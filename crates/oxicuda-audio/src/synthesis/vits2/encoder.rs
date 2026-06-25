//! Posterior encoder, prior (text) encoder, the VAE KL terms and monotonic
//! alignment search for VITS2.
//!
//! These wire the conditional-VAE structure of VITS
//! ([Kim et al. 2021](https://arxiv.org/abs/2106.06103)):
//!
//! * [`PosteriorEncoder`] `q(z | x)` maps an acoustic feature sequence (e.g. a
//!   linear spectrogram) to a diagonal-Gaussian posterior `(m_q, logs_q)` and
//!   draws a reparameterised latent `z = m_q + ε·exp(logs_q)`.
//! * [`PriorEncoder`] is the text encoder `p(z | c)`: a stack of feed-forward
//!   Transformer blocks (reusing [`crate::synthesis::fastspeech2::FftBlock`])
//!   over phoneme embeddings, projected to the prior `(m_p, logs_p)`.
//! * [`gaussian_kl`] is the closed-form KL between two diagonal Gaussians
//!   (`>= 0`, `= 0` iff equal) — the headline VAE KL.
//! * [`flow_kl`] is the VITS Monte-Carlo KL term that incorporates the
//!   flow-transformed latent `z_p = f(z)` and the flow log-determinant.
//! * [`monotonic_alignment_search`] recovers a hard monotonic text→frame
//!   alignment (and per-token durations) by maximising the Gaussian likelihood
//!   of `z_p` under the prior, via dynamic programming.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::synthesis::fastspeech2::FftBlock;
use crate::synthesis::vits2::common::{DenseLayer, conv1d_same, make_normal_vec, relu_inplace};

/// Half of `ln(2π)`.
fn half_ln_2pi() -> f32 {
    0.5 * (2.0 * std::f32::consts::PI).ln()
}

// ─── Posterior encoder ───────────────────────────────────────────────────────

/// VITS posterior encoder `q(z | x)`.
///
/// A pre-projection, two non-causal `conv1d → ReLU` stages and a statistics
/// projection map the acoustic feature `[t, n_feat]` to interleaved mean /
/// log-variance `[t, 2·latent]`, split into `(m_q, logs_q)`. (The reference
/// uses a dilated WaveNet trunk; the convolutional trunk here is the same family
/// of non-causal context mixing.)
#[derive(Debug, Clone)]
pub struct PosteriorEncoder {
    /// Input projection `n_feat → hidden`.
    pre: DenseLayer,
    /// First conv weight `[hidden, hidden, kernel]`.
    conv1_w: Vec<f32>,
    /// First conv bias `[hidden]`.
    conv1_b: Vec<f32>,
    /// Second conv weight `[hidden, hidden, kernel]`.
    conv2_w: Vec<f32>,
    /// Second conv bias `[hidden]`.
    conv2_b: Vec<f32>,
    /// Statistics projection `hidden → 2·latent`.
    proj: DenseLayer,
    /// Input feature dimension.
    n_feat: usize,
    /// Hidden channel count.
    hidden: usize,
    /// Latent dimension.
    latent: usize,
    /// Convolution kernel size (odd).
    kernel: usize,
}

impl PosteriorEncoder {
    /// Construct a posterior encoder.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `n_feat == 0`.
    /// - [`AudioError::InvalidKernelSize`] when `kernel` is `0` or even.
    /// - [`AudioError::Internal`] when `hidden == 0` or `latent == 0`.
    pub fn new(
        n_feat: usize,
        hidden: usize,
        latent: usize,
        kernel: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if n_feat == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if hidden == 0 {
            return Err(AudioError::Internal("PosteriorEncoder: hidden == 0".into()));
        }
        if latent == 0 {
            return Err(AudioError::Internal("PosteriorEncoder: latent == 0".into()));
        }
        if kernel == 0 || kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(kernel));
        }
        let sc = 1.0 / ((hidden * kernel) as f32).sqrt();
        Ok(Self {
            pre: DenseLayer::new(n_feat, hidden, (2.0 / n_feat as f32).sqrt(), rng),
            conv1_w: make_normal_vec(hidden * hidden * kernel, sc, rng),
            conv1_b: vec![0.0_f32; hidden],
            conv2_w: make_normal_vec(hidden * hidden * kernel, sc, rng),
            conv2_b: vec![0.0_f32; hidden],
            proj: DenseLayer::new(hidden, 2 * latent, 1.0 / (hidden as f32).sqrt(), rng),
            n_feat,
            hidden,
            latent,
            kernel,
        })
    }

    /// Latent dimension produced by this encoder.
    #[must_use]
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// Encode `feat` of `[t, n_feat]` into `(m_q, logs_q)`, each `[t, latent]`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `t == 0`.
    /// - [`AudioError::ShapeMismatch`] when `feat.len() != t * n_feat`.
    pub fn forward(&self, feat: &[f32], t: usize) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "PosteriorEncoder: t == 0".into(),
            });
        }
        if feat.len() != t * self.n_feat {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "PosteriorEncoder: feat.len()={} != t*n_feat={}",
                    feat.len(),
                    t * self.n_feat
                ),
            });
        }
        let mut h = self.pre.forward(feat, t);
        relu_inplace(&mut h);
        let mut h = conv1d_same(
            &h,
            t,
            self.hidden,
            self.hidden,
            self.kernel,
            &self.conv1_w,
            &self.conv1_b,
        );
        relu_inplace(&mut h);
        let mut h = conv1d_same(
            &h,
            t,
            self.hidden,
            self.hidden,
            self.kernel,
            &self.conv2_w,
            &self.conv2_b,
        );
        relu_inplace(&mut h);
        let stats = self.proj.forward(&h, t); // [t, 2*latent]
        let mut m_q = vec![0.0_f32; t * self.latent];
        let mut logs_q = vec![0.0_f32; t * self.latent];
        for ti in 0..t {
            let row = &stats[ti * 2 * self.latent..(ti + 1) * 2 * self.latent];
            m_q[ti * self.latent..(ti + 1) * self.latent].copy_from_slice(&row[..self.latent]);
            logs_q[ti * self.latent..(ti + 1) * self.latent].copy_from_slice(&row[self.latent..]);
        }
        Ok((m_q, logs_q))
    }
}

/// Reparameterised sample `z = m + ε·exp(logs)` with `ε ~ N(0, 1)`.
///
/// `m` and `logs` are `[t, latent]`; the result is `[t, latent]`.
#[must_use]
pub fn reparameterize(m: &[f32], logs: &[f32], rng: &mut LcgRng) -> Vec<f32> {
    let mut eps = vec![0.0_f32; m.len()];
    rng.fill_normal(&mut eps);
    let mut z = vec![0.0_f32; m.len()];
    for (i, ((&mi, &li), &ei)) in m.iter().zip(logs.iter()).zip(eps.iter()).enumerate() {
        z[i] = mi + ei * li.exp();
    }
    z
}

// ─── Prior / text encoder ─────────────────────────────────────────────────────

/// VITS prior (text) encoder `p(z | c)`.
///
/// Runs `depth` feed-forward Transformer blocks over the phoneme embeddings and
/// projects the hidden states to the prior `(m_p, logs_p)`. The hidden states
/// are also returned for conditioning the stochastic duration predictor.
#[derive(Debug, Clone)]
pub struct PriorEncoder {
    /// Feed-forward Transformer blocks.
    blocks: Vec<FftBlock>,
    /// Prior statistics projection `embed_dim → 2·latent`.
    proj: DenseLayer,
    /// Phoneme-embedding / hidden dimension.
    embed_dim: usize,
    /// Latent dimension.
    latent: usize,
}

impl PriorEncoder {
    /// Construct a prior encoder.
    ///
    /// # Errors
    ///
    /// - [`AudioError::Internal`] when `depth == 0` or `latent == 0`.
    /// - Any error from [`FftBlock::new`].
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        conv_dim: usize,
        ffn_kernel: usize,
        depth: usize,
        latent: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if depth == 0 {
            return Err(AudioError::Internal("PriorEncoder: depth == 0".into()));
        }
        if latent == 0 {
            return Err(AudioError::Internal("PriorEncoder: latent == 0".into()));
        }
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(FftBlock::new(
                embed_dim, n_heads, conv_dim, ffn_kernel, rng,
            )?);
        }
        Ok(Self {
            blocks,
            proj: DenseLayer::new(embed_dim, 2 * latent, 1.0 / (embed_dim as f32).sqrt(), rng),
            embed_dim,
            latent,
        })
    }

    /// Latent dimension produced by this encoder.
    #[must_use]
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// Encode phonemes `[t, embed_dim]` into `(h_text, m_p, logs_p)`.
    ///
    /// `h_text` is `[t, embed_dim]`; `m_p` and `logs_p` are `[t, latent]`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `t == 0`.
    /// - [`AudioError::ShapeMismatch`] when `phon.len() != t * embed_dim`.
    /// - Propagates errors from the Transformer blocks.
    pub fn forward(&self, phon: &[f32], t: usize) -> AudioResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "PriorEncoder: t == 0".into(),
            });
        }
        if phon.len() != t * self.embed_dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "PriorEncoder: phon.len()={} != t*embed_dim={}",
                    phon.len(),
                    t * self.embed_dim
                ),
            });
        }
        let mut h = phon.to_vec();
        for block in &self.blocks {
            h = block.forward(&h, t)?;
        }
        let stats = self.proj.forward(&h, t);
        let mut m_p = vec![0.0_f32; t * self.latent];
        let mut logs_p = vec![0.0_f32; t * self.latent];
        for ti in 0..t {
            let row = &stats[ti * 2 * self.latent..(ti + 1) * 2 * self.latent];
            m_p[ti * self.latent..(ti + 1) * self.latent].copy_from_slice(&row[..self.latent]);
            logs_p[ti * self.latent..(ti + 1) * self.latent].copy_from_slice(&row[self.latent..]);
        }
        Ok((h, m_p, logs_p))
    }
}

// ─── KL terms ─────────────────────────────────────────────────────────────────

/// Closed-form KL divergence `KL(N(m_q, s_q²) ‖ N(m_p, s_p²))` summed over all
/// elements, with `s = exp(logs)`.
///
/// This diagonal-Gaussian KL is provably non-negative (Gibbs' inequality) and is
/// exactly `0` when the two distributions coincide — the property checked by the
/// KL sanity test.
///
/// # Errors
///
/// [`AudioError::ShapeMismatch`] when the four buffers differ in length.
pub fn gaussian_kl(m_q: &[f32], logs_q: &[f32], m_p: &[f32], logs_p: &[f32]) -> AudioResult<f32> {
    let n = m_q.len();
    if logs_q.len() != n || m_p.len() != n || logs_p.len() != n {
        return Err(AudioError::ShapeMismatch {
            msg: "gaussian_kl: mismatched buffer lengths".into(),
        });
    }
    let mut kl = 0.0_f32;
    for (((&mq, &lq), &mp), &lp) in m_q
        .iter()
        .zip(logs_q.iter())
        .zip(m_p.iter())
        .zip(logs_p.iter())
    {
        // KL = log(s_p/s_q) + (s_q² + (m_q − m_p)²) / (2 s_p²) − 1/2.
        let var_ratio = (2.0 * (lq - lp)).exp(); // s_q² / s_p²
        let mean_term = (mq - mp) * (mq - mp) * (-2.0 * lp).exp();
        kl += (lp - lq) + 0.5 * (var_ratio + mean_term) - 0.5;
    }
    Ok(kl)
}

/// VITS Monte-Carlo KL term using the flow-transformed latent.
///
/// With a single posterior sample `z ~ q` and `z_p = f(z)`, the per-element term
/// is `logs_p − logs_q − 1/2 + 1/2·(z_p − m_p)²·exp(−2 logs_p)`; the flow's
/// log-determinant (which need not vanish for an affine-coupling flow) enters as
/// `− flow_logdet`. This is the KL contribution to the VITS ELBO. It is asserted
/// only to be finite in tests (its value is data/parameter dependent).
///
/// # Errors
///
/// [`AudioError::ShapeMismatch`] when the four buffers differ in length.
pub fn flow_kl(
    z_p: &[f32],
    logs_q: &[f32],
    m_p: &[f32],
    logs_p: &[f32],
    flow_logdet: f32,
) -> AudioResult<f32> {
    let n = z_p.len();
    if logs_q.len() != n || m_p.len() != n || logs_p.len() != n {
        return Err(AudioError::ShapeMismatch {
            msg: "flow_kl: mismatched buffer lengths".into(),
        });
    }
    let mut kl = 0.0_f32;
    for (((&zp, &lq), &mp), &lp) in z_p
        .iter()
        .zip(logs_q.iter())
        .zip(m_p.iter())
        .zip(logs_p.iter())
    {
        let resid = (zp - mp) * (zp - mp) * (-2.0 * lp).exp();
        kl += lp - lq - 0.5 + 0.5 * resid;
    }
    Ok(kl - flow_logdet)
}

/// Per-frame Gaussian log-density `log N(z_p[j]; m_p[i], s_p[i])` summed over the
/// `latent` channels, the local score used by monotonic alignment search.
fn gaussian_log_density(z_row: &[f32], m_row: &[f32], logs_row: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for ((&z, &m), &l) in z_row.iter().zip(m_row.iter()).zip(logs_row.iter()) {
        let diff = z - m;
        acc += -0.5 * diff * diff * (-2.0 * l).exp() - l - half_ln_2pi();
    }
    acc
}

// ─── Monotonic alignment search ──────────────────────────────────────────────

/// Monotonic alignment search (MAS).
///
/// Finds the hard monotonic, surjective text→frame alignment that maximises the
/// total Gaussian log-likelihood `Σ_j log N(z_p[j]; m_p[a(j)], s_p[a(j)])`, then
/// returns the per-text-token frame counts (durations, summing to `t_mel`).
///
/// The dynamic program enforces the VITS alignment constraints: token index is
/// non-decreasing in frame index, advances by at most one per frame, starts at
/// token `0` (frame `0`) and ends at token `t_text − 1` (frame `t_mel − 1`).
///
/// `m_p` / `logs_p` are `[t_text, latent]`; `z_p` is `[t_mel, latent]`.
///
/// # Errors
///
/// - [`AudioError::EmptyInput`] when `t_text == 0` or `t_mel == 0`.
/// - [`AudioError::ShapeMismatch`] for inconsistent buffer lengths.
/// - [`AudioError::InvalidSequenceLength`] when `t_mel < t_text` (no monotonic
///   surjective alignment exists).
pub fn monotonic_alignment_search(
    m_p: &[f32],
    logs_p: &[f32],
    z_p: &[f32],
    t_text: usize,
    t_mel: usize,
    latent: usize,
) -> AudioResult<Vec<usize>> {
    if t_text == 0 || t_mel == 0 {
        return Err(AudioError::EmptyInput {
            msg: "monotonic_alignment_search: empty sequence".into(),
        });
    }
    if m_p.len() != t_text * latent || logs_p.len() != t_text * latent {
        return Err(AudioError::ShapeMismatch {
            msg: "monotonic_alignment_search: prior buffer length".into(),
        });
    }
    if z_p.len() != t_mel * latent {
        return Err(AudioError::ShapeMismatch {
            msg: "monotonic_alignment_search: z_p buffer length".into(),
        });
    }
    if t_mel < t_text {
        return Err(AudioError::InvalidSequenceLength(t_mel));
    }

    let neg_inf = f32::NEG_INFINITY;
    // value[i*t_mel + j] = local Gaussian log-density of frame j under token i.
    let mut value = vec![0.0_f32; t_text * t_mel];
    for i in 0..t_text {
        let m_row = &m_p[i * latent..(i + 1) * latent];
        let logs_row = &logs_p[i * latent..(i + 1) * latent];
        for j in 0..t_mel {
            let z_row = &z_p[j * latent..(j + 1) * latent];
            value[i * t_mel + j] = gaussian_log_density(z_row, m_row, logs_row);
        }
    }

    // q[i*t_mel + j] = best cumulative score reaching (token i, frame j).
    let mut q = vec![neg_inf; t_text * t_mel];
    // from_advance[i][j] = true if the optimum reached (i, j) by advancing the
    // token (i−1 → i); false if it stayed on token i.
    let mut from_advance = vec![false; t_text * t_mel];

    for j in 0..t_mel {
        // Reachability band: i <= j and (t_text-1-i) <= (t_mel-1-j).
        let i_lo = (j + t_text).saturating_sub(t_mel);
        let i_hi = j.min(t_text - 1);
        for i in i_lo..=i_hi {
            let local = value[i * t_mel + j];
            if j == 0 {
                // Only token 0 can occupy frame 0.
                if i == 0 {
                    q[i * t_mel + j] = local;
                }
                continue;
            }
            let stay = q[i * t_mel + (j - 1)];
            let advance = if i > 0 {
                q[(i - 1) * t_mel + (j - 1)]
            } else {
                neg_inf
            };
            let best = stay.max(advance);
            if best > neg_inf {
                q[i * t_mel + j] = local + best;
                // Tie (or stay-better) keeps the token; strict advance wins.
                from_advance[i * t_mel + j] = advance > stay;
            }
        }
    }

    // Backtrack from (t_text-1, t_mel-1).
    let mut durations = vec![0usize; t_text];
    let mut i = t_text - 1;
    for j in (0..t_mel).rev() {
        durations[i] += 1;
        if j > 0 && from_advance[i * t_mel + j] {
            i -= 1;
        }
    }
    Ok(durations)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_encoder_shapes_and_finite() {
        let mut rng = LcgRng::new(1);
        let enc = PosteriorEncoder::new(16, 24, 8, 5, &mut rng).expect("enc");
        let t = 12usize;
        let mut feat = vec![0.0_f32; t * 16];
        LcgRng::new(10).fill_normal(&mut feat);
        let (m_q, logs_q) = enc.forward(&feat, t).expect("forward");
        assert_eq!(m_q.len(), t * 8);
        assert_eq!(logs_q.len(), t * 8);
        assert!(m_q.iter().chain(logs_q.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn prior_encoder_shapes_and_finite() {
        let mut rng = LcgRng::new(2);
        let enc = PriorEncoder::new(16, 2, 32, 9, 2, 8, &mut rng).expect("enc");
        let t = 6usize;
        let mut phon = vec![0.0_f32; t * 16];
        LcgRng::new(20).fill_normal(&mut phon);
        let (h, m_p, logs_p) = enc.forward(&phon, t).expect("forward");
        assert_eq!(h.len(), t * 16);
        assert_eq!(m_p.len(), t * 8);
        assert_eq!(logs_p.len(), t * 8);
        assert!(
            h.iter()
                .chain(m_p.iter())
                .chain(logs_p.iter())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn gaussian_kl_zero_when_equal() {
        // TEST 4: KL == 0 when posterior == prior.
        let mut rng = LcgRng::new(3);
        let n = 40usize;
        let mut m = vec![0.0_f32; n];
        let mut logs = vec![0.0_f32; n];
        rng.fill_normal(&mut m);
        rng.fill_normal(&mut logs);
        let kl = gaussian_kl(&m, &logs, &m, &logs).expect("kl");
        assert!(kl.abs() < 1e-5, "kl should vanish, got {kl}");
    }

    #[test]
    fn gaussian_kl_nonnegative() {
        // TEST 4: KL >= 0 for arbitrary diagonal Gaussians.
        let mut rng = LcgRng::new(4);
        let n = 64usize;
        for _ in 0..16 {
            let mut m_q = vec![0.0_f32; n];
            let mut logs_q = vec![0.0_f32; n];
            let mut m_p = vec![0.0_f32; n];
            let mut logs_p = vec![0.0_f32; n];
            rng.fill_normal(&mut m_q);
            rng.fill_normal(&mut logs_q);
            rng.fill_normal(&mut m_p);
            rng.fill_normal(&mut logs_p);
            // Keep log-sigmas modest so the exp terms stay well-conditioned.
            for v in logs_q.iter_mut().chain(logs_p.iter_mut()) {
                *v *= 0.5;
            }
            let kl = gaussian_kl(&m_q, &logs_q, &m_p, &logs_p).expect("kl");
            assert!(kl >= -1e-4, "kl negative: {kl}");
        }
    }

    #[test]
    fn flow_kl_is_finite() {
        let n = 24usize;
        let mut rng = LcgRng::new(5);
        let mut z_p = vec![0.0_f32; n];
        let mut logs_q = vec![0.0_f32; n];
        let mut m_p = vec![0.0_f32; n];
        let mut logs_p = vec![0.0_f32; n];
        rng.fill_normal(&mut z_p);
        rng.fill_normal(&mut logs_q);
        rng.fill_normal(&mut m_p);
        rng.fill_normal(&mut logs_p);
        let kl = flow_kl(&z_p, &logs_q, &m_p, &logs_p, 0.37).expect("kl");
        assert!(kl.is_finite());
    }

    #[test]
    fn mas_durations_sum_to_t_mel() {
        let t_text = 4usize;
        let t_mel = 11usize;
        let latent = 3usize;
        let mut rng = LcgRng::new(6);
        let mut m_p = vec![0.0_f32; t_text * latent];
        let logs_p = vec![0.0_f32; t_text * latent];
        let mut z_p = vec![0.0_f32; t_mel * latent];
        rng.fill_normal(&mut m_p);
        rng.fill_normal(&mut z_p);
        let dur =
            monotonic_alignment_search(&m_p, &logs_p, &z_p, t_text, t_mel, latent).expect("mas");
        assert_eq!(dur.len(), t_text);
        assert_eq!(dur.iter().sum::<usize>(), t_mel);
        // Surjective: every token must receive at least one frame.
        assert!(dur.iter().all(|&d| d >= 1));
    }

    #[test]
    fn mas_recovers_block_alignment() {
        // Construct z_p so that frames clearly cluster onto 2 well-separated
        // prior means: frames 0..3 near token 0, frames 3..6 near token 1.
        let t_text = 2usize;
        let t_mel = 6usize;
        let latent = 1usize;
        let m_p = vec![0.0_f32, 10.0]; // token 0 mean 0, token 1 mean 10
        let logs_p = vec![0.0_f32, 0.0];
        let z_p = vec![0.1_f32, -0.1, 0.05, 9.9, 10.1, 10.0];
        let dur =
            monotonic_alignment_search(&m_p, &logs_p, &z_p, t_text, t_mel, latent).expect("mas");
        assert_eq!(dur, vec![3, 3], "expected clean 3/3 split, got {dur:?}");
    }

    #[test]
    fn mas_rejects_too_few_frames() {
        let r = monotonic_alignment_search(&[0.0; 3], &[0.0; 3], &[0.0; 2], 3, 2, 1);
        assert!(matches!(r, Err(AudioError::InvalidSequenceLength(2))));
    }

    #[test]
    fn reparameterize_matches_mean_at_zero_logvar() {
        // With logs = ln(0) → −inf is avoided; use very small sigma so z ≈ m.
        let m = vec![1.0_f32, 2.0, 3.0, 4.0];
        let logs = vec![-20.0_f32; 4];
        let z = reparameterize(&m, &logs, &mut LcgRng::new(7));
        for (zi, mi) in z.iter().zip(m.iter()) {
            assert!((zi - mi).abs() < 1e-3);
        }
    }
}
