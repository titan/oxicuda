//! TVAE: a Variational Autoencoder for tabular data.
//!
//! Reference: Xu, Skoularidou, Cuesta-Infante & Veeramachaneni (2019),
//! "Modeling Tabular Data using Conditional GAN", NeurIPS 2019 — the TVAE
//! baseline introduced alongside CTGAN.
//!
//! # Architecture
//!
//! - **Encoder** MLP: `data → (μ, logσ²)` over a latent of dimension `L`.
//! - **Reparameterisation**: `z = μ + exp(½·logσ²) ⊙ ε`, `ε ~ N(0, I)`.
//! - **Decoder** MLP: `z → reconstruction`, which for each *continuous* column
//!   produces a Gaussian mean (paired with a learnable per-column log-σ) and for
//!   each *categorical* column produces softmax logits.
//!
//! # Data layout
//!
//! A row `x` is laid out as `[continuous scalars | one-hot blocks…]`:
//! the first `n_continuous` entries are continuous values, followed by one block
//! of length `cardinality` per categorical column (typically one-hot, but any
//! non-negative soft assignment summing to one is handled by the cross-entropy
//! term).  The decoder output uses the identical layout.
//!
//! # ELBO
//!
//! `ELBO loss = reconstruction + KL`, where the reconstruction term is the
//! Gaussian negative log-likelihood for continuous columns plus the categorical
//! cross-entropy, and `KL(N(μ, σ²) ‖ N(0, I)) = ½ Σ (σ² + μ² − 1 − logσ²)`.
//!
//! [`Tvae::elbo_loss`] is **deterministic**: it evaluates the reconstruction at
//! the posterior mean (`z = μ`, i.e. `ε = 0`), a zero-variance single-sample
//! estimate of the ELBO.  Stochastic latents are available through
//! [`Tvae::reparameterize`] and [`Tvae::sample`].

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::nn::{Mlp, log_softmax, softmax};

/// `ln(2π)`, the constant in the Gaussian negative log-likelihood.
const LN_2PI: f32 = 1.837_877_1;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`Tvae`].
#[derive(Debug, Clone)]
pub struct TvaeConfig {
    /// Number of continuous columns (each a single scalar in the data row).
    pub n_continuous: usize,
    /// Cardinality of each categorical column (one one-hot block per column).
    pub categorical_cardinalities: Vec<usize>,
    /// Latent dimension `L`.
    pub latent_dim: usize,
    /// Hidden width of the encoder / decoder MLPs.
    pub hidden_dim: usize,
    /// Number of hidden layers in the encoder / decoder MLPs.
    pub n_layers: usize,
}

// ─── Tvae ─────────────────────────────────────────────────────────────────────

/// Tabular Variational Autoencoder.
#[derive(Debug, Clone)]
pub struct Tvae {
    config: TvaeConfig,
    /// Total data-row width: `n_continuous + Σ cardinalities`.
    data_dim: usize,
    /// Encoder MLP `data_dim → 2·latent_dim` (concatenated `[μ | logσ²]`).
    encoder: Mlp,
    /// Decoder MLP `latent_dim → data_dim`.
    decoder: Mlp,
    /// Per-continuous-column decoder log-σ, length `n_continuous`.
    log_sigma: Vec<f32>,
}

impl Tvae {
    /// Construct a new TVAE with randomly-initialised encoder / decoder weights.
    ///
    /// # Errors
    /// - [`TabularError::InvalidFeatureCount`] if the data width is zero
    ///   (`n_continuous == 0` and no categorical columns).
    /// - [`TabularError::InvalidEmbedDim`] if `latent_dim == 0`.
    /// - [`TabularError::InvalidParameter`] if `hidden_dim == 0`, or any
    ///   categorical cardinality is zero.
    pub fn new(config: TvaeConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        let cat_sum: usize = config.categorical_cardinalities.iter().sum();
        let data_dim = config.n_continuous + cat_sum;
        if data_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if config.latent_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if config.hidden_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "hidden_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if config.categorical_cardinalities.contains(&0) {
            return Err(TabularError::InvalidParameter {
                name: "categorical_cardinalities".into(),
                msg: "every categorical column must have cardinality ≥ 1".into(),
            });
        }

        // Encoder: data_dim → hidden (×n_layers) → 2·latent_dim.
        let mut enc_dims = Vec::with_capacity(config.n_layers + 2);
        enc_dims.push(data_dim);
        for _ in 0..config.n_layers {
            enc_dims.push(config.hidden_dim);
        }
        enc_dims.push(2 * config.latent_dim);
        let encoder = Mlp::new(&enc_dims, rng)?;

        // Decoder: latent_dim → hidden (×n_layers) → data_dim.
        let mut dec_dims = Vec::with_capacity(config.n_layers + 2);
        dec_dims.push(config.latent_dim);
        for _ in 0..config.n_layers {
            dec_dims.push(config.hidden_dim);
        }
        dec_dims.push(data_dim);
        let decoder = Mlp::new(&dec_dims, rng)?;

        let log_sigma = vec![0.0_f32; config.n_continuous];

        Ok(Self {
            config,
            data_dim,
            encoder,
            decoder,
            log_sigma,
        })
    }

    /// Total data-row width (`n_continuous + Σ cardinalities`).
    #[must_use]
    pub fn data_dim(&self) -> usize {
        self.data_dim
    }

    /// Latent dimension `L`.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    /// Encode a data row into `(μ, logσ²)`, each of length `latent_dim`.
    ///
    /// `logσ²` is clamped to `[-10, 10]` for numerical stability.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != data_dim`.
    pub fn encode(&self, x: &[f32]) -> TabularResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.data_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.data_dim,
                got: x.len(),
            });
        }
        let h = self.encoder.forward(x);
        let l = self.config.latent_dim;
        let mu = h[0..l].to_vec();
        let logvar = h[l..2 * l].iter().map(|&v| v.clamp(-10.0, 10.0)).collect();
        Ok((mu, logvar))
    }

    /// Reparameterise: `z = μ + exp(½·logσ²) ⊙ ε`.
    ///
    /// Deterministic for a fixed `eps`.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if any argument length differs from
    /// `latent_dim`.
    pub fn reparameterize(
        &self,
        mu: &[f32],
        logvar: &[f32],
        eps: &[f32],
    ) -> TabularResult<Vec<f32>> {
        let l = self.config.latent_dim;
        if mu.len() != l || logvar.len() != l || eps.len() != l {
            return Err(TabularError::DimensionMismatch {
                expected: l,
                got: mu.len().min(logvar.len()).min(eps.len()),
            });
        }
        let z = mu
            .iter()
            .zip(logvar.iter())
            .zip(eps.iter())
            .map(|((&m, &lv), &e)| m + (0.5 * lv).exp() * e)
            .collect();
        Ok(z)
    }

    /// Decode a latent vector into a reconstruction laid out like a data row:
    /// continuous means followed by per-column categorical logits.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `z.len() != latent_dim`.
    pub fn decode(&self, z: &[f32]) -> TabularResult<Vec<f32>> {
        if z.len() != self.config.latent_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.latent_dim,
                got: z.len(),
            });
        }
        Ok(self.decoder.forward(z))
    }

    /// Reconstruction term: Gaussian NLL over continuous columns plus
    /// categorical cross-entropy, given a target row `x` and a decoder output.
    fn reconstruction_loss(&self, x: &[f32], recon: &[f32]) -> f32 {
        let mut loss = 0.0_f32;

        // Continuous columns: Gaussian negative log-likelihood.
        for c in 0..self.config.n_continuous {
            let mean = recon[c];
            let log_sigma = self.log_sigma[c];
            let var = (2.0 * log_sigma).exp().max(1e-12);
            let diff = x[c] - mean;
            loss += 0.5 * LN_2PI + log_sigma + (diff * diff) / (2.0 * var);
        }

        // Categorical columns: soft cross-entropy against the one-hot target.
        let mut off = self.config.n_continuous;
        for &card in &self.config.categorical_cardinalities {
            let lsm = log_softmax(&recon[off..off + card]);
            let target = &x[off..off + card];
            for (t, lp) in target.iter().zip(lsm.iter()) {
                loss -= t * lp;
            }
            off += card;
        }
        loss
    }

    /// Evidence lower bound *loss* (negative ELBO) for a data row `x`.
    ///
    /// Deterministic: the reconstruction is evaluated at the posterior mean
    /// (`z = μ`).  Returns `reconstruction + KL`.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != data_dim`, propagated
    /// from [`encode`](Self::encode) / [`decode`](Self::decode).
    pub fn elbo_loss(&self, x: &[f32]) -> TabularResult<f32> {
        let (mu, logvar) = self.encode(x)?;
        let recon = self.decode(&mu)?;
        let recon_loss = self.reconstruction_loss(x, &recon);
        let kl = kl_divergence_standard(&mu, &logvar);
        Ok(recon_loss + kl)
    }

    /// Generate `n` rows by sampling latents `z ~ N(0, I)` and decoding.
    ///
    /// Continuous columns are filled with the decoded Gaussian means; categorical
    /// columns are filled with the softmax of the decoded logits.  Returns a flat
    /// row-major `[n × data_dim]` buffer.
    ///
    /// # Errors
    /// Propagated from [`decode`](Self::decode).
    pub fn sample(&self, n: usize, rng: &mut LcgRng) -> TabularResult<Vec<f32>> {
        let mut out = Vec::with_capacity(n * self.data_dim);
        for _ in 0..n {
            let mut z = vec![0.0_f32; self.config.latent_dim];
            rng.fill_normal(&mut z);
            let recon = self.decode(&z)?;

            let mut row = vec![0.0_f32; self.data_dim];
            row[..self.config.n_continuous].copy_from_slice(&recon[..self.config.n_continuous]);
            let mut off = self.config.n_continuous;
            for &card in &self.config.categorical_cardinalities {
                let probs = softmax(&recon[off..off + card]);
                row[off..off + card].copy_from_slice(&probs);
                off += card;
            }
            out.extend_from_slice(&row);
        }
        Ok(out)
    }
}

// ─── KL divergence ─────────────────────────────────────────────────────────────

/// `KL(N(μ, σ²) ‖ N(0, I)) = ½ Σ_d (exp(logvar_d) + μ_d² − 1 − logvar_d)`.
///
/// Always non-negative, and exactly zero when `μ = 0` and `logvar = 0`.
#[must_use]
pub fn kl_divergence_standard(mu: &[f32], logvar: &[f32]) -> f32 {
    mu.iter()
        .zip(logvar.iter())
        .map(|(&m, &lv)| 0.5 * (lv.exp() + m * m - 1.0 - lv))
        .sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> TvaeConfig {
        TvaeConfig {
            n_continuous: 3,
            categorical_cardinalities: vec![4, 2],
            latent_dim: 5,
            hidden_dim: 16,
            n_layers: 2,
        }
    }

    fn make_model() -> Tvae {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(42);
        Tvae::new(cfg, &mut rng).expect("new should succeed")
    }

    // ── 1. data_dim / latent_dim accessors ───────────────────────────────────
    #[test]
    fn dims_correct() {
        let m = make_model();
        assert_eq!(m.data_dim(), 3 + 4 + 2);
        assert_eq!(m.latent_dim(), 5);
    }

    // ── 2. encode shapes ─────────────────────────────────────────────────────
    #[test]
    fn encode_shapes() {
        let m = make_model();
        let x = vec![0.1_f32; m.data_dim()];
        let (mu, logvar) = m.encode(&x).expect("encode should succeed");
        assert_eq!(mu.len(), 5);
        assert_eq!(logvar.len(), 5);
        assert!(mu.iter().chain(logvar.iter()).all(|v| v.is_finite()));
    }

    // ── 3. encode wrong length errors ────────────────────────────────────────
    #[test]
    fn encode_wrong_len_errs() {
        let m = make_model();
        assert!(matches!(
            m.encode(&[0.0; 4]),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    // ── 4. reparameterize deterministic for fixed eps ────────────────────────
    #[test]
    fn reparameterize_deterministic() {
        let m = make_model();
        let mu = vec![0.5_f32; 5];
        let logvar = vec![0.0_f32; 5];
        let eps = vec![1.0_f32, -1.0, 0.5, 0.0, 2.0];
        let z1 = m
            .reparameterize(&mu, &logvar, &eps)
            .expect("reparameterize should succeed");
        let z2 = m
            .reparameterize(&mu, &logvar, &eps)
            .expect("reparameterize should succeed");
        assert_eq!(z1, z2);
        // logvar = 0 → σ = 1 → z = mu + eps exactly.
        for ((zi, mi), ei) in z1.iter().zip(mu.iter()).zip(eps.iter()) {
            assert!((zi - (mi + ei)).abs() < 1e-6);
        }
    }

    // ── 5. reparameterize wrong length errors ────────────────────────────────
    #[test]
    fn reparameterize_wrong_len_errs() {
        let m = make_model();
        assert!(m.reparameterize(&[0.0; 4], &[0.0; 5], &[0.0; 5]).is_err());
    }

    // ── 6. decode shape matches data layout ──────────────────────────────────
    #[test]
    fn decode_shape() {
        let m = make_model();
        let z = vec![0.2_f32; 5];
        let recon = m.decode(&z).expect("decode should succeed");
        assert_eq!(recon.len(), m.data_dim());
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    // ── 7. KL ≥ 0 ─────────────────────────────────────────────────────────────
    #[test]
    fn kl_non_negative() {
        let mu = vec![0.3_f32, -1.2, 0.7];
        let logvar = vec![0.5_f32, -0.3, 1.1];
        let kl = kl_divergence_standard(&mu, &logvar);
        assert!(kl >= 0.0, "kl = {kl}");
    }

    // ── 8. KL == 0 when mu = 0, logvar = 0 ───────────────────────────────────
    #[test]
    fn kl_zero_at_standard_normal() {
        let mu = vec![0.0_f32; 6];
        let logvar = vec![0.0_f32; 6];
        assert!(kl_divergence_standard(&mu, &logvar).abs() < 1e-6);
    }

    // ── 9. ELBO loss finite ──────────────────────────────────────────────────
    #[test]
    fn elbo_finite() {
        let m = make_model();
        let mut x = vec![0.4_f32; m.data_dim()];
        // make the categorical blocks valid one-hots
        x[3] = 1.0;
        x[4] = 0.0;
        x[5] = 0.0;
        x[6] = 0.0;
        x[7] = 1.0;
        x[8] = 0.0;
        let loss = m.elbo_loss(&x).expect("elbo_loss should succeed");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    // ── 10. ELBO of all-zero input finite ────────────────────────────────────
    #[test]
    fn elbo_all_zero_finite() {
        let m = make_model();
        let x = vec![0.0_f32; m.data_dim()];
        let loss = m.elbo_loss(&x).expect("elbo_loss should succeed");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    // ── 11. sample shape and finite ──────────────────────────────────────────
    #[test]
    fn sample_shape_finite() {
        let m = make_model();
        let mut rng = LcgRng::new(7);
        let out = m.sample(4, &mut rng).expect("sample should succeed");
        assert_eq!(out.len(), 4 * m.data_dim());
        assert!(out.iter().all(|v| v.is_finite()));
        // Each categorical block in each sampled row sums to ~1 (softmax).
        for r in 0..4 {
            let row = &out[r * m.data_dim()..(r + 1) * m.data_dim()];
            let s1: f32 = row[3..7].iter().sum();
            let s2: f32 = row[7..9].iter().sum();
            assert!((s1 - 1.0).abs() < 1e-4);
            assert!((s2 - 1.0).abs() < 1e-4);
        }
    }

    // ── 12. same seed → same sample ──────────────────────────────────────────
    #[test]
    fn sample_same_seed_same_output() {
        let m = make_model();
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let a = m.sample(3, &mut r1).expect("sample should succeed");
        let b = m.sample(3, &mut r2).expect("sample should succeed");
        assert_eq!(a, b);
    }

    // ── 13. diff seed → diff sample ──────────────────────────────────────────
    #[test]
    fn sample_diff_seed_diff_output() {
        let m = make_model();
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(999);
        let a = m.sample(3, &mut r1).expect("sample should succeed");
        let b = m.sample(3, &mut r2).expect("sample should succeed");
        assert_ne!(a, b);
    }

    // ── 14. all-continuous configuration works ───────────────────────────────
    #[test]
    fn all_continuous_config() {
        let cfg = TvaeConfig {
            n_continuous: 6,
            categorical_cardinalities: vec![],
            latent_dim: 3,
            hidden_dim: 8,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(5);
        let m = Tvae::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 6];
        assert!(
            m.elbo_loss(&x)
                .expect("elbo_loss should succeed")
                .is_finite()
        );
        assert_eq!(m.data_dim(), 6);
    }

    // ── 15. all-categorical configuration works ──────────────────────────────
    #[test]
    fn all_categorical_config() {
        let cfg = TvaeConfig {
            n_continuous: 0,
            categorical_cardinalities: vec![3, 3],
            latent_dim: 4,
            hidden_dim: 8,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(6);
        let m = Tvae::new(cfg, &mut rng).expect("new should succeed");
        let mut x = vec![0.0_f32; 6];
        x[0] = 1.0;
        x[3] = 1.0;
        assert!(
            m.elbo_loss(&x)
                .expect("elbo_loss should succeed")
                .is_finite()
        );
    }

    // ── 16. constructor validation ───────────────────────────────────────────
    #[test]
    fn new_rejects_zero_data_dim() {
        let cfg = TvaeConfig {
            n_continuous: 0,
            categorical_cardinalities: vec![],
            latent_dim: 4,
            hidden_dim: 8,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Tvae::new(cfg, &mut rng),
            Err(TabularError::InvalidFeatureCount { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_latent() {
        let cfg = TvaeConfig {
            n_continuous: 3,
            categorical_cardinalities: vec![2],
            latent_dim: 0,
            hidden_dim: 8,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Tvae::new(cfg, &mut rng),
            Err(TabularError::InvalidEmbedDim { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_cardinality() {
        let cfg = TvaeConfig {
            n_continuous: 1,
            categorical_cardinalities: vec![0],
            latent_dim: 4,
            hidden_dim: 8,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(1);
        assert!(Tvae::new(cfg, &mut rng).is_err());
    }
}
