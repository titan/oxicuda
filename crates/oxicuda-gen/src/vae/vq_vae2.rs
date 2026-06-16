//! Hierarchical Vector-Quantised VAE (VQ-VAE-2).
//!
//! Implements the two-level vector quantisation from
//! Razavi et al. (2019) "Generating Diverse High-Fidelity Images with
//! VQ-VAE-2" (NeurIPS 2019).
//!
//! The model maintains two VQ codebooks: a *bottom* one for fine-grained
//! detail and a *top* one for coarse structure.  Both codebooks use
//! Exponential Moving Average (EMA) updates instead of straight-through
//! gradient estimation, which avoids codebook collapse.
//!
//! ## Architecture (CPU linear approximation)
//!
//! ```text
//! x  ──enc_bottom──►  z_bottom  ──VQ_bottom──►  q_bottom
//!                       │                            │
//!                   enc_top                          │
//!                       ▼                            │
//!                     z_top  ──VQ_top──►  q_top      │
//!                                             │      │
//!                                          concat    │
//!                                             ▼      │
//!                                          dec_w  ◄──┘
//!                                             │
//!                                             ▼
//!                                           x_hat
//! ```

use crate::error::{GenError, GenResult};

/// Type alias for the crate-level LCG random number generator.
pub type GenRng = crate::handle::LcgRng;

// ─── Box-Muller helper ───────────────────────────────────────────────────────

/// Sample a single standard-normal deviate using the Box-Muller transform.
fn sample_normal_f64(rng: &mut GenRng) -> f64 {
    let u1 = (rng.next_f32() as f64 + 1e-10_f64).min(1.0 - 1e-10_f64);
    let u2 = rng.next_f32() as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

// ─── VqVae2Config ────────────────────────────────────────────────────────────

/// Configuration for the hierarchical VQ-VAE-2 model.
#[derive(Debug, Clone)]
pub struct VqVae2Config {
    /// Dimensionality of the input / decoded output `x`.
    pub d_input: usize,
    /// Dimensionality of the bottom codebook embeddings.
    pub d_bottom: usize,
    /// Dimensionality of the top codebook embeddings.
    pub d_top: usize,
    /// Number of entries in the bottom codebook.
    pub n_bottom: usize,
    /// Number of entries in the top codebook.
    pub n_top: usize,
    /// Scale factor for the commitment loss (straight-through coefficient).
    pub commitment_weight: f64,
    /// EMA decay rate `γ` for codebook EMA updates (typically 0.99).
    pub ema_decay: f64,
}

// ─── VqCodebookEma ───────────────────────────────────────────────────────────

/// A single VQ codebook updated via Exponential Moving Averages.
///
/// For each codebook entry `e_k ∈ ℝ^{d_emb}`, the EMA update tracks:
///
/// ```text
/// ema_count_k ← γ · ema_count_k + (1 - γ) · |C_k|
/// ema_sum_k   ← γ · ema_sum_k   + (1 - γ) · Σ_{z ∈ C_k} z
/// e_k         ← ema_sum_k / ema_count_k
/// ```
///
/// where `C_k` is the set of encoder outputs assigned to entry `k` in the
/// current batch, and `γ` is `ema_decay`.
#[derive(Debug, Clone)]
pub struct VqCodebookEma {
    /// Flat row-major embedding matrix: `n_entries × d_emb`.
    embeddings: Vec<f64>,
    /// EMA numerator sums for each entry: `n_entries × d_emb`.
    ema_sum: Vec<f64>,
    /// EMA denominator (count) for each entry: `n_entries`.
    ema_count: Vec<f64>,
    /// Number of codebook entries.
    n_entries: usize,
    /// Embedding dimensionality.
    d_emb: usize,
    /// EMA decay `γ`.
    ema_decay: f64,
}

impl VqCodebookEma {
    /// Create a new codebook with randomly initialised embeddings.
    ///
    /// Embeddings are drawn from `N(0, 1)` and EMA state is initialised to
    /// `(embedding, 1.0)`.
    ///
    /// # Errors
    ///
    /// - [`GenError::InvalidCodebookSize`] if `n_entries < 2`.
    /// - [`GenError::EmptyInput`] if `d_emb == 0`.
    pub fn new(
        n_entries: usize,
        d_emb: usize,
        ema_decay: f64,
        rng: &mut GenRng,
    ) -> GenResult<Self> {
        if n_entries < 2 {
            return Err(GenError::InvalidCodebookSize(n_entries));
        }
        if d_emb == 0 {
            return Err(GenError::EmptyInput("d_emb must be > 0"));
        }
        let total = n_entries * d_emb;
        let embeddings: Vec<f64> = (0..total).map(|_| sample_normal_f64(rng)).collect();
        let ema_sum = embeddings.clone();
        let ema_count = vec![1.0_f64; n_entries];

        Ok(Self {
            embeddings,
            ema_sum,
            ema_count,
            n_entries,
            d_emb,
            ema_decay,
        })
    }

    /// Quantise a batch of encoder outputs.
    ///
    /// For each of the `batch` input vectors `z_e[i] ∈ ℝ^{d_emb}`, find the
    /// nearest codebook entry by Euclidean distance and return:
    ///
    /// - `quantised`: the concatenated nearest embeddings `[batch × d_emb]`.
    /// - `indices`:   the codebook index assigned to each input `[batch]`.
    /// - `commit_loss`: mean squared distance from each `z_e` to its nearest
    ///   entry (commitment loss, stop-gradient on embeddings).
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `input.len() != batch * d_emb`.
    pub fn quantize(&self, input: &[f64], batch: usize) -> GenResult<(Vec<f64>, Vec<usize>, f64)> {
        let expected = batch * self.d_emb;
        if input.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: input.len(),
            });
        }

        let mut quantised = vec![0.0_f64; expected];
        let mut indices = vec![0_usize; batch];
        let mut total_commit = 0.0_f64;

        for b in 0..batch {
            let z = &input[b * self.d_emb..(b + 1) * self.d_emb];

            // Find nearest codebook entry
            let mut best_idx = 0_usize;
            let mut best_dist = f64::INFINITY;
            for k in 0..self.n_entries {
                let e = &self.embeddings[k * self.d_emb..(k + 1) * self.d_emb];
                let dist: f64 = z.iter().zip(e.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = k;
                }
            }
            indices[b] = best_idx;
            // Copy nearest embedding into quantised output
            let offset = b * self.d_emb;
            let e_off = best_idx * self.d_emb;
            quantised[offset..offset + self.d_emb]
                .copy_from_slice(&self.embeddings[e_off..e_off + self.d_emb]);

            total_commit += best_dist;
        }

        let commit_loss = if batch > 0 {
            total_commit / (batch as f64)
        } else {
            0.0
        };

        Ok((quantised, indices, commit_loss))
    }

    /// Update codebook embeddings via EMA after a quantisation step.
    ///
    /// `input` is the flat encoder output batch `[batch × d_emb]` and
    /// `indices` are the nearest-codebook assignments from `quantize`.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `input.len() != batch * d_emb`
    ///   or `indices.len() != batch`.
    pub fn ema_update(&mut self, input: &[f64], indices: &[usize], batch: usize) -> GenResult<()> {
        if input.len() != batch * self.d_emb {
            return Err(GenError::DimensionMismatch {
                expected: batch * self.d_emb,
                got: input.len(),
            });
        }
        if indices.len() != batch {
            return Err(GenError::DimensionMismatch {
                expected: batch,
                got: indices.len(),
            });
        }

        let gamma = self.ema_decay;

        // Accumulate batch statistics per codebook entry
        let mut batch_sum = vec![0.0_f64; self.n_entries * self.d_emb];
        let mut batch_count = vec![0.0_f64; self.n_entries];

        for (b, &k) in indices.iter().enumerate().take(batch) {
            if k < self.n_entries {
                batch_count[k] += 1.0;
                let z_off = b * self.d_emb;
                let e_off = k * self.d_emb;
                for d in 0..self.d_emb {
                    batch_sum[e_off + d] += input[z_off + d];
                }
            }
        }

        // EMA update
        #[allow(clippy::needless_range_loop)]
        for k in 0..self.n_entries {
            let e_off = k * self.d_emb;
            self.ema_count[k] = gamma * self.ema_count[k] + (1.0 - gamma) * batch_count[k];
            let count = self.ema_count[k].max(1e-12);
            for d in 0..self.d_emb {
                self.ema_sum[e_off + d] =
                    gamma * self.ema_sum[e_off + d] + (1.0 - gamma) * batch_sum[e_off + d];
                self.embeddings[e_off + d] = self.ema_sum[e_off + d] / count;
            }
        }
        Ok(())
    }

    /// Return the number of codebook entries.
    #[must_use]
    #[inline]
    pub fn n_entries(&self) -> usize {
        self.n_entries
    }

    /// Return the embedding dimensionality.
    #[must_use]
    #[inline]
    pub fn d_emb(&self) -> usize {
        self.d_emb
    }
}

// ─── VqVae2 ──────────────────────────────────────────────────────────────────

/// Hierarchical VQ-VAE-2 model.
///
/// The encoder applies a single linear + tanh projection at each level.
/// The decoder concatenates the bottom and top quantised codes and applies
/// a single linear projection back to `d_input`.
///
/// In a full implementation these projections would be replaced by
/// deep convolutional networks; this linear approximation enables
/// unit-testable CPU inference without CUDA.
#[derive(Debug, Clone)]
pub struct VqVae2 {
    /// Bottom encoder weight matrix (flat row-major, `d_bottom × d_input`).
    enc_bottom_w: Vec<f64>,
    /// Bottom encoder bias (`d_bottom`).
    enc_bottom_b: Vec<f64>,
    /// Top encoder weight matrix (flat row-major, `d_top × d_bottom`).
    enc_top_w: Vec<f64>,
    /// Top encoder bias (`d_top`).
    enc_top_b: Vec<f64>,
    /// Decoder weight matrix (flat row-major, `d_input × (d_bottom + d_top)`).
    dec_w: Vec<f64>,
    /// Decoder bias (`d_input`).
    dec_b: Vec<f64>,
    /// Bottom-level VQ codebook.
    codebook_bottom: VqCodebookEma,
    /// Top-level VQ codebook.
    codebook_top: VqCodebookEma,
    /// Model configuration.
    config: VqVae2Config,
}

impl VqVae2 {
    /// Construct a new `VqVae2` with randomly initialised weights and codebooks.
    ///
    /// # Errors
    ///
    /// - [`GenError::EmptyInput`] if any dimension (`d_input`, `d_bottom`,
    ///   `d_top`) is zero.
    /// - Propagates errors from [`VqCodebookEma::new`].
    pub fn new(config: VqVae2Config, rng: &mut GenRng) -> GenResult<Self> {
        if config.d_input == 0 {
            return Err(GenError::EmptyInput("d_input must be > 0"));
        }
        if config.d_bottom == 0 {
            return Err(GenError::EmptyInput("d_bottom must be > 0"));
        }
        if config.d_top == 0 {
            return Err(GenError::EmptyInput("d_top must be > 0"));
        }

        let d_input = config.d_input;
        let d_bottom = config.d_bottom;
        let d_top = config.d_top;

        // He-init helper
        let he_vec = |fan_in: usize, fan_out: usize, rng: &mut GenRng| -> Vec<f64> {
            let std_dev = (2.0_f64 / (fan_in as f64)).sqrt();
            (0..fan_in * fan_out)
                .map(|_| sample_normal_f64(rng) * std_dev)
                .collect()
        };

        let enc_bottom_w = he_vec(d_input, d_bottom, rng);
        let enc_bottom_b = vec![0.0_f64; d_bottom];
        let enc_top_w = he_vec(d_bottom, d_top, rng);
        let enc_top_b = vec![0.0_f64; d_top];
        let dec_w = he_vec(d_bottom + d_top, d_input, rng);
        let dec_b = vec![0.0_f64; d_input];

        let codebook_bottom = VqCodebookEma::new(config.n_bottom, d_bottom, config.ema_decay, rng)?;
        let codebook_top = VqCodebookEma::new(config.n_top, d_top, config.ema_decay, rng)?;

        Ok(Self {
            enc_bottom_w,
            enc_bottom_b,
            enc_top_w,
            enc_top_b,
            dec_w,
            dec_b,
            codebook_bottom,
            codebook_top,
            config,
        })
    }

    /// Encode a single input `x ∈ ℝ^{d_input}`.
    ///
    /// Returns `(bottom_quant, top_quant, bottom_idx, top_idx, total_loss)`.
    ///
    /// - `bottom_quant`: quantised bottom encoding `[d_bottom]`.
    /// - `top_quant`:    quantised top encoding `[d_top]`.
    /// - `bottom_idx`:   codebook index assigned to the bottom encoding.
    /// - `top_idx`:      codebook index assigned to the top encoding.
    /// - `total_loss`:   `commitment_weight · (commit_bottom + commit_top)`.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `x.len() != d_input`.
    #[allow(clippy::type_complexity)]
    pub fn encode(&self, x: &[f64]) -> GenResult<(Vec<f64>, Vec<f64>, usize, usize, f64)> {
        if x.len() != self.config.d_input {
            return Err(GenError::DimensionMismatch {
                expected: self.config.d_input,
                got: x.len(),
            });
        }

        let d_bottom = self.config.d_bottom;
        let d_top = self.config.d_top;

        // Step 1: bottom encoding  z_bottom = tanh(W_bottom · x + b_bottom)
        let z_bottom = self.linear_tanh(&self.enc_bottom_w, &self.enc_bottom_b, x, d_bottom);

        // Step 2: quantise bottom
        let (bottom_quant, bottom_indices, commit_bottom) =
            self.codebook_bottom.quantize(&z_bottom, 1)?;
        let bottom_idx = bottom_indices[0];

        // Step 3: top encoding  z_top = tanh(W_top · q_bottom + b_top)
        let z_top = self.linear_tanh(&self.enc_top_w, &self.enc_top_b, &bottom_quant, d_top);

        // Step 4: quantise top
        let (top_quant, top_indices, commit_top) = self.codebook_top.quantize(&z_top, 1)?;
        let top_idx = top_indices[0];

        // Step 5: total commitment loss
        let total_loss = self.config.commitment_weight * (commit_bottom + commit_top);

        Ok((bottom_quant, top_quant, bottom_idx, top_idx, total_loss))
    }

    /// Decode concatenated `[bottom_quant ‖ top_quant]` back to input space.
    ///
    /// `out = dec_w · [bottom_quant ‖ top_quant] + dec_b`
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `bottom_quant.len() != d_bottom`
    ///   or `top_quant.len() != d_top`.
    pub fn decode(&self, bottom_quant: &[f64], top_quant: &[f64]) -> GenResult<Vec<f64>> {
        if bottom_quant.len() != self.config.d_bottom {
            return Err(GenError::DimensionMismatch {
                expected: self.config.d_bottom,
                got: bottom_quant.len(),
            });
        }
        if top_quant.len() != self.config.d_top {
            return Err(GenError::DimensionMismatch {
                expected: self.config.d_top,
                got: top_quant.len(),
            });
        }

        let d_input = self.config.d_input;
        let d_concat = self.config.d_bottom + self.config.d_top;

        let mut concat = Vec::with_capacity(d_concat);
        concat.extend_from_slice(bottom_quant);
        concat.extend_from_slice(top_quant);

        let out = (0..d_input)
            .map(|i| {
                let mut acc = self.dec_b[i];
                for (j, &cj) in concat.iter().enumerate() {
                    acc += self.dec_w[i * d_concat + j] * cj;
                }
                acc
            })
            .collect();
        Ok(out)
    }

    /// Return the bottom codebook embedding dimensionality.
    #[must_use]
    #[inline]
    pub fn d_bottom(&self) -> usize {
        self.config.d_bottom
    }

    /// Return the top codebook embedding dimensionality.
    #[must_use]
    #[inline]
    pub fn d_top(&self) -> usize {
        self.config.d_top
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    /// Compute `tanh(W · x + b)`.
    ///
    /// `w` is row-major `out_dim × in_dim`, `x` has length `in_dim`.
    fn linear_tanh(&self, w: &[f64], b: &[f64], x: &[f64], out_dim: usize) -> Vec<f64> {
        let in_dim = x.len();
        (0..out_dim)
            .map(|i| {
                let mut acc = b[i];
                for j in 0..in_dim {
                    acc += w[i * in_dim + j] * x[j];
                }
                acc.tanh()
            })
            .collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> VqVae2Config {
        VqVae2Config {
            d_input: 8,
            d_bottom: 4,
            d_top: 3,
            n_bottom: 8,
            n_top: 4,
            commitment_weight: 0.25,
            ema_decay: 0.99,
        }
    }

    fn make_model() -> VqVae2 {
        let config = make_config();
        let mut rng = GenRng::new(42);
        VqVae2::new(config, &mut rng).expect("new should succeed")
    }

    fn make_codebook(n: usize, d: usize) -> VqCodebookEma {
        let mut rng = GenRng::new(7);
        VqCodebookEma::new(n, d, 0.99, &mut rng).expect("new should succeed")
    }

    // ─── VqCodebookEma tests ─────────────────────────────────────────────────

    #[test]
    fn vq_codebook_quantize_shape() {
        let cb = make_codebook(8, 4);
        let input = vec![0.5_f64; 4 * 3]; // batch=3
        let (q, idx, _) = cb.quantize(&input, 3).expect("quantize should succeed");
        assert_eq!(q.len(), 4 * 3, "quantised shape should be batch*d_emb");
        assert_eq!(idx.len(), 3, "indices length should equal batch");
    }

    #[test]
    fn vq_codebook_commit_loss_nonneg() {
        let cb = make_codebook(8, 4);
        let input = vec![1.0_f64; 4 * 2]; // batch=2
        let (_, _, commit_loss) = cb.quantize(&input, 2).expect("quantize should succeed");
        assert!(
            commit_loss >= 0.0,
            "commit_loss must be non-negative: {commit_loss}"
        );
    }

    #[test]
    fn vq_codebook_indices_valid() {
        let n = 8_usize;
        let d = 4_usize;
        let cb = make_codebook(n, d);
        let mut rng = GenRng::new(13);
        let input: Vec<f64> = (0..d * 5)
            .map(|_| {
                let (a, _) = rng.next_normal_pair();
                a as f64
            })
            .collect();
        let (_, indices, _) = cb.quantize(&input, 5).expect("quantize should succeed");
        for &idx in &indices {
            assert!(idx < n, "index {idx} out of bounds (n_entries={n})");
        }
    }

    #[test]
    fn vq_codebook_ema_update_ok() {
        let mut cb = make_codebook(4, 3);
        let input = vec![0.5_f64; 3 * 2]; // batch=2
        let (_, indices, _) = cb.quantize(&input, 2).expect("quantize should succeed");
        let result = cb.ema_update(&input, &indices, 2);
        assert!(result.is_ok(), "EMA update should succeed: {result:?}");
    }

    #[test]
    fn vq_codebook_n_entries_2_works() {
        // Minimum valid codebook size
        let mut rng = GenRng::new(99);
        let cb = VqCodebookEma::new(2, 4, 0.99, &mut rng).expect("new should succeed");
        assert_eq!(cb.n_entries(), 2);
        let input = vec![0.1_f64; 4];
        let (q, idx, _) = cb.quantize(&input, 1).expect("quantize should succeed");
        assert_eq!(q.len(), 4);
        assert!(idx[0] < 2);
    }

    #[test]
    fn vq_codebook_n_entries_1_error() {
        let mut rng = GenRng::new(1);
        let err = VqCodebookEma::new(1, 4, 0.99, &mut rng);
        assert!(
            matches!(err, Err(GenError::InvalidCodebookSize(1))),
            "expected InvalidCodebookSize(1), got: {err:?}"
        );
    }

    // ─── VqVae2 tests ────────────────────────────────────────────────────────

    #[test]
    fn vq_vae2_encode_shape() {
        let model = make_model();
        let x = vec![0.1_f64; 8];
        let (bq, tq, _, _, _) = model.encode(&x).expect("encode should succeed");
        assert_eq!(bq.len(), 4, "bottom_quant should have d_bottom=4 elements");
        assert_eq!(tq.len(), 3, "top_quant should have d_top=3 elements");
    }

    #[test]
    fn vq_vae2_decode_shape() {
        let model = make_model();
        let bq = vec![0.2_f64; 4];
        let tq = vec![0.3_f64; 3];
        let x_hat = model.decode(&bq, &tq).expect("decode should succeed");
        assert_eq!(
            x_hat.len(),
            8,
            "decoded output should have d_input=8 elements"
        );
    }

    #[test]
    fn vq_vae2_loss_nonneg() {
        let model = make_model();
        let x = vec![0.5_f64; 8];
        let (_, _, _, _, loss) = model.encode(&x).expect("encode should succeed");
        assert!(loss >= 0.0, "commitment loss must be non-negative: {loss}");
    }

    #[test]
    fn vq_vae2_encode_finite() {
        let model = make_model();
        let x = vec![1.0_f64; 8];
        let (bq, tq, _, _, loss) = model.encode(&x).expect("encode should succeed");
        assert!(
            bq.iter().all(|v| v.is_finite()),
            "bottom_quant must be finite"
        );
        assert!(tq.iter().all(|v| v.is_finite()), "top_quant must be finite");
        assert!(loss.is_finite(), "loss must be finite: {loss}");
    }

    #[test]
    fn vq_vae2_decode_finite() {
        let model = make_model();
        let x = vec![0.3_f64; 8];
        let (bq, tq, _, _, _) = model.encode(&x).expect("encode should succeed");
        let x_hat = model.decode(&bq, &tq).expect("decode should succeed");
        assert!(
            x_hat.iter().all(|v| v.is_finite()),
            "decoded output must be finite"
        );
    }

    #[test]
    fn vq_vae2_d_accessors() {
        let model = make_model();
        assert_eq!(model.d_bottom(), 4);
        assert_eq!(model.d_top(), 3);
    }

    #[test]
    fn vq_vae2_encode_dim_mismatch_error() {
        let model = make_model();
        let x = vec![0.1_f64; 5]; // wrong length
        let err = model.encode(&x);
        assert!(
            matches!(
                err,
                Err(GenError::DimensionMismatch {
                    expected: 8,
                    got: 5
                })
            ),
            "expected DimensionMismatch{{8,5}}, got: {err:?}"
        );
    }

    #[test]
    fn vq_vae2_decode_dim_mismatch_error() {
        let model = make_model();
        let bq = vec![0.1_f64; 3]; // wrong: should be 4
        let tq = vec![0.2_f64; 3];
        let err = model.decode(&bq, &tq);
        assert!(
            matches!(
                err,
                Err(GenError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "expected DimensionMismatch{{4,3}}, got: {err:?}"
        );
    }
}
