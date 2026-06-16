//! Autoformer Auto-Correlation attention mechanism.
//!
//! Reference: Wu et al. 2021, "Autoformer: Decomposition Transformers with
//! Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021.
//!
//! Instead of dot-product attention, Autoformer uses **time-delay autocorrelation**:
//! for each head, it computes the cross-correlation between queries and keys in
//! the time domain, selects the top-k lags by their mean correlation amplitude,
//! and aggregates shifted value sequences (circular roll) weighted by softmax
//! over the top-k correlation values.
//!
//! # Simplified DFT-free formulation
//!
//! For correctness without a complex FFT dependency:
//! ```text
//! R_qk[τ] = Σ_t Q_h[t] · K_h[(t + τ) mod T]   (for τ in 0..k)
//! ```
//! This is O(T · k · d_h) and exact for the time-delay aggregation step.
//!
//! # Forward pass
//! 1. Q = x @ W_q,  K = x @ W_k,  V = x @ W_v   [seq_len × d_model]
//! 2. Split into `n_heads` heads of size `d_h = d_model / n_heads`.
//! 3. For each head:
//!    a. Compute correlation `R[τ]` for τ in 0..k (k = max(factor * ln(T), 1))
//!    b. Apply softmax over top-k correlations to get weights.
//!    c. Aggregate: agg_h = Σ_τ softmax_weight_τ * roll(V_h, τ)
//! 4. Concatenate heads, apply W_o.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Autoformer auto-correlation attention block.
#[derive(Debug, Clone)]
pub struct AutocorrConfig {
    /// Total embedding dimension.  Must be divisible by `n_heads`.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Top-k period selection factor.  k = max(factor * ln(seq_len), 1).
    pub factor: usize,
}

// ─── AutocorrelationBlock ────────────────────────────────────────────────────

/// Autoformer auto-correlation attention block.
///
/// Weights layout: `[d_model × d_model]` row-major (output-row, input-col).
#[derive(Debug, Clone)]
pub struct AutocorrelationBlock {
    config: AutocorrConfig,
    /// Query projection weights `[d_model × d_model]`.
    wq: Vec<f32>,
    /// Key projection weights `[d_model × d_model]`.
    wk: Vec<f32>,
    /// Value projection weights `[d_model × d_model]`.
    wv: Vec<f32>,
    /// Output projection weights `[d_model × d_model]`.
    wo: Vec<f32>,
}

impl AutocorrelationBlock {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new block, validating config and initialising weights.
    pub fn new(config: AutocorrConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if config.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: config.d_model,
                n_heads: config.n_heads,
            });
        }
        let d = config.d_model;
        let scale = (2.0_f32 / (d + d) as f32).sqrt();
        let mut mat = |_rows: usize, _cols: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; d * d];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        Ok(Self {
            config,
            wq: mat(d, d),
            wk: mat(d, d),
            wv: mat(d, d),
            wo: mat(d, d),
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Embedding dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Number of attention heads.
    pub fn n_heads(&self) -> usize {
        self.config.n_heads
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Apply a `[d × d]` linear projection (no bias) to `[seq_len × d]`.
    /// Output layout: `[seq_len × d]`.
    fn project(w: &[f32], x: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let x_row = &x[t * d..(t + 1) * d];
            for o in 0..d {
                let w_row = &w[o * d..(o + 1) * d];
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += w_row[k] * x_row[k];
                }
                out[t * d + o] = acc;
            }
        }
        out
    }

    /// Compute time-delay autocorrelation between q_h and k_h for lags 0..k.
    ///
    /// `q_h`, `k_h`: `[seq_len × d_h]`.
    /// Returns correlation vector of length `k`.
    fn autocorrelation(q_h: &[f32], k_h: &[f32], seq_len: usize, d_h: usize, k: usize) -> Vec<f32> {
        let mut corr = vec![0.0f32; k];
        for (tau, c_out) in corr.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for t in 0..seq_len {
                let k_idx = (t + tau) % seq_len;
                // Dot product of q_h[t, :] with k_h[k_idx, :]
                let q_row = &q_h[t * d_h..(t + 1) * d_h];
                let k_row = &k_h[k_idx * d_h..(k_idx + 1) * d_h];
                for d in 0..d_h {
                    sum += q_row[d] * k_row[d];
                }
            }
            *c_out = sum / seq_len as f32;
        }
        corr
    }

    /// Circularly shift v_h (layout `[seq_len × d_h]`) by `tau` positions.
    fn roll(v_h: &[f32], seq_len: usize, d_h: usize, tau: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * d_h];
        for t in 0..seq_len {
            let src = (t + tau) % seq_len;
            let dst_row = &mut out[t * d_h..(t + 1) * d_h];
            let src_row = &v_h[src * d_h..(src + 1) * d_h];
            dst_row.copy_from_slice(src_row);
        }
        out
    }

    /// Softmax over a slice (returns new Vec).
    fn softmax(x: &[f32]) -> Vec<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for e in &mut exps {
                *e /= sum;
            }
        }
        exps
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /// Auto-correlation forward pass.
    ///
    /// `x`: flattened `[seq_len × d_model]` (row-major).
    ///
    /// Returns flattened `[seq_len × d_model]`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> TsResult<Vec<f32>> {
        let d = self.config.d_model;
        let h = self.config.n_heads;
        let d_h = d / h;

        // Validate input size
        if x.len() != seq_len * d {
            return Err(TsError::DimensionMismatch {
                expected: seq_len * d,
                got: x.len(),
            });
        }
        if seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }

        // k = max(factor * ln(seq_len), 1) — at least 1 lag
        let k = ((self.config.factor as f32 * (seq_len as f32).ln()).ceil() as usize).max(1);
        let k = k.min(seq_len); // can't have more lags than seq_len

        // Project to Q, K, V — all [seq_len × d]
        let q = Self::project(&self.wq, x, seq_len, d);
        let k_proj = Self::project(&self.wk, x, seq_len, d);
        let v_proj = Self::project(&self.wv, x, seq_len, d);

        // Process each head independently
        let mut concat_heads = vec![0.0f32; seq_len * d];

        for head in 0..h {
            // Extract head slice: q_h[t, :] = q[t, head*d_h .. (head+1)*d_h]
            let mut q_h = vec![0.0f32; seq_len * d_h];
            let mut kk_h = vec![0.0f32; seq_len * d_h];
            let mut v_h = vec![0.0f32; seq_len * d_h];
            for t in 0..seq_len {
                q_h[t * d_h..(t + 1) * d_h]
                    .copy_from_slice(&q[t * d + head * d_h..t * d + (head + 1) * d_h]);
                kk_h[t * d_h..(t + 1) * d_h]
                    .copy_from_slice(&k_proj[t * d + head * d_h..t * d + (head + 1) * d_h]);
                v_h[t * d_h..(t + 1) * d_h]
                    .copy_from_slice(&v_proj[t * d + head * d_h..t * d + (head + 1) * d_h]);
            }

            // Compute autocorrelation for lags 0..k
            let corr = Self::autocorrelation(&q_h, &kk_h, seq_len, d_h, k);

            // Softmax over correlations to get per-lag weights
            let weights = Self::softmax(&corr);

            // Time-delay aggregation: agg = Σ_τ weight_τ * roll(V_h, τ)
            let mut agg_h = vec![0.0f32; seq_len * d_h];
            for (tau, &w) in weights.iter().enumerate() {
                let rolled = Self::roll(&v_h, seq_len, d_h, tau);
                for i in 0..seq_len * d_h {
                    agg_h[i] += w * rolled[i];
                }
            }

            // Write back into concat_heads at head offset
            for t in 0..seq_len {
                let dst = &mut concat_heads[t * d + head * d_h..t * d + (head + 1) * d_h];
                let src = &agg_h[t * d_h..(t + 1) * d_h];
                dst.copy_from_slice(src);
            }
        }

        // Output projection: [seq_len × d] @ W_o
        let out = Self::project(&self.wo, &concat_heads, seq_len, d);

        // Validate
        for &v in &out {
            if !v.is_finite() {
                return Err(TsError::NonFinite);
            }
        }

        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn make_block(d_model: usize, n_heads: usize, factor: usize) -> AutocorrelationBlock {
        let mut rng = make_rng();
        AutocorrelationBlock::new(
            AutocorrConfig {
                d_model,
                n_heads,
                factor,
            },
            &mut rng,
        )
        .expect("AutocorrelationBlock::new failed")
    }

    fn random_input(seq_len: usize, d_model: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(99);
        (0..seq_len * d_model)
            .map(|_| rng.next_f32() - 0.5)
            .collect()
    }

    #[test]
    fn output_shape() {
        let block = make_block(8, 2, 1);
        let seq_len = 16;
        let x = random_input(seq_len, 8);
        let out = block.forward(&x, seq_len).expect("forward");
        assert_eq!(out.len(), seq_len * 8, "output should be seq_len * d_model");
    }

    #[test]
    fn output_finite() {
        let block = make_block(8, 2, 1);
        let seq_len = 16;
        let x = random_input(seq_len, 8);
        let out = block.forward(&x, seq_len).expect("forward");
        for &v in &out {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn d_model_must_be_divisible() {
        let mut rng = make_rng();
        let result = AutocorrelationBlock::new(
            AutocorrConfig {
                d_model: 7,
                n_heads: 3,
                factor: 1,
            },
            &mut rng,
        );
        assert!(
            result.is_err(),
            "d_model=7, n_heads=3 should fail (not divisible)"
        );
    }

    #[test]
    fn single_head_works() {
        let block = make_block(8, 1, 1);
        let seq_len = 12;
        let x = random_input(seq_len, 8);
        let out = block.forward(&x, seq_len).expect("forward");
        assert_eq!(out.len(), seq_len * 8);
    }

    #[test]
    fn seq_len_1_works() {
        let block = make_block(4, 2, 1);
        let x = vec![0.5f32, -0.5, 1.0, -1.0]; // seq_len=1, d_model=4
        let out = block.forward(&x, 1).expect("forward seq_len=1");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn different_inputs_different_outputs() {
        let block = make_block(8, 2, 1);
        let seq_len = 8;
        let x1 = random_input(seq_len, 8);
        let mut rng2 = LcgRng::new(1234);
        let x2: Vec<f32> = (0..seq_len * 8).map(|_| rng2.next_f32() - 0.5).collect();
        let out1 = block.forward(&x1, seq_len).expect("forward x1");
        let out2 = block.forward(&x2, seq_len).expect("forward x2");
        let diff: f32 = out1.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "different inputs should produce different outputs"
        );
    }

    #[test]
    fn n_heads_zero_error() {
        let mut rng = make_rng();
        let result = AutocorrelationBlock::new(
            AutocorrConfig {
                d_model: 8,
                n_heads: 0,
                factor: 1,
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_heads=0 should return Err");
    }

    #[test]
    fn autocorr_block_not_identity() {
        let block = make_block(8, 2, 1);
        let seq_len = 10;
        let x = random_input(seq_len, 8);
        let out = block.forward(&x, seq_len).expect("forward");
        let diff: f32 = x.iter().zip(&out).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "output should differ from input after random init"
        );
    }

    #[test]
    fn d_model_mismatch_with_input_error() {
        let block = make_block(8, 2, 1);
        let seq_len = 4;
        // x has wrong d_model (6 instead of 8)
        let x = vec![0.0f32; seq_len * 6];
        let result = block.forward(&x, seq_len);
        assert!(result.is_err(), "mismatched x dimension should return Err");
    }

    #[test]
    fn d_model_accessor() {
        let block = make_block(16, 4, 2);
        assert_eq!(block.d_model(), 16);
        assert_eq!(block.n_heads(), 4);
    }
}
