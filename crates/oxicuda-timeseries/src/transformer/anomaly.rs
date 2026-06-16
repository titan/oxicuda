//! Anomaly Transformer — association discrepancy for unsupervised anomaly detection.
//!
//! Reference: Xu et al. 2022, "Anomaly Transformer: Time Series Anomaly Detection
//! with Association Discrepancy", ICLR 2022.
//!
//! Core idea:
//! - **Prior association** (Gaussian kernel): P^prior_{t,i} ∝ exp(-|t-i|²/(2σ²)),
//!   normalised to a proper distribution over i.
//! - **Series association** (learned attention): standard scaled dot-product attention
//!   over embedded time series.
//! - **Association discrepancy**: KL(P^prior ‖ P^series) + KL(P^series ‖ P^prior).
//! - Anomaly score = mean association discrepancy over all positions t.
//!
//! Anomalies exhibit *large* discrepancy because:
//! - Normal points can be well-described by their neighbours (Gaussian prior), while
//! - Anomalous points attract long-range attention in P^series.
//!
//! # Simplified CPU reference implementation
//!
//! No multi-head attention depth — one single-head attention per block for clarity.
//! The full architecture stacks L blocks; here we expose one block plus a forward
//! that computes both the reconstructed output and the anomaly score.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an Anomaly Transformer block.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.  Must divide `d_model`.
    pub n_heads: usize,
    /// Gaussian kernel bandwidth σ for the prior association.
    pub sigma: f64,
}

// ─── AnomalyTransformer ───────────────────────────────────────────────────────

/// Single-block Anomaly Transformer.
///
/// Produces a reconstructed output tensor and a per-position anomaly score.
#[derive(Debug, Clone)]
pub struct AnomalyTransformer {
    config: AnomalyConfig,
    /// Query projection `[d_model × d_model]`.
    wq: Vec<f32>,
    /// Key projection `[d_model × d_model]`.
    wk: Vec<f32>,
    /// Value projection `[d_model × d_model]`.
    wv: Vec<f32>,
    /// Output projection `[d_model × d_model]`.
    wo: Vec<f32>,
    /// FFN layer-1 weights `[d_ffn × d_model]`.
    w_ffn1: Vec<f32>,
    b_ffn1: Vec<f32>,
    /// FFN layer-2 weights `[d_model × d_ffn]`.
    w_ffn2: Vec<f32>,
    b_ffn2: Vec<f32>,
    d_ffn: usize,
}

/// Result of a single Anomaly Transformer forward pass.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Reconstructed output tensor `[seq_len × d_model]` (flattened).
    pub output: Vec<f32>,
    /// Per-position association discrepancy `[seq_len]` (higher → more anomalous).
    pub anomaly_score: Vec<f32>,
    /// Series association matrix `[seq_len × seq_len]` (flattened, row = query).
    pub series_assoc: Vec<f32>,
    /// Prior association matrix `[seq_len × seq_len]` (flattened, row = query).
    pub prior_assoc: Vec<f32>,
}

impl AnomalyTransformer {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new block with random initialisation.
    pub fn new(config: AnomalyConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if config.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: config.d_model,
                n_heads: config.n_heads,
            });
        }
        if config.sigma <= 0.0 {
            return Err(TsError::ShapeMismatch {
                msg: format!("sigma must be > 0, got {}", config.sigma),
            });
        }
        let d = config.d_model;
        let d_ffn = d * 4;
        let scale = (2.0_f32 / (d + d) as f32).sqrt();
        let mut mat = |rows: usize, cols: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; rows * cols];
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
            w_ffn1: mat(d_ffn, d),
            b_ffn1: vec![0.0f32; d_ffn],
            w_ffn2: mat(d, d_ffn),
            b_ffn2: vec![0.0f32; d],
            d_ffn,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Row-wise matrix-vector product: W[out × in] @ x[in] + b[out].
    fn linear(w: &[f32], b: &[f32], x: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
        let mut out = b.to_vec();
        for o in 0..out_d {
            let row = &w[o * in_d..(o + 1) * in_d];
            for k in 0..in_d {
                out[o] += row[k] * x[k];
            }
        }
        out
    }

    /// Project all rows of x[seq_len × d_in] via w[d_out × d_in]: out[seq_len × d_out].
    fn project(w: &[f32], x: &[f32], seq_len: usize, d_in: usize, d_out: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * d_out];
        let b = vec![0.0f32; d_out];
        for t in 0..seq_len {
            let row = Self::linear(w, &b, &x[t * d_in..(t + 1) * d_in], d_in, d_out);
            out[t * d_out..(t + 1) * d_out].copy_from_slice(&row);
        }
        out
    }

    /// Row-wise softmax on a flattened [rows × cols] matrix.
    fn row_softmax(m: &mut [f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let row = &mut m[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }

    /// Layer norm over a vector slice.
    fn layer_norm(v: &[f32]) -> Vec<f32> {
        let n = v.len();
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = (var + 1e-5).sqrt();
        v.iter().map(|&x| (x - mean) / std).collect()
    }

    /// ReLU activation.
    #[inline]
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    // ── Prior association ─────────────────────────────────────────────────────

    /// Build the Gaussian prior association matrix `[seq_len × seq_len]`.
    ///
    /// P^prior[t, i] ∝ exp(-|t-i|² / (2σ²)), row-normalised.
    fn prior_association(seq_len: usize, sigma: f64) -> Vec<f32> {
        let mut mat = vec![0.0f32; seq_len * seq_len];
        let two_sigma_sq = 2.0 * sigma * sigma;
        for t in 0..seq_len {
            let mut row_sum = 0.0f64;
            for i in 0..seq_len {
                let diff = (t as i64 - i as i64) as f64;
                let v = (-(diff * diff) / two_sigma_sq).exp();
                mat[t * seq_len + i] = v as f32;
                row_sum += v;
            }
            // Row-normalise
            if row_sum > 0.0 {
                for i in 0..seq_len {
                    mat[t * seq_len + i] /= row_sum as f32;
                }
            }
        }
        mat
    }

    // ── Association discrepancy ───────────────────────────────────────────────

    /// Symmetric KL divergence: KL(p ‖ q) + KL(q ‖ p) per row, then mean over rows.
    ///
    /// Returns per-position score `[seq_len]`.
    fn association_discrepancy(p_prior: &[f32], p_series: &[f32], seq_len: usize) -> Vec<f32> {
        let eps = 1e-8;
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut kl_pq = 0.0f64;
            let mut kl_qp = 0.0f64;
            for i in 0..seq_len {
                let prior = (p_prior[t * seq_len + i] as f64).max(eps);
                let series = (p_series[t * seq_len + i] as f64).max(eps);
                kl_pq += prior * (prior / series).ln();
                kl_qp += series * (series / prior).ln();
            }
            scores[t] = (kl_pq + kl_qp) as f32;
        }
        scores
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Forward pass.  `x`: `[seq_len × d_model]` flattened.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> TsResult<AnomalyResult> {
        let d = self.config.d_model;
        if x.len() != seq_len * d {
            return Err(TsError::DimensionMismatch {
                expected: seq_len * d,
                got: x.len(),
            });
        }
        if seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }

        // Q, K, V projections — [seq_len × d]
        let q = Self::project(&self.wq, x, seq_len, d, d);
        let k = Self::project(&self.wk, x, seq_len, d, d);
        let v = Self::project(&self.wv, x, seq_len, d, d);

        // Scaled dot-product attention scores: A[t,i] = Q[t,:] · K[i,:] / √d
        let scale = 1.0_f32 / (d as f32).sqrt();
        let mut attn = vec![0.0f32; seq_len * seq_len];
        for t in 0..seq_len {
            for i in 0..seq_len {
                let mut dot = 0.0f32;
                for k_d in 0..d {
                    dot += q[t * d + k_d] * k[i * d + k_d];
                }
                attn[t * seq_len + i] = dot * scale;
            }
        }
        // Row-wise softmax → series association P^series
        Self::row_softmax(&mut attn, seq_len, seq_len);
        let series_assoc = attn.clone();

        // Attend to V: out_attn[t, :] = Σ_i P^series[t,i] * V[i,:]
        let mut attn_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            for i in 0..seq_len {
                let w = attn[t * seq_len + i];
                for kd in 0..d {
                    attn_out[t * d + kd] += w * v[i * d + kd];
                }
            }
        }

        // Output projection + residual + LayerNorm
        let mut output = vec![0.0f32; seq_len * d];
        let b_zero = vec![0.0f32; d];
        for t in 0..seq_len {
            let proj = Self::linear(&self.wo, &b_zero, &attn_out[t * d..(t + 1) * d], d, d);
            // Residual
            let residual: Vec<f32> = proj
                .iter()
                .zip(&x[t * d..(t + 1) * d])
                .map(|(&p, &r)| p + r)
                .collect();
            let normed = Self::layer_norm(&residual);
            output[t * d..(t + 1) * d].copy_from_slice(&normed);
        }

        // FFN + residual + LayerNorm
        for t in 0..seq_len {
            let h1: Vec<f32> = Self::linear(
                &self.w_ffn1,
                &self.b_ffn1,
                &output[t * d..(t + 1) * d],
                d,
                self.d_ffn,
            )
            .into_iter()
            .map(Self::relu)
            .collect();
            let h2 = Self::linear(&self.w_ffn2, &self.b_ffn2, &h1, self.d_ffn, d);
            let residual: Vec<f32> = h2
                .iter()
                .zip(&output[t * d..(t + 1) * d])
                .map(|(&p, &r)| p + r)
                .collect();
            let normed = Self::layer_norm(&residual);
            output[t * d..(t + 1) * d].copy_from_slice(&normed);
        }

        // Prior association
        let prior_assoc = Self::prior_association(seq_len, self.config.sigma);

        // Association discrepancy
        let anomaly_score = Self::association_discrepancy(&prior_assoc, &series_assoc, seq_len);

        // Validate
        for &v in output.iter().chain(anomaly_score.iter()) {
            if !v.is_finite() {
                return Err(TsError::NonFinite);
            }
        }

        Ok(AnomalyResult {
            output,
            anomaly_score,
            series_assoc,
            prior_assoc,
        })
    }

    /// Embedding dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_block() -> AnomalyTransformer {
        let mut rng = make_rng();
        AnomalyTransformer::new(
            AnomalyConfig {
                d_model: 8,
                n_heads: 2,
                sigma: 3.0,
            },
            &mut rng,
        )
        .expect("construction failed")
    }

    fn random_input(seq_len: usize, d_model: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(77);
        (0..seq_len * d_model)
            .map(|_| rng.next_f32() - 0.5)
            .collect()
    }

    #[test]
    fn output_shape() {
        let block = default_block();
        let seq_len = 16;
        let x = random_input(seq_len, 8);
        let result = block.forward(&x, seq_len).expect("forward");
        assert_eq!(result.output.len(), seq_len * 8);
        assert_eq!(result.anomaly_score.len(), seq_len);
    }

    #[test]
    fn output_finite() {
        let block = default_block();
        let seq_len = 12;
        let x = random_input(seq_len, 8);
        let result = block.forward(&x, seq_len).expect("forward");
        for &v in result.output.iter().chain(result.anomaly_score.iter()) {
            assert!(v.is_finite(), "output has non-finite value: {v}");
        }
    }

    #[test]
    fn n_heads_zero_error() {
        let mut rng = make_rng();
        let result = AnomalyTransformer::new(
            AnomalyConfig {
                d_model: 8,
                n_heads: 0,
                sigma: 1.0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_heads=0 should fail");
    }

    #[test]
    fn d_model_not_divisible_error() {
        let mut rng = make_rng();
        let result = AnomalyTransformer::new(
            AnomalyConfig {
                d_model: 7,
                n_heads: 3,
                sigma: 1.0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "d_model=7, n_heads=3 should fail");
    }

    #[test]
    fn prior_assoc_rows_sum_to_one() {
        let block = default_block();
        let seq_len = 10;
        let x = random_input(seq_len, 8);
        let result = block.forward(&x, seq_len).expect("forward");
        for t in 0..seq_len {
            let row_sum: f32 = result.prior_assoc[t * seq_len..(t + 1) * seq_len]
                .iter()
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "prior row {t} sum={row_sum} should be 1.0"
            );
        }
    }

    #[test]
    fn series_assoc_rows_sum_to_one() {
        let block = default_block();
        let seq_len = 10;
        let x = random_input(seq_len, 8);
        let result = block.forward(&x, seq_len).expect("forward");
        for t in 0..seq_len {
            let row_sum: f32 = result.series_assoc[t * seq_len..(t + 1) * seq_len]
                .iter()
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "series row {t} sum={row_sum} should be 1.0"
            );
        }
    }

    #[test]
    fn anomaly_score_nonneg() {
        // KL divergence ≥ 0 by definition.
        let block = default_block();
        let seq_len = 8;
        let x = random_input(seq_len, 8);
        let result = block.forward(&x, seq_len).expect("forward");
        for (t, &s) in result.anomaly_score.iter().enumerate() {
            assert!(s >= 0.0, "anomaly score[{t}] = {s} should be >= 0");
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let block = default_block();
        // x has wrong d_model
        let x = vec![0.0f32; 5 * 6]; // d=6 instead of d=8
        let result = block.forward(&x, 5);
        assert!(result.is_err(), "wrong d_model should return Err");
    }

    #[test]
    fn seq_len_zero_error() {
        let block = default_block();
        let result = block.forward(&[], 0);
        assert!(result.is_err(), "seq_len=0 should return Err");
    }

    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let result = AnomalyTransformer::new(
            AnomalyConfig {
                d_model: 0,
                n_heads: 1,
                sigma: 1.0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "d_model=0 should fail");
    }

    #[test]
    fn sigma_zero_error() {
        let mut rng = make_rng();
        let result = AnomalyTransformer::new(
            AnomalyConfig {
                d_model: 8,
                n_heads: 2,
                sigma: 0.0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "sigma=0 should fail");
    }

    #[test]
    fn d_model_accessor() {
        let block = default_block();
        assert_eq!(block.d_model(), 8);
    }
}
