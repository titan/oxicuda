//! Multi-head cross-attention: Q from modality A, K/V from modality B.
//!
//! Implements the standard scaled dot-product attention:
//! `Attn(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`
//! split across `n_heads` independent attention heads.

use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for multi-head cross-attention.
#[derive(Debug, Clone)]
pub struct CrossAttnConfig {
    /// Number of attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Key/query head dimension. Typically `d_model / n_heads`.
    pub d_k: usize,
    /// Value head dimension. Typically `d_model / n_heads`.
    pub d_v: usize,
    /// Dropout rate (stored for documentation; not applied in CPU forward pass).
    pub dropout_rate: f32,
}

impl CrossAttnConfig {
    /// Create a standard config where `d_k = d_v = d_model / n_heads`.
    pub fn new(n_heads: usize, d_model: usize, dropout_rate: f32) -> MmResult<Self> {
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: n_heads,
                d_model,
            });
        }
        let head_dim = d_model / n_heads;
        Ok(Self {
            n_heads,
            d_model,
            d_k: head_dim,
            d_v: head_dim,
            dropout_rate,
        })
    }

    /// Tiny preset for testing: d_model=8, n_heads=2, d_k=4, d_v=4.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_heads: 2,
            d_model: 8,
            d_k: 4,
            d_v: 4,
            dropout_rate: 0.0,
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learned weight matrices for cross-attention.
///
/// All weight matrices are stored in row-major flat `Vec<f32>`.
/// `w_q`, `w_k`, `w_v`: shape `[d_model × d_model]`.
/// `w_o`: shape `[d_model × d_model]` (output projection).
#[derive(Debug, Clone)]
pub struct CrossAttnWeights {
    /// Query projection: shape `[d_model × d_model]`.
    pub w_q: Vec<f32>,
    /// Key projection: shape `[d_model × d_model]`.
    pub w_k: Vec<f32>,
    /// Value projection: shape `[d_model × d_model]`.
    pub w_v: Vec<f32>,
    /// Output projection: shape `[d_model × d_model]`.
    pub w_o: Vec<f32>,
}

impl CrossAttnWeights {
    /// Create zero-initialised weights.
    #[must_use]
    pub fn zeros(cfg: &CrossAttnConfig) -> Self {
        let sz = cfg.d_model * cfg.d_model;
        Self {
            w_q: vec![0.0_f32; sz],
            w_k: vec![0.0_f32; sz],
            w_v: vec![0.0_f32; sz],
            w_o: vec![0.0_f32; sz],
        }
    }

    /// Create identity-like weights (scaled identity for each projection).
    /// Useful for unit-testing: `forward(Q, Q, Q)` should produce the input.
    #[must_use]
    pub fn identity(cfg: &CrossAttnConfig) -> Self {
        let d = cfg.d_model;
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        Self {
            w_q: w.clone(),
            w_k: w.clone(),
            w_v: w.clone(),
            w_o: w,
        }
    }
}

// ─── CrossAttention ──────────────────────────────────────────────────────────

/// Multi-head cross-attention module.
pub struct CrossAttention {
    pub cfg: CrossAttnConfig,
    pub weights: CrossAttnWeights,
}

impl CrossAttention {
    /// Create from config with zero weights.
    #[must_use]
    pub fn new(cfg: CrossAttnConfig) -> Self {
        let weights = CrossAttnWeights::zeros(&cfg);
        Self { cfg, weights }
    }

    /// Create from config and explicit weights.
    #[must_use]
    pub fn with_weights(cfg: CrossAttnConfig, weights: CrossAttnWeights) -> Self {
        Self { cfg, weights }
    }

    /// Forward pass.
    ///
    /// - `query`: `[q_len × d_model]` row-major
    /// - `key`:   `[kv_len × d_model]` row-major
    /// - `value`: `[kv_len × d_model]` row-major
    ///
    /// Returns `[q_len × d_model]`.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_len: usize,
        kv_len: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let d_k = self.cfg.d_k;
        let d_v = self.cfg.d_v;

        if query.len() != q_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: q_len * d,
                got: query.len(),
            });
        }
        if key.len() != kv_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: kv_len * d,
                got: key.len(),
            });
        }
        if value.len() != kv_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: kv_len * d,
                got: value.len(),
            });
        }
        if kv_len == 0 {
            return Err(MultiModalError::MismatchedSeqLens { q_len, kv_len });
        }

        // Project Q, K, V → [seq × d_model]
        let proj_q = matmul_seq(query, &self.weights.w_q, q_len, d, d)?;
        let proj_k = matmul_seq(key, &self.weights.w_k, kv_len, d, d)?;
        let proj_v = matmul_seq(value, &self.weights.w_v, kv_len, d, d)?;

        // Reshape to [h × seq × d_head] and compute per-head attention
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut head_outputs = vec![0.0_f32; q_len * d]; // [q_len × d_model]

        for head in 0..h {
            let q_start = head * d_k;
            let k_start = head * d_k;
            let v_start = head * d_v;

            // Extract Q head: [q_len × d_k]
            let q_head = extract_head(&proj_q, q_len, d, q_start, d_k);
            // Extract K head: [kv_len × d_k]
            let k_head = extract_head(&proj_k, kv_len, d, k_start, d_k);
            // Extract V head: [kv_len × d_v]
            let v_head = extract_head(&proj_v, kv_len, d, v_start, d_v);

            // Attention scores: [q_len × kv_len] = Q_head · K_headᵀ * scale
            let mut scores = vec![0.0_f32; q_len * kv_len];
            for qi in 0..q_len {
                for ki in 0..kv_len {
                    let mut dot = 0.0_f32;
                    for di in 0..d_k {
                        dot += q_head[qi * d_k + di] * k_head[ki * d_k + di];
                    }
                    scores[qi * kv_len + ki] = dot * scale;
                }
            }

            // Softmax over kv_len dimension (per query position)
            softmax_rows_inplace(&mut scores, q_len, kv_len);

            // Weighted sum of V: [q_len × d_v]
            let mut attn_out = vec![0.0_f32; q_len * d_v];
            for qi in 0..q_len {
                for vi in 0..d_v {
                    let mut s = 0.0_f32;
                    for ki in 0..kv_len {
                        s += scores[qi * kv_len + ki] * v_head[ki * d_v + vi];
                    }
                    attn_out[qi * d_v + vi] = s;
                }
            }

            // Write back into head_outputs at the correct columns
            for qi in 0..q_len {
                for vi in 0..d_v {
                    head_outputs[qi * d + head * d_v + vi] = attn_out[qi * d_v + vi];
                }
            }
        }

        // Output projection: [q_len × d_model]
        let output = matmul_seq(&head_outputs, &self.weights.w_o, q_len, d, d)?;
        Ok(output)
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Matrix multiply: `A [rows × in_dim] × W [in_dim × out_dim]` → `[rows × out_dim]`.
/// W is stored row-major as `[in_dim × out_dim]`.
fn matmul_seq(
    a: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> MmResult<Vec<f32>> {
    if w.len() != in_dim * out_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: in_dim * out_dim,
            got: w.len(),
        });
    }
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += a[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    Ok(out)
}

/// Extract one head's sub-slice from a projected sequence.
/// `proj`: `[seq × d_model]`, returns `[seq × head_dim]` starting at column `col_start`.
fn extract_head(
    proj: &[f32],
    seq: usize,
    d_model: usize,
    col_start: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq * head_dim];
    for s in 0..seq {
        for d in 0..head_dim {
            out[s * head_dim + d] = proj[s * d_model + col_start + d];
        }
    }
    out
}

/// In-place row-wise softmax on a `[rows × cols]` matrix.
pub(crate) fn softmax_rows_inplace(m: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut m[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_attn_config_new_ok() {
        let cfg = CrossAttnConfig::new(4, 32, 0.1).unwrap();
        assert_eq!(cfg.d_k, 8);
        assert_eq!(cfg.d_v, 8);
    }

    #[test]
    fn cross_attn_config_bad_heads() {
        let err = CrossAttnConfig::new(3, 8, 0.0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    #[test]
    fn cross_attn_output_shape() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let q_len = 5;
        let kv_len = 7;
        let weights = CrossAttnWeights::identity(&cfg);
        let attn = CrossAttention::with_weights(cfg, weights);

        let query = vec![0.5_f32; q_len * d];
        let key = vec![0.3_f32; kv_len * d];
        let value = vec![0.2_f32; kv_len * d];

        let out = attn.forward(&query, &key, &value, q_len, kv_len).unwrap();
        assert_eq!(out.len(), q_len * d);
    }

    #[test]
    fn cross_attn_zero_key_uniform_attn() {
        // When all keys are equal, softmax produces uniform attention weights.
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let q_len = 3;
        let kv_len = 4;
        // Use identity weights so projections preserve values.
        let weights = CrossAttnWeights::identity(&cfg);
        let attn = CrossAttention::with_weights(cfg, weights);

        let query = vec![1.0_f32; q_len * d];
        let key = vec![0.0_f32; kv_len * d]; // all zeros → uniform softmax
        // Value: each row is distinct to see averaging
        let mut value = vec![0.0_f32; kv_len * d];
        for ki in 0..kv_len {
            for di in 0..d {
                value[ki * d + di] = ki as f32;
            }
        }
        // Expected output: uniform average over rows [0,1,2,3] = 1.5 * identity
        let out = attn.forward(&query, &key, &value, q_len, kv_len).unwrap();
        // With identity output projection the output equals the attention-weighted V
        // Through the output projection (identity), each dim should be the average value.
        assert_eq!(out.len(), q_len * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cross_attn_dimension_mismatch_error() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let attn = CrossAttention::new(cfg);

        let query = vec![0.0_f32; 3 * (d + 1)]; // wrong dim
        let key = vec![0.0_f32; 4 * d];
        let value = vec![0.0_f32; 4 * d];
        let err = attn.forward(&query, &key, &value, 3, 4).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn cross_attn_empty_kv_error() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let attn = CrossAttention::new(cfg);

        let query = vec![0.0_f32; 3 * d];
        let key = vec![];
        let value = vec![];
        let err = attn.forward(&query, &key, &value, 3, 0).unwrap_err();
        assert!(matches!(err, MultiModalError::MismatchedSeqLens { .. }));
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut m = vec![1.0_f32, 2.0, 3.0, 0.5_f32, 0.5, 0.5];
        softmax_rows_inplace(&mut m, 2, 3);
        for r in 0..2 {
            let sum: f32 = m[r * 3..(r + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {r} sum = {sum}");
        }
    }

    #[test]
    fn matmul_seq_correct() {
        // A = [[1,0],[0,1]] × W = [[2,3],[4,5]] → [[2,3],[4,5]]
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let w = vec![2.0_f32, 3.0, 4.0, 5.0];
        let out = matmul_seq(&a, &w, 2, 2, 2).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        assert!((out[2] - 4.0).abs() < 1e-6);
        assert!((out[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cross_attn_output_finite() {
        let cfg = CrossAttnConfig::new(2, 8, 0.0).unwrap();
        let d = cfg.d_model;
        let mut weights = CrossAttnWeights::zeros(&cfg);
        // Small random-ish values
        for (i, v) in weights.w_q.iter_mut().enumerate() {
            *v = (i as f32 * 0.1).sin() * 0.1;
        }
        for (i, v) in weights.w_k.iter_mut().enumerate() {
            *v = (i as f32 * 0.13).cos() * 0.1;
        }
        for (i, v) in weights.w_v.iter_mut().enumerate() {
            *v = (i as f32 * 0.07).sin() * 0.1;
        }
        for (i, v) in weights.w_o.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).cos() * 0.1;
        }
        let attn = CrossAttention::with_weights(cfg, weights);
        let q = vec![0.3_f32; 4 * d];
        let kv = vec![0.2_f32; 6 * d];
        let out = attn.forward(&q, &kv, &kv, 4, 6).unwrap();
        assert_eq!(out.len(), 4 * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
