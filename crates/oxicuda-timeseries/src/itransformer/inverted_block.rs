//! Inverted Transformer block (Liu et al. 2024).
//!
//! Attention is computed over the **variate axis** (C tokens, each of dim D)
//! rather than the time axis.  This captures multivariate correlations while
//! remaining independent of sequence length after the initial embedding step.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Weights ──────────────────────────────────────────────────────────────────

/// All learnable parameters for one inverted Transformer block.
#[derive(Debug, Clone)]
pub struct InvertedBlockWeights {
    /// LayerNorm scale before inverted MHSA `[D]`.
    pub norm1_g: Vec<f32>,
    /// LayerNorm bias before inverted MHSA `[D]`.
    pub norm1_b: Vec<f32>,
    /// Query projection `[D, D]`.
    pub q_w: Vec<f32>,
    /// Key projection `[D, D]`.
    pub k_w: Vec<f32>,
    /// Value projection `[D, D]`.
    pub v_w: Vec<f32>,
    /// Output projection `[D, D]`.
    pub out_w: Vec<f32>,
    /// LayerNorm scale before FFN `[D]`.
    pub norm2_g: Vec<f32>,
    /// LayerNorm bias before FFN `[D]`.
    pub norm2_b: Vec<f32>,
    /// FFN first layer weight `[4D, D]`.
    pub ff_w1: Vec<f32>,
    /// FFN first layer bias `[4D]`.
    pub ff_b1: Vec<f32>,
    /// FFN second layer weight `[D, 4D]`.
    pub ff_w2: Vec<f32>,
    /// FFN second layer bias `[D]`.
    pub ff_b2: Vec<f32>,
    /// Empty placeholder kept for structural symmetry.
    pub ff_b_empty_placeholder: Vec<f32>,
}

impl InvertedBlockWeights {
    fn new(d: usize, rng: &mut LcgRng) -> Self {
        let d_ff = d * 4;
        let mut init_mat = |rows: usize, cols: usize| -> Vec<f32> {
            let scale = (6.0_f32 / (cols + rows) as f32).sqrt();
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };

        Self {
            norm1_g: vec![1.0_f32; d],
            norm1_b: vec![0.0_f32; d],
            q_w: init_mat(d, d),
            k_w: init_mat(d, d),
            v_w: init_mat(d, d),
            out_w: init_mat(d, d),
            norm2_g: vec![1.0_f32; d],
            norm2_b: vec![0.0_f32; d],
            ff_w1: init_mat(d_ff, d),
            ff_b1: vec![0.0_f32; d_ff],
            ff_w2: init_mat(d, d_ff),
            ff_b2: vec![0.0_f32; d],
            ff_b_empty_placeholder: Vec::new(),
        }
    }
}

// ─── Block ────────────────────────────────────────────────────────────────────

/// One inverted Transformer block operating over the variate axis.
#[derive(Debug, Clone)]
pub struct InvertedBlock {
    /// Learnable parameters.
    pub weights: InvertedBlockWeights,
    /// Token dimension.
    pub d: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl InvertedBlock {
    /// Construct an `InvertedBlock`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidEmbedDim`] when `d == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d % n_heads != 0`.
    pub fn new(d: usize, n_heads: usize, rng: &mut LcgRng) -> TsResult<Self> {
        if d == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if d % n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: d,
                n_heads,
            });
        }
        let weights = InvertedBlockWeights::new(d, rng);
        Ok(Self {
            weights,
            d,
            n_heads,
        })
    }

    /// Apply the block to `tokens: [C, D]` → `[C, D]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `tokens.len() != c * d`.
    pub fn forward(&self, tokens: &[f32], c: usize) -> TsResult<Vec<f32>> {
        let expected = c * self.d;
        if tokens.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: tokens.len(),
            });
        }

        let attn_delta = inv_mhsa(tokens, c, self.d, &self.weights, self.n_heads);
        let mut after_attn: Vec<f32> = tokens
            .iter()
            .zip(attn_delta.iter())
            .map(|(a, b)| a + b)
            .collect();

        let ffn_delta = ffn_block(&after_attn, c, self.d, &self.weights);
        for i in 0..after_attn.len() {
            after_attn[i] += ffn_delta[i];
        }

        Ok(after_attn)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Normalise each row of `x: [N, D]` in-place.
fn layer_norm_vec(x: &mut [f32], d: usize, gamma: &[f32], beta: &[f32]) {
    if d == 0 {
        return;
    }
    let n = x.len() / d;
    for i in 0..n {
        let row = &mut x[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = (var + 1e-5_f32).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

/// GELU activation using the tanh approximation.
#[inline]
fn gelu_exact(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    let inner = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// In-place numerically stable softmax over a row.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = sum.recip();
    for v in row.iter_mut() {
        *v *= inv_sum;
    }
}

/// Multi-head self-attention over C variate tokens of dimension D.
///
/// Returns the attention output delta (pre-LN applied internally).
fn inv_mhsa(
    tokens: &[f32],
    c: usize,
    d: usize,
    w: &InvertedBlockWeights,
    n_heads: usize,
) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut normed = tokens.to_vec();
    layer_norm_vec(&mut normed, d, &w.norm1_g, &w.norm1_b);

    let project = |weight: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; c * d];
        for ci in 0..c {
            for di in 0..d {
                let mut acc = 0.0_f32;
                for k in 0..d {
                    acc += normed[ci * d + k] * weight[di * d + k];
                }
                out[ci * d + di] = acc;
            }
        }
        out
    };

    let q = project(&w.q_w);
    let k_mat = project(&w.k_w);
    let v = project(&w.v_w);

    let mut attn_out = vec![0.0_f32; c * d];

    for h in 0..n_heads {
        let h_start = h * head_dim;

        let mut scores = vec![0.0_f32; c * c];
        for qi in 0..c {
            for ki in 0..c {
                let mut dot = 0.0_f32;
                for hd in 0..head_dim {
                    dot += q[qi * d + h_start + hd] * k_mat[ki * d + h_start + hd];
                }
                scores[qi * c + ki] = dot * scale;
            }
        }

        for qi in 0..c {
            softmax_row(&mut scores[qi * c..(qi + 1) * c]);
        }

        for qi in 0..c {
            for hd in 0..head_dim {
                let mut acc = 0.0_f32;
                for ki in 0..c {
                    acc += scores[qi * c + ki] * v[ki * d + h_start + hd];
                }
                attn_out[qi * d + h_start + hd] += acc;
            }
        }
    }

    let mut out = vec![0.0_f32; c * d];
    for ci in 0..c {
        for di in 0..d {
            let mut acc = 0.0_f32;
            for k in 0..d {
                acc += attn_out[ci * d + k] * w.out_w[di * d + k];
            }
            out[ci * d + di] = acc;
        }
    }
    out
}

/// Row-wise FFN with GELU over `tokens: [C, D]`.
///
/// Returns the FFN output delta (pre-LN applied internally).
fn ffn_block(tokens: &[f32], c: usize, d: usize, w: &InvertedBlockWeights) -> Vec<f32> {
    let d_ff = d * 4;

    let mut normed = tokens.to_vec();
    layer_norm_vec(&mut normed, d, &w.norm2_g, &w.norm2_b);

    let mut hidden = vec![0.0_f32; c * d_ff];
    for ci in 0..c {
        for fi in 0..d_ff {
            let mut acc = w.ff_b1[fi];
            for k in 0..d {
                acc += normed[ci * d + k] * w.ff_w1[fi * d + k];
            }
            hidden[ci * d_ff + fi] = gelu_exact(acc);
        }
    }

    let mut out = vec![0.0_f32; c * d];
    for ci in 0..c {
        for di in 0..d {
            let mut acc = w.ff_b2[di];
            for fi in 0..d_ff {
                acc += hidden[ci * d_ff + fi] * w.ff_w2[di * d_ff + fi];
            }
            out[ci * d + di] = acc;
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    #[test]
    fn inverted_block_output_shape() {
        let mut rng = make_rng();
        let block = InvertedBlock::new(64, 4, &mut rng).expect("build");
        let c = 7;
        let tokens = vec![0.5_f32; c * 64];
        let out = block.forward(&tokens, c).expect("forward");
        assert_eq!(out.len(), c * 64);
    }

    #[test]
    fn inverted_block_output_finite() {
        let mut rng = make_rng();
        let block = InvertedBlock::new(32, 4, &mut rng).expect("build");
        let c = 5;
        let mut tokens = vec![0.0_f32; c * 32];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, c).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn inverted_block_residual_changes_output() {
        let mut rng = make_rng();
        let block = InvertedBlock::new(32, 4, &mut rng).expect("build");
        let c = 4;
        let mut tokens = vec![0.0_f32; c * 32];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, c).expect("forward");
        let same = out
            .iter()
            .zip(tokens.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(
            !same,
            "block output identical to input — residual not applied"
        );
    }

    #[test]
    fn inverted_block_error_zero_d() {
        let mut rng = make_rng();
        assert!(matches!(
            InvertedBlock::new(0, 4, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    #[test]
    fn inverted_block_error_zero_heads() {
        let mut rng = make_rng();
        assert!(matches!(
            InvertedBlock::new(64, 0, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
    }

    #[test]
    fn inverted_block_error_head_dim_mismatch() {
        let mut rng = make_rng();
        assert!(matches!(
            InvertedBlock::new(65, 4, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    #[test]
    fn inverted_block_error_token_len_mismatch() {
        let mut rng = make_rng();
        let block = InvertedBlock::new(32, 4, &mut rng).expect("build");
        let tokens = vec![0.0_f32; 50]; // wrong
        assert!(matches!(
            block.forward(&tokens, 3).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}
