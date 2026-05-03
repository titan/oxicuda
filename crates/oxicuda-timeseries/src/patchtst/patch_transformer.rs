//! PatchTST encoder (Nie et al. 2023).
//!
//! Each variate is processed independently (channel independence).
//! Patches are embedded, positionally encoded with sinusoidal PE,
//! then passed through a stack of pre-LN Transformer layers.
//! A per-variate linear head maps the flattened patch representations
//! to the forecast horizon.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;
use crate::patch::PatchEmbed1d;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Full configuration for a PatchTST encoder.
#[derive(Debug, Clone)]
pub struct PatchTstConfig {
    /// Number of variates (channels).
    pub c: usize,
    /// Input sequence length.
    pub t: usize,
    /// Forecast horizon (steps).
    pub horizon: usize,
    /// Length of each patch (e.g. 16).
    pub patch_len: usize,
    /// Stride between consecutive patches (e.g. 8).
    pub stride: usize,
    /// Token embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of Transformer layers.
    pub n_layers: usize,
    /// FFN hidden expansion factor (applied to `d_model`).
    pub ffn_expansion: usize,
}

impl PatchTstConfig {
    /// Small configuration: `d=64, heads=4, layers=2, expansion=4`.
    pub fn tiny(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            patch_len: 16,
            stride: 8,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            ffn_expansion: 4,
        }
    }

    /// Standard configuration: `d=128, heads=8, layers=3, expansion=4`.
    pub fn base(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            patch_len: 16,
            stride: 8,
            d_model: 128,
            n_heads: 8,
            n_layers: 3,
            ffn_expansion: 4,
        }
    }
}

// ─── Layer weights ────────────────────────────────────────────────────────────

/// All learnable parameters for one pre-LN Transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights {
    /// LayerNorm scale before MHSA `[D]`.
    pub norm1_g: Vec<f32>,
    /// LayerNorm bias before MHSA `[D]`.
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
    /// FFN first layer weight `[D * expansion, D]`.
    pub ff_w1: Vec<f32>,
    /// FFN first layer bias `[D * expansion]`.
    pub ff_b1: Vec<f32>,
    /// FFN second layer weight `[D, D * expansion]`.
    pub ff_w2: Vec<f32>,
    /// FFN second layer bias `[D]`.
    pub ff_b2: Vec<f32>,
}

impl TransformerLayerWeights {
    fn new(d: usize, expansion: usize, rng: &mut LcgRng) -> Self {
        let mut init_mat = |rows: usize, cols: usize| -> Vec<f32> {
            let scale = (6.0_f32 / (cols + rows) as f32).sqrt();
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };

        let d_ff = d * expansion;
        let norm1_g = vec![1.0_f32; d];
        let norm1_b = vec![0.0_f32; d];
        let q_w = init_mat(d, d);
        let k_w = init_mat(d, d);
        let v_w = init_mat(d, d);
        let out_w = init_mat(d, d);
        let norm2_g = vec![1.0_f32; d];
        let norm2_b = vec![0.0_f32; d];
        let ff_w1 = init_mat(d_ff, d);
        let ff_b1 = vec![0.0_f32; d_ff];
        let ff_w2 = init_mat(d, d_ff);
        let ff_b2 = vec![0.0_f32; d];

        Self {
            norm1_g,
            norm1_b,
            q_w,
            k_w,
            v_w,
            out_w,
            norm2_g,
            norm2_b,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
        }
    }
}

// ─── PatchTST model ───────────────────────────────────────────────────────────

/// PatchTST forecasting model.
///
/// Each variate is embedded as a sequence of patches, encoded by a Transformer,
/// then projected to the forecast horizon via a per-variate linear head.
#[derive(Debug, Clone)]
pub struct PatchTst {
    /// Patch embedding (shared across variates).
    pub embed: PatchEmbed1d,
    /// Transformer layer weights.
    pub layers: Vec<TransformerLayerWeights>,
    /// Head weight `[C * horizon, num_patches * d_model]`.
    pub head_w: Vec<f32>,
    /// Head bias `[C * horizon]`.
    pub head_b: Vec<f32>,
    /// Model configuration.
    pub config: PatchTstConfig,
}

impl PatchTst {
    /// Build a PatchTST model from config, initialising all weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    /// - [`TsError::InvalidSequenceLength`] when `t < patch_len`.
    pub fn new(config: PatchTstConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: config.d_model,
                n_heads: config.n_heads,
            });
        }
        if config.t < config.patch_len {
            return Err(TsError::InvalidSequenceLength(config.t));
        }

        let embed = PatchEmbed1d::new(config.patch_len, config.stride, config.d_model, rng)?;
        let np = embed.num_patches(config.t);

        let layers = (0..config.n_layers)
            .map(|_| TransformerLayerWeights::new(config.d_model, config.ffn_expansion, rng))
            .collect();

        let flat_dim = np * config.d_model;
        let head_out = config.c * config.horizon;
        let head_scale = (6.0_f32 / (flat_dim + config.horizon) as f32).sqrt();
        let mut head_w = vec![0.0_f32; head_out * flat_dim];
        rng.fill_normal(&mut head_w);
        for w in &mut head_w {
            *w *= head_scale;
        }
        let head_b = vec![0.0_f32; head_out];

        Ok(Self {
            embed,
            layers,
            head_w,
            head_b,
            config,
        })
    }

    /// Forecast `x: [T, C]` → `[horizon, C]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    /// - [`TsError::InvalidSequenceLength`] when `t < patch_len`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let cfg = &self.config;
        let expected = cfg.t * cfg.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let np = self.embed.num_patches(cfg.t);
        let d = cfg.d_model;
        let flat_dim = np * d;
        let pe = sinusoidal_pos_enc(np, d);

        let mut forecast = vec![0.0_f32; cfg.horizon * cfg.c];

        for ci in 0..cfg.c {
            let series: Vec<f32> = (0..cfg.t).map(|ti| x[ti * cfg.c + ci]).collect();
            let mut tokens = self.embed.forward(&series)?;

            for p in 0..np {
                for di in 0..d {
                    tokens[p * d + di] += pe[p * d + di];
                }
            }

            for lw in &self.layers {
                let attn_out = mhsa_forward(&tokens, np, d, lw, cfg.n_heads);
                for i in 0..tokens.len() {
                    tokens[i] += attn_out[i];
                }
                let ffn_out = ffn_forward(&tokens, np, d, lw, cfg.ffn_expansion);
                for i in 0..tokens.len() {
                    tokens[i] += ffn_out[i];
                }
            }

            let head_row_start = ci * cfg.horizon;
            for hi in 0..cfg.horizon {
                let row = head_row_start + hi;
                let val = self.head_b[row]
                    + self.head_w[row * flat_dim..(row + 1) * flat_dim]
                        .iter()
                        .zip(tokens.iter())
                        .map(|(&w, &t)| w * t)
                        .sum::<f32>();
                forecast[hi * cfg.c + ci] = val;
            }
        }

        Ok(forecast)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// In-place layer normalisation over the last dimension, eps=1e-5.
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let d = gamma.len();
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

/// Sinusoidal positional encoding `[np, d]`.
fn sinusoidal_pos_enc(np: usize, d: usize) -> Vec<f32> {
    let mut pe = vec![0.0_f32; np * d];
    for p in 0..np {
        for i in 0..d / 2 {
            let freq = 10000.0_f32.powf((2 * i) as f32 / d as f32);
            pe[p * d + 2 * i] = (p as f32 / freq).sin();
            pe[p * d + 2 * i + 1] = (p as f32 / freq).cos();
        }
        if d % 2 == 1 {
            let i = d / 2;
            let freq = 10000.0_f32.powf((2 * i) as f32 / d as f32);
            pe[p * d + 2 * i] = (p as f32 / freq).sin();
        }
    }
    pe
}

/// Multi-head self-attention over a `[np, d]` token sequence.
///
/// Pre-LN: normalise input, compute attention, return the delta (caller adds residual).
fn mhsa_forward(
    x: &[f32],
    np: usize,
    d: usize,
    lw: &TransformerLayerWeights,
    n_heads: usize,
) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut normed = x.to_vec();
    layer_norm(&mut normed, &lw.norm1_g, &lw.norm1_b);

    let project = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; np * d];
        for p in 0..np {
            for di in 0..d {
                let mut acc = 0.0_f32;
                for k in 0..d {
                    acc += normed[p * d + k] * w[di * d + k];
                }
                out[p * d + di] = acc;
            }
        }
        out
    };

    let q = project(&lw.q_w);
    let k = project(&lw.k_w);
    let v = project(&lw.v_w);

    let mut attn_out = vec![0.0_f32; np * d];

    for h in 0..n_heads {
        let h_start = h * head_dim;

        let mut scores = vec![0.0_f32; np * np];
        for qi in 0..np {
            for ki in 0..np {
                let mut dot = 0.0_f32;
                for hd in 0..head_dim {
                    dot += q[qi * d + h_start + hd] * k[ki * d + h_start + hd];
                }
                scores[qi * np + ki] = dot * scale;
            }
        }

        for qi in 0..np {
            softmax_row(&mut scores[qi * np..(qi + 1) * np]);
        }

        for qi in 0..np {
            for hd in 0..head_dim {
                let mut acc = 0.0_f32;
                for ki in 0..np {
                    acc += scores[qi * np + ki] * v[ki * d + h_start + hd];
                }
                attn_out[qi * d + h_start + hd] += acc;
            }
        }
    }

    let mut out = vec![0.0_f32; np * d];
    for p in 0..np {
        for di in 0..d {
            let mut acc = 0.0_f32;
            for k in 0..d {
                acc += attn_out[p * d + k] * lw.out_w[di * d + k];
            }
            out[p * d + di] = acc;
        }
    }
    out
}

/// FFN block over a `[np, d]` token sequence.
///
/// Pre-LN: normalise input, apply FFN, return the delta (caller adds residual).
fn ffn_forward(
    x: &[f32],
    np: usize,
    d: usize,
    lw: &TransformerLayerWeights,
    expansion: usize,
) -> Vec<f32> {
    let d_ff = d * expansion;

    let mut normed = x.to_vec();
    layer_norm(&mut normed, &lw.norm2_g, &lw.norm2_b);

    let mut hidden = vec![0.0_f32; np * d_ff];
    for p in 0..np {
        for fi in 0..d_ff {
            let mut acc = lw.ff_b1[fi];
            for k in 0..d {
                acc += normed[p * d + k] * lw.ff_w1[fi * d + k];
            }
            hidden[p * d_ff + fi] = gelu_exact(acc);
        }
    }

    let mut out = vec![0.0_f32; np * d];
    for p in 0..np {
        for di in 0..d {
            let mut acc = lw.ff_b2[di];
            for fi in 0..d_ff {
                acc += hidden[p * d_ff + fi] * lw.ff_w2[di * d_ff + fi];
            }
            out[p * d + di] = acc;
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn patchtst_tiny_output_shape() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig::tiny(3, 64, 12);
        let model = PatchTst::new(cfg.clone(), &mut rng).expect("build");
        let x: Vec<f32> = (0..cfg.t * cfg.c).map(|i| i as f32 * 0.01).collect();
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn patchtst_base_output_shape() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig::base(2, 96, 24);
        let model = PatchTst::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn patchtst_output_finite() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig::tiny(4, 64, 8);
        let model = PatchTst::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.t * cfg.c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn patchtst_horizon_first_layout() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig::tiny(2, 64, 6);
        let model = PatchTst::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![1.0_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
        // [horizon=6, C=2] → len=12
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn patchtst_error_invalid_num_heads() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig {
            n_heads: 0,
            ..PatchTstConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchTst::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
    }

    #[test]
    fn patchtst_error_head_dim_mismatch() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig {
            d_model: 65,
            n_heads: 4,
            ..PatchTstConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchTst::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    #[test]
    fn patchtst_error_seq_too_short() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig {
            t: 10,
            patch_len: 16,
            ..PatchTstConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchTst::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(10)
        ));
    }

    #[test]
    fn patchtst_error_bad_input_len() {
        let mut rng = make_rng();
        let cfg = PatchTstConfig::tiny(2, 64, 8);
        let model = PatchTst::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.0_f32; 50]; // wrong size
        assert!(matches!(
            model.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn sinusoidal_pe_orthogonal_positions() {
        let pe = sinusoidal_pos_enc(8, 16);
        let row0 = &pe[0..16];
        let row1 = &pe[16..32];
        let dot: f32 = row0.iter().zip(row1.iter()).map(|(a, b)| a * b).sum();
        let norm0: f32 = row0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm1: f32 = row1.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (norm0 * norm1 + 1e-8);
        assert!(
            cosine < 0.99,
            "adjacent PE rows are too similar: cosine={cosine}"
        );
    }
}
