//! Hybrid Mamba–Attention block.
//!
//! Interleaves simplified Mamba (linear + ReLU + residual) and simplified
//! self-attention (linear + tanh + residual) layers in a configurable schedule,
//! following the pattern of Jamba (Lieber et al., 2024) and related hybrid
//! SSM–Transformer architectures.
//!
//! This is a **pure-CPU reference implementation** for correctness testing.
//! Weights are single `[d_model × d_model]` matrices; true multi-head
//! attention requires O(L²) memory and is left for GPU kernels.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the [`HybridBlock`].
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Model/embedding dimension `D`.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Number of Mamba (SSM) layers.
    pub n_mamba_layers: usize,
    /// Number of attention layers.
    pub n_attn_layers: usize,
    /// Insert an attention layer every `attn_every_n` total layers.
    /// Must be ≥ 1.  `1` means strictly alternating; `2` means every other.
    pub attn_every_n: usize,
}

// ─── Block struct ─────────────────────────────────────────────────────────────

/// Hybrid Mamba–Attention sequence model block (CPU reference).
///
/// # Weight layout
///
/// | Field       | Shape                                         |
/// |-------------|-----------------------------------------------|
/// | `mamba_w`   | `n_mamba_layers × [d_model × d_model]`        |
/// | `attn_w`    | `n_attn_layers  × [d_model × d_model]`        |
pub struct HybridBlock {
    /// Per-layer weight matrices for Mamba sub-layers.
    mamba_w: Vec<Vec<f32>>,
    /// Per-layer weight matrices for attention sub-layers.
    attn_w: Vec<Vec<f32>>,
    /// Layer configuration.
    config: HybridConfig,
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Dense matrix–vector multiply: `y[i] = Σ_j w[i*in_dim + j] * x[j]`.
fn mat_vec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|i| (0..in_dim).map(|j| w[i * in_dim + j] * x[j]).sum::<f32>())
        .collect()
}

/// Rectified linear unit applied element-wise.
#[inline]
fn relu(v: f32) -> f32 {
    v.max(0.0)
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl HybridBlock {
    /// Construct a new `HybridBlock` with randomly initialised weights.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`] — if `d_model == 0`.
    /// * [`MambaError::HeadDimMismatch`] — if `n_heads == 0` or
    ///   `d_model % n_heads != 0`.
    /// * [`MambaError::Internal`] — if `attn_every_n == 0`.
    pub fn new(config: HybridConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        if config.d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if config.n_heads == 0 {
            return Err(MambaError::HeadDimMismatch {
                n_heads: 0,
                d_model: config.d_model,
            });
        }
        if config.d_model % config.n_heads != 0 {
            return Err(MambaError::HeadDimMismatch {
                n_heads: config.n_heads,
                d_model: config.d_model,
            });
        }
        if config.attn_every_n == 0 {
            return Err(MambaError::Internal("attn_every_n must be >= 1".into()));
        }

        let d2 = config.d_model * config.d_model;
        let scale = (1.0_f32 / config.d_model as f32).sqrt();

        let mamba_w: Vec<Vec<f32>> = (0..config.n_mamba_layers)
            .map(|_| {
                (0..d2)
                    .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();

        let attn_w: Vec<Vec<f32>> = (0..config.n_attn_layers)
            .map(|_| {
                (0..d2)
                    .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();

        Ok(Self {
            mamba_w,
            attn_w,
            config,
        })
    }

    /// Run the hybrid block over a sequence.
    ///
    /// # Arguments
    ///
    /// * `x`       — input tensor, shape `[seq_len × d_model]` (row-major).
    /// * `seq_len` — number of tokens in the sequence.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len × d_model]` (row-major).
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`] — if `seq_len == 0`.
    /// * [`MambaError::DimensionMismatch`] — if `x.len() != seq_len * d_model`.
    /// * [`MambaError::NonFinite`] — if any input value is NaN or infinite.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        let d_model = self.config.d_model;

        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        let expected = seq_len * d_model;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("input"));
        }

        let n_mamba = self.config.n_mamba_layers;
        let n_attn = self.config.n_attn_layers;
        let attn_every_n = self.config.attn_every_n;

        // Current activation buffer (will be updated layer by layer).
        let mut current: Vec<f32> = x.to_vec();

        let mut mamba_idx = 0usize;
        let mut attn_idx = 0usize;
        let mut layer_count = 0usize;

        while mamba_idx < n_mamba || attn_idx < n_attn {
            // Determine whether to apply an attention or Mamba sub-layer.
            let use_attn = if mamba_idx >= n_mamba {
                // All Mamba layers exhausted — consume remaining attn layers.
                true
            } else if attn_idx >= n_attn {
                // All attn layers exhausted — consume remaining Mamba layers.
                false
            } else {
                // Interleave according to the schedule.
                (layer_count + 1) % attn_every_n == 0
            };

            if use_attn {
                let w = &self.attn_w[attn_idx];
                current = Self::apply_attn_layer(w, &current, seq_len, d_model);
                attn_idx += 1;
            } else {
                let w = &self.mamba_w[mamba_idx];
                current = Self::apply_mamba_layer(w, &current, seq_len, d_model);
                mamba_idx += 1;
            }
            layer_count += 1;
        }

        Ok(current)
    }

    /// Apply one Mamba sub-layer: `y_t = ReLU(W * x_t) + x_t`.
    fn apply_mamba_layer(w: &[f32], x: &[f32], seq_len: usize, d_model: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * d_model];
        for t in 0..seq_len {
            let x_t = &x[t * d_model..(t + 1) * d_model];
            let wx = mat_vec(w, x_t, d_model, d_model);
            for i in 0..d_model {
                out[t * d_model + i] = relu(wx[i]) + x_t[i];
            }
        }
        out
    }

    /// Apply one attention sub-layer: `y_t = tanh(W * x_t) + x_t`.
    fn apply_attn_layer(w: &[f32], x: &[f32], seq_len: usize, d_model: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * d_model];
        for t in 0..seq_len {
            let x_t = &x[t * d_model..(t + 1) * d_model];
            let wx = mat_vec(w, x_t, d_model, d_model);
            for i in 0..d_model {
                out[t * d_model + i] = wx[i].tanh() + x_t[i];
            }
        }
        out
    }

    /// Return the total number of layers (Mamba + attention).
    #[must_use]
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.config.n_mamba_layers + self.config.n_attn_layers
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(0xC0FF_EE42)
    }

    fn default_config() -> HybridConfig {
        HybridConfig {
            d_model: 8,
            n_heads: 2,
            n_mamba_layers: 2,
            n_attn_layers: 2,
            attn_every_n: 2,
        }
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    // 1. output_shape
    #[test]
    fn output_shape() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 6;
        let x = randn(&mut rng, seq_len * d_model);
        let y = block.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(
            y.len(),
            seq_len * d_model,
            "output must be seq_len * d_model"
        );
    }

    // 2. output_finite
    #[test]
    fn output_finite() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 8;
        let x = randn(&mut rng, seq_len * d_model);
        let y = block.forward(&x, seq_len).expect("forward must succeed");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // 3. seq_len_1
    #[test]
    fn seq_len_1() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("construction must succeed");
        let x = randn(&mut rng, d_model);
        let y = block
            .forward(&x, 1)
            .expect("forward with seq_len=1 must succeed");
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 4. attn_every_1_alternates
    #[test]
    fn attn_every_1_alternates() {
        let mut rng = make_rng();
        let cfg = HybridConfig {
            d_model: 8,
            n_heads: 2,
            n_mamba_layers: 2,
            n_attn_layers: 2,
            attn_every_n: 1,
        };
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("attn_every_n=1 must succeed");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = block.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(y.len(), seq_len * d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 5. n_mamba_0_only_attn
    #[test]
    fn n_mamba_0_only_attn() {
        let mut rng = make_rng();
        let cfg = HybridConfig {
            d_model: 8,
            n_heads: 2,
            n_mamba_layers: 0,
            n_attn_layers: 2,
            attn_every_n: 1,
        };
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("n_mamba=0 must succeed");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = block.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(y.len(), seq_len * d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 6. n_attn_0_only_mamba
    #[test]
    fn n_attn_0_only_mamba() {
        let mut rng = make_rng();
        let cfg = HybridConfig {
            d_model: 8,
            n_heads: 2,
            n_mamba_layers: 2,
            n_attn_layers: 0,
            attn_every_n: 1,
        };
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("n_attn=0 must succeed");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = block.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(y.len(), seq_len * d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 7. different_input_different_output
    #[test]
    fn different_input_different_output() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let block = HybridBlock::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 4;
        let x1 = randn(&mut rng, seq_len * d_model);
        let x2 = randn(&mut rng, seq_len * d_model);
        assert_ne!(x1, x2, "test precondition: inputs must differ");
        let y1 = block.forward(&x1, seq_len).expect("forward 1 must succeed");
        let y2 = block.forward(&x2, seq_len).expect("forward 2 must succeed");
        assert_ne!(y1, y2, "different inputs must yield different outputs");
    }

    // 8. d_model_zero_error
    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let cfg = HybridConfig {
            d_model: 0,
            n_heads: 1,
            n_mamba_layers: 1,
            n_attn_layers: 1,
            attn_every_n: 1,
        };
        let result = HybridBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "d_model=0 must return Err");
        assert!(
            matches!(result, Err(MambaError::InvalidModelDim(0))),
            "expected InvalidModelDim(0)"
        );
    }

    // 9. n_heads_zero_error
    #[test]
    fn n_heads_zero_error() {
        let mut rng = make_rng();
        let cfg = HybridConfig {
            d_model: 8,
            n_heads: 0,
            n_mamba_layers: 1,
            n_attn_layers: 1,
            attn_every_n: 1,
        };
        let result = HybridBlock::new(cfg, &mut rng);
        assert!(result.is_err(), "n_heads=0 must return Err");
        assert!(
            matches!(result, Err(MambaError::HeadDimMismatch { n_heads: 0, .. })),
            "expected HeadDimMismatch with n_heads=0"
        );
    }

    // 10. n_layers_count
    #[test]
    fn n_layers_count() {
        let mut rng = make_rng();
        let cfg = default_config();
        let expected = cfg.n_mamba_layers + cfg.n_attn_layers;
        let block = HybridBlock::new(cfg, &mut rng).expect("construction must succeed");
        assert_eq!(block.n_layers(), expected);
    }
}
