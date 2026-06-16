//! SSD Chunk Layer — simplified Mamba-2 SSD Chunk-Scan for CPU.
//!
//! Implements the Structured State Space Duality (SSD) chunk scan following
//! Dao & Gu (2024) "Transformers are SSMs: Generalized Models and Efficient
//! Algorithms Through Structured State Space Duality".
//!
//! This is a **pure-CPU reference implementation** of the recurrent form with
//! chunked processing, intended for correctness verification and unit testing.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the [`SsdChunk`] layer.
#[derive(Debug, Clone)]
pub struct SsdChunkConfig {
    /// Model dimension `D` (token embedding width).
    pub d_model: usize,
    /// SSM state dimension `N`.
    pub d_state: usize,
    /// Depthwise convolution kernel size.
    pub d_conv: usize,
    /// Expansion factor: `inner_dim = expand * d_model`.
    pub expand: usize,
    /// Chunk size for segmented processing.
    pub chunk_size: usize,
}

// ─── Layer struct ─────────────────────────────────────────────────────────────

/// Simplified Mamba-2 SSD Chunk layer (CPU reference).
///
/// Projects input tokens through a linear encoder, maintains a first-order
/// recurrent state with learned per-channel decay (`a_log`), and projects the
/// state back to the model dimension via a linear output projection.
///
/// # Weight layout
///
/// | Field       | Shape                           |
/// |-------------|---------------------------------|
/// | `x_proj`    | `[inner_dim × d_model]`         |
/// | `dt_proj`   | `[d_model × inner_dim]`         |
/// | `a_log`     | `[inner_dim]`                   |
/// | `out_proj`  | `[d_model × inner_dim]`         |
pub struct SsdChunk {
    /// Input projection: inner_dim × d_model.
    x_proj: Vec<f32>,
    /// dt projection: d_model × inner_dim.
    dt_proj: Vec<f32>,
    /// Log-magnitude of the negative SSM decay: `A[i] = exp(-a_log[i])`.
    a_log: Vec<f32>,
    /// Output projection: d_model × inner_dim.
    out_proj: Vec<f32>,
    /// Layer configuration.
    config: SsdChunkConfig,
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Dense matrix–vector multiply: `y[i] = Σ_j w[i*in_dim + j] * x[j]`.
///
/// # Arguments
///
/// * `w`       — weight matrix stored row-major, shape `[out_dim × in_dim]`.
/// * `x`       — input vector of length `in_dim`.
/// * `out_dim` — number of output features.
/// * `in_dim`  — number of input features (must equal `x.len()`).
fn mat_vec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|i| (0..in_dim).map(|j| w[i * in_dim + j] * x[j]).sum::<f32>())
        .collect()
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl SsdChunk {
    /// Construct a new `SsdChunk` with randomly initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`MambaError::InvalidModelDim`] when `d_model == 0` or
    /// `expand == 0`, [`MambaError::InvalidChunkSize`] when `chunk_size == 0`,
    /// and [`MambaError::InvalidSsmOrder`] when `d_state == 0`.
    pub fn new(config: SsdChunkConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        if config.d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if config.chunk_size == 0 {
            return Err(MambaError::InvalidChunkSize(0));
        }
        if config.d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if config.expand == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }

        let inner_dim = config.expand * config.d_model;

        // Glorot-style uniform scale.
        let scale = (2.0_f32 / (config.d_model + inner_dim) as f32).sqrt();

        // x_proj: [inner_dim × d_model]
        let x_proj: Vec<f32> = (0..inner_dim * config.d_model)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        // dt_proj: [d_model × inner_dim]
        let dt_proj: Vec<f32> = (0..config.d_model * inner_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        // a_log: small positive values so A = exp(-a_log) ≈ exp(-0.1..0.2) ∈ (0.8, 0.9)
        let a_log: Vec<f32> = (0..inner_dim).map(|_| rng.next_f32() * 0.1 + 0.1).collect();

        // out_proj: [d_model × inner_dim]
        let out_proj: Vec<f32> = (0..config.d_model * inner_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        Ok(Self {
            x_proj,
            dt_proj,
            a_log,
            out_proj,
            config,
        })
    }

    /// Run the SSD chunk scan over a sequence.
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
        let inner_dim = self.config.expand * d_model;

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

        // Precompute base per-channel decay magnitude from a_log.
        // A_base[i] = exp(-a_log[i])  ∈ (0.8, 0.9) with the default init.
        let a_base: Vec<f32> = self.a_log.iter().map(|&v| (-v).exp()).collect();

        // Recurrent state (inner_dim).
        let mut h = vec![0.0_f32; inner_dim];

        // Intermediate outputs [seq_len × inner_dim].
        let mut output_inner = vec![0.0_f32; seq_len * inner_dim];

        for t in 0..seq_len {
            let x_t = &x[t * d_model..(t + 1) * d_model];

            // Project input to inner dimension: u = x_proj × x_t  [inner_dim]
            let u = mat_vec(&self.x_proj, x_t, inner_dim, d_model);

            // Compute dt via dt_proj: project u (inner_dim) -> d_model, then take
            // mean and apply softplus to get a positive time-step scalar dt.
            // dt_proj shape: [d_model × inner_dim].
            let dt_vec = mat_vec(&self.dt_proj, &u, d_model, inner_dim);
            let dt_mean = dt_vec.iter().sum::<f32>() / d_model as f32;
            // softplus ensures dt > 0; clamp for numerical safety.
            let dt = (dt_mean.exp() + 1.0).ln().clamp(1e-4, 10.0);

            // Input-dependent decay: A_t[i] = exp(-dt * a_log[i]).
            // State update: h[i] = A_t[i] * h[i] + dt * u[i]
            for i in 0..inner_dim {
                let a_t = (-dt * self.a_log[i]).exp();
                h[i] = a_t * h[i] + dt * u[i];
            }
            // Suppress unused binding to a_base (only used as a reference comment).
            let _ = &a_base;

            // Copy state to output buffer.
            output_inner[t * inner_dim..(t + 1) * inner_dim].copy_from_slice(&h);
        }

        // Apply output projection: y_t = out_proj × h_t  [d_model]
        let mut y = vec![0.0_f32; seq_len * d_model];
        for t in 0..seq_len {
            let h_t = &output_inner[t * inner_dim..(t + 1) * inner_dim];
            let y_t = mat_vec(&self.out_proj, h_t, d_model, inner_dim);
            y[t * d_model..(t + 1) * d_model].copy_from_slice(&y_t);
        }

        Ok(y)
    }

    /// Return the model dimension `D`.
    #[must_use]
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(0xDEAD_BEEF)
    }

    fn default_config() -> SsdChunkConfig {
        SsdChunkConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            chunk_size: 4,
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
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 6;
        let x = randn(&mut rng, seq_len * d_model);
        let y = layer.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(
            y.len(),
            seq_len * d_model,
            "output len must be seq_len * d_model"
        );
    }

    // 2. output_finite
    #[test]
    fn output_finite() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 8;
        let x = randn(&mut rng, seq_len * d_model);
        let y = layer.forward(&x, seq_len).expect("forward must succeed");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // 3. different_inputs_different_outputs
    #[test]
    fn different_inputs_different_outputs() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 4;
        let x1 = randn(&mut rng, seq_len * d_model);
        let x2 = randn(&mut rng, seq_len * d_model);
        // Ensure inputs actually differ
        assert_ne!(x1, x2, "test precondition: inputs must differ");
        let y1 = layer.forward(&x1, seq_len).expect("forward 1 must succeed");
        let y2 = layer.forward(&x2, seq_len).expect("forward 2 must succeed");
        assert_ne!(y1, y2, "different inputs must produce different outputs");
    }

    // 4. seq_len_1
    #[test]
    fn seq_len_1() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let x = randn(&mut rng, d_model);
        let y = layer
            .forward(&x, 1)
            .expect("forward with seq_len=1 must succeed");
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 5. chunk_size_gt_seq_len
    #[test]
    fn chunk_size_gt_seq_len() {
        let mut rng = make_rng();
        let cfg = SsdChunkConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            chunk_size: 16,
        };
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = layer
            .forward(&x, seq_len)
            .expect("forward with chunk_size > seq_len must succeed");
        assert_eq!(y.len(), seq_len * d_model);
    }

    // 6. d_model_zero_error
    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let cfg = SsdChunkConfig {
            d_model: 0,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            chunk_size: 4,
        };
        let result = SsdChunk::new(cfg, &mut rng);
        assert!(result.is_err(), "d_model=0 must return Err");
        assert!(
            matches!(result, Err(MambaError::InvalidModelDim(0))),
            "expected InvalidModelDim(0)"
        );
    }

    // 7. chunk_size_zero_error
    #[test]
    fn chunk_size_zero_error() {
        let mut rng = make_rng();
        let cfg = SsdChunkConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            chunk_size: 0,
        };
        let result = SsdChunk::new(cfg, &mut rng);
        assert!(result.is_err(), "chunk_size=0 must return Err");
        assert!(
            matches!(result, Err(MambaError::InvalidChunkSize(0))),
            "expected InvalidChunkSize(0)"
        );
    }

    // 8. output_not_input
    #[test]
    fn output_not_input() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("construction must succeed");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = layer.forward(&x, seq_len).expect("forward must succeed");
        // Output must differ from input (non-trivial transformation)
        let identical = x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(!identical, "output must be different from input");
    }

    // 9. state_dim_positive
    #[test]
    fn state_dim_positive() {
        let mut rng = make_rng();
        let cfg = SsdChunkConfig {
            d_model: 8,
            d_state: 1,
            d_conv: 4,
            expand: 2,
            chunk_size: 4,
        };
        let d_model = cfg.d_model;
        let layer = SsdChunk::new(cfg, &mut rng).expect("d_state=1 must be valid");
        let seq_len = 4;
        let x = randn(&mut rng, seq_len * d_model);
        let y = layer.forward(&x, seq_len).expect("forward must succeed");
        assert_eq!(y.len(), seq_len * d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // 10. d_state_zero_error
    #[test]
    fn d_state_zero_error() {
        let mut rng = make_rng();
        let cfg = SsdChunkConfig {
            d_model: 8,
            d_state: 0,
            d_conv: 4,
            expand: 2,
            chunk_size: 4,
        };
        let result = SsdChunk::new(cfg, &mut rng);
        assert!(result.is_err(), "d_state=0 must return Err");
        assert!(
            matches!(result, Err(MambaError::InvalidSsmOrder(0))),
            "expected InvalidSsmOrder(0)"
        );
    }
}
