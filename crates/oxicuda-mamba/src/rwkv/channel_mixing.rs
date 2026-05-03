//! RWKV channel-mixing layer — the FFN equivalent in RWKV-4.
//!
//! # Theory (Peng et al., 2023 — RWKV-4)
//!
//! Channel-mixing replaces the standard FFN by a gated feedforward network
//! using Square-ReLU as the non-linearity:
//!
//! ```text
//! Token shift:
//!   x̃_r[t] = time_mix_r ⊙ x[t] + (1 - time_mix_r) ⊙ x[t-1]
//!   x̃_k[t] = time_mix_k ⊙ x[t] + (1 - time_mix_k) ⊙ x[t-1]
//!
//! Projections:
//!   r[t] = sigmoid(W_r · x̃_r[t])                 — receptance gate ∈ (0,1)
//!   k[t] = W_k · x̃_k[t]                           — key (pre-activation), ∈ ℝ^{d_ffn}
//!   v[t] = W_v · square_relu(k[t])                 — gated value, ∈ ℝ^{d_model}
//!
//! Output:
//!   out[t] = r[t] ⊙ v[t]
//! ```
//!
//! where `square_relu(x) = max(0, x)²`.
//!
//! The token-shift ensures the model can look at the immediately previous token
//! without any explicit attention mechanism.
//!
//! ## Typical dimensions
//!
//! - `d_ffn = 4 * d_model` (matching the expansion ratio of standard FFNs)
//! - Weights: `W_r ∈ ℝ^{d_model × d_model}`, `W_k ∈ ℝ^{d_ffn × d_model}`,
//!   `W_v ∈ ℝ^{d_model × d_ffn}`

use crate::error::{MambaError, MambaResult};

// ─── square_relu ─────────────────────────────────────────────────────────────

/// Square-ReLU activation: `max(0, x)²`.
///
/// This is the non-linearity used in RWKV channel mixing.
/// It is smoother near zero than plain ReLU and scales quadratically,
/// increasing expressiveness without additional parameters.
#[inline]
#[must_use]
pub fn square_relu(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

// ─── ChannelMixingConfig ──────────────────────────────────────────────────────

/// Configuration for an RWKV channel-mixing layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMixingConfig {
    /// Input and output dimension `D`.
    pub d_model: usize,
    /// FFN hidden dimension (typically `4 * d_model`).
    pub d_ffn: usize,
}

impl ChannelMixingConfig {
    /// Create a new configuration with `d_ffn = 4 * d_model`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] if `d_model == 0`
    pub fn new(d_model: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        Ok(Self {
            d_model,
            d_ffn: 4 * d_model,
        })
    }

    /// Override the FFN hidden dimension.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] if `d_ffn == 0`
    pub fn with_d_ffn(mut self, d_ffn: usize) -> MambaResult<Self> {
        if d_ffn == 0 {
            return Err(MambaError::InvalidModelDim(d_ffn));
        }
        self.d_ffn = d_ffn;
        Ok(self)
    }
}

// ─── ChannelMixingWeights ────────────────────────────────────────────────────

/// Learned parameters for an RWKV channel-mixing layer.
#[derive(Debug, Clone)]
pub struct ChannelMixingWeights {
    /// Receptance projection `W_r`: `[d_model, d_model]` (row-major: row=out, col=in).
    pub w_r: Vec<f32>,
    /// Key projection `W_k`: `[d_ffn, d_model]` (row-major).
    pub w_k: Vec<f32>,
    /// Value (down-projection) `W_v`: `[d_model, d_ffn]` (row-major).
    pub w_v: Vec<f32>,
    /// Token-shift blend for receptance `μ_r`: `[d_model]`.
    pub time_mix_r: Vec<f32>,
    /// Token-shift blend for key `μ_k`: `[d_model]`.
    pub time_mix_k: Vec<f32>,
}

impl ChannelMixingWeights {
    /// Allocate all weight tensors as zeros.
    #[must_use]
    pub fn zeros(config: &ChannelMixingConfig) -> Self {
        let d = config.d_model;
        let f = config.d_ffn;
        Self {
            w_r: vec![0.0; d * d],
            w_k: vec![0.0; f * d],
            w_v: vec![0.0; d * f],
            time_mix_r: vec![0.0; d],
            time_mix_k: vec![0.0; d],
        }
    }

    /// Default-initialised weights:
    ///
    /// - `W_r` — scaled identity (`1/D` on diagonal)
    /// - `W_k` — zero matrix (safe expansion to `d_ffn`)
    /// - `W_v` — zero matrix (safe contraction back to `d_model`)
    /// - `time_mix_* = [0.5; D]`
    #[must_use]
    pub fn default_init(config: &ChannelMixingConfig) -> Self {
        let d = config.d_model;
        let f = config.d_ffn;
        let scale = if d > 0 { 1.0 / d as f32 } else { 1.0 };

        // Scaled identity for the receptance gate.
        let mut w_r = vec![0.0_f32; d * d];
        for i in 0..d {
            w_r[i * d + i] = scale;
        }

        // W_k: [d_ffn, d_model] — zero-initialised for safe default.
        let w_k = vec![0.0_f32; f * d];
        // W_v: [d_model, d_ffn] — zero-initialised for safe default.
        let w_v = vec![0.0_f32; d * f];

        Self {
            w_r,
            w_k,
            w_v,
            time_mix_r: vec![0.5; d],
            time_mix_k: vec![0.5; d],
        }
    }

    /// Randomly initialise weights with N(0, 1) samples scaled by `1/sqrt(D)`.
    pub fn random(config: &ChannelMixingConfig, rng: &mut crate::handle::LcgRng) -> Self {
        let d = config.d_model;
        let f = config.d_ffn;
        let scale_d = if d > 0 {
            (d as f32).sqrt().recip()
        } else {
            1.0
        };
        let scale_f = if f > 0 {
            (f as f32).sqrt().recip()
        } else {
            1.0
        };

        let sample = |rng: &mut crate::handle::LcgRng, n: usize, s: f32| -> Vec<f32> {
            let mut buf = vec![0.0_f32; n];
            rng.fill_normal(&mut buf);
            buf.iter_mut().for_each(|v| *v *= s);
            buf
        };

        Self {
            w_r: sample(rng, d * d, scale_d),
            w_k: sample(rng, f * d, scale_d),
            w_v: sample(rng, d * f, scale_f),
            time_mix_r: vec![0.5; d],
            time_mix_k: vec![0.5; d],
        }
    }
}

// ─── ChannelMixingLayer ──────────────────────────────────────────────────────

/// RWKV channel-mixing layer (gated feedforward with Square-ReLU).
pub struct ChannelMixingLayer {
    config: ChannelMixingConfig,
    weights: ChannelMixingWeights,
}

impl ChannelMixingLayer {
    /// Create a new channel-mixing layer, validating weight shapes.
    ///
    /// # Errors
    ///
    /// - [`MambaError::WeightShapeMismatch`] if any weight tensor has the wrong length.
    pub fn new(config: ChannelMixingConfig, weights: ChannelMixingWeights) -> MambaResult<Self> {
        let d = config.d_model;
        let f = config.d_ffn;

        let check = |name: &'static str, got: usize, expected: usize| -> MambaResult<()> {
            if got != expected {
                return Err(MambaError::WeightShapeMismatch {
                    name,
                    expected: vec![expected],
                    got: vec![got],
                });
            }
            Ok(())
        };

        check("w_r", weights.w_r.len(), d * d)?;
        check("w_k", weights.w_k.len(), f * d)?;
        check("w_v", weights.w_v.len(), d * f)?;
        check("time_mix_r", weights.time_mix_r.len(), d)?;
        check("time_mix_k", weights.time_mix_k.len(), d)?;

        Ok(Self { config, weights })
    }

    /// Forward pass: `x: [L * D]` → `output: [L * D]`.
    ///
    /// Applies token-shift blending, key/value/receptance projections,
    /// Square-ReLU non-linearity, and receptance gating.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != L * D` for any valid `L`
    pub fn forward(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        let d = self.config.d_model;
        let f = self.config.d_ffn;

        if d == 0 {
            return Err(MambaError::InvalidModelDim(d));
        }
        if x.len() % d != 0 {
            return Err(MambaError::DimensionMismatch {
                expected: x.len() - (x.len() % d),
                got: x.len(),
            });
        }
        let l = x.len() / d;
        if l == 0 {
            return Err(MambaError::InvalidSeqLen(l));
        }

        let w = &self.weights;

        // ── Token-shift blending ───────────────────────────────────────────────
        let mut shifted_r = vec![0.0_f32; l * d];
        let mut shifted_k = vec![0.0_f32; l * d];

        for t in 0..l {
            let curr = &x[t * d..(t + 1) * d];
            let prev = if t > 0 {
                &x[(t - 1) * d..t * d]
            } else {
                &[] as &[f32]
            };
            for c in 0..d {
                let prev_val = if t > 0 { prev[c] } else { 0.0 };
                shifted_r[t * d + c] =
                    w.time_mix_r[c] * curr[c] + (1.0 - w.time_mix_r[c]) * prev_val;
                shifted_k[t * d + c] =
                    w.time_mix_k[c] * curr[c] + (1.0 - w.time_mix_k[c]) * prev_val;
            }
        }

        // ── Receptance: r = sigmoid(W_r · x̃_r) ───────────────────────────────
        // W_r: [d_model, d_model]; out: [L, d_model]
        let mut r = vec![0.0_f32; l * d];
        for t in 0..l {
            for o in 0..d {
                let mut acc = 0.0_f32;
                for i in 0..d {
                    acc += w.w_r[o * d + i] * shifted_r[t * d + i];
                }
                r[t * d + o] = crate::rwkv::time_mixing::sigmoid(acc);
            }
        }

        // ── Key expansion: k = W_k · x̃_k, shape [L, d_ffn] ──────────────────
        // W_k: [d_ffn, d_model]
        let mut k = vec![0.0_f32; l * f];
        for t in 0..l {
            for o in 0..f {
                let mut acc = 0.0_f32;
                for i in 0..d {
                    acc += w.w_k[o * d + i] * shifted_k[t * d + i];
                }
                k[t * f + o] = acc;
            }
        }

        // ── Square-ReLU: k = square_relu(k) ───────────────────────────────────
        k.iter_mut().for_each(|v| *v = square_relu(*v));

        // ── Value contraction: v = W_v · k, shape [L, d_model] ───────────────
        // W_v: [d_model, d_ffn]
        let mut v = vec![0.0_f32; l * d];
        for t in 0..l {
            for o in 0..d {
                let mut acc = 0.0_f32;
                for i in 0..f {
                    acc += w.w_v[o * f + i] * k[t * f + i];
                }
                v[t * d + o] = acc;
            }
        }

        // ── Gate: out = r ⊙ v ─────────────────────────────────────────────────
        let mut out = vec![0.0_f32; l * d];
        for i in 0..l * d {
            out[i] = r[i] * v[i];
        }

        Ok(out)
    }

    /// Return a reference to the layer configuration.
    #[must_use]
    pub fn config(&self) -> &ChannelMixingConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── square_relu ───────────────────────────────────────────────────────────

    #[test]
    fn square_relu_positive() {
        let v = square_relu(2.0);
        assert!((v - 4.0).abs() < 1e-6, "square_relu(2.0)={v}, expected 4.0");
    }

    #[test]
    fn square_relu_negative() {
        let v = square_relu(-1.0);
        assert_eq!(v, 0.0, "square_relu(-1.0) should be 0.0");
    }

    #[test]
    fn square_relu_zero() {
        let v = square_relu(0.0);
        assert_eq!(v, 0.0, "square_relu(0.0) should be 0.0");
    }

    #[test]
    fn square_relu_monotonic_positive() {
        let xs = [0.0_f32, 0.5, 1.0, 2.0, 3.0, 5.0];
        let ys: Vec<f32> = xs.iter().map(|&x| square_relu(x)).collect();
        for i in 1..ys.len() {
            assert!(ys[i] >= ys[i - 1], "square_relu not monotonic at index {i}");
        }
    }

    #[test]
    fn square_relu_non_negative() {
        for x in [-10.0_f32, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0] {
            let v = square_relu(x);
            assert!(v >= 0.0, "square_relu({x})={v} should be >= 0");
        }
    }

    // ── ChannelMixingConfig ───────────────────────────────────────────────────

    #[test]
    fn channel_config_valid() {
        let cfg = ChannelMixingConfig::new(8).expect("valid config");
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.d_ffn, 32, "default d_ffn should be 4 * d_model");
    }

    #[test]
    fn channel_config_zero_d_model() {
        let err = ChannelMixingConfig::new(0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    #[test]
    fn channel_config_default_d_ffn() {
        let cfg = ChannelMixingConfig::new(16).expect("valid config");
        assert_eq!(cfg.d_ffn, 4 * 16, "d_ffn should be 4*d_model=64");
    }

    #[test]
    fn channel_config_with_d_ffn_override() {
        let cfg = ChannelMixingConfig::new(8)
            .expect("valid")
            .with_d_ffn(24)
            .expect("custom d_ffn");
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.d_ffn, 24);
    }

    #[test]
    fn channel_config_with_d_ffn_zero_error() {
        let err = ChannelMixingConfig::new(8)
            .expect("valid")
            .with_d_ffn(0)
            .expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    // ── ChannelMixingWeights ──────────────────────────────────────────────────

    #[test]
    fn channel_weights_zeros_shape() {
        let cfg = ChannelMixingConfig::new(4).expect("valid");
        let wts = ChannelMixingWeights::zeros(&cfg);
        // w_k: [d_ffn, d_model] = [16, 4] = 64 elements
        assert_eq!(wts.w_k.len(), cfg.d_ffn * cfg.d_model, "w_k size mismatch");
        // w_r: [d_model, d_model] = [4, 4] = 16 elements
        assert_eq!(
            wts.w_r.len(),
            cfg.d_model * cfg.d_model,
            "w_r size mismatch"
        );
        // w_v: [d_model, d_ffn] = [4, 16] = 64 elements
        assert_eq!(wts.w_v.len(), cfg.d_model * cfg.d_ffn, "w_v size mismatch");
        // time_mix_*: [d_model] = 4 elements each
        assert_eq!(
            wts.time_mix_r.len(),
            cfg.d_model,
            "time_mix_r size mismatch"
        );
        assert_eq!(
            wts.time_mix_k.len(),
            cfg.d_model,
            "time_mix_k size mismatch"
        );
    }

    #[test]
    fn channel_weights_random_finite() {
        let cfg = ChannelMixingConfig::new(6).expect("valid");
        let mut rng = LcgRng::new(42);
        let wts = ChannelMixingWeights::random(&cfg, &mut rng);
        assert!(wts.w_r.iter().all(|v| v.is_finite()), "w_r not all finite");
        assert!(wts.w_k.iter().all(|v| v.is_finite()), "w_k not all finite");
        assert!(wts.w_v.iter().all(|v| v.is_finite()), "w_v not all finite");
    }

    // ── ChannelMixingLayer ────────────────────────────────────────────────────

    #[test]
    fn channel_layer_forward_shape() {
        let d = 8_usize;
        let l = 4_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let wts = ChannelMixingWeights::default_init(&cfg);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![0.5_f32; l * d];
        let out = layer.forward(&x).expect("forward ok");
        assert_eq!(out.len(), l * d, "output should have L*D elements");
    }

    #[test]
    fn channel_layer_forward_finite() {
        let d = 8_usize;
        let l = 6_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let mut rng = LcgRng::new(55);
        let wts = ChannelMixingWeights::random(&cfg, &mut rng);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let out = layer.forward(&x).expect("forward finite");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}]={v} not finite");
        }
    }

    #[test]
    fn channel_layer_single_token() {
        let d = 4_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let wts = ChannelMixingWeights::default_init(&cfg);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![1.0_f32, -1.0, 0.5, -0.5];
        let out = layer.forward(&x).expect("single token ok");
        assert_eq!(out.len(), d, "single-token output length");
        assert!(out.iter().all(|v| v.is_finite()), "must be finite");
    }

    #[test]
    fn channel_layer_zero_input() {
        // Zero input through zero-init weights → zero output.
        let d = 4_usize;
        let l = 3_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let wts = ChannelMixingWeights::zeros(&cfg);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        let x = vec![0.0_f32; l * d];
        let out = layer.forward(&x).expect("zero input ok");
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "out[{i}]={v} should be 0 for zero weights + zero input"
            );
        }
    }

    #[test]
    fn channel_layer_deterministic() {
        let d = 6_usize;
        let l = 4_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let wts = ChannelMixingWeights::default_init(&cfg);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        let x: Vec<f32> = (0..l * d).map(|i| i as f32 * 0.1).collect();
        let out_a = layer.forward(&x).expect("forward a");
        let out_b = layer.forward(&x).expect("forward b");
        for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert_eq!(a, b, "non-determinism at index {i}");
        }
    }

    #[test]
    fn channel_layer_config_accessor() {
        let d = 8_usize;
        let cfg = ChannelMixingConfig::new(d).expect("valid");
        let wts = ChannelMixingWeights::zeros(&cfg);
        let layer = ChannelMixingLayer::new(cfg, wts).expect("layer ok");
        assert_eq!(layer.config().d_model, d);
        assert_eq!(layer.config().d_ffn, 4 * d);
    }
}
