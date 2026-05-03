//! Conformer convolution module.
//!
//! Implements the convolution sub-layer used inside each Conformer block:
//!
//! ```text
//! LayerNorm
//!   → pointwise expand (D → 2D)
//!   → GLU gate
//!   → depthwise causal conv (kernel K)
//!   → BatchNorm1d (eval mode)
//!   → Swish
//!   → pointwise reduce (D → D)
//! ```
//!
//! The output is then added back to the residual stream by the parent block.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Layer normalisation over the last dimension of a `[T, D]` matrix.
///
/// `x` — `[T * D]` flat, row-major.  Normalisation is per-row (per timestep).
fn layer_norm_1d(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let d = weight.len();
    let t = x.len().checked_div(d).unwrap_or(0);
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        let row = &x[ti * d..(ti + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (di, (&xv, (&w, &b))) in row.iter().zip(weight.iter().zip(bias.iter())).enumerate() {
            out[ti * d + di] = (xv - mean) * inv_std * w + b;
        }
    }
    out
}

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Swish activation: `x * sigmoid(x)`.
#[inline]
fn swish(x: f32) -> f32 {
    x * sigmoid_stable(x)
}

/// GLU gate: splits `x` in half along the channel axis and applies
/// `first_half * sigmoid(second_half)`.
///
/// `x` — `[T, 2D]` flat.  Returns `[T, D]` flat.
fn glu_split(x: &[f32], t: usize, two_d: usize) -> Vec<f32> {
    let d = two_d / 2;
    let mut out = vec![0.0_f32; t * d];
    for ti in 0..t {
        for di in 0..d {
            let a = x[ti * two_d + di];
            let b = x[ti * two_d + d + di];
            out[ti * d + di] = a * sigmoid_stable(b);
        }
    }
    out
}

/// Depthwise causal 1-D convolution.
///
/// Input/output layout: `[D, T]` flat (channels first).
/// Causal: at time step `t` only frames `[t-K+1 .. t]` are visible;
/// frames before position 0 are treated as zero-padding.
///
/// `weight` — `[D, kernel_size]` (one filter per input channel).
fn depthwise_causal_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    d: usize,
    t: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; d * t];
    for di in 0..d {
        let b = bias[di];
        for ti in 0..t {
            let mut acc = b;
            for k in 0..kernel_size {
                // Causal: kernel index 0 = oldest (furthest back).
                // Position (ti - (kernel_size - 1) + k).
                let src_t = ti + k;
                if src_t < kernel_size - 1 {
                    // Would require a sample before t=0 — zero-pad.
                    continue;
                }
                let abs_t = src_t - (kernel_size - 1);
                acc += weight[di * kernel_size + k] * input[di * t + abs_t];
            }
            out[di * t + ti] = acc;
        }
    }
    out
}

/// Batch normalisation in eval mode (no gradient, fixed running stats).
///
/// `x`   — `[T, D]` flat row-major.
/// Returns `[T, D]` flat.
fn batchnorm_1d_eval(
    x: &[f32],
    t: usize,
    d: usize,
    w: &[f32],
    b: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * d];
    for ti in 0..t {
        for di in 0..d {
            let x_norm = (x[ti * d + di] - mean[di]) / (var[di] + eps).sqrt();
            out[ti * d + di] = x_norm * w[di] + b[di];
        }
    }
    out
}

/// Xavier uniform limit: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

// ─── Public types ────────────────────────────────────────────────────────────

/// Weight tensors for a [`ConvModule`].
#[derive(Debug)]
pub struct ConvModuleWeights {
    /// Layer-norm scale `[D]`.
    pub ln_weight: Vec<f32>,
    /// Layer-norm bias `[D]`.
    pub ln_bias: Vec<f32>,
    /// Pointwise expand kernel `[2D, D]` (1×1 conv as matrix).
    pub pw_expand_w: Vec<f32>,
    /// Pointwise expand bias `[2D]`.
    pub pw_expand_b: Vec<f32>,
    /// Depthwise conv weight `[D, kernel_size]`.
    pub dw_weight: Vec<f32>,
    /// Depthwise conv bias `[D]`.
    pub dw_bias: Vec<f32>,
    /// BatchNorm scale `[D]`.
    pub bn_weight: Vec<f32>,
    /// BatchNorm bias `[D]`.
    pub bn_bias: Vec<f32>,
    /// BatchNorm running mean `[D]`.
    pub bn_running_mean: Vec<f32>,
    /// BatchNorm running variance `[D]`.
    pub bn_running_var: Vec<f32>,
    /// Pointwise reduce kernel `[D, D]` (1×1 conv as matrix).
    pub pw_reduce_w: Vec<f32>,
    /// Pointwise reduce bias `[D]`.
    pub pw_reduce_b: Vec<f32>,
}

/// Conformer convolution sub-module.
#[derive(Debug)]
pub struct ConvModule {
    /// Model dimension `D`.
    pub dim: usize,
    /// Depthwise kernel size `K`.
    pub kernel_size: usize,
    /// All learnable and running-stat parameters.
    pub weights: ConvModuleWeights,
}

impl ConvModule {
    /// Construct a `ConvModule` with Xavier-uniform initialised weights.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidEmbedDim` when `dim == 0`,
    /// or `AudioError::InvalidKernelSize` when `kernel_size == 0`.
    pub fn new(dim: usize, kernel_size: usize, rng: &mut LcgRng) -> AudioResult<Self> {
        if dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if kernel_size == 0 {
            return Err(AudioError::InvalidKernelSize(0));
        }

        // Layer norm: scale=1, bias=0.
        let ln_weight = vec![1.0_f32; dim];
        let ln_bias = vec![0.0_f32; dim];

        // Pointwise expand: D → 2D.  Xavier: fan_in=D, fan_out=2D.
        let lim_pw_exp = xavier_limit(dim, 2 * dim);
        let mut pw_expand_w = vec![0.0_f32; 2 * dim * dim];
        for v in pw_expand_w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_pw_exp;
        }
        let mut pw_expand_b = vec![0.0_f32; 2 * dim];
        for v in pw_expand_b.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_pw_exp;
        }

        // Depthwise conv: fan_in = kernel_size (per channel), fan_out = kernel_size.
        let lim_dw = xavier_limit(kernel_size, kernel_size);
        let mut dw_weight = vec![0.0_f32; dim * kernel_size];
        for v in dw_weight.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_dw;
        }
        let mut dw_bias = vec![0.0_f32; dim];
        for v in dw_bias.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_dw;
        }

        // BatchNorm: scale=1, bias=0, running_mean=0, running_var=1.
        let bn_weight = vec![1.0_f32; dim];
        let bn_bias = vec![0.0_f32; dim];
        let bn_running_mean = vec![0.0_f32; dim];
        let bn_running_var = vec![1.0_f32; dim];

        // Pointwise reduce: D → D.  Xavier: fan_in=D, fan_out=D.
        let lim_pw_red = xavier_limit(dim, dim);
        let mut pw_reduce_w = vec![0.0_f32; dim * dim];
        for v in pw_reduce_w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_pw_red;
        }
        let mut pw_reduce_b = vec![0.0_f32; dim];
        for v in pw_reduce_b.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * lim_pw_red;
        }

        Ok(Self {
            dim,
            kernel_size,
            weights: ConvModuleWeights {
                ln_weight,
                ln_bias,
                pw_expand_w,
                pw_expand_b,
                dw_weight,
                dw_bias,
                bn_weight,
                bn_bias,
                bn_running_mean,
                bn_running_var,
                pw_reduce_w,
                pw_reduce_b,
            },
        })
    }

    /// Apply the convolution module to `x` of shape `[T, D]` (flat row-major).
    ///
    /// # Returns
    ///
    /// `[T, D]` flat output (before the external residual add).
    ///
    /// # Errors
    ///
    /// Returns `AudioError::ShapeMismatch` when `x.len() != t * self.dim`,
    /// or `AudioError::EmptyInput` when `t == 0`.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "ConvModule: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!("ConvModule::forward: x.len()={} != t*d={}", x.len(), t * d),
            });
        }

        // 1. Layer norm  [T, D].
        let normed = layer_norm_1d(x, &self.weights.ln_weight, &self.weights.ln_bias, 1e-5);

        // 2. Pointwise expand: [T, D] × [2D, D]ᵀ → [T, 2D].
        let two_d = 2 * d;
        let mut expanded = vec![0.0_f32; t * two_d];
        for ti in 0..t {
            for o in 0..two_d {
                let mut acc = self.weights.pw_expand_b[o];
                for di in 0..d {
                    acc += self.weights.pw_expand_w[o * d + di] * normed[ti * d + di];
                }
                expanded[ti * two_d + o] = acc;
            }
        }

        // 3. GLU gate → [T, D].
        let gated = glu_split(&expanded, t, two_d);

        // 4. Transpose to [D, T] for depthwise conv.
        let mut dw_in = vec![0.0_f32; d * t];
        for ti in 0..t {
            for di in 0..d {
                dw_in[di * t + ti] = gated[ti * d + di];
            }
        }

        // 5. Depthwise causal conv → [D, T].
        let dw_out = depthwise_causal_conv1d(
            &dw_in,
            &self.weights.dw_weight,
            &self.weights.dw_bias,
            d,
            t,
            self.kernel_size,
        );

        // 6. Transpose back to [T, D].
        let mut after_dw = vec![0.0_f32; t * d];
        for ti in 0..t {
            for di in 0..d {
                after_dw[ti * d + di] = dw_out[di * t + ti];
            }
        }

        // 7. BatchNorm1d eval → [T, D].
        let bn_out = batchnorm_1d_eval(
            &after_dw,
            t,
            d,
            &self.weights.bn_weight,
            &self.weights.bn_bias,
            &self.weights.bn_running_mean,
            &self.weights.bn_running_var,
            1e-5,
        );

        // 8. Swish activation.
        let swished: Vec<f32> = bn_out.iter().map(|&v| swish(v)).collect();

        // 9. Pointwise reduce: [T, D] × [D, D]ᵀ → [T, D].
        let mut reduced = vec![0.0_f32; t * d];
        for ti in 0..t {
            for o in 0..d {
                let mut acc = self.weights.pw_reduce_b[o];
                for di in 0..d {
                    acc += self.weights.pw_reduce_w[o * d + di] * swished[ti * d + di];
                }
                reduced[ti * d + o] = acc;
            }
        }

        Ok(reduced)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Swish / sigmoid ──────────────────────────────────────────────────────

    #[test]
    fn swish_at_zero() {
        assert_eq!(swish(0.0), 0.0);
    }

    #[test]
    fn swish_positive_positive() {
        // For large positive x, swish(x) ≈ x.
        let v = swish(20.0);
        assert!((v - 20.0).abs() < 0.01, "swish(20) should ≈ 20, got {v}");
    }

    #[test]
    fn swish_negative_small() {
        // swish(-5) should be small negative.
        let v = swish(-5.0);
        assert!(v < 0.0 && v > -1.0, "swish(-5) out of expected range: {v}");
    }

    #[test]
    fn sigmoid_stable_half() {
        let v = sigmoid_stable(0.0);
        assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {v}");
    }

    // ── GLU split ────────────────────────────────────────────────────────────

    #[test]
    fn glu_split_shape() {
        // Input [T=4, 2D=8], output should be [T=4, D=4].
        let x = vec![1.0_f32; 4 * 8];
        let out = glu_split(&x, 4, 8);
        assert_eq!(out.len(), 4 * 4);
    }

    #[test]
    fn glu_split_zeros_gate() {
        // When gate half is all 0, sigmoid=0.5, output = input_half * 0.5.
        let mut x = vec![0.0_f32; 2 * 4];
        // First half = 2.0, second half = 0.0.
        for v in x[..4].iter_mut() {
            *v = 2.0;
        }
        let out = glu_split(&x, 1, 8);
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    // ── Layer norm ───────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_mean() {
        let d = 8usize;
        let t = 4usize;
        let x: Vec<f32> = (0..t * d).map(|i| i as f32).collect();
        let w = vec![1.0_f32; d];
        let b = vec![0.0_f32; d];
        let out = layer_norm_1d(&x, &w, &b, 1e-5);
        // Each row mean should be ~0.
        for ti in 0..t {
            let row_mean: f32 = out[ti * d..(ti + 1) * d].iter().sum::<f32>() / d as f32;
            assert!(row_mean.abs() < 1e-4, "row {ti} mean={row_mean}");
        }
    }

    // ── Depthwise causal conv ────────────────────────────────────────────────

    #[test]
    fn depthwise_causal_no_future_leak() {
        // If we zero out frames beyond position P and re-run, all output at
        // positions <= P must be identical (causal invariant).
        let d = 4usize;
        let t = 16usize;
        let ks = 5usize;
        let weight: Vec<f32> = (0..d * ks).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let bias = vec![0.0_f32; d];

        let mut input_full: Vec<f32> = (0..d * t).map(|i| i as f32 * 0.01).collect();
        let out_full = depthwise_causal_conv1d(&input_full, &weight, &bias, d, t, ks);

        // Zero out the last 4 frames.
        let cutoff = 12usize;
        for di in 0..d {
            for ti in cutoff..t {
                input_full[di * t + ti] = 0.0;
            }
        }
        let out_partial = depthwise_causal_conv1d(&input_full, &weight, &bias, d, t, ks);

        // Output at positions strictly before cutoff-(ks-1) must be identical.
        let safe_end = cutoff.saturating_sub(ks - 1);
        for di in 0..d {
            for ti in 0..safe_end {
                let a = out_full[di * t + ti];
                let b = out_partial[di * t + ti];
                assert!(
                    (a - b).abs() < 1e-5,
                    "future leak at d={di} t={ti}: full={a} partial={b}"
                );
            }
        }
    }

    // ── BatchNorm eval ───────────────────────────────────────────────────────

    #[test]
    fn batchnorm_1d_eval_normalises() {
        let d = 4usize;
        let t = 8usize;
        // Input: channel 0 has mean=5, var=4 in running stats.
        let x: Vec<f32> = (0..t * d).map(|i| (i % d) as f32 * 2.0 + 1.0).collect();
        let w = vec![1.0_f32; d];
        let b = vec![0.0_f32; d];
        let mean = vec![1.0_f32; d]; // running mean
        let var = vec![4.0_f32; d]; // running var
        let out = batchnorm_1d_eval(&x, t, d, &w, &b, &mean, &var, 1e-5);
        // All finite.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── ConvModule ───────────────────────────────────────────────────────────

    #[test]
    fn conv_module_build_ok() {
        let mut rng = LcgRng::new(42);
        let m = ConvModule::new(32, 15, &mut rng);
        assert!(m.is_ok(), "ConvModule::new failed: {m:?}");
    }

    #[test]
    fn conv_module_output_shape() {
        let mut rng = LcgRng::new(7);
        let m = ConvModule::new(32, 7, &mut rng).expect("new");
        let t = 20usize;
        let x = vec![0.1_f32; t * 32];
        let out = m.forward(&x, t).expect("forward");
        assert_eq!(out.len(), t * 32, "output shape mismatch");
    }

    #[test]
    fn conv_module_output_finite() {
        let mut rng = LcgRng::new(99);
        let m = ConvModule::new(16, 5, &mut rng).expect("new");
        let t = 12usize;
        let mut x = vec![0.0_f32; t * 16];
        rng.fill_normal(&mut x);
        let out = m.forward(&x, t).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn conv_module_zero_dim_err() {
        let mut rng = LcgRng::new(1);
        let r = ConvModule::new(0, 5, &mut rng);
        assert!(r.is_err());
    }

    #[test]
    fn conv_module_zero_kernel_err() {
        let mut rng = LcgRng::new(1);
        let r = ConvModule::new(8, 0, &mut rng);
        assert!(r.is_err());
    }
}
