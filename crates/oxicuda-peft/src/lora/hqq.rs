//! HQQ — Half-Quadratic Quantization.
//!
//! Reference: Badri H, Shaji A (2023) "Towards 1-bit Machine Learning Models",
//! mobiusml/hqq blog. <https://mobiusml.github.io/hqq_blog/>
//!
//! HQQ frames weight quantization as a half-quadratic splitting problem:
//!
//! ```text
//! min_{q, s, z, β}  ‖w − dequant(q, s, z)‖²  +  β · ‖dequant(q, s, z) − w‖_p^p
//! ```
//!
//! with `p ∈ (0, 1]`. The objective is split alternately:
//!   1. z-update: proximal of the Lₚ shrinkage applied to the residual.
//!   2. scale/zero-update: closed-form 2×2 least squares per group.
//!   3. q-update: round-and-clip given updated scale/zero.
//!   4. β-update: geometric annealing.
//!
//! Quantization is performed group-wise: every `group_size` consecutive weights
//! share one affine `(scale, zero)` pair. The last partial group is honoured
//! when `len % group_size != 0`. This module mirrors the per-group affine pattern
//! in [`crate::lora::qlora::quantize_block`] but solves the joint scale/zero/code
//! problem with `max_iters` alternating minimisations rather than a single absmax
//! pass.

use crate::error::{PeftError, PeftResult};

/// Maximum number of bits this implementation supports per quantized value.
const MAX_NBITS: u32 = 16;

/// Configuration for the HQQ quantizer.
#[derive(Debug, Clone)]
pub struct HqqConfig {
    /// Bits per quantized value. Must satisfy `1 ≤ nbits ≤ 16`. Typical: 2, 3, 4, 8.
    pub nbits: u32,
    /// Number of weights sharing one `(scale, zero)` pair. Must be `> 0`.
    pub group_size: usize,
    /// Shrinkage exponent `p ∈ (0, 1]` in the Lₚ regularisation term.
    pub p: f32,
    /// Number of outer half-quadratic iterations.
    pub max_iters: usize,
    /// Initial value of the half-quadratic coupling parameter β.
    pub beta_init: f32,
    /// Multiplicative growth of β at each outer iteration (`β ← β · beta_growth`).
    pub beta_growth: f32,
}

impl Default for HqqConfig {
    fn default() -> Self {
        Self {
            nbits: 4,
            group_size: 64,
            p: 0.7,
            max_iters: 20,
            beta_init: 1e-1,
            beta_growth: 1.05,
        }
    }
}

/// Result of HQQ quantization.
#[derive(Debug, Clone)]
pub struct HqqQuantized {
    /// Integer quantization codes in `[0, 2^nbits − 1]`. Length equals `len`.
    pub q: Vec<i32>,
    /// Per-group scale factors.
    pub scale: Vec<f32>,
    /// Per-group zero points (additive offsets, in dequantized space).
    pub zero: Vec<f32>,
    /// Group size used during quantization.
    pub group_size: usize,
    /// Bits per quantized value.
    pub nbits: u32,
    /// Original (logical) length of the quantized tensor.
    pub len: usize,
}

/// HQQ algorithm namespace.
pub struct Hqq;

impl Hqq {
    /// Quantize `w` with HQQ using the given configuration.
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] if `w` is empty,
    /// [`PeftError::Internal`] for invalid `nbits` (zero or `> 16`),
    /// or [`PeftError::ZeroBlockSize`] if `cfg.group_size == 0`.
    pub fn quantize(w: &[f32], cfg: &HqqConfig) -> PeftResult<HqqQuantized> {
        validate(w, cfg)?;

        let len = w.len();
        let group_size = cfg.group_size.min(len);
        let n_groups = len.div_ceil(group_size);
        let q_max = (1_u32 << cfg.nbits) - 1;
        let q_max_f = q_max as f32;

        let mut scale = vec![0.0_f32; n_groups];
        let mut zero = vec![0.0_f32; n_groups];

        // Step 1: initialise (scale, zero) via min/max per group.
        for (g, scale_slot) in scale.iter_mut().enumerate() {
            let start = g * group_size;
            let end = (start + group_size).min(len);
            let (lo, hi) = min_max(&w[start..end]);
            let span = (hi - lo).max(f32::EPSILON);
            let s = span / q_max_f;
            *scale_slot = s;
            // dequant = scale · q + zero ⇒ to send `lo` to integer 0 we need zero = lo.
            zero[g] = lo;
        }

        // Initial codes from min/max scale/zero.
        let mut q = vec![0_i32; len];
        for (g, &s) in scale.iter().enumerate() {
            let start = g * group_size;
            let end = (start + group_size).min(len);
            let z = zero[g];
            let inv_s = if s.abs() > f32::EPSILON { 1.0 / s } else { 0.0 };
            for (i, code_slot) in q.iter_mut().enumerate().take(end).skip(start) {
                let code = ((w[i] - z) * inv_s).round();
                *code_slot = clip_code(code as i32, q_max);
            }
        }

        // Step 2: outer half-quadratic loop.
        let mut beta = cfg.beta_init.max(f32::EPSILON);
        let p = cfg.p.clamp(1e-3, 1.0);

        for _ in 0..cfg.max_iters {
            // z-update: produce noisy target `z_target[i] = w[i] + (r[i] − shrink(r[i], β))`
            // where `r[i] = dequant(q[i], scale, zero) − w[i]`.
            let mut z_target = vec![0.0_f32; len];
            for (g, &s) in scale.iter().enumerate() {
                let start = g * group_size;
                let end = (start + group_size).min(len);
                let z = zero[g];
                for i in start..end {
                    let dq = s * (q[i] as f32) + z;
                    let r = dq - w[i];
                    let shrunk = shrinkage(r, beta, p);
                    z_target[i] = w[i] + (r - shrunk);
                }
            }

            // (scale, zero) update — closed-form 2×2 least squares per group.
            //
            // Per-group system:
            //   [Σ q² , Σ q ] [scale]   [Σ q·z_target]
            //   [Σ q  ,  G  ] [zero ] = [Σ z_target  ]
            //
            // det = G · Σ q² − (Σ q)².
            for (g, scale_slot) in scale.iter_mut().enumerate() {
                let start = g * group_size;
                let end = (start + group_size).min(len);
                let group_n = (end - start) as f32;
                let mut sum_q = 0.0_f64;
                let mut sum_q2 = 0.0_f64;
                let mut sum_zt = 0.0_f64;
                let mut sum_qzt = 0.0_f64;
                for i in start..end {
                    let qi = q[i] as f64;
                    let zt = z_target[i] as f64;
                    sum_q += qi;
                    sum_q2 += qi * qi;
                    sum_zt += zt;
                    sum_qzt += qi * zt;
                }
                let det = (group_n as f64) * sum_q2 - sum_q * sum_q;
                if det.abs() > 1e-12 {
                    let new_scale = ((group_n as f64) * sum_qzt - sum_q * sum_zt) / det;
                    let new_zero = (sum_q2 * sum_zt - sum_q * sum_qzt) / det;
                    if new_scale.is_finite() && new_zero.is_finite() {
                        *scale_slot = new_scale as f32;
                        zero[g] = new_zero as f32;
                    }
                }
            }

            // q-update: round-and-clip with refreshed scale/zero against z_target.
            for (g, &s) in scale.iter().enumerate() {
                let start = g * group_size;
                let end = (start + group_size).min(len);
                let z = zero[g];
                let inv_s = if s.abs() > f32::EPSILON { 1.0 / s } else { 0.0 };
                for (i, code_slot) in q.iter_mut().enumerate().take(end).skip(start) {
                    let code = ((z_target[i] - z) * inv_s).round();
                    *code_slot = clip_code(code as i32, q_max);
                }
            }

            beta *= cfg.beta_growth;
        }

        Ok(HqqQuantized {
            q,
            scale,
            zero,
            group_size,
            nbits: cfg.nbits,
            len,
        })
    }

    /// Dequantize an [`HqqQuantized`] back to a contiguous `Vec<f32>` of length `q.len`.
    #[must_use]
    pub fn dequantize(q: &HqqQuantized) -> Vec<f32> {
        let mut out = vec![0.0_f32; q.len];
        if q.group_size == 0 {
            return out;
        }
        for (i, out_slot) in out.iter_mut().enumerate().take(q.len) {
            let g = (i / q.group_size).min(q.scale.len().saturating_sub(1));
            *out_slot = q.scale[g] * (q.q[i] as f32) + q.zero[g];
        }
        out
    }
}

/// Validate the user-supplied configuration and input slice.
fn validate(w: &[f32], cfg: &HqqConfig) -> PeftResult<()> {
    if w.is_empty() {
        return Err(PeftError::EmptyInput);
    }
    if cfg.group_size == 0 {
        return Err(PeftError::ZeroBlockSize);
    }
    if cfg.nbits == 0 || cfg.nbits > MAX_NBITS {
        return Err(PeftError::Internal {
            msg: format!("HQQ nbits={} out of range (1..={MAX_NBITS})", cfg.nbits),
        });
    }
    Ok(())
}

/// Saturating clip of an integer code into `[0, q_max]`.
#[inline]
fn clip_code(code: i32, q_max: u32) -> i32 {
    let q_max_i = q_max as i32;
    code.clamp(0, q_max_i)
}

/// Lₚ proximal shrinkage applied to a scalar residual `r`.
///
/// For `p = 1` this is the classical soft-threshold `sign(r) · max(|r| − 1/β, 0)`.
/// For `p ∈ (0, 1)` we approximate the proximal operator with an iteratively
/// re-weighted soft threshold of magnitude `β · |r|^{p − 1}` (Hong & Wang
/// 2017; see also Marjanovic & Solo 2014).
#[inline]
fn shrinkage(r: f32, beta: f32, p: f32) -> f32 {
    let abs_r = r.abs();
    if abs_r < f32::EPSILON {
        return 0.0;
    }
    let beta_safe = beta.max(f32::EPSILON);
    let threshold = if (p - 1.0).abs() < 1e-6 {
        1.0 / beta_safe
    } else {
        // β · |r|^{p − 1}; for p ∈ (0, 1) this is an iteratively re-weighted soft threshold.
        let weight = abs_r.max(f32::EPSILON).powf(p - 1.0);
        weight / beta_safe
    };
    let magnitude = (abs_r - threshold).max(0.0);
    r.signum() * magnitude
}

/// Compute `(min, max)` across a non-empty slice.
fn min_max(xs: &[f32]) -> (f32, f32) {
    let mut lo = xs[0];
    let mut hi = xs[0];
    for &v in xs.iter().skip(1) {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a deterministic test vector with `len` elements in `[-amp, amp]`.
    fn synthetic_weights(len: usize, amp: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; len];
        for (i, slot) in out.iter_mut().enumerate() {
            // Mix linear ramp with a low-frequency sinusoid for diverse values.
            let t = (i as f32) / (len as f32);
            *slot = amp * (2.0 * t - 1.0) + 0.3 * amp * ((10.0 * t).sin());
        }
        out
    }

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        (s / n).sqrt()
    }

    fn linf(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn round_trip_4bit_within_bound() {
        let w = synthetic_weights(256, 1.0);
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 32,
            p: 0.7,
            max_iters: 30,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        let dq = Hqq::dequantize(&q);
        let err = rmse(&w, &dq);
        // 4-bit affine over 32-element groups should comfortably stay below 0.1 RMSE for amp=1.
        assert!(err < 0.1, "4-bit RMSE {err} not below 0.1");
    }

    #[test]
    fn round_trip_8bit_nearly_exact() {
        let w = synthetic_weights(256, 1.0);
        let cfg = HqqConfig {
            nbits: 8,
            group_size: 64,
            p: 0.7,
            max_iters: 30,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        let dq = Hqq::dequantize(&q);
        let err = linf(&w, &dq);
        // 8-bit affine should keep L∞ error below 1e-2 for amp=1.
        assert!(err < 1e-2, "8-bit L∞ {err} not below 1e-2");
    }

    #[test]
    fn round_trip_2bit_bounded() {
        let w = synthetic_weights(128, 1.0);
        let cfg = HqqConfig {
            nbits: 2,
            group_size: 32,
            p: 0.7,
            max_iters: 30,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        let dq = Hqq::dequantize(&q);
        let err = rmse(&w, &dq);
        // 2-bit affine quantization is coarse; require finite, bounded error.
        assert!(err.is_finite(), "2-bit RMSE non-finite: {err}");
        assert!(err < 1.0, "2-bit RMSE {err} unexpectedly large");
    }

    #[test]
    fn group_size_larger_than_len_single_group() {
        let w = synthetic_weights(20, 0.5);
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 64,
            p: 1.0,
            max_iters: 10,
            beta_init: 1.0,
            beta_growth: 1.1,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        // Group size clamped to len → exactly one group.
        assert_eq!(
            q.scale.len(),
            1,
            "expected one group, got {}",
            q.scale.len()
        );
        assert_eq!(q.zero.len(), 1);
        assert_eq!(q.q.len(), 20);
        assert_eq!(q.group_size, 20);
    }

    #[test]
    fn last_partial_group_handled() {
        // len = 70, group_size = 32 → groups: 32, 32, 6.
        let w = synthetic_weights(70, 1.0);
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 32,
            p: 0.7,
            max_iters: 20,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        assert_eq!(q.scale.len(), 3, "expected 3 groups (32,32,6)");
        assert_eq!(q.q.len(), 70);
        let dq = Hqq::dequantize(&q);
        assert_eq!(dq.len(), 70);
        let err = rmse(&w, &dq);
        assert!(err < 0.1, "partial-group RMSE {err} not below 0.1");
    }

    #[test]
    fn p_half_vs_p_one_both_finite() {
        let w = synthetic_weights(128, 1.0);
        let cfg_half = HqqConfig {
            nbits: 4,
            group_size: 32,
            p: 0.5,
            max_iters: 25,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let cfg_one = HqqConfig {
            p: 1.0,
            ..cfg_half.clone()
        };
        let q_half = Hqq::quantize(&w, &cfg_half).expect("quantize p=0.5");
        let q_one = Hqq::quantize(&w, &cfg_one).expect("quantize p=1.0");
        let dq_half = Hqq::dequantize(&q_half);
        let dq_one = Hqq::dequantize(&q_one);
        for v in dq_half.iter().chain(dq_one.iter()) {
            assert!(v.is_finite(), "non-finite dequant value {v}");
        }
    }

    #[test]
    fn deterministic_same_input() {
        let w = synthetic_weights(128, 1.0);
        let cfg = HqqConfig::default();
        let a = Hqq::quantize(&w, &cfg).expect("a");
        let b = Hqq::quantize(&w, &cfg).expect("b");
        assert_eq!(a.q, b.q);
        assert_eq!(a.scale, b.scale);
        assert_eq!(a.zero, b.zero);
    }

    #[test]
    fn empty_input_errors() {
        let cfg = HqqConfig::default();
        let res = Hqq::quantize(&[], &cfg);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn invalid_nbits_errors() {
        let w = synthetic_weights(16, 1.0);
        let cfg_zero = HqqConfig {
            nbits: 0,
            ..HqqConfig::default()
        };
        let res_zero = Hqq::quantize(&w, &cfg_zero);
        assert!(matches!(res_zero, Err(PeftError::Internal { .. })));

        let cfg_too_big = HqqConfig {
            nbits: 17,
            ..HqqConfig::default()
        };
        let res_big = Hqq::quantize(&w, &cfg_too_big);
        assert!(matches!(res_big, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn zero_group_size_errors() {
        let w = synthetic_weights(16, 1.0);
        let cfg = HqqConfig {
            group_size: 0,
            ..HqqConfig::default()
        };
        let res = Hqq::quantize(&w, &cfg);
        assert!(matches!(res, Err(PeftError::ZeroBlockSize)));
    }

    #[test]
    fn integer_weights_recoverable() {
        // Weights drawn from {0, 1, ..., 15} → 4-bit affine should be exact within one group.
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 16,
            p: 1.0,
            max_iters: 20,
            beta_init: 1.0,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        let dq = Hqq::dequantize(&q);
        let err = linf(&w, &dq);
        assert!(err < 1e-3, "integer-weight L∞ {err} too large");
    }

    #[test]
    fn outer_iters_reduce_residual_monotone_on_average() {
        // Run quantize with different `max_iters` budgets and confirm the
        // half-quadratic loop does not *increase* the squared residual versus
        // the initial absmax solution.
        let w = synthetic_weights(128, 1.0);
        let cfg_init = HqqConfig {
            nbits: 4,
            group_size: 32,
            p: 0.7,
            max_iters: 0, // initial fit only
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let cfg_full = HqqConfig {
            max_iters: 25,
            ..cfg_init.clone()
        };
        let q_init = Hqq::quantize(&w, &cfg_init).expect("init");
        let q_full = Hqq::quantize(&w, &cfg_full).expect("full");
        let err_init = rmse(&w, &Hqq::dequantize(&q_init));
        let err_full = rmse(&w, &Hqq::dequantize(&q_full));
        assert!(
            err_full <= err_init + 1e-4,
            "outer iters increased RMSE: init={err_init} full={err_full}"
        );
    }

    #[test]
    fn quantize_dequant_quantize_idempotent_within_tolerance() {
        let w = synthetic_weights(128, 1.0);
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 32,
            p: 0.7,
            max_iters: 25,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q1 = Hqq::quantize(&w, &cfg).expect("q1");
        let dq1 = Hqq::dequantize(&q1);
        let q2 = Hqq::quantize(&dq1, &cfg).expect("q2");
        let dq2 = Hqq::dequantize(&q2);
        // Re-quantizing an already-quantized signal should converge — small drift only.
        let drift = rmse(&dq1, &dq2);
        assert!(
            drift < 5e-2,
            "quantize-dequant-quantize drift {drift} above tolerance"
        );
    }

    #[test]
    fn dequantized_length_matches_input_length() {
        let w = synthetic_weights(73, 0.7); // not a multiple of any common group size
        let cfg = HqqConfig {
            nbits: 4,
            group_size: 16,
            p: 0.7,
            max_iters: 10,
            beta_init: 0.5,
            beta_growth: 1.05,
        };
        let q = Hqq::quantize(&w, &cfg).expect("quantize");
        let dq = Hqq::dequantize(&q);
        assert_eq!(dq.len(), w.len());
    }

    #[test]
    fn codes_inside_range() {
        let w = synthetic_weights(96, 1.0);
        for &nbits in &[2_u32, 3, 4, 8] {
            let cfg = HqqConfig {
                nbits,
                group_size: 16,
                p: 0.7,
                max_iters: 20,
                beta_init: 0.5,
                beta_growth: 1.05,
            };
            let q = Hqq::quantize(&w, &cfg).expect("quantize");
            let q_max = (1_i32 << nbits) - 1;
            for &code in &q.q {
                assert!(
                    (0..=q_max).contains(&code),
                    "code {code} out of [0, {q_max}] for nbits={nbits}"
                );
            }
        }
    }
}
