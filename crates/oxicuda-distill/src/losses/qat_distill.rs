//! QAT-Distill — Quantisation-Aware Distillation (INT8 / FP8 student).
//!
//! Trains a low-precision student under the supervision of a full-precision teacher.
//! The student weights / activations are passed through a *fake-quantisation* operator
//! during the forward pass (so the network "sees" the rounding error), while gradients
//! flow through unchanged via the **straight-through estimator** (STE). The distillation
//! objective then matches the quantised student logits against the soft teacher targets,
//! so the student learns weights that are robust to quantisation noise.
//!
//! Two quantiser families are supported:
//!
//! * **INT8 affine** — symmetric or asymmetric per-tensor uniform quantisation with a
//!   given number of bits (`n_bits`, typically 8). The real interval `[min, max]` is
//!   mapped onto the integer grid `[q_min, q_max]` via a `scale` and (optionally) a
//!   `zero_point`.
//! * **FP8** — `e4m3` / `e5m2` floating-point formats. The value is decomposed into sign,
//!   exponent and mantissa, the mantissa is rounded to the format width, and the result is
//!   clamped to the representable dynamic range.
//!
//! References: Jacob et al. 2018 ("Quantization and Training of Neural Networks for
//! Efficient Integer-Arithmetic-Only Inference"); Micikevicius et al. 2022 ("FP8 Formats
//! for Deep Learning"); Hinton et al. 2015 (distillation objective).

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{cross_entropy, kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-12;

/// FP8 representation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1 sign, 4 exponent, 3 mantissa bits (bias 7) — higher precision, lower range.
    E4m3,
    /// 1 sign, 5 exponent, 2 mantissa bits (bias 15) — lower precision, higher range.
    E5m2,
}

impl Fp8Format {
    /// Number of explicit mantissa bits.
    #[must_use]
    pub fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4m3 => 3,
            Fp8Format::E5m2 => 2,
        }
    }

    /// Exponent bias.
    #[must_use]
    pub fn exponent_bias(self) -> i32 {
        match self {
            Fp8Format::E4m3 => 7,
            Fp8Format::E5m2 => 15,
        }
    }

    /// Largest finite magnitude representable in the format.
    #[must_use]
    pub fn max_magnitude(self) -> f32 {
        // For e4m3 the spec reserves the all-ones exponent+mantissa pattern, so the
        // maximum finite value is 448.0. For e5m2 it is 57344.0.
        match self {
            Fp8Format::E4m3 => 448.0,
            Fp8Format::E5m2 => 57344.0,
        }
    }

    /// Smallest positive *normal* magnitude (`2^(1 - bias)`).
    #[must_use]
    pub fn min_normal(self) -> f32 {
        2.0_f32.powi(1 - self.exponent_bias())
    }
}

/// Quantiser kind used by the student.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// Affine integer quantisation with `n_bits`; `symmetric` toggles zero-point usage.
    Int { n_bits: u32, symmetric: bool },
    /// 8-bit floating-point quantisation.
    Fp8 { format: Fp8Format },
}

/// Configuration for quantisation-aware distillation.
#[derive(Debug, Clone)]
pub struct QatDistillConfig {
    /// Distillation temperature `T > 0`.
    pub temperature: f32,
    /// Soft-target weight `alpha ∈ [0, 1]`; hard CE weight is `1 − alpha`.
    pub alpha: f32,
    /// Quantiser applied to the student logits before computing the KD loss.
    pub quant: QuantKind,
}

impl QatDistillConfig {
    /// Validate and construct a configuration.
    pub fn new(temperature: f32, alpha: f32, quant: QuantKind) -> DistillResult<Self> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be finite and > 0, got {temperature}"),
            });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(DistillError::InvalidConfig {
                msg: format!("alpha must be in [0, 1], got {alpha}"),
            });
        }
        if let QuantKind::Int { n_bits, .. } = quant
            && !(2..=16).contains(&n_bits)
        {
            return Err(DistillError::InvalidConfig {
                msg: format!("n_bits must be in [2, 16], got {n_bits}"),
            });
        }
        Ok(Self {
            temperature,
            alpha,
            quant,
        })
    }
}

/// Affine integer quantisation parameters derived from a real value range.
#[derive(Debug, Clone, Copy)]
pub struct AffineQuantParams {
    /// Real-valued step size between adjacent quantisation levels.
    pub scale: f32,
    /// Integer code mapping to real value `0.0`.
    pub zero_point: i32,
    /// Minimum integer code.
    pub q_min: i32,
    /// Maximum integer code.
    pub q_max: i32,
}

/// Compute affine quantisation parameters for `[min_val, max_val]` at `n_bits`.
///
/// When `symmetric` the range is made symmetric about zero and `zero_point = 0`.
#[must_use]
pub fn compute_affine_params(
    min_val: f32,
    max_val: f32,
    n_bits: u32,
    symmetric: bool,
) -> AffineQuantParams {
    let levels = (1i64 << n_bits) - 1; // e.g. 255 for 8 bits
    if symmetric {
        let q_max = (levels / 2) as i32; // 127 for 8 bits
        let q_min = -q_max;
        let amax = min_val.abs().max(max_val.abs()).max(EPS);
        let scale = amax / q_max as f32;
        AffineQuantParams {
            scale,
            zero_point: 0,
            q_min,
            q_max,
        }
    } else {
        let q_min = 0i32;
        let q_max = levels as i32; // 255 for 8 bits
        let lo = min_val.min(0.0);
        let hi = max_val.max(0.0);
        let span = (hi - lo).max(EPS);
        let scale = span / (q_max - q_min) as f32;
        // zero_point chosen so that real 0 maps exactly to an integer code.
        let zp = (q_min as f32 - lo / scale).round() as i32;
        let zero_point = zp.clamp(q_min, q_max);
        AffineQuantParams {
            scale,
            zero_point,
            q_min,
            q_max,
        }
    }
}

/// Fake-quantise one value through the affine integer grid (quantise then dequantise).
#[must_use]
pub fn fake_quant_affine_value(x: f32, p: AffineQuantParams) -> f32 {
    let scale = p.scale.max(EPS);
    let q = (x / scale).round() as i32 + p.zero_point;
    let q = q.clamp(p.q_min, p.q_max);
    (q - p.zero_point) as f32 * scale
}

/// Fake-quantise a slice with affine integer quantisation, deriving the range from the data.
#[must_use]
pub fn fake_quant_affine(x: &[f32], n_bits: u32, symmetric: bool) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let min_val = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p = compute_affine_params(min_val, max_val, n_bits, symmetric);
    x.iter().map(|&v| fake_quant_affine_value(v, p)).collect()
}

/// Fake-quantise one value into an FP8 format and back to `f32`.
#[must_use]
pub fn fake_quant_fp8_value(x: f32, format: Fp8Format) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_finite() { 0.0 } else { x };
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let max_mag = format.max_magnitude();
    if ax >= max_mag {
        return sign * max_mag;
    }
    let m_bits = format.mantissa_bits();
    let min_normal = format.min_normal();
    if ax < min_normal {
        // Subnormal region: quantise on the fixed grid `min_normal / 2^m_bits`.
        let step = min_normal / (1i32 << m_bits) as f32;
        let q = (ax / step).round();
        return sign * q * step;
    }
    // Normal region: round the mantissa to `m_bits` at the value's binade.
    let exp = ax.log2().floor() as i32;
    let pow = 2.0_f32.powi(exp);
    let mantissa = ax / pow; // in [1, 2)
    let grid = (1i32 << m_bits) as f32;
    let q_mantissa = (mantissa * grid).round() / grid;
    sign * q_mantissa * pow
}

/// Fake-quantise a slice into an FP8 format and back.
#[must_use]
pub fn fake_quant_fp8(x: &[f32], format: Fp8Format) -> Vec<f32> {
    x.iter().map(|&v| fake_quant_fp8_value(v, format)).collect()
}

/// Apply the configured student quantiser to a slice (forward fake-quant).
#[must_use]
pub fn quantize_student(x: &[f32], quant: QuantKind) -> Vec<f32> {
    match quant {
        QuantKind::Int { n_bits, symmetric } => fake_quant_affine(x, n_bits, symmetric),
        QuantKind::Fp8 { format } => fake_quant_fp8(x, format),
    }
}

/// Quantisation-aware KD loss for one example.
///
/// The student logits are fake-quantised, then the standard Hinton objective is applied:
/// `loss = alpha · T² · KL(p_t || p_s_quant) + (1 − alpha) · CE(s_quant, label)`.
///
/// The straight-through estimator means the returned loss is computed on the quantised
/// logits while training code would route gradients through the *unquantised* path; here
/// we expose [`ste_grad_mask`] for callers that propagate gradients explicitly.
pub fn qat_kd_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    label: usize,
    cfg: &QatDistillConfig,
) -> DistillResult<f32> {
    if student_logits.is_empty() || teacher_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: student_logits.len(),
            got: teacher_logits.len(),
        });
    }
    if label >= student_logits.len() {
        return Err(DistillError::InvalidConfig {
            msg: format!(
                "label {label} out of range for {} classes",
                student_logits.len()
            ),
        });
    }
    let s_q = quantize_student(student_logits, cfg.quant);
    let t = cfg.temperature;
    let p_t = softmax_with_temp(teacher_logits, t);
    let p_s = softmax_with_temp(&s_q, t);
    let soft = kl_divergence(&p_t, &p_s) * t * t;
    let hard = cross_entropy(&s_q, label);
    Ok(cfg.alpha * soft + (1.0 - cfg.alpha) * hard)
}

/// Batched QAT-KD loss (mean over examples).
pub fn qat_kd_loss_batch(
    student_logits: &[Vec<f32>],
    teacher_logits: &[Vec<f32>],
    labels: &[usize],
    cfg: &QatDistillConfig,
) -> DistillResult<f32> {
    if student_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if student_logits.len() != teacher_logits.len() || student_logits.len() != labels.len() {
        return Err(DistillError::DimensionMismatch {
            expected: student_logits.len(),
            got: teacher_logits.len().min(labels.len()),
        });
    }
    let mut total = 0.0_f32;
    for ((s, t), &y) in student_logits
        .iter()
        .zip(teacher_logits.iter())
        .zip(labels.iter())
    {
        total += qat_kd_loss(s, t, y, cfg)?;
    }
    Ok(total / student_logits.len() as f32)
}

/// Straight-through estimator gradient mask.
///
/// Returns `1.0` where the un-quantised input lies inside the clipping range (gradient
/// passes through) and `0.0` where it is clamped (gradient is killed). This is the standard
/// clipped-STE used by QAT to avoid pushing weights further past the saturation point.
#[must_use]
pub fn ste_grad_mask(x: &[f32], quant: QuantKind) -> Vec<f32> {
    match quant {
        QuantKind::Int { n_bits, symmetric } => {
            if x.is_empty() {
                return Vec::new();
            }
            let min_val = x.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p = compute_affine_params(min_val, max_val, n_bits, symmetric);
            let lo = (p.q_min - p.zero_point) as f32 * p.scale;
            let hi = (p.q_max - p.zero_point) as f32 * p.scale;
            x.iter()
                .map(|&v| if v >= lo && v <= hi { 1.0 } else { 0.0 })
                .collect()
        }
        QuantKind::Fp8 { format } => {
            let max_mag = format.max_magnitude();
            x.iter()
                .map(|&v| if v.abs() < max_mag { 1.0 } else { 0.0 })
                .collect()
        }
    }
}

/// Mean absolute quantisation error introduced by the configured quantiser.
///
/// Useful as a diagnostic for how aggressively a tensor is being distorted.
#[must_use]
pub fn quantization_error(x: &[f32], quant: QuantKind) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let q = quantize_student(x, quant);
    x.iter()
        .zip(q.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / x.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip_within_step() {
        // A symmetric 8-bit quantiser should distort values by at most half a step.
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let q = fake_quant_affine(&x, 8, true);
        let amax = x.iter().cloned().fold(0.0_f32, |m, v| m.max(v.abs()));
        let step = amax / 127.0;
        for (a, b) in x.iter().zip(q.iter()) {
            assert!(
                (a - b).abs() <= step * 0.5 + 1e-6,
                "value {a} -> {b} exceeded half-step {step}"
            );
        }
    }

    #[test]
    fn affine_symmetric_zero_maps_to_zero() {
        let p = compute_affine_params(-2.0, 2.0, 8, true);
        assert_eq!(p.zero_point, 0);
        assert!(fake_quant_affine_value(0.0, p).abs() < 1e-7);
    }

    #[test]
    fn affine_asymmetric_zero_preserved() {
        // Even on an asymmetric range, exact zero must dequantise back to ~zero.
        let p = compute_affine_params(-0.5, 3.0, 8, false);
        assert!(fake_quant_affine_value(0.0, p).abs() < p.scale);
    }

    #[test]
    fn fp8_clamps_to_max() {
        let big = 1.0e6_f32;
        assert!((fake_quant_fp8_value(big, Fp8Format::E4m3) - 448.0).abs() < 1e-3);
        assert!((fake_quant_fp8_value(-big, Fp8Format::E5m2) + 57344.0).abs() < 1.0);
    }

    #[test]
    fn fp8_exact_power_of_two_is_exact() {
        // Powers of two are exactly representable in both FP8 formats.
        for &v in &[1.0_f32, 2.0, 4.0, 0.5, 0.25] {
            assert!((fake_quant_fp8_value(v, Fp8Format::E4m3) - v).abs() < 1e-6);
            assert!((fake_quant_fp8_value(v, Fp8Format::E5m2) - v).abs() < 1e-6);
        }
    }

    #[test]
    fn fp8_e4m3_more_precise_than_e5m2() {
        // e4m3 has one more mantissa bit, so it should track 1.1 more closely.
        let v = 1.1_f32;
        let e4 = (fake_quant_fp8_value(v, Fp8Format::E4m3) - v).abs();
        let e5 = (fake_quant_fp8_value(v, Fp8Format::E5m2) - v).abs();
        assert!(e4 <= e5 + 1e-7, "e4m3 err {e4} should be <= e5m2 err {e5}");
    }

    #[test]
    fn qat_loss_reduces_to_ce_when_alpha_zero() {
        let cfg = QatDistillConfig::new(
            1.0,
            0.0,
            QuantKind::Int {
                n_bits: 8,
                symmetric: true,
            },
        )
        .expect("config");
        let s = vec![1.0_f32, 2.0, 3.0, 0.5];
        let t = vec![2.0_f32, 1.0, 3.0, 0.5];
        let loss = qat_kd_loss(&s, &t, 2, &cfg).expect("loss");
        let s_q = quantize_student(&s, cfg.quant);
        let ce = cross_entropy(&s_q, 2);
        assert!((loss - ce).abs() < 1e-5, "loss {loss} vs ce {ce}");
    }

    #[test]
    fn qat_loss_finite_and_nonneg() {
        let cfg = QatDistillConfig::new(
            4.0,
            0.7,
            QuantKind::Fp8 {
                format: Fp8Format::E4m3,
            },
        )
        .expect("config");
        let s = vec![0.3_f32, -1.2, 2.1, 0.0, 1.5];
        let t = vec![1.0_f32, -0.5, 1.8, 0.2, 0.9];
        let loss = qat_kd_loss(&s, &t, 2, &cfg).expect("loss");
        assert!(loss.is_finite() && loss >= 0.0, "loss {loss}");
    }

    #[test]
    fn qat_batch_matches_manual_mean() {
        let cfg = QatDistillConfig::new(
            2.0,
            0.5,
            QuantKind::Int {
                n_bits: 8,
                symmetric: false,
            },
        )
        .expect("config");
        let s = vec![vec![1.0_f32, 2.0, 0.5], vec![0.2_f32, 1.1, 3.0]];
        let t = vec![vec![1.5_f32, 1.0, 0.5], vec![0.5_f32, 0.9, 2.0]];
        let labels = vec![1_usize, 2];
        let batch = qat_kd_loss_batch(&s, &t, &labels, &cfg).expect("batch");
        let m0 = qat_kd_loss(&s[0], &t[0], labels[0], &cfg).expect("l0");
        let m1 = qat_kd_loss(&s[1], &t[1], labels[1], &cfg).expect("l1");
        assert!((batch - 0.5 * (m0 + m1)).abs() < 1e-6);
    }

    #[test]
    fn ste_mask_kills_clamped_values() {
        let quant = QuantKind::Fp8 {
            format: Fp8Format::E4m3,
        };
        let x = vec![0.0_f32, 1.0, 1.0e7];
        let mask = ste_grad_mask(&x, quant);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 0.0);
    }

    #[test]
    fn quantization_error_zero_for_int_grid_points() {
        // Values already on the dequantisation grid have ~zero error after re-quantising.
        let x = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0];
        let q = fake_quant_affine(&x, 8, true);
        let err = quantization_error(
            &q,
            QuantKind::Int {
                n_bits: 8,
                symmetric: true,
            },
        );
        assert!(err < 1e-4, "error {err}");
    }

    #[test]
    fn config_rejects_bad_params() {
        assert!(
            QatDistillConfig::new(
                0.0,
                0.5,
                QuantKind::Int {
                    n_bits: 8,
                    symmetric: true
                }
            )
            .is_err()
        );
        assert!(
            QatDistillConfig::new(
                1.0,
                1.5,
                QuantKind::Int {
                    n_bits: 8,
                    symmetric: true
                }
            )
            .is_err()
        );
        assert!(
            QatDistillConfig::new(
                1.0,
                0.5,
                QuantKind::Int {
                    n_bits: 1,
                    symmetric: true
                }
            )
            .is_err()
        );
    }
}
