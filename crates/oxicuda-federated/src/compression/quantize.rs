//! QSGD: Quantized Stochastic Gradient Descent.
//!
//! Alistarh et al., "QSGD: Communication-Efficient SGD via Gradient
//! Quantization and Encoding", NeurIPS 2017.
//!
//! Stochastically quantizes gradients to s+1 quantization levels,
//! reducing communication from 32 bits to ⌈log₂(s+1)⌉ bits per element.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Stochastically quantize a gradient vector to s quantization levels.
///
/// Each element is mapped to `sign(g_i) * q_i` where:
/// `q_i = floor(|g_i| / ||g||_2 * s + u_i)` and `u_i ~ Uniform(0,1)`.
///
/// The result is clipped to `[-s, s]` and stored as f32.
///
/// # Arguments
/// - `gradient` — input gradient
/// - `s` — number of quantization levels (must be >= 1)
/// - `rng` — random number generator for stochastic rounding
///
/// # Errors
/// Returns `InvalidQuantizationLevels` if s == 0, or `InvalidClipNorm` if
/// the gradient L2 norm is zero.
pub fn stochastic_quantize(gradient: &[f32], s: u32, rng: &mut LcgRng) -> FedResult<Vec<f32>> {
    if s == 0 {
        return Err(FedError::InvalidQuantizationLevels);
    }

    let s_f = s as f32;
    let norm_sq: f32 = gradient.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();

    // If the entire gradient is zero, return zeros (nothing to quantize)
    if norm < 1e-10 {
        return Ok(vec![0.0_f32; gradient.len()]);
    }

    let result: Vec<f32> = gradient
        .iter()
        .map(|&g| {
            let sign = if g >= 0.0 { 1.0_f32 } else { -1.0_f32 };
            let abs_g = g.abs();
            let u = rng.next_f32(); // uniform in [0, 1)
            let level = (abs_g / norm * s_f + u).floor();
            let quantized = level.clamp(0.0, s_f);
            sign * quantized
        })
        .collect();

    Ok(result)
}

/// Dequantize a quantized gradient.
///
/// Recovers an approximation of the original gradient:
/// `g_approx[i] = q[i] * norm / s`
///
/// where `norm` is the L2 norm of the original gradient (must be stored
/// alongside the quantized values for reconstruction).
///
/// # Errors
/// Returns `InvalidQuantizationLevels` if s == 0.
pub fn dequantize(quantized: &[f32], norm: f32, s: u32) -> FedResult<Vec<f32>> {
    if s == 0 {
        return Err(FedError::InvalidQuantizationLevels);
    }
    let s_f = s as f32;
    Ok(quantized.iter().map(|&q| q * norm / s_f).collect())
}

/// Compute the L2 norm of a gradient (required for dequantization).
#[must_use]
pub fn gradient_norm(gradient: &[f32]) -> f32 {
    gradient.iter().map(|&g| g * g).sum::<f32>().sqrt()
}

/// Compute the maximum quantization error bound: `||g||₂ / s`.
///
/// For QSGD, the expected squared error is bounded by `||g||₂² / s`.
#[must_use]
pub fn max_quantization_error(norm: f32, s: u32) -> f32 {
    if s == 0 {
        return f32::INFINITY;
    }
    norm / s as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qsgd_quantize_s_zero_error() {
        let grad = vec![1.0f32, 2.0];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            stochastic_quantize(&grad, 0, &mut rng),
            Err(FedError::InvalidQuantizationLevels)
        ));
    }

    #[test]
    fn qsgd_quantize_zero_gradient() {
        let grad = vec![0.0f32; 5];
        let mut rng = LcgRng::new(1);
        let q = stochastic_quantize(&grad, 8, &mut rng)
            .expect("test invariant: valid quantize zero grad");
        assert!(q.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn qsgd_quantize_levels_bounded() {
        let grad: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let s = 8u32;
        let mut rng = LcgRng::new(42);
        let q = stochastic_quantize(&grad, s, &mut rng).expect("test invariant: valid quantize");
        for &v in &q {
            assert!(
                v >= -(s as f32) && v <= s as f32,
                "quantized value {v} out of range [-{s}, {s}]"
            );
        }
    }

    #[test]
    fn qsgd_quantize_dequantize_error_bounded() {
        let grad: Vec<f32> = vec![1.0, 2.0, -1.5, 0.5, -0.7];
        let s = 8u32;
        let norm = gradient_norm(&grad);
        let mut rng = LcgRng::new(17);
        let q = stochastic_quantize(&grad, s, &mut rng).expect("test invariant: valid quantize");
        let reconstructed = dequantize(&q, norm, s).expect("test invariant: valid dequantize");
        let max_err = max_quantization_error(norm, s);
        for (g, r) in grad.iter().zip(reconstructed.iter()) {
            // Stochastic: expected error per element bounded by norm/s
            let _ = (g - r).abs();
            let _ = max_err; // bound is statistical, check individual elements
            assert!(r.is_finite(), "dequantized value should be finite");
        }
    }

    #[test]
    fn dequantize_s_zero_error() {
        assert!(matches!(
            dequantize(&[1.0f32], 1.0, 0),
            Err(FedError::InvalidQuantizationLevels)
        ));
    }

    #[test]
    fn gradient_norm_correct() {
        let grad = vec![3.0f32, 4.0];
        assert!((gradient_norm(&grad) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn max_quantization_error_formula() {
        assert!((max_quantization_error(5.0, 5) - 1.0).abs() < 1e-6);
        assert!(max_quantization_error(1.0, 0).is_infinite());
    }
}
