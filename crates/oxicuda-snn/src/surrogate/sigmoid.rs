//! Logistic-sigmoid surrogate gradient.
//!
//! `g(v) = α · σ(α(v − v_th)) · (1 − σ(α(v − v_th)))`
//!
//! Uses a numerically stable two-branch sigmoid that avoids overflow at
//! large negative inputs.

use crate::error::{SnnError, SnnResult};

/// Numerically stable logistic sigmoid `σ(x) = 1 / (1 + exp(−x))`.
#[must_use]
pub fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Validate inputs shared by all surrogate gradient kernels.
fn validate(v: &[f32], alpha: f32, grad_out: &[f32]) -> SnnResult<()> {
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "alpha".into(),
            val: alpha,
        });
    }
    if v.len() != grad_out.len() {
        return Err(SnnError::IncompatibleLength {
            a: v.len(),
            b: grad_out.len(),
        });
    }
    Ok(())
}

/// Sigmoid surrogate gradient evaluated element-wise.
pub fn sigmoid_grad(v: &[f32], v_th: f32, alpha: f32, grad_out: &mut [f32]) -> SnnResult<()> {
    validate(v, alpha, grad_out)?;
    for (&vi, g) in v.iter().zip(grad_out.iter_mut()) {
        let x = alpha * (vi - v_th);
        let s = stable_sigmoid(x);
        *g = alpha * s * (1.0 - s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_at_threshold_equals_alpha_over_four() {
        let alpha = 2.0_f32;
        let v_th = 0.5_f32;
        let v = vec![v_th];
        let mut g = vec![0.0_f32; 1];
        sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - alpha / 4.0).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn symmetric_around_threshold() {
        let alpha = 1.5_f32;
        let v_th = 0.0_f32;
        let v = vec![-0.7_f32, 0.7_f32];
        let mut g = vec![0.0_f32; 2];
        sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - g[1]).abs() < 1e-6, "{} vs {}", g[0], g[1]);
    }

    #[test]
    fn finite_at_extremes() {
        let alpha = 1.0_f32;
        let v_th = 0.0_f32;
        let v = vec![-1e6_f32, 1e6_f32];
        let mut g = vec![0.0_f32; 2];
        sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        for &gi in &g {
            assert!(gi.is_finite(), "g not finite: {gi}");
            assert!(gi >= 0.0 && gi <= alpha / 4.0 + 1e-6);
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let v = vec![0.0_f32];
        let mut g = vec![0.0_f32];
        let err = sigmoid_grad(&v, 0.0, 0.0, &mut g);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_length_mismatch() {
        let v = vec![0.0_f32; 2];
        let mut g = vec![0.0_f32; 3];
        let err = sigmoid_grad(&v, 0.0, 1.0, &mut g);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
