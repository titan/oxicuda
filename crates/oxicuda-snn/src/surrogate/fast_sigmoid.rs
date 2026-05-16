//! Fast-sigmoid surrogate gradient.
//!
//! `g(v) = α / (1 + |α(v − v_th)|)²`
//!
//! Identical max value `α` at `v = v_th` to SuperSpike, but with the `α` factor
//! moved inside the absolute value, giving a sharper peak as `α` grows.

use crate::error::{SnnError, SnnResult};

/// Fast-sigmoid surrogate gradient evaluated element-wise.
pub fn fast_sigmoid_grad(v: &[f32], v_th: f32, alpha: f32, grad_out: &mut [f32]) -> SnnResult<()> {
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
    for (&vi, g) in v.iter().zip(grad_out.iter_mut()) {
        let denom = 1.0 + (alpha * (vi - v_th)).abs();
        *g = alpha / (denom * denom);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_at_threshold_equals_alpha() {
        let alpha = 2.5_f32;
        let v_th = -0.2_f32;
        let v = vec![v_th];
        let mut g = vec![0.0_f32; 1];
        fast_sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - alpha).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn symmetric_about_threshold() {
        let alpha = 1.0_f32;
        let v_th = 0.0_f32;
        let v = vec![-0.4_f32, 0.4_f32];
        let mut g = vec![0.0_f32; 2];
        fast_sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - g[1]).abs() < 1e-6);
    }

    #[test]
    fn decreasing_with_distance() {
        let alpha = 1.0_f32;
        let v_th = 0.0_f32;
        let v = vec![0.0_f32, 0.5, 1.0, 5.0];
        let mut g = vec![0.0_f32; 4];
        fast_sigmoid_grad(&v, v_th, alpha, &mut g).expect("ok");
        for w in g.windows(2) {
            assert!(w[0] >= w[1], "not decreasing: {} vs {}", w[0], w[1]);
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let v = vec![0.0_f32];
        let mut g = vec![0.0_f32];
        let err = fast_sigmoid_grad(&v, 0.0, -2.0, &mut g);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
