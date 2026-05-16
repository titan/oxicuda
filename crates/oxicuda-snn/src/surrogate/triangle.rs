//! Triangular surrogate gradient.
//!
//! `g(v) = max(0, 1 − |v − v_th| / α)`
//!
//! Compact support `[v_th − α, v_th + α]`; piecewise-linear; zero elsewhere.

use crate::error::{SnnError, SnnResult};

/// Triangular surrogate gradient evaluated element-wise.
pub fn triangle_grad(v: &[f32], v_th: f32, alpha: f32, grad_out: &mut [f32]) -> SnnResult<()> {
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
        let val = 1.0 - (vi - v_th).abs() / alpha;
        *g = if val < 0.0 { 0.0 } else { val };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_exactly_within_alpha() {
        let alpha = 0.5_f32;
        let v_th = 1.0_f32;
        // Inside support
        let v_in = vec![v_th - 0.49, v_th, v_th + 0.49];
        let mut g_in = vec![0.0_f32; 3];
        triangle_grad(&v_in, v_th, alpha, &mut g_in).expect("ok");
        for &g in &g_in {
            assert!(g > 0.0, "should be positive in support: {g}");
        }
        // Outside support
        let v_out = vec![v_th - alpha - 0.01, v_th + alpha + 0.01, v_th + 2.0];
        let mut g_out = vec![0.0_f32; 3];
        triangle_grad(&v_out, v_th, alpha, &mut g_out).expect("ok");
        for &g in &g_out {
            assert_eq!(g, 0.0, "should be zero outside support: {g}");
        }
    }

    #[test]
    fn max_at_threshold_equals_one() {
        let alpha = 0.7_f32;
        let v_th = 0.0_f32;
        let v = vec![v_th];
        let mut g = vec![0.0_f32; 1];
        triangle_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn boundary_is_zero() {
        let alpha = 0.5_f32;
        let v_th = 0.0_f32;
        let v = vec![-alpha, alpha];
        let mut g = vec![0.0_f32; 2];
        triangle_grad(&v, v_th, alpha, &mut g).expect("ok");
        for &gi in &g {
            assert!(gi.abs() < 1e-6, "boundary should be ~0: {gi}");
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let v = vec![0.0_f32];
        let mut g = vec![0.0_f32];
        let err = triangle_grad(&v, 0.0, 0.0, &mut g);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
