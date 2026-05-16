//! `arctan` surrogate gradient.
//!
//! `g(v) = α / (π · (1 + (α(v − v_th))²))`
//!
//! Maximum value `α/π` at `v = v_th`; decreases monotonically with `|v − v_th|`.

use crate::error::{SnnError, SnnResult};

const INV_PI: f32 = std::f32::consts::FRAC_1_PI;

/// `arctan` surrogate gradient evaluated element-wise.
pub fn atan_grad(v: &[f32], v_th: f32, alpha: f32, grad_out: &mut [f32]) -> SnnResult<()> {
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
        let x = alpha * (vi - v_th);
        *g = alpha * INV_PI / (1.0 + x * x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_at_threshold_equals_alpha_over_pi() {
        let alpha = 2.0_f32;
        let v_th = 0.0_f32;
        let v = vec![v_th];
        let mut g = vec![0.0_f32; 1];
        atan_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - alpha * INV_PI).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn decreasing_away_from_threshold() {
        let alpha = 1.0_f32;
        let v_th = 0.0_f32;
        let v = vec![0.0_f32, 0.5, 1.0, 5.0];
        let mut g = vec![0.0_f32; 4];
        atan_grad(&v, v_th, alpha, &mut g).expect("ok");
        for w in g.windows(2) {
            assert!(w[0] >= w[1], "not decreasing: {} vs {}", w[0], w[1]);
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let v = vec![0.0_f32];
        let mut g = vec![0.0_f32];
        let err = atan_grad(&v, 0.0, -1.0, &mut g);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
