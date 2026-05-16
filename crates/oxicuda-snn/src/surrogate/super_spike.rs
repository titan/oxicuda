//! "SuperSpike" surrogate gradient (Zenke & Ganguli 2018).
//!
//! `g(v) = α / (1 + |v − v_th| · α)²`
//!
//! Smooth, strictly positive, peaked at `v = v_th` with maximum `α`.

use crate::error::{SnnError, SnnResult};

/// SuperSpike surrogate gradient evaluated element-wise.
pub fn super_spike_grad(v: &[f32], v_th: f32, alpha: f32, grad_out: &mut [f32]) -> SnnResult<()> {
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
        let denom = 1.0 + (vi - v_th).abs() * alpha;
        *g = alpha / (denom * denom);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_at_threshold_equals_alpha() {
        let alpha = 1.5_f32;
        let v_th = 0.3_f32;
        let v = vec![v_th];
        let mut g = vec![0.0_f32; 1];
        super_spike_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - alpha).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn positive_everywhere() {
        let alpha = 1.0_f32;
        let v_th = 0.0_f32;
        let v: Vec<f32> = (-50..=50).map(|i| 0.1 * i as f32).collect();
        let mut g = vec![0.0_f32; v.len()];
        super_spike_grad(&v, v_th, alpha, &mut g).expect("ok");
        for &gi in &g {
            assert!(gi > 0.0 && gi.is_finite(), "g={gi}");
        }
    }

    #[test]
    fn symmetric_about_threshold() {
        let alpha = 2.0_f32;
        let v_th = 0.5_f32;
        let v = vec![v_th - 1.0, v_th + 1.0];
        let mut g = vec![0.0_f32; 2];
        super_spike_grad(&v, v_th, alpha, &mut g).expect("ok");
        assert!((g[0] - g[1]).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_alpha() {
        let v = vec![0.0_f32];
        let mut g = vec![0.0_f32];
        let err = super_spike_grad(&v, 0.0, 0.0, &mut g);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
