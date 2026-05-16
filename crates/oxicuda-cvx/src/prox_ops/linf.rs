//! L∞ proximal operator via Moreau decomposition.
//!
//! `prox_{λ||·||_∞}(v) = v − λ · Π_{B_1}(v/λ)` where `B_1` is the L1 unit ball.

use crate::error::{CvxError, CvxResult};
use crate::projection::project_l1_ball;

/// Prox of `λ ||·||_∞`.  Returns `v − projection of v onto the L1 ball of radius λ` (Moreau).
pub fn prox_linf(v: &[f64], lambda: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "L∞ prox requires lambda ≥ 0, got {lambda}"
        )));
    }
    let p = project_l1_ball(v, lambda)?;
    Ok(v.iter().zip(p.iter()).map(|(vi, pi)| vi - pi).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_linf_zero_lambda_zero_norm() {
        let v = vec![1.0, 2.0, 3.0];
        let p = prox_linf(&v, 0.0).expect("ok");
        // With lambda=0, B_1 is the origin → projection is 0 → prox is v.
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - vi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn prox_linf_caps_largest() {
        // L∞ prox should shrink largest magnitude entries.
        let v = vec![5.0, 1.0];
        let p = prox_linf(&v, 1.0).expect("ok");
        // The largest entry should be reduced; the small one less affected.
        assert!(p[0].abs() < v[0].abs() + 1.0e-12);
        // Sum of |v - p| should equal lambda (since we projected onto L1 ball of radius lambda).
        let diff: f64 = v.iter().zip(p.iter()).map(|(vi, pi)| (vi - pi).abs()).sum();
        assert!((diff - 1.0).abs() < 1.0e-10);
    }
}
