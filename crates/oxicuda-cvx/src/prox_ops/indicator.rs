//! Prox for indicator functions: projection onto the underlying convex set.

use crate::error::CvxResult;
use crate::projection::{project_box, project_l1_ball, project_l2_ball, project_simplex};

/// Prox of indicator of box `[lo, hi]`.
pub fn prox_indicator_box(v: &[f64], lo: f64, hi: f64) -> CvxResult<Vec<f64>> {
    project_box(v, lo, hi)
}

/// Prox of indicator of probability simplex with sum z.
pub fn prox_indicator_simplex(v: &[f64], z: f64) -> CvxResult<Vec<f64>> {
    project_simplex(v, z)
}

/// Prox of indicator of L1 ball of radius r.
pub fn prox_indicator_l1_ball(v: &[f64], r: f64) -> CvxResult<Vec<f64>> {
    project_l1_ball(v, r)
}

/// Prox of indicator of L2 ball of radius r.
pub fn prox_indicator_l2_ball(v: &[f64], r: f64) -> CvxResult<Vec<f64>> {
    project_l2_ball(v, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_box_indicator_matches_projection() {
        let v = vec![-2.0, 0.0, 2.0];
        let p = prox_indicator_box(&v, -1.0, 1.0).expect("ok");
        assert_eq!(p, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn prox_simplex_indicator_sums_to_one() {
        let v = vec![1.0, 2.0, 3.0];
        let p = prox_indicator_simplex(&v, 1.0).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-10);
    }
}
