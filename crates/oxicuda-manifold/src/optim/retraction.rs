//! Retraction operators for Riemannian manifolds.

use crate::error::ManifoldResult;
use crate::linalg::householder_qr::{householder_qr, polar_orthogonal};

/// QR retraction on the Stiefel manifold: `qr(Y + Delta).Q[:, :p]`.
pub fn retract_qr_stiefel(
    y: &[f64],
    delta: &[f64],
    n: usize,
    p: usize,
) -> ManifoldResult<Vec<f64>> {
    let mut sum = vec![0.0; n * p];
    for i in 0..n * p {
        sum[i] = y[i] + delta[i];
    }
    let (q, _r) = householder_qr(&sum, n, p)?;
    Ok(q)
}

/// Polar retraction for SPD-like matrices: returns the orthogonal factor.
pub fn retract_polar_spd(m: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    polar_orthogonal(m, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_retract_runs() {
        let n = 3;
        let p = 2;
        let mut y = vec![0.0; n * p];
        y[0] = 1.0;
        y[p + 1] = 1.0;
        let delta = vec![0.01; n * p];
        let q = retract_qr_stiefel(&y, &delta, n, p).expect("ok");
        assert_eq!(q.len(), n * p);
    }

    #[test]
    fn polar_runs() {
        let n = 3;
        let m = vec![2.0, 0.1, 0.0, 0.1, 3.0, 0.0, 0.0, 0.0, 1.5];
        let q = retract_polar_spd(&m, n).expect("ok");
        assert_eq!(q.len(), n * n);
    }
}
