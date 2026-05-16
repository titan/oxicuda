//! Empirical Restricted Isometry Property (RIP) constant estimator.
//!
//! The RIP constant `δ_K` for matrix Φ is the smallest δ such that
//! `(1 - δ) ||x||² ≤ ||Φ x||² ≤ (1 + δ) ||x||²` for all K-sparse x.
//!
//! We approximate `δ_K` by sampling random K-column submatrices and computing extreme
//! singular values via `Σ = svd(Φ_S)` then `δ_S = max(σ_max²-1, 1-σ_min²)`.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;
use crate::linalg::jacobi_svd::jacobi_svd_thin;
use crate::linalg::submat_columns;

/// Empirical RIP constant estimate via random K-subset SVDs.
///
/// Returns the maximum δ observed over `num_samples` random K-subsets.
pub fn rip_estimator(
    phi: &[f64],
    m: usize,
    n: usize,
    k: usize,
    num_samples: usize,
    rng: &mut LcgRng,
) -> CsResult<f64> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if k == 0 || k > n.min(m) {
        return Err(CsError::InvalidSparsity(k));
    }
    if num_samples == 0 {
        return Err(CsError::InvalidParameter("num_samples = 0".into()));
    }
    let mut worst = 0.0_f64;
    let mut indices: Vec<usize> = (0..n).collect();
    for _ in 0..num_samples {
        // Random k-subset via Fisher-Yates prefix.
        for i in 0..k {
            let j = i + rng.next_usize(n - i);
            indices.swap(i, j);
        }
        let mut subset = indices[..k].to_vec();
        subset.sort();
        let sub = submat_columns(phi, m, n, &subset)?;
        let (_u, s, _v) = jacobi_svd_thin(&sub, m, k)?;
        let sigma_max = s[0];
        let sigma_min = *s.last().unwrap_or(&0.0);
        let d_hi = (sigma_max * sigma_max - 1.0).abs();
        let d_lo = (1.0 - sigma_min * sigma_min).abs();
        let d = d_hi.max(d_lo);
        if d > worst {
            worst = d;
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::gaussian_matrix::gaussian_matrix;

    #[test]
    fn rip_estimator_runs() {
        let mut rng = LcgRng::new(2);
        let m = 20;
        let n = 50;
        let phi = gaussian_matrix(m, n, &mut rng).expect("ok");
        let mut rng2 = LcgRng::new(7);
        let d = rip_estimator(&phi, m, n, 3, 20, &mut rng2).expect("ok");
        // For random Gaussian (1/√m) and K=3, RIP constant should be modest (< 2).
        assert!(d.is_finite());
        assert!(d < 2.0);
    }
}
