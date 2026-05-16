//! Projection onto the probability simplex `{x ∈ R^n : Σx_i = z, x_i ≥ 0}`.
//!
//! Reference: Wang & Carreira-Perpiñán (2013), "Projection onto the probability simplex".

use crate::error::{CvxError, CvxResult};

/// Project `v` onto `{x : Σx = z, x ≥ 0}` with `z > 0` (default z=1).
///
/// O(n log n) sort-based algorithm: sort `v` descending into `u`, then find largest
/// `rho` such that `u[rho] - (sum_{i≤rho} u[i] - z) / (rho+1) > 0`, set
/// `tau = (sum_{i≤rho} u[i] - z)/(rho+1)`, project as `max(v - tau, 0)`.
pub fn project_simplex(v: &[f64], z: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !z.is_finite() || z <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "simplex projection requires z > 0, got {z}"
        )));
    }
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum_sum = 0.0_f64;
    let mut rho = 0usize;
    let mut tau = 0.0_f64;
    let mut found = false;
    for (k, &uk) in u.iter().enumerate().take(n) {
        cum_sum += uk;
        let candidate = (cum_sum - z) / (k as f64 + 1.0);
        if uk - candidate > 0.0 {
            rho = k;
            tau = candidate;
            found = true;
        } else {
            break;
        }
    }
    if !found {
        // All entries below threshold — degenerate (shouldn't happen for finite v with z>0).
        return Err(CvxError::NumericalInstability(
            "simplex projection threshold search failed".into(),
        ));
    }
    let _ = rho; // for clarity, but tau is what we use
    Ok(v.iter().map(|x| (x - tau).max(0.0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_uniform() {
        let v = vec![1.0, 1.0, 1.0];
        let p = project_simplex(&v, 1.0).expect("ok");
        for &pi in &p {
            assert!((pi - 1.0 / 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn simplex_already_in() {
        let v = vec![0.5, 0.3, 0.2];
        let p = project_simplex(&v, 1.0).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-12);
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - vi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn simplex_neg_clipped() {
        let v = vec![1.0, -1.0, 1.0];
        let p = project_simplex(&v, 1.0).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-12);
        assert!(p.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn simplex_sums_to_z() {
        let v = vec![5.0, 1.0, -3.0, 2.0];
        let p = project_simplex(&v, 2.0).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 2.0).abs() < 1.0e-10);
    }
}
