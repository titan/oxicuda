//! Group lasso proximal operator.
//!
//! For `g(x) = λ · Σ_g ||x_g||_2`, the prox is block soft-thresholding on each group's L2 norm.

use crate::error::{CvxError, CvxResult};

/// Group lasso prox: groups are contiguous index ranges given as `(start, end)` pairs.
pub fn prox_group_lasso(v: &[f64], groups: &[(usize, usize)], lambda: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "Group lasso requires lambda ≥ 0, got {lambda}"
        )));
    }
    let mut out = v.to_vec();
    for &(s, e) in groups {
        if e > v.len() || s >= e {
            return Err(CvxError::IndexOutOfBounds {
                index: e,
                len: v.len(),
            });
        }
        let mut nrm_sq = 0.0_f64;
        for &val in &v[s..e] {
            nrm_sq += val * val;
        }
        let nrm = nrm_sq.sqrt();
        if nrm <= lambda {
            for slot in &mut out[s..e] {
                *slot = 0.0;
            }
        } else {
            let factor = 1.0 - lambda / nrm;
            for (slot, vi) in out[s..e].iter_mut().zip(v[s..e].iter()) {
                *slot = factor * vi;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_lasso_small_groups_zeroed() {
        let v = vec![0.1, 0.1, 10.0, 10.0];
        let p = prox_group_lasso(&v, &[(0, 2), (2, 4)], 1.0).expect("ok");
        assert!(p[0].abs() < 1.0e-12);
        assert!(p[1].abs() < 1.0e-12);
        // For the (10, 10) group: ||v||=10√2 ≈ 14.14; factor = 1 - 1/14.14
        let nrm = (200.0_f64).sqrt();
        let factor = 1.0 - 1.0 / nrm;
        assert!((p[2] - 10.0 * factor).abs() < 1.0e-9);
        assert!((p[3] - 10.0 * factor).abs() < 1.0e-9);
    }
}
