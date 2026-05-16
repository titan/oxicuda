//! L1 proximal operator (soft-thresholding).

use crate::error::{CvxError, CvxResult};

/// Scalar soft threshold: `sign(v) · max(|v| − λ, 0)`.
#[must_use]
pub fn soft_threshold(v: f64, lambda: f64) -> f64 {
    if !lambda.is_finite() || lambda < 0.0 {
        return v;
    }
    let mag = v.abs() - lambda;
    if mag <= 0.0 {
        0.0
    } else if v >= 0.0 {
        mag
    } else {
        -mag
    }
}

/// Vector L1 prox: element-wise soft threshold by `lambda`.
pub fn prox_l1(v: &[f64], lambda: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "L1 prox requires lambda ≥ 0, got {lambda}"
        )));
    }
    Ok(v.iter().map(|x| soft_threshold(*x, lambda)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_threshold_zeros_in_band() {
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.7, 1.0), 0.0);
    }

    #[test]
    fn soft_threshold_shrinks_outside_band() {
        assert!((soft_threshold(2.0, 1.0) - 1.0).abs() < 1.0e-12);
        assert!((soft_threshold(-3.0, 1.0) + 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn prox_l1_doc_example() {
        let v = [2.0, 0.5, -0.5, -2.0];
        let p = prox_l1(&v, 1.0).expect("ok");
        assert!((p[0] - 1.0).abs() < 1.0e-12);
        assert!(p[1].abs() < 1.0e-12);
        assert!(p[2].abs() < 1.0e-12);
        assert!((p[3] + 1.0).abs() < 1.0e-12);
    }
}
