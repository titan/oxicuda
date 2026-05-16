//! L2 (Tikhonov) proximal operator.

use crate::error::{CvxError, CvxResult};

/// Prox for `g(x) = (λ/2) ||x||²`:  `prox_{(λ/2)||·||²}(v) = v / (1 + λ)`.
pub fn prox_l2(v: &[f64], lambda: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "L2 prox requires lambda ≥ 0, got {lambda}"
        )));
    }
    let factor = 1.0 / (1.0 + lambda);
    Ok(v.iter().map(|x| factor * x).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_l2_scales() {
        let v = vec![1.0, 2.0, 3.0];
        let p = prox_l2(&v, 1.0).expect("ok");
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - 0.5 * vi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn prox_l2_zero_lambda_identity() {
        let v = vec![1.0, -1.0];
        let p = prox_l2(&v, 0.0).expect("ok");
        assert_eq!(p, v);
    }
}
