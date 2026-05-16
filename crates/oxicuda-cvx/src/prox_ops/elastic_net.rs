//! Elastic net proximal operator: `g(x) = λ_1 ||x||_1 + (λ_2/2) ||x||²`.

use crate::error::{CvxError, CvxResult};
use crate::prox_ops::l1::soft_threshold;

/// Elastic-net prox: soft-threshold then scale by `1/(1 + λ_2)`.
pub fn prox_elastic_net(v: &[f64], lambda1: f64, lambda2: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lambda1.is_finite() || !lambda2.is_finite() || lambda1 < 0.0 || lambda2 < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "Elastic net requires non-negative lambdas, got ({lambda1}, {lambda2})"
        )));
    }
    let scale = 1.0 / (1.0 + lambda2);
    Ok(v.iter()
        .map(|x| scale * soft_threshold(*x, lambda1))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_net_reduces_to_l1() {
        let v = vec![2.0, 0.5, -2.0];
        let p = prox_elastic_net(&v, 1.0, 0.0).expect("ok");
        assert!((p[0] - 1.0).abs() < 1.0e-12);
        assert!(p[1].abs() < 1.0e-12);
        assert!((p[2] + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn elastic_net_reduces_to_l2() {
        let v = vec![1.0, 2.0];
        let p = prox_elastic_net(&v, 0.0, 1.0).expect("ok");
        assert!((p[0] - 0.5).abs() < 1.0e-12);
        assert!((p[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn elastic_net_combined() {
        let v = vec![3.0];
        let p = prox_elastic_net(&v, 1.0, 1.0).expect("ok");
        // soft_threshold(3, 1) = 2; then /2 = 1.
        assert!((p[0] - 1.0).abs() < 1.0e-12);
    }
}
