//! DeepSurv-style linear head wrapping the log-risk computation.

use crate::error::{SurvivalError, SurvivalResult};

/// Output of a deep survival linear head.
#[derive(Debug, Clone)]
pub struct DeepSurvOutput {
    pub eta: Vec<f64>,
    pub risk: Vec<f64>,
}

/// Apply a linear head `η = X · w + b` to feature vectors and return log-risk + risk.
pub fn deep_surv_head(
    features: &[Vec<f64>],
    weights: &[f64],
    bias: f64,
) -> SurvivalResult<DeepSurvOutput> {
    if features.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let p = weights.len();
    for (i, row) in features.iter().enumerate() {
        if row.len() != p {
            return Err(SurvivalError::ShapeMismatch {
                expected: vec![p],
                got: vec![row.len()],
            });
        }
        for v in row {
            if !v.is_finite() {
                return Err(SurvivalError::InvalidParameter(format!(
                    "non-finite feature at row {i}"
                )));
            }
        }
    }
    let mut eta = Vec::with_capacity(features.len());
    let mut risk = Vec::with_capacity(features.len());
    for row in features {
        let dot: f64 = row.iter().zip(weights.iter()).map(|(a, b)| a * b).sum();
        let h = dot + bias;
        eta.push(h);
        risk.push(h.exp());
    }
    Ok(DeepSurvOutput { eta, risk })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_simple_linear() {
        let f = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let w = vec![0.5, -0.25];
        let out = deep_surv_head(&f, &w, 0.0).expect("ok");
        assert!((out.eta[0] - (0.5 - 0.5)).abs() < 1.0e-12);
        assert!((out.eta[1] - (1.5 - 1.0)).abs() < 1.0e-12);
    }

    #[test]
    fn head_size_mismatch() {
        let f = vec![vec![1.0, 2.0]];
        let w = vec![0.5];
        assert!(deep_surv_head(&f, &w, 0.0).is_err());
    }

    #[test]
    fn head_rejects_nan_feature() {
        let f = vec![vec![f64::NAN]];
        let w = vec![1.0];
        assert!(deep_surv_head(&f, &w, 0.0).is_err());
    }
}
