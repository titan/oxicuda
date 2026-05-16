//! Unweighted (naive) Brier score at a horizon.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Naive Brier score `BS(t*) = (1/n) Σ (1{Tᵢ > t*} − Ŝᵢ(t*))²`.
///
/// Indicator `1{Tᵢ > t*}` = 1 if the subject is alive at t*. Censored subjects with
/// `Tᵢ ≤ t*` are treated as "indicator = 0" (lower bound). For a proper estimator
/// under censoring use `ipcw_brier_at`.
pub fn brier_score_at(data: &Dataset, s_pred: &[f64], t_star: f64) -> SurvivalResult<f64> {
    if data.len() != s_pred.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![s_pred.len()],
        });
    }
    if !t_star.is_finite() || t_star <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "t_star must be positive".to_string(),
        ));
    }
    let n = data.len() as f64;
    let mut s = 0.0_f64;
    for (i, o) in data.observations.iter().enumerate() {
        let ind = if o.time > t_star { 1.0 } else { 0.0 };
        let diff = ind - s_pred[i];
        s += diff * diff;
    }
    Ok(s / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brier_zero_when_perfect() {
        // Naive Brier: BS = mean (1{T>t*} - s_pred)^2.
        // To get BS=0, set s_pred[i] = 1 if alive at t*, else 0.
        let d = Dataset::from_arrays(&[1.0, 3.0], &[true, true]).expect("ok");
        let s_pred = vec![0.0, 1.0]; // i=0 dead by t=2 (S=0); i=1 alive at t=2 (S=1)
        let b = brier_score_at(&d, &s_pred, 2.0).expect("ok");
        assert!(b < 1.0e-12);
    }

    #[test]
    fn brier_naive_size_mismatch() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(brier_score_at(&d, &[0.5, 0.5], 1.0).is_err());
    }

    #[test]
    fn brier_constant_half_returns_quarter() {
        let d = Dataset::from_arrays(&[1.0, 1.0, 5.0, 5.0], &[true, true, true, true]).expect("ok");
        let s_pred = vec![0.5; 4];
        let b = brier_score_at(&d, &s_pred, 2.0).expect("ok");
        // 2 alive at t=2 → indicators (0,0,1,1); diffs (-0.5, -0.5, 0.5, 0.5); sq=0.25; avg=0.25
        assert!((b - 0.25).abs() < 1.0e-12);
    }
}
