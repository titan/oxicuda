//! Time-dependent AUC.
//!
//! Cumulative-incidence vs dynamic-survivor formulation:
//!   - **Cases**: subjects with `Tᵢ ≤ t*` and `δᵢ=1`.
//!   - **Controls**: subjects with `Tᵢ > t*`.
//!
//! AUC(t*) = P(η_case > η_control).

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute time-dependent AUC at horizon `t_star`.
///
/// Pair-wise (case, control) comparison; tied scores contribute 0.5.
pub fn time_dependent_auc(data: &Dataset, eta: &[f64], t_star: f64) -> SurvivalResult<f64> {
    if data.len() != eta.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![eta.len()],
        });
    }
    let mut cases: Vec<usize> = Vec::new();
    let mut controls: Vec<usize> = Vec::new();
    for (i, o) in data.observations.iter().enumerate() {
        if o.time <= t_star && o.event {
            cases.push(i);
        } else if o.time > t_star {
            controls.push(i);
        }
    }
    if cases.is_empty() || controls.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "no cases or controls for AUC".to_string(),
        ));
    }
    let mut score = 0.0_f64;
    let n_pairs = (cases.len() * controls.len()) as f64;
    for &i in &cases {
        for &j in &controls {
            if eta[i] > eta[j] {
                score += 1.0;
            } else if (eta[i] - eta[j]).abs() < 1.0e-12 {
                score += 0.5;
            }
        }
    }
    Ok(score / n_pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_perfect_one() {
        let d =
            Dataset::from_arrays(&[1.0, 2.0, 5.0, 6.0], &[true, true, false, false]).expect("ok");
        // cases at t<=3: subjects 0,1; controls: 2,3
        let eta = vec![3.0, 4.0, 1.0, 2.0];
        let a = time_dependent_auc(&d, &eta, 3.0).expect("ok");
        assert!((a - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn auc_reverse_zero() {
        let d =
            Dataset::from_arrays(&[1.0, 2.0, 5.0, 6.0], &[true, true, false, false]).expect("ok");
        let eta = vec![1.0, 2.0, 5.0, 6.0];
        let a = time_dependent_auc(&d, &eta, 3.0).expect("ok");
        assert!(a < 0.1);
    }

    #[test]
    fn auc_no_cases_errors() {
        let d = Dataset::from_arrays(&[5.0, 6.0], &[false, false]).expect("ok");
        let eta = vec![1.0, 2.0];
        assert!(time_dependent_auc(&d, &eta, 3.0).is_err());
    }
}
