//! Integrated Brier score `IBS = (1/τ) ∫_0^τ BS(t) dt`.

use crate::calibration::ipcw_brier::ipcw_brier_at;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Integrated Brier across a grid of times `[0, tau]`.
///
/// `s_pred_at[k][i]` = predicted S_i(t_k) for time-point k.
pub fn integrated_brier_score(
    data: &Dataset,
    s_pred_at: &[Vec<f64>],
    times: &[f64],
    tau: f64,
) -> SurvivalResult<f64> {
    if times.len() != s_pred_at.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![times.len()],
            got: vec![s_pred_at.len()],
        });
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "tau must be positive".to_string(),
        ));
    }
    if times.is_empty() {
        return Ok(0.0);
    }
    // Compute BS at each time, then trapezoidal integrate
    let mut bs = Vec::with_capacity(times.len());
    for (k, &t) in times.iter().enumerate() {
        if t <= 0.0 || t > tau {
            bs.push(0.0);
            continue;
        }
        bs.push(ipcw_brier_at(data, &s_pred_at[k], t)?);
    }
    let mut area = 0.0_f64;
    for k in 0..(times.len() - 1) {
        let dt = times[k + 1] - times[k];
        area += 0.5 * (bs[k] + bs[k + 1]) * dt;
    }
    Ok(area / tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibs_returns_finite_for_perfect_at_each_t() {
        let d = Dataset::from_arrays(&[2.0, 4.0], &[true, true]).expect("ok");
        let times = vec![1.0, 3.0];
        // perfect: at t=1 nobody dead (S=1,1); at t=3 subject 0 dead (S=0,1)
        let s_pred_at = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let ibs = integrated_brier_score(&d, &s_pred_at, &times, 5.0).expect("ok");
        // The IPCW Brier uses censoring KM but here all are events so it's stable
        assert!(ibs.is_finite());
        assert!(ibs >= 0.0);
    }

    #[test]
    fn ibs_rejects_negative_tau() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(integrated_brier_score(&d, &[vec![0.5]], &[1.0], -1.0).is_err());
    }

    #[test]
    fn ibs_shape_mismatch() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(integrated_brier_score(&d, &[vec![0.5]], &[1.0, 2.0], 5.0).is_err());
    }
}
