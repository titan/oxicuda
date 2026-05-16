//! Survival losses for use as PyTorch-style training objectives.

use crate::calibration::brier_score::brier_score_at;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Cox negative partial log-likelihood (Breslow).
///
/// `loss(η) = − Σ_{events} (η_i − log S0(t_i))`
pub fn cox_loss(data: &Dataset, eta: &[f64]) -> SurvivalResult<f64> {
    if data.len() != eta.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![eta.len()],
        });
    }
    let n = data.len();
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // S0 at each unique event time
    let w: Vec<f64> = eta.iter().map(|x| x.exp()).collect();
    let mut suffix = vec![0.0_f64; n + 1];
    for k in (0..n).rev() {
        suffix[k] = suffix[k + 1] + w[idx[k]];
    }
    let mut loss = 0.0_f64;
    let mut k = 0usize;
    while k < n {
        let t = data.observations[idx[k]].time;
        let s0 = suffix[k];
        let mut m = k;
        let mut etabsum = 0.0_f64;
        let mut d_count = 0.0_f64;
        while m < n && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                etabsum += eta[idx[m]];
                d_count += 1.0;
            }
            m += 1;
        }
        if d_count > 0.0 && s0 > 0.0 {
            loss -= etabsum - d_count * s0.ln();
        }
        k = m;
    }
    Ok(loss)
}

/// Brier loss = naive Brier score at horizon `t_star`, useful as classification-style loss.
pub fn brier_loss(data: &Dataset, s_pred: &[f64], t_star: f64) -> SurvivalResult<f64> {
    brier_score_at(data, s_pred, t_star)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cox_loss_finite() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true; 3]).expect("ok");
        let eta = vec![0.5, 0.0, -0.5];
        let l = cox_loss(&d, &eta).expect("ok");
        assert!(l.is_finite());
    }

    #[test]
    fn cox_loss_decreases_for_correct_ordering() {
        // higher risk dies first: low loss
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true; 4]).expect("ok");
        let eta_good = vec![2.0, 1.0, -1.0, -2.0];
        let eta_bad = vec![-2.0, -1.0, 1.0, 2.0];
        let lg = cox_loss(&d, &eta_good).expect("ok");
        let lb = cox_loss(&d, &eta_bad).expect("ok");
        assert!(lg < lb);
    }

    #[test]
    fn brier_loss_delegates() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let s_pred = vec![0.5, 0.5];
        let b = brier_loss(&d, &s_pred, 1.5).expect("ok");
        let b2 = brier_score_at(&d, &s_pred, 1.5).expect("ok");
        assert!((b - b2).abs() < 1.0e-12);
    }
}
