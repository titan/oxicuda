//! IPCW (inverse probability of censoring) Brier score at a horizon `t*`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute the IPCW Brier score:
///
/// ```text
///   BS(t*) = (1/n) Σ wᵢ (1{Tᵢ > t*} − Ŝᵢ(t*))²
/// ```
///
/// with weights:
///   - if `Tᵢ ≤ t*` and event=1 → weight = 1/G(Tᵢ)
///   - if `Tᵢ > t*`             → weight = 1/G(t*)
///   - otherwise (censored before t*) → weight = 0
///
/// where `G` is the KM of the censoring distribution.
pub fn ipcw_brier_at(data: &Dataset, s_pred: &[f64], t_star: f64) -> SurvivalResult<f64> {
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
    let mut order = data.order_by_time();
    order.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_total = data.len();
    let mut g_times = Vec::new();
    let mut g_vals = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n_total as f64;
    let mut k = 0usize;
    while k < order.len() {
        let t = data.observations[order[k]].time;
        let mut m = k;
        let mut dc = 0.0_f64;
        while m < order.len() && data.observations[order[m]].time == t {
            if !data.observations[order[m]].event {
                dc += 1.0;
            }
            m += 1;
        }
        if dc > 0.0 && at_risk > 0.0 {
            g_cur *= 1.0 - dc / at_risk;
        }
        g_times.push(t);
        g_vals.push(g_cur);
        at_risk -= (m - k) as f64;
        k = m;
    }
    let g_at = |t: f64| -> f64 {
        let mut v = 1.0_f64;
        for (idx, &gt) in g_times.iter().enumerate() {
            if gt <= t {
                v = g_vals[idx];
            } else {
                break;
            }
        }
        v.max(1.0e-300)
    };
    let n = data.len() as f64;
    let mut s = 0.0_f64;
    for (i, o) in data.observations.iter().enumerate() {
        let weight = if o.time <= t_star && o.event {
            1.0 / g_at(o.time)
        } else if o.time > t_star {
            1.0 / g_at(t_star)
        } else {
            0.0
        };
        if weight <= 0.0 {
            continue;
        }
        // indicator(T > t*) — 1 if alive at horizon, 0 if dead by t*
        let ind = if o.time > t_star { 1.0 } else { 0.0 };
        let diff = ind - s_pred[i];
        s += weight * diff * diff;
    }
    Ok(s / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipcw_brier_no_censoring_reduces_to_naive() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true; 4]).expect("ok");
        let s_pred = vec![0.5; 4];
        let b1 = ipcw_brier_at(&d, &s_pred, 2.5).expect("ok");
        let b2 = crate::calibration::brier_score::brier_score_at(&d, &s_pred, 2.5).expect("ok");
        assert!((b1 - b2).abs() < 1.0e-6);
    }

    #[test]
    fn ipcw_brier_finite_with_censoring() {
        let d =
            Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, false, true, false]).expect("ok");
        let s_pred = vec![0.7; 4];
        let b = ipcw_brier_at(&d, &s_pred, 2.5).expect("ok");
        assert!(b.is_finite());
        assert!(b >= 0.0);
    }
}
