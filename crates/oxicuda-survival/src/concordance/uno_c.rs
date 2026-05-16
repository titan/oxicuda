//! Uno's C-index: IPCW-weighted concordance accounting for censoring distribution.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute Uno's IPCW-weighted C-index up to truncation time `tau`.
///
/// Pairs where the smaller-time subject's `t_i <= tau` and event=true contribute,
/// each weighted by `1 / G(t_i)²` where G is the KM of the censoring distribution.
pub fn uno_c_index(data: &Dataset, eta: &[f64], tau: f64) -> SurvivalResult<f64> {
    if data.len() != eta.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![eta.len()],
        });
    }
    // Build censoring KM G(t)
    let mut order = data.order_by_time();
    order.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = data.len();
    let mut g_times = Vec::new();
    let mut g_vals = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n as f64;
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
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        let ti = data.observations[i].time;
        let ei = data.observations[i].event;
        if !ei || ti > tau {
            continue;
        }
        let g_i = g_at(ti);
        let weight = 1.0 / (g_i * g_i);
        for j in 0..n {
            if i == j {
                continue;
            }
            let tj = data.observations[j].time;
            if tj <= ti {
                continue;
            }
            comparable += weight;
            if eta[i] > eta[j] {
                concordant += weight;
            } else if (eta[i] - eta[j]).abs() < 1.0e-12 {
                concordant += 0.5 * weight;
            }
        }
    }
    if comparable == 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no comparable Uno pairs".to_string(),
        ));
    }
    Ok(concordant / comparable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uno_no_censoring_matches_harrell() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let eta = vec![4.0, 3.0, 2.0, 1.0];
        let c_uno = uno_c_index(&d, &eta, 10.0).expect("ok");
        assert!((c_uno - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn uno_reverse_ranking_low() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let eta = vec![1.0, 2.0, 3.0, 4.0];
        let c_uno = uno_c_index(&d, &eta, 10.0).expect("ok");
        assert!(c_uno < 0.1);
    }

    #[test]
    fn uno_with_censoring_returns_finite() {
        let d = Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[true, false, true, false, true],
        )
        .expect("ok");
        let eta = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let c = uno_c_index(&d, &eta, 10.0).expect("ok");
        assert!(c.is_finite());
        assert!((0.0..=1.0).contains(&c));
    }
}
