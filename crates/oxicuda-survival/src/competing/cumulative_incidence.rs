//! Cumulative incidence function (Aalen-Johansen) for competing risks.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Estimated cumulative incidence for a single cause across event times.
#[derive(Debug, Clone)]
pub struct CifEstimate {
    pub times: Vec<f64>,
    /// CIF F_k(t) = Σ_{t_i ≤ t} S(t_i^-) * d_{k,i} / n_i
    pub cif: Vec<f64>,
    /// Overall survival S(t-) just before each event.
    pub overall_survival_pre: Vec<f64>,
    /// At-risk counts per event time.
    pub at_risk: Vec<f64>,
}

/// Compute CIF for cause `target_cause`.
///
/// `causes[i] = 0` indicates censoring; `> 0` indicates an event of that cause.
pub fn cumulative_incidence(
    data: &Dataset,
    causes: &[u32],
    target_cause: u32,
) -> SurvivalResult<CifEstimate> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if causes.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![causes.len()],
        });
    }
    if target_cause == 0 {
        return Err(SurvivalError::InvalidParameter(
            "target_cause must be > 0".to_string(),
        ));
    }
    let mut order = data.order_by_time();
    order.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = data.len() as f64;
    let mut s_prev = 1.0_f64;
    let mut cif_cur = 0.0_f64;
    let mut times_out = Vec::new();
    let mut cif_out = Vec::new();
    let mut s_pre = Vec::new();
    let mut nrisk_out = Vec::new();
    let mut consumed = 0usize;
    let mut i = 0usize;
    while i < order.len() {
        let t = data.observations[order[i]].time;
        let mut j = i;
        let mut d_total = 0.0_f64;
        let mut d_k = 0.0_f64;
        while j < order.len() && data.observations[order[j]].time == t {
            // event vs censoring decided by causes[idx]; data.observations.event flag must agree
            let row = order[j];
            let c = causes[row];
            if data.observations[row].event && c > 0 {
                d_total += 1.0;
                if c == target_cause {
                    d_k += 1.0;
                }
            }
            j += 1;
        }
        let n_at = n - consumed as f64;
        if n_at <= 0.0 {
            break;
        }
        let s_pre_t = s_prev;
        if d_total > 0.0 {
            cif_cur += s_pre_t * d_k / n_at;
            s_prev *= 1.0 - d_total / n_at;
        }
        times_out.push(t);
        cif_out.push(cif_cur);
        s_pre.push(s_pre_t);
        nrisk_out.push(n_at);
        consumed += j - i;
        i = j;
    }
    Ok(CifEstimate {
        times: times_out,
        cif: cif_out,
        overall_survival_pre: s_pre,
        at_risk: nrisk_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cif_basic_single_cause() {
        // 4 subjects all dying of cause 1 at times 1..4
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![true; 4];
        let causes = vec![1u32, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let cif = cumulative_incidence(&d, &causes, 1).expect("ok");
        // CIF should equal 1 - KM survival = 1 - 0 = 1 by end
        assert!((cif.cif.last().expect("ok") - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn cif_competing_event_reduces_target_cif() {
        // 2 subjects: cause 1 at t=1, cause 2 at t=2
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![true, true, true];
        let causes = vec![1u32, 2, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let cif1 = cumulative_incidence(&d, &causes, 1).expect("ok");
        let cif2 = cumulative_incidence(&d, &causes, 2).expect("ok");
        // sum of CIF1(last) + CIF2(last) should be <= 1
        let total = cif1.cif.last().expect("ok") + cif2.cif.last().expect("ok");
        assert!(total <= 1.0 + 1.0e-9);
        assert!(total > 0.9);
    }

    #[test]
    fn cif_rejects_bad_target() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(cumulative_incidence(&d, &[1], 0).is_err());
    }
}
