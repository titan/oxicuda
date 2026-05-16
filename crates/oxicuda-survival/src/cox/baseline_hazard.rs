//! Breslow's baseline cumulative hazard estimator.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Output of baseline hazard estimation.
#[derive(Debug, Clone)]
pub struct BaselineHazard {
    pub times: Vec<f64>,
    pub cumulative_hazard: Vec<f64>,
}

/// Compute Breslow's estimator of the baseline cumulative hazard given fitted `β`.
///
/// `Ĥ₀(t) = Σ_{t_i ≤ t} d_i / Σ_{R(t_i)} exp(β·x_j)`
pub fn breslow_baseline_hazard(data: &Dataset, beta: &[f64]) -> SurvivalResult<BaselineHazard> {
    let p = beta.len();
    let covariates = data
        .covariates
        .as_ref()
        .ok_or_else(|| SurvivalError::InvalidParameter("dataset has no covariates".to_string()))?;
    if covariates.first().map(|r| r.len()) != Some(p) {
        return Err(SurvivalError::DimensionMismatch {
            a: covariates.first().map(|r| r.len()).unwrap_or(0),
            b: p,
        });
    }
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut s0 = 0.0_f64;
    let mut w_all = vec![0.0_f64; idx.len()];
    for (k, &i) in idx.iter().enumerate() {
        let xi = &covariates[i];
        let dot: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let w = dot.exp();
        w_all[k] = w;
        s0 += w;
    }
    let mut times_out = Vec::new();
    let mut h = Vec::new();
    let mut h_cur = 0.0_f64;
    let mut k = 0usize;
    while k < idx.len() {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        let mut d = 0.0_f64;
        while m < idx.len() && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                d += 1.0;
            }
            m += 1;
        }
        if d > 0.0 && s0 > 0.0 {
            h_cur += d / s0;
        }
        times_out.push(t);
        h.push(h_cur);
        for w in w_all.iter().take(m).skip(k) {
            s0 -= *w;
        }
        k = m;
    }
    Ok(BaselineHazard {
        times: times_out,
        cumulative_hazard: h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_monotone_nondecreasing() {
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![0.5], vec![0.0], vec![-0.5]]),
            None,
        )
        .expect("ok");
        let b = breslow_baseline_hazard(&data, &[0.1]).expect("ok");
        for w in b.cumulative_hazard.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn baseline_reduces_to_na_when_beta_zero() {
        // when β=0, baseline H == Σ d_i / n_i (Nelson-Aalen)
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![1.0], vec![1.0], vec![1.0]]),
            None,
        )
        .expect("ok");
        let b = breslow_baseline_hazard(&data, &[0.0]).expect("ok");
        // H(t1) = 1/3
        assert!((b.cumulative_hazard[0] - 1.0 / 3.0).abs() < 1.0e-10);
        // H(t2) = 1/3 + 1/2
        assert!((b.cumulative_hazard[1] - (1.0 / 3.0 + 0.5)).abs() < 1.0e-10);
    }
}
