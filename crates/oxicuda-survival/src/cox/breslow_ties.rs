//! Cox partial likelihood (Breslow tie handling).
//!
//! For events at time `t_i` with `d_i` tied events:
//! ```text
//!   L_i = exp(Σ_{k ∈ events(t_i)} β·x_k) / [Σ_{j ∈ R(t_i)} exp(β·x_j)]^{d_i}
//! ```

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute the Breslow partial log-likelihood plus its gradient (`score`) and Fisher information.
///
/// Returns `(loglik, score, info)` where `score` has length p and `info` is p×p row-major.
pub fn breslow_log_likelihood(
    data: &Dataset,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
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
    // Sort observations by time descending; we'll iterate ascending though.
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];
    // At time t, risk set is everyone with time >= t. We iterate ascending and shrink.
    // Risk set accumulators: S0 = Σ w, S1 = Σ w x, S2 = Σ w x x^T
    let n = idx.len();
    // Start by including everyone, then remove as we pass each time.
    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];
    let mut s2 = vec![0.0_f64; p * p];
    let mut w_all = vec![0.0_f64; n];
    for (k, &i) in idx.iter().enumerate() {
        let xi = &covariates[i];
        let dot: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let w = dot.exp();
        w_all[k] = w;
        s0 += w;
        for a in 0..p {
            s1[a] += w * xi[a];
            for b in 0..p {
                s2[a * p + b] += w * xi[a] * xi[b];
            }
        }
    }
    // Iterate ascending; at each unique time, compute partial likelihood contribution then remove.
    let mut k = 0usize;
    while k < n {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        // count tied events and sum x for events
        let mut d_count = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];
        let mut etabsum = 0.0_f64;
        while m < n && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                d_count += 1.0;
                let xi = &covariates[idx[m]];
                let dot: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
                etabsum += dot;
                for a in 0..p {
                    x_events[a] += xi[a];
                }
            }
            m += 1;
        }
        if d_count > 0.0 {
            if s0 <= 0.0 {
                return Err(SurvivalError::NumericalInstability(
                    "non-positive risk-set sum in Breslow log-likelihood".to_string(),
                ));
            }
            loglik += etabsum - d_count * s0.ln();
            // gradient: U = Σ_events x - Σ_events s1/s0 = x_events - d_count * x_bar
            let x_bar: Vec<f64> = s1.iter().map(|si| si / s0).collect();
            for a in 0..p {
                score[a] += x_events[a] - d_count * x_bar[a];
            }
            // info: I += d_count * (s2/s0 - x_bar x_bar^T)
            for a in 0..p {
                for b in 0..p {
                    let cov = s2[a * p + b] / s0 - x_bar[a] * x_bar[b];
                    info[a * p + b] += d_count * cov;
                }
            }
        }
        // remove all observations at time t from risk set
        for jj in k..m {
            let xi = &covariates[idx[jj]];
            let w = w_all[jj];
            s0 -= w;
            for a in 0..p {
                s1[a] -= w * xi[a];
                for b in 0..p {
                    s2[a * p + b] -= w * xi[a] * xi[b];
                }
            }
        }
        k = m;
    }
    Ok((loglik, score, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breslow_zero_beta_gives_zero_score() {
        // identical covariates -> score should be ~zero at beta=0
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
        let (ll, sc, info) = breslow_log_likelihood(&data, &[0.0]).expect("ok");
        assert!(sc[0].abs() < 1.0e-10);
        // when all x equal, info should be zero
        assert!(info[0].abs() < 1.0e-10);
        assert!(ll.is_finite());
    }

    #[test]
    fn breslow_log_likelihood_finite() {
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, false).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![0.5], vec![-0.5], vec![1.0]]),
            None,
        )
        .expect("ok");
        let (ll, _, _) = breslow_log_likelihood(&data, &[0.1]).expect("ok");
        assert!(ll.is_finite());
    }

    #[test]
    fn breslow_score_increases_with_better_beta() {
        // Subject with x=1 dies earlier => positive β should give higher likelihood
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
                crate::data::Observation::new(5.0, true).expect("ok"),
            ],
            Some(vec![vec![1.0], vec![0.0], vec![-1.0]]),
            None,
        )
        .expect("ok");
        let (ll0, _, _) = breslow_log_likelihood(&data, &[0.0]).expect("ok");
        let (ll1, _, _) = breslow_log_likelihood(&data, &[1.0]).expect("ok");
        assert!(ll1 > ll0);
    }
}
