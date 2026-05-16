//! Cox partial likelihood (Efron tie handling).
//!
//! With `d` tied events at time `t`, Efron uses:
//! ```text
//!   L_i = exp(Σ events x) / Π_{k=0}^{d-1} [S0 - (k/d) * S0_tied]
//! ```
//! where `S0_tied = Σ_{tied events} exp(β·x)`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute Efron partial log-likelihood + gradient + Fisher information.
pub fn efron_log_likelihood(
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
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = idx.len();
    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];
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
    let mut k = 0usize;
    while k < n {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        let mut d_count = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];
        let mut etabsum = 0.0_f64;
        // sums over tied events
        let mut s0_e = 0.0_f64;
        let mut s1_e = vec![0.0_f64; p];
        let mut s2_e = vec![0.0_f64; p * p];
        while m < n && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                d_count += 1.0;
                let xi = &covariates[idx[m]];
                let dot: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
                etabsum += dot;
                for a in 0..p {
                    x_events[a] += xi[a];
                }
                let w = w_all[m];
                s0_e += w;
                for a in 0..p {
                    s1_e[a] += w * xi[a];
                    for b in 0..p {
                        s2_e[a * p + b] += w * xi[a] * xi[b];
                    }
                }
            }
            m += 1;
        }
        if d_count > 0.0 {
            loglik += etabsum;
            let d = d_count;
            // For each k=0..d-1: adjusted denominator = S0 - (k/d) S0_e
            let mut frac_score = vec![0.0_f64; p];
            let mut frac_info = vec![0.0_f64; p * p];
            for kk in 0..(d as usize) {
                let alpha = kk as f64 / d;
                let denom = s0 - alpha * s0_e;
                if denom <= 0.0 {
                    return Err(SurvivalError::NumericalInstability(
                        "non-positive Efron denominator".to_string(),
                    ));
                }
                loglik -= denom.ln();
                // numerator s1 - alpha * s1_e ; covariance: (s2 - alpha s2_e)/denom - mu mu^T
                let mut mu = vec![0.0_f64; p];
                for a in 0..p {
                    mu[a] = (s1[a] - alpha * s1_e[a]) / denom;
                }
                for a in 0..p {
                    frac_score[a] += mu[a];
                }
                for a in 0..p {
                    for b in 0..p {
                        let s2_eff = (s2[a * p + b] - alpha * s2_e[a * p + b]) / denom;
                        frac_info[a * p + b] += s2_eff - mu[a] * mu[b];
                    }
                }
            }
            for a in 0..p {
                score[a] += x_events[a] - frac_score[a];
            }
            for a in 0..p {
                for b in 0..p {
                    info[a * p + b] += frac_info[a * p + b];
                }
            }
        }
        // remove from risk set
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
    fn efron_reduces_to_breslow_with_unique_times() {
        // when no ties, Efron == Breslow
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![0.5], vec![-0.5], vec![1.0]]),
            None,
        )
        .expect("ok");
        let (le, _, _) = efron_log_likelihood(&data, &[0.3]).expect("ok");
        let (lb, _, _) =
            crate::cox::breslow_ties::breslow_log_likelihood(&data, &[0.3]).expect("ok");
        assert!((le - lb).abs() < 1.0e-12);
    }

    #[test]
    fn efron_handles_ties() {
        // two tied events at t=1
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![0.5], vec![-0.5], vec![1.0]]),
            None,
        )
        .expect("ok");
        let (ll, _, _) = efron_log_likelihood(&data, &[0.1]).expect("ok");
        assert!(ll.is_finite());
    }

    #[test]
    fn efron_score_zero_when_identical_covariates() {
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
        let (_, sc, _) = efron_log_likelihood(&data, &[0.0]).expect("ok");
        assert!(sc[0].abs() < 1.0e-10);
    }
}
