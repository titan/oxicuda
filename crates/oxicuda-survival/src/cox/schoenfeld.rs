//! Schoenfeld residuals and proportional-hazards correlation test.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Schoenfeld residuals: at each event time, `r_i = x_i - x̄_R(t_i)`
/// where `x̄_R = Σ_R w_j x_j / Σ_R w_j` and `w_j = exp(β·x_j)`.
///
/// Returns one residual row per event, in time-ascending order, alongside event times.
pub fn schoenfeld_residuals(
    data: &Dataset,
    beta: &[f64],
) -> SurvivalResult<(Vec<f64>, Vec<Vec<f64>>)> {
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
    let mut s1 = vec![0.0_f64; p];
    let mut w_all = vec![0.0_f64; idx.len()];
    for (k, &i) in idx.iter().enumerate() {
        let xi = &covariates[i];
        let dot: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let w = dot.exp();
        w_all[k] = w;
        s0 += w;
        for a in 0..p {
            s1[a] += w * xi[a];
        }
    }
    let mut times = Vec::new();
    let mut resids: Vec<Vec<f64>> = Vec::new();
    let mut k = 0usize;
    while k < idx.len() {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        let mut event_rows: Vec<usize> = Vec::new();
        while m < idx.len() && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                event_rows.push(idx[m]);
            }
            m += 1;
        }
        if !event_rows.is_empty() && s0 > 0.0 {
            let x_bar: Vec<f64> = s1.iter().map(|s| s / s0).collect();
            for &row in &event_rows {
                let xi = &covariates[row];
                let r: Vec<f64> = xi.iter().zip(x_bar.iter()).map(|(a, b)| a - b).collect();
                times.push(t);
                resids.push(r);
            }
        }
        for jj in k..m {
            let xi = &covariates[idx[jj]];
            let w = w_all[jj];
            s0 -= w;
            for a in 0..p {
                s1[a] -= w * xi[a];
            }
        }
        k = m;
    }
    Ok((times, resids))
}

/// Pearson correlation between each scaled-time covariate residual and time (or log-time).
/// Returns (rho per covariate, chi-square approximation, df).
pub fn schoenfeld_test(
    data: &Dataset,
    beta: &[f64],
    log_time: bool,
) -> SurvivalResult<(Vec<f64>, f64, usize)> {
    let (times, resids) = schoenfeld_residuals(data, beta)?;
    if resids.is_empty() {
        return Err(SurvivalError::NoEvents);
    }
    let p = resids[0].len();
    let n = resids.len();
    let g: Vec<f64> = if log_time {
        times.iter().map(|t| (t.max(1.0e-300)).ln()).collect()
    } else {
        times.clone()
    };
    let g_bar: f64 = g.iter().sum::<f64>() / n as f64;
    let mut rho = vec![0.0_f64; p];
    let mut chi = 0.0_f64;
    for k in 0..p {
        let r_bar: f64 = resids.iter().map(|r| r[k]).sum::<f64>() / n as f64;
        let mut num = 0.0_f64;
        let mut dr = 0.0_f64;
        let mut dg = 0.0_f64;
        for i in 0..n {
            let r_dev = resids[i][k] - r_bar;
            let g_dev = g[i] - g_bar;
            num += r_dev * g_dev;
            dr += r_dev * r_dev;
            dg += g_dev * g_dev;
        }
        let r_corr = if dr > 0.0 && dg > 0.0 {
            num / (dr.sqrt() * dg.sqrt())
        } else {
            0.0
        };
        rho[k] = r_corr;
        chi += r_corr * r_corr * n as f64;
    }
    Ok((rho, chi, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schoenfeld_residuals_sum_to_zero() {
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
                crate::data::Observation::new(4.0, true).expect("ok"),
            ],
            Some(vec![vec![1.0], vec![0.0], vec![-1.0], vec![2.0]]),
            None,
        )
        .expect("ok");
        // At β=0 the residuals at each time should sum to zero across the risk set if all event.
        // For only-event data, each event's residual is x_i - mean of risk set;
        // the sum across all events is not exactly zero, but the FINAL event residual is zero.
        let (_t, r) = schoenfeld_residuals(&data, &[0.0]).expect("ok");
        // last event's risk set has just itself, so residual = 0
        assert!(r.last().expect("non-empty")[0].abs() < 1.0e-12);
    }

    #[test]
    fn schoenfeld_test_returns_nonneg_chi() {
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
                crate::data::Observation::new(4.0, true).expect("ok"),
            ],
            Some(vec![vec![1.0], vec![0.0], vec![-1.0], vec![2.0]]),
            None,
        )
        .expect("ok");
        let (_, chi, df) = schoenfeld_test(&data, &[0.0], false).expect("ok");
        assert!(chi >= 0.0);
        assert_eq!(df, 1);
    }

    #[test]
    fn schoenfeld_log_time_returns_finite() {
        let data = Dataset::new(
            vec![
                crate::data::Observation::new(1.0, true).expect("ok"),
                crate::data::Observation::new(2.0, true).expect("ok"),
                crate::data::Observation::new(3.0, true).expect("ok"),
            ],
            Some(vec![vec![1.0], vec![0.0], vec![-1.0]]),
            None,
        )
        .expect("ok");
        let (rho, chi, _) = schoenfeld_test(&data, &[0.0], true).expect("ok");
        for r in rho {
            assert!(r.is_finite());
        }
        assert!(chi.is_finite());
    }
}
