//! K-sample log-rank test.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;

/// Result of a log-rank test.
#[derive(Debug, Clone)]
pub struct LogRankResult {
    /// Observed minus expected for each group.
    pub observed_minus_expected: Vec<f64>,
    /// Chi-square statistic with `K-1` degrees of freedom.
    pub chi_square: f64,
    /// Degrees of freedom (K-1).
    pub df: usize,
    /// One-sided p-value upper bound for the χ² (computed only for χ² <= 1000).
    pub p_value: f64,
}

/// K-sample log-rank test on a dataset with explicit group labels per observation.
///
/// At each unique event time `t_i`, the K-vector of observed minus expected is
/// `O_k - E_k = d_{ki} - n_{ki} * d_i / n_i`, and the K×K covariance matrix
/// `V_i` follows the multivariate hypergeometric variance.
/// The χ² statistic is `(O-E)^T V^{-1} (O-E)` on the first K-1 entries (rank-deficient by 1).
pub fn log_rank_test(data: &Dataset, groups: &[usize]) -> SurvivalResult<LogRankResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if groups.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![groups.len()],
        });
    }
    let k = groups.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(
            "need at least 2 groups".to_string(),
        ));
    }
    if !groups.iter().any(|&g| g < k) {
        return Err(SurvivalError::InvalidParameter(
            "group indices invalid".to_string(),
        ));
    }
    let n = data.len();
    let mut order = data.order_by_time();
    order.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut oe = vec![0.0_f64; k];
    // V is K×K; we accumulate per-time contributions
    let mut v = vec![0.0_f64; k * k];
    let mut at_risk_per_group: Vec<f64> = vec![0.0_f64; k];
    for &i in &order {
        let g = groups[i];
        if g >= k {
            return Err(SurvivalError::IndexOutOfBounds { index: g, len: k });
        }
        at_risk_per_group[g] += 1.0;
    }
    let mut total_at_risk = n as f64;
    let mut i = 0usize;
    while i < order.len() {
        let t = data.observations[order[i]].time;
        // gather all observations at time t
        let mut j = i;
        let mut d_total = 0.0_f64;
        let mut d_per_group: Vec<f64> = vec![0.0_f64; k];
        while j < order.len() && data.observations[order[j]].time == t {
            let g = groups[order[j]];
            if data.observations[order[j]].event {
                d_total += 1.0;
                d_per_group[g] += 1.0;
            }
            j += 1;
        }
        // n_per_group BEFORE removing failures and censorings (= at_risk_per_group at this moment)
        let n_per_group: Vec<f64> = at_risk_per_group.clone();
        let n_total = total_at_risk;
        if n_total > 1.0 && d_total > 0.0 {
            // Observed - Expected
            for g in 0..k {
                let e = n_per_group[g] * d_total / n_total;
                oe[g] += d_per_group[g] - e;
            }
            // Hypergeometric variance contribution: V_{g,h} =
            //   d_t * (n_t - d_t) / (n_t - 1) * (n_{g} * δ_{gh} / n_t  − n_g n_h / n_t²)
            let common = d_total * (n_total - d_total) / (n_total - 1.0);
            for g in 0..k {
                for h in 0..k {
                    let term = if g == h {
                        n_per_group[g] * (n_total - n_per_group[g]) / (n_total * n_total)
                    } else {
                        -n_per_group[g] * n_per_group[h] / (n_total * n_total)
                    };
                    v[g * k + h] += common * term;
                }
            }
        }
        // Now remove at-risk (events + censorings) at time t for next step
        for jj in i..j {
            let g = groups[order[jj]];
            at_risk_per_group[g] -= 1.0;
            total_at_risk -= 1.0;
        }
        i = j;
    }
    // Reduce to K-1 dimensional space by dropping last row/col
    let chi_square = if k > 1 {
        let km1 = k - 1;
        let mut v_red = vec![0.0_f64; km1 * km1];
        for r in 0..km1 {
            for c in 0..km1 {
                v_red[r * km1 + c] = v[r * k + c];
            }
        }
        let v_inv = match gauss_jordan_inverse(&v_red, km1) {
            Ok(m) => m,
            Err(_) => return Err(SurvivalError::SingularMatrix),
        };
        let mut s = 0.0_f64;
        for r in 0..km1 {
            for c in 0..km1 {
                s += oe[r] * v_inv[r * km1 + c] * oe[c];
            }
        }
        s
    } else {
        0.0
    };
    let p = chi_square_survival(chi_square, k - 1);
    Ok(LogRankResult {
        observed_minus_expected: oe,
        chi_square,
        df: k - 1,
        p_value: p,
    })
}

/// Approximate upper-tail probability `P(χ²_df >= x)` using the Wilson-Hilferty cube-root
/// approximation: `((x/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))` ~ N(0,1).
pub(crate) fn chi_square_survival(x: f64, df: usize) -> f64 {
    if df == 0 || x <= 0.0 {
        return 1.0;
    }
    let dff = df as f64;
    let z = ((x / dff).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt();
    1.0 - phi(z)
}

fn phi(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, ~ 1.5e-7 accuracy
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_rank_identical_groups_zero_chi_square() {
        // two groups with the same observations: O-E should sum near 0, χ² ~ 0
        let times = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let events = vec![true, true, true, true, true, true];
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = log_rank_test(&d, &groups).expect("ok");
        assert!(r.chi_square < 1.0e-9, "chi2={}", r.chi_square);
        assert_eq!(r.df, 1);
    }

    #[test]
    fn log_rank_two_group_strong_diff() {
        // group 0 dies fast, group 1 doesn't
        let times = vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0];
        let events = vec![true, true, true, true, true, true];
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = log_rank_test(&d, &groups).expect("ok");
        assert!(r.chi_square > 2.0);
    }

    #[test]
    fn log_rank_three_group() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let events = vec![true; 9];
        let groups = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = log_rank_test(&d, &groups).expect("ok");
        assert_eq!(r.df, 2);
    }

    #[test]
    fn log_rank_chi_square_invariant_under_relabel() {
        // swap group labels: χ² should be identical
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true; 5];
        let groups_a = vec![0usize, 1, 0, 1, 0];
        let groups_b: Vec<usize> = groups_a.iter().map(|g| 1 - g).collect();
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let ra = log_rank_test(&d, &groups_a).expect("ok");
        let rb = log_rank_test(&d, &groups_b).expect("ok");
        assert!((ra.chi_square - rb.chi_square).abs() < 1.0e-10);
    }

    #[test]
    fn chi_square_survival_extremes() {
        assert!((chi_square_survival(0.0, 1) - 1.0).abs() < 1.0e-10);
        assert!(chi_square_survival(50.0, 1) < 1.0e-6);
    }
}
