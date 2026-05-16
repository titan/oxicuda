//! Kolmogorov-Smirnov goodness-of-fit tests (one-sample and two-sample).

use crate::error::{StatsError, StatsResult};

/// Result of a KS test.
#[derive(Debug, Clone, Copy)]
pub struct KsResult {
    pub d_statistic: f64,
    pub p_value: f64,
}

/// Asymptotic Kolmogorov distribution: `Q_KS(z) = 2 * sum_{j>=1} (-1)^(j-1) * exp(-2 j^2 z^2)`.
fn ks_p_value(d: f64, n: f64) -> f64 {
    if d <= 1e-15 {
        return 1.0;
    }
    // Stephens correction
    let z = d * (n.sqrt() + 0.12 + 0.11 / n.sqrt());
    let z2 = z * z;
    let mut sum = 0.0;
    let mut prev_sign = 1.0;
    for j in 1..=100 {
        let term = (-2.0 * (j as f64).powi(2) * z2).exp();
        sum += prev_sign * term;
        prev_sign = -prev_sign;
        if term < 1e-16 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// One-sample KS: compare empirical CDF of `x` with `cdf(t)`.
pub fn ks_one_sample(x: &[f64], cdf: impl Fn(f64) -> f64) -> StatsResult<KsResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut d_max: f64 = 0.0;
    for (i, &val) in sorted.iter().enumerate() {
        let f_n_upper = (i + 1) as f64 / n as f64;
        let f_n_lower = i as f64 / n as f64;
        let f_t = cdf(val);
        let d1 = (f_n_upper - f_t).abs();
        let d2 = (f_n_lower - f_t).abs();
        let d = d1.max(d2);
        if d > d_max {
            d_max = d;
        }
    }
    let p = ks_p_value(d_max, n as f64);
    Ok(KsResult {
        d_statistic: d_max,
        p_value: p,
    })
}

/// Two-sample KS: compare empirical CDFs of `x1` and `x2`.
pub fn ks_two_sample(x1: &[f64], x2: &[f64]) -> StatsResult<KsResult> {
    if x1.is_empty() || x2.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut s1: Vec<f64> = x1.to_vec();
    let mut s2: Vec<f64> = x2.to_vec();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n1 = s1.len() as f64;
    let n2 = s2.len() as f64;
    // Build the union of unique sample points and compute CDF differences at each.
    let mut union: Vec<f64> = Vec::with_capacity(s1.len() + s2.len());
    union.extend_from_slice(&s1);
    union.extend_from_slice(&s2);
    union.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    union.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    let mut d_max: f64 = 0.0;
    for &t in &union {
        // F_n(t) = (number of samples <= t) / n
        let c1 = s1.iter().filter(|&&v| v <= t).count() as f64 / n1;
        let c2 = s2.iter().filter(|&&v| v <= t).count() as f64 / n2;
        let d = (c1 - c2).abs();
        if d > d_max {
            d_max = d;
        }
    }
    let n_eff = (n1 * n2 / (n1 + n2)).max(1.0);
    let p = ks_p_value(d_max, n_eff);
    Ok(KsResult {
        d_statistic: d_max,
        p_value: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::normal::Normal;

    #[test]
    fn ks_one_sample_uniform_vs_normal() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64 + 0.5) / 20.0).collect();
        let n = Normal::standard();
        let r = ks_one_sample(&x, |t| n.cdf(t)).expect("ok");
        // Big disagreement; small p-value
        assert!(r.d_statistic > 0.3);
    }

    #[test]
    fn ks_two_sample_same_distribution() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let r = ks_two_sample(&x, &y).expect("ok");
        assert!(r.d_statistic.abs() < 1e-9);
        assert!(r.p_value > 0.5);
    }

    #[test]
    fn ks_two_sample_shifted() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = (21..=40).map(|i| i as f64).collect();
        let r = ks_two_sample(&x, &y).expect("ok");
        assert!(r.d_statistic > 0.9);
    }
}
