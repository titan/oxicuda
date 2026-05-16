//! Anderson-Darling goodness-of-fit test.

use crate::error::{StatsError, StatsResult};

/// Result of Anderson-Darling test (A^2 statistic + asymptotic p).
#[derive(Debug, Clone, Copy)]
pub struct AndersonDarlingResult {
    pub a_squared: f64,
    pub p_value_approx: f64,
}

/// Anderson-Darling A^2 statistic against a continuous CDF.
///
/// Uses Stephens' modification A*^2 = A^2 (1 + 0.75/n + 2.25/n^2) for normality;
/// here we use the unmodified A^2 followed by asymptotic p-approximation.
pub fn anderson_darling(x: &[f64], cdf: impl Fn(f64) -> f64) -> StatsResult<AndersonDarlingResult> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let n_f = n as f64;
    let mut sum = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        let i_f = (i + 1) as f64;
        let f = cdf(v).clamp(1e-300, 1.0 - 1e-15);
        let f_complement = cdf(sorted[n - i - 1]).clamp(1e-300, 1.0 - 1e-15);
        sum += (2.0 * i_f - 1.0) * (f.ln() + (1.0 - f_complement).ln());
    }
    let a2 = -n_f - sum / n_f;
    // Approximate p-value from D'Agostino-Stephens 1986 table (rough)
    let p = if a2 < 0.2 {
        1.0 - (-13.436 + 101.14 * a2 - 223.73 * a2 * a2).exp()
    } else if a2 < 0.34 {
        1.0 - (-8.318 + 42.796 * a2 - 59.938 * a2 * a2).exp()
    } else if a2 < 0.6 {
        (0.9177 - 4.279 * a2 - 1.38 * a2 * a2).exp()
    } else {
        (1.2937 - 5.709 * a2 + 0.0186 * a2 * a2).exp()
    };
    Ok(AndersonDarlingResult {
        a_squared: a2,
        p_value_approx: p.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::normal::Normal;

    #[test]
    fn anderson_darling_finite() {
        let x = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let n = Normal::standard();
        let r = anderson_darling(&x, |t| n.cdf(t)).expect("ok");
        assert!(r.a_squared.is_finite());
        assert!((0.0..=1.0).contains(&r.p_value_approx));
    }

    #[test]
    fn anderson_darling_clear_misfit() {
        let x = [10.0, 11.0, 12.0, 13.0, 14.0];
        let n = Normal::standard();
        let r = anderson_darling(&x, |t| n.cdf(t)).expect("ok");
        // Very far in the tail -> large A^2
        assert!(r.a_squared > 1.0);
    }
}
