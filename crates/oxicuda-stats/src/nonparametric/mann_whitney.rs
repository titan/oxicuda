//! Mann-Whitney U test (rank-sum test).

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::nonparametric::{rank_with_ties, tie_correction_sum};

/// Result of a Mann-Whitney U test.
#[derive(Debug, Clone, Copy)]
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z: f64,
    pub p_value_two_sided: f64,
}

/// Mann-Whitney U via rank sum with continuity correction and tie adjustment.
pub fn mann_whitney_u(x1: &[f64], x2: &[f64]) -> StatsResult<MannWhitneyResult> {
    if x1.is_empty() || x2.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let mut combined = Vec::with_capacity(x1.len() + x2.len());
    combined.extend_from_slice(x1);
    combined.extend_from_slice(x2);
    let ranks = rank_with_ties(&combined);
    let r1: f64 = ranks.iter().take(x1.len()).sum();
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let u2 = n1 * n2 - u1;
    let u = u1.min(u2);
    let mu = n1 * n2 / 2.0;
    let n = n1 + n2;
    let t_corr = tie_correction_sum(&ranks);
    let sigma2 = (n1 * n2 / 12.0) * ((n + 1.0) - t_corr / (n * (n - 1.0)));
    if sigma2 <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "Mann-Whitney: degenerate ranks".into(),
        ));
    }
    let sigma = sigma2.sqrt();
    // Continuity-corrected z
    let z = (u - mu + 0.5 * (u - mu).signum()) / sigma;
    let dist = Normal::standard();
    let p_two = 2.0 * (1.0 - dist.cdf(z.abs()));
    Ok(MannWhitneyResult {
        u_statistic: u,
        z,
        p_value_two_sided: p_two.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mann_whitney_distinct_groups() {
        let x1 = [1.0, 2.0, 3.0];
        let x2 = [4.0, 5.0, 6.0];
        let r = mann_whitney_u(&x1, &x2).expect("ok");
        // Clear separation; p should be small
        assert!(r.p_value_two_sided < 0.2);
    }

    #[test]
    fn mann_whitney_identical_groups() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = mann_whitney_u(&x1, &x2).expect("ok");
        // No effect; p large
        assert!(r.p_value_two_sided > 0.5);
    }
}
