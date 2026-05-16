//! Kruskal-Wallis H test.

use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};
use crate::nonparametric::{rank_with_ties, tie_correction_sum};

/// Result of a Kruskal-Wallis test.
#[derive(Debug, Clone, Copy)]
pub struct KruskalWallisResult {
    pub h_statistic: f64,
    pub df: f64,
    pub p_value: f64,
}

/// Kruskal-Wallis H test for `k >= 2` independent groups.
pub fn kruskal_wallis(groups: &[&[f64]]) -> StatsResult<KruskalWallisResult> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: groups.len(),
            need: 2,
        });
    }
    let mut combined = Vec::new();
    let mut sizes = Vec::with_capacity(groups.len());
    for g in groups {
        if g.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        sizes.push(g.len());
        combined.extend_from_slice(g);
    }
    let n = combined.len() as f64;
    if n < 2.0 {
        return Err(StatsError::InsufficientSampleSize {
            got: combined.len(),
            need: 2,
        });
    }
    let ranks = rank_with_ties(&combined);
    let mut start = 0usize;
    let mut h = 0.0;
    for (k, &sz) in sizes.iter().enumerate() {
        let rk: f64 = ranks[start..start + sz].iter().sum();
        h += rk * rk / sz as f64;
        start += sz;
        let _ = k;
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    // Tie correction
    let t_corr = tie_correction_sum(&ranks);
    let denom = 1.0 - t_corr / (n * n * n - n);
    if denom.abs() > 1e-12 {
        h /= denom;
    }
    let df = (groups.len() - 1) as f64;
    let p = 1.0 - ChiSquared::new(df)?.cdf(h.max(0.0))?;
    Ok(KruskalWallisResult {
        h_statistic: h,
        df,
        p_value: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kw_three_distinct_groups() {
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[4.0, 5.0, 6.0];
        let g3: &[f64] = &[7.0, 8.0, 9.0];
        let r = kruskal_wallis(&[g1, g2, g3]).expect("ok");
        assert!(r.p_value < 0.1);
    }

    #[test]
    fn kw_identical_groups() {
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[1.0, 2.0, 3.0];
        let r = kruskal_wallis(&[g1, g2]).expect("ok");
        assert!(r.p_value > 0.5);
    }
}
