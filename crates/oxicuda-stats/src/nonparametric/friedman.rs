//! Friedman test for repeated-measures (within-block) ranks.

use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};
use crate::nonparametric::rank_with_ties;

/// Result of a Friedman test.
#[derive(Debug, Clone, Copy)]
pub struct FriedmanResult {
    pub q_statistic: f64,
    pub df: f64,
    pub p_value: f64,
}

/// Friedman chi-squared test. `data[i][j]` = block i, treatment j.
pub fn friedman(data: &[Vec<f64>]) -> StatsResult<FriedmanResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let k = data[0].len();
    if k < 2 {
        return Err(StatsError::InsufficientSampleSize { got: k, need: 2 });
    }
    let mut rank_sums = vec![0.0; k];
    for row in data {
        if row.len() != k {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n, k],
                got: vec![n, row.len()],
            });
        }
        let ranks = rank_with_ties(row);
        for (j, &r) in ranks.iter().enumerate() {
            rank_sums[j] += r;
        }
    }
    let n_f = n as f64;
    let k_f = k as f64;
    let sum_sq: f64 = rank_sums.iter().map(|r| r * r).sum();
    let q = 12.0 / (n_f * k_f * (k_f + 1.0)) * sum_sq - 3.0 * n_f * (k_f + 1.0);
    let df = (k - 1) as f64;
    let p = 1.0 - ChiSquared::new(df)?.cdf(q.max(0.0))?;
    Ok(FriedmanResult {
        q_statistic: q,
        df,
        p_value: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friedman_distinct_treatments() {
        // 5 blocks, 3 treatments; treatment 3 always best, treatment 1 always worst
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
            vec![1.2, 2.2, 3.2],
            vec![0.8, 1.8, 2.8],
        ];
        let r = friedman(&data).expect("ok");
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn friedman_no_effect() {
        let data = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
        ];
        let r = friedman(&data).expect("ok");
        assert!(r.p_value > 0.9);
    }
}
