//! One-way and two-way ANOVA.

use crate::distributions::f_dist::FDist;
use crate::error::{StatsError, StatsResult};

/// Result of an ANOVA F-test.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    pub ss_between: f64,
    pub ss_within: f64,
    pub df_between: f64,
    pub df_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
    pub f_statistic: f64,
    pub p_value: f64,
}

/// One-way ANOVA.
///
/// Each `group` is a slice of observations.
pub fn one_way_anova(groups: &[&[f64]]) -> StatsResult<AnovaResult> {
    if groups.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: groups.len(),
            need: 2,
        });
    }
    let k = groups.len();
    let mut n_total = 0usize;
    let mut grand_sum = 0.0f64;
    for g in groups {
        if g.len() < 2 {
            return Err(StatsError::InsufficientSampleSize {
                got: g.len(),
                need: 2,
            });
        }
        n_total += g.len();
        grand_sum += g.iter().sum::<f64>();
    }
    let grand_mean = grand_sum / n_total as f64;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for g in groups {
        let n_i = g.len() as f64;
        let mean_i = g.iter().sum::<f64>() / n_i;
        ss_between += n_i * (mean_i - grand_mean).powi(2);
        for &v in *g {
            ss_within += (v - mean_i).powi(2);
        }
    }
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within.max(1.0);
    if ms_within <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "within-group variance is zero".into(),
        ));
    }
    let f = ms_between / ms_within;
    let p = 1.0 - FDist::new(df_between, df_within)?.cdf(f)?;
    Ok(AnovaResult {
        ss_between,
        ss_within,
        df_between,
        df_within,
        ms_between,
        ms_within,
        f_statistic: f,
        p_value: p,
    })
}

/// Two-way ANOVA result with row, column, and interaction effects.
#[derive(Debug, Clone)]
pub struct TwoWayResult {
    pub ss_rows: f64,
    pub ss_cols: f64,
    pub ss_inter: f64,
    pub ss_error: f64,
    pub df_rows: f64,
    pub df_cols: f64,
    pub df_inter: f64,
    pub df_error: f64,
    pub f_rows: f64,
    pub f_cols: f64,
    pub f_inter: f64,
    pub p_rows: f64,
    pub p_cols: f64,
    pub p_inter: f64,
}

/// Balanced two-way ANOVA. `cells[i][j]` is a vector of `n` replicates for row i, column j.
pub fn two_way_anova(cells: &[Vec<Vec<f64>>]) -> StatsResult<TwoWayResult> {
    let r = cells.len();
    if r < 2 {
        return Err(StatsError::InsufficientSampleSize { got: r, need: 2 });
    }
    let c = cells[0].len();
    if c < 2 {
        return Err(StatsError::InsufficientSampleSize { got: c, need: 2 });
    }
    let n = cells[0][0].len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    // Check balance
    for row in cells {
        if row.len() != c {
            return Err(StatsError::ShapeMismatch {
                expected: vec![r, c],
                got: vec![r, row.len()],
            });
        }
        for col in row {
            if col.len() != n {
                return Err(StatsError::InsufficientSampleSize {
                    got: col.len(),
                    need: n,
                });
            }
        }
    }
    let total = (r * c * n) as f64;
    let mut grand_sum = 0.0;
    for row in cells {
        for col in row {
            grand_sum += col.iter().sum::<f64>();
        }
    }
    let grand_mean = grand_sum / total;
    // Row, col means
    let mut row_means = vec![0.0; r];
    let mut col_means = vec![0.0; c];
    let mut cell_means = vec![vec![0.0; c]; r];
    for (i, row) in cells.iter().enumerate() {
        let mut sum_i = 0.0;
        for (j, col) in row.iter().enumerate() {
            let m = col.iter().sum::<f64>() / n as f64;
            cell_means[i][j] = m;
            sum_i += m * n as f64;
        }
        row_means[i] = sum_i / (c * n) as f64;
    }
    for j in 0..c {
        let mut s = 0.0;
        for (i, row) in cells.iter().enumerate() {
            s += cell_means[i][j] * n as f64;
            let _ = row;
        }
        col_means[j] = s / (r * n) as f64;
    }
    let mut ss_rows = 0.0;
    for &m in &row_means {
        ss_rows += (c * n) as f64 * (m - grand_mean).powi(2);
    }
    let mut ss_cols = 0.0;
    for &m in &col_means {
        ss_cols += (r * n) as f64 * (m - grand_mean).powi(2);
    }
    let mut ss_inter = 0.0;
    let mut ss_error = 0.0;
    for (i, row) in cells.iter().enumerate() {
        for (j, col) in row.iter().enumerate() {
            let cm = cell_means[i][j];
            ss_inter += n as f64 * (cm - row_means[i] - col_means[j] + grand_mean).powi(2);
            for &v in col {
                ss_error += (v - cm).powi(2);
            }
        }
    }
    let df_rows = (r - 1) as f64;
    let df_cols = (c - 1) as f64;
    let df_inter = (r - 1) as f64 * (c - 1) as f64;
    let df_error = (r * c * (n - 1)) as f64;
    let ms_error = ss_error / df_error.max(1.0);
    if ms_error <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "zero within-cell variance".into(),
        ));
    }
    let f_rows = (ss_rows / df_rows) / ms_error;
    let f_cols = (ss_cols / df_cols) / ms_error;
    let f_inter = (ss_inter / df_inter) / ms_error;
    let p_rows = 1.0 - FDist::new(df_rows, df_error)?.cdf(f_rows)?;
    let p_cols = 1.0 - FDist::new(df_cols, df_error)?.cdf(f_cols)?;
    let p_inter = 1.0 - FDist::new(df_inter, df_error)?.cdf(f_inter)?;
    Ok(TwoWayResult {
        ss_rows,
        ss_cols,
        ss_inter,
        ss_error,
        df_rows,
        df_cols,
        df_inter,
        df_error,
        f_rows,
        f_cols,
        f_inter,
        p_rows,
        p_cols,
        p_inter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_way_known_case() {
        // scipy f_oneway: groups [1,2,3], [3,4,5], [5,6,7]
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[3.0, 4.0, 5.0];
        let g3: &[f64] = &[5.0, 6.0, 7.0];
        let r = one_way_anova(&[g1, g2, g3]).expect("ok");
        // scipy gives F = 12.0 with df=(2, 6), p ~ 0.008
        assert!((r.f_statistic - 12.0).abs() < 1e-9);
        assert!(r.p_value < 0.01);
    }

    #[test]
    fn one_way_zero_when_identical() {
        // Means equal but variance > 0 -> F small, p close to 1
        let g1: &[f64] = &[1.0, 2.0, 3.0];
        let g2: &[f64] = &[1.0, 2.0, 3.0];
        let r = one_way_anova(&[g1, g2]).expect("ok");
        assert!(r.f_statistic.abs() < 1e-12);
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn two_way_finite() {
        let cells = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![2.0, 3.0], vec![4.0, 5.0]],
        ];
        let r = two_way_anova(&cells).expect("ok");
        assert!(r.f_rows.is_finite());
        assert!(r.f_cols.is_finite());
    }
}
