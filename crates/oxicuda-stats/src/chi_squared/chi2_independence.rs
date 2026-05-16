//! Chi-squared test of independence in an r x c contingency table.

use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};

/// Result of a chi-squared independence test.
#[derive(Debug, Clone)]
pub struct Chi2IndependenceResult {
    pub chi_squared: f64,
    pub df: f64,
    pub p_value: f64,
    pub expected: Vec<f64>,
}

/// Chi-squared independence test on an r x c contingency table (row-major counts).
pub fn chi2_independence(
    observed: &[f64],
    r: usize,
    c: usize,
) -> StatsResult<Chi2IndependenceResult> {
    if r < 2 || c < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: r * c,
            need: 4,
        });
    }
    if observed.len() != r * c {
        return Err(StatsError::ShapeMismatch {
            expected: vec![r, c],
            got: vec![observed.len()],
        });
    }
    let mut row_sums = vec![0.0; r];
    let mut col_sums = vec![0.0; c];
    let mut grand = 0.0;
    for i in 0..r {
        for j in 0..c {
            let v = observed[i * c + j];
            if v < 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: "observed".into(),
                    reason: format!("negative count at ({i},{j})"),
                });
            }
            row_sums[i] += v;
            col_sums[j] += v;
            grand += v;
        }
    }
    if grand <= 0.0 {
        return Err(StatsError::NumericalInstability("total count zero".into()));
    }
    let mut expected = vec![0.0; r * c];
    let mut chi2 = 0.0;
    for i in 0..r {
        for j in 0..c {
            let e = row_sums[i] * col_sums[j] / grand;
            expected[i * c + j] = e;
            if e > 0.0 {
                let o = observed[i * c + j];
                chi2 += (o - e).powi(2) / e;
            }
        }
    }
    let df = ((r - 1) * (c - 1)) as f64;
    let p = 1.0 - ChiSquared::new(df)?.cdf(chi2)?;
    Ok(Chi2IndependenceResult {
        chi_squared: chi2,
        df,
        p_value: p,
        expected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2_2x2_no_association() {
        // Perfect independence: rows scale columns proportionally
        let obs = [50.0, 50.0, 100.0, 100.0];
        let r = chi2_independence(&obs, 2, 2).expect("ok");
        assert!(r.chi_squared.abs() < 1e-9);
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn chi2_2x2_strong_association() {
        let obs = [90.0, 10.0, 10.0, 90.0];
        let r = chi2_independence(&obs, 2, 2).expect("ok");
        assert!(r.chi_squared > 100.0);
        assert!(r.p_value < 0.001);
    }
}
