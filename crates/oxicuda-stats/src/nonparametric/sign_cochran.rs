//! Sign test and Cochran's Q test.
//!
//! - **Sign test** — a distribution-free test that the median of the paired
//!   differences `d_i = x_i − y_i` is zero. It counts how many differences are
//!   positive vs. negative (ties are discarded) and, under the null, the count
//!   of positives follows `Binomial(m, ½)` where `m` is the number of non-zero
//!   differences. The exact two-sided p-value sums the binomial point masses in
//!   both tails; a normal approximation with continuity correction is also
//!   returned for large `m`.
//!
//! - **Cochran's Q test** — extends McNemar's test to `k ≥ 3` related binary
//!   treatments measured on the same `n` subjects (a `n × k` matrix of 0/1
//!   responses). It tests whether the proportion of "successes" is the same
//!   across treatments. The statistic
//!   `Q = (k−1)(k·Σ C_j² − T²) / (k·T − Σ R_i²)`
//!   is chi-squared-distributed with `k−1` degrees of freedom, where `C_j` is
//!   the column total, `R_i` the row total and `T` the grand total.
//!
//! ## References
//! - Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.).
//! - Cochran, W. G. (1950). "The comparison of percentages in matched samples."
//!   Biometrika 37.

use crate::distributions::binomial::Binomial;
use crate::distributions::chi_squared::ChiSquared;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Result of a paired sign test.
#[derive(Debug, Clone, Copy)]
pub struct SignTestResult {
    /// Number of positive differences (`x_i > y_i`).
    pub n_positive: usize,
    /// Number of negative differences (`x_i < y_i`).
    pub n_negative: usize,
    /// Number of ties (`x_i == y_i`), discarded from the test.
    pub n_ties: usize,
    /// Exact two-sided p-value from the `Binomial(m, ½)` null.
    pub p_value_exact: f64,
    /// Two-sided p-value from the normal approximation (continuity-corrected).
    pub p_value_normal: f64,
}

/// Result of Cochran's Q test.
#[derive(Debug, Clone, Copy)]
pub struct CochranQResult {
    /// The `Q` statistic.
    pub q_statistic: f64,
    /// Degrees of freedom (`k − 1`).
    pub df: f64,
    /// Upper-tail p-value from the chi-squared distribution.
    pub p_value: f64,
}

// ─── Sign test ────────────────────────────────────────────────────────────────

/// Paired sign test for the null hypothesis that the median difference
/// `median(x − y) = 0`.
///
/// `x` and `y` must be the same length. Tied pairs (`|x_i − y_i| ≤ tie_eps`)
/// are excluded. Requires at least one non-tied pair.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if `x.len() != y.len()`.
/// - [`StatsError::EmptyInput`] if the inputs are empty.
/// - [`StatsError::InsufficientSampleSize`] if every pair is a tie.
/// - [`StatsError::NonFiniteValue`] on non-finite data.
pub fn sign_test(x: &[f64], y: &[f64], tie_eps: f64) -> StatsResult<SignTestResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut n_tie = 0usize;
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
        let d = xi - yi;
        if d.abs() <= tie_eps {
            n_tie += 1;
        } else if d > 0.0 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    let m = n_pos + n_neg;
    if m == 0 {
        return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
    }

    // Exact two-sided p-value: 2 · min(P(K ≤ k_min), P(K ≥ k_min)) capped at 1,
    // computed by summing Binomial(m, ½) point masses around the smaller tail.
    let dist = Binomial::new(m, 0.5)?;
    let k_small = n_pos.min(n_neg);
    let mut tail = 0.0;
    for k in 0..=k_small {
        tail += dist.pmf(k);
    }
    let p_exact = (2.0 * tail).min(1.0);

    // Normal approximation with continuity correction.
    let mean = m as f64 * 0.5;
    let sd = (m as f64 * 0.25).sqrt();
    let p_normal = if sd > 0.0 {
        let z = ((k_small as f64 + 0.5) - mean) / sd; // k_small < mean ⇒ z < 0
        let std = Normal::standard();
        (2.0 * std.cdf(z)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Ok(SignTestResult {
        n_positive: n_pos,
        n_negative: n_neg,
        n_ties: n_tie,
        p_value_exact: p_exact,
        p_value_normal: p_normal,
    })
}

// ─── Cochran's Q ──────────────────────────────────────────────────────────────

/// Cochran's Q test for `k ≥ 3` related binary treatments on `n` subjects.
///
/// `data` is a flat row-major `n × k` matrix of 0/1 responses (`data[i*k + j]`
/// is subject `i`'s response to treatment `j`). Any non-zero entry is treated
/// as a success (`1`).
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if `k < 3` or `n == 0`.
/// - [`StatsError::ShapeMismatch`] if `data.len() != n * k`.
/// - [`StatsError::NumericalInstability`] if every subject has a constant row
///   (the denominator vanishes — `Q` is undefined / degenerate).
pub fn cochran_q(data: &[f64], n: usize, k: usize) -> StatsResult<CochranQResult> {
    if k < 3 {
        return Err(StatsError::InvalidParameter {
            name: "k".into(),
            reason: "Cochran's Q requires at least 3 treatments".into(),
        });
    }
    if n == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n".into(),
            reason: "need at least one subject".into(),
        });
    }
    if data.len() != n * k {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, k],
            got: vec![data.len()],
        });
    }

    let mut col_totals = vec![0.0_f64; k];
    let mut row_totals = vec![0.0_f64; n];
    let mut grand_total = 0.0_f64;
    let mut sum_row_sq = 0.0_f64;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..k {
            let bit = if data[i * k + j] != 0.0 { 1.0 } else { 0.0 };
            col_totals[j] += bit;
            row += bit;
        }
        row_totals[i] = row;
        grand_total += row;
        sum_row_sq += row * row;
    }
    let sum_col_sq: f64 = col_totals.iter().map(|&c| c * c).sum();

    let kf = k as f64;
    let numerator = (kf - 1.0) * (kf * sum_col_sq - grand_total * grand_total);
    let denominator = kf * grand_total - sum_row_sq;
    if denominator.abs() < 1e-12 {
        return Err(StatsError::NumericalInstability(
            "cochran_q: degenerate data (all rows constant)".into(),
        ));
    }
    let q = (numerator / denominator).max(0.0);
    let df = (k - 1) as f64;
    let p = 1.0 - ChiSquared::new(df)?.cdf(q)?;
    Ok(CochranQResult {
        q_statistic: q,
        df,
        p_value: p.clamp(0.0, 1.0),
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_all_positive_significant() {
        // Every pair increases → strongly reject the null.
        let x: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = sign_test(&x, &y, 0.0).expect("ok");
        assert_eq!(r.n_positive, 8);
        assert_eq!(r.n_negative, 0);
        // P = 2 · (1/2)^8 = 2/256 ≈ 0.0078.
        assert!(
            (r.p_value_exact - 2.0 / 256.0).abs() < 1e-9,
            "p={}",
            r.p_value_exact
        );
    }

    #[test]
    fn sign_test_balanced_high_p() {
        // Equal positives and negatives → fail to reject.
        let x: Vec<f64> = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let y: Vec<f64> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let r = sign_test(&x, &y, 0.0).expect("ok");
        assert_eq!(r.n_positive, 3);
        assert_eq!(r.n_negative, 3);
        assert!(r.p_value_exact > 0.5, "p={}", r.p_value_exact);
    }

    #[test]
    fn sign_test_counts_ties() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![1.0, 2.0, 1.0, 5.0]; // 2 ties, 1 pos, 1 neg
        let r = sign_test(&x, &y, 0.0).expect("ok");
        assert_eq!(r.n_ties, 2);
        assert_eq!(r.n_positive, 1);
        assert_eq!(r.n_negative, 1);
    }

    #[test]
    fn sign_test_exact_p_known_value() {
        // 1 positive out of 5 non-tied: tail = P(K≤1) = (1+5)/32 = 6/32;
        // two-sided = 12/32 = 0.375.
        let x: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 0.0];
        let y: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let r = sign_test(&x, &y, 0.0).expect("ok");
        assert_eq!(r.n_positive, 1);
        assert_eq!(r.n_negative, 4);
        assert!(
            (r.p_value_exact - 12.0 / 32.0).abs() < 1e-9,
            "p={}",
            r.p_value_exact
        );
    }

    #[test]
    fn sign_test_normal_in_range() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 + 0.6).collect();
        let y: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let r = sign_test(&x, &y, 0.0).expect("ok");
        assert!((0.0..=1.0).contains(&r.p_value_normal));
        assert!((0.0..=1.0).contains(&r.p_value_exact));
    }

    #[test]
    fn sign_test_dimension_mismatch_error() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![1.0];
        assert!(matches!(
            sign_test(&x, &y, 0.0).unwrap_err(),
            StatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn sign_test_empty_error() {
        assert!(matches!(
            sign_test(&[], &[], 0.0).unwrap_err(),
            StatsError::EmptyInput
        ));
    }

    #[test]
    fn sign_test_all_ties_error() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            sign_test(&x, &y, 0.0).unwrap_err(),
            StatsError::InsufficientSampleSize { .. }
        ));
    }

    #[test]
    fn sign_test_non_finite_error() {
        let x: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        let y: Vec<f64> = vec![0.0, 1.0, 2.0];
        assert!(matches!(
            sign_test(&x, &y, 0.0).unwrap_err(),
            StatsError::NonFiniteValue(_)
        ));
    }

    #[test]
    fn sign_test_tie_eps_threshold() {
        // Differences of 0.05 are below tie_eps=0.1 → treated as ties.
        let x: Vec<f64> = vec![1.05, 2.05, 3.5];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let r = sign_test(&x, &y, 0.1).expect("ok");
        assert_eq!(r.n_ties, 2);
        assert_eq!(r.n_positive, 1);
    }

    #[test]
    fn cochran_q_constant_rows_degenerate() {
        // Every subject responds identically across treatments: each row is all
        // 0s or all 1s, so the denominator `k·T − Σ R_i²` vanishes ⇒ degenerate.
        let n = 4;
        let k = 3;
        let data = vec![
            1.0, 1.0, 1.0, // subject 0
            0.0, 0.0, 0.0, // subject 1
            1.0, 1.0, 1.0, // subject 2
            0.0, 0.0, 0.0, // subject 3
        ];
        assert!(matches!(
            cochran_q(&data, n, k),
            Err(StatsError::NumericalInstability(_))
        ));
    }

    #[test]
    fn cochran_q_strong_difference_significant() {
        // Treatment 0 always 1, treatment 2 always 0, treatment 1 mixed.
        let n = 6;
        let k = 3;
        let data = vec![
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            0.0,
        ];
        let r = cochran_q(&data, n, k).expect("ok");
        assert!(r.q_statistic > 0.0);
        assert_eq!(r.df, 2.0);
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    #[test]
    fn cochran_q_no_difference_high_p() {
        // All columns have the same success proportion (balanced design).
        let n = 4;
        let k = 3;
        let data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = cochran_q(&data, n, k).expect("ok");
        // Column totals all equal 2 → numerator 0 → Q = 0 → large p.
        assert!(r.q_statistic.abs() < 1e-9);
        assert!(r.p_value > 0.9, "p={}", r.p_value);
    }

    #[test]
    fn cochran_q_treats_nonzero_as_success() {
        let n = 3;
        let k = 3;
        let data = vec![2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let r = cochran_q(&data, n, k).expect("ok");
        assert!(r.q_statistic.is_finite() && r.q_statistic >= 0.0);
    }

    #[test]
    fn cochran_q_too_few_treatments_error() {
        let data = vec![1.0, 0.0, 1.0, 1.0];
        assert!(matches!(
            cochran_q(&data, 2, 2).unwrap_err(),
            StatsError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn cochran_q_zero_subjects_error() {
        assert!(matches!(
            cochran_q(&[], 0, 3).unwrap_err(),
            StatsError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn cochran_q_shape_mismatch_error() {
        let data = vec![1.0, 0.0, 1.0]; // length 3 ≠ n*k = 6
        assert!(matches!(
            cochran_q(&data, 2, 3).unwrap_err(),
            StatsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn cochran_q_deterministic() {
        let n = 4;
        let k = 3;
        let data = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let a = cochran_q(&data, n, k).expect("ok");
        let b = cochran_q(&data, n, k).expect("ok");
        assert_eq!(a.q_statistic, b.q_statistic);
        assert_eq!(a.p_value, b.p_value);
    }
}
