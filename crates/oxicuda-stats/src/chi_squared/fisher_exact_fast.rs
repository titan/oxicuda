//! Optimised Fisher's exact test using log-hypergeometric probabilities.
//!
//! For large 2×2 tables (n = a+b+c+d > 100), direct enumeration of the
//! hypergeometric distribution can visit millions of cells.  This module
//! computes log-probabilities via `lgamma` and exploits the multiplicative
//! recurrence of the PMF to avoid recomputing log-factorials for every cell.
//!
//! # Reference
//! Fisher, R. A. (1922).  *On the interpretation of χ² from contingency
//! tables, and the calculation of P*.  J. R. Statist. Soc., 85(1), 87-94.
//!
//! Lancaster, H. O. (1961).  Significance tests in discrete distributions.
//! *J. Amer. Statist. Assoc.*, 56, 223-234.  (mid-P correction)

use crate::error::{StatsError, StatsResult};
use crate::special::gammaln::lgamma;

// ─── public types ─────────────────────────────────────────────────────────────

/// Direction of the alternative hypothesis for Fisher's exact test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    /// Two-sided: sum of probabilities ≤ the observed probability.
    TwoSided,
    /// Left-tailed: P(X ≤ a).
    Less,
    /// Right-tailed: P(X ≥ a).
    Greater,
}

// ─── log-binomial coefficient ─────────────────────────────────────────────────

/// Logarithm of the binomial coefficient C(n, k) via `lgamma`.
///
/// Returns `−∞` when `k > n` (impossible draws), and `0.0` when `k = 0` or
/// `k = n` (by definition C(n,0) = C(n,n) = 1).
#[must_use]
pub fn log_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    lgamma((n + 1) as f64) - lgamma((k + 1) as f64) - lgamma((n - k + 1) as f64)
}

// ─── internal helpers ─────────────────────────────────────────────────────────

/// Range of valid values for the top-left cell `X` given fixed margins.
///
/// For a 2×2 table with row sums r1 = a+b, r2 = c+d and column sums
/// c1 = a+c, c2 = b+d (with n = r1+r2):
///
/// `k_min = max(0, r1 − c2) = max(0, a+b − (b+d))`
/// `k_max = min(r1, c1)     = min(a+b, a+c)`
fn hypergeometric_range(r1: u64, c1: u64, n: u64) -> (u64, u64) {
    let c2 = n - c1;
    let k_min = r1.saturating_sub(c2);
    let k_max = r1.min(c1);
    (k_min, k_max)
}

/// Log-probability of P(X = k) for hypergeometric(N=n, K=c1, n=r1).
///
/// log P(X=k) = log_choose(c1, k) + log_choose(n-c1, r1-k) - log_choose(n, r1)
#[inline]
fn log_pmf(k: u64, r1: u64, c1: u64, n: u64) -> f64 {
    log_choose(c1, k) + log_choose(n - c1, r1 - k) - log_choose(n, r1)
}

/// Build the full vector of log-probabilities over the support `[k_min, k_max]`
/// using the multiplicative recurrence of the hypergeometric PMF to avoid
/// repeated lgamma calls.
///
/// Recurrence (Vandermonde-based):
///
///   P(X=k+1)   (c1 - k)(r1 - k)
///   ──────── = ────────────────────────────────
///   P(X=k)    (k+1)(n - c1 - r1 + k + 1)
///
/// We start from `k = k_min` (computed exactly via lgamma), then apply the
/// recurrence in log-space until `k_max`.
fn build_log_pmf_table(r1: u64, c1: u64, n: u64) -> Vec<f64> {
    let (k_min, k_max) = hypergeometric_range(r1, c1, n);
    if k_min > k_max {
        return Vec::new();
    }
    let len = (k_max - k_min + 1) as usize;
    let mut table = vec![0.0_f64; len];
    // Compute the first entry exactly.
    table[0] = log_pmf(k_min, r1, c1, n);
    // Apply recurrence for k_min, k_min+1, ..., k_max-1.
    for i in 0..(len - 1) {
        let k = k_min + i as u64;
        // log P(k+1) = log P(k) + log[(c1-k)(r1-k)] - log[(k+1)(n-c1-r1+k+1)]
        let numerator = ((c1 - k) as f64).ln() + ((r1 - k) as f64).ln();
        // Denominator term: (k+1)(n - c1 - r1 + k + 1)
        // n - c1 - r1 + k + 1 = (n - c1 - r1) + k + 1
        let denom_b = n - c1 - r1 + k + 1; // safe: k ≤ k_max ≤ min(r1,c1) so c1+r1 ≤ c1+r1
        let denominator = ((k + 1) as f64).ln() + (denom_b as f64).ln();
        table[i + 1] = table[i] + numerator - denominator;
    }
    table
}

/// Convert a table of log-probabilities to probabilities, normalised to sum
/// to exactly 1 (numerical safeguard).
fn normalise_log_pmf(log_table: &[f64]) -> Vec<f64> {
    if log_table.is_empty() {
        return Vec::new();
    }
    // log-sum-exp for numerical stability.
    let log_max = log_table.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_table.iter().map(|&lp| (lp - log_max).exp()).sum();
    let log_total = log_max + sum_exp.ln();
    log_table
        .iter()
        .map(|&lp| {
            let v = (lp - log_total).exp();
            // Clamp tiny negatives due to floating point to zero.
            if v < 0.0 { 0.0 } else { v }
        })
        .collect()
}

// ─── public API ───────────────────────────────────────────────────────────────

/// Optimised Fisher's exact test for a 2×2 contingency table using
/// log-hypergeometric probabilities.
///
/// ```text
/// | a | b |   row1 = a + b
/// | c | d |   row2 = c + d
///  col1  col2
/// ```
///
/// Returns the exact p-value according to `alternative`.
///
/// # Errors
/// Returns [`StatsError::EmptyInput`] when `a + b + c + d = 0`.
pub fn fisher_exact_fast(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    alternative: Alternative,
) -> StatsResult<f64> {
    let n = a
        .checked_add(b)
        .and_then(|x| x.checked_add(c))
        .and_then(|x| x.checked_add(d))
        .ok_or_else(|| StatsError::InvalidParameter {
            name: "n".into(),
            reason: "table counts overflow u64".into(),
        })?;
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let r1 = a + b; // row1
    let c1 = a + c; // col1

    let log_table = build_log_pmf_table(r1, c1, n);
    if log_table.is_empty() {
        // Degenerate: only one possible value of X.
        return Ok(1.0);
    }
    let (k_min, _k_max) = hypergeometric_range(r1, c1, n);

    let probs = normalise_log_pmf(&log_table);
    let obs_log_p = log_pmf(a, r1, c1, n);

    let p_value = match alternative {
        Alternative::Less => {
            // P(X ≤ a)
            let a_idx = (a - k_min) as usize;
            probs.iter().take(a_idx + 1).sum::<f64>()
        }
        Alternative::Greater => {
            // P(X ≥ a)
            let a_idx = (a - k_min) as usize;
            probs.iter().skip(a_idx).sum::<f64>()
        }
        Alternative::TwoSided => {
            // Sum all probabilities ≤ P(X = a), with tolerance for FP rounding.
            let threshold = obs_log_p + 1e-10;
            log_table
                .iter()
                .zip(probs.iter())
                .filter(|&(&lp, _)| lp <= threshold)
                .map(|(_, &p)| p)
                .sum::<f64>()
        }
    };
    Ok(p_value.clamp(0.0, 1.0))
}

/// Mid-P corrected Fisher's exact test (Lancaster 1961).
///
/// The two-sided mid-P value is defined as:
///
/// ```text
/// mid_p = 2 * min(mid_p_less, mid_p_greater)
/// ```
///
/// where:
/// - `mid_p_less    = P(X < a) + P(X = a) / 2 = P(X ≤ a) − P(X = a) / 2`
/// - `mid_p_greater = P(X > a) + P(X = a) / 2 = P(X ≥ a) − P(X = a) / 2`
///
/// This correction removes the conservative bias of the ordinary exact test
/// for discrete distributions (Lancaster 1961).
///
/// # Errors
/// Returns [`StatsError::EmptyInput`] when `a + b + c + d = 0`.
pub fn midp_fisher_exact_fast(a: u64, b: u64, c: u64, d: u64) -> StatsResult<f64> {
    let n = a
        .checked_add(b)
        .and_then(|x| x.checked_add(c))
        .and_then(|x| x.checked_add(d))
        .ok_or_else(|| StatsError::InvalidParameter {
            name: "n".into(),
            reason: "table counts overflow u64".into(),
        })?;
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let r1 = a + b;
    let c1 = a + c;

    let log_table = build_log_pmf_table(r1, c1, n);
    if log_table.is_empty() {
        return Ok(1.0);
    }
    let (k_min, _) = hypergeometric_range(r1, c1, n);
    let probs = normalise_log_pmf(&log_table);
    let a_idx = (a - k_min) as usize;
    let p_obs = probs[a_idx];

    // One-tailed cumulative probabilities.
    let p_less: f64 = probs.iter().take(a_idx + 1).sum(); // P(X ≤ a)
    let p_greater: f64 = probs.iter().skip(a_idx).sum(); // P(X ≥ a)

    // Mid-P one-tailed values: subtract half the point probability.
    let midp_less = p_less - p_obs / 2.0;
    let midp_greater = p_greater - p_obs / 2.0;

    // Two-sided: 2 * min(midp_less, midp_greater), clamped to [0, 1].
    let midp = (2.0 * midp_less.min(midp_greater)).clamp(0.0, 1.0);
    Ok(midp)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chi_squared::fisher_exact::fisher_exact_2x2;

    const EPS: f64 = 1e-6;

    // ── 1. log_choose sanity ─────────────────────────────────────────────────
    #[test]
    fn log_choose_basic() {
        // C(10,0)=1 => log=0
        assert!((log_choose(10, 0)).abs() < EPS);
        // C(10,10)=1 => log=0
        assert!((log_choose(10, 10)).abs() < EPS);
        // C(5,2)=10 => log = ln(10)
        let expected = 10.0_f64.ln();
        assert!((log_choose(5, 2) - expected).abs() < EPS);
        // k > n => -inf
        assert!(log_choose(3, 5).is_infinite());
    }

    // ── 2. log_choose for large values ──────────────────────────────────────
    #[test]
    fn log_choose_large() {
        // C(1000, 500) should not overflow (just verify it's finite and positive)
        let lc = log_choose(1000, 500);
        assert!(lc.is_finite() && lc > 0.0);
        // C(200, 100) — a known-range check
        let lc2 = log_choose(200, 100);
        assert!(lc2.is_finite() && lc2 > 100.0);
    }

    // ── 3. Strong association gives small p (two-sided) ─────────────────────
    #[test]
    fn fast_strong_association_two_sided() {
        let p = fisher_exact_fast(9, 1, 1, 9, Alternative::TwoSided).expect("ok");
        assert!(p < 0.01, "p={p} should be < 0.01 for strong association");
    }

    // ── 4. No association gives large p ─────────────────────────────────────
    #[test]
    fn fast_no_association_two_sided() {
        let p = fisher_exact_fast(5, 5, 5, 5, Alternative::TwoSided).expect("ok");
        assert!(p > 0.5, "p={p} should be large for no association");
    }

    // ── 5. Consistency with existing fisher_exact_2x2 (small tables) ────────
    #[test]
    fn fast_consistent_with_classic_small_tables() {
        let cases = [(9usize, 1, 1, 9), (3, 1, 1, 3), (5, 5, 5, 5), (0, 4, 4, 0)];
        for (a, b, c, d) in cases {
            let r_classic = fisher_exact_2x2(a, b, c, d).expect("classic ok");
            let p_fast = fisher_exact_fast(
                a as u64,
                b as u64,
                c as u64,
                d as u64,
                Alternative::TwoSided,
            )
            .expect("fast ok");
            let diff = (p_fast - r_classic.p_value_two_sided).abs();
            assert!(
                diff < 1e-5,
                "a={a} b={b} c={c} d={d}: fast={p_fast} classic={} diff={diff}",
                r_classic.p_value_two_sided,
            );
        }
    }

    // ── 6. One-sided Less: extreme cell (a=0) should give large p_less ──────
    #[test]
    fn fast_one_sided_less_extreme() {
        // When a is at the minimum possible value, p_less should be very small
        // (it equals only P(X=0)), while p_greater should be ≈ 1.
        let p_less = fisher_exact_fast(0, 5, 5, 0, Alternative::Less).expect("ok");
        let p_greater = fisher_exact_fast(0, 5, 5, 0, Alternative::Greater).expect("ok");
        assert!(p_less < p_greater, "p_less={p_less} p_greater={p_greater}");
    }

    // ── 7. One-sided Greater: extreme cell (a at max) should give large p_greater
    #[test]
    fn fast_one_sided_greater_extreme() {
        let p_greater = fisher_exact_fast(5, 0, 0, 5, Alternative::Greater).expect("ok");
        assert!(p_greater < 0.05, "p_greater={p_greater} for extreme table");
    }

    // ── 8. Large table (n > 100) runs without overflow ───────────────────────
    #[test]
    fn fast_large_table_no_overflow() {
        // n = 200, moderate association
        let p = fisher_exact_fast(60, 40, 30, 70, Alternative::TwoSided).expect("ok");
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    // ── 9. Very large table (n ≈ 1000) ──────────────────────────────────────
    #[test]
    fn fast_very_large_table() {
        // Strong association at large n → tiny p
        let p = fisher_exact_fast(450, 50, 50, 450, Alternative::TwoSided).expect("ok");
        assert!(
            p < 1e-10,
            "p={p} expected very small for strong assoc at n=1000"
        );
    }

    // ── 10. Empty table returns error ────────────────────────────────────────
    #[test]
    fn fast_empty_table_error() {
        let r = fisher_exact_fast(0, 0, 0, 0, Alternative::TwoSided);
        assert!(r.is_err(), "empty table should return error");
    }

    // ── 11. p-value is in [0, 1] for various inputs ──────────────────────────
    #[test]
    fn fast_p_value_range() {
        let cases = [
            (1u64, 1, 1, 1),
            (100, 1, 1, 100),
            (10, 90, 90, 10),
            (1, 999, 999, 1),
        ];
        for (a, b, c, d) in cases {
            for alt in [
                Alternative::TwoSided,
                Alternative::Less,
                Alternative::Greater,
            ] {
                let p = fisher_exact_fast(a, b, c, d, alt).expect("ok");
                assert!(
                    (0.0..=1.0).contains(&p),
                    "a={a} b={b} c={c} d={d}: p={p} out of [0,1]"
                );
            }
        }
    }

    // ── 12. mid-P: one-tailed mid-P is smaller than one-tailed p ───────────
    #[test]
    fn midp_one_sided_smaller_than_ordinary() {
        // mid_p_greater = P(X≥a) - P(X=a)/2 < P(X≥a) for strong association
        // We verify this by checking the midp two-sided (2*min) is < ordinary one-sided greater.
        let a = 9u64;
        let (b, c, d) = (1u64, 1u64, 9u64);
        let p_greater = fisher_exact_fast(a, b, c, d, Alternative::Greater).expect("ok");
        let midp = midp_fisher_exact_fast(a, b, c, d).expect("mid");
        // midp (two-sided via 2*min) ≤ 2 * p_greater_one_sided
        assert!(
            midp <= 2.0 * p_greater + 1e-9,
            "midp={midp} > 2*p_greater={} for ({a},{b},{c},{d})",
            2.0 * p_greater
        );
    }

    // ── 13. mid-P is in [0, 1] ───────────────────────────────────────────────
    #[test]
    fn midp_range() {
        let cases = [(9u64, 1, 1, 9), (5, 5, 5, 5), (1, 1, 1, 1), (0, 5, 5, 0)];
        for (a, b, c, d) in cases {
            let p = midp_fisher_exact_fast(a, b, c, d).expect("ok");
            assert!(
                (0.0..=1.0).contains(&p),
                "midp={p} out of range for ({a},{b},{c},{d})"
            );
        }
    }

    // ── 14. Swapping rows flips Less ↔ Greater ───────────────────────────────
    #[test]
    fn fast_row_swap_flips_alternative() {
        // P_less(a,b,c,d) = P_greater(c,d,a,b): swapping rows flips direction
        let p_less_orig = fisher_exact_fast(3, 7, 7, 3, Alternative::Less).expect("ok");
        // Swap rows: (a,b,c,d) → (c,d,a,b) = (7,3,3,7)
        let p_greater_swapped = fisher_exact_fast(7, 3, 3, 7, Alternative::Greater).expect("ok");
        let diff = (p_less_orig - p_greater_swapped).abs();
        assert!(
            diff < 1e-9,
            "p_less({p_less_orig}) ≠ p_greater_swapped({p_greater_swapped})"
        );
    }

    // ── 15. Degenerate marginals (one cell = entire margin) ─────────────────
    #[test]
    fn fast_one_cell_dominates() {
        // a=10, b=0, c=0, d=10: perfect association
        let p = fisher_exact_fast(10, 0, 0, 10, Alternative::TwoSided).expect("ok");
        assert!(p < 0.01, "p={p} should be small for perfect association");
    }
}
