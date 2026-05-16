//! Fisher's exact test for a 2 x 2 contingency table.

use crate::error::{StatsError, StatsResult};
use crate::special::gammaln::lgamma;

/// Result of Fisher's exact test.
#[derive(Debug, Clone, Copy)]
pub struct FisherExactResult {
    pub odds_ratio: f64,
    pub p_value_two_sided: f64,
    pub p_value_one_sided_less: f64,
    pub p_value_one_sided_greater: f64,
}

/// Hypergeometric probability of observing exactly `k` successes out of `n` draws
/// from a population of size `n_total` with `k_total` successes.
fn ln_hyper(k: usize, n: usize, n_total: usize, k_total: usize) -> f64 {
    let ln_choose = |a: usize, b: usize| -> f64 {
        if b > a {
            f64::NEG_INFINITY
        } else {
            lgamma((a + 1) as f64) - lgamma((b + 1) as f64) - lgamma((a - b + 1) as f64)
        }
    };
    let l1 = ln_choose(k_total, k);
    let l2 = ln_choose(n_total - k_total, n - k);
    let l3 = ln_choose(n_total, n);
    l1 + l2 - l3
}

/// Fisher's exact test for a 2x2 table:
/// | a | b |
/// | c | d |
pub fn fisher_exact_2x2(a: usize, b: usize, c: usize, d: usize) -> StatsResult<FisherExactResult> {
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let n = row1 + row2;
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    let k_total = col1;
    // We iterate over all possible values of a' = k for fixed margins
    let k_max = row1.min(k_total);
    let k_min = k_total.saturating_sub(row2);
    let observed_lp = ln_hyper(a, row1, n, k_total);
    let mut p_two = 0.0;
    let mut p_less = 0.0;
    let mut p_greater = 0.0;
    for k in k_min..=k_max {
        let lp = ln_hyper(k, row1, n, k_total);
        if lp.is_finite() {
            let p = lp.exp();
            // Two-sided: sum probabilities <= observed prob
            if lp <= observed_lp + 1e-12 {
                p_two += p;
            }
            if k <= a {
                p_less += p;
            }
            if k >= a {
                p_greater += p;
            }
        }
    }
    let odds_ratio = if c == 0 || b == 0 {
        if a == 0 && d == 0 {
            f64::NAN
        } else if c == 0 && b == 0 {
            f64::INFINITY
        } else {
            // Conventional handling: small constant smoothing returns NaN/Inf appropriately
            f64::INFINITY
        }
    } else {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    };
    Ok(FisherExactResult {
        odds_ratio,
        p_value_two_sided: p_two.clamp(0.0, 1.0),
        p_value_one_sided_less: p_less.clamp(0.0, 1.0),
        p_value_one_sided_greater: p_greater.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_exact_strong_assoc() {
        // 9, 1 / 1, 9 -> strong association
        let r = fisher_exact_2x2(9, 1, 1, 9).expect("ok");
        assert!(r.p_value_two_sided < 0.01);
    }

    #[test]
    fn fisher_exact_no_assoc() {
        let r = fisher_exact_2x2(5, 5, 5, 5).expect("ok");
        assert!(r.p_value_two_sided > 0.5);
        assert!((r.odds_ratio - 1.0).abs() < 1e-12);
    }
}
