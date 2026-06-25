//! Fisher-Z conditional-independence test calibration.
//!
//! Two independent checks:
//!
//! 1. **Critical-value calibration.** The production [`crate::discovery::pc::fisher_z_test`]
//!    compares the standardized Fisher-Z statistic against a hard-coded normal
//!    critical value (1.645 / 1.96 / 2.576). We re-derive the exact two-sided
//!    quantiles from an erf-based normal CDF and confirm the constants are the
//!    right percentiles. This is the CPU analogue of "calibration vs. exact
//!    percentile from R `pcalg`" — `pcalg::gaussCItest` uses precisely
//!    `qnorm(1 − α/2)`.
//!
//! 2. **Empirical type-I error.** On data drawn from *independent* Gaussians the
//!    test rejects (declares dependence) with probability ≈ α. We estimate that
//!    rejection rate by Monte-Carlo and check it brackets the nominal level.

use crate::discovery::pc::{fisher_z_test, partial_corr};
use crate::handle::LcgRng;
use crate::verification::reference::two_sided_z_quantile;

/// The exact two-sided normal critical value `qnorm(1 − α/2)` for the level the
/// production test snaps `alpha` to (≤0.01 → 2.576, ≤0.05 → 1.96, else 1.645).
#[must_use]
pub fn exact_fisher_z_critical(alpha: f32) -> f64 {
    let level = if alpha <= 0.01 {
        0.01
    } else if alpha <= 0.05 {
        0.05
    } else {
        0.10
    };
    two_sided_z_quantile(level)
}

/// Monte-Carlo estimate of the test's type-I error: fraction of independent
/// Gaussian samples for which `fisher_z_test` (with empty conditioning set)
/// declares dependence.
#[must_use]
pub fn empirical_type_one_error(
    n_samples: usize,
    n_trials: usize,
    alpha: f32,
    rng: &mut LcgRng,
) -> f64 {
    let mut rejections = 0usize;
    for _ in 0..n_trials {
        let mut x = vec![0.0_f32; n_samples];
        let mut y = vec![0.0_f32; n_samples];
        for i in 0..n_samples {
            x[i] = rng.next_normal();
            y[i] = rng.next_normal(); // independent of x
        }
        let r = partial_corr(&x, &y, &[], n_samples);
        if fisher_z_test(r, n_samples, 0, alpha) {
            rejections += 1;
        }
    }
    rejections as f64 / n_trials as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardcoded_constants_are_correct_percentiles() {
        // The constants baked into fisher_z_test must equal qnorm(1 − α/2).
        assert!(
            (exact_fisher_z_critical(0.05) - 1.96).abs() < 2e-3,
            "0.05 critical = {}",
            exact_fisher_z_critical(0.05)
        );
        assert!(
            (exact_fisher_z_critical(0.01) - 2.576).abs() < 2e-3,
            "0.01 critical = {}",
            exact_fisher_z_critical(0.01)
        );
        assert!(
            (exact_fisher_z_critical(0.10) - 1.645).abs() < 2e-3,
            "0.10 critical = {}",
            exact_fisher_z_critical(0.10)
        );
    }

    #[test]
    fn test_decision_agrees_with_exact_threshold() {
        // Construct a Fisher-Z statistic landing just inside / outside the exact
        // 5% critical value and confirm the production decision matches.
        let n = 103usize; // df = n - 0 - 3 = 100
        let df = (n as f64 - 3.0).sqrt();
        let z_crit = exact_fisher_z_critical(0.05);
        // Pick z just above and below the critical |z| = z_crit / sqrt(df).
        let z_reject = (z_crit / df) * 1.10;
        let z_accept = (z_crit / df) * 0.90;
        // Invert Fisher transform z = atanh(r) => r = tanh(z).
        let r_reject = z_reject.tanh() as f32;
        let r_accept = z_accept.tanh() as f32;
        assert!(
            fisher_z_test(r_reject, n, 0, 0.05),
            "should reject at r={r_reject}"
        );
        assert!(
            !fisher_z_test(r_accept, n, 0, 0.05),
            "should accept at r={r_accept}"
        );
    }

    #[test]
    fn empirical_type_one_error_brackets_nominal() {
        // On independent Gaussians the rejection rate should sit near alpha.
        // With 600 trials the binomial sd at α=0.05 is ~0.0089, so a ±0.04 band
        // is a comfortable, non-flaky bracket while still being a real check
        // (a broken test that always/never rejects would fail this).
        let mut rng = LcgRng::new(20_260_621);
        let rate = empirical_type_one_error(120, 600, 0.05, &mut rng);
        assert!(
            (0.01..0.12).contains(&rate),
            "empirical type-I error {rate} not near 0.05"
        );
    }

    #[test]
    fn larger_alpha_rejects_more_often() {
        // Monotonicity: a more permissive level rejects independent data more.
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let rate_strict = empirical_type_one_error(120, 400, 0.01, &mut rng_a);
        let rate_loose = empirical_type_one_error(120, 400, 0.10, &mut rng_b);
        assert!(
            rate_loose >= rate_strict,
            "loose {rate_loose} should reject >= strict {rate_strict}"
        );
    }
}
