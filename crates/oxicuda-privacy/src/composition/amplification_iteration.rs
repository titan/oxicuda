//! Privacy amplification by iteration (Feldman-Mironov-Talwar 2018).
//!
//! Reference: Feldman, Mironov, Talwar & Thakurta (2018), "Privacy Amplification
//! by Iteration", FOCS 2018, arXiv:1808.06651.
//!
//! # Setting
//! Consider a sequence of *contractive noisy iterations* (CNI):
//!
//! `X_{t} = ψ_t(X_{t−1}) + Z_t`,  with each `ψ_t` 1-Lipschitz (a contraction)
//! and `Z_t ~ 𝒩(0, σ²·I)`.
//!
//! This is the abstract form of (projected) noisy SGD on a convex loss: each
//! gradient step `x ↦ x − η∇f(x)` is a contraction when `f` is convex and the
//! step size is small enough, and the Gaussian noise is the privacy noise.
//!
//! Two inputs that differ only in the data used at a single step `k` (out of
//! `T` total) start at distance `0`, are pushed apart by at most the per-step
//! sensitivity `s` at step `k`, and are then *contracted back together* by the
//! remaining `T − k` noisy iterations.
//!
//! # Main bound (Theorem 22 / Corollary 23)
//! Splitting the gap `s` over the `T − k` post-injection steps with a uniform
//! schedule `a_i = s / (T − k)`, the Rényi divergence of order `α` between the
//! two output distributions is bounded by
//!
//! `R_α ≤ Σ_{i=1}^{T−k} α · a_i² / (2σ²)  =  α · s² / (2σ² · (T − k))`.
//!
//! Compared with the *last-step* (non-amplified) Gaussian RDP `α·s²/(2σ²)`,
//! iterating `T − k` further contractive noisy steps **divides** the privacy
//! loss by `(T − k)`. A record that participates early (small `k`) is therefore
//! far better protected than one used at the final step (`k = T`, no
//! amplification).

use crate::error::{PrivacyError, PrivacyResult};

/// Configuration for privacy amplification by iteration on a CNI sequence.
#[derive(Debug, Clone)]
pub struct IterationAmplificationConfig {
    /// Total number of contractive noisy iterations `T ≥ 1`.
    pub total_steps: usize,
    /// Per-step Gaussian noise standard deviation `σ > 0`.
    pub sigma: f64,
    /// Per-step sensitivity `s > 0` (L2 distance the differing step can inject;
    /// for noisy SGD with learning rate `η` and gradient clip `C`, `s = 2ηC/B`
    /// for batch size `B`, or `s = 2ηC` for full-record substitution).
    pub sensitivity: f64,
}

impl IterationAmplificationConfig {
    /// Construct and validate an [`IterationAmplificationConfig`].
    ///
    /// # Errors
    /// - `InvalidParameter` if `total_steps == 0` or `sigma ≤ 0`.
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    pub fn new(total_steps: usize, sigma: f64, sensitivity: f64) -> PrivacyResult<Self> {
        if total_steps == 0 {
            return Err(PrivacyError::InvalidParameter(
                "total_steps must be ≥ 1".into(),
            ));
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive and finite, got {sigma}"
            )));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        Ok(Self {
            total_steps,
            sigma,
            sensitivity,
        })
    }
}

/// Amplification-by-iteration RDP bounds.
#[derive(Debug, Clone)]
pub struct IterationAmplification;

impl IterationAmplification {
    /// RDP at order `alpha` for a record **last used at step `inject_step`**
    /// (1-based, `1 ≤ inject_step ≤ total_steps`), after which
    /// `total_steps − inject_step` contractive noisy iterations amplify privacy.
    ///
    /// `R_α = α · s² / (2σ² · max(1, T − k))`.
    ///
    /// When `inject_step == total_steps` (the differing data is used at the
    /// final step) there is no amplification and this reduces to the standard
    /// Gaussian-mechanism RDP `α·s²/(2σ²)`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha ≤ 1.0` or non-finite.
    /// - `IndexOutOfRange` if `inject_step == 0` or `inject_step > total_steps`.
    pub fn rdp_at_step(
        cfg: &IterationAmplificationConfig,
        inject_step: usize,
        alpha: f64,
    ) -> PrivacyResult<f64> {
        if !alpha.is_finite() || alpha <= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "alpha must be > 1 for RDP, got {alpha}"
            )));
        }
        if inject_step == 0 || inject_step > cfg.total_steps {
            return Err(PrivacyError::IndexOutOfRange(inject_step, cfg.total_steps));
        }
        let remaining = cfg.total_steps - inject_step;
        let divisor = (remaining.max(1)) as f64;
        let s = cfg.sensitivity;
        Ok(alpha * s * s / (2.0 * cfg.sigma * cfg.sigma * divisor))
    }

    /// Worst-case (over the injection step) amplified RDP at order `alpha`.
    ///
    /// The worst case is always the **last** step (`inject_step = total_steps`),
    /// where no contraction follows, so this returns the un-amplified Gaussian
    /// RDP `α·s²/(2σ²)`. It is provided as the conservative bound to use when the
    /// adversary may choose which step a record participates in.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha ≤ 1.0` or non-finite.
    pub fn worst_case_rdp(cfg: &IterationAmplificationConfig, alpha: f64) -> PrivacyResult<f64> {
        Self::rdp_at_step(cfg, cfg.total_steps, alpha)
    }

    /// Best-case amplified RDP at order `alpha` for a record used at the very
    /// **first** step (`inject_step = 1`), maximally amplified by the remaining
    /// `total_steps − 1` iterations.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha ≤ 1.0` or non-finite.
    pub fn best_case_rdp(cfg: &IterationAmplificationConfig, alpha: f64) -> PrivacyResult<f64> {
        Self::rdp_at_step(cfg, 1, alpha)
    }

    /// Multiplicative amplification factor of iterating from `inject_step`:
    /// `1 / max(1, T − k)`. A factor `< 1` means privacy improved relative to
    /// the un-amplified last-step bound.
    ///
    /// # Errors
    /// - `IndexOutOfRange` if `inject_step == 0` or `inject_step > total_steps`.
    pub fn amplification_factor(
        cfg: &IterationAmplificationConfig,
        inject_step: usize,
    ) -> PrivacyResult<f64> {
        if inject_step == 0 || inject_step > cfg.total_steps {
            return Err(PrivacyError::IndexOutOfRange(inject_step, cfg.total_steps));
        }
        let remaining = cfg.total_steps - inject_step;
        Ok(1.0 / (remaining.max(1) as f64))
    }

    /// Convert an amplified RDP value at order `alpha` to an `(ε, δ)`-DP
    /// guarantee using the Canonne-Kamath-Steinke (2020) tight conversion:
    ///
    /// `ε = rdp + ln((α−1)/α) − (ln δ + ln α)/(α−1)`,  clamped to `≥ 0`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha ≤ 1.0`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn rdp_to_epsilon(rdp: f64, alpha: f64, delta: f64) -> PrivacyResult<f64> {
        if !alpha.is_finite() || alpha <= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "alpha must be > 1, got {alpha}"
            )));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let eps = rdp + ((alpha - 1.0) / alpha).ln() - (delta.ln() + alpha.ln()) / (alpha - 1.0);
        Ok(eps.max(0.0))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(t: usize, sigma: f64, s: f64) -> IterationAmplificationConfig {
        IterationAmplificationConfig::new(t, sigma, s).expect("cfg")
    }

    // 1. Last step is not amplified: equals plain Gaussian RDP α·s²/(2σ²).
    #[test]
    fn last_step_no_amplification() {
        let c = cfg(100, 2.0, 1.0);
        let alpha = 5.0;
        let got = IterationAmplification::rdp_at_step(&c, 100, alpha).expect("r");
        let want = alpha * 1.0 / (2.0 * 4.0);
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    // 2. Earlier injection is more amplified (smaller RDP).
    #[test]
    fn earlier_injection_more_amplified() {
        let c = cfg(100, 1.0, 1.0);
        let early = IterationAmplification::rdp_at_step(&c, 1, 4.0).expect("r");
        let late = IterationAmplification::rdp_at_step(&c, 90, 4.0).expect("r");
        assert!(early < late, "early should amplify more: {early} < {late}");
    }

    // 3. First-step amplification divides last-step RDP by (T − 1).
    #[test]
    fn first_step_divides_by_t_minus_1() {
        let c = cfg(50, 1.0, 1.0);
        let first = IterationAmplification::best_case_rdp(&c, 3.0).expect("r");
        let last = IterationAmplification::worst_case_rdp(&c, 3.0).expect("r");
        assert!(
            (first * 49.0 - last).abs() < 1e-9,
            "first·(T−1) should equal last: {} vs {last}",
            first * 49.0
        );
    }

    // 4. Larger noise → smaller amplified RDP.
    #[test]
    fn more_noise_smaller_rdp() {
        let lo = cfg(20, 1.0, 1.0);
        let hi = cfg(20, 4.0, 1.0);
        let r_lo = IterationAmplification::rdp_at_step(&lo, 5, 2.0).expect("r");
        let r_hi = IterationAmplification::rdp_at_step(&hi, 5, 2.0).expect("r");
        assert!(r_hi < r_lo, "{r_hi} < {r_lo}");
    }

    // 5. Larger sensitivity → larger RDP (quadratic).
    #[test]
    fn larger_sensitivity_larger_rdp() {
        let small = cfg(20, 1.0, 1.0);
        let big = cfg(20, 1.0, 2.0);
        let r_small = IterationAmplification::rdp_at_step(&small, 5, 2.0).expect("r");
        let r_big = IterationAmplification::rdp_at_step(&big, 5, 2.0).expect("r");
        // Doubling s quadruples RDP.
        assert!(
            (r_big - 4.0 * r_small).abs() < 1e-9,
            "{r_big} vs {}",
            4.0 * r_small
        );
    }

    // 6. Amplification factor is 1 at the last step, < 1 earlier.
    #[test]
    fn amplification_factor_values() {
        let c = cfg(10, 1.0, 1.0);
        assert!(
            (IterationAmplification::amplification_factor(&c, 10).expect("f") - 1.0).abs() < 1e-12
        );
        let early = IterationAmplification::amplification_factor(&c, 1).expect("f");
        assert!(early < 1.0, "early factor should be < 1, got {early}");
        // T=10, inject at step 1 → remaining 9 → factor 1/9.
        assert!((early - 1.0 / 9.0).abs() < 1e-12, "factor {early}");
    }

    // 7. RDP scales linearly with the Rényi order.
    #[test]
    fn rdp_linear_in_alpha() {
        let c = cfg(30, 1.5, 1.0);
        let r2 = IterationAmplification::rdp_at_step(&c, 5, 2.0).expect("r");
        let r6 = IterationAmplification::rdp_at_step(&c, 5, 6.0).expect("r");
        assert!(
            (r6 - 3.0 * r2).abs() < 1e-9,
            "α-linear: {r6} vs {}",
            3.0 * r2
        );
    }

    // 8. RDP → (ε, δ) conversion is finite and non-negative.
    #[test]
    fn rdp_to_epsilon_finite() {
        let c = cfg(1000, 1.0, 1.0);
        let r = IterationAmplification::best_case_rdp(&c, 10.0).expect("r");
        let eps = IterationAmplification::rdp_to_epsilon(r, 10.0, 1e-5).expect("e");
        assert!(eps.is_finite() && eps >= 0.0, "ε = {eps}");
    }

    // 9. Amplification lowers the final ε vs the worst case.
    #[test]
    fn amplification_lowers_epsilon() {
        let c = cfg(500, 1.0, 1.0);
        let alpha = 8.0;
        let r_best = IterationAmplification::best_case_rdp(&c, alpha).expect("r");
        let r_worst = IterationAmplification::worst_case_rdp(&c, alpha).expect("r");
        let eps_best = IterationAmplification::rdp_to_epsilon(r_best, alpha, 1e-5).expect("e");
        let eps_worst = IterationAmplification::rdp_to_epsilon(r_worst, alpha, 1e-5).expect("e");
        assert!(eps_best < eps_worst, "{eps_best} < {eps_worst}");
    }

    // 10. T = 1 (single step) admits only step 1, with no amplification.
    #[test]
    fn single_step_no_amplification() {
        let c = cfg(1, 2.0, 1.0);
        let r = IterationAmplification::rdp_at_step(&c, 1, 4.0).expect("r");
        let want = 4.0 * 1.0 / (2.0 * 4.0);
        assert!((r - want).abs() < 1e-12, "got {r}, want {want}");
    }

    // 11. Error paths: bad config, bad α, bad step index, bad δ.
    #[test]
    fn error_paths() {
        assert!(IterationAmplificationConfig::new(0, 1.0, 1.0).is_err());
        assert!(IterationAmplificationConfig::new(5, 0.0, 1.0).is_err());
        assert!(IterationAmplificationConfig::new(5, 1.0, 0.0).is_err());
        let c = cfg(10, 1.0, 1.0);
        assert!(IterationAmplification::rdp_at_step(&c, 5, 1.0).is_err());
        assert!(IterationAmplification::rdp_at_step(&c, 0, 2.0).is_err());
        assert!(IterationAmplification::rdp_at_step(&c, 11, 2.0).is_err());
        assert!(IterationAmplification::amplification_factor(&c, 0).is_err());
        assert!(IterationAmplification::rdp_to_epsilon(0.1, 2.0, 0.0).is_err());
        assert!(IterationAmplification::rdp_to_epsilon(0.1, 1.0, 1e-5).is_err());
    }

    // 12. More total iterations → stronger amplification for an early record.
    #[test]
    fn more_iterations_stronger_amplification() {
        let short = cfg(10, 1.0, 1.0);
        let long = cfg(1000, 1.0, 1.0);
        let r_short = IterationAmplification::best_case_rdp(&short, 4.0).expect("r");
        let r_long = IterationAmplification::best_case_rdp(&long, 4.0).expect("r");
        assert!(
            r_long < r_short,
            "longer training amplifies more: {r_long} < {r_short}"
        );
    }
}
