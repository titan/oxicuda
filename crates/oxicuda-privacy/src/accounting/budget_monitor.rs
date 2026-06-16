//! Runtime privacy-budget monitor with a composition circuit-breaker.
//!
//! Tracks the cumulative `(ε, δ)` spend across a sequence of differentially
//! private queries and *refuses* (returns `BudgetExhausted`) any query that
//! would push the cumulative spend past the configured total budget — the
//! "circuit breaker" — without committing it.
//!
//! Two accumulation modes are supported:
//! - **Basic composition**: cumulative ε and δ are the plain sums of the
//!   per-query spends (`Σ εᵢ`, `Σ δᵢ`).
//! - **Advanced composition** (Dwork, Rothblum & Vadhan, 2010): the cumulative
//!   ε grows sub-linearly,
//!
//!   ```text
//!       ε = Σ εᵢ(e^{εᵢ} − 1) + sqrt( 2 ln(1/δ') · Σ εᵢ² ),
//!   ```
//!
//!   at the cost of an extra slack `δ'` added to the cumulative δ. For many
//!   small queries this is far tighter than the linear basic-composition sum.
//!
//! References:
//! - Dwork, Rothblum & Vadhan (2010), "Boosting and Differential Privacy",
//!   FOCS.
//! - Dwork & Roth (2014), "The Algorithmic Foundations of Differential
//!   Privacy", Theorem 3.20 (advanced composition).

use crate::error::{PrivacyError, PrivacyResult};

/// Composition accounting mode for a [`BudgetMonitor`].
#[derive(Debug, Clone, Copy)]
pub enum CompositionMode {
    /// Basic (linear) composition: cumulative ε and δ are plain sums.
    Basic,
    /// Advanced (Dwork–Rothblum–Vadhan) composition with reserved slack `δ'`.
    Advanced {
        /// Failure-probability slack `δ' ∈ (0, total_delta)` reserved for the
        /// advanced-composition bound.
        slack_delta: f64,
    },
}

/// Runtime privacy-budget monitor enforcing a `(ε, δ)` total via a
/// circuit-breaker on each query.
#[derive(Debug, Clone)]
pub struct BudgetMonitor {
    total_epsilon: f64,
    total_delta: f64,
    mode: CompositionMode,
    /// `Σ εᵢ` over committed queries.
    sum_epsilon: f64,
    /// `Σ δᵢ` over committed queries.
    sum_delta: f64,
    /// `Σ εᵢ(e^{εᵢ} − 1)` — advanced-composition drift term.
    sum_drift: f64,
    /// `Σ εᵢ²` — advanced-composition variance term.
    sum_epsilon_sq: f64,
    /// Number of committed queries.
    count: usize,
}

impl BudgetMonitor {
    /// Create a basic-composition budget monitor with the given totals.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `total_epsilon` is non-finite or `≤ 0`.
    /// - `InvalidDelta` if `total_delta ∉ [0, 1)`.
    pub fn new(total_epsilon: f64, total_delta: f64) -> PrivacyResult<Self> {
        if total_epsilon <= 0.0 || !total_epsilon.is_finite() {
            return Err(PrivacyError::NonPositiveEpsilon(total_epsilon));
        }
        if !(0.0..1.0).contains(&total_delta) {
            return Err(PrivacyError::InvalidDelta(total_delta));
        }
        Ok(Self {
            total_epsilon,
            total_delta,
            mode: CompositionMode::Basic,
            sum_epsilon: 0.0,
            sum_delta: 0.0,
            sum_drift: 0.0,
            sum_epsilon_sq: 0.0,
            count: 0,
        })
    }

    /// Create an advanced-composition (Dwork–Rothblum–Vadhan) budget monitor.
    ///
    /// `slack_delta` (`δ'`) is the failure-probability slack reserved from the
    /// δ budget for the advanced-composition bound; it must satisfy
    /// `0 < slack_delta < total_delta`.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `total_epsilon` is non-finite or `≤ 0`.
    /// - `InvalidDelta` if `total_delta ∉ (0, 1)`.
    /// - `InvalidParameter` if `slack_delta ∉ (0, total_delta)`.
    pub fn new_advanced(
        total_epsilon: f64,
        total_delta: f64,
        slack_delta: f64,
    ) -> PrivacyResult<Self> {
        if total_epsilon <= 0.0 || !total_epsilon.is_finite() {
            return Err(PrivacyError::NonPositiveEpsilon(total_epsilon));
        }
        if !(total_delta > 0.0 && total_delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(total_delta));
        }
        if !(slack_delta > 0.0 && slack_delta < total_delta) {
            return Err(PrivacyError::InvalidParameter(format!(
                "advanced-composition slack δ' must satisfy 0 < δ' < total_delta={total_delta}, got {slack_delta}"
            )));
        }
        let mut monitor = Self::new(total_epsilon, total_delta)?;
        monitor.mode = CompositionMode::Advanced { slack_delta };
        Ok(monitor)
    }

    /// Cumulative `(ε, δ)` for a hypothetical accumulator state under the
    /// configured composition mode.
    fn spend_of(
        &self,
        sum_eps: f64,
        sum_del: f64,
        drift: f64,
        eps_sq: f64,
        count: usize,
    ) -> (f64, f64) {
        match self.mode {
            CompositionMode::Basic => (sum_eps, sum_del),
            CompositionMode::Advanced { slack_delta } => {
                if count == 0 {
                    (0.0, 0.0)
                } else {
                    let eps = drift + (2.0 * eps_sq * (1.0 / slack_delta).ln()).sqrt();
                    (eps, sum_del + slack_delta)
                }
            }
        }
    }

    /// Cumulative `(ε, δ)` spent so far under the configured composition mode.
    #[must_use]
    pub fn spent(&self) -> (f64, f64) {
        self.spend_of(
            self.sum_epsilon,
            self.sum_delta,
            self.sum_drift,
            self.sum_epsilon_sq,
            self.count,
        )
    }

    /// Remaining `(ε, δ)` budget headroom (each clamped at zero).
    #[must_use]
    pub fn remaining(&self) -> (f64, f64) {
        let (spent_eps, spent_del) = self.spent();
        (
            (self.total_epsilon - spent_eps).max(0.0),
            (self.total_delta - spent_del).max(0.0),
        )
    }

    /// Configured total `(ε, δ)` budget.
    #[must_use]
    pub fn total(&self) -> (f64, f64) {
        (self.total_epsilon, self.total_delta)
    }

    /// Number of committed queries.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The configured composition mode.
    #[must_use]
    pub fn mode(&self) -> CompositionMode {
        self.mode
    }

    /// Attempt to spend `(epsilon, delta)`, committing only if the resulting
    /// cumulative spend stays within the total budget (the circuit breaker).
    ///
    /// On rejection the committed spend is left **unchanged** and the returned
    /// `BudgetExhausted` reports the projected (would-be) spend and total for
    /// whichever dimension (ε or δ) was first exceeded.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon` is non-finite or `≤ 0`.
    /// - `InvalidDelta` if `delta ∉ [0, 1)`.
    /// - `BudgetExhausted` if committing would exceed the ε or δ budget.
    pub fn try_spend(&mut self, epsilon: f64, delta: f64) -> PrivacyResult<()> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if !(0.0..1.0).contains(&delta) {
            return Err(PrivacyError::InvalidDelta(delta));
        }

        // Tentative accumulator state after this query.
        let new_sum_eps = self.sum_epsilon + epsilon;
        let new_sum_del = self.sum_delta + delta;
        let new_drift = self.sum_drift + epsilon * (epsilon.exp() - 1.0);
        let new_eps_sq = self.sum_epsilon_sq + epsilon * epsilon;
        let new_count = self.count + 1;

        let (proj_eps, proj_del) =
            self.spend_of(new_sum_eps, new_sum_del, new_drift, new_eps_sq, new_count);

        // Small tolerances admit exact-boundary spends despite float rounding.
        let eps_tol = 1e-12 * (1.0 + self.total_epsilon.abs());
        let del_tol = 1e-15 + 1e-12 * self.total_delta.abs();
        if proj_eps > self.total_epsilon + eps_tol {
            return Err(PrivacyError::BudgetExhausted {
                spent: proj_eps,
                total: self.total_epsilon,
            });
        }
        if proj_del > self.total_delta + del_tol {
            return Err(PrivacyError::BudgetExhausted {
                spent: proj_del,
                total: self.total_delta,
            });
        }

        // Commit.
        self.sum_epsilon = new_sum_eps;
        self.sum_delta = new_sum_del;
        self.sum_drift = new_drift;
        self.sum_epsilon_sq = new_eps_sq;
        self.count = new_count;
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // (a) spending within budget succeeds and decrements remaining correctly.
    #[test]
    fn spend_within_budget_decrements() {
        let mut m = BudgetMonitor::new(1.0, 1e-5).expect("new");
        m.try_spend(0.3, 1e-6).expect("spend");
        let (se, sd) = m.spent();
        assert!((se - 0.3).abs() < 1e-12, "spent ε {se}");
        assert!((sd - 1e-6).abs() < 1e-15, "spent δ {sd}");
        let (re, rd) = m.remaining();
        assert!((re - 0.7).abs() < 1e-12, "remaining ε {re}");
        assert!((rd - 9e-6).abs() < 1e-15, "remaining δ {rd}");
        assert_eq!(m.count(), 1);
    }

    // (b) the step crossing the total is refused and does NOT commit.
    #[test]
    fn crossing_step_refused_no_commit() {
        let mut m = BudgetMonitor::new(1.0, 1e-5).expect("new");
        m.try_spend(0.6, 0.0).expect("first");
        let before = m.spent();
        let err = m.try_spend(0.6, 0.0);
        assert!(matches!(err, Err(PrivacyError::BudgetExhausted { .. })));
        let after = m.spent();
        assert!(
            (before.0 - after.0).abs() < 1e-15 && (before.1 - after.1).abs() < 1e-15,
            "spent must be unchanged after a refused spend: {before:?} vs {after:?}"
        );
        assert_eq!(m.count(), 1, "count unchanged after refusal");
    }

    // (c) an exact-boundary spend is allowed; the next is refused.
    #[test]
    fn exact_boundary_allowed() {
        let mut m = BudgetMonitor::new(1.0, 1e-5).expect("new");
        // 0.25 is exactly representable; four of them sum to exactly 1.0.
        for _ in 0..4 {
            m.try_spend(0.25, 0.0).expect("boundary spend");
        }
        let (se, _) = m.spent();
        assert!(
            (se - 1.0).abs() < 1e-12,
            "should be exactly at budget, got {se}"
        );
        assert!(
            m.try_spend(0.25, 0.0).is_err(),
            "spending past full budget must fail"
        );
    }

    // δ budget is enforced independently of ε.
    #[test]
    fn delta_budget_enforced() {
        let mut m = BudgetMonitor::new(10.0, 1e-6).expect("new");
        m.try_spend(0.1, 8e-7).expect("ok");
        // ε is fine but δ would exceed.
        let err = m.try_spend(0.1, 5e-7);
        assert!(matches!(err, Err(PrivacyError::BudgetExhausted { .. })));
        assert_eq!(m.count(), 1);
    }

    // (d) advanced composition gives a smaller cumulative ε than the naive sum.
    #[test]
    fn advanced_beats_basic_for_many_small_steps() {
        let mut basic = BudgetMonitor::new(10.0, 1e-3).expect("basic");
        let mut advanced = BudgetMonitor::new_advanced(10.0, 1e-3, 1e-6).expect("advanced");
        let step_eps = 0.05;
        let step_del = 1e-7;
        for _ in 0..100 {
            basic.try_spend(step_eps, step_del).expect("basic spend");
            advanced
                .try_spend(step_eps, step_del)
                .expect("advanced spend");
        }
        assert_eq!(basic.count(), 100);
        assert_eq!(advanced.count(), 100);
        let eps_basic = basic.spent().0;
        let eps_advanced = advanced.spent().0;
        assert!(
            (eps_basic - 5.0).abs() < 1e-9,
            "basic ε should be 5.0, got {eps_basic}"
        );
        assert!(
            eps_advanced < eps_basic,
            "advanced ε {eps_advanced} should be < basic ε {eps_basic}"
        );
    }

    // (e) invalid totals are rejected.
    #[test]
    fn invalid_totals_rejected() {
        assert!(matches!(
            BudgetMonitor::new(0.0, 1e-5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            BudgetMonitor::new(-1.0, 1e-5),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            BudgetMonitor::new(1.0, -0.1),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(matches!(
            BudgetMonitor::new(1.0, 1.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(BudgetMonitor::new(1.0, 1.5).is_err());
    }

    // Advanced-mode slack validation.
    #[test]
    fn advanced_slack_validation() {
        // slack must be strictly less than total_delta and > 0.
        assert!(BudgetMonitor::new_advanced(1.0, 1e-5, 0.0).is_err());
        assert!(BudgetMonitor::new_advanced(1.0, 1e-5, 1e-5).is_err());
        assert!(BudgetMonitor::new_advanced(1.0, 1e-5, 2e-5).is_err());
        // advanced needs a positive δ budget for the slack.
        assert!(BudgetMonitor::new_advanced(1.0, 0.0, 0.0).is_err());
        assert!(BudgetMonitor::new_advanced(1.0, 1e-5, 1e-6).is_ok());
    }

    // try_spend rejects malformed per-query parameters.
    #[test]
    fn try_spend_rejects_bad_params() {
        let mut m = BudgetMonitor::new(1.0, 1e-5).expect("new");
        assert!(matches!(
            m.try_spend(0.0, 0.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            m.try_spend(-0.1, 0.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            m.try_spend(0.1, -0.1),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(matches!(
            m.try_spend(0.1, 1.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        // δ = 0 (pure-ε query) is allowed.
        assert!(m.try_spend(0.1, 0.0).is_ok());
    }

    // A fresh advanced monitor reports zero spend and full remaining.
    #[test]
    fn advanced_fresh_state() {
        let m = BudgetMonitor::new_advanced(2.0, 1e-4, 1e-6).expect("new");
        let (se, sd) = m.spent();
        assert!(
            se.abs() < 1e-15 && sd.abs() < 1e-15,
            "fresh spend must be 0"
        );
        let (re, rd) = m.remaining();
        assert!((re - 2.0).abs() < 1e-12 && (rd - 1e-4).abs() < 1e-15);
        assert!(matches!(m.mode(), CompositionMode::Advanced { .. }));
    }
}
