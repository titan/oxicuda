//! Adaptive Sparse Vector Technique (SVT) with budget-aware thresholding.
//!
//! # References
//! - Dwork C, Naor M, Reingold O, Rothblum GN, Vadhan S (2009),
//!   *"On the Complexity of Differentially Private Data Release"*, STOC 2009 —
//!   the original Sparse Vector Technique on which the adaptive variant builds.
//! - Lyu M, Su D, Li N (2017),
//!   *"Understanding the Sparse Vector Technique for Differential Privacy"*,
//!   PVLDB 10(6):637–648, §3 — the budget-aware allocation
//!   `noise_scale = 4·k·Δ / ε_query` used here, together with the one-time
//!   threshold noise `Lap(2·Δ / ε_threshold)`.
//! - Kaplan H, Mansour Y, Nissim K (2023),
//!   *"Sparse-Vector with Adaptive Thresholding"*, ALT 2023 — the **soft**
//!   adaptation rule used in this implementation: a public-state moving
//!   average of *unprivatised* query values is tracked for diagnostics
//!   while the comparison `q̃ ≥ T̃` continues to use the one-time noisy
//!   threshold (preserving the SVT privacy guarantee).
//!
//! # Algorithm
//!
//! ```text
//!     ε = ε_T + ε_Q
//!     ε_T = threshold_budget_frac · ε                   (1)
//!     ε_Q = (1 − threshold_budget_frac) · ε              (2)
//!     T̃   = T₀ + Lap(2·Δ / ε_T)                          (3, one-time)
//!     q̃ⱼ  = qⱼ + Lap(4·k·Δ / ε_Q)                        (4, per query)
//!     respondⱼ = q̃ⱼ ≥ T̃                                  (5)
//!     T_{j+1} = T_j + adapt_rate · (qⱼ − T_j) if responded  (soft, public)
//! ```
//!
//! The total privacy cost is `(ε, 0)`-DP since
//! `Lap(4·k·Δ / ε_Q)` is the canonical noise scale that admits at most
//! `k` `True` answers under the Lyu-Su-Li accounting.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration of the adaptive SVT.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveSvtConfig {
    /// Total privacy budget `ε > 0` split between threshold and query noise.
    pub epsilon_total: f64,
    /// Global sensitivity `Δ > 0` of each query function.
    pub sensitivity: f64,
    /// Cap on `True` responses `k ≥ 1`.
    pub max_responses: usize,
    /// Soft threshold update rate, `adapt_rate ∈ [0, 1]`.
    ///
    /// - `0.0` — disables adaptation (classical SVT behaviour).
    /// - `1.0` — replaces the public threshold with the most recent
    ///   above-threshold raw query value.
    pub adapt_rate: f64,
    /// Fraction of `epsilon_total` allocated to the noisy threshold; must
    /// lie strictly in `(0, 1)`.
    pub threshold_budget_frac: f64,
}

impl AdaptiveSvtConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` when `epsilon_total` is non-positive or
    ///   non-finite.
    /// - `NonPositiveSensitivity` when `sensitivity` is non-positive or
    ///   non-finite.
    /// - `InvalidParameter` when `max_responses == 0`, `adapt_rate ∉ [0,1]`,
    ///   or `threshold_budget_frac ∉ (0,1)` (or non-finite).
    pub fn validate(&self) -> PrivacyResult<()> {
        if !(self.epsilon_total.is_finite() && self.epsilon_total > 0.0) {
            return Err(PrivacyError::NonPositiveEpsilon(self.epsilon_total));
        }
        if !(self.sensitivity.is_finite() && self.sensitivity > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(self.sensitivity));
        }
        if self.max_responses == 0 {
            return Err(PrivacyError::InvalidParameter(
                "max_responses must be ≥ 1".into(),
            ));
        }
        if !self.adapt_rate.is_finite() || !(0.0..=1.0).contains(&self.adapt_rate) {
            return Err(PrivacyError::InvalidParameter(format!(
                "adapt_rate must lie in [0, 1] and be finite, got {}",
                self.adapt_rate
            )));
        }
        if !self.threshold_budget_frac.is_finite()
            || self.threshold_budget_frac <= 0.0
            || self.threshold_budget_frac >= 1.0
        {
            return Err(PrivacyError::InvalidParameter(format!(
                "threshold_budget_frac must lie in (0, 1) and be finite, got {}",
                self.threshold_budget_frac
            )));
        }
        Ok(())
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Mutable session state of the adaptive SVT.
#[derive(Debug, Clone)]
pub struct AdaptiveSvtState {
    /// Public-state moving average of the most-recent above-threshold raw
    /// query values (diagnostic only, **not** used in the privacy
    /// comparison).
    pub current_threshold: f64,
    /// One-time noisy threshold `T̃ = T₀ + Lap(2·Δ / ε_T)` used in every
    /// per-query comparison.
    pub noisy_threshold: f64,
    /// Number of additional `True` responses still allowed.
    pub remaining_responses: usize,
    /// Initial query budget `ε_Q = (1 − threshold_budget_frac) · ε_total`
    /// (kept constant — the per-query Laplace scale never changes).
    pub query_budget: f64,
    /// Sequence of boolean responses emitted so far.
    pub responses: Vec<bool>,
    /// Total queries processed (including post-budget no-ops).
    pub queries_seen: usize,
}

// ─── Adaptive SVT ──────────────────────────────────────────────────────────────

/// Stateless adaptive SVT mechanism — holds only the configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveSvt {
    /// Active validated configuration.
    pub cfg: AdaptiveSvtConfig,
}

impl AdaptiveSvt {
    /// Construct an adaptive SVT and initialise its session state.
    ///
    /// Draws the one-time noisy threshold from
    /// `Lap(2·Δ / ε_T)` where `ε_T = threshold_budget_frac · ε_total`.
    ///
    /// # Errors
    /// - Validation errors from [`AdaptiveSvtConfig::validate`].
    /// - `InvalidParameter` if `initial_threshold` is non-finite.
    pub fn new(
        cfg: AdaptiveSvtConfig,
        initial_threshold: f64,
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<(Self, AdaptiveSvtState)> {
        cfg.validate()?;
        if !initial_threshold.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "initial_threshold must be finite, got {initial_threshold}"
            )));
        }
        let epsilon_t = cfg.threshold_budget_frac * cfg.epsilon_total;
        let query_budget = (1.0 - cfg.threshold_budget_frac) * cfg.epsilon_total;
        let threshold_scale = 2.0 * cfg.sensitivity / epsilon_t;
        let noise = handle.generate_laplace_noise(threshold_scale, 1)?;
        let noisy_threshold = initial_threshold + noise[0];
        let state = AdaptiveSvtState {
            current_threshold: initial_threshold,
            noisy_threshold,
            remaining_responses: cfg.max_responses,
            query_budget,
            responses: Vec::new(),
            queries_seen: 0,
        };
        Ok((Self { cfg }, state))
    }

    /// Process a single query value, returning the boolean response.
    ///
    /// Once `remaining_responses == 0`, every subsequent call returns
    /// `Ok(false)` and only increments `queries_seen` — no further noise
    /// is drawn, so no additional privacy budget is consumed.
    ///
    /// # Errors
    /// `InvalidParameter` when `query_value` is non-finite.
    pub fn process_query(
        &self,
        state: &mut AdaptiveSvtState,
        query_value: f64,
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<bool> {
        if !query_value.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "query_value must be finite, got {query_value}"
            )));
        }
        state.queries_seen += 1;
        if state.remaining_responses == 0 {
            state.responses.push(false);
            return Ok(false);
        }
        let k = self.cfg.max_responses as f64;
        // Lyu-Su-Li §3: per-query Laplace noise scale `4·k·Δ / ε_Q`
        // (the factor 4 accounts for the two-sided query/threshold
        // comparison plus the worst-case sensitivity in the streaming
        // setting).
        let query_scale = 4.0 * k * self.cfg.sensitivity / state.query_budget;
        let noise = handle.generate_laplace_noise(query_scale, 1)?;
        let noisy_query = query_value + noise[0];
        let above = noisy_query >= state.noisy_threshold;
        if above {
            state.remaining_responses -= 1;
            // Soft adaptation of the **public** threshold tracker
            // (diagnostics only — does not affect the privacy proof).
            state.current_threshold +=
                self.cfg.adapt_rate * (query_value - state.current_threshold);
        }
        state.responses.push(above);
        Ok(above)
    }

    /// Borrow the active validated configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveSvtConfig {
        &self.cfg
    }

    /// Total privacy budget `ε_T + ε_Q == ε_total`.
    #[must_use]
    pub fn total_epsilon(&self) -> f64 {
        self.cfg.epsilon_total
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(
        epsilon_total: f64,
        sensitivity: f64,
        max_responses: usize,
        adapt_rate: f64,
        threshold_budget_frac: f64,
    ) -> AdaptiveSvtConfig {
        AdaptiveSvtConfig {
            epsilon_total,
            sensitivity,
            max_responses,
            adapt_rate,
            threshold_budget_frac,
        }
    }

    // 1. Exhausting max_responses halts True responses.
    #[test]
    fn test_max_responses_halts_true_responses() {
        let cfg = make_cfg(2.0, 1.0, 3, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 7);
        let (svt, mut state) = AdaptiveSvt::new(cfg, -1_000_000.0, &mut handle).expect("new");
        // Use very large queries so each is almost surely above the noisy
        // threshold once it's drawn.
        let mut true_count = 0usize;
        for _ in 0..20 {
            let resp = svt
                .process_query(&mut state, 1_000.0, &mut handle)
                .expect("ok");
            if resp {
                true_count += 1;
            }
        }
        assert!(
            true_count <= cfg.max_responses,
            "exceeded cap: {true_count} > {}",
            cfg.max_responses
        );
        assert_eq!(state.remaining_responses, 0);
        // Subsequent calls return false without erroring.
        let post = svt
            .process_query(&mut state, 1_000.0, &mut handle)
            .expect("ok");
        assert!(!post);
    }

    // 2. adapt_rate = 0 reduces to standard SVT (threshold static).
    #[test]
    fn test_adapt_rate_zero_static_threshold() {
        let cfg = make_cfg(4.0, 1.0, 5, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 13);
        let initial_threshold = 5.0;
        let (svt, mut state) = AdaptiveSvt::new(cfg, initial_threshold, &mut handle).expect("new");
        for _ in 0..10 {
            let _ = svt
                .process_query(&mut state, 100.0, &mut handle)
                .expect("ok");
        }
        assert!(
            (state.current_threshold - initial_threshold).abs() < 1e-12,
            "current_threshold drifted with adapt_rate=0: {}",
            state.current_threshold
        );
    }

    // 3. cfg.epsilon_total ≤ 0 → NonPositiveEpsilon.
    #[test]
    fn test_validation_rejects_bad_epsilon() {
        let mut handle = PrivacyHandle::new(80, 0);
        let r = AdaptiveSvt::new(make_cfg(0.0, 1.0, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveEpsilon(_))));
        let r = AdaptiveSvt::new(make_cfg(-1.0, 1.0, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveEpsilon(_))));
        let r = AdaptiveSvt::new(make_cfg(f64::NAN, 1.0, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveEpsilon(_))));
    }

    // 4. cfg.sensitivity ≤ 0 → NonPositiveSensitivity.
    #[test]
    fn test_validation_rejects_bad_sensitivity() {
        let mut handle = PrivacyHandle::new(80, 0);
        let r = AdaptiveSvt::new(make_cfg(1.0, 0.0, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveSensitivity(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, -1.0, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveSensitivity(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, f64::NAN, 1, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::NonPositiveSensitivity(_))));
    }

    // 5. cfg.max_responses == 0 → InvalidParameter.
    #[test]
    fn test_validation_rejects_zero_max_responses() {
        let mut handle = PrivacyHandle::new(80, 0);
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 0, 0.0, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 6. cfg.adapt_rate out of [0, 1] → InvalidParameter.
    #[test]
    fn test_validation_rejects_bad_adapt_rate() {
        let mut handle = PrivacyHandle::new(80, 0);
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, -0.1, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, 1.5, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, f64::NAN, 0.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 7. cfg.threshold_budget_frac out of (0, 1) → InvalidParameter.
    #[test]
    fn test_validation_rejects_bad_threshold_budget_frac() {
        let mut handle = PrivacyHandle::new(80, 0);
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, 0.0, 0.0), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, 0.0, 1.0), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, 0.0, -0.1), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(make_cfg(1.0, 1.0, 1, 0.0, 1.5), 0.0, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 8. Deterministic with fixed RNG seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = make_cfg(2.0, 1.0, 5, 0.3, 0.5);
        let mut h_a = PrivacyHandle::new(80, 4242);
        let mut h_b = PrivacyHandle::new(80, 4242);
        let (svt_a, mut state_a) = AdaptiveSvt::new(cfg, 0.0, &mut h_a).expect("a");
        let (svt_b, mut state_b) = AdaptiveSvt::new(cfg, 0.0, &mut h_b).expect("b");
        let queries = [3.0, -2.0, 7.5, 4.0, 11.0, -3.0, 8.0];
        for &q in &queries {
            let ra = svt_a.process_query(&mut state_a, q, &mut h_a).expect("a");
            let rb = svt_b.process_query(&mut state_b, q, &mut h_b).expect("b");
            assert_eq!(ra, rb);
        }
        assert!((state_a.noisy_threshold - state_b.noisy_threshold).abs() < 1e-15);
        assert!((state_a.current_threshold - state_b.current_threshold).abs() < 1e-15);
        assert_eq!(state_a.responses, state_b.responses);
    }

    // 9. Large positive queries return True with high probability.
    #[test]
    fn test_large_positive_queries_return_true() {
        let cfg = make_cfg(20.0, 1.0, 4, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 11);
        let (svt, mut state) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        let r1 = svt
            .process_query(&mut state, 1.0e6, &mut handle)
            .expect("ok");
        let r2 = svt
            .process_query(&mut state, 1.0e6, &mut handle)
            .expect("ok");
        assert!(r1, "first large query should be True");
        assert!(r2, "second large query should be True");
    }

    // 10. Large negative queries return False.
    #[test]
    fn test_large_negative_queries_return_false() {
        let cfg = make_cfg(20.0, 1.0, 4, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 11);
        let (svt, mut state) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        for _ in 0..5 {
            let r = svt
                .process_query(&mut state, -1.0e6, &mut handle)
                .expect("ok");
            assert!(!r, "large negative query should be False");
        }
        // No `True` responses ⇒ budget unconsumed.
        assert_eq!(state.remaining_responses, cfg.max_responses);
    }

    // 11. current_threshold drifts toward observed query mean.
    #[test]
    fn test_current_threshold_drifts_toward_mean() {
        let cfg = make_cfg(20.0, 1.0, 10, 0.5, 0.5);
        let mut handle = PrivacyHandle::new(80, 21);
        let initial_threshold = 0.0;
        let (svt, mut state) = AdaptiveSvt::new(cfg, initial_threshold, &mut handle).expect("new");
        // Queries all roughly 100; with high ε noise is tiny so each is True.
        let target = 100.0;
        for _ in 0..6 {
            let _ = svt
                .process_query(&mut state, target, &mut handle)
                .expect("ok");
        }
        assert!(
            state.current_threshold > 0.5 * target,
            "current_threshold did not drift toward {target}: {}",
            state.current_threshold
        );
        assert!(state.current_threshold <= target + 1.0);
    }

    // 12. responses.len() == queries_seen (always).
    #[test]
    fn test_responses_length_matches_queries_seen() {
        let cfg = make_cfg(2.0, 1.0, 2, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 33);
        let (svt, mut state) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        for q in [10.0, -5.0, 1.0, 7.0, -100.0, 50.0].iter() {
            let _ = svt.process_query(&mut state, *q, &mut handle).expect("ok");
        }
        assert_eq!(state.responses.len(), state.queries_seen);
        assert_eq!(state.queries_seen, 6);
    }

    // 13. True count ≤ max_responses (invariant).
    #[test]
    fn test_true_count_does_not_exceed_max() {
        let cfg = make_cfg(5.0, 1.0, 2, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 99);
        let (svt, mut state) = AdaptiveSvt::new(cfg, -1.0e6, &mut handle).expect("new");
        let mut true_count = 0usize;
        for _ in 0..50 {
            let r = svt
                .process_query(&mut state, 1.0e6, &mut handle)
                .expect("ok");
            if r {
                true_count += 1;
            }
        }
        assert!(
            true_count <= cfg.max_responses,
            "{true_count} > {}",
            cfg.max_responses
        );
    }

    // 14. Non-finite query → InvalidParameter.
    #[test]
    fn test_non_finite_query_value_errors() {
        let cfg = make_cfg(1.0, 1.0, 1, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 1);
        let (svt, mut state) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        let r = svt.process_query(&mut state, f64::NAN, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = svt.process_query(&mut state, f64::INFINITY, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 15. Non-finite initial_threshold → InvalidParameter.
    #[test]
    fn test_non_finite_initial_threshold_errors() {
        let cfg = make_cfg(1.0, 1.0, 1, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 1);
        let r = AdaptiveSvt::new(cfg, f64::NAN, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = AdaptiveSvt::new(cfg, f64::INFINITY, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 16. config() accessor returns the input config.
    #[test]
    fn test_config_accessor_returns_input() {
        let cfg = make_cfg(1.5, 2.0, 4, 0.25, 0.6);
        let mut handle = PrivacyHandle::new(80, 0);
        let (svt, _) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        let back = svt.config();
        assert!((back.epsilon_total - cfg.epsilon_total).abs() < 1e-12);
        assert!((back.sensitivity - cfg.sensitivity).abs() < 1e-12);
        assert_eq!(back.max_responses, cfg.max_responses);
        assert!((back.adapt_rate - cfg.adapt_rate).abs() < 1e-12);
        assert!((back.threshold_budget_frac - cfg.threshold_budget_frac).abs() < 1e-12);
        assert!((svt.total_epsilon() - cfg.epsilon_total).abs() < 1e-12);
    }

    // 17. query_budget equals (1 − threshold_budget_frac) · ε_total.
    #[test]
    fn test_query_budget_split() {
        let cfg = make_cfg(2.0, 1.0, 4, 0.0, 0.25);
        let mut handle = PrivacyHandle::new(80, 0);
        let (_svt, state) = AdaptiveSvt::new(cfg, 0.0, &mut handle).expect("new");
        // (1 − 0.25) · 2.0 = 1.5
        assert!((state.query_budget - 1.5).abs() < 1e-12);
    }

    // 18. Post-budget queries still increment queries_seen without
    //     consuming additional Laplace draws (no noise generated).
    #[test]
    fn test_post_budget_queries_no_op() {
        let cfg = make_cfg(5.0, 1.0, 1, 0.0, 0.5);
        let mut handle = PrivacyHandle::new(80, 5);
        let (svt, mut state) = AdaptiveSvt::new(cfg, -1.0e6, &mut handle).expect("new");
        // Force the single True.
        let r = svt
            .process_query(&mut state, 1.0e6, &mut handle)
            .expect("ok");
        assert!(r);
        assert_eq!(state.remaining_responses, 0);
        let queries_before = state.queries_seen;
        // Capture handle RNG state.
        let mut rng_before = handle.rng.clone();
        let r2 = svt
            .process_query(&mut state, 1.0e6, &mut handle)
            .expect("ok");
        assert!(!r2);
        assert_eq!(state.queries_seen, queries_before + 1);
        // RNG state should be unchanged (no Laplace draw).
        assert_eq!(rng_before.next_f64(), handle.rng.next_f64());
    }
}
