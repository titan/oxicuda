//! Sparse-Vector Technique composed across **multiple parallel streams**.
//!
//! References:
//! - Dwork & Roth (2014), "The Algorithmic Foundations of Differential
//!   Privacy", §3.6 (AboveThreshold / SVT) and Theorem 3.20 (advanced
//!   composition).
//! - Lyu, Su & Li (2017), "Understanding the Sparse Vector Technique for
//!   Differential Privacy", PVLDB — correct SVT budgeting.
//!
//! The single-stream SVT lives in [`crate::selection::sparse_vector`].  In
//! practice one often runs *several* SVT instances at once — e.g. one
//! above-threshold monitor per feature, per shard, or per metric — over the
//! same sensitive dataset.  Because the streams touch the same data, their
//! privacy costs **compose**.  This module:
//!
//! 1. Holds `S` independent [`crate::selection::sparse_vector::SvtState`]
//!    streams, each with its own per-stream budget `ε_s` and `k_s`-True cap.
//! 2. Routes a query to a chosen stream and returns its above-threshold
//!    indicator, halting that stream after its `k_s` True answers.
//! 3. Reports the **composed** total privacy cost across all streams under
//!    either basic composition (`ε_total = Σ_s ε_s`) or advanced composition
//!    (Dwork–Rothblum–Vadhan), which for many small-`ε_s` streams is far tighter.
//!
//! # Privacy
//! Each stream `s` is `(ε_s, 0)`-DP by the SVT guarantee.  Running all `S`
//! streams is the (adaptive) composition of `S` mechanisms with parameters
//! `(ε_s, 0)`, so the released transcript of *all* streams is
//! `(Σ_s ε_s, 0)`-DP under basic composition, or
//! `(ε', δ')`-DP under advanced composition (which trades a small `δ'` for a
//! sub-linear `ε'`).

use crate::composition::advanced::{CompositionResult, basic_compose, strong_compose};
use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;
use crate::selection::sparse_vector::{SvtConfig, SvtState};

/// Composition mode for multi-stream SVT total-budget accounting.
#[derive(Debug, Clone, Copy)]
pub enum SvtCompositionMode {
    /// Basic linear composition: `ε_total = Σ_s ε_s` (and `δ = 0`).
    Basic,
    /// Advanced (Dwork–Rothblum–Vadhan) composition with reserved slack `δ'`.
    Advanced {
        /// Failure-probability slack `δ' ∈ (0, 1)`.
        slack_delta: f64,
    },
}

/// A collection of independent SVT streams sharing one sensitive dataset.
pub struct MultiStreamSvt {
    configs: Vec<SvtConfig>,
    states: Vec<SvtState>,
}

impl MultiStreamSvt {
    /// Initialise one SVT stream per supplied config (each draws its own noisy
    /// threshold from `rng`).
    ///
    /// # Errors
    /// - `EmptyInput` if `configs` is empty.
    /// - Propagates `SvtState::new` errors.
    pub fn new(configs: Vec<SvtConfig>, rng: &mut LcgRng) -> PrivacyResult<Self> {
        if configs.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let mut states = Vec::with_capacity(configs.len());
        for cfg in &configs {
            states.push(SvtState::new(cfg, rng)?);
        }
        Ok(Self { configs, states })
    }

    /// Number of streams.
    #[must_use]
    pub fn num_streams(&self) -> usize {
        self.configs.len()
    }

    /// True answers returned so far by stream `s`.
    ///
    /// # Errors
    /// - `IndexOutOfRange` if `s ≥ num_streams`.
    pub fn answered(&self, s: usize) -> PrivacyResult<usize> {
        self.states
            .get(s)
            .map(|st| st.answered)
            .ok_or(PrivacyError::IndexOutOfRange(s, self.states.len()))
    }

    /// Route a query value to stream `s` and return its above-threshold
    /// indicator (`Some(true/false)`), or `Some` behaviour per
    /// [`SvtState::query`].  Returns `Ok(None)` if the stream has already
    /// exhausted its `k_s`-True budget (the stream halts gracefully rather than
    /// erroring).
    ///
    /// # Errors
    /// - `IndexOutOfRange` if `s ≥ num_streams`.
    pub fn query(
        &mut self,
        s: usize,
        query_val: f64,
        rng: &mut LcgRng,
    ) -> PrivacyResult<Option<bool>> {
        if s >= self.configs.len() {
            return Err(PrivacyError::IndexOutOfRange(s, self.configs.len()));
        }
        let cfg = self.configs[s].clone();
        let state = &mut self.states[s];
        if state.answered >= cfg.k {
            return Ok(None);
        }
        state.query(query_val, &cfg, rng)
    }

    /// The per-stream privacy parameters `(ε_s, 0)`.
    #[must_use]
    pub fn stream_budgets(&self) -> Vec<f64> {
        self.configs.iter().map(|c| c.epsilon).collect()
    }

    /// Composed total privacy cost across all streams.
    ///
    /// - `Basic`: `(Σ_s ε_s, 0)`.
    /// - `Advanced { δ' }`: applies strong composition treating all streams as
    ///   the *worst-case* per-mechanism `ε₀ = maxₛ ε_s` over `k = S` mechanisms;
    ///   when the per-stream budgets are heterogeneous this is a valid upper
    ///   bound (each stream's true cost ≤ `ε₀`).
    ///
    /// # Errors
    /// - `InvalidParameter` if `Advanced.slack_delta ∉ (0, 1)`.
    pub fn total_budget(&self, mode: SvtCompositionMode) -> PrivacyResult<CompositionResult> {
        match mode {
            SvtCompositionMode::Basic => {
                let total_eps: f64 = self.configs.iter().map(|c| c.epsilon).sum();
                Ok(CompositionResult::new(total_eps, 0.0))
            }
            SvtCompositionMode::Advanced { slack_delta } => {
                if !(slack_delta > 0.0 && slack_delta < 1.0) {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "slack_delta must be in (0,1), got {slack_delta}"
                    )));
                }
                let eps0 = self
                    .configs
                    .iter()
                    .map(|c| c.epsilon)
                    .fold(0.0f64, f64::max);
                let k = self.configs.len();
                strong_compose(eps0, 0.0, k, slack_delta)
            }
        }
    }

    /// Convenience: basic-composition total via [`basic_compose`] using the
    /// (homogeneous) assumption `ε₀ = maxₛ ε_s`.  Mainly for comparison with
    /// the exact heterogeneous sum from [`Self::total_budget`].
    #[must_use]
    pub fn basic_budget_homogeneous(&self) -> CompositionResult {
        let eps0 = self
            .configs
            .iter()
            .map(|c| c.epsilon)
            .fold(0.0f64, f64::max);
        basic_compose(eps0, 0.0, self.configs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_streams() -> Vec<SvtConfig> {
        vec![
            SvtConfig::new(0.5, 0.0, 2, 1.0).expect("s0"),
            SvtConfig::new(0.5, 1.0, 2, 1.0).expect("s1"),
            SvtConfig::new(1.0, -1.0, 3, 1.0).expect("s2"),
        ]
    }

    #[test]
    fn test_construct_and_count() {
        let mut rng = LcgRng::new(7);
        let m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
        assert_eq!(m.num_streams(), 3);
        for s in 0..3 {
            assert_eq!(m.answered(s).expect("a"), 0);
        }
    }

    #[test]
    fn test_streams_are_independent() {
        // Driving stream 0 to its True-cap must not halt stream 1.
        let mut rng = LcgRng::new(11);
        let mut m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
        // Stream 0 cap is 2; force True with a huge value until None.
        let mut s0_true = 0;
        for _ in 0..20 {
            match m.query(0, 1e9, &mut rng).expect("q") {
                Some(true) => s0_true += 1,
                Some(false) => {}
                None => break,
            }
        }
        assert!(s0_true <= 2, "stream 0 returned {s0_true} True > cap 2");
        // Stream 1 should still respond (not halted by stream 0).
        let r = m.query(1, 1e9, &mut rng).expect("q1");
        assert!(r.is_some(), "stream 1 should still be active");
    }

    #[test]
    fn test_basic_total_is_sum_of_epsilons() {
        let mut rng = LcgRng::new(3);
        let m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
        let total = m.total_budget(SvtCompositionMode::Basic).expect("total");
        assert!(
            (total.epsilon - (0.5 + 0.5 + 1.0)).abs() < 1e-12,
            "basic total ε={} should equal 2.0",
            total.epsilon
        );
        assert!((total.delta - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_advanced_total_tighter_for_many_small_streams() {
        // Many small-ε streams: advanced composition should beat the linear sum.
        let configs: Vec<SvtConfig> = (0..50)
            .map(|_| SvtConfig::new(0.1, 0.0, 1, 1.0).expect("s"))
            .collect();
        let mut rng = LcgRng::new(99);
        let m = MultiStreamSvt::new(configs, &mut rng).expect("multi");
        let basic = m.total_budget(SvtCompositionMode::Basic).expect("basic");
        let advanced = m
            .total_budget(SvtCompositionMode::Advanced { slack_delta: 1e-6 })
            .expect("adv");
        // basic ε = 50·0.1 = 5.0.
        assert!((basic.epsilon - 5.0).abs() < 1e-9);
        assert!(
            advanced.epsilon < basic.epsilon,
            "advanced ε={} should be < basic ε={}",
            advanced.epsilon,
            basic.epsilon
        );
        assert!(advanced.delta > 0.0, "advanced reserves δ' slack");
    }

    #[test]
    fn test_query_out_of_range_errors() {
        let mut rng = LcgRng::new(5);
        let mut m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
        assert!(m.query(99, 1.0, &mut rng).is_err());
        assert!(m.answered(99).is_err());
    }

    #[test]
    fn test_empty_configs_error_and_bad_slack() {
        let mut rng = LcgRng::new(1);
        assert!(MultiStreamSvt::new(vec![], &mut rng).is_err());
        let m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
        assert!(
            m.total_budget(SvtCompositionMode::Advanced { slack_delta: 0.0 })
                .is_err()
        );
    }

    #[test]
    fn test_determinism_same_seed() {
        let run = || {
            let mut rng = LcgRng::new(2024);
            let mut m = MultiStreamSvt::new(make_streams(), &mut rng).expect("multi");
            let mut out = Vec::new();
            for i in 0..15 {
                let s = i % 3;
                out.push(m.query(s, (i as f64) - 5.0, &mut rng).expect("q"));
            }
            out
        };
        assert_eq!(run(), run());
    }
}
