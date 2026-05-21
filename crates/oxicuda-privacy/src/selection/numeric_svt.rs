//! Numeric Sparse Vector Technique (SVT with noisy values).
//!
//! # Reference
//! - Lyu M, Su D, Li N (2017),
//!   *"Understanding the Sparse Vector Technique for Differential
//!   Privacy"*, PVLDB 10(6):637–648, Algorithm 3 ("SVT with Noisy
//!   Values"), <https://www.vldb.org/pvldb/vol10/p637-lyu.pdf>.
//!
//! Unlike the classical SVT (`selection::sparse_vector`) which only
//! releases a `true / false` indicator, the **numeric SVT** also
//! releases a noisy *value* for each above-threshold query, partitioning
//! its privacy budget across three independent Laplace noise channels:
//!
//! ```text
//!     ε = ε₁ + ε₂ + ε₃
//! ```
//!
//! - `ε₁` — single noisy threshold `T̃ = T + Lap(2Δ / ε₁)`.
//! - `ε₂` — split across `k` query comparisons as `Lap(4kΔ / ε₂)`.
//! - `ε₃` — split across `k` value releases as `Lap(2kΔ / ε₃)`.
//!
//! The mechanism halts after `k` above-threshold queries (responses).
//! Subsequent calls return `Ok(None)` without consuming any additional
//! budget.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration of the Numeric SVT.
#[derive(Debug, Clone)]
pub struct NumericSvtConfig {
    /// `ε₁ > 0` — budget for the noisy threshold.
    pub epsilon_threshold: f64,
    /// `ε₂ > 0` — total budget for the `k` query comparisons.
    pub epsilon_query: f64,
    /// `ε₃ > 0` — total budget for the `k` value releases.
    pub epsilon_value: f64,
    /// Threshold `T` for above-threshold detection.
    pub threshold: f64,
    /// Global sensitivity `Δ > 0` of each query.
    pub sensitivity: f64,
    /// Maximum number of above-threshold responses `k ≥ 1`.
    pub max_responses: usize,
}

impl NumericSvtConfig {
    /// Validate and construct a config.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if any `ε` is non-positive or non-finite.
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0` or non-finite.
    /// - `InvalidParameter` if `max_responses == 0`.
    pub fn new(
        epsilon_threshold: f64,
        epsilon_query: f64,
        epsilon_value: f64,
        threshold: f64,
        sensitivity: f64,
        max_responses: usize,
    ) -> PrivacyResult<Self> {
        for &(name, e) in &[
            ("epsilon_threshold", epsilon_threshold),
            ("epsilon_query", epsilon_query),
            ("epsilon_value", epsilon_value),
        ] {
            if !(e.is_finite() && e > 0.0) {
                return Err(PrivacyError::InvalidParameter(format!(
                    "{name} must be > 0 and finite, got {e}"
                )));
            }
        }
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if !threshold.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "threshold must be finite, got {threshold}"
            )));
        }
        if max_responses == 0 {
            return Err(PrivacyError::InvalidParameter(
                "max_responses must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            epsilon_threshold,
            epsilon_query,
            epsilon_value,
            threshold,
            sensitivity,
            max_responses,
        })
    }
}

/// Response emitted for an above-threshold query.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericSvtResponse {
    /// Zero-based index of the query in the stream order (auto-incremented).
    pub query_index: usize,
    /// Noisy released value `q + η`.
    pub value: f64,
    /// Raw input `q` (debug only; the value is *not* private and should
    /// not be logged in production).
    pub raw_query: f64,
}

/// Streaming Numeric SVT state.
///
/// One instance corresponds to one privacy session and may answer at
/// most `cfg.max_responses` above-threshold queries.
#[derive(Debug)]
pub struct NumericSvt {
    cfg: NumericSvtConfig,
    noisy_threshold: f64,
    returned: usize,
    next_index: usize,
}

impl NumericSvt {
    /// Initialise a Numeric SVT session by drawing the noisy threshold
    /// `T̃ = T + Lap(2Δ / ε₁)`.
    ///
    /// # Errors
    /// Propagates validation errors from the config.
    pub fn new(cfg: NumericSvtConfig, rng: &mut LcgRng) -> PrivacyResult<Self> {
        // Re-validate so a hand-constructed `NumericSvtConfig` cannot
        // bypass `new` and crash the sampler.
        let cfg = NumericSvtConfig::new(
            cfg.epsilon_threshold,
            cfg.epsilon_query,
            cfg.epsilon_value,
            cfg.threshold,
            cfg.sensitivity,
            cfg.max_responses,
        )?;
        let threshold_scale = 2.0 * cfg.sensitivity / cfg.epsilon_threshold;
        let noisy_threshold = cfg.threshold + laplace_sample(threshold_scale, rng);
        Ok(Self {
            cfg,
            noisy_threshold,
            returned: 0,
            next_index: 0,
        })
    }

    /// Reference to the validated configuration.
    #[must_use]
    pub fn config(&self) -> &NumericSvtConfig {
        &self.cfg
    }

    /// Process a single query value.
    ///
    /// Returns `Ok(None)` once the budget of `max_responses` is exhausted
    /// or when the noisy query falls below the noisy threshold; returns
    /// `Ok(Some(response))` for an above-threshold release with noisy
    /// value `q + η`.
    pub fn process_query(
        &mut self,
        q: f64,
        rng: &mut LcgRng,
    ) -> PrivacyResult<Option<NumericSvtResponse>> {
        if !q.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "query value must be finite, got {q}"
            )));
        }
        let index = self.next_index;
        self.next_index += 1;
        if self.returned >= self.cfg.max_responses {
            return Ok(None);
        }
        let k = self.cfg.max_responses as f64;
        let query_scale = 4.0 * k * self.cfg.sensitivity / self.cfg.epsilon_query;
        let nu = laplace_sample(query_scale, rng);
        if q + nu < self.noisy_threshold {
            return Ok(None);
        }
        let value_scale = 2.0 * k * self.cfg.sensitivity / self.cfg.epsilon_value;
        let eta = laplace_sample(value_scale, rng);
        let released = q + eta;
        self.returned += 1;
        Ok(Some(NumericSvtResponse {
            query_index: index,
            value: released,
            raw_query: q,
        }))
    }

    /// Process a batch of queries in stream order.
    ///
    /// Returns only the above-threshold responses; the length of the
    /// returned vector is `≤ min(queries.len(), max_responses)`.
    pub fn batch(
        &mut self,
        queries: &[f64],
        rng: &mut LcgRng,
    ) -> PrivacyResult<Vec<NumericSvtResponse>> {
        let mut out = Vec::with_capacity(queries.len().min(self.cfg.max_responses));
        for &q in queries {
            match self.process_query(q, rng)? {
                Some(resp) => out.push(resp),
                None => {
                    if self.returned >= self.cfg.max_responses {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Total `(ε, 0)`-DP budget: `ε₁ + ε₂ + ε₃`.
    #[must_use]
    pub fn total_epsilon(&self) -> f64 {
        self.cfg.epsilon_threshold + self.cfg.epsilon_query + self.cfg.epsilon_value
    }

    /// Number of additional above-threshold responses still allowed.
    #[must_use]
    pub fn remaining_budget(&self) -> usize {
        self.cfg.max_responses.saturating_sub(self.returned)
    }

    /// Number of responses released so far.
    #[must_use]
    pub fn responses_returned(&self) -> usize {
        self.returned
    }

    /// Number of queries processed so far (including below-threshold ones
    /// and post-budget no-ops).
    #[must_use]
    pub fn queries_processed(&self) -> usize {
        self.next_index
    }

    /// Read-only view of the noisy threshold (debug / testing).
    #[must_use]
    pub fn noisy_threshold(&self) -> f64 {
        self.noisy_threshold
    }
}

/// Sample `L ~ Lap(0, scale)` via the inverse CDF.
///
/// `scale > 0` must hold; the public API validates this through
/// `NumericSvtConfig::new`.
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    debug_assert!(scale > 0.0);
    let u = rng.next_f64() - 0.5;
    let abs_u = u.abs().min(0.5 - f64::EPSILON);
    -scale * u.signum() * (1.0 - 2.0 * abs_u).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_basic(threshold: f64, k: usize) -> NumericSvtConfig {
        NumericSvtConfig::new(0.5, 0.5, 0.5, threshold, 1.0, k).expect("ok")
    }

    #[test]
    fn test_config_validation_rejects_bad_epsilon() {
        assert!(NumericSvtConfig::new(0.0, 0.5, 0.5, 0.0, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(-0.1, 0.5, 0.5, 0.0, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, -1.0, 0.5, 0.0, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.0, 0.0, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, f64::NAN, 0.0, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, f64::INFINITY, 0.0, 1.0, 1).is_err());
    }

    #[test]
    fn test_config_validation_rejects_bad_sensitivity() {
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, 0.0, 0.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, 0.0, -1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, 0.0, f64::NAN, 1).is_err());
    }

    #[test]
    fn test_config_validation_rejects_zero_k() {
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, 0.0, 1.0, 0).is_err());
    }

    #[test]
    fn test_config_validation_rejects_bad_threshold() {
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, f64::NAN, 1.0, 1).is_err());
        assert!(NumericSvtConfig::new(0.5, 0.5, 0.5, f64::INFINITY, 1.0, 1).is_err());
    }

    #[test]
    fn test_above_threshold_returns_some_with_finite_value() {
        let cfg = cfg_basic(0.0, 3);
        let mut rng = LcgRng::new(42);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        // Very large query → almost surely above noisy threshold.
        let resp = svt.process_query(1_000.0, &mut rng).expect("ok");
        let resp = resp.expect("expected above-threshold release");
        assert!(resp.value.is_finite());
        assert_eq!(resp.query_index, 0);
        assert!((resp.raw_query - 1_000.0).abs() < 1e-12);
    }

    #[test]
    fn test_below_threshold_returns_none() {
        let cfg = cfg_basic(1_000_000.0, 3);
        let mut rng = LcgRng::new(7);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        // Very small query relative to threshold → returns None.
        let resp = svt.process_query(-1_000_000.0, &mut rng).expect("ok");
        assert!(resp.is_none());
        // Budget not consumed.
        assert_eq!(svt.responses_returned(), 0);
        assert_eq!(svt.remaining_budget(), 3);
    }

    #[test]
    fn test_budget_exhausts_at_max_responses() {
        let cfg = cfg_basic(-1_000_000.0, 2);
        let mut rng = LcgRng::new(11);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        let mut got = 0usize;
        for _ in 0..20 {
            if let Some(_resp) = svt.process_query(1_000.0, &mut rng).expect("ok") {
                got += 1;
            }
        }
        assert_eq!(got, 2);
        assert_eq!(svt.responses_returned(), 2);
        assert_eq!(svt.remaining_budget(), 0);
        // Further calls still succeed but return None.
        let post = svt.process_query(1_000.0, &mut rng).expect("ok");
        assert!(post.is_none());
    }

    #[test]
    fn test_total_epsilon_is_sum() {
        let cfg = NumericSvtConfig::new(0.1, 0.2, 0.3, 0.0, 1.0, 4).expect("ok");
        let mut rng = LcgRng::new(0);
        let svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        assert!((svt.total_epsilon() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_k_equals_one_special_case() {
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, -1_000.0, 1.0, 1).expect("ok");
        let mut rng = LcgRng::new(99);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        let r1 = svt.process_query(500.0, &mut rng).expect("ok");
        assert!(r1.is_some());
        // The next query should be None (budget exhausted).
        let r2 = svt.process_query(500.0, &mut rng).expect("ok");
        assert!(r2.is_none());
        assert_eq!(svt.remaining_budget(), 0);
    }

    #[test]
    fn test_sensitivity_scaling_increases_noise() {
        // With sensitivity = 10× the released value variance about q
        // should be roughly 10² times larger.  We estimate the variance
        // of (released - q) across many trials.
        let mut var_low = 0.0_f64;
        let mut var_high = 0.0_f64;
        let trials = 256usize;
        let queries = 200usize;
        for trial in 0..trials {
            let cfg_low =
                NumericSvtConfig::new(1.0, 1.0, 1.0, -1_000_000.0, 1.0, queries).expect("ok");
            let cfg_high =
                NumericSvtConfig::new(1.0, 1.0, 1.0, -1_000_000.0, 10.0, queries).expect("ok");
            let mut rng_low = LcgRng::new(trial as u64 * 2 + 1);
            let mut rng_high = LcgRng::new(trial as u64 * 2 + 2);
            let mut svt_low = NumericSvt::new(cfg_low, &mut rng_low).expect("ok");
            let mut svt_high = NumericSvt::new(cfg_high, &mut rng_high).expect("ok");
            for _ in 0..queries {
                if let Some(r) = svt_low.process_query(0.0, &mut rng_low).expect("ok") {
                    var_low += r.value * r.value;
                }
                if let Some(r) = svt_high.process_query(0.0, &mut rng_high).expect("ok") {
                    var_high += r.value * r.value;
                }
            }
        }
        // Sanity: high-sensitivity variance is meaningfully larger.
        assert!(
            var_high > 4.0 * var_low,
            "var_high {var_high} not >> var_low {var_low}"
        );
    }

    #[test]
    fn test_batch_equivalent_to_sequential() {
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, -100.0, 1.0, 5).expect("ok");
        let queries = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut rng_seq = LcgRng::new(2026);
        let mut svt_seq = NumericSvt::new(cfg.clone(), &mut rng_seq).expect("ok");
        let mut seq = Vec::new();
        for &q in &queries {
            if let Some(r) = svt_seq.process_query(q, &mut rng_seq).expect("ok") {
                seq.push(r);
            }
            if svt_seq.remaining_budget() == 0 {
                break;
            }
        }

        let mut rng_batch = LcgRng::new(2026);
        let mut svt_batch = NumericSvt::new(cfg, &mut rng_batch).expect("ok");
        let batch = svt_batch.batch(&queries, &mut rng_batch).expect("ok");

        assert_eq!(seq, batch);
    }

    #[test]
    fn test_released_value_centered_near_q() {
        // Mean of released values (over many trials) should be near q
        // because η ~ Lap(0, scale) is mean-zero.
        let trials = 1_000usize;
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for trial in 0..trials {
            let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, -1e6, 1.0, 1).expect("ok");
            let mut rng = LcgRng::new(trial as u64 + 1);
            let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
            if let Some(r) = svt.process_query(50.0, &mut rng).expect("ok") {
                sum += r.value;
                count += 1;
            }
        }
        assert!(count > 800, "too few above-threshold releases: {count}");
        let mean = sum / count as f64;
        // Theoretical scale = 2·k·Δ / ε = 2·1·1/1 = 2; SE ≈ scale·√2 / √count.
        let se = 2.0_f64.sqrt() * 2.0 / (count as f64).sqrt();
        assert!((mean - 50.0).abs() < 5.0 * se, "mean = {mean}, SE = {se}");
    }

    #[test]
    fn test_indices_monotonically_increase() {
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, -1e6, 1.0, 10).expect("ok");
        let mut rng = LcgRng::new(3);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        let queries: Vec<f64> = (0..10).map(|i| i as f64 + 1000.0).collect();
        let out = svt.batch(&queries, &mut rng).expect("ok");
        assert!(!out.is_empty());
        for w in out.windows(2) {
            assert!(w[0].query_index < w[1].query_index, "non-monotone");
        }
    }

    #[test]
    fn test_empty_batch_yields_empty() {
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, 0.0, 1.0, 3).expect("ok");
        let mut rng = LcgRng::new(1);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        let out = svt.batch(&[], &mut rng).expect("ok");
        assert!(out.is_empty());
        assert_eq!(svt.remaining_budget(), 3);
    }

    #[test]
    fn test_threshold_below_all_returns_first_k() {
        // With a very low threshold, the first k queries should *all* be
        // released.
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, -1e9, 1.0, 4).expect("ok");
        let mut rng = LcgRng::new(31);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        let queries: Vec<f64> = (0..20).map(|i| 1_000.0 + i as f64).collect();
        let out = svt.batch(&queries, &mut rng).expect("ok");
        assert_eq!(out.len(), 4);
        // The first four query indices in the stream should match.
        for (i, r) in out.iter().enumerate() {
            assert_eq!(r.query_index, i);
        }
    }

    #[test]
    fn test_non_finite_query_returns_error() {
        let cfg = cfg_basic(0.0, 3);
        let mut rng = LcgRng::new(5);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        assert!(svt.process_query(f64::NAN, &mut rng).is_err());
        assert!(svt.process_query(f64::INFINITY, &mut rng).is_err());
        assert!(svt.process_query(f64::NEG_INFINITY, &mut rng).is_err());
    }

    #[test]
    fn test_queries_processed_counter_increments() {
        let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, 1e9, 1.0, 5).expect("ok");
        let mut rng = LcgRng::new(0);
        let mut svt = NumericSvt::new(cfg, &mut rng).expect("ok");
        // All queries are well below threshold ⇒ no responses; but the
        // index counter still advances.
        for _ in 0..5 {
            let _ = svt.process_query(0.0, &mut rng).expect("ok");
        }
        assert_eq!(svt.queries_processed(), 5);
        assert_eq!(svt.responses_returned(), 0);
    }

    #[test]
    fn test_config_accessor_returns_input() {
        let cfg = NumericSvtConfig::new(0.7, 0.8, 0.9, 5.0, 2.0, 3).expect("ok");
        let mut rng = LcgRng::new(0);
        let svt = NumericSvt::new(cfg.clone(), &mut rng).expect("ok");
        let back = svt.config();
        assert!((back.epsilon_threshold - cfg.epsilon_threshold).abs() < 1e-12);
        assert!((back.epsilon_query - cfg.epsilon_query).abs() < 1e-12);
        assert!((back.epsilon_value - cfg.epsilon_value).abs() < 1e-12);
        assert!((back.threshold - cfg.threshold).abs() < 1e-12);
        assert!((back.sensitivity - cfg.sensitivity).abs() < 1e-12);
        assert_eq!(back.max_responses, cfg.max_responses);
    }

    #[test]
    fn test_noisy_threshold_centered_around_threshold() {
        // Over many independent sessions the average noisy threshold
        // should be close to the configured threshold (Lap is mean-zero).
        let mut sum = 0.0_f64;
        let trials = 2000usize;
        for trial in 0..trials {
            let cfg = NumericSvtConfig::new(1.0, 1.0, 1.0, 7.0, 1.0, 1).expect("ok");
            let mut rng = LcgRng::new(trial as u64 + 1);
            let svt = NumericSvt::new(cfg, &mut rng).expect("ok");
            sum += svt.noisy_threshold();
        }
        let mean = sum / trials as f64;
        // Threshold noise scale = 2·Δ / ε₁ = 2; SE = scale·√2/√trials.
        let se = 2.0_f64.sqrt() * 2.0 / (trials as f64).sqrt();
        assert!((mean - 7.0).abs() < 5.0 * se, "mean = {mean}, SE = {se}");
    }
}
