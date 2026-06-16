//! Best-of-N (rejection) sampling for reward-guided generation.
//!
//! References:
//! * Nakano et al. 2021, "WebGPT: Browser-assisted question-answering with human
//!   feedback", arXiv:2112.09332 (best-of-n re-ranking against a reward model).
//! * Gao et al. 2023, "Scaling Laws for Reward Model Overoptimization",
//!   arXiv:2210.10760 (best-of-n as a KL-bounded inference-time policy).
//!
//! The pattern is: draw `n_samples` candidate continuations, score each with a
//! reward model, and keep the candidate selected by an aggregation policy.  In
//! the simplest case the policy is `argmax` (keep the single highest-reward
//! candidate); richer policies aggregate the candidate rewards into a scalar
//! summary (mean, top-k mean, softmax-weighted average) that characterises the
//! quality of the sampled set as a whole.
//!
//! This module also exposes [`BestOfN::expected_best_reward`], the order
//! statistic E[max of `n_samples` i.i.d. draws] computed from an empirical
//! reward distribution.  It is the theoretical reward attainable by best-of-n
//! decoding and grows monotonically with `n_samples`.

use crate::error::{RlhfError, RlhfResult};

// ── Aggregation strategy ──────────────────────────────────────────────────────

/// How to aggregate a set of candidate rewards into a single scalar summary.
#[derive(Debug, Clone)]
pub enum ScoreAggregation {
    /// Maximum reward over the candidate set: `max_i r_i`.
    ///
    /// This is the canonical best-of-n score — the reward of the kept candidate.
    Max,
    /// Arithmetic mean over the candidate set: `(1/n) Σ_i r_i`.
    Mean,
    /// Softmax-weighted average: `Σ_i softmax(r_i / τ)_i · r_i`.
    ///
    /// As `τ → 0` this tends to [`ScoreAggregation::Max`]; as `τ → ∞` it tends to
    /// [`ScoreAggregation::Mean`].  `temperature` must be strictly positive.
    SoftmaxWeighted { temperature: f32 },
    /// Mean of the top-`k` rewards (`k` largest candidates).
    ///
    /// `k` must satisfy `1 ≤ k ≤ n`.
    TopKMean { k: usize },
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`BestOfN`].
#[derive(Debug, Clone)]
pub struct BestOfNConfig {
    /// Number of candidates drawn per prompt (must be ≥ 1).
    pub n_samples: usize,
    /// Aggregation policy applied to candidate rewards.
    pub aggregation: ScoreAggregation,
}

// ── Sampler ───────────────────────────────────────────────────────────────────

/// Best-of-N reward-guided selector.
#[derive(Debug, Clone)]
pub struct BestOfN {
    cfg: BestOfNConfig,
}

impl BestOfN {
    /// Construct a [`BestOfN`] selector from a config.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::Internal`] — `cfg.n_samples == 0`.
    /// * [`RlhfError::InvalidTemp`] — `SoftmaxWeighted` with `temperature ≤ 0`.
    pub fn new(cfg: BestOfNConfig) -> RlhfResult<Self> {
        if cfg.n_samples == 0 {
            return Err(RlhfError::Internal {
                msg: "n_samples must be >= 1".into(),
            });
        }
        if let ScoreAggregation::SoftmaxWeighted { temperature } = cfg.aggregation
            && (temperature <= 0.0 || !temperature.is_finite())
        {
            return Err(RlhfError::InvalidTemp { temp: temperature });
        }
        Ok(Self { cfg })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &BestOfNConfig {
        &self.cfg
    }

    /// Index of the selected candidate.
    ///
    /// Selection is always the best single candidate: the index of the maximum
    /// reward.  Ties resolve to the lowest index.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::EmptyInput`] — `rewards` is empty.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    pub fn select(&self, rewards: &[f32]) -> RlhfResult<usize> {
        if rewards.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        argmax_first(rewards)
    }

    /// Aggregate a candidate-reward set into a single scalar per the policy.
    ///
    /// * [`ScoreAggregation::Max`] → `max_i r_i`.
    /// * [`ScoreAggregation::Mean`] → `(1/n) Σ_i r_i`.
    /// * [`ScoreAggregation::TopKMean`] → mean of the `k` largest rewards.
    /// * [`ScoreAggregation::SoftmaxWeighted`] → `Σ_i softmax(r_i / τ)_i · r_i`.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::EmptyInput`] — `rewards` is empty.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    /// * [`RlhfError::Internal`] — `TopKMean` with `k == 0` or `k > rewards.len()`.
    /// * [`RlhfError::InvalidTemp`] — `SoftmaxWeighted` with non-positive temperature.
    pub fn aggregate_score(&self, rewards: &[f32]) -> RlhfResult<f32> {
        if rewards.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        for &r in rewards {
            if r.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
        }
        let score = match &self.cfg.aggregation {
            ScoreAggregation::Max => rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max),

            ScoreAggregation::Mean => rewards.iter().sum::<f32>() / rewards.len() as f32,

            ScoreAggregation::TopKMean { k } => {
                let k = *k;
                if k == 0 || k > rewards.len() {
                    return Err(RlhfError::Internal {
                        msg: format!(
                            "TopKMean k must satisfy 1 <= k <= {}, got {k}",
                            rewards.len()
                        ),
                    });
                }
                let mut sorted = rewards.to_vec();
                // Descending sort; NaN already rejected above so total order holds.
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                sorted[..k].iter().sum::<f32>() / k as f32
            }

            ScoreAggregation::SoftmaxWeighted { temperature } => {
                let temperature = *temperature;
                if temperature <= 0.0 || !temperature.is_finite() {
                    return Err(RlhfError::InvalidTemp { temp: temperature });
                }
                // Numerically stable softmax weighting: subtract the max logit.
                let max_logit = rewards
                    .iter()
                    .map(|&r| r / temperature)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut weight_sum = 0.0_f32;
                let mut weighted_reward = 0.0_f32;
                for &r in rewards {
                    let w = ((r / temperature) - max_logit).exp();
                    weight_sum += w;
                    weighted_reward += w * r;
                }
                if weight_sum <= 0.0 {
                    return Err(RlhfError::Internal {
                        msg: "softmax weight sum collapsed to zero".into(),
                    });
                }
                weighted_reward / weight_sum
            }
        };
        if score.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(score)
    }

    /// Multi-objective selection: each candidate carries a reward vector.
    ///
    /// `candidate_rewards` is `n_candidates × n_objectives` row-major.  Each
    /// candidate's objective vector is averaged into a scalar utility, then the
    /// best candidate (argmax utility, ties → lowest index) is returned.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::EmptyInput`] — `n_candidates == 0` or `n_objectives == 0`.
    /// * [`RlhfError::DimensionMismatch`] — `candidate_rewards.len() != n_candidates * n_objectives`.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    pub fn select_multi(
        &self,
        candidate_rewards: &[f32],
        n_candidates: usize,
        n_objectives: usize,
    ) -> RlhfResult<usize> {
        if n_candidates == 0 || n_objectives == 0 {
            return Err(RlhfError::EmptyInput);
        }
        let expected = n_candidates * n_objectives;
        if candidate_rewards.len() != expected {
            return Err(RlhfError::DimensionMismatch {
                expected,
                got: candidate_rewards.len(),
            });
        }
        let mut utilities = Vec::with_capacity(n_candidates);
        for c in 0..n_candidates {
            let base = c * n_objectives;
            let row = &candidate_rewards[base..base + n_objectives];
            let mut sum = 0.0_f32;
            for &v in row {
                if v.is_nan() {
                    return Err(RlhfError::NanEncountered);
                }
                sum += v;
            }
            utilities.push(sum / n_objectives as f32);
        }
        argmax_first(&utilities)
    }

    /// Monte-Carlo-free estimate of E[max of `n_samples` i.i.d. draws] from the
    /// empirical reward distribution `samples`.
    ///
    /// The empirical samples are treated as the population.  Sorting them
    /// ascending as order statistics `r_(1) ≤ … ≤ r_(m)` and writing the
    /// empirical CDF as `F_i = i / m` (with `F_0 = 0`), the expected maximum of
    /// `n = n_samples` i.i.d. draws is the closed form
    ///
    /// ```text
    /// E[max of n] = Σ_i r_(i) · (F_i^n − F_{i-1}^n).
    /// ```
    ///
    /// `(F_i^n − F_{i-1}^n)` is exactly the probability that the maximum of `n`
    /// draws equals the `i`-th order statistic, so the estimator is a proper
    /// probability-weighted average bounded by `[min, max]`.  For `n = 1` it
    /// equals the empirical mean; it increases monotonically with `n` toward the
    /// maximum.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::EmptyInput`] — `samples` is empty.
    /// * [`RlhfError::NanEncountered`] — any sample is NaN.
    pub fn expected_best_reward(&self, samples: &[f32]) -> RlhfResult<f32> {
        if samples.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        for &s in samples {
            if s.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
        }
        let m = samples.len();
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.cfg.n_samples as f64;
        let m_f = m as f64;
        let mut acc = 0.0_f64;
        let mut prev_cdf_pow = 0.0_f64; // F_0^n = 0
        for (i, &r) in sorted.iter().enumerate() {
            // F_i = (i + 1) / m  for the (i+1)-th order statistic (1-indexed).
            let cdf = ((i + 1) as f64) / m_f;
            let cdf_pow = cdf.powf(n);
            let weight = cdf_pow - prev_cdf_pow;
            acc += f64::from(r) * weight;
            prev_cdf_pow = cdf_pow;
        }
        let out = acc as f32;
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Index of the maximum value (ties → lowest index).
///
/// # Errors
///
/// * [`RlhfError::EmptyInput`] — `values` is empty.
/// * [`RlhfError::NanEncountered`] — any value is NaN.
fn argmax_first(values: &[f32]) -> RlhfResult<usize> {
    if values.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        // Strict `>` keeps the lowest index on ties.
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n_samples: usize, aggregation: ScoreAggregation) -> BestOfN {
        BestOfN::new(BestOfNConfig {
            n_samples,
            aggregation,
        })
        .expect("value should be present")
    }

    // ── select ──────────────────────────────────────────────────────────────

    #[test]
    fn select_returns_argmax_index() {
        let bon = make(4, ScoreAggregation::Max);
        let rewards = [0.1_f32, 0.9, 0.3, 0.5];
        assert_eq!(bon.select(&rewards).expect("select should succeed"), 1);
    }

    #[test]
    fn select_tie_resolves_to_lowest_index() {
        let bon = make(3, ScoreAggregation::Max);
        let rewards = [0.7_f32, 0.7, 0.2];
        assert_eq!(
            bon.select(&rewards).expect("select should succeed"),
            0,
            "tie should resolve to lowest index"
        );
    }

    #[test]
    fn select_single_candidate_index_zero() {
        let bon = make(1, ScoreAggregation::Max);
        assert_eq!(bon.select(&[42.0_f32]).expect("select should succeed"), 0);
    }

    #[test]
    fn select_all_equal_index_zero() {
        let bon = make(5, ScoreAggregation::Mean);
        let rewards = [1.0_f32, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(bon.select(&rewards).expect("select should succeed"), 0);
    }

    // ── aggregate_score ───────────────────────────────────────────────────────

    #[test]
    fn aggregate_max_equals_max() {
        let bon = make(4, ScoreAggregation::Max);
        let rewards = [0.1_f32, 0.9, 0.3, 0.5];
        assert!(
            (bon.aggregate_score(&rewards)
                .expect("aggregate_score should succeed")
                - 0.9)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn aggregate_mean_equals_mean() {
        let bon = make(4, ScoreAggregation::Mean);
        let rewards = [1.0_f32, 2.0, 3.0, 4.0];
        assert!(
            (bon.aggregate_score(&rewards)
                .expect("aggregate_score should succeed")
                - 2.5)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn aggregate_topk_mean_of_top_two() {
        let bon = make(4, ScoreAggregation::TopKMean { k: 2 });
        let rewards = [1.0_f32, 4.0, 2.0, 3.0];
        // top-2 = {4, 3}; mean = 3.5
        assert!(
            (bon.aggregate_score(&rewards)
                .expect("aggregate_score should succeed")
                - 3.5)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn aggregate_topk_full_equals_mean() {
        let bon = make(3, ScoreAggregation::TopKMean { k: 3 });
        let rewards = [1.0_f32, 2.0, 3.0];
        assert!(
            (bon.aggregate_score(&rewards)
                .expect("aggregate_score should succeed")
                - 2.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn aggregate_softmax_in_min_max_range() {
        let bon = make(4, ScoreAggregation::SoftmaxWeighted { temperature: 1.0 });
        let rewards = [0.0_f32, 1.0, 2.0, 3.0];
        let score = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        assert!(
            (0.0..=3.0).contains(&score),
            "softmax-weighted score {score} must lie in [min, max]"
        );
        // Strictly above the mean because higher rewards get more weight.
        assert!(
            score > 1.5,
            "softmax-weighted score {score} should exceed mean 1.5"
        );
    }

    #[test]
    fn aggregate_softmax_tends_to_max_as_temp_small() {
        let bon = make(4, ScoreAggregation::SoftmaxWeighted { temperature: 1e-3 });
        let rewards = [0.0_f32, 1.0, 2.0, 5.0];
        let score = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        assert!(
            (score - 5.0).abs() < 1e-2,
            "as temperature → 0 the score should approach max 5.0, got {score}"
        );
    }

    #[test]
    fn aggregate_softmax_large_rewards_no_overflow() {
        // Stable softmax must not overflow for large logits.
        let bon = make(3, ScoreAggregation::SoftmaxWeighted { temperature: 1.0 });
        let rewards = [100.0_f32, 200.0, 300.0];
        let score = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        assert!(score.is_finite() && (100.0..=300.0).contains(&score));
    }

    // ── select_multi ──────────────────────────────────────────────────────────

    #[test]
    fn select_multi_averages_objectives_then_argmaxes() {
        let bon = make(3, ScoreAggregation::Max);
        // 3 candidates × 2 objectives:
        // c0 = [1, 1] → 1.0 ; c1 = [0, 4] → 2.0 ; c2 = [1, 2] → 1.5
        let candidate_rewards = [1.0_f32, 1.0, 0.0, 4.0, 1.0, 2.0];
        assert_eq!(
            bon.select_multi(&candidate_rewards, 3, 2)
                .expect("select_multi should succeed"),
            1
        );
    }

    #[test]
    fn select_multi_single_objective_matches_select() {
        let bon = make(3, ScoreAggregation::Max);
        let rewards = [0.1_f32, 0.9, 0.3];
        let multi = bon
            .select_multi(&rewards, 3, 1)
            .expect("select_multi should succeed");
        assert_eq!(multi, bon.select(&rewards).expect("select should succeed"));
    }

    #[test]
    fn select_multi_tie_lowest_index() {
        let bon = make(2, ScoreAggregation::Max);
        // c0 = [1, 1] → 1.0 ; c1 = [2, 0] → 1.0 — tie → index 0
        let candidate_rewards = [1.0_f32, 1.0, 2.0, 0.0];
        assert_eq!(
            bon.select_multi(&candidate_rewards, 2, 2)
                .expect("select_multi should succeed"),
            0
        );
    }

    // ── expected_best_reward ──────────────────────────────────────────────────

    #[test]
    fn expected_best_between_mean_and_max() {
        let bon = make(5, ScoreAggregation::Max);
        let samples = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let max = 5.0_f32;
        let ebr = bon
            .expected_best_reward(&samples)
            .expect("expected_best_reward should succeed");
        assert!(
            ebr >= mean - 1e-5 && ebr <= max + 1e-5,
            "expected_best_reward {ebr} must lie in [mean {mean}, max {max}]"
        );
    }

    #[test]
    fn expected_best_n_one_equals_mean() {
        let bon = make(1, ScoreAggregation::Max);
        let samples = [1.0_f32, 2.0, 3.0, 4.0];
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let ebr = bon
            .expected_best_reward(&samples)
            .expect("expected_best_reward should succeed");
        assert!(
            (ebr - mean).abs() < 1e-5,
            "for n_samples=1 expected_best_reward {ebr} should equal mean {mean}"
        );
    }

    #[test]
    fn expected_best_increases_with_n() {
        let samples = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let ebr_2 = make(2, ScoreAggregation::Max)
            .expected_best_reward(&samples)
            .expect("value should be present");
        let ebr_10 = make(10, ScoreAggregation::Max)
            .expected_best_reward(&samples)
            .expect("value should be present");
        let ebr_100 = make(100, ScoreAggregation::Max)
            .expected_best_reward(&samples)
            .expect("value should be present");
        assert!(
            ebr_2 < ebr_10 && ebr_10 < ebr_100,
            "expected_best_reward should increase with n_samples: {ebr_2} < {ebr_10} < {ebr_100}"
        );
    }

    #[test]
    fn expected_best_large_n_approaches_max() {
        let bon = make(10_000, ScoreAggregation::Max);
        let samples = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let ebr = bon
            .expected_best_reward(&samples)
            .expect("expected_best_reward should succeed");
        assert!(
            (ebr - 5.0).abs() < 1e-2,
            "for very large n the expected best should approach max 5.0, got {ebr}"
        );
    }

    #[test]
    fn expected_best_single_sample_equals_value() {
        let bon = make(7, ScoreAggregation::Max);
        let value = 4.25_f32;
        let ebr = bon
            .expected_best_reward(&[value])
            .expect("expected_best_reward should succeed");
        assert!((ebr - value).abs() < 1e-5);
    }

    // ── Determinism ─────────────────────────────────────────────────────────────

    #[test]
    fn deterministic_repeated_calls() {
        let bon = make(8, ScoreAggregation::SoftmaxWeighted { temperature: 0.7 });
        let rewards = [0.3_f32, 0.8, 0.1, 0.5, 0.9, 0.2, 0.6, 0.4];
        let a = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        let b = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        let c = bon
            .aggregate_score(&rewards)
            .expect("aggregate_score should succeed");
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(b.to_bits(), c.to_bits());
        assert_eq!(
            bon.select(&rewards).expect("select should succeed"),
            bon.select(&rewards).expect("select should succeed")
        );
    }

    // ── Error paths ─────────────────────────────────────────────────────────────

    #[test]
    fn err_empty_rewards_select_and_aggregate() {
        let bon = make(3, ScoreAggregation::Max);
        assert!(matches!(bon.select(&[]), Err(RlhfError::EmptyInput)));
        assert!(matches!(
            bon.aggregate_score(&[]),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn err_n_samples_zero() {
        let res = BestOfN::new(BestOfNConfig {
            n_samples: 0,
            aggregation: ScoreAggregation::Max,
        });
        assert!(matches!(res, Err(RlhfError::Internal { .. })));
    }

    #[test]
    fn err_topk_k_zero() {
        let bon = make(3, ScoreAggregation::TopKMean { k: 0 });
        assert!(matches!(
            bon.aggregate_score(&[1.0_f32, 2.0, 3.0]),
            Err(RlhfError::Internal { .. })
        ));
    }

    #[test]
    fn err_topk_k_exceeds_len() {
        let bon = make(3, ScoreAggregation::TopKMean { k: 5 });
        assert!(matches!(
            bon.aggregate_score(&[1.0_f32, 2.0, 3.0]),
            Err(RlhfError::Internal { .. })
        ));
    }

    #[test]
    fn err_softmax_temperature_non_positive_on_new() {
        let res = BestOfN::new(BestOfNConfig {
            n_samples: 4,
            aggregation: ScoreAggregation::SoftmaxWeighted { temperature: 0.0 },
        });
        assert!(matches!(res, Err(RlhfError::InvalidTemp { .. })));
        let res_neg = BestOfN::new(BestOfNConfig {
            n_samples: 4,
            aggregation: ScoreAggregation::SoftmaxWeighted { temperature: -1.0 },
        });
        assert!(matches!(res_neg, Err(RlhfError::InvalidTemp { .. })));
    }

    #[test]
    fn err_select_multi_length_mismatch() {
        let bon = make(2, ScoreAggregation::Max);
        // expects 2*2 = 4 entries, give 3
        assert!(matches!(
            bon.select_multi(&[1.0_f32, 2.0, 3.0], 2, 2),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_select_multi_n_objectives_zero() {
        let bon = make(2, ScoreAggregation::Max);
        assert!(matches!(
            bon.select_multi(&[], 2, 0),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn err_select_multi_n_candidates_zero() {
        let bon = make(2, ScoreAggregation::Max);
        assert!(matches!(
            bon.select_multi(&[], 0, 2),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn err_nan_reward_select() {
        let bon = make(3, ScoreAggregation::Max);
        assert!(matches!(
            bon.select(&[1.0_f32, f32::NAN, 2.0]),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn err_nan_sample_expected_best() {
        let bon = make(3, ScoreAggregation::Max);
        assert!(matches!(
            bon.expected_best_reward(&[1.0_f32, f32::NAN]),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn err_empty_samples_expected_best() {
        let bon = make(3, ScoreAggregation::Max);
        assert!(matches!(
            bon.expected_best_reward(&[]),
            Err(RlhfError::EmptyInput)
        ));
    }
}
