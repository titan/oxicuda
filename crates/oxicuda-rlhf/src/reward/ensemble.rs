//! Reward-model ensembling and uncertainty estimation for RLHF robustness.
//!
//! Reference:
//! * Coste et al. 2023, "Reward Model Ensembles Help Mitigate Overoptimization",
//!   arXiv:2310.02743.
//!
//! A single learned reward model is a noisy, exploitable proxy for the true
//! objective: a policy optimised against it tends to "overoptimise" — exploiting
//! reward-model errors rather than improving genuine quality.  Aggregating `K`
//! independently-trained reward models and penalising their disagreement yields a
//! more conservative, harder-to-exploit signal.
//!
//! This module aggregates per-model rewards ([`EnsembleAgg::Mean`],
//! [`EnsembleAgg::Min`], [`EnsembleAgg::WeightedMean`]), measures cross-model
//! disagreement as the population standard deviation ([`RewardEnsemble::uncertainty`]),
//! and forms a pessimistic uncertainty-penalised reward
//! `aggregate − λ · uncertainty` ([`RewardEnsemble::penalized_reward`]).

use crate::error::{RlhfError, RlhfResult};

// ── Aggregation strategy ──────────────────────────────────────────────────────

/// How to combine the `K` per-model rewards into a single ensemble reward.
#[derive(Debug, Clone)]
pub enum EnsembleAgg {
    /// Arithmetic mean over the models: `(1/K) Σ_k r_k`.
    Mean,
    /// Minimum over the models: `min_k r_k`.
    ///
    /// The most conservative aggregation — the ensemble is only as confident as
    /// its most pessimistic member, strongly suppressing reward hacking.
    Min,
    /// Weighted mean: `Σ_k w_k r_k` with weights normalised to sum to one.
    ///
    /// Useful when models have differing reliability.
    WeightedMean,
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`RewardEnsemble`].
#[derive(Debug, Clone)]
pub struct RewardEnsembleConfig {
    /// Number of reward models in the ensemble (must be ≥ 1).
    pub n_models: usize,
    /// Uncertainty penalty coefficient `λ ≥ 0` applied to the cross-model std.
    pub uncertainty_penalty: f32,
    /// Aggregation policy.
    pub aggregation: EnsembleAgg,
}

// ── Ensemble ──────────────────────────────────────────────────────────────────

/// Reward-model ensemble with disagreement-based uncertainty penalisation.
#[derive(Debug, Clone)]
pub struct RewardEnsemble {
    cfg: RewardEnsembleConfig,
    weights: Vec<f32>,
}

impl RewardEnsemble {
    /// Construct an ensemble with uniform weights.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::Internal`] — `cfg.n_models == 0`.
    /// * [`RlhfError::InvalidLambda`] — `cfg.uncertainty_penalty < 0`.
    pub fn new(cfg: RewardEnsembleConfig) -> RlhfResult<Self> {
        validate_cfg(&cfg)?;
        let weights = vec![1.0_f32 / cfg.n_models as f32; cfg.n_models];
        Ok(Self { cfg, weights })
    }

    /// Construct an ensemble with explicit per-model weights (for
    /// [`EnsembleAgg::WeightedMean`]).  Weights are stored as-given and
    /// normalised at aggregation time.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::Internal`] — `cfg.n_models == 0`.
    /// * [`RlhfError::InvalidLambda`] — `cfg.uncertainty_penalty < 0`.
    /// * [`RlhfError::DimensionMismatch`] — `weights.len() != cfg.n_models`.
    /// * [`RlhfError::RewardNormFailed`] — weights sum to ≤ 0 or contain a NaN.
    pub fn with_weights(cfg: RewardEnsembleConfig, weights: Vec<f32>) -> RlhfResult<Self> {
        validate_cfg(&cfg)?;
        if weights.len() != cfg.n_models {
            return Err(RlhfError::DimensionMismatch {
                expected: cfg.n_models,
                got: weights.len(),
            });
        }
        let mut sum = 0.0_f32;
        for &w in &weights {
            if w.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(RlhfError::RewardNormFailed {
                msg: "ensemble weights must sum to a positive value".into(),
            });
        }
        Ok(Self { cfg, weights })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &RewardEnsembleConfig {
        &self.cfg
    }

    /// Borrow the (un-normalised) per-model weights.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Aggregate the `K` per-model rewards per the configured policy.
    ///
    /// * [`EnsembleAgg::Mean`] → `(1/K) Σ_k r_k`.
    /// * [`EnsembleAgg::Min`] → `min_k r_k`.
    /// * [`EnsembleAgg::WeightedMean`] → `Σ_k (w_k / Σ w) r_k`.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::DimensionMismatch`] — `model_rewards.len() != n_models`.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    pub fn aggregate(&self, model_rewards: &[f32]) -> RlhfResult<f32> {
        self.check_rewards(model_rewards)?;
        let k = self.cfg.n_models;
        let score = match self.cfg.aggregation {
            EnsembleAgg::Mean => model_rewards.iter().sum::<f32>() / k as f32,

            EnsembleAgg::Min => model_rewards.iter().copied().fold(f32::INFINITY, f32::min),

            EnsembleAgg::WeightedMean => {
                let weight_total: f32 = self.weights.iter().sum();
                // `with_weights`/`new` guarantee weight_total > 0.
                let weighted: f32 = model_rewards
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&r, &w)| r * w)
                    .sum();
                weighted / weight_total
            }
        };
        if score.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(score)
    }

    /// Cross-model uncertainty: the population standard deviation of the `K`
    /// model rewards, `sqrt((1/K) Σ_k (r_k − mean)²)`.
    ///
    /// Zero when all models agree.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::DimensionMismatch`] — `model_rewards.len() != n_models`.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    pub fn uncertainty(&self, model_rewards: &[f32]) -> RlhfResult<f32> {
        self.check_rewards(model_rewards)?;
        let k = self.cfg.n_models as f32;
        let mean = model_rewards.iter().sum::<f32>() / k;
        let var = model_rewards
            .iter()
            .map(|&r| {
                let d = r - mean;
                d * d
            })
            .sum::<f32>()
            / k;
        let std = var.max(0.0).sqrt();
        if std.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(std)
    }

    /// Pessimistic uncertainty-penalised reward:
    /// `aggregate(model_rewards) − λ · uncertainty(model_rewards)`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`RewardEnsemble::aggregate`] and
    /// [`RewardEnsemble::uncertainty`].
    pub fn penalized_reward(&self, model_rewards: &[f32]) -> RlhfResult<f32> {
        let agg = self.aggregate(model_rewards)?;
        let unc = self.uncertainty(model_rewards)?;
        let out = agg - self.cfg.uncertainty_penalty * unc;
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }

    /// Batch penalised reward.
    ///
    /// `rewards` is `n_items × n_models` row-major; returns one penalised reward
    /// per item.
    ///
    /// # Errors
    ///
    /// * [`RlhfError::EmptyInput`] — `n_items == 0`.
    /// * [`RlhfError::DimensionMismatch`] — `rewards.len() != n_items * n_models`.
    /// * [`RlhfError::NanEncountered`] — any reward is NaN.
    pub fn batch_penalized(&self, rewards: &[f32], n_items: usize) -> RlhfResult<Vec<f32>> {
        if n_items == 0 {
            return Err(RlhfError::EmptyInput);
        }
        let k = self.cfg.n_models;
        let expected = n_items * k;
        if rewards.len() != expected {
            return Err(RlhfError::DimensionMismatch {
                expected,
                got: rewards.len(),
            });
        }
        let mut out = Vec::with_capacity(n_items);
        for item in 0..n_items {
            let base = item * k;
            let row = &rewards[base..base + k];
            out.push(self.penalized_reward(row)?);
        }
        Ok(out)
    }

    /// Internal: validate a per-model reward slice (length + NaN).
    fn check_rewards(&self, model_rewards: &[f32]) -> RlhfResult<()> {
        if model_rewards.len() != self.cfg.n_models {
            return Err(RlhfError::DimensionMismatch {
                expected: self.cfg.n_models,
                got: model_rewards.len(),
            });
        }
        for &r in model_rewards {
            if r.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Validate the shared config invariants (`n_models ≥ 1`, `λ ≥ 0`).
fn validate_cfg(cfg: &RewardEnsembleConfig) -> RlhfResult<()> {
    if cfg.n_models == 0 {
        return Err(RlhfError::Internal {
            msg: "n_models must be >= 1".into(),
        });
    }
    if cfg.uncertainty_penalty < 0.0 || !cfg.uncertainty_penalty.is_finite() {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.uncertainty_penalty,
        });
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n_models: usize, lambda: f32, aggregation: EnsembleAgg) -> RewardEnsemble {
        RewardEnsemble::new(RewardEnsembleConfig {
            n_models,
            uncertainty_penalty: lambda,
            aggregation,
        })
        .unwrap()
    }

    // ── aggregate ───────────────────────────────────────────────────────────

    #[test]
    fn aggregate_mean_equals_mean() {
        let ens = make(4, 0.0, EnsembleAgg::Mean);
        let rewards = [1.0_f32, 2.0, 3.0, 4.0];
        assert!((ens.aggregate(&rewards).unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn aggregate_min_equals_min() {
        let ens = make(4, 0.0, EnsembleAgg::Min);
        let rewards = [3.0_f32, 1.0, 4.0, 2.0];
        assert!((ens.aggregate(&rewards).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_mean_uniform_equals_mean() {
        // `new` gives uniform weights → WeightedMean must equal the plain mean.
        let ens = make(4, 0.0, EnsembleAgg::WeightedMean);
        let rewards = [1.0_f32, 2.0, 3.0, 4.0];
        assert!((ens.aggregate(&rewards).unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn weighted_mean_honors_weights() {
        let cfg = RewardEnsembleConfig {
            n_models: 3,
            uncertainty_penalty: 0.0,
            aggregation: EnsembleAgg::WeightedMean,
        };
        // weights [1, 2, 1] sum 4 → normalised [0.25, 0.5, 0.25]
        // rewards [4, 0, 4] → 0.25*4 + 0.5*0 + 0.25*4 = 2.0
        let ens = RewardEnsemble::with_weights(cfg, vec![1.0, 2.0, 1.0]).unwrap();
        let rewards = [4.0_f32, 0.0, 4.0];
        assert!(
            (ens.aggregate(&rewards).unwrap() - 2.0).abs() < 1e-6,
            "weighted mean = {}, expected 2.0",
            ens.aggregate(&rewards).unwrap()
        );
    }

    // ── uncertainty ───────────────────────────────────────────────────────────

    #[test]
    fn uncertainty_equals_population_std() {
        let ens = make(4, 0.0, EnsembleAgg::Mean);
        // rewards [2, 4, 4, 6] → mean 4, variance = (4+0+0+4)/4 = 2, std = sqrt(2)
        let rewards = [2.0_f32, 4.0, 4.0, 6.0];
        let expected = 2.0_f32.sqrt();
        assert!(
            (ens.uncertainty(&rewards).unwrap() - expected).abs() < 1e-5,
            "uncertainty = {}, expected {expected}",
            ens.uncertainty(&rewards).unwrap()
        );
    }

    #[test]
    fn uncertainty_zero_when_models_agree() {
        let ens = make(5, 0.0, EnsembleAgg::Mean);
        let rewards = [3.3_f32, 3.3, 3.3, 3.3, 3.3];
        assert!(
            ens.uncertainty(&rewards).unwrap().abs() < 1e-6,
            "uncertainty should be 0 when all models agree"
        );
    }

    #[test]
    fn uncertainty_single_model_zero() {
        let ens = make(1, 1.0, EnsembleAgg::Mean);
        assert!(ens.uncertainty(&[5.0_f32]).unwrap().abs() < 1e-6);
    }

    // ── penalized_reward ──────────────────────────────────────────────────────

    #[test]
    fn penalized_equals_aggregate_minus_lambda_std() {
        let lambda = 0.5_f32;
        let ens = make(4, lambda, EnsembleAgg::Mean);
        let rewards = [2.0_f32, 4.0, 4.0, 6.0];
        let agg = ens.aggregate(&rewards).unwrap();
        let std = ens.uncertainty(&rewards).unwrap();
        let expected = agg - lambda * std;
        assert!(
            (ens.penalized_reward(&rewards).unwrap() - expected).abs() < 1e-5,
            "penalized = {}, expected {expected}",
            ens.penalized_reward(&rewards).unwrap()
        );
    }

    #[test]
    fn penalized_below_aggregate_when_disagreement() {
        let ens = make(3, 1.0, EnsembleAgg::Mean);
        let rewards = [1.0_f32, 3.0, 5.0]; // std > 0
        let agg = ens.aggregate(&rewards).unwrap();
        let pen = ens.penalized_reward(&rewards).unwrap();
        assert!(
            pen < agg,
            "penalized {pen} should be strictly below aggregate {agg} when λ>0 and disagreement>0"
        );
    }

    #[test]
    fn penalized_equals_aggregate_when_lambda_zero() {
        let ens = make(3, 0.0, EnsembleAgg::Mean);
        let rewards = [1.0_f32, 3.0, 5.0];
        let agg = ens.aggregate(&rewards).unwrap();
        let pen = ens.penalized_reward(&rewards).unwrap();
        assert!(
            (pen - agg).abs() < 1e-6,
            "penalized {pen} should equal aggregate {agg} when λ=0"
        );
    }

    #[test]
    fn penalized_single_model_equals_reward() {
        let ens = make(1, 2.0, EnsembleAgg::Mean);
        let pen = ens.penalized_reward(&[7.0_f32]).unwrap();
        assert!(
            (pen - 7.0).abs() < 1e-6,
            "single model penalized should equal the reward (no disagreement)"
        );
    }

    #[test]
    fn min_aggregation_le_mean() {
        let rewards = [1.0_f32, 3.0, 5.0, 7.0];
        let mean = make(4, 0.0, EnsembleAgg::Mean).aggregate(&rewards).unwrap();
        let min = make(4, 0.0, EnsembleAgg::Min).aggregate(&rewards).unwrap();
        assert!(
            min <= mean,
            "Min aggregation {min} must be <= Mean aggregation {mean} (conservative)"
        );
    }

    // ── batch_penalized ───────────────────────────────────────────────────────

    #[test]
    fn batch_penalized_length_equals_n_items() {
        let ens = make(2, 0.5, EnsembleAgg::Mean);
        // 3 items × 2 models
        let rewards = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ens.batch_penalized(&rewards, 3).unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn batch_matches_per_item_penalized() {
        let ens = make(2, 0.7, EnsembleAgg::Min);
        let rewards = [1.0_f32, 5.0, 2.0, 2.0, 9.0, 1.0];
        let batch = ens.batch_penalized(&rewards, 3).unwrap();
        for item in 0..3 {
            let row = &rewards[item * 2..item * 2 + 2];
            let single = ens.penalized_reward(row).unwrap();
            assert!(
                (batch[item] - single).abs() < 1e-6,
                "batch[{item}] = {} should match per-item {single}",
                batch[item]
            );
        }
    }

    // ── Determinism ─────────────────────────────────────────────────────────────

    #[test]
    fn deterministic_repeated_calls() {
        let ens = make(4, 0.3, EnsembleAgg::Mean);
        let rewards = [0.5_f32, 1.5, 2.5, 3.5];
        let a = ens.penalized_reward(&rewards).unwrap();
        let b = ens.penalized_reward(&rewards).unwrap();
        let c = ens.penalized_reward(&rewards).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(b.to_bits(), c.to_bits());
    }

    // ── Error paths ─────────────────────────────────────────────────────────────

    #[test]
    fn err_model_rewards_wrong_length() {
        let ens = make(3, 0.0, EnsembleAgg::Mean);
        assert!(matches!(
            ens.aggregate(&[1.0_f32, 2.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            ens.uncertainty(&[1.0_f32, 2.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_models_zero() {
        let res = RewardEnsemble::new(RewardEnsembleConfig {
            n_models: 0,
            uncertainty_penalty: 0.0,
            aggregation: EnsembleAgg::Mean,
        });
        assert!(matches!(res, Err(RlhfError::Internal { .. })));
    }

    #[test]
    fn err_lambda_negative() {
        let res = RewardEnsemble::new(RewardEnsembleConfig {
            n_models: 3,
            uncertainty_penalty: -0.1,
            aggregation: EnsembleAgg::Mean,
        });
        assert!(matches!(res, Err(RlhfError::InvalidLambda { .. })));
    }

    #[test]
    fn err_weights_wrong_length() {
        let cfg = RewardEnsembleConfig {
            n_models: 3,
            uncertainty_penalty: 0.0,
            aggregation: EnsembleAgg::WeightedMean,
        };
        let res = RewardEnsemble::with_weights(cfg, vec![1.0, 1.0]);
        assert!(matches!(res, Err(RlhfError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_zero_sum_weights() {
        let cfg = RewardEnsembleConfig {
            n_models: 2,
            uncertainty_penalty: 0.0,
            aggregation: EnsembleAgg::WeightedMean,
        };
        // sum = 0 → must error
        let res = RewardEnsemble::with_weights(cfg, vec![1.0, -1.0]);
        assert!(matches!(res, Err(RlhfError::RewardNormFailed { .. })));
    }

    #[test]
    fn err_batch_length_mismatch() {
        let ens = make(2, 0.0, EnsembleAgg::Mean);
        // expects 3*2 = 6, give 5
        assert!(matches!(
            ens.batch_penalized(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], 3),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_batch_n_items_zero() {
        let ens = make(2, 0.0, EnsembleAgg::Mean);
        assert!(matches!(
            ens.batch_penalized(&[], 0),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn err_nan_reward() {
        let ens = make(2, 0.0, EnsembleAgg::Mean);
        assert!(matches!(
            ens.aggregate(&[1.0_f32, f32::NAN]),
            Err(RlhfError::NanEncountered)
        ));
    }
}
