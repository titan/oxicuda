//! Length-controlled DPO: Disentangling length from quality in preference
//! optimisation.
//!
//! References:
//! - Park et al. 2024, "Disentangling Length from Quality in Direct Preference
//!   Optimization", arXiv:2403.19159.
//! - Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-
//!   Free Reward", arXiv:2405.14734 (SimPO target-reward margin γ).
//!
//! Standard DPO is susceptible to "length exploitation": the policy can
//! increase the log-ratio by simply producing longer chosen responses.
//! Two complementary techniques mitigate this:
//!
//! 1. **Length normalisation** (`normalize_by_length = true`): divide each
//!    log-prob by its sequence length before computing the DPO log-ratio.
//!    This converts token-sum log-probs into per-token average log-probs,
//!    making the signal invariant to sequence length.
//!
//! 2. **Length penalty** (`length_lambda > 0`): add an explicit L1 penalty on
//!    the absolute difference in chosen / rejected sequence lengths.
//!
//! 3. **SimPO margin** (`target_reward_margin > 0`): subtract a constant γ
//!    from the logit, requiring the policy to achieve at least γ reward margin
//!    between chosen and rejected.
//!
//! Per-pair loss:
//! ```text
//! logit = β * (norm_c - norm_rc - norm_r + norm_rr) - γ
//! loss  = -log σ(logit) + λ * |len_chosen - len_rejected|
//! ```
//! where `norm_x = logp_x / len` if `normalize_by_length`, else `norm_x = logp_x`.

use crate::dpo::step_dpo::log_sigmoid;
use crate::error::{RlhfError, RlhfResult};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for Length-controlled DPO.
#[derive(Debug, Clone)]
pub struct LengthDpoConfig {
    /// DPO temperature β (same role as in `DpoConfig`).
    ///
    /// Must be positive and finite.
    pub beta: f32,

    /// Weight λ for the length-difference penalty (`≥ 0`, finite).
    ///
    /// Setting `length_lambda = 0.0` disables the explicit length penalty.
    pub length_lambda: f32,

    /// If `true`, divide each log-prob by its sequence length before computing
    /// the log-ratio (per-token normalisation a la SimPO / Park et al.).
    pub normalize_by_length: bool,

    /// SimPO target-reward margin γ (default 0.0).
    ///
    /// The logit is shifted as `logit = β * (log_ratio) - γ`.
    /// Positive γ requires the chosen response to achieve a higher reward
    /// margin before contributing to a low loss.
    pub target_reward_margin: f32,
}

impl Default for LengthDpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            length_lambda: 0.01,
            normalize_by_length: true,
            target_reward_margin: 0.0,
        }
    }
}

// ── Input types ───────────────────────────────────────────────────────────────

/// An extended preference pair that carries sequence lengths alongside
/// log-probabilities.
///
/// All log-probs represent per-sequence sums (not per-token averages).
/// Length normalisation (if requested) is performed inside [`LengthDpo`].
#[derive(Debug, Clone)]
pub struct LengthPair {
    /// log π_θ(y_chosen | x): policy log-prob of chosen response.
    pub chosen_logp: f32,
    /// log π_ref(y_chosen | x): reference log-prob of chosen response.
    pub ref_chosen_logp: f32,
    /// log π_θ(y_rejected | x): policy log-prob of rejected response.
    pub rejected_logp: f32,
    /// log π_ref(y_rejected | x): reference log-prob of rejected response.
    pub ref_rejected_logp: f32,
    /// Number of tokens in the chosen response (> 0 if length-normalising).
    pub chosen_len: usize,
    /// Number of tokens in the rejected response (> 0 if length-normalising).
    pub rejected_len: usize,
}

/// A batch of [`LengthPair`] instances.
#[derive(Debug, Clone)]
pub struct LengthDpoBatch {
    /// Preference pairs in this batch.
    pub pairs: Vec<LengthPair>,
}

impl LengthDpoBatch {
    /// Create a new batch from the given pairs.
    pub fn new(pairs: Vec<LengthPair>) -> Self {
        Self { pairs }
    }

    /// Number of pairs in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns `true` if the batch contains no pairs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Length-controlled DPO loss computation.
///
/// Provides length normalisation, length penalty, and SimPO margin support
/// as independent, composable knobs controlled via [`LengthDpoConfig`].
pub struct LengthDpo;

impl LengthDpo {
    /// Compute the (optionally) length-normalised log-prob.
    ///
    /// If `normalize` is `true`: returns `logp / seq_len as f32`.
    /// If `normalize` is `false`: returns `logp` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::DimensionMismatch`] if `normalize` is `true` and
    /// `seq_len == 0` (division by zero).
    pub fn normalize_logp(logp: f32, seq_len: usize, normalize: bool) -> RlhfResult<f32> {
        if normalize {
            if seq_len == 0 {
                return Err(RlhfError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
            Ok(logp / seq_len as f32)
        } else {
            Ok(logp)
        }
    }

    /// Compute the DPO log-ratio logit for a single pair with optional length
    /// normalisation and SimPO target-reward margin.
    ///
    /// ```text
    /// norm_c   = normalize_logp(chosen_logp,     chosen_len,   normalize)
    /// norm_rc  = normalize_logp(ref_chosen_logp,  chosen_len,   normalize)
    /// norm_r   = normalize_logp(rejected_logp,   rejected_len, normalize)
    /// norm_rr  = normalize_logp(ref_rejected_logp, rejected_len, normalize)
    ///
    /// logit = β * ((norm_c - norm_rc) - (norm_r - norm_rr)) - γ
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates errors from [`LengthDpo::normalize_logp`].
    pub fn length_log_ratio(pair: &LengthPair, cfg: &LengthDpoConfig) -> RlhfResult<f32> {
        let norm_chosen =
            Self::normalize_logp(pair.chosen_logp, pair.chosen_len, cfg.normalize_by_length)?;
        let norm_ref_chosen = Self::normalize_logp(
            pair.ref_chosen_logp,
            pair.chosen_len,
            cfg.normalize_by_length,
        )?;
        let norm_rejected = Self::normalize_logp(
            pair.rejected_logp,
            pair.rejected_len,
            cfg.normalize_by_length,
        )?;
        let norm_ref_rejected = Self::normalize_logp(
            pair.ref_rejected_logp,
            pair.rejected_len,
            cfg.normalize_by_length,
        )?;

        let log_ratio_chosen = norm_chosen - norm_ref_chosen;
        let log_ratio_rejected = norm_rejected - norm_ref_rejected;
        let logit = cfg.beta * (log_ratio_chosen - log_ratio_rejected) - cfg.target_reward_margin;
        Ok(logit)
    }

    /// Length penalty for a single pair.
    ///
    /// Returns `lambda * |chosen_len - rejected_len|`.
    #[inline]
    pub fn length_penalty(pair: &LengthPair, lambda: f32) -> f32 {
        let diff = if pair.chosen_len >= pair.rejected_len {
            (pair.chosen_len - pair.rejected_len) as f32
        } else {
            (pair.rejected_len - pair.chosen_len) as f32
        };
        lambda * diff
    }

    /// Loss for a single pair.
    ///
    /// ```text
    /// loss = -log_sigmoid(logit) + length_penalty(pair, length_lambda)
    /// ```
    ///
    /// # Errors
    ///
    /// - [`RlhfError::InvalidBeta`] — `beta ≤ 0` or non-finite.
    /// - [`RlhfError::InvalidLambda`] — `length_lambda < 0` or non-finite.
    /// - [`RlhfError::NanEncountered`] — any log-prob is NaN.
    /// - Propagates errors from [`LengthDpo::normalize_logp`].
    pub fn loss_per_pair(pair: &LengthPair, cfg: &LengthDpoConfig) -> RlhfResult<f32> {
        Self::validate_config(cfg)?;

        // Check for NaN in inputs
        if pair.chosen_logp.is_nan()
            || pair.ref_chosen_logp.is_nan()
            || pair.rejected_logp.is_nan()
            || pair.ref_rejected_logp.is_nan()
        {
            return Err(RlhfError::NanEncountered);
        }

        let logit = Self::length_log_ratio(pair, cfg)?;
        let dpo_loss = -log_sigmoid(logit);
        let pen = Self::length_penalty(pair, cfg.length_lambda);
        let loss = dpo_loss + pen;
        if loss.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(loss)
    }

    /// Mean loss over a batch of preference pairs.
    ///
    /// # Errors
    ///
    /// - [`RlhfError::EmptyInput`] — batch is empty.
    /// - Propagates errors from [`LengthDpo::loss_per_pair`].
    pub fn loss(batch: &LengthDpoBatch, cfg: &LengthDpoConfig) -> RlhfResult<f32> {
        if batch.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        Self::validate_config(cfg)?;

        let mut total = 0.0_f32;
        for pair in &batch.pairs {
            total += Self::loss_per_pair(pair, cfg)?;
        }
        let mean = total / batch.len() as f32;
        if mean.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(mean)
    }

    /// Compute the implicit reward difference for a single pair.
    ///
    /// ```text
    /// reward_diff = β * (norm_chosen - norm_rejected) - γ
    /// ```
    ///
    /// where `norm_x` is the (optionally) length-normalised policy log-prob
    /// (without subtracting the reference, unlike the DPO log-ratio).
    ///
    /// A positive value indicates the policy assigns relatively higher
    /// (length-normalised) probability to the chosen response than to the
    /// rejected one, minus the required margin.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`LengthDpo::normalize_logp`].
    pub fn implicit_reward_diff(pair: &LengthPair, cfg: &LengthDpoConfig) -> RlhfResult<f32> {
        let norm_chosen =
            Self::normalize_logp(pair.chosen_logp, pair.chosen_len, cfg.normalize_by_length)?;
        let norm_rejected = Self::normalize_logp(
            pair.rejected_logp,
            pair.rejected_len,
            cfg.normalize_by_length,
        )?;
        Ok(cfg.beta * (norm_chosen - norm_rejected) - cfg.target_reward_margin)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn validate_config(cfg: &LengthDpoConfig) -> RlhfResult<()> {
        if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: cfg.beta });
        }
        if !cfg.length_lambda.is_finite() || cfg.length_lambda < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: cfg.length_lambda,
            });
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_pair(
        chosen_logp: f32,
        ref_chosen_logp: f32,
        rejected_logp: f32,
        ref_rejected_logp: f32,
        chosen_len: usize,
        rejected_len: usize,
    ) -> LengthPair {
        LengthPair {
            chosen_logp,
            ref_chosen_logp,
            rejected_logp,
            ref_rejected_logp,
            chosen_len,
            rejected_len,
        }
    }

    // ── normalize_logp ────────────────────────────────────────────────────────

    #[test]
    fn normalize_logp_divides_by_len_when_normalize_true() {
        let result =
            LengthDpo::normalize_logp(-10.0, 5, true).expect("normalize_logp should succeed");
        assert!(
            (result - (-2.0_f32)).abs() < 1e-6,
            "normalize_logp(-10.0, 5, true) = {result}, expected -2.0"
        );
    }

    #[test]
    fn normalize_logp_returns_unchanged_when_normalize_false() {
        let result =
            LengthDpo::normalize_logp(-10.0, 5, false).expect("normalize_logp should succeed");
        assert!(
            (result - (-10.0_f32)).abs() < 1e-6,
            "normalize_logp(-10.0, 5, false) = {result}, expected -10.0"
        );
    }

    #[test]
    fn normalize_logp_zero_len_returns_error() {
        let result = LengthDpo::normalize_logp(-10.0, 0, true);
        assert!(
            matches!(result, Err(RlhfError::DimensionMismatch { .. })),
            "seq_len=0 with normalize=true should return DimensionMismatch"
        );
    }

    #[test]
    fn normalize_logp_zero_len_false_normalize_ok() {
        // When normalize=false, seq_len=0 is fine (no division)
        let result = LengthDpo::normalize_logp(-5.0, 0, false);
        assert!(
            result.is_ok(),
            "seq_len=0 with normalize=false should succeed"
        );
    }

    // ── length_penalty ────────────────────────────────────────────────────────

    #[test]
    fn length_penalty_equal_lengths_is_zero() {
        let pair = make_pair(-1.0, -1.0, -2.0, -2.0, 10, 10);
        let pen = LengthDpo::length_penalty(&pair, 0.01);
        assert!(pen.abs() < 1e-6, "equal lengths → penalty=0, got {pen}");
    }

    #[test]
    fn length_penalty_different_lengths_positive() {
        let pair = make_pair(-1.0, -1.0, -2.0, -2.0, 15, 10);
        let pen = LengthDpo::length_penalty(&pair, 0.01);
        assert!(
            (pen - 0.05_f32).abs() < 1e-5,
            "diff=5, lambda=0.01 → penalty=0.05, got {pen}"
        );
    }

    #[test]
    fn length_penalty_lambda_zero_is_always_zero() {
        let pair = make_pair(-1.0, -1.0, -2.0, -2.0, 100, 10);
        let pen = LengthDpo::length_penalty(&pair, 0.0);
        assert!(pen.abs() < 1e-6, "lambda=0 → penalty=0, got {pen}");
    }

    #[test]
    fn length_penalty_symmetric_chosen_shorter() {
        // rejected > chosen: penalty should be same magnitude
        let pair_a = make_pair(-1.0, -1.0, -2.0, -2.0, 15, 10);
        let pair_b = make_pair(-1.0, -1.0, -2.0, -2.0, 10, 15);
        let pen_a = LengthDpo::length_penalty(&pair_a, 1.0);
        let pen_b = LengthDpo::length_penalty(&pair_b, 1.0);
        assert!(
            (pen_a - pen_b).abs() < 1e-6,
            "penalty should be symmetric: pen_a={pen_a}, pen_b={pen_b}"
        );
    }

    // ── loss_per_pair ─────────────────────────────────────────────────────────

    #[test]
    fn loss_per_pair_lambda_zero_matches_plain_dpo() {
        // With lambda=0, normalize=false: should equal standard -log_sigmoid(logit)
        let pair = make_pair(-1.0, -1.1, -2.0, -1.9, 10, 10);
        let cfg = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let loss = LengthDpo::loss_per_pair(&pair, &cfg).expect("loss_per_pair should succeed");
        // Manual: logit = 0.1 * ((-1.0 - -1.1) - (-2.0 - -1.9)) = 0.1 * (0.1 - (-0.1)) = 0.02
        let logit = 0.1_f32 * ((-1.0_f32 - -1.1_f32) - (-2.0_f32 - -1.9_f32));
        let expected = -log_sigmoid(logit);
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn loss_per_pair_normalize_false_matches_dpo_formula() {
        let pair = make_pair(-0.5, -1.0, -1.5, -1.0, 8, 8);
        let cfg = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let loss = LengthDpo::loss_per_pair(&pair, &cfg).expect("loss_per_pair should succeed");
        let logit = 0.1_f32 * ((-0.5_f32 - -1.0_f32) - (-1.5_f32 - -1.0_f32));
        let expected = -log_sigmoid(logit);
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss={loss}, expected={expected}"
        );
    }

    // ── loss (batch) ──────────────────────────────────────────────────────────

    #[test]
    fn loss_is_mean_not_sum() {
        let pair_a = make_pair(-1.0, -1.1, -2.0, -1.9, 10, 10);
        let pair_b = make_pair(-0.5, -0.6, -1.5, -1.4, 8, 8);
        let cfg = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let loss_a = LengthDpo::loss_per_pair(&pair_a, &cfg).expect("loss_per_pair should succeed");
        let loss_b = LengthDpo::loss_per_pair(&pair_b, &cfg).expect("loss_per_pair should succeed");
        let batch = LengthDpoBatch::new(vec![pair_a, pair_b]);
        let batch_loss = LengthDpo::loss(&batch, &cfg).expect("loss should succeed");
        let expected = (loss_a + loss_b) / 2.0;
        assert!(
            (batch_loss - expected).abs() < 1e-5,
            "batch_loss={batch_loss}, expected mean={expected}"
        );
    }

    // ── implicit_reward_diff ──────────────────────────────────────────────────

    #[test]
    fn implicit_reward_diff_preferred_over_rejected_positive() {
        // chosen_logp > rejected_logp → positive reward diff
        let pair = make_pair(-0.5, -1.0, -2.0, -1.0, 10, 10);
        let cfg = LengthDpoConfig {
            beta: 1.0,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let diff = LengthDpo::implicit_reward_diff(&pair, &cfg)
            .expect("implicit_reward_diff should succeed");
        assert!(
            diff > 0.0,
            "preferred chosen → positive reward diff, got {diff}"
        );
    }

    #[test]
    fn implicit_reward_diff_equal_logps_equals_neg_margin() {
        let pair = make_pair(-1.0, -1.0, -1.0, -1.0, 10, 10);
        let cfg = LengthDpoConfig {
            beta: 0.5,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.3,
        };
        let diff = LengthDpo::implicit_reward_diff(&pair, &cfg)
            .expect("implicit_reward_diff should succeed");
        // β * (norm_c - norm_r) - γ = 0.5 * 0 - 0.3 = -0.3
        assert!(
            (diff - (-0.3_f32)).abs() < 1e-5,
            "equal logps → reward_diff = -gamma, got {diff}"
        );
    }

    #[test]
    fn target_reward_margin_shifts_loss_upward() {
        let pair = make_pair(-1.0, -1.1, -2.0, -1.9, 10, 10);
        let cfg_no_margin = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let cfg_with_margin = LengthDpoConfig {
            target_reward_margin: 1.0,
            ..cfg_no_margin.clone()
        };
        let loss_no =
            LengthDpo::loss_per_pair(&pair, &cfg_no_margin).expect("loss_per_pair should succeed");
        let loss_with = LengthDpo::loss_per_pair(&pair, &cfg_with_margin)
            .expect("loss_per_pair should succeed");
        assert!(
            loss_with > loss_no,
            "positive margin shifts logit down → higher loss; no_margin={loss_no}, with_margin={loss_with}"
        );
    }

    #[test]
    fn normalize_by_length_changes_loss() {
        // chosen_len=10, rejected_len=5 with different per-token log-probs
        let pair = make_pair(-10.0, -11.0, -5.0, -4.5, 10, 5);
        let cfg_no_norm = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        let cfg_norm = LengthDpoConfig {
            normalize_by_length: true,
            ..cfg_no_norm.clone()
        };
        let loss_no_norm =
            LengthDpo::loss_per_pair(&pair, &cfg_no_norm).expect("loss_per_pair should succeed");
        let loss_norm =
            LengthDpo::loss_per_pair(&pair, &cfg_norm).expect("loss_per_pair should succeed");
        // They should differ because normalisation changes the log-ratio
        let differ = (loss_no_norm - loss_norm).abs() > 1e-5;
        assert!(
            differ,
            "normalize_by_length should change the loss; no_norm={loss_no_norm}, norm={loss_norm}"
        );
    }

    // ── Error conditions ──────────────────────────────────────────────────────

    #[test]
    fn loss_empty_batch_returns_error() {
        let batch = LengthDpoBatch::new(vec![]);
        let cfg = LengthDpoConfig::default();
        assert!(
            matches!(LengthDpo::loss(&batch, &cfg), Err(RlhfError::EmptyInput)),
            "empty batch should return EmptyInput"
        );
    }

    #[test]
    fn loss_invalid_beta_returns_error() {
        let batch = LengthDpoBatch::new(vec![make_pair(-1.0, -1.1, -2.0, -1.9, 5, 5)]);
        let cfg = LengthDpoConfig {
            beta: -0.1,
            ..Default::default()
        };
        assert!(
            matches!(
                LengthDpo::loss(&batch, &cfg),
                Err(RlhfError::InvalidBeta { .. })
            ),
            "negative beta should return InvalidBeta"
        );
    }

    #[test]
    fn loss_invalid_lambda_returns_error() {
        let batch = LengthDpoBatch::new(vec![make_pair(-1.0, -1.1, -2.0, -1.9, 5, 5)]);
        let cfg = LengthDpoConfig {
            beta: 0.1,
            length_lambda: -0.1,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        assert!(
            matches!(
                LengthDpo::loss(&batch, &cfg),
                Err(RlhfError::InvalidLambda { .. })
            ),
            "negative length_lambda should return InvalidLambda"
        );
    }

    #[test]
    fn loss_nan_logp_returns_error() {
        let batch = LengthDpoBatch::new(vec![make_pair(f32::NAN, -1.1, -2.0, -1.9, 5, 5)]);
        let cfg = LengthDpoConfig {
            beta: 0.1,
            length_lambda: 0.0,
            normalize_by_length: false,
            target_reward_margin: 0.0,
        };
        assert!(
            matches!(
                LengthDpo::loss(&batch, &cfg),
                Err(RlhfError::NanEncountered)
            ),
            "NaN logp should return NanEncountered"
        );
    }

    // ── LengthDpoBatch helpers ────────────────────────────────────────────────

    #[test]
    fn batch_len_and_is_empty() {
        let empty = LengthDpoBatch::new(vec![]);
        assert!(empty.is_empty(), "empty batch: is_empty() should be true");
        assert_eq!(empty.len(), 0);

        let batch = LengthDpoBatch::new(vec![
            make_pair(-1.0, -1.1, -2.0, -1.9, 5, 5),
            make_pair(-0.5, -0.6, -1.5, -1.4, 8, 8),
        ]);
        assert!(
            !batch.is_empty(),
            "non-empty batch: is_empty() should be false"
        );
        assert_eq!(batch.len(), 2);
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = LengthDpoConfig::default();
        assert!((cfg.beta - 0.1_f32).abs() < 1e-6, "default beta=0.1");
        assert!(
            (cfg.length_lambda - 0.01_f32).abs() < 1e-6,
            "default length_lambda=0.01"
        );
        assert!(cfg.normalize_by_length, "default normalize_by_length=true");
        assert!(cfg.target_reward_margin.abs() < 1e-6, "default margin=0.0");
    }
}
