//! Cringe Loss: Learning what language NOT to model.
//!
//! Reference: Adolphs et al. 2022, "The CRINGE Loss: Learning what language NOT
//! to model", arXiv:2211.05100.
//!
//! Combines two complementary training signals:
//!
//! 1. **Positive NLL**: Maximize probability on good/preferred continuations via
//!    standard negative log-likelihood: `L_pos = -log π(y_pos|x)`.
//!
//! 2. **Negative hinge**: Penalize high probability on forbidden/cringe
//!    continuations: `L_neg = max(0, log π(y_neg|x) - margin)`.
//!    The margin (typically negative, e.g. -1.0) defines the log-prob threshold
//!    above which a negative sample is penalized.
//!
//! Combined batch loss:
//! ```text
//! L = α * mean(L_pos) + (1 - α) * mean(L_neg)
//! ```
//! where α is `positive_weight`.

use crate::error::{RlhfError, RlhfResult};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the Cringe loss.
#[derive(Debug, Clone)]
pub struct CringeConfig {
    /// Margin for the negative hinge (log-prob threshold).
    ///
    /// A negative value (e.g. -1.0) means: the policy must drive
    /// `log π(y_neg|x)` below `margin` before the penalty disappears.
    /// Must be finite.
    pub margin: f32,

    /// Weight α ∈ (0, 1) on the positive NLL loss.
    ///
    /// The negative hinge receives weight `(1 - α)`.
    /// Must be strictly in the open interval (0, 1).
    pub positive_weight: f32,
}

impl Default for CringeConfig {
    fn default() -> Self {
        Self {
            margin: -1.0,
            positive_weight: 0.5,
        }
    }
}

// ── Sample and Batch ──────────────────────────────────────────────────────────

/// A single (positive, optional-negative) Cringe training example.
#[derive(Debug, Clone)]
pub struct CringeSample {
    /// log π(y_pos|x): log-probability of the desired continuation under the
    /// current policy.
    pub log_prob_pos: f32,

    /// log π(y_neg|x): log-probability of the forbidden continuation.
    ///
    /// `None` means this sample only contributes to the positive loss term.
    pub log_prob_neg: Option<f32>,
}

/// A batch of Cringe training examples.
#[derive(Debug, Clone)]
pub struct CringeBatch {
    /// Individual training examples in this batch.
    pub samples: Vec<CringeSample>,
}

impl CringeBatch {
    /// Create a new batch from the given samples.
    pub fn new(samples: Vec<CringeSample>) -> Self {
        Self { samples }
    }

    /// Number of samples in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if the batch contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ── Loss computation ──────────────────────────────────────────────────────────

/// Cringe loss computation.
///
/// Implements the algorithm from Adolphs et al. 2022:
/// - Positive arm: standard NLL on preferred continuations.
/// - Negative arm: hinge-on-log-prob for forbidden continuations.
pub struct CringeLoss;

impl CringeLoss {
    /// Positive NLL loss for a single sample.
    ///
    /// Returns `-log_prob_pos`.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::NanEncountered`] if the computed loss is NaN.
    pub fn positive_loss(log_prob_pos: f32) -> RlhfResult<f32> {
        let loss = -log_prob_pos;
        if loss.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(loss)
    }

    /// Negative hinge loss for a single sample.
    ///
    /// Returns `max(0, log_prob_neg - margin)`.
    ///
    /// When `log_prob_neg < margin` the hinge is inactive (returns 0.0).
    /// When `log_prob_neg > margin` the hinge is active and returns
    /// the excess above the margin.
    #[inline]
    pub fn negative_hinge(log_prob_neg: f32, margin: f32) -> f32 {
        let excess = log_prob_neg - margin;
        if excess > 0.0 { excess } else { 0.0 }
    }

    /// Pseudo-gradient of the negative hinge w.r.t. `log_prob_neg`.
    ///
    /// Returns `1.0` when the hinge is active (`log_prob_neg > margin`),
    /// and `0.0` when it is inactive (`log_prob_neg <= margin`).
    ///
    /// (At the boundary the subgradient is taken as 0.)
    #[inline]
    pub fn negative_hinge_grad(log_prob_neg: f32, margin: f32) -> f32 {
        if log_prob_neg > margin { 1.0 } else { 0.0 }
    }

    /// Compute the Cringe loss for a batch.
    ///
    /// # Returns
    ///
    /// A tuple `(total_loss, pos_loss_mean, neg_loss_mean)` where:
    /// - `total_loss = α * pos_loss_mean + (1 - α) * neg_loss_mean`
    /// - `pos_loss_mean` = mean of `-log π(y_pos|x)` over all samples
    /// - `neg_loss_mean` = mean of `max(0, log π(y_neg|x) - margin)` over
    ///   samples that have a negative example; 0.0 if none do.
    ///
    /// # Errors
    ///
    /// - [`RlhfError::EmptyInput`] — batch has no samples.
    /// - [`RlhfError::InvalidMargin`] — `margin` is not finite.
    /// - [`RlhfError::InvalidLambda`] — `positive_weight` is not in (0, 1).
    /// - [`RlhfError::NanEncountered`] — any intermediate loss value is NaN.
    pub fn compute(batch: &CringeBatch, cfg: &CringeConfig) -> RlhfResult<(f32, f32, f32)> {
        // ── Validation ────────────────────────────────────────────────────────
        if batch.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if !cfg.margin.is_finite() {
            return Err(RlhfError::InvalidMargin { margin: cfg.margin });
        }
        if cfg.positive_weight <= 0.0
            || cfg.positive_weight >= 1.0
            || !cfg.positive_weight.is_finite()
        {
            return Err(RlhfError::InvalidLambda {
                lambda: cfg.positive_weight,
            });
        }

        // ── Positive loss ─────────────────────────────────────────────────────
        let mut pos_sum = 0.0_f32;
        for sample in &batch.samples {
            let lp = -sample.log_prob_pos;
            if lp.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
            pos_sum += lp;
        }
        let pos_mean = pos_sum / batch.len() as f32;

        // ── Negative hinge loss ───────────────────────────────────────────────
        let mut neg_sum = 0.0_f32;
        let mut neg_count = 0_usize;
        for sample in &batch.samples {
            if let Some(lp_neg) = sample.log_prob_neg {
                let hinge = Self::negative_hinge(lp_neg, cfg.margin);
                if hinge.is_nan() {
                    return Err(RlhfError::NanEncountered);
                }
                neg_sum += hinge;
                neg_count += 1;
            }
        }
        let neg_mean = if neg_count == 0 {
            0.0_f32
        } else {
            neg_sum / neg_count as f32
        };

        // ── Combined loss ─────────────────────────────────────────────────────
        let alpha = cfg.positive_weight;
        let total = alpha * pos_mean + (1.0 - alpha) * neg_mean;
        if total.is_nan() {
            return Err(RlhfError::NanEncountered);
        }

        Ok((total, pos_mean, neg_mean))
    }

    /// Convenience: compute only the positive NLL loss over a slice of
    /// log-probabilities.
    ///
    /// Returns `mean(-log_prob)` over the slice.
    ///
    /// # Errors
    ///
    /// - [`RlhfError::EmptyInput`] — slice is empty.
    /// - [`RlhfError::NanEncountered`] — any computed loss is NaN.
    pub fn positive_only(log_probs: &[f32]) -> RlhfResult<f32> {
        if log_probs.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        let mut sum = 0.0_f32;
        for &lp in log_probs {
            let loss = -lp;
            if loss.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
            sum += loss;
        }
        Ok(sum / log_probs.len() as f32)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── positive_loss ─────────────────────────────────────────────────────────

    #[test]
    fn positive_loss_finite_log_prob() {
        // L_pos = -log_prob_pos
        let lp = -0.5_f32;
        let loss = CringeLoss::positive_loss(lp).expect("positive_loss should succeed");
        assert!(
            (loss - 0.5_f32).abs() < 1e-6,
            "positive_loss({lp}) = {loss}, expected 0.5"
        );
    }

    #[test]
    fn positive_loss_negative_lp_gives_positive_loss() {
        let lp = -2.3_f32;
        let loss = CringeLoss::positive_loss(lp).expect("positive_loss should succeed");
        assert!(
            (loss - 2.3_f32).abs() < 1e-6,
            "positive_loss({lp}) = {loss}, expected 2.3"
        );
    }

    #[test]
    fn positive_loss_nan_returns_error() {
        let result = CringeLoss::positive_loss(f32::NAN);
        assert!(
            matches!(result, Err(RlhfError::NanEncountered)),
            "NaN log_prob_pos should return NanEncountered"
        );
    }

    // ── negative_hinge ────────────────────────────────────────────────────────

    #[test]
    fn negative_hinge_inactive_when_below_margin() {
        // log_prob_neg = -2.0, margin = -1.0 → excess = -1.0 < 0 → 0.0
        let hinge = CringeLoss::negative_hinge(-2.0, -1.0);
        assert!(
            hinge.abs() < 1e-6,
            "hinge should be 0 when log_prob_neg < margin, got {hinge}"
        );
    }

    #[test]
    fn negative_hinge_active_when_above_margin() {
        // log_prob_neg = -0.5, margin = -1.0 → excess = 0.5 → 0.5
        let hinge = CringeLoss::negative_hinge(-0.5, -1.0);
        assert!(
            (hinge - 0.5_f32).abs() < 1e-6,
            "hinge should be 0.5, got {hinge}"
        );
    }

    #[test]
    fn negative_hinge_at_boundary_is_zero() {
        // log_prob_neg == margin → excess = 0.0 → 0.0
        let hinge = CringeLoss::negative_hinge(-1.0, -1.0);
        assert!(
            hinge.abs() < 1e-6,
            "hinge at boundary should be 0, got {hinge}"
        );
    }

    // ── negative_hinge_grad ───────────────────────────────────────────────────

    #[test]
    fn negative_hinge_grad_active_returns_one() {
        let grad = CringeLoss::negative_hinge_grad(-0.5, -1.0);
        assert!(
            (grad - 1.0_f32).abs() < 1e-6,
            "grad should be 1.0 when active, got {grad}"
        );
    }

    #[test]
    fn negative_hinge_grad_inactive_returns_zero() {
        let grad = CringeLoss::negative_hinge_grad(-2.0, -1.0);
        assert!(
            grad.abs() < 1e-6,
            "grad should be 0.0 when inactive, got {grad}"
        );
    }

    #[test]
    fn negative_hinge_grad_at_boundary_returns_zero() {
        // Subgradient convention: take 0 at boundary
        let grad = CringeLoss::negative_hinge_grad(-1.0, -1.0);
        assert!(
            grad.abs() < 1e-6,
            "grad at boundary should be 0.0, got {grad}"
        );
    }

    // ── compute: positive-only batch ──────────────────────────────────────────

    #[test]
    fn compute_positive_only_batch_neg_mean_is_zero() {
        let batch = CringeBatch::new(vec![
            CringeSample {
                log_prob_pos: -1.0,
                log_prob_neg: None,
            },
            CringeSample {
                log_prob_pos: -2.0,
                log_prob_neg: None,
            },
        ]);
        let cfg = CringeConfig::default();
        let (total, pos_mean, neg_mean) =
            CringeLoss::compute(&batch, &cfg).expect("compute should succeed");
        assert!(
            neg_mean.abs() < 1e-6,
            "neg_mean should be 0 when no negatives, got {neg_mean}"
        );
        // pos_mean = mean(1.0, 2.0) = 1.5
        assert!(
            (pos_mean - 1.5_f32).abs() < 1e-6,
            "pos_mean should be 1.5, got {pos_mean}"
        );
        // total = 0.5 * 1.5 + 0.5 * 0.0 = 0.75
        assert!(
            (total - 0.75_f32).abs() < 1e-6,
            "total should be 0.75, got {total}"
        );
    }

    #[test]
    fn compute_mixed_pos_and_neg() {
        // pos samples: log_prob_pos = -1.0, -2.0
        // neg sample:  log_prob_neg = -0.5 (margin = -1.0 → hinge = 0.5)
        let batch = CringeBatch::new(vec![
            CringeSample {
                log_prob_pos: -1.0,
                log_prob_neg: Some(-0.5),
            },
            CringeSample {
                log_prob_pos: -2.0,
                log_prob_neg: None,
            },
        ]);
        let cfg = CringeConfig::default(); // margin=-1.0, alpha=0.5
        let (total, pos_mean, neg_mean) =
            CringeLoss::compute(&batch, &cfg).expect("compute should succeed");

        // pos_mean = mean(1.0, 2.0) = 1.5
        assert!((pos_mean - 1.5_f32).abs() < 1e-6, "pos_mean = {pos_mean}");
        // neg_mean = 0.5 (only one negative)
        assert!((neg_mean - 0.5_f32).abs() < 1e-6, "neg_mean = {neg_mean}");
        // total = 0.5 * 1.5 + 0.5 * 0.5 = 1.0
        assert!(
            (total - 1.0_f32).abs() < 1e-6,
            "total = {total}, expected 1.0"
        );
    }

    #[test]
    fn compute_balanced_weighting() {
        // With positive_weight = 0.5: equal blend of pos and neg
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: -1.0,
            log_prob_neg: Some(-0.5),
        }]);
        let cfg = CringeConfig {
            margin: -1.0,
            positive_weight: 0.5,
        };
        let (total, pos_mean, neg_mean) =
            CringeLoss::compute(&batch, &cfg).expect("compute should succeed");
        let expected = 0.5 * pos_mean + 0.5 * neg_mean;
        assert!(
            (total - expected).abs() < 1e-6,
            "total = {total}, expected = {expected}"
        );
    }

    #[test]
    fn compute_returns_correct_tuple_structure() {
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: -1.0,
            log_prob_neg: Some(-0.3),
        }]);
        let cfg = CringeConfig::default();
        let result = CringeLoss::compute(&batch, &cfg);
        assert!(result.is_ok(), "compute should succeed");
        let (total, pos_mean, neg_mean) = result.expect("result should be present");
        assert!(total.is_finite(), "total should be finite");
        assert!(pos_mean.is_finite(), "pos_mean should be finite");
        assert!(neg_mean.is_finite(), "neg_mean should be finite");
    }

    #[test]
    fn compute_all_negatives_no_pos_only_path() {
        // All samples have both pos and neg — ensure neg arm is fully averaged
        let batch = CringeBatch::new(vec![
            CringeSample {
                log_prob_pos: -1.0,
                log_prob_neg: Some(-0.5),
            },
            CringeSample {
                log_prob_pos: -2.0,
                log_prob_neg: Some(-1.5),
            },
        ]);
        let cfg = CringeConfig::default(); // margin=-1.0
        // neg for sample 1: hinge(-0.5, -1.0) = 0.5
        // neg for sample 2: hinge(-1.5, -1.0) = 0.0 (below margin)
        let (_, _, neg_mean) = CringeLoss::compute(&batch, &cfg).expect("compute should succeed");
        let expected_neg_mean = (0.5_f32 + 0.0_f32) / 2.0;
        assert!(
            (neg_mean - expected_neg_mean).abs() < 1e-6,
            "neg_mean = {neg_mean}, expected {expected_neg_mean}"
        );
    }

    // ── positive_only ─────────────────────────────────────────────────────────

    #[test]
    fn positive_only_matches_manual_mean() {
        let log_probs = [-1.0_f32, -2.0, -3.0];
        let result = CringeLoss::positive_only(&log_probs).expect("positive_only should succeed");
        let expected = (1.0_f32 + 2.0 + 3.0) / 3.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "positive_only = {result}, expected {expected}"
        );
    }

    #[test]
    fn positive_only_single_sample() {
        let log_probs = [-0.7_f32];
        let result = CringeLoss::positive_only(&log_probs).expect("positive_only should succeed");
        assert!(
            (result - 0.7_f32).abs() < 1e-6,
            "positive_only single = {result}"
        );
    }

    // ── Error conditions ──────────────────────────────────────────────────────

    #[test]
    fn compute_empty_batch_returns_error() {
        let batch = CringeBatch::new(vec![]);
        let cfg = CringeConfig::default();
        assert!(
            matches!(
                CringeLoss::compute(&batch, &cfg),
                Err(RlhfError::EmptyInput)
            ),
            "empty batch should return EmptyInput"
        );
    }

    #[test]
    fn positive_only_empty_returns_error() {
        assert!(
            matches!(CringeLoss::positive_only(&[]), Err(RlhfError::EmptyInput)),
            "empty slice should return EmptyInput"
        );
    }

    #[test]
    fn compute_infinite_margin_returns_error() {
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: -1.0,
            log_prob_neg: None,
        }]);
        let cfg = CringeConfig {
            margin: f32::INFINITY,
            positive_weight: 0.5,
        };
        assert!(
            matches!(
                CringeLoss::compute(&batch, &cfg),
                Err(RlhfError::InvalidMargin { .. })
            ),
            "infinite margin should return InvalidMargin"
        );
    }

    #[test]
    fn compute_positive_weight_zero_returns_error() {
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: -1.0,
            log_prob_neg: None,
        }]);
        let cfg = CringeConfig {
            margin: -1.0,
            positive_weight: 0.0,
        };
        assert!(
            matches!(
                CringeLoss::compute(&batch, &cfg),
                Err(RlhfError::InvalidLambda { .. })
            ),
            "positive_weight=0.0 should return InvalidLambda"
        );
    }

    #[test]
    fn compute_positive_weight_one_returns_error() {
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: -1.0,
            log_prob_neg: None,
        }]);
        let cfg = CringeConfig {
            margin: -1.0,
            positive_weight: 1.0,
        };
        assert!(
            matches!(
                CringeLoss::compute(&batch, &cfg),
                Err(RlhfError::InvalidLambda { .. })
            ),
            "positive_weight=1.0 should return InvalidLambda"
        );
    }

    #[test]
    fn compute_nan_log_prob_pos_returns_error() {
        let batch = CringeBatch::new(vec![CringeSample {
            log_prob_pos: f32::NAN,
            log_prob_neg: None,
        }]);
        let cfg = CringeConfig::default();
        assert!(
            matches!(
                CringeLoss::compute(&batch, &cfg),
                Err(RlhfError::NanEncountered)
            ),
            "NaN log_prob_pos should return NanEncountered"
        );
    }

    // ── CringeBatch helpers ───────────────────────────────────────────────────

    #[test]
    fn batch_len_and_is_empty() {
        let empty = CringeBatch::new(vec![]);
        assert!(empty.is_empty(), "empty batch: is_empty() should be true");
        assert_eq!(empty.len(), 0);

        let batch = CringeBatch::new(vec![
            CringeSample {
                log_prob_pos: -1.0,
                log_prob_neg: None,
            },
            CringeSample {
                log_prob_pos: -2.0,
                log_prob_neg: None,
            },
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
        let cfg = CringeConfig::default();
        assert!(
            (cfg.margin - (-1.0_f32)).abs() < 1e-6,
            "default margin should be -1.0, got {}",
            cfg.margin
        );
        assert!(
            (cfg.positive_weight - 0.5_f32).abs() < 1e-6,
            "default positive_weight should be 0.5, got {}",
            cfg.positive_weight
        );
    }
}
