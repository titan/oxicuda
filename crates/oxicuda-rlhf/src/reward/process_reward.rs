//! Process Reward Modelling (PRM) loss for step-level reward learning.
//!
//! References:
//! * Lightman et al. 2023, "Let's Verify Step by Step", arXiv:2305.20050.
//! * Wang et al. 2024, "Math-Shepherd: Verify and Reinforce LLMs Step-by-step",
//!   arXiv:2312.08935.
//!
//! Unlike Outcome Reward Models (ORM) which only score the final answer,
//! Process Reward Models score every reasoning step individually.  The training
//! objective is binary cross-entropy (BCE) at each step, optionally combined
//! with a solution-level BCE term.
//!
//! At inference time, per-step logits are aggregated into a solution-level
//! score via [`PrmAggregation`] (product, minimum, mean, last-step, or a
//! weighted average).  Solutions can be ranked by [`prm_rank_solutions`].

use crate::error::{RlhfError, RlhfResult};

// ── Aggregation strategy ──────────────────────────────────────────────────────

/// How to aggregate per-step PRM scores into a solution-level probability.
#[derive(Debug, Clone)]
pub enum PrmAggregation {
    /// Product of per-step probabilities: P(correct) = Π_t σ(logit_t).
    ///
    /// This is the strictest aggregation: the entire chain is deemed correct
    /// only when every step is correct.
    Product,
    /// Minimum (most conservative): min_t σ(logit_t).
    ///
    /// Bottlenecked by the worst step.
    Minimum,
    /// Arithmetic mean: (1/T) Σ_t σ(logit_t).
    Mean,
    /// Only the final step's score (ORM-equivalent when applied to the last step).
    LastStep,
    /// Weighted average with explicit per-step weights (must sum > 0).
    Weighted { weights: Vec<f32> },
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for PRM training.
#[derive(Debug, Clone)]
pub struct PrmConfig {
    /// Maximum number of steps per solution (default 32).
    ///
    /// `prm_loss` returns an error when `n_steps > n_steps_max`.
    pub n_steps_max: usize,
    /// Aggregation strategy for converting per-step scores to a solution score.
    pub aggregation: PrmAggregation,
    /// Weight for the step-level BCE loss term (default 1.0).
    pub step_loss_weight: f32,
    /// Weight for the solution-level BCE loss term (default 0.0).
    ///
    /// When `> 0`, the last step's logit is used as the solution-level score.
    pub solution_loss_weight: f32,
    /// Label smoothing ε: effective_label = y * (1 − ε) + ε / 2 (default 0.0).
    pub label_smoothing: f32,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            n_steps_max: 32,
            aggregation: PrmAggregation::Product,
            step_loss_weight: 1.0,
            solution_loss_weight: 0.0,
            label_smoothing: 0.0,
        }
    }
}

// ── Model output ──────────────────────────────────────────────────────────────

/// PRM model output: one raw logit per reasoning step.
///
/// Positive logit → the model predicts that step is correct.
#[derive(Debug, Clone)]
pub struct PrmOutput {
    /// Raw model logit for each step.
    pub step_logits: Vec<f32>,
}

impl PrmOutput {
    /// Construct a [`PrmOutput`] from a vector of per-step logits.
    pub fn new(step_logits: Vec<f32>) -> Self {
        Self { step_logits }
    }

    /// Number of steps in this output.
    pub fn n_steps(&self) -> usize {
        self.step_logits.len()
    }
}

// ── Labels ────────────────────────────────────────────────────────────────────

/// Ground-truth labels for a PRM training example.
#[derive(Debug, Clone)]
pub struct PrmLabel {
    /// Binary correctness label for each step ∈ [0.0, 1.0].
    ///
    /// 1.0 = step is correct, 0.0 = step is incorrect. Soft labels (values
    /// strictly between 0 and 1) are also allowed, e.g. from human raters.
    pub step_labels: Vec<f32>,
    /// Optional solution-level label consumed by the `solution_loss_weight` term.
    ///
    /// When `None` or when `cfg.solution_loss_weight == 0`, no solution-level
    /// loss is added.
    pub solution_label: Option<f32>,
}

impl PrmLabel {
    /// Number of step labels.
    pub fn n_steps(&self) -> usize {
        self.step_labels.len()
    }
}

// ── Loss result ───────────────────────────────────────────────────────────────

/// Result of a single PRM loss computation.
#[derive(Debug, Clone)]
pub struct PrmLossResult {
    /// Total weighted loss: `step_loss_weight * step_loss + solution_loss_weight * solution_loss`.
    pub total_loss: f32,
    /// Mean step-level BCE (before `step_loss_weight` scaling).
    pub step_loss: f32,
    /// Solution-level BCE (before `solution_loss_weight` scaling); 0.0 if not applicable.
    pub solution_loss: f32,
    /// Per-step BCE losses (length = n_steps).
    pub per_step_losses: Vec<f32>,
}

// ── Numerics ──────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid: σ(x) = 1 / (1 + exp(−x)).
///
/// Uses two branches to avoid floating-point overflow:
/// * x ≥ 0: 1 / (1 + exp(−x))       — exp(−x) stays ≤ 1
/// * x < 0: exp(x) / (1 + exp(x))   — exp(x) stays ≤ 1
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0_f32 / (1.0_f32 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0_f32 + ex)
    }
}

/// Binary cross-entropy from a raw logit with optional label smoothing.
///
/// Effective label after smoothing:
/// ```text
/// smoothed = label * (1 − label_smoothing) + label_smoothing / 2
/// ```
///
/// Loss (numerically stable):
/// ```text
/// bce = −smoothed * log σ(logit) − (1 − smoothed) * log σ(−logit)
/// ```
///
/// Both `log σ(logit)` and `log σ(−logit)` are computed with the two-branch
/// log-sigmoid formula to guarantee finite results for extreme logits.
#[inline]
pub fn bce_with_logit(logit: f32, label: f32, label_smoothing: f32) -> f32 {
    let smoothed = label * (1.0 - label_smoothing) + label_smoothing / 2.0;
    let log_sig_pos = log_sigmoid_stable(logit);
    let log_sig_neg = log_sigmoid_stable(-logit);
    -smoothed * log_sig_pos - (1.0 - smoothed) * log_sig_neg
}

/// Internal helper: numerically stable log-sigmoid.
#[inline]
fn log_sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0_f32 + (-x).exp()).ln()
    } else {
        x - (1.0_f32 + x.exp()).ln()
    }
}

// ── Core functions ────────────────────────────────────────────────────────────

/// Compute the PRM training loss for a single (output, label) pair.
///
/// # Errors
///
/// * [`RlhfError::EmptyInput`] — zero steps.
/// * [`RlhfError::DimensionMismatch`] — `output.n_steps() != label.n_steps()`.
/// * [`RlhfError::Internal`] — `n_steps > n_steps_max`.
/// * [`RlhfError::NanEncountered`] — any value is NaN.
pub fn prm_loss(
    output: &PrmOutput,
    label: &PrmLabel,
    cfg: &PrmConfig,
) -> RlhfResult<PrmLossResult> {
    let n = output.n_steps();
    if n == 0 || label.n_steps() == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if n != label.n_steps() {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: label.n_steps(),
        });
    }
    if n > cfg.n_steps_max {
        return Err(RlhfError::Internal {
            msg: format!("n_steps {n} exceeds n_steps_max {}", cfg.n_steps_max),
        });
    }

    // ── Per-step BCE ──────────────────────────────────────────────────────────
    let mut per_step_losses = Vec::with_capacity(n);
    for t in 0..n {
        let logit = output.step_logits[t];
        let lbl = label.step_labels[t];
        if !logit.is_finite() || !lbl.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        let l = bce_with_logit(logit, lbl, cfg.label_smoothing);
        if l.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        per_step_losses.push(l);
    }

    // ── Step-level loss (mean) ─────────────────────────────────────────────────
    let step_loss = per_step_losses.iter().sum::<f32>() / n as f32;

    // ── Solution-level loss (last-step logit vs solution_label) ───────────────
    let solution_loss = if cfg.solution_loss_weight > 0.0 {
        if let Some(sol_lbl) = label.solution_label {
            let sol_logit = output.step_logits[n - 1];
            let sl = bce_with_logit(sol_logit, sol_lbl, cfg.label_smoothing);
            if sl.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
            sl
        } else {
            0.0
        }
    } else {
        0.0
    };

    let total_loss = cfg.step_loss_weight * step_loss + cfg.solution_loss_weight * solution_loss;

    if total_loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }

    Ok(PrmLossResult {
        total_loss,
        step_loss,
        solution_loss,
        per_step_losses,
    })
}

/// Compute the mean PRM loss over a batch of (output, label) pairs.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if `outputs` is empty, or propagates any
/// error from [`prm_loss`].  Returns an error if the lengths of `outputs` and
/// `labels` differ.
pub fn prm_loss_batch(
    outputs: &[PrmOutput],
    labels: &[PrmLabel],
    cfg: &PrmConfig,
) -> RlhfResult<f32> {
    if outputs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if outputs.len() != labels.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: outputs.len(),
            got: labels.len(),
        });
    }
    let total: f32 = outputs
        .iter()
        .zip(labels.iter())
        .map(|(o, l)| prm_loss(o, l, cfg).map(|r| r.total_loss))
        .collect::<RlhfResult<Vec<f32>>>()?
        .into_iter()
        .sum();
    let mean = total / outputs.len() as f32;
    if mean.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(mean)
}

/// Aggregate per-step logits into a solution-level probability in [0, 1].
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if the output has no steps,
/// [`RlhfError::DimensionMismatch`] if a `Weighted` aggregation weight vector
/// has wrong length, and [`RlhfError::NanEncountered`] if any score is NaN.
pub fn prm_aggregate_score(output: &PrmOutput, cfg: &PrmConfig) -> RlhfResult<f32> {
    let n = output.n_steps();
    if n == 0 {
        return Err(RlhfError::EmptyInput);
    }

    let probs: Vec<f32> = output.step_logits.iter().map(|&l| sigmoid(l)).collect();

    let score = match &cfg.aggregation {
        PrmAggregation::Product => probs.iter().copied().fold(1.0_f32, |acc, p| acc * p),

        PrmAggregation::Minimum => probs.iter().copied().fold(f32::INFINITY, f32::min),

        PrmAggregation::Mean => probs.iter().sum::<f32>() / n as f32,

        PrmAggregation::LastStep => probs[n - 1],

        PrmAggregation::Weighted { weights } => {
            if weights.len() != n {
                return Err(RlhfError::DimensionMismatch {
                    expected: n,
                    got: weights.len(),
                });
            }
            let weighted_sum: f32 = probs.iter().zip(weights.iter()).map(|(&p, &w)| p * w).sum();
            let weight_total: f32 = weights.iter().sum();
            weighted_sum / weight_total
        }
    };

    if score.is_nan() {
        return Err(RlhfError::NanEncountered);
    }

    // Clamp to [0, 1] for numerical safety (floating-point product can drift).
    Ok(score.clamp(0.0, 1.0))
}

/// Rank multiple candidate solutions by their aggregated PRM score.
///
/// Returns a vector of indices into `outputs`, sorted in descending order of
/// score (best solution first).  Uses a stable sort so that equal scores
/// preserve the original order.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if `outputs` is empty, or propagates any
/// error from [`prm_aggregate_score`].
pub fn prm_rank_solutions(outputs: &[PrmOutput], cfg: &PrmConfig) -> RlhfResult<Vec<usize>> {
    if outputs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }

    let scores: Vec<f32> = outputs
        .iter()
        .map(|o| prm_aggregate_score(o, cfg))
        .collect::<RlhfResult<Vec<f32>>>()?;

    let mut indices: Vec<usize> = (0..outputs.len()).collect();
    // Stable sort descending — treat NaN as lowest (shouldn't occur after
    // the aggregate_score check above, but guard anyway).
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(indices)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sigmoid ───────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_at_zero() {
        assert!(
            (sigmoid(0.0) - 0.5).abs() < 1e-6,
            "σ(0) should be 0.5, got {}",
            sigmoid(0.0)
        );
    }

    #[test]
    fn sigmoid_large_positive_near_one() {
        assert!(
            (sigmoid(100.0) - 1.0).abs() < 1e-5,
            "σ(100) should be ~1, got {}",
            sigmoid(100.0)
        );
    }

    #[test]
    fn sigmoid_large_negative_near_zero() {
        assert!(
            sigmoid(-100.0) < 1e-5,
            "σ(-100) should be ~0, got {}",
            sigmoid(-100.0)
        );
    }

    #[test]
    fn sigmoid_is_monotone() {
        let vals: Vec<f32> = (-5..=5).map(|i| sigmoid(i as f32)).collect();
        for w in vals.windows(2) {
            assert!(w[1] > w[0], "sigmoid must be strictly increasing");
        }
    }

    // ── bce_with_logit ────────────────────────────────────────────────────────

    #[test]
    fn bce_logit_zero_label_one_equals_ln2() {
        // logit=0 → σ(0)=0.5 → BCE = -log(0.5) = ln2 ≈ 0.6931
        let bce = bce_with_logit(0.0, 1.0, 0.0);
        let expected = 2.0_f32.ln();
        assert!(
            (bce - expected).abs() < 1e-5,
            "BCE(logit=0, label=1) = {bce}, expected {expected}"
        );
    }

    #[test]
    fn bce_large_positive_logit_label_one_near_zero() {
        // logit=100, label=1 → loss ≈ 0
        let bce = bce_with_logit(100.0, 1.0, 0.0);
        assert!(bce < 1e-4, "BCE(100, 1) should be ~0, got {bce}");
    }

    #[test]
    fn bce_large_negative_logit_label_zero_near_zero() {
        // logit=-100, label=0 → loss = -log σ(-(-100)) = -log σ(100) ≈ 0
        let bce = bce_with_logit(-100.0, 0.0, 0.0);
        assert!(bce < 1e-4, "BCE(-100, 0) should be ~0, got {bce}");
    }

    #[test]
    fn bce_symmetry_at_logit_zero_label_half() {
        // With label=0.5 and logit=0: smoothed=0.5, log_sig_pos=log_sig_neg=-ln2
        // BCE = -0.5*(-ln2) - 0.5*(-ln2) = ln2
        let bce = bce_with_logit(0.0, 0.5, 0.0);
        let expected = 2.0_f32.ln();
        assert!(
            (bce - expected).abs() < 1e-5,
            "BCE(0, 0.5, 0) = {bce}, expected {expected}"
        );
    }

    #[test]
    fn bce_label_smoothing_moves_toward_half() {
        // Smoothed label for (label=1, smooth=0.1) = 0.95 < 1
        // → loss should be higher than without smoothing at logit=0
        let bce_no_smooth = bce_with_logit(0.0, 1.0, 0.0);
        let bce_smooth = bce_with_logit(0.0, 1.0, 0.1);
        // With label=1, smoothing makes effective label = 0.95.
        // BCE(0.95) = -0.95*log(0.5) - 0.05*log(0.5) = ln2 for both here —
        // but smoothed shifts effective target closer to 0.5, so when logit≠0
        // the effect becomes visible. At logit=0 both give ln2 by symmetry;
        // let's check with a positive logit instead.
        let bce_no_smooth_pos = bce_with_logit(2.0, 1.0, 0.0);
        let bce_smooth_pos = bce_with_logit(2.0, 1.0, 0.1);
        // Smoothing pulls effective label below 1.0, slightly increasing the loss.
        assert!(
            bce_smooth_pos > bce_no_smooth_pos,
            "smoothing should increase loss when logit is correct"
        );
        // Both must be finite
        assert!(bce_no_smooth.is_finite());
        assert!(bce_smooth.is_finite());
    }

    // ── prm_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn prm_loss_three_steps_all_zero_logits_label_one() {
        // logit=0 for all 3 steps, label=1 → each step BCE = ln2
        let output = PrmOutput::new(vec![0.0, 0.0, 0.0]);
        let label = PrmLabel {
            step_labels: vec![1.0, 1.0, 1.0],
            solution_label: None,
        };
        let cfg = PrmConfig::default();
        let result = prm_loss(&output, &label, &cfg).expect("prm_loss should succeed");
        let expected_step_loss = 2.0_f32.ln();
        assert!(
            (result.step_loss - expected_step_loss).abs() < 1e-5,
            "step_loss = {}, expected {}",
            result.step_loss,
            expected_step_loss
        );
        for &l in &result.per_step_losses {
            assert!(
                (l - expected_step_loss).abs() < 1e-5,
                "per_step_loss = {l}, expected {expected_step_loss}"
            );
        }
    }

    #[test]
    fn prm_loss_large_logit_label_one_near_zero() {
        // logit=10, label=1 → nearly zero loss
        let output = PrmOutput::new(vec![10.0]);
        let label = PrmLabel {
            step_labels: vec![1.0],
            solution_label: None,
        };
        let cfg = PrmConfig::default();
        let result = prm_loss(&output, &label, &cfg).expect("prm_loss should succeed");
        assert!(
            result.total_loss < 1e-3,
            "high-confidence correct step should have ~0 loss, got {}",
            result.total_loss
        );
    }

    #[test]
    fn prm_loss_dimension_mismatch_returns_error() {
        let output = PrmOutput::new(vec![0.0, 0.0]);
        let label = PrmLabel {
            step_labels: vec![1.0],
            solution_label: None,
        };
        let cfg = PrmConfig::default();
        assert!(
            matches!(
                prm_loss(&output, &label, &cfg),
                Err(RlhfError::DimensionMismatch { .. })
            ),
            "dimension mismatch should return DimensionMismatch"
        );
    }

    #[test]
    fn prm_loss_exceeds_n_steps_max_returns_error() {
        let n = 33_usize; // > default n_steps_max=32
        let output = PrmOutput::new(vec![0.0; n]);
        let label = PrmLabel {
            step_labels: vec![1.0; n],
            solution_label: None,
        };
        let cfg = PrmConfig::default();
        assert!(
            matches!(
                prm_loss(&output, &label, &cfg),
                Err(RlhfError::Internal { .. })
            ),
            "n_steps > n_steps_max should return Internal error"
        );
    }

    // ── prm_aggregate_score ───────────────────────────────────────────────────

    #[test]
    fn aggregate_product_decreases_with_more_uncertain_steps() {
        let cfg_4 = PrmConfig {
            aggregation: PrmAggregation::Product,
            ..Default::default()
        };
        // 2 steps with logit=0 (prob=0.5 each): product = 0.25
        let out2 = PrmOutput::new(vec![0.0, 0.0]);
        let score2 =
            prm_aggregate_score(&out2, &cfg_4).expect("prm_aggregate_score should succeed");
        // 4 steps: product = 0.0625
        let out4 = PrmOutput::new(vec![0.0, 0.0, 0.0, 0.0]);
        let score4 =
            prm_aggregate_score(&out4, &cfg_4).expect("prm_aggregate_score should succeed");
        assert!(
            score4 < score2,
            "product score should decrease with more uncertain steps: {score4} < {score2}"
        );
    }

    #[test]
    fn aggregate_minimum_less_than_mean() {
        let cfg_min = PrmConfig {
            aggregation: PrmAggregation::Minimum,
            ..Default::default()
        };
        let cfg_mean = PrmConfig {
            aggregation: PrmAggregation::Mean,
            ..Default::default()
        };
        // logits vary: one bad step
        let output = PrmOutput::new(vec![5.0, 5.0, -5.0]);
        let min_score =
            prm_aggregate_score(&output, &cfg_min).expect("prm_aggregate_score should succeed");
        let mean_score =
            prm_aggregate_score(&output, &cfg_mean).expect("prm_aggregate_score should succeed");
        assert!(
            min_score < mean_score,
            "min score {min_score} should be less than mean score {mean_score}"
        );
    }

    #[test]
    fn aggregate_last_step_equals_sigmoid_last_logit() {
        let cfg = PrmConfig {
            aggregation: PrmAggregation::LastStep,
            ..Default::default()
        };
        let output = PrmOutput::new(vec![1.0, 2.0, 3.0]);
        let score = prm_aggregate_score(&output, &cfg).expect("prm_aggregate_score should succeed");
        let expected = sigmoid(3.0);
        assert!(
            (score - expected).abs() < 1e-6,
            "LastStep score = {score}, expected sigmoid(3.0) = {expected}"
        );
    }

    #[test]
    fn aggregate_weighted_respects_custom_weights() {
        let weights = vec![1.0, 2.0]; // second step weighted 2x
        let cfg = PrmConfig {
            aggregation: PrmAggregation::Weighted { weights },
            ..Default::default()
        };
        let output = PrmOutput::new(vec![0.0, 100.0]); // step2 nearly certain
        let score = prm_aggregate_score(&output, &cfg).expect("prm_aggregate_score should succeed");
        // Expected: (1*0.5 + 2*1.0) / 3 = 2.5 / 3 ≈ 0.833
        let expected = (1.0 * sigmoid(0.0) + 2.0 * sigmoid(100.0)) / 3.0;
        assert!(
            (score - expected.clamp(0.0, 1.0)).abs() < 1e-4,
            "weighted score = {score}, expected ≈ {expected}"
        );
    }

    // ── prm_rank_solutions ────────────────────────────────────────────────────

    #[test]
    fn rank_solutions_best_first() {
        let cfg = PrmConfig::default();
        // Output 0: all correct (logit=5), Output 1: all wrong (logit=-5)
        let out0 = PrmOutput::new(vec![5.0, 5.0]);
        let out1 = PrmOutput::new(vec![-5.0, -5.0]);
        let ranked =
            prm_rank_solutions(&[out0, out1], &cfg).expect("prm_rank_solutions should succeed");
        assert_eq!(ranked[0], 0, "best solution (index 0) should rank first");
        assert_eq!(ranked[1], 1);
    }

    #[test]
    fn rank_solutions_single_index_zero() {
        let cfg = PrmConfig::default();
        let out = PrmOutput::new(vec![1.0]);
        let ranked = prm_rank_solutions(&[out], &cfg).expect("prm_rank_solutions should succeed");
        assert_eq!(ranked, vec![0], "single solution should rank as [0]");
    }

    #[test]
    fn rank_solutions_three_candidates_correct_order() {
        let cfg = PrmConfig {
            aggregation: PrmAggregation::Mean,
            ..Default::default()
        };
        // Scores: out0 → mean prob ≈ 0.7, out1 → 0.5, out2 → 0.27
        let out0 = PrmOutput::new(vec![1.0, 1.0]); // mean sigmoid(1.0) ≈ 0.731
        let out1 = PrmOutput::new(vec![0.0, 0.0]); // mean 0.5
        let out2 = PrmOutput::new(vec![-1.0, -1.0]); // mean sigmoid(-1.0) ≈ 0.269
        let ranked = prm_rank_solutions(&[out0, out1, out2], &cfg)
            .expect("prm_rank_solutions should succeed");
        assert_eq!(ranked[0], 0, "highest-logit solution should rank first");
        assert_eq!(ranked[1], 1);
        assert_eq!(ranked[2], 2);
    }

    // ── prm_loss_batch ────────────────────────────────────────────────────────

    #[test]
    fn batch_mean_loss_matches_individual() {
        let cfg = PrmConfig::default();
        let out0 = PrmOutput::new(vec![0.0, 0.0]);
        let lbl0 = PrmLabel {
            step_labels: vec![1.0, 1.0],
            solution_label: None,
        };
        let out1 = PrmOutput::new(vec![1.0, -1.0]);
        let lbl1 = PrmLabel {
            step_labels: vec![1.0, 0.0],
            solution_label: None,
        };
        let l0 = prm_loss(&out0, &lbl0, &cfg)
            .expect("prm_loss should succeed")
            .total_loss;
        let l1 = prm_loss(&out1, &lbl1, &cfg)
            .expect("prm_loss should succeed")
            .total_loss;
        let expected = (l0 + l1) / 2.0;
        let batch = prm_loss_batch(&[out0, out1], &[lbl0, lbl1], &cfg)
            .expect("prm_loss_batch should succeed");
        assert!(
            (batch - expected).abs() < 1e-5,
            "batch mean = {batch}, expected {expected}"
        );
    }

    #[test]
    fn batch_empty_returns_error() {
        let cfg = PrmConfig::default();
        assert!(
            matches!(prm_loss_batch(&[], &[], &cfg), Err(RlhfError::EmptyInput)),
            "empty batch should return EmptyInput"
        );
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_fields() {
        let cfg = PrmConfig::default();
        assert_eq!(cfg.n_steps_max, 32, "default n_steps_max should be 32");
        assert!(
            matches!(cfg.aggregation, PrmAggregation::Product),
            "default aggregation should be Product"
        );
        assert!(
            (cfg.step_loss_weight - 1.0).abs() < 1e-6,
            "default step_loss_weight should be 1.0"
        );
        assert!(
            cfg.solution_loss_weight.abs() < 1e-6,
            "default solution_loss_weight should be 0.0"
        );
        assert!(
            cfg.label_smoothing.abs() < 1e-6,
            "default label_smoothing should be 0.0"
        );
    }

    // ── n_steps() helper ──────────────────────────────────────────────────────

    #[test]
    fn n_steps_matches_logit_count() {
        let out = PrmOutput::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.n_steps(), 4);
    }

    // ── NaN guard ─────────────────────────────────────────────────────────────

    #[test]
    fn prm_loss_nan_logit_returns_error() {
        let output = PrmOutput::new(vec![f32::NAN]);
        let label = PrmLabel {
            step_labels: vec![1.0],
            solution_label: None,
        };
        let cfg = PrmConfig::default();
        assert!(
            matches!(
                prm_loss(&output, &label, &cfg),
                Err(RlhfError::NanEncountered)
            ),
            "NaN logit should return NanEncountered"
        );
    }
}
