//! Step-DPO: Step-wise Preference Optimization for chain-of-thought reasoning.
//!
//! Reference: Lai et al. 2024, "Step-DPO: Step-wise Preference Optimization for
//! Long-chain Reasoning of LLMs", arXiv:2406.18629.
//!
//! Unlike standard DPO which treats an entire response as chosen or rejected,
//! Step-DPO assigns fine-grained credit at the level of individual reasoning
//! steps. Each step k contributes a scaled DPO loss term:
//!
//! ```text
//! L = Σ_k w_k * (-log σ(β * (log π_θ(s_k^w|ctx^w) - log π_ref(s_k^w|ctx^w)
//!                         - log π_θ(s_k^l|ctx^l) + log π_ref(s_k^l|ctx^l))))
//! ```
//!
//! with aggregation (weighted mean / sum / last-step) controlled by
//! [`StepReduceMode`].

use crate::error::{RlhfError, RlhfResult};

// ── Step weighting ────────────────────────────────────────────────────────────

/// How to weight individual reasoning steps.
#[derive(Debug, Clone)]
pub enum StepWeightScheme {
    /// All steps weighted equally: w_k = 1.0 for all k.
    Uniform,
    /// Exponential decay: w_k = gamma^k (k = 0, 1, …).
    ///
    /// gamma ∈ (0, 1] discounts earlier steps; gamma = 1.0 is identical to
    /// `Uniform`.
    ExponentialDecay { gamma: f32 },
    /// Inverse-position weighting: w_k = 1.0 / (k + 1).
    InversePosition,
    /// Caller-supplied per-step weights (must have exactly n_steps entries).
    Explicit { weights: Vec<f32> },
}

// ── Reduction mode ────────────────────────────────────────────────────────────

/// How to aggregate the per-step losses into a scalar.
#[derive(Debug, Clone)]
pub enum StepReduceMode {
    /// Weighted mean over steps: L = (Σ_k w_k * loss_k) / (Σ_k w_k).
    WeightedMean,
    /// Weighted sum: L = Σ_k w_k * loss_k.
    WeightedSum,
    /// Use only the last step's loss (ignores all weights).
    LastStep,
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the Step-DPO loss.
#[derive(Debug, Clone)]
pub struct StepDpoConfig {
    /// KL-regularisation temperature β (default 0.1).
    ///
    /// Must be positive and finite; identical in meaning to `DpoConfig::beta`.
    pub beta: f32,
    /// How to weight individual reasoning steps (default [`StepWeightScheme::Uniform`]).
    pub weight_scheme: StepWeightScheme,
    /// How to reduce the weighted per-step losses (default [`StepReduceMode::WeightedMean`]).
    pub reduce: StepReduceMode,
}

impl Default for StepDpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            weight_scheme: StepWeightScheme::Uniform,
            reduce: StepReduceMode::WeightedMean,
        }
    }
}

// ── Input ─────────────────────────────────────────────────────────────────────

/// Step-level preference pair for a single reasoning problem.
///
/// Every field is a `Vec<f32>` of length `n_steps`.  All four vecs must be the
/// same length and non-empty.
#[derive(Debug, Clone)]
pub struct StepPair {
    /// log π_θ(s_k^w | ctx_k^w) — policy log-prob of each chosen step.
    pub chosen_step_logps: Vec<f32>,
    /// log π_θ(s_k^l | ctx_k^l) — policy log-prob of each rejected step.
    pub rejected_step_logps: Vec<f32>,
    /// log π_ref(s_k^w | ctx_k^w) — reference log-prob of each chosen step.
    pub ref_chosen_step_logps: Vec<f32>,
    /// log π_ref(s_k^l | ctx_k^l) — reference log-prob of each rejected step.
    pub ref_rejected_step_logps: Vec<f32>,
}

impl StepPair {
    /// Number of reasoning steps in this pair (derived from chosen length).
    pub fn n_steps(&self) -> usize {
        self.chosen_step_logps.len()
    }

    /// Validate that all four vectors have the same positive length.
    pub fn validate(&self) -> RlhfResult<()> {
        let n = self.chosen_step_logps.len();
        if n == 0 {
            return Err(RlhfError::EmptyInput);
        }
        if self.rejected_step_logps.len() != n {
            return Err(RlhfError::MismatchedPairLength {
                chosen: n,
                rejected: self.rejected_step_logps.len(),
            });
        }
        if self.ref_chosen_step_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: self.ref_chosen_step_logps.len(),
            });
        }
        if self.ref_rejected_step_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: self.ref_rejected_step_logps.len(),
            });
        }
        Ok(())
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Per-pair output of the Step-DPO computation.
#[derive(Debug, Clone)]
pub struct StepDpoOutput {
    /// Aggregated scalar loss for this pair.
    pub loss: f32,
    /// Per-step DPO loss values (before weighting).
    pub per_step_losses: Vec<f32>,
    /// Per-step weights that were applied.
    pub per_step_weights: Vec<f32>,
    /// Average implicit reward margin across steps:
    /// mean_k β * ((log π_θ/π_ref chosen) − (log π_θ/π_ref rejected)).
    pub mean_margin: f32,
}

// ── Numerics ──────────────────────────────────────────────────────────────────

/// Numerically stable log-sigmoid: log σ(x) = log(1 / (1 + exp(−x))).
///
/// Uses the standard two-branch form to avoid cancellation and overflow:
/// * x ≥ 0: −log(1 + exp(−x))
/// * x < 0: x − log(1 + exp(x))
#[inline]
pub fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0_f32 + (-x).exp()).ln()
    } else {
        x - (1.0_f32 + x.exp()).ln()
    }
}

/// Compute the implicit reward for a single step:
///
/// r_k = β * ((log π_θ(s_k^w) − log π_ref(s_k^w)) − (log π_θ(s_k^l) − log π_ref(s_k^l)))
#[inline]
pub fn step_implicit_reward(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    beta: f32,
) -> f32 {
    beta * ((chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp))
}

// ── Loss computation ──────────────────────────────────────────────────────────

/// Compute the Step-DPO loss for a single preference pair.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if the pair has no steps,
/// [`RlhfError::InvalidBeta`] if β ≤ 0 or non-finite,
/// [`RlhfError::DimensionMismatch`] if the `Explicit` weight vector length
/// does not match `n_steps`, and [`RlhfError::NanEncountered`] if any
/// intermediate or final value is NaN.
pub fn step_dpo_loss(pair: &StepPair, cfg: &StepDpoConfig) -> RlhfResult<StepDpoOutput> {
    pair.validate()?;

    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }

    let n = pair.n_steps();

    // ── Step 1: per-step log-ratios and losses ────────────────────────────────
    let mut per_step_losses = Vec::with_capacity(n);
    let mut log_ratios = Vec::with_capacity(n);

    for k in 0..n {
        let log_ratio = step_implicit_reward(
            pair.chosen_step_logps[k],
            pair.ref_chosen_step_logps[k],
            pair.rejected_step_logps[k],
            pair.ref_rejected_step_logps[k],
            cfg.beta,
        );
        log_ratios.push(log_ratio);
        per_step_losses.push(-log_sigmoid(log_ratio));
    }

    // ── Step 2: per-step weights ──────────────────────────────────────────────
    let per_step_weights: Vec<f32> = match &cfg.weight_scheme {
        StepWeightScheme::Uniform => vec![1.0_f32; n],

        StepWeightScheme::ExponentialDecay { gamma } => {
            let g = *gamma;
            let mut w = 1.0_f32;
            let mut weights = Vec::with_capacity(n);
            for _ in 0..n {
                weights.push(w);
                w *= g;
            }
            weights
        }

        StepWeightScheme::InversePosition => (0..n).map(|k| 1.0_f32 / (k as f32 + 1.0)).collect(),

        StepWeightScheme::Explicit { weights } => {
            if weights.len() != n {
                return Err(RlhfError::DimensionMismatch {
                    expected: n,
                    got: weights.len(),
                });
            }
            weights.clone()
        }
    };

    // ── Step 3: aggregate loss ────────────────────────────────────────────────
    let loss = match cfg.reduce {
        StepReduceMode::WeightedMean => {
            let weighted_sum: f32 = per_step_losses
                .iter()
                .zip(per_step_weights.iter())
                .map(|(&l, &w)| w * l)
                .sum();
            let weight_total: f32 = per_step_weights.iter().sum();
            weighted_sum / weight_total
        }

        StepReduceMode::WeightedSum => per_step_losses
            .iter()
            .zip(per_step_weights.iter())
            .map(|(&l, &w)| w * l)
            .sum(),

        StepReduceMode::LastStep => per_step_losses[n - 1],
    };

    // ── Step 4: mean margin (unweighted) ─────────────────────────────────────
    let mean_margin: f32 = log_ratios.iter().sum::<f32>() / n as f32;

    // ── Validity check ────────────────────────────────────────────────────────
    if loss.is_nan() || mean_margin.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    for &l in &per_step_losses {
        if l.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
    }

    Ok(StepDpoOutput {
        loss,
        per_step_losses,
        per_step_weights,
        mean_margin,
    })
}

/// Compute the mean Step-DPO loss over a batch of preference pairs.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if `pairs` is empty, or propagates any
/// error from [`step_dpo_loss`].
pub fn step_dpo_loss_batch(pairs: &[StepPair], cfg: &StepDpoConfig) -> RlhfResult<f32> {
    if pairs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let total: f32 = pairs
        .iter()
        .map(|pair| step_dpo_loss(pair, cfg).map(|o| o.loss))
        .collect::<RlhfResult<Vec<f32>>>()?
        .into_iter()
        .sum();
    let mean = total / pairs.len() as f32;
    if mean.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(mean)
}

// ── Gradient ────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid `σ(x)`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Gradient of the aggregated Step-DPO loss w.r.t. every per-step log-probability.
///
/// Finite-difference verified against [`step_dpo_loss`]`(..).loss`. Each field is
/// a `Vec<f32>` of length `n_steps`, aligned with the corresponding input vector.
#[derive(Debug, Clone)]
pub struct StepDpoGrad {
    /// `∂L/∂(chosen step log-prob)` for each step.
    pub d_chosen_step_logps: Vec<f32>,
    /// `∂L/∂(rejected step log-prob)` for each step.
    pub d_rejected_step_logps: Vec<f32>,
    /// `∂L/∂(reference chosen step log-prob)` for each step.
    pub d_ref_chosen_step_logps: Vec<f32>,
    /// `∂L/∂(reference rejected step log-prob)` for each step.
    pub d_ref_rejected_step_logps: Vec<f32>,
}

/// Analytic gradient of [`step_dpo_loss`].
///
/// Per step `loss_k = −log σ(r_k)` with `r_k = β·((c_k − rc_k) − (l_k − rr_k))`,
/// so `dloss_k/dr_k = −σ(−r_k)` and the four chained partials follow the
/// `+β, −β, −β, +β` pattern. The aggregation contributes a per-step coefficient
/// `a_k`: for [`StepReduceMode::WeightedMean`] `a_k = w_k / Σw`, for
/// [`StepReduceMode::WeightedSum`] `a_k = w_k`, and for
/// [`StepReduceMode::LastStep`] `a_k = 1` on the last step and `0` elsewhere
/// (the per-step weights themselves are constants of the position / scheme and
/// carry no gradient). The forward's own computed `per_step_weights` are reused
/// so the coefficients match exactly.
///
/// # Errors
///
/// Mirrors [`step_dpo_loss`]: validation of the pair / `beta` / explicit-weight
/// length, plus [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn step_dpo_grad(pair: &StepPair, cfg: &StepDpoConfig) -> RlhfResult<StepDpoGrad> {
    // Reuse the forward for validation and the exact per-step weights.
    let out = step_dpo_loss(pair, cfg)?;
    let n = pair.n_steps();
    let weights = &out.per_step_weights;

    let coeffs: Vec<f32> = match cfg.reduce {
        StepReduceMode::WeightedMean => {
            let wsum: f32 = weights.iter().sum();
            weights.iter().map(|&w| w / wsum).collect()
        }
        StepReduceMode::WeightedSum => weights.clone(),
        StepReduceMode::LastStep => (0..n).map(|k| if k == n - 1 { 1.0 } else { 0.0 }).collect(),
    };

    let beta = cfg.beta;
    let mut d_chosen_step_logps = Vec::with_capacity(n);
    let mut d_rejected_step_logps = Vec::with_capacity(n);
    let mut d_ref_chosen_step_logps = Vec::with_capacity(n);
    let mut d_ref_rejected_step_logps = Vec::with_capacity(n);
    for ((((&coeff, &c), &rc), &l), &rr) in coeffs
        .iter()
        .zip(pair.chosen_step_logps.iter())
        .zip(pair.ref_chosen_step_logps.iter())
        .zip(pair.rejected_step_logps.iter())
        .zip(pair.ref_rejected_step_logps.iter())
    {
        let r_k = step_implicit_reward(c, rc, l, rr, beta);
        // d = a_k · (dloss_k/dr_k) · (∂r_k/∂c_k) = a_k · (−σ(−r_k)) · β.
        let d = coeff * (-sigmoid(-r_k)) * beta;
        if !d.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_chosen_step_logps.push(d);
        d_ref_chosen_step_logps.push(-d);
        d_rejected_step_logps.push(-d);
        d_ref_rejected_step_logps.push(d);
    }

    Ok(StepDpoGrad {
        d_chosen_step_logps,
        d_rejected_step_logps,
        d_ref_chosen_step_logps,
        d_ref_rejected_step_logps,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple uniform pair.
    fn make_pair(n: usize, chosen_lp: f32, rejected_lp: f32) -> StepPair {
        StepPair {
            chosen_step_logps: vec![chosen_lp; n],
            rejected_step_logps: vec![rejected_lp; n],
            ref_chosen_step_logps: vec![chosen_lp; n],
            ref_rejected_step_logps: vec![rejected_lp; n],
        }
    }

    // ── log_sigmoid tests ─────────────────────────────────────────────────────

    #[test]
    fn log_sigmoid_at_zero() {
        // log σ(0) = -ln 2 ≈ -0.6931
        let expected = -(2.0_f32.ln());
        assert!(
            (log_sigmoid(0.0) - expected).abs() < 1e-5,
            "log_sigmoid(0) = {}, expected {}",
            log_sigmoid(0.0),
            expected
        );
    }

    #[test]
    fn log_sigmoid_large_positive() {
        // log σ(100) ≈ 0
        assert!(
            log_sigmoid(100.0).abs() < 1e-5,
            "log_sigmoid(100) should be ~0, got {}",
            log_sigmoid(100.0)
        );
    }

    #[test]
    fn log_sigmoid_large_negative() {
        // log σ(-100) ≈ -100
        let v = log_sigmoid(-100.0);
        assert!(
            (v - (-100.0_f32)).abs() < 1e-2,
            "log_sigmoid(-100) = {v}, expected ~-100"
        );
    }

    #[test]
    fn log_sigmoid_is_finite_for_extreme_inputs() {
        assert!(log_sigmoid(f32::MAX / 2.0).is_finite());
        assert!(log_sigmoid(-f32::MAX / 2.0).is_finite());
    }

    // ── step_implicit_reward tests ────────────────────────────────────────────

    #[test]
    fn implicit_reward_equal_logps_is_zero() {
        let r = step_implicit_reward(-1.0, -1.0, -2.0, -2.0, 0.5);
        assert!(
            r.abs() < 1e-6,
            "equal log-probs should give zero margin, got {r}"
        );
    }

    #[test]
    fn implicit_reward_chosen_better() {
        // chosen logp higher → positive margin
        let r = step_implicit_reward(-0.5, -1.0, -1.5, -1.0, 1.0);
        assert!(r > 0.0, "chosen better → positive margin, got {r}");
    }

    // ── StepPair::validate tests ──────────────────────────────────────────────

    #[test]
    fn validate_fails_empty() {
        let pair = StepPair {
            chosen_step_logps: vec![],
            rejected_step_logps: vec![],
            ref_chosen_step_logps: vec![],
            ref_rejected_step_logps: vec![],
        };
        assert!(
            matches!(pair.validate(), Err(RlhfError::EmptyInput)),
            "empty vecs should return EmptyInput"
        );
    }

    #[test]
    fn validate_fails_length_mismatch_rejected() {
        let pair = StepPair {
            chosen_step_logps: vec![-1.0, -2.0],
            rejected_step_logps: vec![-1.0],
            ref_chosen_step_logps: vec![-1.0, -2.0],
            ref_rejected_step_logps: vec![-1.0, -2.0],
        };
        assert!(
            pair.validate().is_err(),
            "mismatched rejected length should error"
        );
    }

    #[test]
    fn validate_fails_length_mismatch_ref_chosen() {
        let pair = StepPair {
            chosen_step_logps: vec![-1.0, -2.0],
            rejected_step_logps: vec![-1.0, -2.0],
            ref_chosen_step_logps: vec![-1.0],
            ref_rejected_step_logps: vec![-1.0, -2.0],
        };
        assert!(
            pair.validate().is_err(),
            "mismatched ref_chosen length should error"
        );
    }

    // ── step_dpo_loss: single-step ────────────────────────────────────────────

    #[test]
    fn single_step_equals_standard_dpo() {
        // With n=1 chosen_lp=-0.5, rejected_lp=-2.0 (ref both -1.0)
        let pair = StepPair {
            chosen_step_logps: vec![-0.5],
            rejected_step_logps: vec![-2.0],
            ref_chosen_step_logps: vec![-1.0],
            ref_rejected_step_logps: vec![-1.0],
        };
        let cfg = StepDpoConfig::default();
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        assert!(out.loss.is_finite(), "single-step loss should be finite");
        // Verify it equals -log_sigmoid(beta * (chosen_delta - rejected_delta))
        let delta = 0.1 * ((-0.5 - (-1.0)) - (-2.0 - (-1.0)));
        let expected = -log_sigmoid(delta);
        assert!(
            (out.loss - expected).abs() < 1e-5,
            "single-step loss {}, expected {}",
            out.loss,
            expected
        );
    }

    // ── step_dpo_loss: Uniform weights + WeightedMean ─────────────────────────

    #[test]
    fn uniform_weighted_mean_equals_mean_losses() {
        let pair = StepPair {
            chosen_step_logps: vec![-0.5, -1.0, -1.5],
            rejected_step_logps: vec![-2.0, -2.5, -3.0],
            ref_chosen_step_logps: vec![-1.0, -1.0, -1.0],
            ref_rejected_step_logps: vec![-1.0, -1.0, -1.0],
        };
        let cfg = StepDpoConfig {
            reduce: StepReduceMode::WeightedMean,
            ..Default::default()
        };
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        let expected_mean = out.per_step_losses.iter().sum::<f32>() / 3.0;
        assert!(
            (out.loss - expected_mean).abs() < 1e-5,
            "WeightedMean with Uniform should equal mean(per_step_losses); got {} vs {}",
            out.loss,
            expected_mean
        );
    }

    // ── step_dpo_loss: ExponentialDecay weights ───────────────────────────────

    #[test]
    fn exponential_decay_weights_match_formula() {
        let n = 4_usize;
        let pair = make_pair(n, -1.0, -2.0);
        let gamma = 0.9_f32;
        let cfg = StepDpoConfig {
            weight_scheme: StepWeightScheme::ExponentialDecay { gamma },
            reduce: StepReduceMode::WeightedSum,
            ..Default::default()
        };
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        // All steps identical, so check weights
        let expected_weights: Vec<f32> = (0..n).map(|k| gamma.powi(k as i32)).collect();
        for (k, (&w, &ew)) in out
            .per_step_weights
            .iter()
            .zip(expected_weights.iter())
            .enumerate()
        {
            assert!((w - ew).abs() < 1e-5, "weight[{k}] = {w}, expected {ew}");
        }
    }

    // ── step_dpo_loss: InversePosition weights ────────────────────────────────

    #[test]
    fn inverse_position_weights_correct() {
        let n = 3_usize;
        let pair = make_pair(n, -1.0, -2.0);
        let cfg = StepDpoConfig {
            weight_scheme: StepWeightScheme::InversePosition,
            ..Default::default()
        };
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        assert!((out.per_step_weights[0] - 1.0).abs() < 1e-5, "w[0]=1.0");
        assert!((out.per_step_weights[1] - 0.5).abs() < 1e-5, "w[1]=0.5");
        assert!(
            (out.per_step_weights[2] - 1.0 / 3.0).abs() < 1e-5,
            "w[2]=1/3"
        );
    }

    // ── step_dpo_loss: LastStep ───────────────────────────────────────────────

    #[test]
    fn last_step_reduce_equals_last_per_step_loss() {
        let pair = StepPair {
            chosen_step_logps: vec![-0.5, -1.0, -1.5],
            rejected_step_logps: vec![-2.0, -2.5, -3.0],
            ref_chosen_step_logps: vec![-1.0, -1.0, -1.0],
            ref_rejected_step_logps: vec![-1.0, -1.0, -1.0],
        };
        let cfg = StepDpoConfig {
            reduce: StepReduceMode::LastStep,
            ..Default::default()
        };
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        let n = out.per_step_losses.len();
        assert!(
            (out.loss - out.per_step_losses[n - 1]).abs() < 1e-6,
            "LastStep loss {} != per_step_losses[last] {}",
            out.loss,
            out.per_step_losses[n - 1]
        );
    }

    // ── step_dpo_loss: WeightedSum ────────────────────────────────────────────

    #[test]
    fn weighted_sum_reduce_correct() {
        let pair = StepPair {
            chosen_step_logps: vec![-0.5, -1.0],
            rejected_step_logps: vec![-2.0, -2.5],
            ref_chosen_step_logps: vec![-1.0, -1.0],
            ref_rejected_step_logps: vec![-1.0, -1.0],
        };
        let cfg = StepDpoConfig {
            weight_scheme: StepWeightScheme::Explicit {
                weights: vec![2.0, 3.0],
            },
            reduce: StepReduceMode::WeightedSum,
            ..Default::default()
        };
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        let expected = 2.0 * out.per_step_losses[0] + 3.0 * out.per_step_losses[1];
        assert!(
            (out.loss - expected).abs() < 1e-5,
            "WeightedSum = {}, expected {expected}",
            out.loss
        );
    }

    // ── step_dpo_loss: NaN handling ───────────────────────────────────────────

    #[test]
    fn nan_logp_returns_nan_encountered() {
        let pair = StepPair {
            chosen_step_logps: vec![f32::NAN],
            rejected_step_logps: vec![-1.0],
            ref_chosen_step_logps: vec![-1.0],
            ref_rejected_step_logps: vec![-1.0],
        };
        let cfg = StepDpoConfig::default();
        assert!(
            matches!(step_dpo_loss(&pair, &cfg), Err(RlhfError::NanEncountered)),
            "NaN log-prob should return NanEncountered"
        );
    }

    // ── step_dpo_loss_batch ────────────────────────────────────────────────────

    #[test]
    fn batch_empty_returns_error() {
        let cfg = StepDpoConfig::default();
        assert!(
            matches!(step_dpo_loss_batch(&[], &cfg), Err(RlhfError::EmptyInput)),
            "empty batch should return EmptyInput"
        );
    }

    #[test]
    fn batch_mean_loss_correct() {
        let cfg = StepDpoConfig::default();
        let pair_a = make_pair(2, -0.5, -2.0);
        let pair_b = make_pair(2, -1.0, -3.0);
        let loss_a = step_dpo_loss(&pair_a, &cfg)
            .expect("step_dpo_loss should succeed")
            .loss;
        let loss_b = step_dpo_loss(&pair_b, &cfg)
            .expect("step_dpo_loss should succeed")
            .loss;
        let batch_loss = step_dpo_loss_batch(&[pair_a, pair_b], &cfg)
            .expect("step_dpo_loss_batch should succeed");
        let expected = (loss_a + loss_b) / 2.0;
        assert!(
            (batch_loss - expected).abs() < 1e-5,
            "batch mean = {batch_loss}, expected {expected}"
        );
    }

    // ── margin direction tests ─────────────────────────────────────────────────

    #[test]
    fn chosen_better_than_rejected_positive_margin_low_loss() {
        // chosen_lp > ref_chosen_lp means policy up-weights chosen step
        let pair = StepPair {
            chosen_step_logps: vec![-0.2],
            rejected_step_logps: vec![-3.0],
            ref_chosen_step_logps: vec![-1.0],
            ref_rejected_step_logps: vec![-1.0],
        };
        let cfg = StepDpoConfig::default();
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        assert!(out.mean_margin > 0.0, "chosen better → positive margin");
        // Loss should be less than ln 2 (the maximum at margin=0)
        assert!(
            out.loss < 2.0_f32.ln(),
            "aligned pair should have loss < ln2"
        );
    }

    #[test]
    fn rejected_better_than_chosen_negative_margin_high_loss() {
        let pair = StepPair {
            chosen_step_logps: vec![-3.0],
            rejected_step_logps: vec![-0.2],
            ref_chosen_step_logps: vec![-1.0],
            ref_rejected_step_logps: vec![-1.0],
        };
        let cfg = StepDpoConfig::default();
        let out = step_dpo_loss(&pair, &cfg).expect("step_dpo_loss should succeed");
        assert!(out.mean_margin < 0.0, "rejected better → negative margin");
        assert!(
            out.loss > 2.0_f32.ln(),
            "unaligned pair should have loss > ln2"
        );
    }

    // ── Explicit weight length mismatch ───────────────────────────────────────

    #[test]
    fn explicit_weights_wrong_length_returns_error() {
        let pair = make_pair(3, -1.0, -2.0);
        let cfg = StepDpoConfig {
            weight_scheme: StepWeightScheme::Explicit {
                weights: vec![1.0, 2.0],
            }, // length 2 ≠ 3
            ..Default::default()
        };
        assert!(
            matches!(
                step_dpo_loss(&pair, &cfg),
                Err(RlhfError::DimensionMismatch { .. })
            ),
            "wrong explicit weight length should return DimensionMismatch"
        );
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_fields() {
        let cfg = StepDpoConfig::default();
        assert!((cfg.beta - 0.1).abs() < 1e-6, "default beta should be 0.1");
        assert!(
            matches!(cfg.weight_scheme, StepWeightScheme::Uniform),
            "default weight_scheme should be Uniform"
        );
        assert!(
            matches!(cfg.reduce, StepReduceMode::WeightedMean),
            "default reduce should be WeightedMean"
        );
    }

    // ── Invalid beta ──────────────────────────────────────────────────────────

    #[test]
    fn invalid_beta_returns_error() {
        let pair = make_pair(2, -1.0, -2.0);
        let cfg = StepDpoConfig {
            beta: -0.1,
            ..Default::default()
        };
        assert!(
            matches!(
                step_dpo_loss(&pair, &cfg),
                Err(RlhfError::InvalidBeta { .. })
            ),
            "negative beta should return InvalidBeta"
        );
    }

    // ── n_steps() helper ──────────────────────────────────────────────────────

    #[test]
    fn n_steps_returns_correct_count() {
        let pair = make_pair(5, -1.0, -2.0);
        assert_eq!(pair.n_steps(), 5);
    }
}

#[cfg(test)]
mod grad_tests {
    use super::*;

    fn central_diff(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        ((f(x + h) as f64 - f(x - h) as f64) / (2.0 * h as f64)) as f32
    }

    fn assert_close(analytic: f32, fd: f32, label: &str) {
        let denom = analytic.abs().max(1e-3);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel <= 1e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    fn mk() -> StepPair {
        StepPair {
            chosen_step_logps: vec![-0.5, -1.0, -1.4],
            rejected_step_logps: vec![-2.0, -2.5, -3.1],
            ref_chosen_step_logps: vec![-1.0, -1.1, -1.2],
            ref_rejected_step_logps: vec![-1.0, -1.1, -1.2],
        }
    }

    fn loss_of(pair: &StepPair, cfg: &StepDpoConfig) -> f32 {
        step_dpo_loss(pair, cfg).expect("loss").loss
    }

    fn check_all(cfg: &StepDpoConfig) {
        let pair = mk();
        let g = step_dpo_grad(&pair, cfg).expect("grad");
        let n = pair.n_steps();
        let h = 1e-2;
        for k in 0..n {
            let fd_c = central_diff(
                |v| {
                    let mut p = mk();
                    p.chosen_step_logps[k] = v;
                    loss_of(&p, cfg)
                },
                pair.chosen_step_logps[k],
                h,
            );
            let fd_l = central_diff(
                |v| {
                    let mut p = mk();
                    p.rejected_step_logps[k] = v;
                    loss_of(&p, cfg)
                },
                pair.rejected_step_logps[k],
                h,
            );
            let fd_rc = central_diff(
                |v| {
                    let mut p = mk();
                    p.ref_chosen_step_logps[k] = v;
                    loss_of(&p, cfg)
                },
                pair.ref_chosen_step_logps[k],
                h,
            );
            let fd_rr = central_diff(
                |v| {
                    let mut p = mk();
                    p.ref_rejected_step_logps[k] = v;
                    loss_of(&p, cfg)
                },
                pair.ref_rejected_step_logps[k],
                h,
            );
            assert_close(g.d_chosen_step_logps[k], fd_c, "d_chosen");
            assert_close(g.d_rejected_step_logps[k], fd_l, "d_rejected");
            assert_close(g.d_ref_chosen_step_logps[k], fd_rc, "d_ref_chosen");
            assert_close(g.d_ref_rejected_step_logps[k], fd_rr, "d_ref_rejected");
        }
    }

    #[test]
    fn step_dpo_grad_matches_fd_weighted_mean() {
        check_all(&StepDpoConfig::default());
    }

    #[test]
    fn step_dpo_grad_matches_fd_weighted_sum_explicit() {
        check_all(&StepDpoConfig {
            beta: 0.2,
            weight_scheme: StepWeightScheme::Explicit {
                weights: vec![2.0, 0.5, 1.5],
            },
            reduce: StepReduceMode::WeightedSum,
        });
    }

    #[test]
    fn step_dpo_grad_matches_fd_exponential_decay() {
        check_all(&StepDpoConfig {
            beta: 0.15,
            weight_scheme: StepWeightScheme::ExponentialDecay { gamma: 0.8 },
            reduce: StepReduceMode::WeightedMean,
        });
    }

    #[test]
    fn step_dpo_grad_last_step_zeroes_other_steps() {
        let cfg = StepDpoConfig {
            reduce: StepReduceMode::LastStep,
            ..Default::default()
        };
        let g = step_dpo_grad(&mk(), &cfg).expect("grad");
        // Only the last step carries gradient.
        assert_eq!(g.d_chosen_step_logps[0], 0.0);
        assert_eq!(g.d_chosen_step_logps[1], 0.0);
        assert!(g.d_chosen_step_logps[2] < 0.0, "last step pushes chosen up");
        check_all(&cfg);
    }

    #[test]
    fn step_dpo_grad_chosen_sign_is_descent() {
        let g = step_dpo_grad(&mk(), &StepDpoConfig::default()).expect("grad");
        for (&dc, &dl) in g
            .d_chosen_step_logps
            .iter()
            .zip(g.d_rejected_step_logps.iter())
        {
            assert!(dc < 0.0, "raising a chosen step lowers loss");
            assert!(dl > 0.0, "raising a rejected step raises loss");
        }
    }

    #[test]
    fn step_dpo_grad_invalid_beta_errors() {
        let cfg = StepDpoConfig {
            beta: -0.1,
            ..Default::default()
        };
        assert!(matches!(
            step_dpo_grad(&mk(), &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}
