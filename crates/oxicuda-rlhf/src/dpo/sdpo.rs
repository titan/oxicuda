//! sDPO: stepwise (staged) Direct Preference Optimisation.
//!
//! Reference: Kim et al. 2024, "sDPO: Don't Use Your Data All at Once",
//! arXiv:2403.19270.
//!
//! Standard DPO uses a single, fixed reference model — the original SFT
//! checkpoint — for the entire run. sDPO instead splits the preference data
//! into `n_stages` chunks and trains on them **one stage at a time**. The
//! defining property is the *reference handoff*: at stage `k` the reference
//! log-probs are the **policy** log-probs produced at the end of stage
//! `k − 1`, i.e. the previously aligned model becomes the reference for the
//! next stage rather than keeping the frozen SFT model throughout.
//!
//! ```text
//! stage 0:   ref = π_SFT
//! stage 1:   ref = π_(after stage 0)
//! stage 2:   ref = π_(after stage 1)
//! ...
//! ```
//!
//! Because each stage starts from a strictly more aligned reference, the
//! implicit-reward lower bound is tightened stage over stage, which Kim et al.
//! show empirically yields a better-aligned final model than feeding all data
//! to a single DPO run.
//!
//! Each stage's loss is exactly the standard DPO loss (mean over the batch of
//! `-log σ(β · ((π_chosen − ref_chosen) − (π_rejected − ref_rejected)))`),
//! evaluated against that stage's reference. This module reuses
//! [`dpo_log_ratio`] and [`log_sigmoid`] so a single-stage sDPO run is bit-for-bit
//! identical to [`crate::dpo::dpo::dpo_loss`].

use crate::dpo::dpo::dpo_log_ratio;
use crate::dpo::step_dpo::log_sigmoid;
use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PairBatch;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the staged DPO loss.
#[derive(Debug, Clone)]
pub struct SdpoConfig {
    /// KL-regularisation temperature β. Must be positive and finite; identical
    /// in meaning to [`crate::dpo::dpo::DpoConfig::beta`].
    pub beta: f32,
    /// Number of training stages the preference data is split into.
    ///
    /// This is informational metadata for the staged schedule; the loss
    /// functions below operate on whatever stages they are handed and validate
    /// against it where it is meaningful (see [`sdpo_total_loss`]).
    pub n_stages: usize,
}

impl SdpoConfig {
    /// Validate β (> 0, finite) and `n_stages` (≥ 1).
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if self.n_stages == 0 {
            return Err(RlhfError::EmptyInput);
        }
        Ok(())
    }
}

// ── Reference handoff ─────────────────────────────────────────────────────────

/// Produce the next stage's reference log-probs from the previous stage's
/// policy log-probs — the sDPO *reference handoff*.
///
/// At the end of stage `k − 1` the aligned policy's chosen/rejected log-probs
/// become the reference for stage `k`. This is a pure copy: the returned
/// `(ref_chosen, ref_rejected)` vectors equal the supplied
/// `(prev_policy_logps_chosen, prev_policy_logps_rejected)`. Keeping it as an
/// explicit function makes the handoff a named, testable step rather than an
/// implicit assignment.
#[must_use]
pub fn sdpo_update_reference(
    prev_policy_logps_chosen: &[f32],
    prev_policy_logps_rejected: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    (
        prev_policy_logps_chosen.to_vec(),
        prev_policy_logps_rejected.to_vec(),
    )
}

// ── Per-stage / total loss ─────────────────────────────────────────────────────

/// Compute the DPO loss for a single sDPO stage.
///
/// This is the standard DPO loss — the mean over the batch of
/// `-log σ(dpo_log_ratio(...))` — using the reference log-probs carried in
/// `stage_batch`. Within the staged schedule those reference log-probs are the
/// previous stage's policy log-probs (produced via [`sdpo_update_reference`]);
/// for stage 0 they are the SFT reference. The function itself is agnostic to
/// where the reference came from.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if `stage_batch` has no pairs,
/// [`RlhfError::InvalidBeta`] if β ≤ 0 or non-finite, and
/// [`RlhfError::NanEncountered`] if the result is NaN. (A [`PairBatch`] is
/// length-validated at construction, so the four vectors are always equal
/// length here.)
pub fn sdpo_stage_loss(stage_batch: &PairBatch, cfg: &SdpoConfig) -> RlhfResult<f32> {
    if stage_batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let total: f32 = stage_batch
        .chosen_logps
        .iter()
        .zip(stage_batch.rejected_logps.iter())
        .zip(stage_batch.ref_chosen_logps.iter())
        .zip(stage_batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta);
            -log_sigmoid(logit)
        })
        .sum();
    let loss = total / stage_batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean implicit-reward margin for a stage:
/// `mean_i β · ((π_chosen_i − ref_chosen_i) − (π_rejected_i − ref_rejected_i))`.
///
/// A positive margin means the policy assigns relatively more probability mass
/// to the chosen than the rejected responses (relative to the reference),
/// which is the alignment objective.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for an empty batch and
/// [`RlhfError::InvalidBeta`] for invalid β.
pub fn sdpo_stage_margin(stage_batch: &PairBatch, cfg: &SdpoConfig) -> RlhfResult<f32> {
    if stage_batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let total: f32 = stage_batch
        .chosen_logps
        .iter()
        .zip(stage_batch.rejected_logps.iter())
        .zip(stage_batch.ref_chosen_logps.iter())
        .zip(stage_batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta))
        .sum();
    Ok(total / stage_batch.len() as f32)
}

/// Compute the total sDPO loss as the **mean** of the per-stage DPO losses.
///
/// The mean (rather than the sum) is used so that a single-stage run reduces
/// exactly to [`crate::dpo::dpo::dpo_loss`]. Each stage carries its own
/// reference log-probs, which in a real staged run are the previous stage's
/// policy log-probs (the handoff).
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] if `stages` is empty,
/// [`RlhfError::DimensionMismatch`] if `cfg.n_stages` is set and does not match
/// `stages.len()`, [`RlhfError::InvalidBeta`] for invalid β, and propagates any
/// error from [`sdpo_stage_loss`].
pub fn sdpo_total_loss(stages: &[PairBatch], cfg: &SdpoConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if stages.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if stages.len() != cfg.n_stages {
        return Err(RlhfError::DimensionMismatch {
            expected: cfg.n_stages,
            got: stages.len(),
        });
    }
    let total: f32 = stages
        .iter()
        .map(|stage| sdpo_stage_loss(stage, cfg))
        .collect::<RlhfResult<Vec<f32>>>()?
        .into_iter()
        .sum();
    let mean = total / stages.len() as f32;
    if mean.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(mean)
}

// ── Stage gradient ──────────────────────────────────────────────────────────

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

/// Gradient of a single sDPO stage loss w.r.t. the four per-pair log-probs.
///
/// Finite-difference verified against [`sdpo_stage_loss`].
#[derive(Debug, Clone, Copy)]
pub struct SdpoStageGrad {
    /// `∂L/∂(policy chosen log-prob)`.
    pub d_chosen_logp: f32,
    /// `∂L/∂(policy rejected log-prob)`.
    pub d_rejected_logp: f32,
    /// `∂L/∂(reference chosen log-prob)`.
    pub d_ref_chosen_logp: f32,
    /// `∂L/∂(reference rejected log-prob)`.
    pub d_ref_rejected_logp: f32,
}

/// Analytic gradient of [`sdpo_stage_loss`] — the standard DPO gradient taken
/// against *this stage's* (handed-off) reference.
///
/// Per pair, with `L = −log σ(β·Δ)` and `Δ = (c − rc) − (l − rr)`,
/// `dL/dΔ = −β·σ(−β·Δ)`; the four partials follow the `+1, −1, −1, +1` pattern
/// and are scaled by `1 / batch.len()` for the mean reduction. (The staged
/// reference only changes *which* `ref_*` values are supplied; the closed form
/// is identical to [`crate::dpo::dpo::dpo_grad`].)
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for an empty stage, [`RlhfError::InvalidBeta`]
/// for invalid β, and [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn sdpo_stage_grad(
    stage_batch: &PairBatch,
    cfg: &SdpoConfig,
) -> RlhfResult<Vec<SdpoStageGrad>> {
    if stage_batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let inv_n = 1.0 / stage_batch.len() as f32;
    let mut grads = Vec::with_capacity(stage_batch.len());
    for (((&clp, &rlp), &rclp), &rrlp) in stage_batch
        .chosen_logps
        .iter()
        .zip(stage_batch.rejected_logps.iter())
        .zip(stage_batch.ref_chosen_logps.iter())
        .zip(stage_batch.ref_rejected_logps.iter())
    {
        let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta);
        let d_delta = -cfg.beta * sigmoid(-logit) * inv_n;
        let g = SdpoStageGrad {
            d_chosen_logp: d_delta,
            d_rejected_logp: -d_delta,
            d_ref_chosen_logp: -d_delta,
            d_ref_rejected_logp: d_delta,
        };
        if !g.d_chosen_logp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(g);
    }
    Ok(grads)
}

// ── Staged driver ──────────────────────────────────────────────────────────────

/// Stateful driver for the sDPO staged loop.
///
/// Holds the *current reference* chosen/rejected log-probs. Each call to
/// [`StagedDpo::run_stage`]:
///
/// 1. computes that stage's DPO loss using the **held** reference (overriding
///    whatever reference the incoming batch carried — this is the sDPO
///    property), then
/// 2. performs the reference handoff: the held reference is replaced by *this*
///    stage's policy log-probs, so the next stage trains against the
///    just-aligned model.
///
/// Before the first stage the held reference is the SFT reference, supplied via
/// [`StagedDpo::new`].
#[derive(Debug, Clone)]
pub struct StagedDpo {
    beta: f32,
    ref_chosen_logps: Vec<f32>,
    ref_rejected_logps: Vec<f32>,
    stages_run: usize,
}

impl StagedDpo {
    /// Create a driver seeded with the initial (SFT) reference log-probs.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::InvalidBeta`] if β ≤ 0 or non-finite, and
    /// [`RlhfError::MismatchedPairLength`] if the two reference vectors differ
    /// in length.
    pub fn new(
        beta: f32,
        init_ref_chosen_logps: Vec<f32>,
        init_ref_rejected_logps: Vec<f32>,
    ) -> RlhfResult<Self> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta });
        }
        if init_ref_chosen_logps.len() != init_ref_rejected_logps.len() {
            return Err(RlhfError::MismatchedPairLength {
                chosen: init_ref_chosen_logps.len(),
                rejected: init_ref_rejected_logps.len(),
            });
        }
        Ok(Self {
            beta,
            ref_chosen_logps: init_ref_chosen_logps,
            ref_rejected_logps: init_ref_rejected_logps,
            stages_run: stages_run_init(),
        })
    }

    /// The chosen-side reference log-probs currently held by the driver.
    #[must_use]
    pub fn ref_chosen_logps(&self) -> &[f32] {
        &self.ref_chosen_logps
    }

    /// The rejected-side reference log-probs currently held by the driver.
    #[must_use]
    pub fn ref_rejected_logps(&self) -> &[f32] {
        &self.ref_rejected_logps
    }

    /// Number of stages run so far.
    #[must_use]
    pub fn stages_run(&self) -> usize {
        self.stages_run
    }

    /// Run one sDPO stage against the held reference, then hand off the
    /// reference to this stage's policy log-probs.
    ///
    /// `policy_chosen_logps` / `policy_rejected_logps` are the current policy's
    /// log-probs on this stage's chosen / rejected responses. The held
    /// reference is used in place of any reference the data carried.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::EmptyInput`] if the policy vectors are empty,
    /// [`RlhfError::MismatchedPairLength`] if `policy_chosen_logps` and
    /// `policy_rejected_logps` differ in length,
    /// [`RlhfError::DimensionMismatch`] if they do not match the held
    /// reference's length, and [`RlhfError::NanEncountered`] on a NaN loss.
    pub fn run_stage(
        &mut self,
        policy_chosen_logps: &[f32],
        policy_rejected_logps: &[f32],
    ) -> RlhfResult<f32> {
        let n = policy_chosen_logps.len();
        if n == 0 {
            return Err(RlhfError::EmptyInput);
        }
        if policy_rejected_logps.len() != n {
            return Err(RlhfError::MismatchedPairLength {
                chosen: n,
                rejected: policy_rejected_logps.len(),
            });
        }
        if self.ref_chosen_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: self.ref_chosen_logps.len(),
                got: n,
            });
        }

        // ── Step 1: stage loss against the *held* reference ───────────────────
        let total: f32 = policy_chosen_logps
            .iter()
            .zip(policy_rejected_logps.iter())
            .zip(self.ref_chosen_logps.iter())
            .zip(self.ref_rejected_logps.iter())
            .map(|(((&clp, &rlp), &rclp), &rrlp)| {
                let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, self.beta);
                -log_sigmoid(logit)
            })
            .sum();
        let loss = total / n as f32;
        if loss.is_nan() {
            return Err(RlhfError::NanEncountered);
        }

        // ── Step 2: reference handoff (ref ← this stage's policy) ─────────────
        let (next_ref_chosen, next_ref_rejected) =
            sdpo_update_reference(policy_chosen_logps, policy_rejected_logps);
        self.ref_chosen_logps = next_ref_chosen;
        self.ref_rejected_logps = next_ref_rejected;
        self.stages_run += 1;

        Ok(loss)
    }
}

#[inline]
fn stages_run_init() -> usize {
    0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpo::dpo::{DpoConfig, dpo_loss};

    fn batch(
        chosen: Vec<f32>,
        rejected: Vec<f32>,
        ref_chosen: Vec<f32>,
        ref_rejected: Vec<f32>,
    ) -> PairBatch {
        PairBatch::new(chosen, rejected, ref_chosen, ref_rejected)
            .expect("valid batch in test fixture")
    }

    // 1. Single-stage sDPO equals standard dpo_loss exactly.
    #[test]
    fn single_stage_equals_dpo_loss() {
        let b = batch(
            vec![-0.5, -1.0, -1.5],
            vec![-2.0, -2.5, -3.0],
            vec![-1.0, -1.1, -1.2],
            vec![-1.0, -1.1, -1.2],
        );
        let beta = 0.3_f32;
        let sdpo_cfg = SdpoConfig { beta, n_stages: 1 };
        let dpo_cfg = DpoConfig { beta };
        let single = batch(
            b.chosen_logps.clone(),
            b.rejected_logps.clone(),
            b.ref_chosen_logps.clone(),
            b.ref_rejected_logps.clone(),
        );
        let sdpo =
            sdpo_total_loss(std::slice::from_ref(&b), &sdpo_cfg).expect("value should be present");
        let dpo = dpo_loss(&single, &dpo_cfg).expect("dpo_loss should succeed");
        assert!(
            (sdpo - dpo).abs() < 1e-6,
            "single-stage sDPO {sdpo} must equal dpo_loss {dpo}"
        );
    }

    // 2. sdpo_stage_loss equals dpo_loss for the same batch.
    #[test]
    fn stage_loss_equals_dpo_loss() {
        let b = batch(
            vec![-0.5, -2.0],
            vec![-2.5, -1.0],
            vec![-1.0, -1.0],
            vec![-1.0, -1.0],
        );
        let beta = 0.15_f32;
        let stage = sdpo_stage_loss(&b, &SdpoConfig { beta, n_stages: 1 })
            .expect("sdpo_stage_loss should succeed");
        let dpo = dpo_loss(&b, &DpoConfig { beta }).expect("dpo_loss should succeed");
        assert!((stage - dpo).abs() < 1e-6, "stage {stage} vs dpo {dpo}");
    }

    // 3. sdpo_update_reference copies the supplied previous-policy logps.
    #[test]
    fn update_reference_copies_previous_policy() {
        let prev_chosen = vec![-0.3, -0.7, -1.1];
        let prev_rejected = vec![-2.2, -2.6, -3.0];
        let (ref_c, ref_r) = sdpo_update_reference(&prev_chosen, &prev_rejected);
        assert_eq!(ref_c, prev_chosen);
        assert_eq!(ref_r, prev_rejected);
    }

    // 4. After run_stage, the held reference equals the just-passed policy logps.
    #[test]
    fn run_stage_hands_off_reference() {
        let mut driver =
            StagedDpo::new(0.2, vec![-1.0, -1.0], vec![-1.0, -1.0]).expect("new should succeed");
        let policy_chosen = vec![-0.4, -0.6];
        let policy_rejected = vec![-2.4, -2.6];
        let _ = driver
            .run_stage(&policy_chosen, &policy_rejected)
            .expect("run_stage should succeed");
        assert_eq!(
            driver.ref_chosen_logps(),
            policy_chosen.as_slice(),
            "held chosen reference must equal this stage's policy chosen logps"
        );
        assert_eq!(
            driver.ref_rejected_logps(),
            policy_rejected.as_slice(),
            "held rejected reference must equal this stage's policy rejected logps"
        );
    }

    // 5. run_stage initial loss uses the seeded (SFT) reference, equals dpo_loss.
    #[test]
    fn run_stage_first_loss_uses_seeded_reference() {
        let beta = 0.25_f32;
        let init_ref_chosen = vec![-1.0, -1.2];
        let init_ref_rejected = vec![-1.0, -1.2];
        let policy_chosen = vec![-0.5, -0.7];
        let policy_rejected = vec![-2.5, -2.7];
        let mut driver = StagedDpo::new(beta, init_ref_chosen.clone(), init_ref_rejected.clone())
            .expect("value should be present");
        let loss = driver
            .run_stage(&policy_chosen, &policy_rejected)
            .expect("run_stage should succeed");
        let equiv = batch(
            policy_chosen.clone(),
            policy_rejected.clone(),
            init_ref_chosen,
            init_ref_rejected,
        );
        let dpo = dpo_loss(&equiv, &DpoConfig { beta }).expect("dpo_loss should succeed");
        assert!((loss - dpo).abs() < 1e-6, "first stage {loss} vs dpo {dpo}");
    }

    // 6. stages_run increments per call.
    #[test]
    fn stages_run_increments() {
        let mut driver = StagedDpo::new(0.1, vec![-1.0], vec![-1.0]).expect("new should succeed");
        assert_eq!(driver.stages_run(), 0);
        let _ = driver
            .run_stage(&[-0.5], &[-2.0])
            .expect("run_stage should succeed");
        assert_eq!(driver.stages_run(), 1);
        let _ = driver
            .run_stage(&[-0.4], &[-2.5])
            .expect("run_stage should succeed");
        assert_eq!(driver.stages_run(), 2);
    }

    // 7. When the policy improves each stage, per-stage loss decreases monotonically.
    #[test]
    fn improving_policy_gives_monotonic_decreasing_loss() {
        let beta = 0.5_f32;
        let mut driver = StagedDpo::new(beta, vec![-1.0], vec![-1.0]).expect("new should succeed");
        // Stage configs: chosen logp rises (toward 0), rejected logp falls.
        let stages = [
            (-0.9_f32, -1.1_f32),
            (-0.7, -1.5),
            (-0.5, -2.0),
            (-0.3, -2.6),
        ];
        let mut losses = Vec::new();
        for &(c, r) in &stages {
            losses.push(
                driver
                    .run_stage(&[c], &[r])
                    .expect("run_stage should succeed"),
            );
        }
        for w in losses.windows(2) {
            assert!(
                w[1] < w[0],
                "loss should decrease as policy improves: {:?}",
                losses
            );
        }
    }

    // 8. Implicit-reward margin is positive when chosen is up-weighted vs rejected.
    #[test]
    fn margin_positive_when_chosen_upweighted() {
        let b = batch(vec![-0.2], vec![-3.0], vec![-1.0], vec![-1.0]);
        let m = sdpo_stage_margin(
            &b,
            &SdpoConfig {
                beta: 1.0,
                n_stages: 1,
            },
        )
        .expect("value should be present");
        assert!(m > 0.0, "chosen up-weighted → positive margin, got {m}");
    }

    // 9. Implicit-reward margin is negative when rejected is up-weighted.
    #[test]
    fn margin_negative_when_rejected_upweighted() {
        let b = batch(vec![-3.0], vec![-0.2], vec![-1.0], vec![-1.0]);
        let m = sdpo_stage_margin(
            &b,
            &SdpoConfig {
                beta: 1.0,
                n_stages: 1,
            },
        )
        .expect("value should be present");
        assert!(m < 0.0, "rejected up-weighted → negative margin, got {m}");
    }

    // 10. Multi-stage total loss equals the mean of per-stage losses.
    #[test]
    fn total_loss_is_mean_of_stage_losses() {
        let beta = 0.2_f32;
        let cfg = SdpoConfig { beta, n_stages: 2 };
        let s0 = batch(vec![-0.5], vec![-2.0], vec![-1.0], vec![-1.0]);
        let s1 = batch(vec![-0.4], vec![-2.5], vec![-0.5], vec![-2.0]);
        let l0 = sdpo_stage_loss(&s0, &cfg).expect("sdpo_stage_loss should succeed");
        let l1 = sdpo_stage_loss(&s1, &cfg).expect("sdpo_stage_loss should succeed");
        let total = sdpo_total_loss(&[s0, s1], &cfg).expect("sdpo_total_loss should succeed");
        assert!(
            (total - (l0 + l1) / 2.0).abs() < 1e-6,
            "total {total} vs mean {}",
            (l0 + l1) / 2.0
        );
    }

    // 11. Empty stages slice → EmptyInput.
    #[test]
    fn total_loss_empty_stages_errors() {
        let cfg = SdpoConfig {
            beta: 0.1,
            n_stages: 0,
        };
        assert!(matches!(
            sdpo_total_loss(&[], &cfg),
            Err(RlhfError::EmptyInput)
        ));
    }

    // 12. n_stages mismatch vs stages.len() → DimensionMismatch.
    #[test]
    fn total_loss_stage_count_mismatch_errors() {
        let cfg = SdpoConfig {
            beta: 0.1,
            n_stages: 3,
        };
        let s0 = batch(vec![-0.5], vec![-2.0], vec![-1.0], vec![-1.0]);
        assert!(matches!(
            sdpo_total_loss(std::slice::from_ref(&s0), &cfg),
            Err(RlhfError::DimensionMismatch {
                expected: 3,
                got: 1
            })
        ));
    }

    // 13. β ≤ 0 → InvalidBeta in stage loss.
    #[test]
    fn stage_loss_invalid_beta_errors() {
        let b = batch(vec![-0.5], vec![-2.0], vec![-1.0], vec![-1.0]);
        assert!(matches!(
            sdpo_stage_loss(
                &b,
                &SdpoConfig {
                    beta: 0.0,
                    n_stages: 1
                }
            ),
            Err(RlhfError::InvalidBeta { .. })
        ));
        assert!(matches!(
            sdpo_stage_loss(
                &b,
                &SdpoConfig {
                    beta: -0.3,
                    n_stages: 1
                }
            ),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 14. β ≤ 0 → InvalidBeta in total loss (via config validate).
    #[test]
    fn total_loss_invalid_beta_errors() {
        let s0 = batch(vec![-0.5], vec![-2.0], vec![-1.0], vec![-1.0]);
        let cfg = SdpoConfig {
            beta: -1.0,
            n_stages: 1,
        };
        assert!(matches!(
            sdpo_total_loss(std::slice::from_ref(&s0), &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 15. StagedDpo::new rejects invalid beta.
    #[test]
    fn new_rejects_invalid_beta() {
        assert!(matches!(
            StagedDpo::new(0.0, vec![-1.0], vec![-1.0]),
            Err(RlhfError::InvalidBeta { .. })
        ));
        assert!(matches!(
            StagedDpo::new(f32::NAN, vec![-1.0], vec![-1.0]),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 16. StagedDpo::new rejects mismatched reference lengths.
    #[test]
    fn new_rejects_ref_length_mismatch() {
        assert!(matches!(
            StagedDpo::new(0.1, vec![-1.0, -1.0], vec![-1.0]),
            Err(RlhfError::MismatchedPairLength { .. })
        ));
    }

    // 17. run_stage rejects per-stage policy length mismatch.
    #[test]
    fn run_stage_rejects_policy_length_mismatch() {
        let mut driver =
            StagedDpo::new(0.1, vec![-1.0, -1.0], vec![-1.0, -1.0]).expect("new should succeed");
        assert!(matches!(
            driver.run_stage(&[-0.5, -0.6], &[-2.0]),
            Err(RlhfError::MismatchedPairLength { .. })
        ));
    }

    // 18. run_stage rejects policy/reference length mismatch.
    #[test]
    fn run_stage_rejects_ref_length_mismatch() {
        let mut driver =
            StagedDpo::new(0.1, vec![-1.0, -1.0], vec![-1.0, -1.0]).expect("new should succeed");
        assert!(matches!(
            driver.run_stage(&[-0.5], &[-2.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 19. run_stage rejects empty policy input.
    #[test]
    fn run_stage_rejects_empty_policy() {
        let mut driver = StagedDpo::new(0.1, vec![], vec![]).expect("new should succeed");
        assert!(matches!(
            driver.run_stage(&[], &[]),
            Err(RlhfError::EmptyInput)
        ));
    }

    // 20. Two-stage driver: second stage trains against the first stage's policy.
    #[test]
    fn second_stage_trains_against_first_policy() {
        let beta = 0.4_f32;
        let mut driver = StagedDpo::new(beta, vec![-1.0], vec![-1.0]).expect("new should succeed");
        let p0_chosen = vec![-0.6];
        let p0_rejected = vec![-2.0];
        let _ = driver
            .run_stage(&p0_chosen, &p0_rejected)
            .expect("run_stage should succeed");
        // Stage 1: reference is now p0; loss computed manually must match.
        let p1_chosen = vec![-0.4];
        let p1_rejected = vec![-2.6];
        let loss1 = driver
            .run_stage(&p1_chosen, &p1_rejected)
            .expect("run_stage should succeed");
        let equiv = batch(
            p1_chosen.clone(),
            p1_rejected.clone(),
            p0_chosen.clone(),
            p0_rejected.clone(),
        );
        let manual = dpo_loss(&equiv, &DpoConfig { beta }).expect("dpo_loss should succeed");
        assert!(
            (loss1 - manual).abs() < 1e-6,
            "stage-1 loss {loss1} must use stage-0 policy as reference (manual {manual})"
        );
    }

    // 21. NaN policy logp → NanEncountered from run_stage.
    #[test]
    fn run_stage_nan_returns_error() {
        let mut driver = StagedDpo::new(0.1, vec![-1.0], vec![-1.0]).expect("new should succeed");
        assert!(matches!(
            driver.run_stage(&[f32::NAN], &[-2.0]),
            Err(RlhfError::NanEncountered)
        ));
    }

    // 22. Equal policy/reference logps → loss is exactly -log σ(0) = ln 2.
    #[test]
    fn equal_logps_loss_is_ln2() {
        let b = batch(vec![-1.0], vec![-1.0], vec![-1.0], vec![-1.0]);
        let loss = sdpo_stage_loss(
            &b,
            &SdpoConfig {
                beta: 0.5,
                n_stages: 1,
            },
        )
        .expect("value should be present");
        assert!(
            (loss - std::f32::consts::LN_2).abs() < 1e-6,
            "zero-margin loss should be ln 2, got {loss}"
        );
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

    fn bat() -> PairBatch {
        PairBatch::new(
            vec![-0.5_f32, -1.2],
            vec![-2.0_f32, -0.8],
            vec![-0.6_f32, -1.0],
            vec![-1.1_f32, -0.9],
        )
        .expect("batch")
    }

    #[test]
    fn sdpo_stage_grad_matches_fd() {
        let cfg = SdpoConfig {
            beta: 0.3,
            n_stages: 1,
        };
        let b = bat();
        let grads = sdpo_stage_grad(&b, &cfg).expect("grads");
        assert_eq!(grads.len(), 2);
        let h = 1e-2;
        let idx = 1usize;
        let fd_c = central_diff(
            |v| {
                let mut c = b.chosen_logps.clone();
                c[idx] = v;
                let nb = PairBatch::new(
                    c,
                    b.rejected_logps.clone(),
                    b.ref_chosen_logps.clone(),
                    b.ref_rejected_logps.clone(),
                )
                .expect("b");
                sdpo_stage_loss(&nb, &cfg).expect("loss")
            },
            b.chosen_logps[idx],
            h,
        );
        let fd_rc = central_diff(
            |v| {
                let mut rc = b.ref_chosen_logps.clone();
                rc[idx] = v;
                let nb = PairBatch::new(
                    b.chosen_logps.clone(),
                    b.rejected_logps.clone(),
                    rc,
                    b.ref_rejected_logps.clone(),
                )
                .expect("b");
                sdpo_stage_loss(&nb, &cfg).expect("loss")
            },
            b.ref_chosen_logps[idx],
            h,
        );
        assert_close(grads[idx].d_chosen_logp, fd_c, "d_chosen");
        assert_close(grads[idx].d_ref_chosen_logp, fd_rc, "d_ref_chosen");
    }

    #[test]
    fn sdpo_stage_grad_signs() {
        let cfg = SdpoConfig {
            beta: 0.5,
            n_stages: 1,
        };
        let b = PairBatch::new(vec![-1.0], vec![-1.0], vec![-1.0], vec![-1.0]).expect("b");
        let g = sdpo_stage_grad(&b, &cfg).expect("grad");
        assert!(g[0].d_chosen_logp < 0.0);
        assert!(g[0].d_rejected_logp > 0.0);
    }

    #[test]
    fn sdpo_stage_grad_invalid_beta_errors() {
        let cfg = SdpoConfig {
            beta: 0.0,
            n_stages: 1,
        };
        assert!(matches!(
            sdpo_stage_grad(&bat(), &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}
