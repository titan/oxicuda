//! Online DPO with rejection sampling.
//!
//! References:
//! * Guo et al. 2024, "Direct Language Model Alignment from Online AI
//!   Feedback", arXiv:2402.04792 (online / iterative DPO).
//! * Liu et al. 2024, "Statistical Rejection Sampling Improves Preference
//!   Optimization" (RSO), arXiv:2309.06657.
//!
//! In offline DPO the chosen/rejected pairs are fixed in advance. In the
//! *online* / rejection-sampling setting we instead draw `N` candidate
//! responses from the current policy, score each with a reward model, and
//! **synthesize** preference pairs on the fly: the highest-reward candidate is
//! treated as chosen and a lower-reward candidate as rejected. The standard DPO
//! loss is then computed on the synthesized pair(s).
//!
//! This module is deterministic: candidates are *provided* by the caller as
//! three parallel slices — the policy log-prob, the reference log-prob, and the
//! scalar reward of each candidate — so no RNG is involved here. Pair selection
//! is governed by [`PairingMode`].
//!
//! Tie-breaking rule: `argmax`/`argmin` over rewards both resolve to the
//! **lowest index** among tied candidates. Under [`PairingMode::Threshold`] a
//! reward gap below the margin yields [`RlhfError::NoValidPair`] instead.

use crate::dpo::dpo::{DpoConfig, dpo_log_ratio, dpo_loss_per_pair};
use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PreferencePair;

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

// ── Pairing mode ────────────────────────────────────────────────────────────

/// How preference pairs are synthesized from the scored candidates.
#[derive(Debug, Clone, PartialEq)]
pub enum PairingMode {
    /// Single pair: best (argmax reward) vs worst (argmin reward).
    BestWorst,
    /// `n − 1` pairs: best vs each of the other candidates.
    BestVsRest,
    /// Like [`PairingMode::BestWorst`] but only emits the pair when the reward
    /// gap `max − min` is at least the given margin; otherwise the pair is
    /// rejected with [`RlhfError::NoValidPair`].
    Threshold(f32),
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for online DPO with rejection sampling.
#[derive(Debug, Clone)]
pub struct OnlineDpoConfig {
    /// KL-regularisation temperature β. Must be positive and finite.
    pub beta: f32,
    /// Expected number of candidate responses per prompt. Must be ≥ 2; the
    /// provided slices must have exactly this length.
    pub n_candidates: usize,
    /// Strategy for synthesizing preference pairs from the candidates.
    pub pairing: PairingMode,
}

impl OnlineDpoConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if self.n_candidates < 2 {
            return Err(RlhfError::DimensionMismatch {
                expected: 2,
                got: self.n_candidates,
            });
        }
        Ok(())
    }
}

// ── Input validation ──────────────────────────────────────────────────────────

/// Validate the three parallel candidate slices against the config.
///
/// Ensures all three slices are non-empty, equal length, match
/// `cfg.n_candidates`, and that there are at least two candidates.
fn validate_inputs(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    cfg: &OnlineDpoConfig,
) -> RlhfResult<()> {
    cfg.validate()?;
    let n = candidate_logps.len();
    if n == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if ref_logps.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: ref_logps.len(),
        });
    }
    if rewards.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: rewards.len(),
        });
    }
    if n != cfg.n_candidates {
        return Err(RlhfError::DimensionMismatch {
            expected: cfg.n_candidates,
            got: n,
        });
    }
    if n < 2 {
        return Err(RlhfError::DimensionMismatch {
            expected: 2,
            got: n,
        });
    }
    Ok(())
}

/// Index of the maximum reward, breaking ties toward the lowest index.
///
/// `rewards` must be non-empty (callers validate this first).
fn argmax_low_index(rewards: &[f32]) -> RlhfResult<usize> {
    let mut best = 0_usize;
    let mut best_val = *rewards.first().ok_or(RlhfError::EmptyInput)?;
    for (i, &r) in rewards.iter().enumerate().skip(1) {
        if r > best_val {
            best_val = r;
            best = i;
        }
    }
    Ok(best)
}

/// Index of the minimum reward, breaking ties toward the lowest index.
///
/// `rewards` must be non-empty (callers validate this first).
fn argmin_low_index(rewards: &[f32]) -> RlhfResult<usize> {
    let mut worst = 0_usize;
    let mut worst_val = *rewards.first().ok_or(RlhfError::EmptyInput)?;
    for (i, &r) in rewards.iter().enumerate().skip(1) {
        if r < worst_val {
            worst_val = r;
            worst = i;
        }
    }
    Ok(worst)
}

/// Build a [`PreferencePair`] from candidate index `chosen_idx` (the chosen
/// response) and `rejected_idx` (the rejected response).
fn pair_from_indices(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    chosen_idx: usize,
    rejected_idx: usize,
) -> RlhfResult<PreferencePair> {
    let chosen_logp = *candidate_logps
        .get(chosen_idx)
        .ok_or(RlhfError::EmptyInput)?;
    let rejected_logp = *candidate_logps
        .get(rejected_idx)
        .ok_or(RlhfError::EmptyInput)?;
    let ref_chosen_logp = *ref_logps.get(chosen_idx).ok_or(RlhfError::EmptyInput)?;
    let ref_rejected_logp = *ref_logps.get(rejected_idx).ok_or(RlhfError::EmptyInput)?;
    Ok(PreferencePair {
        chosen_logp,
        rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
    })
}

// ── Pair construction ─────────────────────────────────────────────────────────

/// Synthesize a single preference pair from the scored candidates.
///
/// chosen = `argmax` reward, rejected = `argmin` reward (ties → lowest index).
/// For [`PairingMode::Threshold`]`(m)`, returns [`RlhfError::NoValidPair`] when
/// the reward gap `max − min` is strictly less than `m`. (For
/// [`PairingMode::BestVsRest`] this returns the best-vs-worst pair; use
/// [`online_dpo_pairs`] to obtain the full set of `n − 1` pairs.)
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for empty candidate slices,
/// [`RlhfError::DimensionMismatch`] for length mismatch / `n_candidates < 2`,
/// [`RlhfError::InvalidBeta`] for invalid β, [`RlhfError::NanEncountered`] if a
/// reward is NaN, and [`RlhfError::NoValidPair`] when a `Threshold` margin is
/// not met.
pub fn build_preference_pair(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    cfg: &OnlineDpoConfig,
) -> RlhfResult<PreferencePair> {
    validate_inputs(candidate_logps, ref_logps, rewards, cfg)?;
    if rewards.iter().any(|r| r.is_nan()) {
        return Err(RlhfError::NanEncountered);
    }

    let chosen_idx = argmax_low_index(rewards)?;
    let rejected_idx = argmin_low_index(rewards)?;

    if let PairingMode::Threshold(margin) = cfg.pairing {
        if !margin.is_finite() || margin < 0.0 {
            return Err(RlhfError::InvalidMargin { margin });
        }
        let gap = rewards[chosen_idx] - rewards[rejected_idx];
        if gap < margin {
            return Err(RlhfError::NoValidPair {
                msg: format!("reward gap {gap} below threshold margin {margin}"),
            });
        }
    }

    pair_from_indices(candidate_logps, ref_logps, chosen_idx, rejected_idx)
}

/// Synthesize the full set of preference pairs for [`PairingMode::BestVsRest`]:
/// the best candidate (argmax reward) paired against each of the other `n − 1`
/// candidates.
///
/// Each emitted pair has the best candidate as chosen and one of the remaining
/// candidates as rejected. The chosen reward is therefore ≥ every rejected
/// reward by construction.
///
/// # Errors
///
/// Returns the same input-validation errors as [`build_preference_pair`].
/// Returns [`RlhfError::Internal`] if called with a non-`BestVsRest` pairing
/// mode (use [`online_dpo_step`] for the single-pair modes).
pub fn online_dpo_pairs(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    cfg: &OnlineDpoConfig,
) -> RlhfResult<Vec<PreferencePair>> {
    validate_inputs(candidate_logps, ref_logps, rewards, cfg)?;
    if rewards.iter().any(|r| r.is_nan()) {
        return Err(RlhfError::NanEncountered);
    }
    if cfg.pairing != PairingMode::BestVsRest {
        return Err(RlhfError::Internal {
            msg: "online_dpo_pairs requires PairingMode::BestVsRest".to_string(),
        });
    }

    let best_idx = argmax_low_index(rewards)?;
    let n = candidate_logps.len();
    let mut pairs = Vec::with_capacity(n - 1);
    for j in 0..n {
        if j == best_idx {
            continue;
        }
        pairs.push(pair_from_indices(candidate_logps, ref_logps, best_idx, j)?);
    }
    Ok(pairs)
}

// ── End-to-end step ─────────────────────────────────────────────────────────

/// Run one online-DPO step for the single-pair modes
/// ([`PairingMode::BestWorst`] / [`PairingMode::Threshold`]): synthesize the
/// best-vs-worst preference pair and return it alongside its DPO loss.
///
/// The DPO loss is `-log σ(β · ((π_chosen − ref_chosen) − (π_rejected −
/// ref_rejected)))`, computed via [`dpo_loss_per_pair`]. It is non-negative
/// (since `-log σ(x) ≥ 0`) and finite for finite inputs.
///
/// # Errors
///
/// Returns [`RlhfError::Internal`] if called with [`PairingMode::BestVsRest`]
/// (use [`online_dpo_pairs`] then [`dpo_loss_per_pair`] per pair), plus any
/// error from [`build_preference_pair`] / [`dpo_loss_per_pair`].
pub fn online_dpo_step(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    cfg: &OnlineDpoConfig,
) -> RlhfResult<(PreferencePair, f32)> {
    if cfg.pairing == PairingMode::BestVsRest {
        return Err(RlhfError::Internal {
            msg: "online_dpo_step is for single-pair modes; use online_dpo_pairs for BestVsRest"
                .to_string(),
        });
    }
    let pair = build_preference_pair(candidate_logps, ref_logps, rewards, cfg)?;
    let loss = dpo_loss_per_pair(&pair, &DpoConfig { beta: cfg.beta })?;
    Ok((pair, loss))
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/// Gradient of the online-DPO step loss w.r.t. the candidate / reference log-probs.
///
/// Finite-difference verified against [`online_dpo_step`]'s returned loss. Both
/// vectors have the full candidate length; every entry is `0` except the chosen
/// (argmax-reward) and rejected (argmin-reward) candidates.
#[derive(Debug, Clone)]
pub struct OnlineDpoGrad {
    /// `∂L/∂(candidate policy log-prob)` for each candidate.
    pub d_candidate_logps: Vec<f32>,
    /// `∂L/∂(candidate reference log-prob)` for each candidate.
    pub d_ref_logps: Vec<f32>,
}

/// Analytic gradient of the single-pair online-DPO step ([`PairingMode::BestWorst`]
/// / [`PairingMode::Threshold`]).
///
/// Pair synthesis (argmax/argmin over the *rewards*) is held fixed — perturbing a
/// log-prob does not change which candidates are chosen/rejected — so the loss is
/// exactly the DPO loss on the selected pair, and its gradient is the standard
/// `dL/dΔ = −β·σ(−β·Δ)` distributed to the chosen / rejected candidate (and their
/// references) with the `+1, −1, −1, +1` sign pattern. All other candidates get
/// gradient `0`.
///
/// # Errors
///
/// Returns [`RlhfError::Internal`] for [`PairingMode::BestVsRest`] (mirrors
/// [`online_dpo_step`]), plus any error from [`build_preference_pair`].
pub fn online_dpo_grad(
    candidate_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    cfg: &OnlineDpoConfig,
) -> RlhfResult<OnlineDpoGrad> {
    if cfg.pairing == PairingMode::BestVsRest {
        return Err(RlhfError::Internal {
            msg: "online_dpo_grad is for single-pair modes; use online_dpo_pairs for BestVsRest"
                .to_string(),
        });
    }
    // Validates inputs / threshold / NaN and returns the pair being differentiated.
    let pair = build_preference_pair(candidate_logps, ref_logps, rewards, cfg)?;
    let chosen_idx = argmax_low_index(rewards)?;
    let rejected_idx = argmin_low_index(rewards)?;

    let logit = dpo_log_ratio(
        pair.chosen_logp,
        pair.ref_chosen_logp,
        pair.rejected_logp,
        pair.ref_rejected_logp,
        cfg.beta,
    );
    // dL/dΔ = −β·σ(−β·Δ).
    let d_delta = -cfg.beta * sigmoid(-logit);

    let n = candidate_logps.len();
    let mut d_candidate_logps = vec![0.0_f32; n];
    let mut d_ref_logps = vec![0.0_f32; n];
    // Accumulate (handles the degenerate chosen_idx == rejected_idx case → 0).
    d_candidate_logps[chosen_idx] += d_delta;
    d_candidate_logps[rejected_idx] += -d_delta;
    d_ref_logps[chosen_idx] += -d_delta;
    d_ref_logps[rejected_idx] += d_delta;

    if d_candidate_logps
        .iter()
        .chain(d_ref_logps.iter())
        .any(|g| !g.is_finite())
    {
        return Err(RlhfError::NanEncountered);
    }
    Ok(OnlineDpoGrad {
        d_candidate_logps,
        d_ref_logps,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_bw(beta: f32, n: usize) -> OnlineDpoConfig {
        OnlineDpoConfig {
            beta,
            n_candidates: n,
            pairing: PairingMode::BestWorst,
        }
    }

    // 1. Best/worst chosen correctly by reward.
    #[test]
    fn best_worst_selected_by_reward() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.5, -2.5, -3.5];
        let rewards = vec![0.2, 0.9, 0.1];
        let cfg = cfg_bw(0.1, 3);
        let pair = build_preference_pair(&logps, &refs, &rewards, &cfg)
            .expect("build_preference_pair should succeed");
        // best reward at index 1, worst at index 2
        assert!(
            (pair.chosen_logp - (-2.0)).abs() < 1e-6,
            "chosen logp = index 1"
        );
        assert!(
            (pair.rejected_logp - (-3.0)).abs() < 1e-6,
            "rejected logp = index 2"
        );
        assert!((pair.ref_chosen_logp - (-2.5)).abs() < 1e-6);
        assert!((pair.ref_rejected_logp - (-3.5)).abs() < 1e-6);
    }

    // 2. Chosen reward ≥ rejected reward always (here strictly greater).
    #[test]
    fn chosen_reward_geq_rejected() {
        let logps = vec![-1.0, -1.0, -1.0, -1.0];
        let refs = vec![-1.0, -1.0, -1.0, -1.0];
        let rewards = vec![0.3, 0.5, 0.1, 0.4];
        let cfg = cfg_bw(0.2, 4);
        let chosen = argmax_low_index(&rewards).expect("argmax_low_index should succeed");
        let rejected = argmin_low_index(&rewards).expect("argmin_low_index should succeed");
        let _ = build_preference_pair(&logps, &refs, &rewards, &cfg)
            .expect("build_preference_pair should succeed");
        assert!(rewards[chosen] >= rewards[rejected]);
        assert_eq!(chosen, 1);
        assert_eq!(rejected, 2);
    }

    // 3. Tied rewards resolve to low index (argmax and argmin both index 0).
    #[test]
    fn tied_rewards_low_index_rule() {
        let rewards = vec![0.5, 0.5, 0.5];
        assert_eq!(
            argmax_low_index(&rewards).expect("argmax_low_index should succeed"),
            0
        );
        assert_eq!(
            argmin_low_index(&rewards).expect("argmin_low_index should succeed"),
            0
        );
    }

    // 4. Tied rewards under Threshold → NoValidPair (gap 0 < margin).
    #[test]
    fn tied_rewards_threshold_no_valid_pair() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.5, 0.5];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::Threshold(0.01),
        };
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::NoValidPair { .. })
        ));
    }

    // 5. Built pair feeds dpo_loss_per_pair and yields finite ≥0 loss.
    #[test]
    fn built_pair_yields_finite_nonneg_loss() {
        let logps = vec![-0.5, -3.0];
        let refs = vec![-1.0, -1.0];
        let rewards = vec![0.9, 0.1];
        let cfg = cfg_bw(0.5, 2);
        let (_, loss) =
            online_dpo_step(&logps, &refs, &rewards, &cfg).expect("online_dpo_step should succeed");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(loss >= 0.0, "DPO loss = -log σ ≥ 0, got {loss}");
    }

    // 6. BestVsRest yields n-1 pairs.
    #[test]
    fn best_vs_rest_yields_n_minus_one_pairs() {
        let logps = vec![-1.0, -2.0, -3.0, -4.0];
        let refs = vec![-1.0, -1.0, -1.0, -1.0];
        let rewards = vec![0.1, 0.9, 0.2, 0.3];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 4,
            pairing: PairingMode::BestVsRest,
        };
        let pairs = online_dpo_pairs(&logps, &refs, &rewards, &cfg)
            .expect("online_dpo_pairs should succeed");
        assert_eq!(pairs.len(), 3, "BestVsRest → n-1 pairs");
        // Every pair's chosen is the best (index 1, logp -2.0).
        for p in &pairs {
            assert!((p.chosen_logp - (-2.0)).abs() < 1e-6);
        }
    }

    // 7. BestVsRest: chosen reward ≥ each rejected reward.
    #[test]
    fn best_vs_rest_chosen_dominates() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.0, -1.0, -1.0];
        let rewards = vec![0.4, 0.9, 0.7];
        let best = argmax_low_index(&rewards).expect("argmax_low_index should succeed");
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 3,
            pairing: PairingMode::BestVsRest,
        };
        let pairs = online_dpo_pairs(&logps, &refs, &rewards, &cfg)
            .expect("online_dpo_pairs should succeed");
        // best index = 1; rejected indices are 0 and 2.
        assert_eq!(best, 1);
        assert_eq!(pairs.len(), 2);
    }

    // 8. Threshold met → pair returned.
    #[test]
    fn threshold_met_returns_pair() {
        let logps = vec![-0.5, -3.0];
        let refs = vec![-1.0, -1.0];
        let rewards = vec![0.9, 0.1];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::Threshold(0.5),
        };
        let pair = build_preference_pair(&logps, &refs, &rewards, &cfg)
            .expect("build_preference_pair should succeed");
        assert!((pair.chosen_logp - (-0.5)).abs() < 1e-6);
    }

    // 9. Threshold not met → NoValidPair.
    #[test]
    fn threshold_not_met_no_valid_pair() {
        let logps = vec![-0.5, -3.0];
        let refs = vec![-1.0, -1.0];
        let rewards = vec![0.55, 0.5]; // gap 0.05 < 0.5
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::Threshold(0.5),
        };
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::NoValidPair { .. })
        ));
    }

    // 10. n_candidates < 2 → Err.
    #[test]
    fn n_candidates_too_small_errors() {
        let logps = vec![-1.0];
        let refs = vec![-1.0];
        let rewards = vec![0.5];
        let cfg = cfg_bw(0.1, 1);
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 11. Length mismatch among the three slices → Err (logps vs refs).
    #[test]
    fn length_mismatch_logps_refs_errors() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.1, 0.2, 0.3];
        let cfg = cfg_bw(0.1, 3);
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 12. Length mismatch among the three slices → Err (rewards short).
    #[test]
    fn length_mismatch_rewards_errors() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.0, -2.0, -3.0];
        let rewards = vec![0.1, 0.2];
        let cfg = cfg_bw(0.1, 3);
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 13. Empty rewards → Err.
    #[test]
    fn empty_rewards_errors() {
        let logps: Vec<f32> = vec![];
        let refs: Vec<f32> = vec![];
        let rewards: Vec<f32> = vec![];
        let cfg = cfg_bw(0.1, 2);
        // n_candidates(2) != 0 caught, but empty also returns EmptyInput first.
        let res = build_preference_pair(&logps, &refs, &rewards, &cfg);
        assert!(
            matches!(res, Err(RlhfError::EmptyInput))
                || matches!(res, Err(RlhfError::DimensionMismatch { .. })),
            "empty rewards should error"
        );
    }

    // 14. Deterministic given inputs (no RNG): two calls give identical pairs.
    #[test]
    fn deterministic_pair_selection() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.1, -2.1, -3.1];
        let rewards = vec![0.3, 0.9, 0.1];
        let cfg = cfg_bw(0.2, 3);
        let p1 = build_preference_pair(&logps, &refs, &rewards, &cfg)
            .expect("build_preference_pair should succeed");
        let p2 = build_preference_pair(&logps, &refs, &rewards, &cfg)
            .expect("build_preference_pair should succeed");
        assert!((p1.chosen_logp - p2.chosen_logp).abs() < 1e-12);
        assert!((p1.rejected_logp - p2.rejected_logp).abs() < 1e-12);
        assert!((p1.ref_chosen_logp - p2.ref_chosen_logp).abs() < 1e-12);
        assert!((p1.ref_rejected_logp - p2.ref_rejected_logp).abs() < 1e-12);
    }

    // 15. Invalid beta → Err.
    #[test]
    fn invalid_beta_errors() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.5, 0.1];
        let cfg = cfg_bw(0.0, 2);
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 16. NaN reward → NanEncountered.
    #[test]
    fn nan_reward_errors() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![f32::NAN, 0.1];
        let cfg = cfg_bw(0.1, 2);
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::NanEncountered)
        ));
    }

    // 17. online_dpo_step rejects BestVsRest.
    #[test]
    fn step_rejects_best_vs_rest() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.5, 0.1];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::BestVsRest,
        };
        assert!(matches!(
            online_dpo_step(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::Internal { .. })
        ));
    }

    // 18. online_dpo_pairs rejects non-BestVsRest modes.
    #[test]
    fn pairs_rejects_best_worst() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.5, 0.1];
        let cfg = cfg_bw(0.1, 2);
        assert!(matches!(
            online_dpo_pairs(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::Internal { .. })
        ));
    }

    // 19. Aligned pair (high chosen reward + high chosen logp) gives lower loss
    //     than a misaligned pair where the policy down-weights the chosen.
    #[test]
    fn aligned_pair_lower_loss_than_misaligned() {
        // Aligned: chosen has higher policy logp than reference.
        let logps_a = vec![-0.2, -3.0];
        let refs_a = vec![-1.0, -1.0];
        let rewards = vec![0.9, 0.1];
        let cfg = cfg_bw(0.5, 2);
        let (_, loss_a) = online_dpo_step(&logps_a, &refs_a, &rewards, &cfg)
            .expect("online_dpo_step should succeed");

        // Misaligned: chosen has lower policy logp than reference.
        let logps_m = vec![-3.0, -0.2];
        let refs_m = vec![-1.0, -1.0];
        let (_, loss_m) = online_dpo_step(&logps_m, &refs_m, &rewards, &cfg)
            .expect("online_dpo_step should succeed");
        assert!(
            loss_a < loss_m,
            "aligned loss {loss_a} should be < misaligned {loss_m}"
        );
    }

    // 20. Negative threshold margin → InvalidMargin.
    #[test]
    fn negative_threshold_margin_errors() {
        let logps = vec![-1.0, -2.0];
        let refs = vec![-1.0, -2.0];
        let rewards = vec![0.9, 0.1];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::Threshold(-0.5),
        };
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::InvalidMargin { .. })
        ));
    }

    // 21. n_candidates mismatch vs provided slice length → Err.
    #[test]
    fn n_candidates_mismatch_slice_len_errors() {
        let logps = vec![-1.0, -2.0, -3.0];
        let refs = vec![-1.0, -2.0, -3.0];
        let rewards = vec![0.1, 0.2, 0.3];
        let cfg = cfg_bw(0.1, 4); // claims 4 but slices are length 3
        assert!(matches!(
            build_preference_pair(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 22. Threshold gap exactly equal to margin → pair returned (>= rule).
    #[test]
    fn threshold_gap_equal_margin_returns_pair() {
        let logps = vec![-0.5, -3.0];
        let refs = vec![-1.0, -1.0];
        let rewards = vec![0.6, 0.1]; // gap exactly 0.5
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::Threshold(0.5),
        };
        let pair = build_preference_pair(&logps, &refs, &rewards, &cfg);
        assert!(pair.is_ok(), "gap == margin should be accepted");
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

    #[test]
    fn online_dpo_grad_matches_fd() {
        let logps = vec![-0.5_f32, -2.0, -3.0];
        let refs = vec![-1.0_f32, -1.5, -1.2];
        let rewards = vec![0.9_f32, 0.1, 0.5]; // chosen=0, rejected=1
        let cfg = OnlineDpoConfig {
            beta: 0.5,
            n_candidates: 3,
            pairing: PairingMode::BestWorst,
        };
        let g = online_dpo_grad(&logps, &refs, &rewards, &cfg).expect("grad");
        let h = 1e-2;
        for i in 0..logps.len() {
            let fd_lp = central_diff(
                |v| {
                    let mut l = logps.clone();
                    l[i] = v;
                    online_dpo_step(&l, &refs, &rewards, &cfg).expect("step").1
                },
                logps[i],
                h,
            );
            let fd_ref = central_diff(
                |v| {
                    let mut r = refs.clone();
                    r[i] = v;
                    online_dpo_step(&logps, &r, &rewards, &cfg).expect("step").1
                },
                refs[i],
                h,
            );
            assert_close(g.d_candidate_logps[i], fd_lp, "d_candidate");
            assert_close(g.d_ref_logps[i], fd_ref, "d_ref");
        }
    }

    #[test]
    fn online_dpo_grad_zero_off_selected() {
        let logps = vec![-0.5_f32, -2.0, -3.0];
        let refs = vec![-1.0_f32, -1.5, -1.2];
        let rewards = vec![0.9_f32, 0.1, 0.5]; // chosen=0, rejected=1; index 2 unused
        let cfg = OnlineDpoConfig {
            beta: 0.5,
            n_candidates: 3,
            pairing: PairingMode::BestWorst,
        };
        let g = online_dpo_grad(&logps, &refs, &rewards, &cfg).expect("grad");
        assert_eq!(g.d_candidate_logps[2], 0.0, "unselected candidate grad = 0");
        assert_eq!(g.d_ref_logps[2], 0.0);
        // Chosen pushed up (negative), rejected pushed down (positive).
        assert!(g.d_candidate_logps[0] < 0.0);
        assert!(g.d_candidate_logps[1] > 0.0);
    }

    #[test]
    fn online_dpo_grad_rejects_best_vs_rest() {
        let logps = vec![-1.0_f32, -2.0];
        let refs = vec![-1.0_f32, -2.0];
        let rewards = vec![0.5_f32, 0.1];
        let cfg = OnlineDpoConfig {
            beta: 0.1,
            n_candidates: 2,
            pairing: PairingMode::BestVsRest,
        };
        assert!(matches!(
            online_dpo_grad(&logps, &refs, &rewards, &cfg),
            Err(RlhfError::Internal { .. })
        ));
    }
}
