//! RLAIF (Reinforcement Learning from AI Feedback) reward-modelling helpers.
//!
//! References:
//! * Bai et al. 2022, "Constitutional AI: Harmlessness from AI Feedback",
//!   arXiv:2212.08073 — RLAIF reward model trained on AI-labelled preferences.
//! * Lee et al. 2023, "RLAIF: Scaling RLHF with AI Feedback",
//!   arXiv:2309.00267 — soft AI-preference labels, position-bias debiasing,
//!   self-consistency.
//!
//! RLAIF replaces the human annotator with an *AI judge* (an off-the-shelf LLM)
//! that is shown two candidate responses and asked which is better. Rather than
//! a hard 0/1 label, the judge's normalised log-probability of choosing "A" over
//! "B" gives a **soft preference label** `p ∈ (0, 1)`. A reward model is then
//! trained on these soft labels with a soft-target Bradley-Terry cross-entropy.
//!
//! This module provides the CPU helpers that turn AI-judge outputs into reward
//! supervision:
//!
//! * [`soft_preference_from_logits`] — judge logits for "A"/"B" → soft label
//!   `σ(logit_A − logit_B)`.
//! * [`debias_position`] — average the soft labels from the two presentation
//!   orders (A-first and B-first) to cancel the judge's positional bias.
//! * [`self_consistency_label`] — aggregate several independent judge samples
//!   into one label (mean) with a disagreement (variance) score.
//! * [`soft_bt_reward_loss`] — soft-label Bradley-Terry cross-entropy
//!   `−[p·log σ(Δr) + (1−p)·log σ(−Δr)]` for training the reward model, where
//!   `Δr = r_A − r_B`.
//!
//! Everything is deterministic and validated; soft labels are clamped away from
//! the open-interval endpoints to keep the cross-entropy finite.

use crate::error::{RlhfError, RlhfResult};

/// Lower/upper clamp for soft labels so `log p` / `log(1−p)` stay finite.
const LABEL_EPS: f32 = 1e-6;

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically-stable `log σ(x)`.
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

// ── Soft preference from judge logits ───────────────────────────────────────

/// Soft preference label that response A is better than B, from the AI judge's
/// logits for the "A"/"B" choice: `p = σ(logit_a − logit_b)`.
///
/// The result is clamped to `[LABEL_EPS, 1 − LABEL_EPS]`.
///
/// # Errors
///
/// Returns [`RlhfError::NanEncountered`] for NaN logits.
pub fn soft_preference_from_logits(logit_a: f32, logit_b: f32) -> RlhfResult<f32> {
    if logit_a.is_nan() || logit_b.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    let p = sigmoid(logit_a - logit_b);
    Ok(p.clamp(LABEL_EPS, 1.0 - LABEL_EPS))
}

// ── Position-bias debiasing ─────────────────────────────────────────────────

/// Debias a pair of soft labels by averaging the two presentation orders.
///
/// `p_ab` is the judge's probability that A is better when A is shown *first*;
/// `p_ba_for_a` is the probability that A is better when A is shown *second*
/// (i.e. already converted to "A-is-better" orientation, `1 − p(B|B-first)`).
/// The debiased label is their mean, which cancels a constant positional bias.
///
/// # Errors
///
/// Returns [`RlhfError::NanEncountered`] for NaN inputs and
/// [`RlhfError::InvalidLambda`] if either input is outside `[0, 1]`.
pub fn debias_position(p_ab: f32, p_ba_for_a: f32) -> RlhfResult<f32> {
    for &p in &[p_ab, p_ba_for_a] {
        if p.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(RlhfError::InvalidLambda { lambda: p });
        }
    }
    let avg = 0.5 * (p_ab + p_ba_for_a);
    Ok(avg.clamp(LABEL_EPS, 1.0 - LABEL_EPS))
}

// ── Self-consistency aggregation ────────────────────────────────────────────

/// Aggregate several independent judge samples of the same pair into a single
/// soft label (mean) plus a disagreement score (population variance).
///
/// A low variance means the judge is self-consistent (confident); a high
/// variance flags an unreliable preference the caller may down-weight.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for no samples,
/// [`RlhfError::NanEncountered`] for NaN, and [`RlhfError::InvalidLambda`] for a
/// sample outside `[0, 1]`.
pub fn self_consistency_label(samples: &[f32]) -> RlhfResult<(f32, f32)> {
    if samples.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    for &p in samples {
        if p.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(RlhfError::InvalidLambda { lambda: p });
        }
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let var = samples
        .iter()
        .map(|&p| (p - mean) * (p - mean))
        .sum::<f32>()
        / n;
    Ok((mean.clamp(LABEL_EPS, 1.0 - LABEL_EPS), var))
}

// ── Soft-label Bradley-Terry reward loss ────────────────────────────────────

/// Soft-label Bradley-Terry cross-entropy for one pair:
/// `−[p·log σ(Δr) + (1−p)·log σ(−Δr)]`, with `Δr = reward_a − reward_b` and `p`
/// the soft AI-preference that A is better.
///
/// Reduces to the hard-label BT loss `−log σ(Δr)` when `p = 1`, and is minimised
/// (for given `p`) at `σ(Δr) = p`.
///
/// # Errors
///
/// Returns [`RlhfError::NanEncountered`] for NaN inputs and
/// [`RlhfError::InvalidLambda`] if `p ∉ [0, 1]`.
pub fn soft_bt_pair_loss(reward_a: f32, reward_b: f32, p: f32) -> RlhfResult<f32> {
    if reward_a.is_nan() || reward_b.is_nan() || p.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(RlhfError::InvalidLambda { lambda: p });
    }
    let delta = reward_a - reward_b;
    let loss = -(p * log_sigmoid(delta) + (1.0 - p) * log_sigmoid(-delta));
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean soft-label Bradley-Terry cross-entropy over a batch of pairs.
///
/// `rewards_a`, `rewards_b`, and `soft_labels` must be equal length.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], [`RlhfError::DimensionMismatch`], and
/// any error from [`soft_bt_pair_loss`].
pub fn soft_bt_reward_loss(
    rewards_a: &[f32],
    rewards_b: &[f32],
    soft_labels: &[f32],
) -> RlhfResult<f32> {
    let n = rewards_a.len();
    if n == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if rewards_b.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: rewards_b.len(),
        });
    }
    if soft_labels.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: soft_labels.len(),
        });
    }
    let mut total = 0.0_f32;
    for ((&ra, &rb), &p) in rewards_a
        .iter()
        .zip(rewards_b.iter())
        .zip(soft_labels.iter())
    {
        total += soft_bt_pair_loss(ra, rb, p)?;
    }
    Ok(total / n as f32)
}

// ── Soft-label Bradley-Terry gradient ───────────────────────────────────────

/// Gradient of the per-pair soft-BT loss w.r.t. the two rewards.
///
/// `L = −[p·log σ(Δ) + (1−p)·log σ(−Δ)]` with `Δ = r_a − r_b`. The cross-entropy
/// gradient collapses to `dL/dΔ = σ(Δ) − p`, so `∂L/∂r_a = σ(Δ) − p` and
/// `∂L/∂r_b = p − σ(Δ)`. The soft label `p` is held constant. Finite-difference
/// verified against [`soft_bt_pair_loss`].
///
/// # Errors
///
/// Returns [`RlhfError::NanEncountered`] for NaN inputs and
/// [`RlhfError::InvalidLambda`] if `p ∉ [0, 1]`.
pub fn soft_bt_pair_grad(reward_a: f32, reward_b: f32, p: f32) -> RlhfResult<(f32, f32)> {
    if reward_a.is_nan() || reward_b.is_nan() || p.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(RlhfError::InvalidLambda { lambda: p });
    }
    let delta = reward_a - reward_b;
    let d_delta = sigmoid(delta) - p;
    if !d_delta.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok((d_delta, -d_delta))
}

/// Gradient of the mean soft-BT reward loss w.r.t. the per-pair rewards.
///
/// Finite-difference verified against [`soft_bt_reward_loss`].
#[derive(Debug, Clone)]
pub struct SoftBtGrad {
    /// `∂L/∂r_a` for each pair (mean-scaled).
    pub d_rewards_a: Vec<f32>,
    /// `∂L/∂r_b` for each pair (mean-scaled).
    pub d_rewards_b: Vec<f32>,
}

/// Analytic gradient of the mean-reduced [`soft_bt_reward_loss`].
///
/// Each per-pair partial `σ(Δ_i) − p_i` is scaled by `1 / n` for the mean.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`], [`RlhfError::DimensionMismatch`], and any
/// error from [`soft_bt_pair_grad`].
pub fn soft_bt_reward_grad(
    rewards_a: &[f32],
    rewards_b: &[f32],
    soft_labels: &[f32],
) -> RlhfResult<SoftBtGrad> {
    let n = rewards_a.len();
    if n == 0 {
        return Err(RlhfError::EmptyInput);
    }
    if rewards_b.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: rewards_b.len(),
        });
    }
    if soft_labels.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: soft_labels.len(),
        });
    }
    let inv_n = 1.0 / n as f32;
    let mut d_rewards_a = Vec::with_capacity(n);
    let mut d_rewards_b = Vec::with_capacity(n);
    for ((&ra, &rb), &p) in rewards_a
        .iter()
        .zip(rewards_b.iter())
        .zip(soft_labels.iter())
    {
        let (da, db) = soft_bt_pair_grad(ra, rb, p)?;
        d_rewards_a.push(da * inv_n);
        d_rewards_b.push(db * inv_n);
    }
    Ok(SoftBtGrad {
        d_rewards_a,
        d_rewards_b,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Soft preference from equal logits is 0.5.
    #[test]
    fn equal_logits_half() {
        let p = soft_preference_from_logits(1.5, 1.5).expect("p");
        assert!((p - 0.5).abs() < 1e-6, "p {p}");
    }

    // 2. A-favoured logits give p > 0.5; B-favoured give p < 0.5.
    #[test]
    fn logit_ordering() {
        let a = soft_preference_from_logits(3.0, 0.0).expect("a");
        let b = soft_preference_from_logits(0.0, 3.0).expect("b");
        assert!(a > 0.5 && b < 0.5, "a {a} b {b}");
        assert!((a + b - 1.0).abs() < 1e-5, "symmetry a+b≈1");
    }

    // 3. Soft label is clamped strictly inside (0,1).
    #[test]
    fn soft_label_clamped() {
        let p = soft_preference_from_logits(100.0, -100.0).expect("p");
        assert!(p < 1.0 && p > 0.0, "p {p}");
        assert!((p - (1.0 - LABEL_EPS)).abs() < 1e-7);
    }

    // 4. Position debiasing averages the two orders.
    #[test]
    fn debias_averages_orders() {
        // A-first says 0.8, A-second (re-oriented) says 0.6 → debiased 0.7.
        let d = debias_position(0.8, 0.6).expect("d");
        assert!((d - 0.7).abs() < 1e-6, "d {d}");
    }

    // 5. Debiasing cancels a symmetric positional bias.
    #[test]
    fn debias_cancels_symmetric_bias() {
        // True preference 0.5 but judge favours whatever is first by +0.2:
        // A-first → 0.7 (A is first); when B is first, p(A better)=0.3.
        let d = debias_position(0.7, 0.3).expect("d");
        assert!(
            (d - 0.5).abs() < 1e-6,
            "symmetric bias should cancel to 0.5, got {d}"
        );
    }

    // 6. Self-consistency mean + variance.
    #[test]
    fn self_consistency_stats() {
        let (mean, var) = self_consistency_label(&[0.6, 0.8, 0.7]).expect("sc");
        assert!((mean - 0.7).abs() < 1e-6, "mean {mean}");
        // population variance of {0.6,0.8,0.7} = (0.01+0.01+0)/3 ≈ 0.006667
        assert!((var - 0.006_666_7).abs() < 1e-4, "var {var}");
    }

    // 7. Agreeing samples → near-zero variance.
    #[test]
    fn agreeing_samples_low_variance() {
        let (_, var) = self_consistency_label(&[0.9, 0.9, 0.9, 0.9]).expect("sc");
        assert!(var < 1e-9, "agreement → var≈0, got {var}");
    }

    // 8. Soft BT loss with p=1 equals hard BT loss −log σ(Δr).
    #[test]
    fn soft_bt_p1_equals_hard() {
        let ra = 2.0_f32;
        let rb = 0.5_f32;
        let soft = soft_bt_pair_loss(ra, rb, 1.0 - LABEL_EPS).expect("soft");
        let hard = -log_sigmoid(ra - rb);
        assert!((soft - hard).abs() < 1e-3, "soft {soft} vs hard {hard}");
    }

    // 9. Soft BT loss minimised when σ(Δr) = p.
    #[test]
    fn soft_bt_minimised_at_calibrated_reward() {
        // p = 0.7 → optimal Δr = logit(0.7) ≈ 0.8473.
        let p = 0.7_f32;
        let opt_delta = (p / (1.0 - p)).ln();
        let at_opt = soft_bt_pair_loss(opt_delta, 0.0, p).expect("opt");
        let too_high = soft_bt_pair_loss(opt_delta + 1.0, 0.0, p).expect("hi");
        let too_low = soft_bt_pair_loss(opt_delta - 1.0, 0.0, p).expect("lo");
        assert!(at_opt < too_high, "loss should rise above optimum");
        assert!(at_opt < too_low, "loss should rise below optimum");
    }

    // 10. Soft BT loss symmetric under (A,B,p) ↔ (B,A,1−p).
    #[test]
    fn soft_bt_symmetry() {
        let l1 = soft_bt_pair_loss(2.0, 0.5, 0.7).expect("l1");
        let l2 = soft_bt_pair_loss(0.5, 2.0, 0.3).expect("l2");
        assert!((l1 - l2).abs() < 1e-6, "l1 {l1} l2 {l2}");
    }

    // 11. Batch soft BT loss is the mean of per-pair losses.
    #[test]
    fn batch_is_mean_of_pairs() {
        let ra = [2.0_f32, 1.0];
        let rb = [0.5_f32, 1.5];
        let p = [0.8_f32, 0.4];
        let batch = soft_bt_reward_loss(&ra, &rb, &p).expect("batch");
        let l0 = soft_bt_pair_loss(ra[0], rb[0], p[0]).expect("l0");
        let l1 = soft_bt_pair_loss(ra[1], rb[1], p[1]).expect("l1");
        assert!((batch - (l0 + l1) / 2.0).abs() < 1e-6);
    }

    // 12. Lower-reward-on-preferred yields higher loss than aligned rewards.
    #[test]
    fn aligned_rewards_lower_loss() {
        // p favours A strongly.
        let aligned = soft_bt_reward_loss(&[3.0], &[0.0], &[0.9]).expect("aligned");
        let misaligned = soft_bt_reward_loss(&[0.0], &[3.0], &[0.9]).expect("misaligned");
        assert!(aligned < misaligned, "aligned {aligned} vs {misaligned}");
    }

    // 13. p out of [0,1] rejected.
    #[test]
    fn invalid_p_errors() {
        assert!(matches!(
            soft_bt_pair_loss(1.0, 0.0, 1.5),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            self_consistency_label(&[0.5, 1.2]),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            debias_position(0.5, -0.1),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 14. Length mismatch / empty rejected.
    #[test]
    fn shape_errors() {
        assert!(matches!(
            soft_bt_reward_loss(&[1.0, 2.0], &[0.0], &[0.5, 0.5]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            soft_bt_reward_loss(&[], &[], &[]),
            Err(RlhfError::EmptyInput)
        ));
        assert!(matches!(
            self_consistency_label(&[]),
            Err(RlhfError::EmptyInput)
        ));
    }

    // 15. NaN inputs rejected.
    #[test]
    fn nan_errors() {
        assert!(matches!(
            soft_preference_from_logits(f32::NAN, 1.0),
            Err(RlhfError::NanEncountered)
        ));
        assert!(matches!(
            soft_bt_pair_loss(f32::NAN, 0.0, 0.5),
            Err(RlhfError::NanEncountered)
        ));
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
    fn soft_bt_pair_grad_matches_fd() {
        let (ra, rb, p) = (2.0_f32, 0.5, 0.7);
        let (da, db) = soft_bt_pair_grad(ra, rb, p).expect("grad");
        let h = 1e-2;
        let fd_a = central_diff(|v| soft_bt_pair_loss(v, rb, p).expect("l"), ra, h);
        let fd_b = central_diff(|v| soft_bt_pair_loss(ra, v, p).expect("l"), rb, h);
        assert_close(da, fd_a, "d_reward_a");
        assert_close(db, fd_b, "d_reward_b");
    }

    #[test]
    fn soft_bt_grad_zero_at_calibrated_reward() {
        // L is minimised when σ(Δ) = p → gradient vanishes there.
        let p = 0.7_f32;
        let opt_delta = (p / (1.0 - p)).ln();
        let (da, db) = soft_bt_pair_grad(opt_delta, 0.0, p).expect("grad");
        assert!(da.abs() < 1e-4, "da {da}");
        assert!(db.abs() < 1e-4, "db {db}");
    }

    #[test]
    fn soft_bt_reward_grad_matches_fd() {
        let ra = [2.0_f32, 1.0];
        let rb = [0.5_f32, 1.5];
        let p = [0.8_f32, 0.4];
        let g = soft_bt_reward_grad(&ra, &rb, &p).expect("grad");
        let h = 1e-2;
        for i in 0..ra.len() {
            let fd_a = central_diff(
                |v| {
                    let mut a = ra.to_vec();
                    a[i] = v;
                    soft_bt_reward_loss(&a, &rb, &p).expect("loss")
                },
                ra[i],
                h,
            );
            assert_close(g.d_rewards_a[i], fd_a, "batch d_reward_a");
        }
        // Antisymmetry per pair.
        for (&da, &db) in g.d_rewards_a.iter().zip(g.d_rewards_b.iter()) {
            assert!((da + db).abs() < 1e-7);
        }
    }

    #[test]
    fn soft_bt_grad_invalid_p_errors() {
        assert!(matches!(
            soft_bt_pair_grad(1.0, 0.0, 1.5),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            soft_bt_reward_grad(&[1.0], &[0.0], &[]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }
}
