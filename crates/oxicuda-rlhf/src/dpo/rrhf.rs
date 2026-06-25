//! RRHF — Rank Responses to align Human Feedback (Yuan et al. 2023).
//!
//! Reference: Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., & Huang, F. (2023).
//! *RRHF: Rank Responses to Align Language Models with Human Feedback without tears*.
//! NeurIPS 2023. <https://arxiv.org/abs/2304.05302>
//!
//! RRHF is a lightweight alternative to PPO that needs **no reference model, no reward model
//! at training time, and no value network** — just a set of candidate responses per prompt,
//! each annotated with a scalar reward. For a prompt with `k` sampled responses, let
//!
//! ```text
//!   p_i = ( Σ_t log π_θ(y_{i,t} | x, y_{i,<t}) ) / |y_i|          (length-normalised log-prob)
//! ```
//!
//! be the policy's *conditional, length-normalised* log-probability of response `i`, and let
//! `r_i` be its reward. RRHF optimises a pairwise **ranking (hinge) loss** that pushes the
//! model to score higher-reward responses above lower-reward ones:
//!
//! ```text
//!   L_rank = Σ_{ r_i < r_j }  max( 0,  p_i − p_j )
//! ```
//!
//! and adds an SFT-style term that maximises the length-normalised log-prob of the
//! **best-reward** response `i⋆ = argmax_i r_i`:
//!
//! ```text
//!   L_ft = − p_{i⋆}
//! ```
//!
//! The total objective is `L = L_rank + β · L_ft` (the paper uses `β = 1`).

use crate::error::{RlhfError, RlhfResult};

/// Configuration for RRHF.
#[derive(Debug, Clone)]
pub struct RrhfConfig {
    /// Weight `β` of the SFT (best-response) term relative to the ranking loss.
    ///
    /// Must be finite and `≥ 0`. The paper uses `1.0`.
    pub ft_weight: f32,
}

impl Default for RrhfConfig {
    fn default() -> Self {
        Self { ft_weight: 1.0 }
    }
}

impl RrhfConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.ft_weight.is_finite() || self.ft_weight < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.ft_weight,
            });
        }
        Ok(())
    }
}

/// A candidate set for one prompt: parallel arrays of summed log-probs, token lengths and
/// rewards (one entry per sampled response).
#[derive(Debug, Clone)]
pub struct RrhfSample {
    /// `Σ_t log π_θ(y_t | …)` for each response (summed, *not* averaged).
    pub sum_logps: Vec<f32>,
    /// Token length `|y_i|` of each response (must be `> 0`).
    pub lengths: Vec<usize>,
    /// Scalar reward `r_i` of each response.
    pub rewards: Vec<f32>,
}

impl RrhfSample {
    /// Number of candidate responses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sum_logps.len()
    }

    /// Returns `true` if there are no candidates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sum_logps.is_empty()
    }

    fn validate(&self) -> RlhfResult<()> {
        let k = self.sum_logps.len();
        if k == 0 {
            return Err(RlhfError::EmptyInput);
        }
        if self.lengths.len() != k {
            return Err(RlhfError::DimensionMismatch {
                expected: k,
                got: self.lengths.len(),
            });
        }
        if self.rewards.len() != k {
            return Err(RlhfError::DimensionMismatch {
                expected: k,
                got: self.rewards.len(),
            });
        }
        for &len in &self.lengths {
            if len == 0 {
                return Err(RlhfError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
        }
        for &v in &self.sum_logps {
            if v.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
        }
        for &v in &self.rewards {
            if v.is_nan() {
                return Err(RlhfError::NanEncountered);
            }
        }
        Ok(())
    }
}

/// Length-normalised conditional log-probabilities `p_i = sum_logp_i / |y_i|`.
///
/// # Errors
/// Propagates validation errors from [`RrhfSample`].
pub fn length_normalized_scores(sample: &RrhfSample) -> RlhfResult<Vec<f32>> {
    sample.validate()?;
    Ok(sample
        .sum_logps
        .iter()
        .zip(sample.lengths.iter())
        .map(|(&lp, &len)| lp / len as f32)
        .collect())
}

/// Pairwise ranking (hinge) loss `Σ_{r_i < r_j} max(0, p_i − p_j)`.
///
/// `scores` are the length-normalised log-probs `p`, `rewards` the scalar rewards `r`. Only
/// strictly-ordered reward pairs (`r_i < r_j`) contribute.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if `scores` is empty.
/// - [`RlhfError::DimensionMismatch`] if `scores.len() != rewards.len()`.
pub fn ranking_loss(scores: &[f32], rewards: &[f32]) -> RlhfResult<f32> {
    if scores.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if scores.len() != rewards.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: scores.len(),
            got: rewards.len(),
        });
    }
    let k = scores.len();
    let mut total = 0.0_f32;
    for i in 0..k {
        for j in 0..k {
            // Penalise when response j is preferred (higher reward) but the model scores i
            // at least as high: hinge on (p_i − p_j).
            if rewards[i] < rewards[j] {
                let margin = scores[i] - scores[j];
                if margin > 0.0 {
                    total += margin;
                }
            }
        }
    }
    Ok(total)
}

/// Index of the response with the maximum reward (ties broken by lowest index).
fn argmax_reward(rewards: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = rewards[0];
    for (i, &r) in rewards.iter().enumerate().skip(1) {
        if r > best_val {
            best_val = r;
            best = i;
        }
    }
    best
}

/// SFT (best-response) loss `− p_{i⋆}` where `i⋆ = argmax_i r_i`.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if `scores` is empty.
/// - [`RlhfError::DimensionMismatch`] if `scores.len() != rewards.len()`.
pub fn ft_loss(scores: &[f32], rewards: &[f32]) -> RlhfResult<f32> {
    if scores.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if scores.len() != rewards.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: scores.len(),
            got: rewards.len(),
        });
    }
    let best = argmax_reward(rewards);
    Ok(-scores[best])
}

/// Full RRHF loss for a single prompt's candidate set: `L_rank + β · L_ft`.
///
/// # Errors
/// - [`RlhfError::InvalidLambda`] if `cfg.ft_weight` is invalid.
/// - Propagates validation errors from [`RrhfSample`].
/// - [`RlhfError::NanEncountered`] if the result is NaN.
pub fn rrhf_loss(sample: &RrhfSample, cfg: &RrhfConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    let scores = length_normalized_scores(sample)?;
    let rank = ranking_loss(&scores, &sample.rewards)?;
    let ft = ft_loss(&scores, &sample.rewards)?;
    let loss = rank + cfg.ft_weight * ft;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean RRHF loss over a batch of candidate sets.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if the batch is empty.
/// - Propagates per-sample errors from [`rrhf_loss`].
pub fn rrhf_loss_batch(samples: &[RrhfSample], cfg: &RrhfConfig) -> RlhfResult<f32> {
    if samples.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let mut total = 0.0_f32;
    for s in samples {
        total += rrhf_loss(s, cfg)?;
    }
    Ok(total / samples.len() as f32)
}

/// Sub-gradient of the pairwise ranking (hinge) loss w.r.t. the scores `p`.
///
/// `L_rank = Σ_{r_i < r_j} max(0, p_i − p_j)`. Each active term (`r_i < r_j` and
/// `p_i > p_j`) contributes `+1` to `∂/∂p_i` and `−1` to `∂/∂p_j`; on the flat
/// side and at the hinge kink the contribution is `0`. Finite-difference
/// verified against [`ranking_loss`].
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if `scores` is empty.
/// - [`RlhfError::DimensionMismatch`] if `scores.len() != rewards.len()`.
pub fn ranking_grad(scores: &[f32], rewards: &[f32]) -> RlhfResult<Vec<f32>> {
    if scores.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if scores.len() != rewards.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: scores.len(),
            got: rewards.len(),
        });
    }
    let k = scores.len();
    let mut grad = vec![0.0_f32; k];
    for i in 0..k {
        for j in 0..k {
            if rewards[i] < rewards[j] {
                let margin = scores[i] - scores[j];
                if margin > 0.0 {
                    grad[i] += 1.0;
                    grad[j] -= 1.0;
                }
            }
        }
    }
    Ok(grad)
}

/// Gradient of the full RRHF loss w.r.t. the per-response summed log-probs.
///
/// Finite-difference verified against [`rrhf_loss`].
#[derive(Debug, Clone)]
pub struct RrhfGrad {
    /// `∂L/∂(sum_logp_i)` for each candidate response.
    pub d_sum_logps: Vec<f32>,
}

/// Analytic (sub-)gradient of [`rrhf_loss`] w.r.t. the summed log-probs.
///
/// `L = L_rank(p) + β · (−p_{i⋆})` with `p_i = sum_logp_i / |y_i|` and
/// `i⋆ = argmax_i r_i`. So `∂L/∂p_i = ranking_grad_i − β·[i = i⋆]`, and chaining
/// through the length normalisation gives `∂L/∂sum_logp_i = (∂L/∂p_i) / |y_i|`.
/// The rewards and lengths are held constant (they select the ordering and the
/// best response, not differentiated).
///
/// # Errors
/// - [`RlhfError::InvalidLambda`] if `cfg.ft_weight` is invalid.
/// - Propagates validation errors from [`RrhfSample`].
/// - [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn rrhf_grad(sample: &RrhfSample, cfg: &RrhfConfig) -> RlhfResult<RrhfGrad> {
    cfg.validate()?;
    let scores = length_normalized_scores(sample)?;
    let rank_grad = ranking_grad(&scores, &sample.rewards)?;
    let best = argmax_reward(&sample.rewards);
    let mut d_sum_logps = Vec::with_capacity(sample.len());
    for (i, &len) in sample.lengths.iter().enumerate() {
        let mut d_score = rank_grad[i];
        if i == best {
            d_score -= cfg.ft_weight;
        }
        let g = d_score / len as f32;
        if !g.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_sum_logps.push(g);
    }
    Ok(RrhfGrad { d_sum_logps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(sum_logps: &[f32], lengths: &[usize], rewards: &[f32]) -> RrhfSample {
        RrhfSample {
            sum_logps: sum_logps.to_vec(),
            lengths: lengths.to_vec(),
            rewards: rewards.to_vec(),
        }
    }

    #[test]
    fn length_normalized_divides_by_length() {
        let s = sample(&[-10.0, -6.0], &[5, 2], &[1.0, 0.0]);
        let p = length_normalized_scores(&s).expect("length_normalized_scores should succeed");
        assert!((p[0] - (-2.0)).abs() < 1e-6, "p0={}", p[0]);
        assert!((p[1] - (-3.0)).abs() < 1e-6, "p1={}", p[1]);
    }

    #[test]
    fn length_normalized_empty_errors() {
        let s = sample(&[], &[], &[]);
        assert!(matches!(
            length_normalized_scores(&s),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn length_normalized_zero_length_errors() {
        let s = sample(&[-1.0], &[0], &[1.0]);
        assert!(matches!(
            length_normalized_scores(&s),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn length_normalized_mismatched_rewards_errors() {
        let s = sample(&[-1.0, -2.0], &[1, 1], &[1.0]);
        assert!(matches!(
            length_normalized_scores(&s),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ranking_loss_zero_when_well_ordered() {
        // Higher reward → higher score → no violation.
        let scores = [-1.0_f32, -2.0, -3.0]; // p0 > p1 > p2
        let rewards = [3.0_f32, 2.0, 1.0]; // r0 > r1 > r2
        let l = ranking_loss(&scores, &rewards).expect("ranking_loss should succeed");
        assert!(l.abs() < 1e-6, "well-ordered → zero ranking loss, got {l}");
    }

    #[test]
    fn ranking_loss_positive_when_violated() {
        // Reward prefers index 1, but model scores index 0 higher → violation.
        let scores = [0.0_f32, -1.0];
        let rewards = [1.0_f32, 2.0]; // r1 > r0 but p0 > p1
        let l = ranking_loss(&scores, &rewards).expect("ranking_loss should succeed");
        assert!(
            (l - 1.0).abs() < 1e-6,
            "violation magnitude = p0 - p1 = 1.0, got {l}"
        );
    }

    #[test]
    fn ranking_loss_ignores_equal_rewards() {
        let scores = [5.0_f32, -5.0];
        let rewards = [1.0_f32, 1.0]; // equal → no constraint
        let l = ranking_loss(&scores, &rewards).expect("ranking_loss should succeed");
        assert!(l.abs() < 1e-6, "equal rewards → no loss, got {l}");
    }

    #[test]
    fn ranking_loss_dimension_mismatch_errors() {
        assert!(matches!(
            ranking_loss(&[1.0, 2.0], &[1.0]),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ft_loss_uses_best_reward_response() {
        let scores = [-2.0_f32, -1.0, -3.0];
        let rewards = [1.0_f32, 5.0, 2.0]; // best is index 1
        let l = ft_loss(&scores, &rewards).expect("ft_loss should succeed");
        assert!(
            (l - 1.0).abs() < 1e-6,
            "ft = -p_best = -(-1.0) = 1.0, got {l}"
        );
    }

    #[test]
    fn ft_loss_ties_pick_lowest_index() {
        let scores = [-1.0_f32, -2.0];
        let rewards = [3.0_f32, 3.0]; // tie → index 0
        let l = ft_loss(&scores, &rewards).expect("ft_loss should succeed");
        assert!((l - 1.0).abs() < 1e-6, "tie picks index 0, ft=1.0, got {l}");
    }

    #[test]
    fn rrhf_loss_finite_and_combines_terms() {
        let s = sample(&[-2.0, -4.0], &[2, 2], &[2.0, 1.0]);
        let cfg = RrhfConfig { ft_weight: 1.0 };
        let loss = rrhf_loss(&s, &cfg).expect("rrhf_loss should succeed");
        // scores = [-1, -2]; rank=0 (well ordered); ft = -(-1)=1 → loss=1
        assert!((loss - 1.0).abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn rrhf_loss_higher_when_misranked() {
        // Two candidates where the lower-reward one currently has higher norm log-prob.
        let good = sample(&[-2.0, -4.0], &[2, 2], &[2.0, 1.0]); // aligned
        let bad = sample(&[-4.0, -2.0], &[2, 2], &[2.0, 1.0]); // model prefers the worse one
        let cfg = RrhfConfig { ft_weight: 1.0 };
        let l_good = rrhf_loss(&good, &cfg).expect("rrhf_loss should succeed");
        let l_bad = rrhf_loss(&bad, &cfg).expect("rrhf_loss should succeed");
        assert!(
            l_bad > l_good,
            "misranked must have higher loss: good={l_good}, bad={l_bad}"
        );
    }

    #[test]
    fn rrhf_loss_ft_weight_zero_is_pure_ranking() {
        let s = sample(&[-4.0, -2.0], &[2, 2], &[2.0, 1.0]); // misranked → rank>0
        let cfg = RrhfConfig { ft_weight: 0.0 };
        let loss = rrhf_loss(&s, &cfg).expect("rrhf_loss should succeed");
        let scores = length_normalized_scores(&s).expect("length_normalized_scores should succeed");
        let rank = ranking_loss(&scores, &s.rewards).expect("ranking_loss should succeed");
        assert!(
            (loss - rank).abs() < 1e-6,
            "ft_weight=0 → loss == ranking_loss"
        );
    }

    #[test]
    fn rrhf_loss_invalid_ft_weight_errors() {
        let s = sample(&[-2.0, -4.0], &[2, 2], &[2.0, 1.0]);
        let cfg = RrhfConfig { ft_weight: -1.0 };
        assert!(matches!(
            rrhf_loss(&s, &cfg),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    #[test]
    fn rrhf_loss_nan_logp_errors() {
        let s = sample(&[f32::NAN, -4.0], &[2, 2], &[2.0, 1.0]);
        let cfg = RrhfConfig::default();
        assert!(matches!(
            rrhf_loss(&s, &cfg),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn rrhf_loss_batch_is_mean() {
        let s1 = sample(&[-2.0, -4.0], &[2, 2], &[2.0, 1.0]);
        let s2 = sample(&[-4.0, -2.0], &[2, 2], &[2.0, 1.0]);
        let cfg = RrhfConfig::default();
        let l1 = rrhf_loss(&s1, &cfg).expect("rrhf_loss should succeed");
        let l2 = rrhf_loss(&s2, &cfg).expect("rrhf_loss should succeed");
        let mean = rrhf_loss_batch(&[s1, s2], &cfg).expect("rrhf_loss_batch should succeed");
        assert!(
            (mean - (l1 + l2) / 2.0).abs() < 1e-5,
            "batch mean mismatch: {mean}"
        );
    }

    #[test]
    fn rrhf_loss_batch_empty_errors() {
        let cfg = RrhfConfig::default();
        assert!(matches!(
            rrhf_loss_batch(&[], &cfg),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn sample_len_and_is_empty() {
        let s = sample(&[-1.0, -2.0], &[1, 1], &[1.0, 0.0]);
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        let empty = sample(&[], &[], &[]);
        assert!(empty.is_empty());
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

    fn sample(sum_logps: &[f32], lengths: &[usize], rewards: &[f32]) -> RrhfSample {
        RrhfSample {
            sum_logps: sum_logps.to_vec(),
            lengths: lengths.to_vec(),
            rewards: rewards.to_vec(),
        }
    }

    #[test]
    fn ranking_grad_matches_fd_misranked() {
        // Misranked, well-separated margins so no hinge flips under ±h.
        let scores = [-2.0_f32, -1.0, -3.0];
        let rewards = [3.0_f32, 1.0, 2.0];
        let g = ranking_grad(&scores, &rewards).expect("grad");
        let h = 1e-2;
        for i in 0..scores.len() {
            let fd = central_diff(
                |v| {
                    let mut s = scores;
                    s[i] = v;
                    ranking_loss(&s, &rewards).expect("loss")
                },
                scores[i],
                h,
            );
            assert_close(g[i], fd, "ranking_grad");
        }
        // Closed form: g0 = -1, g1 = +2, g2 = -1.
        assert!((g[0] + 1.0).abs() < 1e-7);
        assert!((g[1] - 2.0).abs() < 1e-7);
        assert!((g[2] + 1.0).abs() < 1e-7);
    }

    #[test]
    fn ranking_grad_zero_when_well_ordered() {
        let scores = [-1.0_f32, -2.0, -3.0];
        let rewards = [3.0_f32, 2.0, 1.0];
        let g = ranking_grad(&scores, &rewards).expect("grad");
        for &gi in &g {
            assert_eq!(gi, 0.0, "well-ordered → zero ranking gradient");
        }
    }

    #[test]
    fn rrhf_grad_matches_fd() {
        // Misranked sample, distinct rewards, length-normalised.
        let s = sample(&[-2.0, -1.0, -3.0], &[1, 1, 1], &[3.0, 1.0, 2.0]);
        let cfg = RrhfConfig { ft_weight: 1.0 };
        let g = rrhf_grad(&s, &cfg).expect("grad");
        let h = 1e-2;
        for i in 0..s.len() {
            let fd = central_diff(
                |v| {
                    let mut ss = s.clone();
                    ss.sum_logps[i] = v;
                    rrhf_loss(&ss, &cfg).expect("loss")
                },
                s.sum_logps[i],
                h,
            );
            assert_close(g.d_sum_logps[i], fd, "rrhf_grad");
        }
    }

    #[test]
    fn rrhf_grad_length_normalised() {
        // Non-unit lengths; chosen so the normalised scores [-2,-1,-3] are
        // distinct and the hinge margins are well separated from 0.
        let s = sample(&[-4.0, -1.0, -9.0], &[2, 1, 3], &[3.0, 1.0, 2.0]);
        let cfg = RrhfConfig { ft_weight: 0.5 };
        let g = rrhf_grad(&s, &cfg).expect("grad");
        let h = 1e-2;
        for i in 0..s.len() {
            let fd = central_diff(
                |v| {
                    let mut ss = s.clone();
                    ss.sum_logps[i] = v;
                    rrhf_loss(&ss, &cfg).expect("loss")
                },
                s.sum_logps[i],
                h,
            );
            assert_close(g.d_sum_logps[i], fd, "rrhf_grad_len");
        }
    }

    #[test]
    fn rrhf_grad_best_response_pushed_up() {
        // ft term pushes the best-reward response's log-prob up (negative grad).
        let s = sample(&[-1.0, -2.0], &[1, 1], &[2.0, 1.0]); // well-ordered → rank grad 0
        let cfg = RrhfConfig { ft_weight: 1.0 };
        let g = rrhf_grad(&s, &cfg).expect("grad");
        assert!(g.d_sum_logps[0] < 0.0, "best response pushed up");
    }

    #[test]
    fn rrhf_grad_invalid_ft_weight_errors() {
        let s = sample(&[-2.0, -4.0], &[2, 2], &[2.0, 1.0]);
        let cfg = RrhfConfig { ft_weight: -1.0 };
        assert!(matches!(
            rrhf_grad(&s, &cfg),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }
}
