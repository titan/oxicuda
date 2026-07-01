use crate::error::{RlhfError, RlhfResult};

pub fn win_rate(chosen_rewards: &[f32], rejected_rewards: &[f32]) -> RlhfResult<f32> {
    if chosen_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_rewards.len() != rejected_rewards.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_rewards.len(),
            rejected: rejected_rewards.len(),
        });
    }
    let wins: f32 = chosen_rewards
        .iter()
        .zip(rejected_rewards.iter())
        .filter(|&(&c, &r)| c > r)
        .count() as f32;
    Ok(wins / chosen_rewards.len() as f32)
}

pub fn reward_gap(chosen_rewards: &[f32], rejected_rewards: &[f32]) -> RlhfResult<f32> {
    if chosen_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_rewards.len() != rejected_rewards.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_rewards.len(),
            rejected: rejected_rewards.len(),
        });
    }
    let gap: f32 = chosen_rewards
        .iter()
        .zip(rejected_rewards.iter())
        .map(|(&c, &r)| c - r)
        .sum::<f32>()
        / chosen_rewards.len() as f32;
    Ok(gap)
}

pub fn kl_from_ref(log_probs: &[f32], ref_log_probs: &[f32]) -> RlhfResult<f32> {
    if log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if log_probs.len() != ref_log_probs.len() {
        return Err(RlhfError::DimensionMismatch {
            expected: log_probs.len(),
            got: ref_log_probs.len(),
        });
    }
    let kl: f32 = log_probs
        .iter()
        .zip(ref_log_probs.iter())
        .map(|(&lp, &rlp)| lp.exp() * (lp - rlp))
        .sum::<f32>();
    if kl.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(kl)
}

pub fn perplexity(token_log_probs: &[f32]) -> RlhfResult<f32> {
    if token_log_probs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mean_lp = token_log_probs.iter().sum::<f32>() / token_log_probs.len() as f32;
    let ppl = (-mean_lp).exp();
    if ppl.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(ppl)
}

pub struct AlignmentMetrics {
    pub win_rate: f32,
    pub reward_gap: f32,
    pub kl_from_ref: f32,
    pub chosen_reward_mean: f32,
    pub rejected_reward_mean: f32,
}

pub fn compute_alignment_metrics(
    chosen_rs: &[f32],
    rejected_rs: &[f32],
    lps: &[f32],
    ref_lps: &[f32],
) -> RlhfResult<AlignmentMetrics> {
    if chosen_rs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let wr = win_rate(chosen_rs, rejected_rs)?;
    let rg = reward_gap(chosen_rs, rejected_rs)?;
    let kl = kl_from_ref(lps, ref_lps)?;
    let chosen_mean = chosen_rs.iter().sum::<f32>() / chosen_rs.len() as f32;
    let rejected_mean = rejected_rs.iter().sum::<f32>() / rejected_rs.len() as f32;
    Ok(AlignmentMetrics {
        win_rate: wr,
        reward_gap: rg,
        kl_from_ref: kl,
        chosen_reward_mean: chosen_mean,
        rejected_reward_mean: rejected_mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── win_rate ─────────────────────────────────────────────────────────────

    #[test]
    fn win_rate_perfect_alignment() {
        // Every chosen > rejected → win rate must equal 1.0.
        let chosen = [2.0_f32, 3.0, 5.0];
        let rejected = [1.0_f32, 1.5, 4.0];
        let wr = win_rate(&chosen, &rejected).expect("valid inputs");
        assert!(
            (wr - 1.0).abs() < 1e-6,
            "perfect alignment must give win_rate=1.0, got {wr}"
        );
    }

    #[test]
    fn win_rate_zero_alignment() {
        // Every chosen < rejected → win rate must equal 0.0.
        let chosen = [0.0_f32, 1.0, 2.0];
        let rejected = [1.0_f32, 2.0, 3.0];
        let wr = win_rate(&chosen, &rejected).expect("valid inputs");
        assert!(
            wr.abs() < 1e-6,
            "fully inverted alignment must give win_rate=0.0, got {wr}"
        );
    }

    #[test]
    fn win_rate_partial_analytic() {
        // Pairs: (3>2) ✓, (1<2) ✗, (5>4) ✓  → 2/3.
        let chosen = [3.0_f32, 1.0, 5.0];
        let rejected = [2.0_f32, 2.0, 4.0];
        let wr = win_rate(&chosen, &rejected).expect("valid inputs");
        let expected = 2.0_f32 / 3.0;
        assert!(
            (wr - expected).abs() < 1e-6,
            "expected win_rate={expected}, got {wr}"
        );
    }

    #[test]
    fn win_rate_empty_returns_error() {
        let err = win_rate(&[], &[]).expect_err("empty input must error");
        assert!(
            matches!(err, RlhfError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    #[test]
    fn win_rate_length_mismatch_returns_error() {
        let err = win_rate(&[1.0], &[1.0, 2.0]).expect_err("mismatched lengths must error");
        assert!(
            matches!(err, RlhfError::MismatchedPairLength { .. }),
            "expected MismatchedPairLength, got {err:?}"
        );
    }

    // ── reward_gap ───────────────────────────────────────────────────────────

    #[test]
    fn reward_gap_analytic() {
        // chosen=[2.0, 3.0], rejected=[1.0, 2.0]
        // differences = [1.0, 1.0], mean = 1.0.
        let chosen = [2.0_f32, 3.0];
        let rejected = [1.0_f32, 2.0];
        let gap = reward_gap(&chosen, &rejected).expect("valid inputs");
        assert!(
            (gap - 1.0).abs() < 1e-6,
            "expected reward_gap=1.0, got {gap}"
        );
    }

    // ── kl_from_ref ──────────────────────────────────────────────────────────

    #[test]
    fn kl_from_ref_equal_distributions_is_zero() {
        // KL(P ‖ P) = Σ p_i · (log p_i − log p_i) = 0 for any P.
        let lp = [-1.0_f32, -2.0, -3.0];
        let ref_lp = [-1.0_f32, -2.0, -3.0];
        let kl = kl_from_ref(&lp, &ref_lp).expect("valid inputs");
        assert!(
            kl.abs() < 1e-6,
            "KL of identical distributions must be 0, got {kl}"
        );
    }

    #[test]
    fn kl_from_ref_nonneg_for_proper_distribution() {
        // KL(P ‖ Q) ≥ 0 by Gibbs inequality.
        // P = [0.8, 0.2], Q = [0.5, 0.5] → KL ≈ 0.193.
        let lp = [0.8_f32.ln(), 0.2_f32.ln()];
        let ref_lp = [0.5_f32.ln(), 0.5_f32.ln()];
        let kl = kl_from_ref(&lp, &ref_lp).expect("valid inputs");
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {kl}");
        assert!(
            kl.is_finite(),
            "KL must be finite for valid distributions, got {kl}"
        );
    }

    // ── perplexity ───────────────────────────────────────────────────────────

    #[test]
    fn perplexity_analytic_exact() {
        // All log-probs = −1 → mean_lp = −1 → ppl = exp(1) = e.
        let lp = [-1.0_f32, -1.0, -1.0];
        let ppl = perplexity(&lp).expect("valid inputs");
        let expected = 1.0_f32.exp();
        assert!(
            (ppl - expected).abs() < 1e-5,
            "expected perplexity=exp(1)={expected:.6}, got {ppl}"
        );
    }

    // ── compute_alignment_metrics ─────────────────────────────────────────────

    #[test]
    fn compute_alignment_metrics_integration() {
        // chosen=[3.0, 2.0], rejected=[1.0, 1.0]: both chosen > rejected.
        // win_rate = 1.0, reward_gap = mean([2.0, 1.0]) = 1.5,
        // kl = 0 (equal log-probs), chosen_mean = 2.5, rejected_mean = 1.0.
        let chosen = [3.0_f32, 2.0];
        let rejected = [1.0_f32, 1.0];
        let lp = [-0.5_f32, -0.5];
        let ref_lp = [-0.5_f32, -0.5];
        let m = compute_alignment_metrics(&chosen, &rejected, &lp, &ref_lp).expect("valid inputs");
        assert!(
            (m.win_rate - 1.0).abs() < 1e-6,
            "win_rate: expected 1.0, got {}",
            m.win_rate
        );
        assert!(
            (m.reward_gap - 1.5).abs() < 1e-6,
            "reward_gap: expected 1.5, got {}",
            m.reward_gap
        );
        assert!(
            m.kl_from_ref.abs() < 1e-6,
            "kl_from_ref: expected 0.0 for equal log-probs, got {}",
            m.kl_from_ref
        );
        assert!(
            (m.chosen_reward_mean - 2.5).abs() < 1e-6,
            "chosen_reward_mean: expected 2.5, got {}",
            m.chosen_reward_mean
        );
        assert!(
            (m.rejected_reward_mean - 1.0).abs() < 1e-6,
            "rejected_reward_mean: expected 1.0, got {}",
            m.rejected_reward_mean
        );
    }
}
