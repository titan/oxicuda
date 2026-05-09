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
