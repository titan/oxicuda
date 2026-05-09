use crate::error::{RlhfError, RlhfResult};

pub struct OrpoConfig {
    pub lambda: f32,
}

pub fn log_odds(lp: f32) -> f32 {
    let clamped = lp.clamp(-30.0, -1e-6);
    let p = clamped.exp();
    let odds = p / (1.0 - p + 1e-7);
    odds.max(1e-7).ln()
}

pub fn orpo_loss(
    chosen_logps: &[f32],
    rejected_logps: &[f32],
    sft_loss: f32,
    cfg: &OrpoConfig,
) -> RlhfResult<f32> {
    if chosen_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_logps.len() != rejected_logps.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_logps.len(),
            rejected: rejected_logps.len(),
        });
    }
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(RlhfError::InvalidLambda { lambda: cfg.lambda });
    }
    if !sft_loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }

    let odds_ratio_sum: f32 = chosen_logps
        .iter()
        .zip(rejected_logps.iter())
        .map(|(&clp, &rlp)| {
            let lo_c = log_odds(clp);
            let lo_r = log_odds(rlp);
            let log_ratio = lo_c - lo_r;
            -log_sigmoid_stable(log_ratio)
        })
        .sum();
    let odds_penalty = odds_ratio_sum / chosen_logps.len() as f32;
    let loss = sft_loss + cfg.lambda * odds_penalty;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

fn log_sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}
