use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::{PairBatch, PreferencePair};

pub struct DpoConfig {
    pub beta: f32,
}

pub fn dpo_log_ratio(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    beta: f32,
) -> f32 {
    let log_ratio_chosen = chosen_lp - ref_chosen_lp;
    let log_ratio_rejected = rejected_lp - ref_rejected_lp;
    beta * (log_ratio_chosen - log_ratio_rejected)
}

pub fn dpo_loss_per_pair(pair: &PreferencePair, cfg: &DpoConfig) -> RlhfResult<f32> {
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let logit = dpo_log_ratio(
        pair.chosen_logp,
        pair.ref_chosen_logp,
        pair.rejected_logp,
        pair.ref_rejected_logp,
        cfg.beta,
    );
    let loss = -log_sigmoid(logit);
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

pub fn dpo_loss(batch: &PairBatch, cfg: &DpoConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let total: f32 = batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta);
            -log_sigmoid(logit)
        })
        .sum();
    let loss = total / batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}
