use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PairBatch;

pub struct IpoConfig {
    pub beta: f32,
}

pub fn ipo_loss(batch: &PairBatch, cfg: &IpoConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let target = 1.0 / (2.0 * cfg.beta);
    let total: f32 = batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let h = (clp - rclp) - (rlp - rrlp);
            let diff = h - target;
            diff * diff
        })
        .sum();
    let loss = total / batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}
