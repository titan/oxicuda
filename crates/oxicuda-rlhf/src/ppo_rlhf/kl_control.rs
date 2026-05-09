use crate::error::{RlhfError, RlhfResult};

pub struct KlController {
    pub beta: f32,
    pub target_kl: f32,
    pub k_beta: f32,
}

impl KlController {
    pub fn new(init_beta: f32, target_kl: f32) -> Self {
        Self {
            beta: init_beta,
            target_kl,
            k_beta: 0.2,
        }
    }

    pub fn update_beta(&mut self, current_kl: f32) {
        let proportional_error = (current_kl - self.target_kl) / self.target_kl;
        self.beta *= 1.0 + self.k_beta * proportional_error;
        self.beta = self.beta.max(1e-6);
    }
}

pub fn kl_divergence_from_logps(log_probs: &[f32], ref_log_probs: &[f32]) -> RlhfResult<f32> {
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
        .map(|(&lp, &rlp)| lp - rlp)
        .sum::<f32>()
        / log_probs.len() as f32;
    if kl.is_nan() {
        return Err(RlhfError::KlDivergence {
            msg: "NaN in KL computation".into(),
        });
    }
    Ok(kl)
}
