use crate::error::{RlhfError, RlhfResult};

pub struct SimpoConfig {
    pub beta: f32,
    pub gamma: f32,
}

pub fn simpo_loss(
    chosen_sum_logps: &[f32],
    rejected_sum_logps: &[f32],
    chosen_lengths: &[usize],
    rejected_lengths: &[usize],
    cfg: &SimpoConfig,
) -> RlhfResult<f32> {
    if chosen_sum_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let n = chosen_sum_logps.len();
    if rejected_sum_logps.len() != n {
        return Err(RlhfError::MismatchedPairLength {
            chosen: n,
            rejected: rejected_sum_logps.len(),
        });
    }
    if chosen_lengths.len() != n || rejected_lengths.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: chosen_lengths.len(),
        });
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.gamma.is_finite() {
        return Err(RlhfError::InvalidMargin { margin: cfg.gamma });
    }
    let total: f32 = chosen_sum_logps
        .iter()
        .zip(rejected_sum_logps.iter())
        .zip(chosen_lengths.iter())
        .zip(rejected_lengths.iter())
        .map(|(((&cslp, &rslp), &cl), &rl)| {
            let cl_f = cl.max(1) as f32;
            let rl_f = rl.max(1) as f32;
            let norm_c = cslp / cl_f;
            let norm_r = rslp / rl_f;
            let logit = cfg.beta * (norm_c - norm_r) - cfg.gamma;
            -log_sigmoid_stable(logit)
        })
        .sum();
    let loss = total / n as f32;
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
