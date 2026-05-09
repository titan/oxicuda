use crate::error::{RlhfError, RlhfResult};

pub struct KtoConfig {
    pub beta: f32,
    pub lambda_d: f32,
    pub lambda_u: f32,
}

pub fn kto_loss(
    desirable_rewards: &[f32],
    undesirable_rewards: &[f32],
    cfg: &KtoConfig,
) -> RlhfResult<f32> {
    if desirable_rewards.is_empty() && undesirable_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.lambda_d.is_finite() || cfg.lambda_d < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_d,
        });
    }
    if !cfg.lambda_u.is_finite() || cfg.lambda_u < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_u,
        });
    }
    let z0 = std::f32::consts::LN_2;

    let desirable_loss = if desirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = desirable_rewards
            .iter()
            .map(|&r| {
                let arg = cfg.beta * (r - z0);
                1.0 - sigmoid(arg)
            })
            .sum();
        cfg.lambda_d * sum / desirable_rewards.len() as f32
    };

    let undesirable_loss = if undesirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = undesirable_rewards
            .iter()
            .map(|&r| {
                let arg = cfg.beta * (z0 - r);
                1.0 - sigmoid(arg)
            })
            .sum();
        cfg.lambda_u * sum / undesirable_rewards.len() as f32
    };

    let loss = desirable_loss + undesirable_loss;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
