use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;

pub fn inner_sgd_step(params: &[f32], grads: &[f32], lr: f32) -> MetaResult<Vec<f32>> {
    if params.len() != grads.len() {
        return Err(MetaError::DimensionMismatch {
            expected: params.len(),
            got: grads.len(),
        });
    }
    if lr <= 0.0 || !lr.is_finite() {
        return Err(MetaError::InvalidLr { lr });
    }
    Ok(params
        .iter()
        .zip(grads.iter())
        .map(|(&p, &g)| p - lr * g)
        .collect())
}

pub fn multi_step_inner<F>(
    params: Vec<f32>,
    f: &F,
    lr: f32,
    n_steps: usize,
    eps: f32,
) -> MetaResult<Vec<f32>>
where
    F: Fn(&[f32]) -> f32,
{
    if lr <= 0.0 || !lr.is_finite() {
        return Err(MetaError::InvalidLr { lr });
    }
    let mut current = params;
    for _ in 0..n_steps {
        let grad = fd_gradient(&current, f, eps);
        current = inner_sgd_step(&current, &grad, lr)?;
    }
    Ok(current)
}

pub fn cross_entropy_loss(logits: &[f32], labels: &[u32], n_classes: usize) -> MetaResult<f32> {
    if n_classes == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "n_classes must be > 0".into(),
        });
    }
    if logits.len() != labels.len() * n_classes {
        return Err(MetaError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: logits.len(),
        });
    }

    let n = labels.len();
    if n == 0 {
        return Err(MetaError::EmptySupport);
    }

    let mut total_loss = 0.0_f32;

    for (i, &lbl) in labels.iter().enumerate() {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp == 0.0 {
            return Err(MetaError::NanEncountered {
                context: "cross_entropy sum_exp is zero".into(),
            });
        }
        let log_prob = (exps[lbl as usize] / sum_exp).ln();
        if !log_prob.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "cross_entropy log_prob is non-finite".into(),
            });
        }
        total_loss -= log_prob;
    }

    Ok(total_loss / n as f32)
}
