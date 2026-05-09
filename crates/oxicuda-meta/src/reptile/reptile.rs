use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::inner_sgd_step;

pub struct ReptileConfig {
    pub inner_lr: f32,
    pub n_inner_steps: usize,
    pub step_size: f32,
}

fn task_loss_flat(
    params: &[f32],
    support_x: &[f32],
    support_y: &[u32],
    n_classes: usize,
    feat_dim: usize,
) -> f32 {
    let n_support = support_y.len();
    if n_support == 0 {
        return 0.0;
    }
    let w = &params[..n_classes * feat_dim];
    let b = &params[n_classes * feat_dim..];

    let mut loss = 0.0_f32;
    for (s, feat) in support_x.chunks(feat_dim).enumerate() {
        let mut logits = vec![0.0_f32; n_classes];
        for c in 0..n_classes {
            let row = &w[c * feat_dim..(c + 1) * feat_dim];
            logits[c] = row
                .iter()
                .zip(feat.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b[c];
        }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&z| (z - max_l).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        if sum_e > 0.0 {
            let lp = (exps[support_y[s] as usize] / sum_e).ln();
            if lp.is_finite() {
                loss -= lp;
            }
        }
    }
    loss / n_support as f32
}

pub fn reptile_update(
    params: &[f32],
    task_data: &[(Vec<f32>, Vec<u32>)],
    n_classes: usize,
    feat_dim: usize,
    cfg: &ReptileConfig,
) -> MetaResult<Vec<f32>> {
    if task_data.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    if cfg.inner_lr <= 0.0 || !cfg.inner_lr.is_finite() {
        return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
    }
    if !cfg.step_size.is_finite() {
        return Err(MetaError::InvalidLr { lr: cfg.step_size });
    }

    let n_params = params.len();
    let n_tasks = task_data.len() as f32;
    let mut avg_adapted = vec![0.0_f32; n_params];

    for (support_x, support_y) in task_data {
        let mut adapted = params.to_vec();
        for _ in 0..cfg.n_inner_steps {
            let f = |p: &[f32]| task_loss_flat(p, support_x, support_y, n_classes, feat_dim);
            let grad = fd_gradient(&adapted, &f, 1e-4);
            adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
        }
        for (avg, &a) in avg_adapted.iter_mut().zip(adapted.iter()) {
            *avg += a / n_tasks;
        }
    }

    // θ ← θ + ε * (avg_θ' - θ)
    Ok(params
        .iter()
        .zip(avg_adapted.iter())
        .map(|(&p, &avg)| p + cfg.step_size * (avg - p))
        .collect())
}
