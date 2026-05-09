use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::inner_sgd_step;

pub struct FoMamlConfig {
    pub inner_lr: f32,
    pub n_inner_steps: usize,
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
    let mut logits = vec![0.0_f32; n_support * n_classes];
    for (s, feat) in support_x.chunks(feat_dim).enumerate() {
        let w = &params[..n_classes * feat_dim];
        let b = &params[n_classes * feat_dim..];
        for c in 0..n_classes {
            let row = &w[c * feat_dim..(c + 1) * feat_dim];
            logits[s * n_classes + c] = row
                .iter()
                .zip(feat.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b[c];
        }
    }
    let mut loss = 0.0_f32;
    for (s, &lbl) in support_y.iter().enumerate() {
        let row = &logits[s * n_classes..(s + 1) * n_classes];
        let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&z| (z - max_l).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        if sum_e > 0.0 {
            let lp = (exps[lbl as usize] / sum_e).ln();
            if lp.is_finite() {
                loss -= lp;
            }
        }
    }
    loss / n_support as f32
}

fn fomaml_adapt(
    params: &[f32],
    support_x: &[f32],
    support_y: &[u32],
    n_classes: usize,
    feat_dim: usize,
    cfg: &FoMamlConfig,
) -> MetaResult<Vec<f32>> {
    let mut adapted = params.to_vec();
    for _ in 0..cfg.n_inner_steps {
        let f = |p: &[f32]| task_loss_flat(p, support_x, support_y, n_classes, feat_dim);
        let grad = fd_gradient(&adapted, &f, 1e-4);
        adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
    }
    Ok(adapted)
}

pub fn fomaml_update(
    params: &[f32],
    task_data: &[(Vec<f32>, Vec<u32>)],
    n_classes: usize,
    feat_dim: usize,
    outer_lr: f32,
    cfg: &FoMamlConfig,
) -> MetaResult<Vec<f32>> {
    if task_data.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    if outer_lr <= 0.0 || !outer_lr.is_finite() {
        return Err(MetaError::InvalidLr { lr: outer_lr });
    }

    let n_params = params.len();
    let n_tasks = task_data.len() as f32;
    let mut meta_grad = vec![0.0_f32; n_params];

    for (support_x, support_y) in task_data {
        // FOMAML: gradient at adapted params (no second-order terms)
        let adapted = fomaml_adapt(params, support_x, support_y, n_classes, feat_dim, cfg)?;
        let f = |p: &[f32]| task_loss_flat(p, support_x, support_y, n_classes, feat_dim);
        let task_grad = fd_gradient(&adapted, &f, 1e-4);
        for (mg, &tg) in meta_grad.iter_mut().zip(task_grad.iter()) {
            *mg += tg / n_tasks;
        }
    }

    Ok(params
        .iter()
        .zip(meta_grad.iter())
        .map(|(&p, &g)| p - outer_lr * g)
        .collect())
}
