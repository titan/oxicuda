use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::inner_sgd_step;

pub struct AnilConfig {
    pub inner_lr: f32,
    pub n_inner_steps: usize,
    pub feat_dim: usize,
    pub n_classes: usize,
}

fn head_loss(
    head_params: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    feat_dim: usize,
    n_classes: usize,
) -> f32 {
    let n_support = support_y.len();
    if n_support == 0 {
        return 0.0;
    }
    let w = &head_params[..n_classes * feat_dim];
    let b = &head_params[n_classes * feat_dim..];

    let mut loss = 0.0_f32;
    for (s, feat) in support_feats.chunks(feat_dim).enumerate() {
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
            let lbl = support_y[s] as usize;
            let lp = (exps[lbl] / sum_e).ln();
            if lp.is_finite() {
                loss -= lp;
            }
        }
    }
    loss / n_support as f32
}

pub fn anil_adapt_head(
    head_params: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    cfg: &AnilConfig,
) -> MetaResult<Vec<f32>> {
    if cfg.inner_lr <= 0.0 {
        return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
    }
    let expected = cfg.n_classes * cfg.feat_dim + cfg.n_classes;
    if head_params.len() != expected {
        return Err(MetaError::DimensionMismatch {
            expected,
            got: head_params.len(),
        });
    }

    let mut adapted = head_params.to_vec();
    for _ in 0..cfg.n_inner_steps {
        let f = |p: &[f32]| head_loss(p, support_feats, support_y, cfg.feat_dim, cfg.n_classes);
        let grad = fd_gradient(&adapted, &f, 1e-4);
        adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
    }
    Ok(adapted)
}

pub fn anil_meta_update(
    head_params: &[f32],
    task_feats: &[(Vec<f32>, Vec<u32>)],
    outer_lr: f32,
    cfg: &AnilConfig,
) -> MetaResult<Vec<f32>> {
    if task_feats.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    if outer_lr <= 0.0 || !outer_lr.is_finite() {
        return Err(MetaError::InvalidLr { lr: outer_lr });
    }

    let n_params = head_params.len();
    let n_tasks = task_feats.len() as f32;
    let mut meta_grad = vec![0.0_f32; n_params];

    for (support_feats, support_y) in task_feats {
        let adapted = anil_adapt_head(head_params, support_feats, support_y, cfg)?;
        let f = |p: &[f32]| head_loss(p, support_feats, support_y, cfg.feat_dim, cfg.n_classes);
        let task_grad = fd_gradient(&adapted, &f, 1e-4);
        for (mg, &tg) in meta_grad.iter_mut().zip(task_grad.iter()) {
            *mg += tg / n_tasks;
        }
    }

    Ok(head_params
        .iter()
        .zip(meta_grad.iter())
        .map(|(&p, &g)| p - outer_lr * g)
        .collect())
}
