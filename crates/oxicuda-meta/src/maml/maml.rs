use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::{cross_entropy_loss, inner_sgd_step};

pub struct MamlConfig {
    pub inner_lr: f32,
    pub n_inner_steps: usize,
}

pub(crate) fn task_loss_at_params(
    params: &[f32],
    support_x: &[f32],
    support_y: &[u32],
    n_classes: usize,
    feat_dim: usize,
) -> f32 {
    let n_support = support_y.len();
    if n_support == 0 || feat_dim == 0 {
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
    cross_entropy_loss(&logits, support_y, n_classes).unwrap_or(f32::MAX)
}

pub fn maml_adapt(
    params: &[f32],
    support_x: &[f32],
    support_y: &[u32],
    n_classes: usize,
    feat_dim: usize,
    cfg: &MamlConfig,
) -> MetaResult<Vec<f32>> {
    if cfg.inner_lr <= 0.0 {
        return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
    }
    let n_params = n_classes * feat_dim + n_classes;
    if params.len() != n_params {
        return Err(MetaError::DimensionMismatch {
            expected: n_params,
            got: params.len(),
        });
    }

    let mut adapted = params.to_vec();
    for _ in 0..cfg.n_inner_steps {
        let f = |p: &[f32]| task_loss_at_params(p, support_x, support_y, n_classes, feat_dim);
        let grad = fd_gradient(&adapted, &f, 1e-4);
        adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
    }
    Ok(adapted)
}

pub fn maml_meta_update(
    params: &[f32],
    task_data: &[(Vec<f32>, Vec<u32>)],
    n_classes: usize,
    feat_dim: usize,
    outer_lr: f32,
    cfg: &MamlConfig,
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
        // Second-order approximation: finite-diff around theta, computing loss
        // after n_inner_steps of SGD starting from theta
        let f = |p: &[f32]| {
            let mut inner = p.to_vec();
            for _ in 0..cfg.n_inner_steps {
                let g = fd_gradient(
                    &inner,
                    &|ip: &[f32]| {
                        task_loss_at_params(ip, support_x, support_y, n_classes, feat_dim)
                    },
                    1e-4,
                );
                for (param, gi) in inner.iter_mut().zip(g.iter()) {
                    *param -= cfg.inner_lr * gi;
                }
            }
            task_loss_at_params(&inner, support_x, support_y, n_classes, feat_dim)
        };

        let outer_grad = fd_gradient(params, &f, 1e-4);
        for (mg, &og) in meta_grad.iter_mut().zip(outer_grad.iter()) {
            *mg += og / n_tasks;
        }
    }

    Ok(params
        .iter()
        .zip(meta_grad.iter())
        .map(|(&p, &g)| p - outer_lr * g)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maml_adapt_changes_params_unit() {
        let n_classes = 2;
        let feat_dim = 4;
        let n_params = n_classes * feat_dim + n_classes;
        let params: Vec<f32> = (0..n_params).map(|i| i as f32 * 0.1).collect();
        let support_x = vec![1.0_f32; n_classes * feat_dim];
        let support_y: Vec<u32> = (0..n_classes as u32).collect();
        let cfg = MamlConfig {
            inner_lr: 0.01,
            n_inner_steps: 2,
        };
        let adapted = maml_adapt(&params, &support_x, &support_y, n_classes, feat_dim, &cfg)
            .expect("maml_adapt should succeed");
        assert_ne!(params, adapted);
    }
}
