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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const N_CLASSES: usize = 2;
    const FEAT_DIM: usize = 3;
    const N_PARAMS: usize = N_CLASSES * FEAT_DIM + N_CLASSES;

    fn base_params() -> Vec<f32> {
        vec![0.2, -0.1, 0.05, 0.1, -0.2, 0.15, 0.0, 0.0]
    }

    fn task_a() -> (Vec<f32>, Vec<u32>) {
        (vec![1.0, 0.5, -0.5, -0.3, 0.8, 0.2], vec![0_u32, 1])
    }

    fn task_b() -> (Vec<f32>, Vec<u32>) {
        (vec![-0.7, 0.4, 0.9, 0.6, -0.2, -0.1], vec![1_u32, 0])
    }

    // Recover the meta-gradient from the outer step: θ' = θ − η·g ⇒ g = (θ − θ')/η.
    fn recovered_meta_grad(params: &[f32], updated: &[f32], outer_lr: f32) -> Vec<f32> {
        params
            .iter()
            .zip(updated.iter())
            .map(|(&p, &u)| (p - u) / outer_lr)
            .collect()
    }

    // DEFINING FOMAML PROPERTY: the meta-gradient is ∇L evaluated at the
    // ADAPTED params (no Hessian / second-order term). On a single task the
    // recovered meta-grad must equal fd_gradient(loss, θ_adapted) exactly.
    #[test]
    fn fomaml_meta_grad_is_gradient_at_adapted_params() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = FoMamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 3,
        };
        let outer_lr = 0.1_f32;
        let adapted = fomaml_adapt(&params, &sx, &sy, N_CLASSES, FEAT_DIM, &cfg).expect("adapt");
        assert_ne!(adapted, params, "inner loop must adapt params");
        let f = |p: &[f32]| task_loss_flat(p, &sx, &sy, N_CLASSES, FEAT_DIM);
        let grad_at_adapted = fd_gradient(&adapted, &f, 1e-4);
        let updated = fomaml_update(
            &params,
            &[(sx.clone(), sy.clone())],
            N_CLASSES,
            FEAT_DIM,
            outer_lr,
            &cfg,
        )
        .expect("fomaml");
        let meta_grad = recovered_meta_grad(&params, &updated, outer_lr);
        for i in 0..N_PARAMS {
            assert!(
                (meta_grad[i] - grad_at_adapted[i]).abs() < 1e-4,
                "meta-grad must be ∇L at θ_adapted at {i}: {} vs {}",
                meta_grad[i],
                grad_at_adapted[i]
            );
        }
    }

    // CONTRAST: the FOMAML meta-grad (at θ_adapted) differs from the gradient
    // at θ₀ whenever adaptation actually moved the params (not yet converged).
    #[test]
    fn fomaml_meta_grad_differs_from_theta0_gradient() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = FoMamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 3,
        };
        let outer_lr = 0.1_f32;
        let f = |p: &[f32]| task_loss_flat(p, &sx, &sy, N_CLASSES, FEAT_DIM);
        let grad_at_theta0 = fd_gradient(&params, &f, 1e-4);
        let updated = fomaml_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, outer_lr, &cfg)
            .expect("fomaml");
        let meta_grad = recovered_meta_grad(&params, &updated, outer_lr);
        let differs = (0..N_PARAMS).any(|i| (meta_grad[i] - grad_at_theta0[i]).abs() > 1e-3);
        assert!(
            differs,
            "FOMAML meta-grad (at θ_adapted) must differ from the θ₀ gradient when not converged"
        );
    }

    // With n_inner_steps = 0 there is no adaptation, so θ_adapted = θ₀ and the
    // FOMAML meta-grad reduces EXACTLY to the gradient at θ₀.
    #[test]
    fn fomaml_zero_inner_steps_reduces_to_theta0_gradient() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = FoMamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 0,
        };
        let outer_lr = 0.1_f32;
        let f = |p: &[f32]| task_loss_flat(p, &sx, &sy, N_CLASSES, FEAT_DIM);
        let grad_at_theta0 = fd_gradient(&params, &f, 1e-4);
        let updated = fomaml_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, outer_lr, &cfg)
            .expect("fomaml");
        let meta_grad = recovered_meta_grad(&params, &updated, outer_lr);
        for i in 0..N_PARAMS {
            assert!(
                (meta_grad[i] - grad_at_theta0[i]).abs() < 1e-4,
                "no-adapt FOMAML must equal θ₀ gradient at {i}: {} vs {}",
                meta_grad[i],
                grad_at_theta0[i]
            );
        }
    }

    // Multi-task: the meta-gradient is the MEAN of per-task gradients at each
    // task's adapted params.
    #[test]
    fn fomaml_multi_task_averages_adapted_gradients() {
        let params = base_params();
        let (sx_a, sy_a) = task_a();
        let (sx_b, sy_b) = task_b();
        let cfg = FoMamlConfig {
            inner_lr: 0.1,
            n_inner_steps: 2,
        };
        let outer_lr = 0.1_f32;
        let adapt_a =
            fomaml_adapt(&params, &sx_a, &sy_a, N_CLASSES, FEAT_DIM, &cfg).expect("adapt a");
        let adapt_b =
            fomaml_adapt(&params, &sx_b, &sy_b, N_CLASSES, FEAT_DIM, &cfg).expect("adapt b");
        let fa = |p: &[f32]| task_loss_flat(p, &sx_a, &sy_a, N_CLASSES, FEAT_DIM);
        let fb = |p: &[f32]| task_loss_flat(p, &sx_b, &sy_b, N_CLASSES, FEAT_DIM);
        let ga = fd_gradient(&adapt_a, &fa, 1e-4);
        let gb = fd_gradient(&adapt_b, &fb, 1e-4);
        let tasks = vec![(sx_a, sy_a), (sx_b, sy_b)];
        let updated =
            fomaml_update(&params, &tasks, N_CLASSES, FEAT_DIM, outer_lr, &cfg).expect("fomaml");
        let meta_grad = recovered_meta_grad(&params, &updated, outer_lr);
        for i in 0..N_PARAMS {
            let mean = ga[i] / 2.0 + gb[i] / 2.0;
            assert!(
                (meta_grad[i] - mean).abs() < 1e-4,
                "multi-task meta-grad mean mismatch at {i}: {} vs {}",
                meta_grad[i],
                mean
            );
        }
    }

    #[test]
    fn fomaml_deterministic_and_finite() {
        let mut rng = LcgRng::new(99);
        let params: Vec<f32> = (0..N_PARAMS).map(|_| rng.next_f32() - 0.5).collect();
        let (sx, sy) = task_a();
        let cfg = FoMamlConfig {
            inner_lr: 0.05,
            n_inner_steps: 2,
        };
        let tasks = vec![(sx, sy)];
        let u1 = fomaml_update(&params, &tasks, N_CLASSES, FEAT_DIM, 0.1, &cfg).expect("fomaml");
        let u2 = fomaml_update(&params, &tasks, N_CLASSES, FEAT_DIM, 0.1, &cfg).expect("fomaml");
        assert_eq!(u1, u2, "fomaml_update must be deterministic");
        assert!(u1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fomaml_rejects_bad_args() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = FoMamlConfig {
            inner_lr: 0.05,
            n_inner_steps: 1,
        };
        assert!(matches!(
            fomaml_update(&params, &[], N_CLASSES, FEAT_DIM, 0.1, &cfg),
            Err(MetaError::EmptySupport)
        ));
        assert!(matches!(
            fomaml_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, 0.0, &cfg),
            Err(MetaError::InvalidLr { .. })
        ));
    }
}
