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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const N_CLASSES: usize = 2;
    const FEAT_DIM: usize = 3;
    // params layout = W (N_CLASSES * FEAT_DIM) then b (N_CLASSES).
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

    // Reproduce the EXACT inner loop the implementation runs (same private loss,
    // same fd eps, same SGD step) so θ_adapted is byte-identical.
    fn replicate_adapt(params: &[f32], sx: &[f32], sy: &[u32], cfg: &ReptileConfig) -> Vec<f32> {
        let mut adapted = params.to_vec();
        for _ in 0..cfg.n_inner_steps {
            let f = |p: &[f32]| task_loss_flat(p, sx, sy, N_CLASSES, FEAT_DIM);
            let grad = fd_gradient(&adapted, &f, 1e-4);
            adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr).expect("sgd step");
        }
        adapted
    }

    // Reptile meta-update is EXACTLY θ + ε·(θ_adapted − θ): the result lies on
    // the segment from θ to θ_adapted at fraction ε.
    #[test]
    fn reptile_update_is_segment_at_fraction_eps() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 3,
            step_size: 0.4,
        };
        let adapted = replicate_adapt(&params, &sx, &sy, &cfg);
        let updated =
            reptile_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        assert_eq!(updated.len(), N_PARAMS);
        let mut moved = false;
        for i in 0..N_PARAMS {
            let expected = params[i] + cfg.step_size * (adapted[i] - params[i]);
            assert_eq!(updated[i], expected, "must equal θ + ε·(θ' − θ) at {i}");
            // strictly between θ and θ_adapted (same sign, smaller magnitude).
            let step = updated[i] - params[i];
            let full = adapted[i] - params[i];
            if full != 0.0 {
                moved = true;
                assert!(
                    step * full >= 0.0,
                    "step must point toward θ_adapted at {i}"
                );
                assert!(
                    step.abs() <= full.abs() + 1e-6,
                    "ε<1 must undershoot at {i}"
                );
            }
        }
        assert!(
            moved,
            "inner loop must actually move at least one parameter"
        );
    }

    // ε = 0 ⇒ no change (exact).
    #[test]
    fn reptile_eps_zero_no_change() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 3,
            step_size: 0.0,
        };
        let updated =
            reptile_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        assert_eq!(updated, params, "ε=0 must leave θ unchanged");
    }

    // ε = 1 ⇒ exactly θ_adapted.
    #[test]
    fn reptile_eps_one_equals_adapted() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 3,
            step_size: 1.0,
        };
        let adapted = replicate_adapt(&params, &sx, &sy, &cfg);
        let updated =
            reptile_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        for i in 0..N_PARAMS {
            let expected = params[i] + 1.0 * (adapted[i] - params[i]);
            assert_eq!(updated[i], expected);
            assert!(
                (updated[i] - adapted[i]).abs() < 1e-5,
                "ε=1 must reach θ_adapted at {i}: {} vs {}",
                updated[i],
                adapted[i]
            );
        }
    }

    // Multi-task: θ_adapted is the MEAN of per-task adapted weights, and the
    // update is θ + ε·(mean(θ') − θ).
    #[test]
    fn reptile_multi_task_averages_adapted() {
        let params = base_params();
        let (sx_a, sy_a) = task_a();
        let (sx_b, sy_b) = task_b();
        let cfg = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 2,
            step_size: 0.5,
        };
        let a = replicate_adapt(&params, &sx_a, &sy_a, &cfg);
        let b = replicate_adapt(&params, &sx_b, &sy_b, &cfg);
        let tasks = vec![(sx_a, sy_a), (sx_b, sy_b)];
        let updated = reptile_update(&params, &tasks, N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        for i in 0..N_PARAMS {
            let avg = a[i] / 2.0 + b[i] / 2.0;
            let expected = params[i] + cfg.step_size * (avg - params[i]);
            assert!(
                (updated[i] - expected).abs() < 1e-6,
                "multi-task mean mismatch at {i}: {} vs {}",
                updated[i],
                expected
            );
        }
    }

    // Sanity that Reptile moves toward an improved solution: the inner loop
    // reduces the support loss, so θ_adapted has lower task loss than θ.
    #[test]
    fn reptile_inner_loop_reduces_support_loss() {
        let params = base_params();
        let (sx, sy) = task_a();
        let cfg = ReptileConfig {
            inner_lr: 0.1,
            n_inner_steps: 5,
            step_size: 0.5,
        };
        let adapted = replicate_adapt(&params, &sx, &sy, &cfg);
        let loss0 = task_loss_flat(&params, &sx, &sy, N_CLASSES, FEAT_DIM);
        let loss1 = task_loss_flat(&adapted, &sx, &sy, N_CLASSES, FEAT_DIM);
        assert!(
            loss1 < loss0,
            "inner loop must reduce support loss: {loss1} >= {loss0}"
        );
    }

    #[test]
    fn reptile_deterministic_and_finite() {
        let mut rng = LcgRng::new(7);
        let params: Vec<f32> = (0..N_PARAMS).map(|_| rng.next_f32() - 0.5).collect();
        let (sx, sy) = task_a();
        let cfg = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 3,
            step_size: 0.3,
        };
        let tasks = vec![(sx, sy)];
        let u1 = reptile_update(&params, &tasks, N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        let u2 = reptile_update(&params, &tasks, N_CLASSES, FEAT_DIM, &cfg).expect("reptile");
        assert_eq!(u1, u2, "reptile_update must be deterministic");
        assert!(u1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reptile_rejects_bad_args() {
        let params = base_params();
        let (sx, sy) = task_a();
        let good = ReptileConfig {
            inner_lr: 0.05,
            n_inner_steps: 1,
            step_size: 0.5,
        };
        assert!(matches!(
            reptile_update(&params, &[], N_CLASSES, FEAT_DIM, &good),
            Err(MetaError::EmptySupport)
        ));
        let bad_lr = ReptileConfig {
            inner_lr: 0.0,
            n_inner_steps: 1,
            step_size: 0.5,
        };
        assert!(matches!(
            reptile_update(&params, &[(sx, sy)], N_CLASSES, FEAT_DIM, &bad_lr),
            Err(MetaError::InvalidLr { .. })
        ));
    }
}
