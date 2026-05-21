use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;

#[derive(Debug, Clone)]
pub struct MetaSgdConfig {
    pub inner_lr_init: f32,
    pub meta_lr: f32,
    pub inner_steps: usize,
    pub fd_eps: f32,
    pub clip_alpha: f32,
}

impl Default for MetaSgdConfig {
    fn default() -> Self {
        Self {
            inner_lr_init: 0.01,
            meta_lr: 0.001,
            inner_steps: 1,
            fd_eps: 1e-4,
            clip_alpha: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetaSgdState {
    pub params: Vec<f32>,
    pub alpha: Vec<f32>,
    pub n_params: usize,
}

impl MetaSgdState {
    pub fn new(n_params: usize, init_lr: f32) -> MetaResult<Self> {
        if n_params == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_params must be > 0".into(),
            });
        }
        Ok(Self {
            params: vec![0.0_f32; n_params],
            alpha: vec![init_lr; n_params],
            n_params,
        })
    }

    pub fn from_params(params: Vec<f32>, init_lr: f32) -> MetaResult<Self> {
        let n = params.len();
        if n == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_params must be > 0".into(),
            });
        }
        Ok(Self {
            alpha: vec![init_lr; n],
            n_params: n,
            params,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetaSgdResult {
    pub params: Vec<f32>,
    pub alpha: Vec<f32>,
    pub mean_task_loss: f32,
}

pub struct MetaSgd;

impl MetaSgd {
    /// Inner-loop adaptation for one task.
    /// Returns adapted parameters θ' after `inner_steps` steps.
    /// θ'_{k+1} = θ'_k - α ⊙ fd_gradient(task_loss, θ'_k)
    pub fn inner_adapt<F>(
        state: &MetaSgdState,
        task_loss: &F,
        cfg: &MetaSgdConfig,
    ) -> MetaResult<Vec<f32>>
    where
        F: Fn(&[f32]) -> f32,
    {
        if cfg.inner_steps == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "inner_steps must be > 0".into(),
            });
        }
        if cfg.fd_eps <= 0.0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "fd_eps must be > 0".into(),
            });
        }

        let n = state.n_params;
        let mut theta_prime = state.params.clone();

        for _ in 0..cfg.inner_steps {
            let grad = fd_gradient(&theta_prime, task_loss, cfg.fd_eps);
            for j in 0..n {
                theta_prime[j] -= state.alpha[j] * grad[j];
            }
        }

        Ok(theta_prime)
    }

    /// Meta-update over a batch of tasks.
    /// 1. For each task i: compute θ'_i = inner_adapt(θ, α, task_i)
    /// 2. Compute meta-gradient for θ: g_θ = (1/N) Σ_i fd_gradient(task_i, θ'_i)
    /// 3. Compute meta-gradient for α: `g_α_i[j] = -grad_theta_base[j] * grad_theta_adapted[j]`
    ///    g_α = (1/N) Σ_i g_α_i
    /// 4. θ -= meta_lr * g_θ
    /// 5. α -= meta_lr * g_α; then clamp α to [0, clip_alpha]
    /// 6. Return MetaSgdResult with updated θ, α, and mean task loss at adapted params.
    pub fn meta_update<F>(
        state: &mut MetaSgdState,
        tasks: &[F],
        cfg: &MetaSgdConfig,
    ) -> MetaResult<MetaSgdResult>
    where
        F: Fn(&[f32]) -> f32,
    {
        if tasks.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        if cfg.meta_lr <= 0.0 || !cfg.meta_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: cfg.meta_lr });
        }
        if cfg.inner_steps == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "inner_steps must be > 0".into(),
            });
        }
        if cfg.fd_eps <= 0.0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "fd_eps must be > 0".into(),
            });
        }

        let n = state.n_params;
        let n_tasks = tasks.len() as f32;

        let mut g_theta = vec![0.0_f32; n];
        let mut g_alpha = vec![0.0_f32; n];
        let mut total_loss = 0.0_f32;

        for task in tasks {
            // Compute adapted parameters θ'_i
            let theta_prime = Self::inner_adapt(state, task, cfg)?;

            // Meta-gradient for θ: fd_gradient at adapted params
            let grad_adapted = fd_gradient(&theta_prime, task, cfg.fd_eps);

            // Base gradient at θ for α meta-gradient computation
            let grad_base = fd_gradient(&state.params, task, cfg.fd_eps);

            // Accumulate meta-gradient for θ
            for j in 0..n {
                g_theta[j] += grad_adapted[j] / n_tasks;
            }

            // Accumulate meta-gradient for α: chain rule approximation
            // ∂L_i(θ'_i)/∂α_j = (∂L_i/∂θ'_{i,j}) * (-∂_{θ_j} L_i(θ))
            for j in 0..n {
                g_alpha[j] += (-grad_base[j] * grad_adapted[j]) / n_tasks;
            }

            // Accumulate task loss at adapted params
            total_loss += task(&theta_prime);
        }

        // Update θ
        for (p, &gt) in state.params.iter_mut().zip(g_theta.iter()) {
            *p -= cfg.meta_lr * gt;
        }

        // Update α and clamp to [0, clip_alpha]
        for (a, &ga) in state.alpha.iter_mut().zip(g_alpha.iter()) {
            *a -= cfg.meta_lr * ga;
            *a = a.clamp(0.0, cfg.clip_alpha);
        }

        let mean_task_loss = total_loss / n_tasks;

        Ok(MetaSgdResult {
            params: state.params.clone(),
            alpha: state.alpha.clone(),
            mean_task_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_task(center: f32) -> impl Fn(&[f32]) -> f32 {
        move |params: &[f32]| params.iter().map(|&p| (p - center) * (p - center)).sum()
    }

    #[test]
    fn state_new_valid() {
        let state = MetaSgdState::new(5, 0.01).unwrap();
        assert_eq!(state.params.len(), 5);
        assert_eq!(state.alpha.len(), 5);
        for &a in &state.alpha {
            assert!((a - 0.01_f32).abs() < 1e-6);
        }
        assert_eq!(state.n_params, 5);
    }

    #[test]
    fn state_from_params() {
        let p = vec![1.0_f32, 2.0, 3.0];
        let state = MetaSgdState::from_params(p.clone(), 0.01).unwrap();
        assert_eq!(state.params, p);
        assert_eq!(state.n_params, 3);
        assert_eq!(state.alpha.len(), 3);
    }

    #[test]
    fn inner_adapt_reduces_loss() {
        // Simple convex task: Σ(x_i - 1)^2 -> minimum at all ones
        let task = quadratic_task(1.0);
        let cfg = MetaSgdConfig::default();
        let state = MetaSgdState::new(4, 0.01).unwrap();
        let loss_before = task(&state.params);
        let adapted = MetaSgd::inner_adapt(&state, &task, &cfg).unwrap();
        let loss_after = task(&adapted);
        assert!(loss_after < loss_before);
    }

    #[test]
    fn inner_adapt_length_preserved() {
        let task = quadratic_task(2.0);
        let cfg = MetaSgdConfig::default();
        let state = MetaSgdState::new(7, 0.01).unwrap();
        let adapted = MetaSgd::inner_adapt(&state, &task, &cfg).unwrap();
        assert_eq!(adapted.len(), 7);
    }

    #[test]
    fn alpha_init() {
        let state = MetaSgdState::new(10, 0.05).unwrap();
        for &a in &state.alpha {
            assert!((a - 0.05_f32).abs() < 1e-7);
        }
    }

    #[test]
    fn meta_update_returns_result() {
        let task = quadratic_task(3.0);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn meta_update_result_length() {
        let task = quadratic_task(1.5);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(6, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
        assert_eq!(result.params.len(), 6);
    }

    #[test]
    fn meta_update_changes_params() {
        // Non-trivial task: quadratic away from zero
        let task = quadratic_task(5.0);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(3, 0.01).unwrap();
        let params_before = state.params.clone();
        MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
        // At least one parameter should have changed
        let changed = state
            .params
            .iter()
            .zip(params_before.iter())
            .any(|(&after, &before)| (after - before).abs() > 1e-9);
        assert!(changed);
    }

    #[test]
    fn alpha_clamped_nonneg() {
        let task = quadratic_task(2.0);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(5, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
        for &a in &result.alpha {
            assert!(a >= 0.0);
        }
    }

    #[test]
    fn alpha_clamped_upper() {
        let task = quadratic_task(2.0);
        let cfg = MetaSgdConfig {
            clip_alpha: 1.0,
            ..MetaSgdConfig::default()
        };
        let mut state = MetaSgdState::new(5, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
        for &a in &result.alpha {
            assert!(a <= 1.0);
        }
    }

    #[test]
    fn multi_step_inner() {
        let task = quadratic_task(1.0);
        let cfg = MetaSgdConfig {
            inner_steps: 3,
            ..MetaSgdConfig::default()
        };
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn single_task() {
        let task = quadratic_task(2.0);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(3, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn mean_task_loss_finite() {
        let task = quadratic_task(1.0);
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
        assert!(result.mean_task_loss.is_finite());
    }

    #[test]
    fn empty_tasks_err() {
        type TaskFn = Box<dyn Fn(&[f32]) -> f32>;
        let cfg = MetaSgdConfig::default();
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let tasks: Vec<TaskFn> = vec![];
        let result = MetaSgd::meta_update(&mut state, tasks.as_slice(), &cfg);
        assert!(matches!(result, Err(MetaError::EmptySupport)));
    }

    #[test]
    fn invalid_meta_lr_err() {
        let task = quadratic_task(1.0);
        let cfg = MetaSgdConfig {
            meta_lr: 0.0,
            ..MetaSgdConfig::default()
        };
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg);
        assert!(matches!(result, Err(MetaError::InvalidLr { .. })));
    }

    #[test]
    fn invalid_inner_steps_err() {
        let task = quadratic_task(1.0);
        let cfg = MetaSgdConfig {
            inner_steps: 0,
            ..MetaSgdConfig::default()
        };
        let mut state = MetaSgdState::new(4, 0.01).unwrap();
        let result = MetaSgd::meta_update(&mut state, &[task], &cfg);
        assert!(matches!(
            result,
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn quadratic_convergence() {
        // Repeated meta_update on Σ(x-1)^2 task should decrease loss over rounds
        let cfg = MetaSgdConfig {
            inner_lr_init: 0.05,
            meta_lr: 0.01,
            inner_steps: 2,
            fd_eps: 1e-4,
            clip_alpha: 1.0,
        };
        let mut state = MetaSgdState::new(4, 0.05).unwrap();

        let mut prev_loss = f32::MAX;
        let mut n_improvements = 0_usize;

        for _ in 0..30 {
            let task = quadratic_task(1.0);
            let result = MetaSgd::meta_update(&mut state, &[task], &cfg).unwrap();
            let loss = result.mean_task_loss;
            if loss < prev_loss {
                n_improvements += 1;
            }
            prev_loss = loss;
        }
        // Loss should decrease for the majority of rounds
        assert!(n_improvements >= 15);
    }
}
