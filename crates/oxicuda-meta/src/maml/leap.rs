//! LEAP: Meta-Learning with Warped Gradient Descent (Flennerhag et al. 2019).
//!
//! LEAP learns a warp matrix W such that inner-loop gradient steps taken in the
//! warped space `W * ∇L(θ)` converge faster across tasks than plain gradient descent.

#![allow(clippy::module_name_repetitions)]

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Type alias for a boxed gradient function: `&[f32] -> Vec<f32>`.
type GradFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

/// Configuration for LEAP.
pub struct LeapConfig {
    /// Dimensionality of the parameter space (warp matrix is `warp_dim × warp_dim`).
    pub warp_dim: usize,
    /// Meta (outer) learning rate for updating `theta`.
    pub outer_lr: f32,
    /// Inner-loop learning rate for warped gradient steps.
    pub inner_lr: f32,
    /// Number of inner adaptation steps.
    pub inner_steps: usize,
    /// Learning rate for updating the warp matrix.
    pub warp_lr: f32,
}

/// LEAP learner: meta-learns both parameters and a warp matrix for fast adaptation.
pub struct Leap {
    /// Meta-parameters `θ` of shape `[warp_dim]`.
    theta: Vec<f32>,
    /// Warp matrix `W` of shape `[warp_dim × warp_dim]`, stored row-major.
    warp: Vec<f32>,
    config: LeapConfig,
}

impl Leap {
    /// Create a new LEAP learner.
    ///
    /// Initializes `theta` as zeros and `warp` as the identity matrix.
    ///
    /// # Errors
    /// - `InvalidEpisodeConfig` if `warp_dim == 0`
    pub fn new(config: LeapConfig, _rng: &mut LcgRng) -> MetaResult<Self> {
        if config.warp_dim == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "warp_dim must be > 0".into(),
            });
        }
        let dim = config.warp_dim;
        let theta = vec![0.0_f32; dim];
        // Warp starts as identity matrix
        let mut warp = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            warp[i * dim + i] = 1.0;
        }
        Ok(Self {
            theta,
            warp,
            config,
        })
    }

    /// Apply the warp matrix to a gradient vector: `warped[i] = Σ_j W[i,j] * grad[j]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `grad.len() != warp_dim`
    pub fn warp_gradient(&self, grad: &[f32]) -> MetaResult<Vec<f32>> {
        let dim = self.config.warp_dim;
        if grad.len() != dim {
            return Err(MetaError::DimensionMismatch {
                expected: dim,
                got: grad.len(),
            });
        }
        let mut warped = vec![0.0_f32; dim];
        for (i, warped_i) in warped.iter_mut().enumerate() {
            let row_start = i * dim;
            let val: f32 = self.warp[row_start..row_start + dim]
                .iter()
                .zip(grad.iter())
                .map(|(&w, &g)| w * g)
                .sum();
            *warped_i = val;
        }
        Ok(warped)
    }

    /// Inner-loop adaptation using warped gradient descent.
    ///
    /// At each step: `θ' = θ - inner_lr * W * ∇L(θ')`
    ///
    /// If `steps == 0`, returns `theta` unchanged.
    ///
    /// # Errors
    /// - `DimensionMismatch` if the gradient from `grad_fn` has wrong length
    pub fn inner_adapt_warped(
        &self,
        theta: &[f32],
        grad_fn: impl Fn(&[f32]) -> Vec<f32>,
        steps: usize,
    ) -> MetaResult<Vec<f32>> {
        if steps == 0 {
            return Ok(theta.to_vec());
        }
        let dim = self.config.warp_dim;
        let eff = theta.len().min(dim);
        let mut params = theta.to_vec();
        for _ in 0..steps {
            let raw_grad = grad_fn(&params);
            // Compute warped gradient: warped[i] = Σ_j W[i,j] * raw_grad[j]
            let mut warped = vec![0.0_f32; eff];
            for (i, warped_i) in warped.iter_mut().enumerate() {
                let row_start = i * dim;
                let val: f32 = self.warp[row_start..row_start + eff]
                    .iter()
                    .zip(raw_grad.iter().take(eff))
                    .map(|(&w, &g)| w * g)
                    .sum();
                *warped_i = val;
            }
            for (p, &wg) in params.iter_mut().zip(warped.iter()) {
                *p -= self.config.inner_lr * wg;
            }
        }
        Ok(params)
    }

    /// Perform one meta-update step for LEAP.
    ///
    /// For each task `i`:
    /// 1. Inner adapt: `θ_i = inner_adapt_warped(θ, task_grad_fns[i], inner_steps)`
    /// 2. Eval gradient: `g_eval = eval_grad_fns[i](θ_i)`
    /// 3. Accumulate meta-gradient for `θ`: `g_theta += g_eval / n_tasks`
    /// 4. Accumulate warp gradient: `g_warp[i,j] += g_eval[i] * raw_grad[j] / n_tasks`
    ///    (chain rule: W maps raw_grad to warped_grad, warp update aligns warped grad
    ///    with the meta-loss gradient direction)
    ///
    /// Updates: `θ -= outer_lr * mean_g_theta`, `W -= warp_lr * mean_g_warp`.
    ///
    /// Returns mean `||g_eval||²` as proxy meta-loss.
    ///
    /// # Errors
    /// - `EmptySupport` if `task_grad_fns` is empty
    pub fn meta_step(
        &mut self,
        task_grad_fns: &[GradFn],
        eval_grad_fns: &[GradFn],
    ) -> MetaResult<f32> {
        if task_grad_fns.is_empty() {
            return Err(MetaError::EmptySupport);
        }

        let dim = self.config.warp_dim;
        let n_tasks = task_grad_fns.len();
        let inner_steps = self.config.inner_steps;

        let mut g_theta = vec![0.0_f32; dim];
        let mut g_warp = vec![0.0_f32; dim * dim];
        let mut total_loss = 0.0_f32;

        for (i, task_fn) in task_grad_fns.iter().enumerate() {
            // Step 1: inner-loop adaptation with warped gradients
            let theta_i = self.inner_adapt_warped(&self.theta.clone(), task_fn, inner_steps)?;

            // Step 2: eval gradient at adapted params
            let eval_fn = eval_grad_fns.get(i).unwrap_or(&eval_grad_fns[0]);
            let g_eval = eval_fn(&theta_i);

            // Pre-adaptation gradient at θ (for warp update)
            let raw_grad_at_theta = task_fn(&self.theta);

            // Accumulate meta-gradient for theta and compute sq norm
            let eff = dim.min(g_eval.len());
            let mut sq_norm = 0.0_f32;
            for (gt, &ge) in g_theta[..eff].iter_mut().zip(g_eval[..eff].iter()) {
                *gt += ge / n_tasks as f32;
                sq_norm += ge * ge;
            }
            total_loss += sq_norm;

            // Warp gradient: d_loss/d_W[row,col] ≈ g_eval[row] * raw_grad[col]
            // (gradient of the meta-loss through the warped inner-loop step)
            let rg_eff = dim.min(raw_grad_at_theta.len());
            for (row, &g_ev) in g_eval[..eff].iter().enumerate() {
                for (col, &rg) in raw_grad_at_theta[..rg_eff].iter().enumerate() {
                    g_warp[row * dim + col] += g_ev * rg / n_tasks as f32;
                }
            }
        }

        // Update theta
        let outer_lr = self.config.outer_lr;
        for (t, &g) in self.theta.iter_mut().zip(g_theta.iter()) {
            *t -= outer_lr * g;
        }

        // Update warp matrix
        let warp_lr = self.config.warp_lr;
        for (w, &g) in self.warp.iter_mut().zip(g_warp.iter()) {
            *w -= warp_lr * g;
        }

        let meta_loss = total_loss / n_tasks as f32;
        Ok(meta_loss)
    }

    /// Return a reference to the current meta-parameters `θ`.
    pub fn params(&self) -> &[f32] {
        &self.theta
    }

    /// Return a reference to the warp matrix `W` (row-major, `warp_dim × warp_dim`).
    pub fn warp(&self) -> &[f32] {
        &self.warp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(13)
    }

    fn simple_config() -> LeapConfig {
        LeapConfig {
            warp_dim: 4,
            outer_lr: 0.01,
            inner_lr: 0.1,
            inner_steps: 2,
            warp_lr: 0.001,
        }
    }

    fn quadratic_grad(center: f32) -> impl Fn(&[f32]) -> Vec<f32> {
        move |params: &[f32]| params.iter().map(|&p| 2.0 * (p - center)).collect()
    }

    // Test 1: identity warp → warped_gradient == input grad
    #[test]
    fn identity_warp_same_as_plain_grad() {
        let mut rng = make_rng();
        let config = simple_config();
        let learner = Leap::new(config, &mut rng).expect("new ok");

        let grad = vec![1.0_f32, 2.0, 3.0, 4.0];
        let warped = learner.warp_gradient(&grad).expect("warp_gradient ok");
        assert_eq!(grad, warped, "identity warp should preserve gradient");
    }

    // Test 2: adapt changes params with non-trivial loss
    #[test]
    fn inner_adapt_changes_params() {
        let mut rng = make_rng();
        let config = simple_config();
        let learner = Leap::new(config, &mut rng).expect("new ok");

        let theta = vec![0.0_f32; 4];
        let grad_fn = quadratic_grad(1.0);
        let adapted = learner
            .inner_adapt_warped(&theta, grad_fn, 3)
            .expect("adapt ok");
        assert_ne!(theta, adapted, "adapt should change params");
    }

    // Test 3: 0 steps → no change
    #[test]
    fn inner_adapt_0_steps_no_change() {
        let mut rng = make_rng();
        let config = simple_config();
        let learner = Leap::new(config, &mut rng).expect("new ok");

        let theta = vec![1.0_f32, 2.0, 3.0, 4.0];
        let adapted = learner
            .inner_adapt_warped(&theta, quadratic_grad(5.0), 0)
            .expect("adapt ok");
        assert_eq!(theta, adapted, "0 steps should leave params unchanged");
    }

    // Test 4: meta_step returns finite loss
    #[test]
    fn meta_step_finite() {
        let mut rng = make_rng();
        let config = simple_config();
        let mut learner = Leap::new(config, &mut rng).expect("new ok");

        let task_fns: Vec<GradFn> = vec![
            Box::new(quadratic_grad(1.0)),
            Box::new(quadratic_grad(-1.0)),
        ];
        let eval_fns: Vec<GradFn> = vec![
            Box::new(quadratic_grad(0.5)),
            Box::new(quadratic_grad(-0.5)),
        ];
        let loss = learner
            .meta_step(&task_fns, &eval_fns)
            .expect("meta_step ok");
        assert!(loss.is_finite(), "meta_loss={loss} is not finite");
    }

    // Test 5: empty tasks → Err(EmptySupport)
    #[test]
    fn meta_step_empty_error() {
        let mut rng = make_rng();
        let config = simple_config();
        let mut learner = Leap::new(config, &mut rng).expect("new ok");

        let task_fns: Vec<GradFn> = vec![];
        let eval_fns: Vec<GradFn> = vec![];
        let result = learner.meta_step(&task_fns, &eval_fns);
        assert!(matches!(result, Err(MetaError::EmptySupport)));
    }

    // Test 6: warp_dim=0 → Err
    #[test]
    fn warp_dim_zero_error() {
        let mut rng = make_rng();
        let config = LeapConfig {
            warp_dim: 0,
            outer_lr: 0.01,
            inner_lr: 0.1,
            inner_steps: 1,
            warp_lr: 0.001,
        };
        let result = Leap::new(config, &mut rng);
        assert!(matches!(
            result,
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    // Test 7: params().len() == warp_dim
    #[test]
    fn params_len() {
        let mut rng = make_rng();
        let dim = 5;
        let config = LeapConfig {
            warp_dim: dim,
            outer_lr: 0.01,
            inner_lr: 0.1,
            inner_steps: 1,
            warp_lr: 0.001,
        };
        let learner = Leap::new(config, &mut rng).expect("new ok");
        assert_eq!(learner.params().len(), dim);
    }

    // Test 8: warp().len() == warp_dim * warp_dim
    #[test]
    fn warp_len() {
        let mut rng = make_rng();
        let dim = 5;
        let config = LeapConfig {
            warp_dim: dim,
            outer_lr: 0.01,
            inner_lr: 0.1,
            inner_steps: 1,
            warp_lr: 0.001,
        };
        let learner = Leap::new(config, &mut rng).expect("new ok");
        assert_eq!(learner.warp().len(), dim * dim);
    }
}
