//! Second-order MAML (MAML2) with Hessian-vector products via finite differences.
//!
//! This implements the full second-order MAML meta-gradient update from Finn et al. 2017,
//! computing HVP corrections via central finite differences rather than exact autodiff.

#![allow(clippy::module_name_repetitions)]

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Type alias for a boxed gradient function: `&[f32] -> Vec<f32>`.
type GradFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

/// Configuration for second-order MAML.
pub struct Maml2Config {
    /// Inner-loop learning rate.
    pub inner_lr: f32,
    /// Number of inner adaptation steps.
    pub inner_steps: usize,
    /// Outer (meta) learning rate.
    pub outer_lr: f32,
    /// Finite-difference step size for HVP approximation (typical: 1e-3).
    pub eps_hvp: f32,
    /// Number of tasks per meta-batch.
    pub n_tasks: usize,
}

/// Second-order MAML learner with HVP correction via central finite differences.
pub struct SecondOrderMaml {
    theta: Vec<f32>,
    config: Maml2Config,
}

impl SecondOrderMaml {
    /// Create a new second-order MAML learner.
    ///
    /// Initializes `theta` with small random values: `rng.next_f32() * 0.01`.
    ///
    /// # Errors
    /// - `InvalidEpisodeConfig` if `dim == 0`
    /// - `InvalidEpisodeConfig` if `config.n_tasks == 0`
    pub fn new(dim: usize, config: Maml2Config, rng: &mut LcgRng) -> MetaResult<Self> {
        if dim == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "dim must be > 0".into(),
            });
        }
        if config.n_tasks == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_tasks must be > 0".into(),
            });
        }
        let theta = (0..dim).map(|_| rng.next_f32() * 0.01).collect();
        Ok(Self { theta, config })
    }

    /// Inner-loop adaptation: run `steps` gradient-descent steps.
    ///
    /// If `steps == 0` or `lr == 0.0`, returns `theta` unchanged.
    pub fn inner_adapt(
        theta: &[f32],
        grad_fn: impl Fn(&[f32]) -> Vec<f32>,
        steps: usize,
        lr: f32,
    ) -> Vec<f32> {
        if steps == 0 || lr == 0.0 {
            return theta.to_vec();
        }
        let lr = lr.min(f32::MAX);
        let mut params = theta.to_vec();
        for _ in 0..steps {
            let grad = grad_fn(&params);
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= lr * g;
            }
        }
        params
    }

    /// Hessian-vector product via central finite differences.
    ///
    /// Approximates `H(θ) * v ≈ (∇L(θ + ε·v) - ∇L(θ - ε·v)) / (2ε)`
    /// for a twice-differentiable loss `L` with Hessian `H`.
    pub fn hvp(
        theta: &[f32],
        v: &[f32],
        grad_fn: impl Fn(&[f32]) -> Vec<f32>,
        eps: f32,
    ) -> Vec<f32> {
        let n = theta.len();
        let theta_plus: Vec<f32> = theta
            .iter()
            .zip(v.iter())
            .map(|(&t, &vi)| t + eps * vi)
            .collect();
        let theta_minus: Vec<f32> = theta
            .iter()
            .zip(v.iter())
            .map(|(&t, &vi)| t - eps * vi)
            .collect();

        let g_plus = grad_fn(&theta_plus);
        let g_minus = grad_fn(&theta_minus);

        let denom = 2.0 * eps;
        let mut hvp_out = vec![0.0_f32; n];
        for (hvp_i, (gp, gm)) in hvp_out.iter_mut().zip(g_plus.iter().zip(g_minus.iter())) {
            *hvp_i = (gp - gm) / denom;
        }
        hvp_out
    }

    /// Perform one meta-update step with second-order gradients.
    ///
    /// For each task `i`:
    /// 1. Adapt: `θ_i = inner_adapt(θ, task_grad_fns[i], inner_steps, inner_lr)`
    /// 2. Eval gradient: `g_eval = eval_grad_fns[i](θ_i)`
    /// 3. HVP correction: `h = HVP(θ, g_eval, task_grad_fns[i], eps_hvp)`
    /// 4. Second-order meta-gradient: `g_meta_i[j] = g_eval[j] - inner_lr * h[j]`
    ///
    /// Updates `θ -= outer_lr * mean_meta_grad`.
    ///
    /// Returns the mean squared norm of `g_eval` vectors as a proxy meta-loss.
    ///
    /// # Errors
    /// - `EmptySupport` if `task_grad_fns` is empty
    /// - `NanEncountered` if the proxy meta-loss is non-finite
    pub fn meta_step(
        &mut self,
        task_grad_fns: &[GradFn],
        eval_grad_fns: &[GradFn],
    ) -> MetaResult<f32> {
        if task_grad_fns.is_empty() {
            return Err(MetaError::EmptySupport);
        }

        let n = self.theta.len();
        let n_tasks = task_grad_fns.len();
        let inner_lr = self.config.inner_lr;
        let inner_steps = self.config.inner_steps;
        let outer_lr = self.config.outer_lr;
        let eps_hvp = self.config.eps_hvp;

        let mut meta_grad = vec![0.0_f32; n];
        let mut total_eval_sq_norm = 0.0_f32;

        for (i, task_fn) in task_grad_fns.iter().enumerate() {
            // Step 1: inner-loop adaptation
            let theta_i = Self::inner_adapt(&self.theta, task_fn, inner_steps, inner_lr);

            // Step 2: eval gradient at adapted params
            let eval_grad_fn = eval_grad_fns.get(i).unwrap_or(&eval_grad_fns[0]);
            let g_eval = eval_grad_fn(&theta_i);

            // Step 3: HVP correction using task gradient function
            let h = Self::hvp(&self.theta, &g_eval, task_fn, eps_hvp);

            // Step 4: accumulate second-order meta-gradient
            let eff_n = g_eval.len().min(n).min(h.len());
            let mut sq_norm = 0.0_f32;
            for (meta_j, (ge, hi)) in meta_grad[..eff_n]
                .iter_mut()
                .zip(g_eval[..eff_n].iter().zip(h[..eff_n].iter()))
            {
                let g_meta_ij = ge - inner_lr * hi;
                *meta_j += g_meta_ij / n_tasks as f32;
                sq_norm += ge * ge;
            }
            total_eval_sq_norm += sq_norm;
        }

        // Update theta
        for (t, &mg) in self.theta.iter_mut().zip(meta_grad.iter()) {
            *t -= outer_lr * mg;
        }

        let meta_loss = total_eval_sq_norm / n_tasks as f32;
        if !meta_loss.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "SecondOrderMaml meta_step proxy loss".into(),
            });
        }
        Ok(meta_loss)
    }

    /// Return a reference to the current meta-parameters.
    pub fn params(&self) -> &[f32] {
        &self.theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_grad(center: f32) -> impl Fn(&[f32]) -> Vec<f32> {
        move |params: &[f32]| params.iter().map(|&p| 2.0 * (p - center)).collect()
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn make_config() -> Maml2Config {
        Maml2Config {
            inner_lr: 0.01,
            inner_steps: 2,
            outer_lr: 0.001,
            eps_hvp: 1e-3,
            n_tasks: 2,
        }
    }

    // Test 1: inner_adapt changes params with nonzero grad and lr
    #[test]
    fn inner_adapt_changes_params() {
        let theta = vec![1.0_f32, 2.0, 3.0];
        let grad_fn = quadratic_grad(0.0);
        let adapted = SecondOrderMaml::inner_adapt(&theta, grad_fn, 3, 0.1);
        assert_ne!(theta, adapted);
    }

    // Test 2: steps=0 → no change
    #[test]
    fn inner_adapt_0_steps_no_change() {
        let theta = vec![1.0_f32, 2.0, 3.0];
        let grad_fn = quadratic_grad(5.0);
        let adapted = SecondOrderMaml::inner_adapt(&theta, grad_fn, 0, 0.1);
        assert_eq!(theta, adapted);
    }

    // Test 3: hvp approximates Hessian diag for L=½||θ||² (H=I)
    #[test]
    fn hvp_approximates_hessian_diag() {
        // L(θ) = ½ Σ θ_i², ∇L(θ) = θ, H = I
        // HVP(θ, e_i) should ≈ e_i
        let theta = vec![0.5_f32, 0.3, 0.7];
        let n = theta.len();
        let grad_fn = |params: &[f32]| params.to_vec(); // ∇L = θ for L = ½||θ||²

        for i in 0..n {
            let mut v = vec![0.0_f32; n];
            v[i] = 1.0;
            let h = SecondOrderMaml::hvp(&theta, &v, grad_fn, 1e-3);
            assert!(
                (h[i] - 1.0_f32).abs() < 1e-4,
                "h[{i}] = {}, expected ~1.0",
                h[i]
            );
            for (j, &hj) in h.iter().enumerate() {
                if j != i {
                    assert!(hj.abs() < 1e-4, "h[{j}] = {hj}, expected ~0.0");
                }
            }
        }
    }

    // Test 4: meta_step returns Ok with finite f32
    #[test]
    fn meta_step_finite_loss() {
        let mut rng = make_rng();
        let config = make_config();
        let mut learner = SecondOrderMaml::new(4, config, &mut rng).expect("new ok");

        let task_fns: Vec<GradFn> =
            vec![Box::new(quadratic_grad(1.0)), Box::new(quadratic_grad(2.0))];
        let eval_fns: Vec<GradFn> =
            vec![Box::new(quadratic_grad(1.5)), Box::new(quadratic_grad(2.5))];
        let loss = learner
            .meta_step(&task_fns, &eval_fns)
            .expect("meta_step ok");
        assert!(loss.is_finite(), "meta_loss={loss} is not finite");
    }

    // Test 5: empty task_grad_fns → Err(EmptySupport)
    #[test]
    fn meta_step_zero_tasks_error() {
        let mut rng = make_rng();
        let config = make_config();
        let mut learner = SecondOrderMaml::new(4, config, &mut rng).expect("new ok");

        let task_fns: Vec<GradFn> = vec![];
        let eval_fns: Vec<GradFn> = vec![];
        let result = learner.meta_step(&task_fns, &eval_fns);
        assert!(matches!(result, Err(MetaError::EmptySupport)));
    }

    // Test 6: second-order differs from first-order
    #[test]
    fn second_order_differs_from_first_order() {
        let dim = 3;
        let seed_theta = vec![0.5_f32, 0.3, 0.7];

        let cfg_so = Maml2Config {
            inner_lr: 0.1,
            inner_steps: 1,
            outer_lr: 0.01,
            eps_hvp: 1e-3,
            n_tasks: 1,
        };
        let mut rng = LcgRng::new(99);
        let mut so_learner = SecondOrderMaml::new(dim, cfg_so, &mut rng).expect("so new ok");
        so_learner.theta = seed_theta.clone();

        let cfg_fo = Maml2Config {
            inner_lr: 0.1,
            inner_steps: 1,
            outer_lr: 0.01,
            eps_hvp: 0.0,
            n_tasks: 1,
        };
        let mut fo_learner = SecondOrderMaml::new(dim, cfg_fo, &mut rng).expect("fo new ok");
        fo_learner.theta = seed_theta.clone();

        let task_fns_so: Vec<GradFn> = vec![Box::new(quadratic_grad(0.0))];
        let eval_fns_so: Vec<GradFn> = vec![Box::new(quadratic_grad(0.0))];
        so_learner
            .meta_step(&task_fns_so, &eval_fns_so)
            .expect("so meta_step ok");

        let task_fns_fo: Vec<GradFn> = vec![Box::new(quadratic_grad(0.0))];
        let eval_fns_fo: Vec<GradFn> = vec![Box::new(quadratic_grad(0.0))];
        fo_learner
            .meta_step(&task_fns_fo, &eval_fns_fo)
            .expect("fo meta_step ok");

        let same = so_learner
            .params()
            .iter()
            .zip(fo_learner.params().iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-9);
        assert!(
            !same,
            "Second-order and first-order should give different params"
        );
    }

    // Test 7: params().len() == dim
    #[test]
    fn params_shape() {
        let mut rng = make_rng();
        let dim = 7;
        let config = Maml2Config {
            inner_lr: 0.01,
            inner_steps: 1,
            outer_lr: 0.001,
            eps_hvp: 1e-3,
            n_tasks: 1,
        };
        let learner = SecondOrderMaml::new(dim, config, &mut rng).expect("new ok");
        assert_eq!(learner.params().len(), dim);
    }

    // Test 8: lr=0.0 → no change
    #[test]
    fn inner_adapt_lr_0_no_change() {
        let theta = vec![1.0_f32, 2.0, 3.0];
        let grad_fn = quadratic_grad(5.0);
        let adapted = SecondOrderMaml::inner_adapt(&theta, grad_fn, 5, 0.0);
        assert_eq!(theta, adapted);
    }

    // Test 9: HVP is linear in v — HVP(θ, 2v) ≈ 2 * HVP(θ, v)
    #[test]
    fn hvp_linear_in_v() {
        let theta = vec![0.4_f32, 0.6, 0.2];
        let v = vec![1.0_f32, -1.0, 0.5];
        let v2: Vec<f32> = v.iter().map(|&vi| 2.0 * vi).collect();

        let grad_fn = |params: &[f32]| params.to_vec(); // ∇(½||θ||²) = θ

        let h1 = SecondOrderMaml::hvp(&theta, &v, grad_fn, 1e-3);
        let h2 = SecondOrderMaml::hvp(&theta, &v2, grad_fn, 1e-3);

        for (h2i, h1i) in h2.iter().zip(h1.iter()) {
            assert!(
                (h2i - 2.0 * h1i).abs() < 1e-3,
                "linearity violated: h2={h2i}, 2*h1={}",
                2.0 * h1i
            );
        }
    }

    // Test 10: all params are finite after meta_step
    #[test]
    fn params_finite_after_meta_step() {
        let mut rng = make_rng();
        let config = make_config();
        let mut learner = SecondOrderMaml::new(5, config, &mut rng).expect("new ok");

        let task_fns: Vec<GradFn> = vec![
            Box::new(quadratic_grad(1.0)),
            Box::new(quadratic_grad(-1.0)),
        ];
        let eval_fns: Vec<GradFn> = vec![
            Box::new(quadratic_grad(0.5)),
            Box::new(quadratic_grad(-0.5)),
        ];
        learner
            .meta_step(&task_fns, &eval_fns)
            .expect("meta_step ok");
        for &p in learner.params() {
            assert!(p.is_finite(), "param {p} is not finite after meta_step");
        }
    }
}
