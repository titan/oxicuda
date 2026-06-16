//! iMAML — Implicit MAML (Rajeswaran, Finn, Kakade & Levine, NeurIPS 2019).
//!
//! # Background
//!
//! Standard MAML backpropagates *through* the inner-loop optimisation, so the
//! meta-gradient memory and compute grow with the number of inner steps.
//! **Implicit MAML** ("Meta-Learning with Implicit Gradients") instead defines
//! the adapted parameters as the minimiser of a *proximally-regularised* inner
//! objective and recovers the meta-gradient analytically via the implicit
//! function theorem — independent of the optimisation path.
//!
//! The inner problem for a task is
//!
//! ```text
//! θ*(θ_meta) = argmin_φ  L_train(φ) + (λ/2) ‖φ − θ_meta‖²
//! ```
//!
//! At the optimum `∇L_train(θ*) + λ(θ* − θ_meta) = 0`.  Differentiating w.r.t.
//! `θ_meta` and applying the implicit function theorem gives the meta-gradient
//!
//! ```text
//! d L_val / d θ_meta = (I + (1/λ) ∇²L_train(θ*))⁻¹ ∇L_val(θ*)
//! ```
//!
//! The matrix `I + (1/λ) H` is never formed explicitly: the linear system
//!
//! ```text
//! (I + (1/λ) H) g = ∇L_val(θ*)
//! ```
//!
//! is solved with **conjugate gradient** (CG), which only needs Hessian-vector
//! products `H v`.  Those products are computed by central finite differences of
//! the gradient, `H v ≈ (∇L(θ*+εv) − ∇L(θ*−εv)) / 2ε`, so the caller only ever
//! supplies *gradient* closures — no second-order autodiff required.
//!
//! This module operates on flat `Vec<f32>` parameter buffers and gradient
//! closures `&[f32] → Vec<f32>`, matching the rest of `oxicuda-meta`'s MAML
//! family (see [`crate::maml::second_order`]).

use crate::error::{MetaError, MetaResult};

/// Boxed gradient closure: maps a parameter vector to its gradient.
pub type GradFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyper-parameters for iMAML.
#[derive(Debug, Clone)]
pub struct ImamlConfig {
    /// Proximal regularisation strength `λ > 0`.  Larger `λ` keeps the adapted
    /// parameters closer to the meta-initialisation and better-conditions the
    /// implicit linear system.
    pub lambda: f32,
    /// Inner-loop learning rate for solving the proximal problem.
    pub inner_lr: f32,
    /// Number of inner gradient-descent steps used to approximate `θ*`.
    pub inner_steps: usize,
    /// Maximum conjugate-gradient iterations for the implicit solve.
    pub cg_iters: usize,
    /// CG residual tolerance (early stop when `‖r‖² < tol`).
    pub cg_tol: f32,
    /// Finite-difference step `ε` for Hessian-vector products.
    pub hvp_eps: f32,
    /// Outer (meta) learning rate.
    pub outer_lr: f32,
}

impl Default for ImamlConfig {
    fn default() -> Self {
        Self {
            lambda: 2.0,
            inner_lr: 0.1,
            inner_steps: 16,
            cg_iters: 8,
            cg_tol: 1e-8,
            hvp_eps: 1e-3,
            outer_lr: 0.1,
        }
    }
}

impl ImamlConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`MetaError::InvalidLr`] — if `lambda`, `inner_lr`, or `outer_lr` is
    ///   non-positive / non-finite.
    /// * [`MetaError::InvalidEpisodeConfig`] — if `cg_iters == 0`.
    pub fn validate(&self) -> MetaResult<()> {
        for &(name, v) in &[
            ("lambda", self.lambda),
            ("inner_lr", self.inner_lr),
            ("outer_lr", self.outer_lr),
        ] {
            if v <= 0.0 || !v.is_finite() {
                let _ = name;
                return Err(MetaError::InvalidLr { lr: v });
            }
        }
        if self.cg_iters == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "cg_iters must be > 0".into(),
            });
        }
        Ok(())
    }
}

// ─── Core building blocks ────────────────────────────────────────────────────

/// Solve the proximal inner problem
/// `argmin_φ L_train(φ) + (λ/2)‖φ − θ_meta‖²` by gradient descent.
///
/// Returns the adapted parameters `θ*`.
///
/// # Errors
///
/// [`MetaError::InvalidLr`] if `inner_lr` is non-positive/non-finite;
/// [`MetaError::NanEncountered`] if a non-finite value appears.
pub fn proximal_inner_solve(
    theta_meta: &[f32],
    train_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    config: &ImamlConfig,
) -> MetaResult<Vec<f32>> {
    if config.inner_lr <= 0.0 || !config.inner_lr.is_finite() {
        return Err(MetaError::InvalidLr {
            lr: config.inner_lr,
        });
    }
    let n = theta_meta.len();
    let mut phi = theta_meta.to_vec();
    for _ in 0..config.inner_steps {
        let g = train_grad(&phi);
        if g.len() != n {
            return Err(MetaError::DimensionMismatch {
                expected: n,
                got: g.len(),
            });
        }
        // Proximal gradient: ∇L + λ(φ − θ_meta).
        for i in 0..n {
            let prox = config.lambda * (phi[i] - theta_meta[i]);
            phi[i] -= config.inner_lr * (g[i] + prox);
        }
    }
    if phi.iter().any(|v| !v.is_finite()) {
        return Err(MetaError::NanEncountered {
            context: "proximal_inner_solve produced non-finite parameters".into(),
        });
    }
    Ok(phi)
}

/// Hessian-vector product `H v ≈ (∇L(θ+εv) − ∇L(θ−εv)) / (2ε)`.
///
/// # Errors
///
/// [`MetaError::DimensionMismatch`] if `theta.len() ≠ v.len()` or a gradient
/// has the wrong length.
pub fn hessian_vector_product(
    theta: &[f32],
    v: &[f32],
    grad: &dyn Fn(&[f32]) -> Vec<f32>,
    eps: f32,
) -> MetaResult<Vec<f32>> {
    let n = theta.len();
    if v.len() != n {
        return Err(MetaError::DimensionMismatch {
            expected: n,
            got: v.len(),
        });
    }
    let plus: Vec<f32> = theta.iter().zip(v).map(|(&t, &vi)| t + eps * vi).collect();
    let minus: Vec<f32> = theta.iter().zip(v).map(|(&t, &vi)| t - eps * vi).collect();
    let gp = grad(&plus);
    let gm = grad(&minus);
    if gp.len() != n || gm.len() != n {
        return Err(MetaError::DimensionMismatch {
            expected: n,
            got: gp.len().min(gm.len()),
        });
    }
    let denom = 2.0 * eps;
    Ok(gp
        .iter()
        .zip(gm.iter())
        .map(|(&a, &b)| (a - b) / denom)
        .collect())
}

/// Solve `(I + (1/λ) H) g = b` for `g` via conjugate gradient, where `H` is the
/// Hessian of `L_train` at `theta_star` (accessed only through HVPs).
///
/// `b` is typically `∇L_val(θ*)`.  Returns the implicit meta-gradient direction.
///
/// # Errors
///
/// Propagates HVP dimension errors; [`MetaError::NanEncountered`] on divergence.
pub fn conjugate_gradient_implicit(
    theta_star: &[f32],
    b: &[f32],
    train_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    config: &ImamlConfig,
) -> MetaResult<Vec<f32>> {
    let n = theta_star.len();
    if b.len() != n {
        return Err(MetaError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let inv_lambda = 1.0 / config.lambda;

    // Operator A v = v + (1/λ) H v.
    let apply_a = |v: &[f32]| -> MetaResult<Vec<f32>> {
        let hv = hessian_vector_product(theta_star, v, train_grad, config.hvp_eps)?;
        Ok(v.iter()
            .zip(hv.iter())
            .map(|(&vi, &hvi)| vi + inv_lambda * hvi)
            .collect())
    };

    // CG from x0 = 0 → r0 = b, p0 = b.
    let mut x = vec![0.0_f32; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    for _ in 0..config.cg_iters {
        if rs_old < config.cg_tol {
            break;
        }
        let ap = apply_a(&p)?;
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-20 {
            // Curvature breakdown: stop with the current iterate.
            break;
        }
        let alpha = rs_old / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if !rs_new.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "conjugate_gradient_implicit diverged".into(),
            });
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Ok(x)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ─── Per-task meta-gradient ──────────────────────────────────────────────────

/// Compute the iMAML meta-gradient for a single task.
///
/// 1. Solve the proximal inner problem to obtain `θ*`.
/// 2. Evaluate the validation gradient `b = ∇L_val(θ*)`.
/// 3. Solve `(I + (1/λ)H) g = b` by CG; `g` is the task meta-gradient.
///
/// # Errors
///
/// Propagates inner-solve / CG errors.
pub fn imaml_task_gradient(
    theta_meta: &[f32],
    train_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    val_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    config: &ImamlConfig,
) -> MetaResult<Vec<f32>> {
    config.validate()?;
    let theta_star = proximal_inner_solve(theta_meta, train_grad, config)?;
    let b = val_grad(&theta_star);
    if b.len() != theta_meta.len() {
        return Err(MetaError::DimensionMismatch {
            expected: theta_meta.len(),
            got: b.len(),
        });
    }
    conjugate_gradient_implicit(&theta_star, &b, train_grad, config)
}

// ─── Meta-update over a task batch ───────────────────────────────────────────

/// iMAML meta-learner holding the shared initialisation `θ_meta`.
pub struct Imaml {
    theta: Vec<f32>,
    config: ImamlConfig,
}

/// One task for an iMAML meta-update: its train and validation gradient closures.
pub struct ImamlTask {
    /// Gradient of the *support* (train) loss `∇L_train(φ)`.
    pub train_grad: GradFn,
    /// Gradient of the *query* (validation) loss `∇L_val(φ)`.
    pub val_grad: GradFn,
}

impl Imaml {
    /// Create an iMAML learner over `dim` parameters initialised to `init`.
    ///
    /// # Errors
    ///
    /// * [`MetaError::InvalidEpisodeConfig`] — if `init` is empty.
    /// * Propagates [`ImamlConfig::validate`].
    pub fn new(init: Vec<f32>, config: ImamlConfig) -> MetaResult<Self> {
        if init.is_empty() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "init parameter vector must be non-empty".into(),
            });
        }
        config.validate()?;
        Ok(Self {
            theta: init,
            config,
        })
    }

    /// Read-only view of the current meta-parameters.
    #[inline]
    #[must_use]
    pub fn theta(&self) -> &[f32] {
        &self.theta
    }

    /// Perform one meta-update averaging the per-task implicit gradients and
    /// taking an outer SGD step `θ ← θ − outer_lr · ḡ`.
    ///
    /// Returns the squared norm of the averaged meta-gradient (a convergence
    /// proxy).
    ///
    /// # Errors
    ///
    /// * [`MetaError::EmptySupport`] — if `tasks` is empty.
    /// * Propagates per-task gradient errors.
    pub fn meta_step(&mut self, tasks: &[ImamlTask]) -> MetaResult<f32> {
        if tasks.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        let n = self.theta.len();
        let mut acc = vec![0.0_f32; n];
        for task in tasks {
            let g = imaml_task_gradient(
                &self.theta,
                task.train_grad.as_ref(),
                task.val_grad.as_ref(),
                &self.config,
            )?;
            if g.len() != n {
                return Err(MetaError::DimensionMismatch {
                    expected: n,
                    got: g.len(),
                });
            }
            for (a, &gi) in acc.iter_mut().zip(g.iter()) {
                *a += gi;
            }
        }
        let inv = 1.0 / tasks.len() as f32;
        let mut sq_norm = 0.0_f32;
        for (t, a) in self.theta.iter_mut().zip(acc.iter()) {
            let g = a * inv;
            *t -= self.config.outer_lr * g;
            sq_norm += g * g;
        }
        if !sq_norm.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "imaml meta_step produced non-finite gradient".into(),
            });
        }
        Ok(sq_norm)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadratic train loss `½ Σ a_i (φ_i − c_i)²` with gradient `a (φ − c)`.
    fn quad_grad(a: Vec<f32>, c: Vec<f32>) -> GradFn {
        Box::new(move |phi: &[f32]| {
            phi.iter()
                .zip(a.iter().zip(c.iter()))
                .map(|(&p, (&ai, &ci))| ai * (p - ci))
                .collect()
        })
    }

    #[test]
    fn config_default_is_valid() {
        assert!(ImamlConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_bad_lambda() {
        let c = ImamlConfig {
            lambda: 0.0,
            ..ImamlConfig::default()
        };
        assert!(matches!(c.validate(), Err(MetaError::InvalidLr { .. })));
    }

    #[test]
    fn config_rejects_zero_cg_iters() {
        let c = ImamlConfig {
            cg_iters: 0,
            ..ImamlConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn proximal_solve_moves_toward_train_min() {
        // train min at c = [5, -3]; with proximal pull to θ_meta=0 the optimum
        // lies strictly between 0 and c.
        let cfg = ImamlConfig {
            lambda: 1.0,
            inner_lr: 0.1,
            inner_steps: 200,
            ..ImamlConfig::default()
        };
        let g = quad_grad(vec![1.0, 1.0], vec![5.0, -3.0]);
        let theta_meta = vec![0.0, 0.0];
        let star = proximal_inner_solve(&theta_meta, g.as_ref(), &cfg).expect("solve");
        // Closed form for ½(φ-c)² + λ/2 φ²:  φ* = c/(1+λ) = c/2.
        assert!((star[0] - 2.5).abs() < 1e-2, "φ*_0 = {}", star[0]);
        assert!((star[1] - (-1.5)).abs() < 1e-2, "φ*_1 = {}", star[1]);
    }

    #[test]
    fn proximal_solve_rejects_bad_lr() {
        let cfg = ImamlConfig {
            inner_lr: 0.0,
            ..ImamlConfig::default()
        };
        let g = quad_grad(vec![1.0], vec![1.0]);
        assert!(proximal_inner_solve(&[0.0], g.as_ref(), &cfg).is_err());
    }

    #[test]
    fn hvp_of_quadratic_is_diagonal_scaling() {
        // For L = ½ Σ a_i φ_i², H = diag(a); H v = a ⊙ v.
        let g = quad_grad(vec![2.0, 3.0, 0.5], vec![0.0, 0.0, 0.0]);
        let theta = vec![1.0, 1.0, 1.0];
        let v = vec![1.0, 1.0, 1.0];
        let hv = hessian_vector_product(&theta, &v, g.as_ref(), 1e-2).expect("hvp");
        assert!((hv[0] - 2.0).abs() < 1e-2);
        assert!((hv[1] - 3.0).abs() < 1e-2);
        assert!((hv[2] - 0.5).abs() < 1e-2);
    }

    #[test]
    fn hvp_rejects_dim_mismatch() {
        let g = quad_grad(vec![1.0], vec![0.0]);
        assert!(hessian_vector_product(&[1.0], &[1.0, 2.0], g.as_ref(), 1e-3).is_err());
    }

    #[test]
    fn cg_solves_diagonal_system_exactly() {
        // H = diag(a). Operator A = I + (1/λ)H = diag(1 + a/λ).
        // Solving A g = b gives g_i = b_i / (1 + a_i/λ).
        let cfg = ImamlConfig {
            lambda: 2.0,
            cg_iters: 50,
            cg_tol: 1e-12,
            hvp_eps: 1e-2,
            ..ImamlConfig::default()
        };
        let a = vec![4.0_f32, 1.0, 9.0];
        let g = quad_grad(a.clone(), vec![0.0; 3]);
        let theta_star = vec![0.5, 0.5, 0.5];
        let b = vec![1.0, 1.0, 1.0];
        let sol = conjugate_gradient_implicit(&theta_star, &b, g.as_ref(), &cfg).expect("cg");
        for i in 0..3 {
            let expected = 1.0 / (1.0 + a[i] / cfg.lambda);
            assert!((sol[i] - expected).abs() < 1e-2, "sol[{i}]={}", sol[i]);
        }
    }

    #[test]
    fn cg_with_zero_hessian_returns_b() {
        // If H = 0 (constant gradient ⇒ a=0), A = I and g = b.
        let cfg = ImamlConfig::default();
        let g = quad_grad(vec![0.0, 0.0], vec![0.0, 0.0]);
        let b = vec![3.0, -7.0];
        let sol = conjugate_gradient_implicit(&[1.0, 1.0], &b, g.as_ref(), &cfg).expect("cg");
        assert!((sol[0] - 3.0).abs() < 1e-3);
        assert!((sol[1] - (-7.0)).abs() < 1e-3);
    }

    #[test]
    fn task_gradient_runs_end_to_end() {
        let cfg = ImamlConfig::default();
        let train = quad_grad(vec![1.0, 1.0], vec![2.0, 2.0]);
        let val = quad_grad(vec![1.0, 1.0], vec![2.0, 2.0]);
        let theta = vec![0.0, 0.0];
        let g = imaml_task_gradient(&theta, train.as_ref(), val.as_ref(), &cfg).expect("g");
        assert_eq!(g.len(), 2);
        assert!(g.iter().all(|v| v.is_finite()));
        // Validation min is at 2; from θ=0 the meta-gradient should point in the
        // negative direction (so an outer step *increases* θ toward 2).
        assert!(
            g[0] < 0.0 && g[1] < 0.0,
            "meta-grad should be negative: {g:?}"
        );
    }

    #[test]
    fn new_rejects_empty_init() {
        assert!(matches!(
            Imaml::new(vec![], ImamlConfig::default()),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn meta_step_rejects_empty_tasks() {
        let mut m = Imaml::new(vec![0.0, 0.0], ImamlConfig::default()).expect("m");
        assert!(matches!(m.meta_step(&[]), Err(MetaError::EmptySupport)));
    }

    #[test]
    fn meta_step_moves_theta_toward_task_optimum() {
        // Single task whose train & val both minimise at c = [3, -2].
        let cfg = ImamlConfig {
            outer_lr: 0.5,
            ..ImamlConfig::default()
        };
        let mut m = Imaml::new(vec![0.0, 0.0], cfg).expect("m");
        let theta0 = m.theta().to_vec();
        for _ in 0..30 {
            let task = ImamlTask {
                train_grad: quad_grad(vec![1.0, 1.0], vec![3.0, -2.0]),
                val_grad: quad_grad(vec![1.0, 1.0], vec![3.0, -2.0]),
            };
            m.meta_step(std::slice::from_ref(&task)).expect("step");
        }
        let theta1 = m.theta();
        // θ should move toward c = [3, -2].
        assert!(theta1[0] > theta0[0], "θ_0 should increase toward 3");
        assert!(theta1[1] < theta0[1], "θ_1 should decrease toward -2");
    }

    #[test]
    fn meta_step_returns_decreasing_norm_near_optimum() {
        // Starting near the optimum yields a small meta-gradient norm.
        let cfg = ImamlConfig::default();
        let mut m = Imaml::new(vec![3.0, -2.0], cfg).expect("m");
        let task = ImamlTask {
            train_grad: quad_grad(vec![1.0, 1.0], vec![3.0, -2.0]),
            val_grad: quad_grad(vec![1.0, 1.0], vec![3.0, -2.0]),
        };
        let norm = m.meta_step(std::slice::from_ref(&task)).expect("step");
        assert!(
            norm < 1e-2,
            "near-optimum meta-grad norm should be tiny: {norm}"
        );
    }

    #[test]
    fn larger_lambda_shrinks_adaptation() {
        // Bigger λ keeps θ* closer to θ_meta.
        let g = quad_grad(vec![1.0], vec![10.0]);
        let theta_meta = vec![0.0];
        let small = proximal_inner_solve(
            &theta_meta,
            g.as_ref(),
            &ImamlConfig {
                lambda: 0.5,
                inner_steps: 300,
                inner_lr: 0.05,
                ..ImamlConfig::default()
            },
        )
        .expect("s");
        let large = proximal_inner_solve(
            &theta_meta,
            g.as_ref(),
            &ImamlConfig {
                lambda: 5.0,
                inner_steps: 300,
                inner_lr: 0.05,
                ..ImamlConfig::default()
            },
        )
        .expect("l");
        assert!(
            large[0].abs() < small[0].abs(),
            "larger λ should adapt less: {} vs {}",
            large[0],
            small[0]
        );
    }
}
