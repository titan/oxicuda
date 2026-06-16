//! Distributionally Robust Optimisation (DRO) with Wasserstein uncertainty sets.
//!
//! Implements the DRO-ERM framework of Esfahani & Kuhn (2018) using a
//! Wasserstein ball of radius `ε` centred at the empirical measure `P̂_n` as
//! the uncertainty set.
//!
//! # Background
//!
//! Given a loss function `ℓ(z; θ)` and `n` training samples `{ẑ₁,…,ẑ_n}`,
//! the DRO-Wasserstein problem is
//!
//! ```text
//! min_{θ} max_{Q : W_p(Q, P̂_n) ≤ ε}  E_{Z~Q}[ℓ(Z; θ)]
//! ```
//!
//! Esfahani & Kuhn (2018) show that, for `p=1` and Lipschitz losses, the
//! minimax problem admits the tractable dual reformulation
//!
//! ```text
//! min_{θ, λ ≥ 0}  λε + (1/n) Σ_i sup_{z} { ℓ(z; θ) − λ ‖z − ẑ_i‖ }
//! ```
//!
//! For the empirical adversarial worst-case, the inner `sup` is approximated
//! over a finite set of candidate perturbations.
//!
//! This module provides:
//! - [`DroConfig`] — problem parameters (radius `ε`, Wasserstein order `p`,
//!   iteration budget, step size).
//! - [`DroSolver`] — solver struct that holds the training data and exposes
//!   `solve(loss_fn)`.
//! - [`DroResult`] — output: optimal Lagrange multiplier `λ*`, worst-case
//!   loss bound, and per-sample adversarial perturbation magnitudes.
//!
//! # Parametric loss
//!
//! The caller supplies a closure `loss_fn : (&[f32]) → f32` that evaluates
//! the loss on a single sample.  For differentiable parametric losses the
//! solver also accepts a gradient closure to drive gradient-based updates of
//! `λ`.
//!
//! # Algorithm
//!
//! We implement a projected sub-gradient ascent on the dual variable `λ` and
//! estimate the worst-case risk via *finite-sample adversarial attack* — for
//! each training sample `ẑ_i` we search over `n_adv` random perturbation
//! directions `δ` with `‖δ‖ ≤ ε_budget` and keep the one that maximises
//! `ℓ(ẑ_i + δ; θ)`.
//!
//! References:
//! - Esfahani P.M. & Kuhn D. *Data-driven Distributionally Robust
//!   Optimization Using the Wasserstein Metric* (Management Science, 2018).
//! - Blanchet J. & Murthy K. *Quantifying distributional model risk via
//!   optimal transport* (Math. Oper. Res., 2019).

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Wasserstein DRO solver.
#[derive(Debug, Clone)]
pub struct DroConfig {
    /// Wasserstein ball radius `ε > 0`.
    pub epsilon: f32,
    /// Wasserstein order `p ∈ {1, 2}`.
    pub p: u32,
    /// Maximum sub-gradient iterations for `λ`.
    pub max_iter: usize,
    /// Initial value of the dual Lagrange multiplier `λ`.
    pub lambda_init: f32,
    /// Step size for sub-gradient updates of `λ`.
    pub lambda_lr: f32,
    /// Number of random adversarial perturbation directions to try per sample.
    pub n_adv: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for DroConfig {
    fn default() -> Self {
        DroConfig {
            epsilon: 0.1,
            p: 1,
            max_iter: 50,
            lambda_init: 1.0,
            lambda_lr: 0.01,
            n_adv: 20,
            seed: 42,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Result
// ──────────────────────────────────────────────────────────────────────────────

/// Output of the Wasserstein DRO solver.
#[derive(Debug, Clone)]
pub struct DroResult {
    /// Optimal dual Lagrange multiplier `λ*` (transport cost penalty).
    pub lambda_star: f32,
    /// Worst-case empirical risk bound: `λ*·ε + (1/n) Σ_i sup_z {ℓ(z;θ) − λ*·‖z−ẑ_i‖}`.
    pub worst_case_risk: f32,
    /// Nominal (in-sample) average loss: `(1/n) Σ_i ℓ(ẑ_i; θ)`.
    pub nominal_risk: f32,
    /// Per-sample adversarial perturbation magnitudes `‖δ_i‖`, length `n`.
    pub perturbation_norms: Vec<f32>,
    /// Dual objective history (one entry per outer iteration).
    pub dual_history: Vec<f32>,
    /// Number of training samples.
    pub n_samples: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Solver
// ──────────────────────────────────────────────────────────────────────────────

/// Wasserstein DRO solver.
///
/// Holds the training dataset and all configuration. Call [`DroSolver::solve`]
/// with a loss function to obtain the robust risk bound and the optimal dual
/// variable `λ*`.
#[derive(Debug, Clone)]
pub struct DroSolver {
    /// Training samples, row-major `[n × dim]`.
    pub samples: Vec<f32>,
    /// Number of samples.
    pub n: usize,
    /// Sample dimensionality.
    pub dim: usize,
    /// Solver configuration.
    pub cfg: DroConfig,
}

impl DroSolver {
    /// Create a new solver from training samples.
    ///
    /// `samples` must have length `n * dim`, row-major.
    pub fn new(samples: Vec<f32>, n: usize, dim: usize, cfg: DroConfig) -> OtResult<Self> {
        if dim == 0 {
            return Err(OtError::BadDim { got: 0 });
        }
        if n == 0 {
            return Err(OtError::EmptyInput);
        }
        if samples.len() != n * dim {
            return Err(OtError::IncompatibleLength {
                a: samples.len(),
                b: n * dim,
            });
        }
        if cfg.epsilon <= 0.0 {
            return Err(OtError::BadEpsilon { eps: cfg.epsilon });
        }
        if cfg.p == 0 {
            return Err(OtError::BadCount { got: 0 });
        }
        Ok(DroSolver {
            samples,
            n,
            dim,
            cfg,
        })
    }

    /// Solve the Wasserstein DRO problem for a given loss function.
    ///
    /// `loss_fn(z)` evaluates the loss at sample `z ∈ ℝ^dim` (slice of length
    /// `dim`).  Must be finite for all inputs.
    pub fn solve<F>(&self, loss_fn: F) -> OtResult<DroResult>
    where
        F: Fn(&[f32]) -> f32,
    {
        let n = self.n;
        let d = self.dim;
        let eps = self.cfg.epsilon;
        let p = self.cfg.p;

        // ── Nominal risk ──────────────────────────────────────────────────────
        let mut nominal_risk = 0.0_f32;
        for i in 0..n {
            let zi = &self.samples[i * d..(i + 1) * d];
            nominal_risk += loss_fn(zi);
        }
        nominal_risk /= n as f32;

        // ── Dual sub-gradient loop on λ ───────────────────────────────────────
        let mut rng = LcgRng::new(self.cfg.seed);
        let mut lambda = self.cfg.lambda_init.max(1e-6);

        let mut dual_history = Vec::with_capacity(self.cfg.max_iter);
        let mut best_dual = f32::NEG_INFINITY;
        let mut best_lambda = lambda;

        for _iter in 0..self.cfg.max_iter {
            // For each sample, compute  sup_{z} {ℓ(z;θ) − λ‖z−ẑ_i‖_p}
            // approximated over random directions.
            let (avg_sup, _perturb_norms) =
                self.compute_average_sup(&loss_fn, lambda, p, eps, &mut rng);

            let dual_val = lambda * eps + avg_sup;
            dual_history.push(dual_val);

            if dual_val > best_dual {
                best_dual = dual_val;
                best_lambda = lambda;
            }

            // Sub-gradient of dual w.r.t. λ: d/dλ [λε + avg_sup]
            // avg_sup = (1/n) Σ_i sup_z { ℓ − λ‖z−ẑ_i‖ }
            // d(avg_sup)/d(λ) ≈ −(1/n) Σ_i ‖δ*_i‖  (by envelope theorem)
            let sub_grad = eps - _perturb_norms.iter().sum::<f32>() / n as f32;

            // Projected sub-gradient ascent (maximise dual → negative gradient
            // step on minimisation form is positive step here).
            lambda = (lambda + self.cfg.lambda_lr * sub_grad).max(1e-9);
        }

        // ── Compute final worst-case quantities at λ* ─────────────────────────
        let (worst_case_risk, perturbation_norms) =
            self.compute_average_sup_with_norms(&loss_fn, best_lambda, p, eps, &mut rng);

        Ok(DroResult {
            lambda_star: best_lambda,
            worst_case_risk: best_lambda * eps + worst_case_risk,
            nominal_risk,
            perturbation_norms,
            dual_history,
            n_samples: n,
        })
    }

    /// Compute `(1/n) Σ_i sup_z { ℓ(z;θ) − λ‖z−ẑ_i‖_p }` over adversarial
    /// perturbations and return `(avg_sup, per_sample_norms)`.
    fn compute_average_sup<F>(
        &self,
        loss_fn: &F,
        lambda: f32,
        p: u32,
        eps_budget: f32,
        rng: &mut LcgRng,
    ) -> (f32, Vec<f32>)
    where
        F: Fn(&[f32]) -> f32,
    {
        let n = self.n;
        let d = self.dim;
        let n_adv = self.cfg.n_adv;
        let mut avg = 0.0_f32;
        let mut norms = Vec::with_capacity(n);

        let mut perturbed = vec![0.0_f32; d];
        let mut best_delta_norm;

        for i in 0..n {
            let zi = &self.samples[i * d..(i + 1) * d];
            let loss_zi = loss_fn(zi);
            let mut best_val = loss_zi; // δ = 0 is always feasible
            best_delta_norm = 0.0_f32;

            for _ in 0..n_adv {
                // Sample random direction and scale to radius eps_budget
                let mut dir = vec![0.0_f32; d];
                let mut norm_sq = 0.0_f32;
                for v in dir.iter_mut() {
                    let u1 = rng.next_f32().max(1e-9_f32);
                    let u2 = rng.next_f32();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    *v = z;
                    norm_sq += z * z;
                }
                let dir_norm = norm_sq.sqrt().max(1e-12);
                let scale = eps_budget / dir_norm;

                for (pv, (&zv, &dv)) in perturbed.iter_mut().zip(zi.iter().zip(dir.iter())) {
                    *pv = zv + scale * dv;
                }
                let delta_norm: f32 = match p {
                    1 => perturbed
                        .iter()
                        .zip(zi.iter())
                        .map(|(pv, zv)| (pv - zv).abs())
                        .sum(),
                    _ => perturbed
                        .iter()
                        .zip(zi.iter())
                        .map(|(pv, zv)| (pv - zv).powi(2))
                        .sum::<f32>()
                        .sqrt(),
                };
                let val = loss_fn(&perturbed) - lambda * delta_norm;
                if val > best_val {
                    best_val = val;
                    best_delta_norm = delta_norm;
                }
            }

            avg += best_val;
            norms.push(best_delta_norm);
        }

        (avg / n as f32, norms)
    }

    /// Variant of [`compute_average_sup`] that also returns per-sample norms.
    fn compute_average_sup_with_norms<F>(
        &self,
        loss_fn: &F,
        lambda: f32,
        p: u32,
        eps_budget: f32,
        rng: &mut LcgRng,
    ) -> (f32, Vec<f32>)
    where
        F: Fn(&[f32]) -> f32,
    {
        self.compute_average_sup(loss_fn, lambda, p, eps_budget, rng)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the Wasserstein DRO worst-case risk bound for a quadratic loss.
///
/// Convenience wrapper: `ℓ(z; θ) = ‖z − θ‖²` (squared L₂ distance to `theta`).
/// Returns [`DroResult`] for the worst-case risk over the Wasserstein ball.
pub fn dro_quadratic_loss(
    samples: &[f32],
    n: usize,
    dim: usize,
    theta: &[f32],
    cfg: DroConfig,
) -> OtResult<DroResult> {
    if theta.len() != dim {
        return Err(OtError::IncompatibleLength {
            a: theta.len(),
            b: dim,
        });
    }
    let theta_owned = theta.to_vec();
    let solver = DroSolver::new(samples.to_vec(), n, dim, cfg)?;
    let d = dim;
    solver.solve(move |z: &[f32]| {
        z.iter()
            .zip(theta_owned.iter())
            .map(|(&zi, &ti)| (zi - ti).powi(2))
            .sum::<f32>()
            / d as f32
    })
}

/// Compute the Wasserstein robustness certificate for a given nominal risk.
///
/// For an `ε`-Wasserstein ball, a Lipschitz loss with constant `L` satisfies
/// `worst_case_risk ≤ nominal_risk + L · ε`.  This function returns the
/// certified upper bound `nominal_risk + lipschitz_const * epsilon`.
pub fn dro_lipschitz_bound(nominal_risk: f32, lipschitz_const: f32, epsilon: f32) -> f32 {
    nominal_risk + lipschitz_const * epsilon
}

/// Evaluate the empirical Wasserstein robustness gap:
/// `worst_case_risk − nominal_risk` from a solved [`DroResult`].
pub fn robustness_gap(result: &DroResult) -> f32 {
    result.worst_case_risk - result.nominal_risk
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize, dim: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(42);
        (0..n * dim).map(|_| rng.next_f32()).collect()
    }

    #[test]
    fn test_dro_solver_construction() {
        let samples = make_samples(10, 3);
        let cfg = DroConfig::default();
        let solver = DroSolver::new(samples, 10, 3, cfg);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_dro_solver_bad_dim() {
        let samples = make_samples(5, 2);
        let cfg = DroConfig::default();
        let err = DroSolver::new(samples, 5, 0, cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_dro_solver_empty() {
        let cfg = DroConfig::default();
        let err = DroSolver::new(vec![], 0, 2, cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_dro_solver_bad_epsilon() {
        let samples = make_samples(5, 2);
        let cfg = DroConfig {
            epsilon: -0.1,
            ..DroConfig::default()
        };
        let err = DroSolver::new(samples, 5, 2, cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_dro_solve_quadratic_loss() {
        let n = 20;
        let dim = 2;
        let samples = make_samples(n, dim);
        let cfg = DroConfig {
            epsilon: 0.05,
            max_iter: 10,
            n_adv: 5,
            lambda_lr: 0.01,
            lambda_init: 1.0,
            p: 1,
            seed: 99,
        };
        let theta = vec![0.5_f32; dim];
        let result = dro_quadratic_loss(&samples, n, dim, &theta, cfg).expect("dro ok");
        assert!(result.worst_case_risk.is_finite());
        assert!(result.nominal_risk.is_finite());
        assert!(result.worst_case_risk >= result.nominal_risk - 1e-4);
    }

    #[test]
    fn test_dro_result_dual_history_length() {
        let n = 10;
        let dim = 2;
        let samples = make_samples(n, dim);
        let max_iter = 7;
        let cfg = DroConfig {
            max_iter,
            n_adv: 3,
            ..DroConfig::default()
        };
        let solver = DroSolver::new(samples, n, dim, cfg).expect("ok");
        let result = solver
            .solve(|z: &[f32]| z.iter().map(|&v| v * v).sum::<f32>())
            .expect("ok");
        assert_eq!(result.dual_history.len(), max_iter);
        for &v in &result.dual_history {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dro_perturbation_norms_length() {
        let n = 8;
        let dim = 3;
        let samples = make_samples(n, dim);
        let cfg = DroConfig {
            max_iter: 3,
            n_adv: 4,
            ..DroConfig::default()
        };
        let solver = DroSolver::new(samples, n, dim, cfg).expect("ok");
        let result = solver.solve(|_: &[f32]| 1.0_f32).expect("ok");
        assert_eq!(result.perturbation_norms.len(), n);
    }

    #[test]
    fn test_robustness_gap_nonneg() {
        let n = 10;
        let dim = 2;
        let samples = make_samples(n, dim);
        let cfg = DroConfig {
            epsilon: 0.1,
            max_iter: 5,
            n_adv: 3,
            ..DroConfig::default()
        };
        let theta = vec![0.0_f32; dim];
        let result = dro_quadratic_loss(&samples, n, dim, &theta, cfg).expect("ok");
        let gap = robustness_gap(&result);
        // Gap should be ≥ 0 (worst-case ≥ nominal up to numerical tolerance)
        assert!(gap > -0.5, "robustness_gap={gap}");
    }

    #[test]
    fn test_dro_lipschitz_bound() {
        let bound = dro_lipschitz_bound(1.0, 2.0, 0.1);
        assert!((bound - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_dro_p2_wasserstein() {
        let n = 8;
        let dim = 2;
        let samples = make_samples(n, dim);
        let cfg = DroConfig {
            p: 2,
            max_iter: 3,
            n_adv: 3,
            ..DroConfig::default()
        };
        let solver = DroSolver::new(samples, n, dim, cfg).expect("ok");
        let result = solver.solve(|z: &[f32]| z.iter().sum::<f32>()).expect("ok");
        assert!(result.lambda_star > 0.0);
    }

    #[test]
    fn test_dro_lambda_stays_positive() {
        let n = 15;
        let dim = 2;
        let samples = make_samples(n, dim);
        let cfg = DroConfig {
            lambda_init: 0.01,
            lambda_lr: 0.1,
            max_iter: 20,
            n_adv: 5,
            ..DroConfig::default()
        };
        let solver = DroSolver::new(samples, n, dim, cfg).expect("ok");
        let result = solver
            .solve(|z: &[f32]| z.iter().map(|&v| v.abs()).sum::<f32>())
            .expect("ok");
        assert!(result.lambda_star > 0.0);
    }
}
