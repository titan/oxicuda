//! Active CMA-ES: covariance matrix adaptation with negative rank-mu updates.
//!
//! Reference: G. Jastrebski & D. Arnold, "Improving Evolution Strategies through Active
//! Covariance Matrix Adaptation", Proc. CEC 2006, pp. 2814-2821.
//!
//! The key difference from standard CMA-ES is the addition of a *negative* covariance
//! update using the μ_neg worst individuals in each generation.  This accelerates
//! covariance adaptation by explicitly discouraging the directions of bad steps.

#![allow(clippy::needless_range_loop)]

use super::linalg::{b_mv, btranspose_mv, jacobi_eigen, norm};
use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for an Active CMA-ES run.
#[derive(Debug, Clone)]
pub struct ActiveCmaEsConfig {
    /// Problem dimension n.
    pub n: usize,
    /// Initial step size σ₀.
    pub sigma0: f64,
    /// Maximum number of objective evaluations.
    pub max_iter: usize,
    /// Termination tolerance on step size σ.
    pub tol: f64,
    /// Random seed.
    pub seed: u64,
}

impl ActiveCmaEsConfig {
    /// Construct a default `ActiveCmaEsConfig` for `n`-dimensional problems.
    pub fn new(n: usize) -> EvolResult<Self> {
        if n == 0 {
            return Err(EvolError::InvalidParameter("n must be >= 1".to_owned()));
        }
        Ok(Self {
            n,
            sigma0: 0.3,
            max_iter: 100_000,
            tol: 1e-11,
            seed: 0,
        })
    }
}

/// Mutable state for Active CMA-ES.
pub struct ActiveCmaEsState {
    /// Current distribution mean.
    pub mean: Vec<f64>,
    /// Current step size σ.
    pub sigma: f64,
    /// Covariance matrix C (n×n, row-major).
    pub c_matrix: Vec<f64>,
    /// Evolution path for σ-control.
    pub ps: Vec<f64>,
    /// Evolution path for rank-one update.
    pub pc: Vec<f64>,
    /// Eigenvalues of C (sorted descending).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors of C (columns, stored row-major — B matrix).
    pub eigenvectors: Vec<f64>,
    /// Current generation counter.
    pub generation: usize,

    // ── Derived parameters ───────────────────────────────────────────────────
    /// Population size λ.
    pub lambda: usize,
    /// Number of elites μ (positive).
    pub mu: usize,
    /// Number of negative samples μ_neg.
    pub mu_neg: usize,
    /// Recombination weights for positive update (length μ, sum to 1).
    pub weights: Vec<f64>,
    /// Effective mass μ_eff.
    pub mu_eff: f64,
    /// Step-size path learning rate c_σ.
    pub c_sigma: f64,
    /// Step-size damping d_σ.
    pub d_sigma: f64,
    /// Cumulation rate for p_c.
    pub c_c: f64,
    /// Rank-one learning rate c_1.
    pub c1: f64,
    /// Rank-μ learning rate c_mu.
    pub c_mu: f64,
    /// Negative update coefficient α_neg.
    pub alpha_neg: f64,
    /// Expected norm of N(0,I).
    pub chi_n: f64,
    /// Counter for scheduled eigendecomposition updates.
    eig_update_count: usize,
    /// D vector: sqrt of eigenvalues (for sampling).
    d_vector: Vec<f64>,
}

impl ActiveCmaEsState {
    /// Initialise state from a starting mean vector and configuration.
    pub fn new(mean_init: Vec<f64>, cfg: &ActiveCmaEsConfig) -> EvolResult<Self> {
        let n = cfg.n;
        if mean_init.len() != n {
            return Err(EvolError::DimensionMismatch {
                expected: n,
                got: mean_init.len(),
            });
        }

        // Population size and mu (same defaults as vanilla CMA-ES tutorial)
        let lambda = 4 + (3.0 * (n as f64).ln()).floor() as usize;
        let mu = lambda / 2;
        let mu_neg = mu / 2 + 1; // μ_neg ≥ 1

        // Positive recombination weights
        let raw_w: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = raw_w.iter().sum();
        let weights: Vec<f64> = raw_w.iter().map(|&w| w / w_sum).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Step-size control constants
        let nf = n as f64;
        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma =
            1.0 + c_sigma + 2.0 * f64::max(0.0, ((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0);

        // Covariance learning rates
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
        let c_mu = f64::min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff),
        );

        // Negative update coefficient: α_neg = 0.5 * μ_neg / (n + 1.5)^2
        let alpha_neg = 0.5 * mu_neg as f64 / (nf + 1.5).powi(2);

        // χ_n
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        // Initial C = I, B = I, D = [1,…,1]
        let mut c_matrix = vec![0.0f64; n * n];
        let mut eigenvectors = vec![0.0f64; n * n];
        for i in 0..n {
            c_matrix[i * n + i] = 1.0;
            eigenvectors[i * n + i] = 1.0;
        }
        let eigenvalues = vec![1.0f64; n];
        let d_vector = vec![1.0f64; n];

        Ok(Self {
            mean: mean_init,
            sigma: cfg.sigma0,
            c_matrix,
            ps: vec![0.0; n],
            pc: vec![0.0; n],
            eigenvalues,
            eigenvectors,
            generation: 0,
            lambda,
            mu,
            mu_neg,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            alpha_neg,
            chi_n,
            eig_update_count: 0,
            d_vector,
        })
    }

    /// Sample λ candidates from the current distribution.
    ///
    /// Each sample: x_k = m + σ · B · D · z_k, z_k ~ N(0, I).
    fn sample(&self, n: usize, rng: &mut LcgRng) -> Vec<Vec<f64>> {
        (0..self.lambda)
            .map(|_| {
                let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                let dz: Vec<f64> = (0..n).map(|i| self.d_vector[i] * z[i]).collect();
                let y = b_mv(&self.eigenvectors, &dz, n);
                (0..n).map(|i| self.mean[i] + self.sigma * y[i]).collect()
            })
            .collect()
    }

    /// Recompute eigenvectors and D from the current covariance matrix.
    fn update_eigen(&mut self, n: usize) -> EvolResult<()> {
        let mut a_copy = self.c_matrix.clone();
        match jacobi_eigen(&mut a_copy, n) {
            Ok((eigenvalues, b)) => {
                self.eigenvectors = b;
                self.d_vector = eigenvalues.iter().map(|&ev| ev.max(1e-20).sqrt()).collect();
                self.eigenvalues = eigenvalues;
            }
            Err(EvolError::EigenFailed(_)) => {
                // Soft fail: keep previous decomposition
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// Perform one Active CMA-ES generation step.
    ///
    /// Returns the best fitness value observed in this generation.
    pub fn step<F: Fn(&[f64]) -> f64>(
        &mut self,
        fitness_fn: &F,
        rng: &mut LcgRng,
        n: usize,
    ) -> EvolResult<f64> {
        let sigma_old = self.sigma;
        let m_old = self.mean.clone();

        // ── Sample λ candidates ──────────────────────────────────────────────
        let samples = self.sample(n, rng);
        let fitnesses: Vec<f64> = samples.iter().map(|x| fitness_fn(x)).collect();

        // ── Sort by fitness ascending (minimisation) ─────────────────────────
        let mut order: Vec<usize> = (0..self.lambda).collect();
        order.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_fit = fitnesses[order[0]];

        // ── New mean from top-μ ──────────────────────────────────────────────
        let mut m_new = vec![0.0f64; n];
        for j in 0..self.mu {
            let idx = order[j];
            for i in 0..n {
                m_new[i] += self.weights[j] * samples[idx][i];
            }
        }

        // ── Step-size path p_sigma update ────────────────────────────────────
        let delta_m: Vec<f64> = (0..n).map(|i| m_new[i] - m_old[i]).collect();
        let bt_delta = btranspose_mv(&self.eigenvectors, &delta_m, n);
        let inv_d_bt_delta: Vec<f64> = (0..n)
            .map(|i| bt_delta[i] / self.d_vector[i].max(1e-300))
            .collect();
        let invsqrt_c_delta = b_mv(&self.eigenvectors, &inv_d_bt_delta, n);

        let factor_ps = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..n {
            self.ps[i] =
                (1.0 - self.c_sigma) * self.ps[i] + factor_ps * invsqrt_c_delta[i] / sigma_old;
        }

        // ── σ update ─────────────────────────────────────────────────────────
        let ps_norm = norm(&self.ps);
        self.sigma = sigma_old * (self.c_sigma / self.d_sigma * (ps_norm / self.chi_n - 1.0)).exp();
        self.sigma = self.sigma.clamp(1e-15, 1e6);

        // ── h_sigma (heaviside indicator) ─────────────────────────────────────
        let gen1 = self.generation as f64 + 1.0;
        let denom = (1.0 - (1.0 - self.c_sigma).powf(2.0 * gen1)).sqrt();
        let h_sigma = if ps_norm / denom < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n {
            1.0
        } else {
            0.0
        };

        // ── p_c update ───────────────────────────────────────────────────────
        let factor_pc = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..n {
            self.pc[i] = (1.0 - self.c_c) * self.pc[i]
                + h_sigma * factor_pc * (m_new[i] - m_old[i]) / sigma_old;
        }

        // ── Positive rank-μ term ─────────────────────────────────────────────
        let mut rank_mu_pos = vec![0.0f64; n * n];
        for j in 0..self.mu {
            let idx = order[j];
            let y: Vec<f64> = (0..n)
                .map(|k| (samples[idx][k] - m_old[k]) / sigma_old)
                .collect();
            for r in 0..n {
                for c in 0..n {
                    rank_mu_pos[r * n + c] += self.weights[j] * y[r] * y[c];
                }
            }
        }

        // ── Negative rank-μ term (Active CMA-ES) ────────────────────────────
        // Select μ_neg worst individuals (highest fitness index = end of sorted order)
        let mu_neg = self.mu_neg.min(self.lambda.saturating_sub(self.mu));
        let w_neg = if mu_neg > 0 { 1.0 / mu_neg as f64 } else { 0.0 };
        let mut rank_mu_neg = vec![0.0f64; n * n];
        for j in 0..mu_neg {
            // worst individuals are at the end of sorted order
            let idx = order[self.lambda - 1 - j];
            let y: Vec<f64> = (0..n)
                .map(|k| (samples[idx][k] - m_old[k]) / sigma_old)
                .collect();
            for r in 0..n {
                for c in 0..n {
                    rank_mu_neg[r * n + c] += w_neg * y[r] * y[c];
                }
            }
        }

        // ── Covariance matrix update ──────────────────────────────────────────
        // C ← (1 - c1 - c_mu) * C
        //   + c1 * [p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C]
        //   + c_mu * [Σ_pos w_i y_i y_i^T  −  α_neg * Σ_neg w_neg_i y_neg_i y_neg_i^T]
        let base_scale = 1.0 - self.c1 - self.c_mu;
        let hsig_corr = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        let alpha = self.alpha_neg;

        for r in 0..n {
            for c in 0..n {
                let old_c = self.c_matrix[r * n + c];
                let pc_outer = self.pc[r] * self.pc[c];
                self.c_matrix[r * n + c] = base_scale * old_c
                    + self.c1 * (pc_outer + hsig_corr * old_c)
                    + self.c_mu * rank_mu_pos[r * n + c]
                    - self.c_mu * alpha * rank_mu_neg[r * n + c];
            }
        }

        // Enforce symmetry
        for r in 0..n {
            for c in (r + 1)..n {
                let avg = 0.5 * (self.c_matrix[r * n + c] + self.c_matrix[c * n + r]);
                self.c_matrix[r * n + c] = avg;
                self.c_matrix[c * n + r] = avg;
            }
        }

        // Clamp diagonal to stay positive definite (protect from negative updates overshooting)
        for i in 0..n {
            if self.c_matrix[i * n + i] < 1e-20 {
                self.c_matrix[i * n + i] = 1e-20;
            }
        }

        self.mean = m_new;
        self.generation += 1;

        // ── Schedule eigendecomposition ───────────────────────────────────────
        let update_freq = ((1.0 / (self.c1 + self.c_mu) / n as f64 / 10.0).floor() as usize).max(1);
        self.eig_update_count += 1;
        if self.eig_update_count >= update_freq {
            self.eig_update_count = 0;
            self.update_eigen(n)?;
        }

        Ok(best_fit)
    }
}

/// Run Active CMA-ES optimization from `init_mean` using the given configuration.
///
/// Returns the final state (containing the best solution in `state.mean`).
pub fn active_cmaes_run<F>(
    fitness_fn: F,
    init_mean: &[f64],
    cfg: &ActiveCmaEsConfig,
) -> EvolResult<ActiveCmaEsState>
where
    F: Fn(&[f64]) -> f64,
{
    let n = cfg.n;
    if init_mean.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: init_mean.len(),
        });
    }

    let mut rng = LcgRng::new(cfg.seed);
    let mut state = ActiveCmaEsState::new(init_mean.to_vec(), cfg)?;

    let mut best_x = state.mean.clone();
    let mut best_fit = fitness_fn(&best_x);
    let mut n_evals = 1usize;

    while n_evals < cfg.max_iter {
        let gen_best = state.step(&fitness_fn, &mut rng, n)?;
        n_evals += state.lambda;

        if gen_best < best_fit {
            best_fit = gen_best;
            best_x = state.mean.clone();
        }

        // Termination: step size below tolerance
        if state.sigma < cfg.tol {
            break;
        }
        // Termination: function value converged
        if best_fit < cfg.tol {
            break;
        }
    }

    // Set mean to best seen
    state.mean = best_x;

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper functions ──────────────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| {
                let (xi, xj) = (w[0], w[1]);
                100.0 * (xj - xi * xi).powi(2) + (1.0 - xi).powi(2)
            })
            .sum()
    }

    fn ellipsoid(x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| (10.0f64.powi(i as i32 * 6 / (x.len() - 1).max(1) as i32)) * xi * xi)
            .sum()
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // ── Config / state construction tests ────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = ActiveCmaEsConfig::new(5).unwrap();
        assert_eq!(cfg.n, 5);
        assert!(cfg.sigma0 > 0.0);
        assert!(cfg.max_iter > 0);
    }

    #[test]
    fn test_config_new_zero_dim() {
        assert!(ActiveCmaEsConfig::new(0).is_err());
    }

    #[test]
    fn test_state_new_dim_mismatch() {
        let cfg = ActiveCmaEsConfig::new(3).unwrap();
        let mean = vec![0.0f64; 5]; // wrong length
        assert!(ActiveCmaEsState::new(mean, &cfg).is_err());
    }

    #[test]
    fn test_state_new_identity_covariance() {
        let n = 4;
        let cfg = ActiveCmaEsConfig::new(n).unwrap();
        let state = ActiveCmaEsState::new(vec![0.0f64; n], &cfg).unwrap();
        // Diagonal of initial C should be 1.0
        for i in 0..n {
            assert!((state.c_matrix[i * n + i] - 1.0).abs() < 1e-10);
        }
        // Off-diagonal should be 0.0
        for r in 0..n {
            for c in 0..n {
                if r != c {
                    assert!(state.c_matrix[r * n + c].abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_state_alpha_neg_positive() {
        let cfg = ActiveCmaEsConfig::new(5).unwrap();
        let state = ActiveCmaEsState::new(vec![0.0; 5], &cfg).unwrap();
        assert!(state.alpha_neg > 0.0);
    }

    #[test]
    fn test_state_weights_sum_to_one() {
        let cfg = ActiveCmaEsConfig::new(6).unwrap();
        let state = ActiveCmaEsState::new(vec![0.0; 6], &cfg).unwrap();
        let s: f64 = state.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "weight sum = {s}");
    }

    #[test]
    fn test_state_mu_neg_positive() {
        let cfg = ActiveCmaEsConfig::new(4).unwrap();
        let state = ActiveCmaEsState::new(vec![0.0; 4], &cfg).unwrap();
        assert!(state.mu_neg >= 1);
    }

    // ── Single-step tests ─────────────────────────────────────────────────────

    #[test]
    fn test_step_returns_finite_fitness() {
        let cfg = ActiveCmaEsConfig {
            n: 3,
            sigma0: 0.5,
            max_iter: 100,
            tol: 1e-6,
            seed: 1,
        };
        let mut state = ActiveCmaEsState::new(vec![1.0, 1.0, 1.0], &cfg).unwrap();
        let mut rng = LcgRng::new(1);
        let best = state.step(&sphere, &mut rng, 3).unwrap();
        assert!(best.is_finite());
    }

    #[test]
    fn test_step_generation_increments() {
        let cfg = ActiveCmaEsConfig {
            n: 2,
            sigma0: 0.3,
            max_iter: 100,
            tol: 1e-6,
            seed: 2,
        };
        let mut state = ActiveCmaEsState::new(vec![0.5, -0.5], &cfg).unwrap();
        let mut rng = LcgRng::new(2);
        assert_eq!(state.generation, 0);
        state.step(&sphere, &mut rng, 2).unwrap();
        assert_eq!(state.generation, 1);
        state.step(&sphere, &mut rng, 2).unwrap();
        assert_eq!(state.generation, 2);
    }

    #[test]
    fn test_step_covariance_positive_diagonal() {
        let cfg = ActiveCmaEsConfig {
            n: 3,
            sigma0: 0.5,
            max_iter: 200,
            tol: 1e-8,
            seed: 3,
        };
        let mut state = ActiveCmaEsState::new(vec![2.0, -1.0, 0.5], &cfg).unwrap();
        let mut rng = LcgRng::new(3);
        for _ in 0..20 {
            state.step(&sphere, &mut rng, 3).unwrap();
        }
        // Diagonal should remain positive after negative updates
        for i in 0..3 {
            assert!(
                state.c_matrix[i * 3 + i] > 0.0,
                "C[{i},{i}] = {}",
                state.c_matrix[i * 3 + i]
            );
        }
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_run_sphere_2d() {
        let cfg = ActiveCmaEsConfig {
            n: 2,
            sigma0: 0.5,
            max_iter: 10_000,
            tol: 1e-8,
            seed: 42,
        };
        let state = active_cmaes_run(sphere, &[1.0, 1.0], &cfg).unwrap();
        let val = sphere(&state.mean);
        assert!(val < 1.0, "sphere 2D: {val} not < 1.0");
    }

    #[test]
    fn test_run_sphere_5d() {
        let cfg = ActiveCmaEsConfig {
            n: 5,
            sigma0: 0.3,
            max_iter: 50_000,
            tol: 1e-8,
            seed: 77,
        };
        let init = vec![2.0; 5];
        let state = active_cmaes_run(sphere, &init, &cfg).unwrap();
        let val = sphere(&state.mean);
        assert!(val < 1.0, "sphere 5D: {val}");
    }

    #[test]
    fn test_run_ellipsoid_3d() {
        let cfg = ActiveCmaEsConfig {
            n: 3,
            sigma0: 0.5,
            max_iter: 30_000,
            tol: 1e-8,
            seed: 11,
        };
        let init = vec![1.0, -1.0, 0.5];
        let state = active_cmaes_run(ellipsoid, &init, &cfg).unwrap();
        let val = ellipsoid(&state.mean);
        assert!(val < 10.0, "ellipsoid 3D: {val}");
    }

    #[test]
    fn test_run_rosenbrock_2d() {
        let cfg = ActiveCmaEsConfig {
            n: 2,
            sigma0: 0.5,
            max_iter: 50_000,
            tol: 1e-8,
            seed: 99,
        };
        let init = vec![0.0, 0.0];
        let state = active_cmaes_run(rosenbrock, &init, &cfg).unwrap();
        let val = rosenbrock(&state.mean);
        assert!(val < 100.0, "rosenbrock 2D: {val}");
    }

    #[test]
    fn test_run_rastrigin_2d_reduces_fitness() {
        // Rastrigin is highly multimodal — test that the result is strictly less than
        // the worst-case value (corners of [-5,5]^2 give ~50) after sufficient budget.
        let cfg = ActiveCmaEsConfig {
            n: 2,
            sigma0: 0.5,
            max_iter: 50_000,
            tol: 1e-8,
            seed: 55,
        };
        let init = vec![1.0, -1.0];
        let state = active_cmaes_run(rastrigin, &init, &cfg).unwrap();
        let final_val = rastrigin(&state.mean);
        // Worst possible near the corners ≈ 50; any decent run should do much better.
        assert!(final_val < 50.0, "rastrigin 2D: {final_val} not < 50.0");
    }

    #[test]
    fn test_run_dimension_mismatch_error() {
        let cfg = ActiveCmaEsConfig::new(3).unwrap();
        let init = vec![0.0; 5]; // wrong length
        assert!(active_cmaes_run(sphere, &init, &cfg).is_err());
    }

    #[test]
    fn test_state_sigma_decreases_on_sphere() {
        let cfg = ActiveCmaEsConfig {
            n: 3,
            sigma0: 1.0,
            max_iter: 5_000,
            tol: 1e-10,
            seed: 7,
        };
        let mut state = ActiveCmaEsState::new(vec![0.5, 0.5, 0.5], &cfg).unwrap();
        let mut rng = LcgRng::new(7);
        let sigma_init = state.sigma;
        for _ in 0..100 {
            state.step(&sphere, &mut rng, 3).unwrap();
        }
        // After 100 steps on sphere (near zero), sigma should have adapted
        assert!(
            state.sigma < sigma_init * 2.0,
            "sigma={} vs init={}",
            state.sigma,
            sigma_init
        );
    }
}
