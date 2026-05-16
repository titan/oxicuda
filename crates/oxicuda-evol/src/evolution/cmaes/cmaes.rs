//! Full CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
//!
//! Reference: N. Hansen, "The CMA Evolution Strategy: A Tutorial", 2016.
//! <https://arxiv.org/abs/1604.00772>

#![allow(clippy::needless_range_loop)]

use super::linalg::{b_mv, btranspose_mv, jacobi_eigen, norm};
use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for a CMA-ES run.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Number of decision variables (problem dimension n).
    pub n_dims: usize,
    /// Population size λ = 4 + ⌊3 ln n⌋.
    pub pop_size: usize,
    /// Number of elites μ = ⌊λ/2⌋.
    pub mu: usize,
    /// Initial step size σ₀.
    pub sigma_init: f64,
    /// Maximum number of objective evaluations.
    pub max_evals: usize,
    /// Termination criterion: change in function value.
    pub tol_fun: f64,
    /// Termination criterion: change in mean vector.
    pub tol_x: f64,
}

impl CmaEsConfig {
    /// Build a default `CmaEsConfig` for an `n`-dimensional problem.
    ///
    /// Population size and μ are set to the widely-used defaults from the CMA-ES tutorial.
    pub fn new(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        let pop_size = 4 + (3.0 * (n_dims as f64).ln()).floor() as usize;
        let mu = pop_size / 2;
        Ok(Self {
            n_dims,
            pop_size,
            mu,
            sigma_init: 0.3,
            max_evals: 100_000,
            tol_fun: 1e-12,
            tol_x: 1e-11,
        })
    }
}

/// Mutable algorithm state for a single CMA-ES run.
pub struct CmaEsState {
    /// Current distribution mean m.
    pub mean: Vec<f64>,
    /// Current step-size σ.
    pub sigma: f64,
    /// Cumulative path for C update.
    pub p_c: Vec<f64>,
    /// Cumulative path for σ update.
    pub p_sigma: Vec<f64>,
    /// Covariance matrix C (n×n, row-major).
    pub c_matrix: Vec<f64>,
    /// Eigenvectors B (columns = eigenvectors of C).
    pub b_matrix: Vec<f64>,
    /// Diagonal D: sqrt of eigenvalues of C.
    pub d_vector: Vec<f64>,
    /// Counter used to schedule eigendecomposition updates.
    pub eig_update_count: usize,
    /// Recombination weights (μ values, normalised to sum to 1).
    pub weights: Vec<f64>,
    /// Effective variance selection mass μ_eff = (∑w_i)² / ∑w_i².
    pub mu_eff: f64,
    // ── CMA-ES step-size control constants ──────────────────────────────────
    /// c_σ: step-size path learning rate.
    pub c_sigma: f64,
    /// d_σ: step-size damping.
    pub d_sigma: f64,
    // ── Covariance update constants ─────────────────────────────────────────
    /// c_c: cumulation path for C learning rate.
    pub c_c: f64,
    /// c₁: rank-one update learning rate.
    pub c1: f64,
    /// c_μ: rank-μ update learning rate.
    pub c_mu: f64,
    /// χ_n = E[‖N(0,I)‖] ≈ √n (1 − 1/(4n) + 1/(21n²)).
    pub chi_n: f64,
    /// Total number of objective evaluations performed.
    pub n_evals: usize,
    /// Current generation index.
    pub generation: usize,
}

impl CmaEsState {
    /// Initialise CMA-ES state centred at `mean_init`.
    pub fn new(mean_init: Vec<f64>, cfg: &CmaEsConfig) -> EvolResult<Self> {
        let n = cfg.n_dims;
        if mean_init.len() != n {
            return Err(EvolError::DimensionMismatch {
                expected: n,
                got: mean_init.len(),
            });
        }
        if cfg.mu == 0 {
            return Err(EvolError::InvalidParameter("mu must be >= 1".to_owned()));
        }

        // ── Weights ──────────────────────────────────────────────────────────
        let mu = cfg.mu;
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // ── Step-size control ────────────────────────────────────────────────
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma =
            1.0 + c_sigma + 2.0 * f64::max(0.0, ((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0);

        // ── Covariance learning rates ────────────────────────────────────────
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = f64::min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff),
        );

        // ── χ_n ──────────────────────────────────────────────────────────────
        let nf = n as f64;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        // ── Initial C = I, B = I, D = [1,…,1] ───────────────────────────────
        let mut c_matrix = vec![0.0f64; n * n];
        let mut b_matrix = vec![0.0f64; n * n];
        for i in 0..n {
            c_matrix[i * n + i] = 1.0;
            b_matrix[i * n + i] = 1.0;
        }
        let d_vector = vec![1.0f64; n];

        Ok(Self {
            mean: mean_init,
            sigma: cfg.sigma_init,
            p_c: vec![0.0; n],
            p_sigma: vec![0.0; n],
            c_matrix,
            b_matrix,
            d_vector,
            eig_update_count: 0,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
            n_evals: 0,
            generation: 0,
        })
    }

    /// Sample λ candidate solutions from the current distribution.
    ///
    /// Each sample `x_k = m + σ · B · D · z_k` where `z_k ~ N(0, I)`.
    pub fn sample(&self, cfg: &CmaEsConfig, rng: &mut LcgRng) -> Vec<Vec<f64>> {
        let n = cfg.n_dims;
        (0..cfg.pop_size)
            .map(|_| {
                // z ~ N(0, I)
                let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                // y = B * D * z
                let dz: Vec<f64> = (0..n).map(|i| self.d_vector[i] * z[i]).collect();
                let y = b_mv(&self.b_matrix, &dz, n);
                // x = m + sigma * y
                (0..n).map(|i| self.mean[i] + self.sigma * y[i]).collect()
            })
            .collect()
    }

    /// Perform one CMA-ES update step given a set of samples and their fitness values.
    pub fn update(
        &mut self,
        samples: &[Vec<f64>],
        fitnesses: &[f64],
        cfg: &CmaEsConfig,
    ) -> EvolResult<()> {
        let n = cfg.n_dims;
        let mu = cfg.mu;
        let lambda = cfg.pop_size;

        if samples.len() != lambda {
            return Err(EvolError::DimensionMismatch {
                expected: lambda,
                got: samples.len(),
            });
        }

        // ── Sort by fitness (ascending = minimisation) ───────────────────────
        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sigma_old = self.sigma;
        let m_old = self.mean.clone();

        // ── New mean: weighted combination of top-μ samples ──────────────────
        let mut m_new = vec![0.0f64; n];
        for j in 0..mu {
            let idx = order[j];
            for i in 0..n {
                m_new[i] += self.weights[j] * samples[idx][i];
            }
        }

        // ── Step-size path p_sigma update ────────────────────────────────────
        // invsqrt_C * (m_new - m_old) = B * diag(1/D) * B^T * delta_m
        let delta_m: Vec<f64> = (0..n).map(|i| m_new[i] - m_old[i]).collect();
        let bt_delta = btranspose_mv(&self.b_matrix, &delta_m, n);
        let inv_d_bt_delta: Vec<f64> = (0..n)
            .map(|i| bt_delta[i] / self.d_vector[i].max(1e-300))
            .collect();
        let invsqrt_c_delta = b_mv(&self.b_matrix, &inv_d_bt_delta, n);

        let factor_ps = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..n {
            self.p_sigma[i] =
                (1.0 - self.c_sigma) * self.p_sigma[i] + factor_ps * invsqrt_c_delta[i] / sigma_old;
        }

        // ── σ update ─────────────────────────────────────────────────────────
        let ps_norm = norm(&self.p_sigma);
        self.sigma = sigma_old * (self.c_sigma / self.d_sigma * (ps_norm / self.chi_n - 1.0)).exp();
        // Clamp sigma to avoid numerical explosion
        self.sigma = self.sigma.clamp(1e-15, 1e6);

        // ── h_sigma (heaviside) ──────────────────────────────────────────────
        let gen1 = self.generation as f64 + 1.0;
        let denom = (1.0 - (1.0 - self.c_sigma).powf(2.0 * gen1)).sqrt();
        let h_sigma = if ps_norm / denom < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n {
            1.0
        } else {
            0.0
        };

        // ── Cumulation path p_c update ────────────────────────────────────────
        let factor_pc = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..n {
            self.p_c[i] = (1.0 - self.c_c) * self.p_c[i]
                + h_sigma * factor_pc * (m_new[i] - m_old[i]) / sigma_old;
        }

        // ── Covariance matrix update ──────────────────────────────────────────
        // C ← (1-c1-c_mu) * C + c1 * (p_c * p_c^T + (1-h_sigma)*c_c*(2-c_c)*C)
        //   + c_mu * Σ_i w_i * y_i * y_i^T
        let base_scale = 1.0 - self.c1 - self.c_mu;
        let hsig_correction = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);

        // Rank-μ terms
        let mut rank_mu = vec![0.0f64; n * n];
        for j in 0..mu {
            let idx = order[j];
            let y: Vec<f64> = (0..n)
                .map(|k| (samples[idx][k] - m_old[k]) / sigma_old)
                .collect();
            for r in 0..n {
                for c in 0..n {
                    rank_mu[r * n + c] += self.weights[j] * y[r] * y[c];
                }
            }
        }

        // Update C
        for r in 0..n {
            for c in 0..n {
                let old_c = self.c_matrix[r * n + c];
                let pc_outer = self.p_c[r] * self.p_c[c];
                self.c_matrix[r * n + c] = base_scale * old_c
                    + self.c1 * (pc_outer + hsig_correction * old_c)
                    + self.c_mu * rank_mu[r * n + c];
            }
        }
        // Enforce symmetry (numerical cleanup)
        for r in 0..n {
            for c in (r + 1)..n {
                let avg = 0.5 * (self.c_matrix[r * n + c] + self.c_matrix[c * n + r]);
                self.c_matrix[r * n + c] = avg;
                self.c_matrix[c * n + r] = avg;
            }
        }

        // Update mean
        self.mean = m_new;
        self.generation += 1;
        self.n_evals += lambda;

        // ── Eigendecomposition update ─────────────────────────────────────────
        // Schedule: every floor(1 / (c1 + c_mu) / n / 10) generations (min 1).
        let update_freq = ((1.0 / (self.c1 + self.c_mu) / n as f64 / 10.0).floor() as usize).max(1);
        self.eig_update_count += 1;
        if self.eig_update_count >= update_freq {
            self.eig_update_count = 0;
            self.update_eigen(n)?;
        }

        Ok(())
    }

    /// Recompute B and D from the current covariance matrix C.
    fn update_eigen(&mut self, n: usize) -> EvolResult<()> {
        let mut a_copy = self.c_matrix.clone();
        match jacobi_eigen(&mut a_copy, n) {
            Ok((eigenvalues, b)) => {
                self.b_matrix = b;
                self.d_vector = eigenvalues.iter().map(|&ev| ev.max(1e-20).sqrt()).collect();
            }
            Err(EvolError::EigenFailed(_)) => {
                // Soft fail: keep previous B/D to avoid crashing
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// Run full CMA-ES optimization.
    ///
    /// Returns `(best_x, best_fitness)`.
    pub fn run<F: Fn(&[f64]) -> f64>(
        &mut self,
        objective: F,
        cfg: &CmaEsConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<(Vec<f64>, f64)> {
        let mut best_x = self.mean.clone();
        let mut best_fit = objective(&best_x);
        self.n_evals = 1;

        while self.n_evals < cfg.max_evals {
            let samples = self.sample(cfg, rng);
            let fitnesses: Vec<f64> = samples.iter().map(|x| objective(x)).collect();
            self.n_evals += fitnesses.len();

            // Track best
            for (x, &f) in samples.iter().zip(fitnesses.iter()) {
                if f < best_fit {
                    best_fit = f;
                    best_x = x.clone();
                }
            }

            // Termination checks before update
            if best_fit < cfg.tol_fun {
                break;
            }

            self.update(&samples, &fitnesses, cfg)?;

            // Convergence: sigma too small
            if self.sigma < cfg.tol_x {
                break;
            }
        }

        Ok((best_x, best_fit))
    }
}
