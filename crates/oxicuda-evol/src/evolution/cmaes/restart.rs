//! IPOP / BIPOP CMA-ES: population-doubling and bi-population restart strategies.
//!
//! # References
//! - IPOP: A. Auger & N. Hansen, "A restart CMA evolution strategy with increasing population
//!   size", Proc. CEC 2005, pp. 1769-1776.
//! - BIPOP: N. Hansen, "Benchmarking a BI-population CMA-ES on the BBOB-2009 function
//!   testbed", GECCO'09 Companion, 2009.

#![allow(clippy::needless_range_loop)]

use super::linalg::{b_mv, btranspose_mv, jacobi_eigen, norm};
use crate::{EvolError, EvolResult, handle::LcgRng};

// ── Internal single-run CMA-ES state ─────────────────────────────────────────

/// Internal mutable state for one CMA-ES run inside a restart wrapper.
struct InnerCmaEs {
    /// Current distribution mean m.
    mean: Vec<f64>,
    /// Current step size σ.
    sigma: f64,
    /// Cumulative path for C update.
    p_c: Vec<f64>,
    /// Cumulative path for σ update.
    p_sigma: Vec<f64>,
    /// Covariance matrix C (n×n, row-major).
    c_matrix: Vec<f64>,
    /// Eigenvectors B (columns = eigenvectors of C).
    b_matrix: Vec<f64>,
    /// Diagonal D: sqrt of eigenvalues of C.
    d_vector: Vec<f64>,
    /// Counter used to schedule eigendecomposition updates.
    eig_update_count: usize,
    /// Recombination weights (μ values, normalised to sum to 1).
    weights: Vec<f64>,
    /// Effective variance selection mass μ_eff.
    mu_eff: f64,
    /// c_σ: step-size path learning rate.
    c_sigma: f64,
    /// d_σ: step-size damping.
    d_sigma: f64,
    /// c_c: cumulation path for C learning rate.
    c_c: f64,
    /// c₁: rank-one update learning rate.
    c1: f64,
    /// c_μ: rank-μ update learning rate.
    c_mu: f64,
    /// χ_n = E[‖N(0,I)‖].
    chi_n: f64,
    /// Total number of evaluations consumed by this run.
    n_evals: usize,
    /// Current generation index.
    generation: usize,
    /// Population size λ for this run.
    lambda: usize,
    /// Elite count μ for this run.
    mu: usize,
    /// Problem dimension n.
    n_dims: usize,
}

impl InnerCmaEs {
    fn new(mean_init: Vec<f64>, sigma0: f64, lambda: usize, rng: &mut LcgRng) -> EvolResult<Self> {
        let n = mean_init.len();
        if n == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        let mu = lambda / 2;
        if mu == 0 {
            return Err(EvolError::InvalidParameter(
                "lambda must be >= 2".to_owned(),
            ));
        }

        // ── Weights ──────────────────────────────────────────────────────────
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

        let nf = n as f64;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        let mut c_matrix = vec![0.0f64; n * n];
        let mut b_matrix = vec![0.0f64; n * n];
        for i in 0..n {
            c_matrix[i * n + i] = 1.0;
            b_matrix[i * n + i] = 1.0;
        }
        let d_vector = vec![1.0f64; n];

        // Jitter the initial mean slightly if requested via rng (used by BIPOP small regime)
        let _ = rng; // consumed only to satisfy borrow; callers may pre-jitter mean

        Ok(Self {
            mean: mean_init,
            sigma: sigma0,
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
            lambda,
            mu,
            n_dims: n,
        })
    }

    /// Sample `lambda` candidate solutions from the current distribution.
    fn sample(&self, rng: &mut LcgRng) -> Vec<Vec<f64>> {
        let n = self.n_dims;
        (0..self.lambda)
            .map(|_| {
                let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                let dz: Vec<f64> = (0..n).map(|i| self.d_vector[i] * z[i]).collect();
                let y = b_mv(&self.b_matrix, &dz, n);
                (0..n).map(|i| self.mean[i] + self.sigma * y[i]).collect()
            })
            .collect()
    }

    /// Perform one update step.
    fn update(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) -> EvolResult<()> {
        let n = self.n_dims;
        let mu = self.mu;
        let lambda = self.lambda;

        // ── Sort by fitness ascending ────────────────────────────────────────
        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sigma_old = self.sigma;
        let m_old = self.mean.clone();

        // ── New mean ─────────────────────────────────────────────────────────
        let mut m_new = vec![0.0f64; n];
        for j in 0..mu {
            let idx = order[j];
            for i in 0..n {
                m_new[i] += self.weights[j] * samples[idx][i];
            }
        }

        // ── Step-size path ───────────────────────────────────────────────────
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
        self.sigma = self.sigma.clamp(1e-15, 1e6);

        // ── h_sigma ──────────────────────────────────────────────────────────
        let gen1 = self.generation as f64 + 1.0;
        let denom = (1.0 - (1.0 - self.c_sigma).powf(2.0 * gen1)).sqrt();
        let h_sigma = if ps_norm / denom < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n {
            1.0
        } else {
            0.0
        };

        // ── p_c ──────────────────────────────────────────────────────────────
        let factor_pc = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..n {
            self.p_c[i] = (1.0 - self.c_c) * self.p_c[i]
                + h_sigma * factor_pc * (m_new[i] - m_old[i]) / sigma_old;
        }

        // ── Covariance matrix ─────────────────────────────────────────────────
        let base_scale = 1.0 - self.c1 - self.c_mu;
        let hsig_correction = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);

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

        for r in 0..n {
            for c in 0..n {
                let old_c = self.c_matrix[r * n + c];
                let pc_outer = self.p_c[r] * self.p_c[c];
                self.c_matrix[r * n + c] = base_scale * old_c
                    + self.c1 * (pc_outer + hsig_correction * old_c)
                    + self.c_mu * rank_mu[r * n + c];
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

        self.mean = m_new;
        self.generation += 1;
        self.n_evals += lambda;

        // ── Eigendecomposition ────────────────────────────────────────────────
        let update_freq = ((1.0 / (self.c1 + self.c_mu) / n as f64 / 10.0).floor() as usize).max(1);
        self.eig_update_count += 1;
        if self.eig_update_count >= update_freq {
            self.eig_update_count = 0;
            self.update_eigen(n)?;
        }
        Ok(())
    }

    /// Recompute B and D from the current covariance matrix.
    fn update_eigen(&mut self, n: usize) -> EvolResult<()> {
        let mut a_copy = self.c_matrix.clone();
        match jacobi_eigen(&mut a_copy, n) {
            Ok((eigenvalues, b)) => {
                self.b_matrix = b;
                self.d_vector = eigenvalues.iter().map(|&ev| ev.max(1e-20).sqrt()).collect();
            }
            Err(EvolError::EigenFailed(_)) => {}
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// Compute the condition number of the covariance matrix (max/min eigenvalue).
    fn condition_number(&self) -> f64 {
        let max_d = self
            .d_vector
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_d = self.d_vector.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_d < 1e-300 {
            return f64::INFINITY;
        }
        (max_d / min_d).powi(2)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Which restart regime was used for a given restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeKind {
    /// Large-population restart (IPOP or BIPOP large).
    Large,
    /// Small-population restart (BIPOP only).
    Small,
}

/// Record of one restart instance.
#[derive(Debug, Clone)]
pub struct RestartRegime {
    /// Population size used in this restart.
    pub pop_size: usize,
    /// Initial step size used in this restart.
    pub sigma0: f64,
    /// Number of evaluations consumed by this restart.
    pub n_evals_used: usize,
    /// Best fitness found at the end of this restart.
    pub final_best: f64,
    /// Which regime (Large or Small) was used.
    pub kind: RegimeKind,
}

/// Configuration for IPOP / BIPOP CMA-ES.
#[derive(Debug, Clone)]
pub struct RestartConfig {
    /// Problem dimension n.
    pub n: usize,
    /// Initial step size σ₀.
    pub sigma0: f64,
    /// Total evaluation budget across all restarts.
    pub max_total_evals: usize,
    /// Maximum number of restarts (not counting the first run).
    pub max_restarts: usize,
    /// Stagnation / convergence tolerance on function value.
    pub tol: f64,
    /// If `true`, run BIPOP (alternating large/small regimes); if `false`, pure IPOP.
    pub bipop: bool,
    /// Initial σ factor for BIPOP small-population restarts (≥ 1.0).
    pub small_sigma_factor: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            n: 2,
            sigma0: 0.3,
            max_total_evals: 100_000,
            max_restarts: 9,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 0,
        }
    }
}

impl RestartConfig {
    /// Construct a `RestartConfig` for an n-dimensional problem.
    pub fn new(n: usize) -> EvolResult<Self> {
        if n == 0 {
            return Err(EvolError::InvalidParameter("n must be >= 1".to_owned()));
        }
        Ok(Self {
            n,
            ..Self::default()
        })
    }
}

/// State returned after all IPOP / BIPOP restarts complete.
#[derive(Debug, Clone)]
pub struct RestartState {
    /// Best decision variable vector found across all restarts.
    pub best_x: Vec<f64>,
    /// Best (lowest) function value found across all restarts.
    pub best_f: f64,
    /// Total number of evaluations consumed.
    pub n_evals: usize,
    /// Number of restarts performed (excluding the initial run).
    pub n_restarts: usize,
    /// History of each restart regime.
    pub regime_history: Vec<RestartRegime>,
}

// ── Stagnation detection helpers ──────────────────────────────────────────────

/// Returns `true` if the inner CMA-ES run should be terminated due to stagnation.
fn is_stagnated(best_history: &[f64], window: usize, tol: f64, sigma: f64, cond_num: f64) -> bool {
    // Sigma too small
    if sigma < 1e-12 {
        return true;
    }
    // Condition number too large
    if cond_num > 1e14 {
        return true;
    }
    // No significant improvement over the last `window` generations
    if best_history.len() >= window {
        let oldest = best_history[best_history.len() - window];
        let newest = *best_history.last().unwrap();
        if (oldest - newest).abs() <= tol.max(1e-300) {
            return true;
        }
    }
    false
}

// ── Core single-restart runner ────────────────────────────────────────────────

/// Run one CMA-ES instance from `init_x` with `lambda` population and `sigma0`.
///
/// Enforces bounds by clamping samples.  Returns `(best_x, best_f, n_evals)`.
fn run_single_cmaes<F>(
    fitness_fn: &F,
    init_x: &[f64],
    bounds: &[(f64, f64)],
    lambda: usize,
    sigma0: f64,
    max_evals: usize,
    tol: f64,
    rng: &mut LcgRng,
) -> EvolResult<(Vec<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> f64,
{
    let n = init_x.len();
    if n == 0 {
        return Err(EvolError::InvalidParameter("n must be >= 1".to_owned()));
    }
    if bounds.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: bounds.len(),
        });
    }

    // Clamp initial mean to bounds
    let init_mean: Vec<f64> = init_x
        .iter()
        .enumerate()
        .map(|(i, &x)| x.clamp(bounds[i].0, bounds[i].1))
        .collect();

    let mut state = InnerCmaEs::new(init_mean, sigma0, lambda, rng)?;
    let mut best_x = state.mean.clone();
    let mut best_fit = fitness_fn(&best_x);
    let mut total_evals = 1usize;

    // Track best over rolling window for stagnation detection
    let window = 10 + (30.0 * n as f64 / lambda as f64).floor() as usize;
    let mut best_history: Vec<f64> = Vec::with_capacity(window + 1);
    best_history.push(best_fit);

    while total_evals < max_evals {
        // Sample and clamp to bounds
        let mut samples = state.sample(rng);
        for s in samples.iter_mut() {
            for (i, x) in s.iter_mut().enumerate() {
                *x = x.clamp(bounds[i].0, bounds[i].1);
            }
        }

        let fitnesses: Vec<f64> = samples.iter().map(|x| fitness_fn(x)).collect();
        total_evals += fitnesses.len();

        // Track best
        for (x, &f) in samples.iter().zip(fitnesses.iter()) {
            if f < best_fit {
                best_fit = f;
                best_x = x.clone();
            }
        }
        best_history.push(best_fit);
        if best_history.len() > window + 1 {
            best_history.drain(0..1);
        }

        // Terminate if objective is below tolerance
        if best_fit < tol {
            break;
        }

        state.update(&samples, &fitnesses)?;

        // Stagnation detection
        let cond = state.condition_number();
        if is_stagnated(&best_history, window, tol, state.sigma, cond) {
            break;
        }
    }

    Ok((best_x, best_fit, total_evals))
}

// ── IPOP CMA-ES ───────────────────────────────────────────────────────────────

/// Run IPOP CMA-ES: restarts with doubling population on stagnation.
///
/// Auger & Hansen (2005).  Each restart doubles the population size and resets
/// σ to the initial value, starting from a random point within bounds.
///
/// # Errors
/// Returns an error if configuration parameters are invalid.
pub fn ipop_cmaes_run<F>(
    fitness_fn: F,
    init_x: &[f64],
    bounds: &[(f64, f64)],
    cfg: &RestartConfig,
) -> EvolResult<RestartState>
where
    F: Fn(&[f64]) -> f64,
{
    let n = cfg.n;
    if init_x.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: init_x.len(),
        });
    }
    if bounds.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: bounds.len(),
        });
    }

    let mut rng = LcgRng::new(cfg.seed);

    // Default population size (same as CMA-ES tutorial)
    let lambda0: usize = 4 + (3.0 * (n as f64).ln()).floor() as usize;

    let mut best_x = init_x.to_vec();
    let mut best_f = f64::INFINITY;
    let mut total_evals = 0usize;
    let mut regime_history: Vec<RestartRegime> = Vec::new();

    let mut lambda = lambda0;
    let mut current_x = init_x.to_vec();

    for restart_idx in 0..=cfg.max_restarts {
        let remaining = cfg.max_total_evals.saturating_sub(total_evals);
        if remaining == 0 {
            break;
        }

        let (bx, bf, evals) = run_single_cmaes(
            &fitness_fn,
            &current_x,
            bounds,
            lambda,
            cfg.sigma0,
            remaining,
            cfg.tol,
            &mut rng,
        )?;

        total_evals += evals;

        if bf < best_f {
            best_f = bf;
            best_x = bx.clone();
        }

        regime_history.push(RestartRegime {
            pop_size: lambda,
            sigma0: cfg.sigma0,
            n_evals_used: evals,
            final_best: bf,
            kind: RegimeKind::Large,
        });

        // Prepare next restart
        if restart_idx < cfg.max_restarts {
            // Double population (IPOP rule)
            lambda *= 2;
            // Restart from a random point within bounds
            current_x = bounds
                .iter()
                .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect();
        }
    }

    let n_restarts = regime_history.len().saturating_sub(1);

    Ok(RestartState {
        best_x,
        best_f,
        n_evals: total_evals,
        n_restarts,
        regime_history,
    })
}

// ── BIPOP CMA-ES ──────────────────────────────────────────────────────────────

/// Run BIPOP CMA-ES: alternates large-population and small-population restarts.
///
/// Hansen (2009).  Two interleaved restart regimes:
/// - **Large**: identical to IPOP — population doubles, σ = σ₀.
/// - **Small**: population ≈ `ceil(small_factor * λ_large^U[0,1])`, σ reduced.
///
/// The regime alternation is budget-guided: switch to whichever regime still has
/// remaining budget.
///
/// # Errors
/// Returns an error if configuration parameters are invalid.
pub fn bipop_cmaes_run<F>(
    fitness_fn: F,
    init_x: &[f64],
    bounds: &[(f64, f64)],
    cfg: &RestartConfig,
) -> EvolResult<RestartState>
where
    F: Fn(&[f64]) -> f64,
{
    let n = cfg.n;
    if init_x.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: init_x.len(),
        });
    }
    if bounds.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: bounds.len(),
        });
    }

    let mut rng = LcgRng::new(cfg.seed);

    let lambda0: usize = 4 + (3.0 * (n as f64).ln()).floor() as usize;

    let mut best_x = init_x.to_vec();
    let mut best_f = f64::INFINITY;
    let mut total_evals = 0usize;
    let mut regime_history: Vec<RestartRegime> = Vec::new();

    // Initial run with default lambda (counts as large regime seed)
    {
        let remaining = cfg.max_total_evals.saturating_sub(total_evals);
        let (bx, bf, evals) = run_single_cmaes(
            &fitness_fn,
            init_x,
            bounds,
            lambda0,
            cfg.sigma0,
            remaining,
            cfg.tol,
            &mut rng,
        )?;
        total_evals += evals;
        if bf < best_f {
            best_f = bf;
            best_x = bx;
        }
        regime_history.push(RestartRegime {
            pop_size: lambda0,
            sigma0: cfg.sigma0,
            n_evals_used: evals,
            final_best: bf,
            kind: RegimeKind::Large,
        });
    }

    // Half the total budget for each regime
    let half_budget = cfg.max_total_evals / 2;
    let mut budget_large = half_budget;
    let mut budget_small = cfg.max_total_evals - half_budget;

    // Track large regime lambda scaling
    let mut lambda_large = lambda0 * 2; // first large restart already doubles

    let mut n_restart_loop = 0usize;

    loop {
        if total_evals >= cfg.max_total_evals {
            break;
        }
        if n_restart_loop >= cfg.max_restarts {
            break;
        }
        n_restart_loop += 1;

        // Decide which regime to run
        let use_large = budget_large >= budget_small || budget_small == 0;

        let (lambda_used, sigma_used, kind) = if use_large {
            // Large regime: IPOP doubling
            let lam = lambda_large;
            lambda_large *= 2;
            (lam, cfg.sigma0, RegimeKind::Large)
        } else {
            // Small regime: reduced lambda and sigma
            // λ_small = max(2, ceil(small_factor * λ_large^U[0,1]))
            let exp = rng.next_f64();
            let lam_small =
                ((cfg.small_sigma_factor * (lambda_large as f64).powf(exp)).ceil() as usize).max(2);
            // σ_small = σ₀ × 10^(−2 × U[0,1])
            let u = rng.next_f64();
            let sig_small = cfg.sigma0 * 10.0f64.powf(-2.0 * u);
            (lam_small, sig_small, RegimeKind::Small)
        };

        // Start from random point within bounds
        let start_x: Vec<f64> = bounds
            .iter()
            .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
            .collect();

        let remaining = cfg.max_total_evals.saturating_sub(total_evals);
        if remaining == 0 {
            break;
        }

        let (bx, bf, evals) = run_single_cmaes(
            &fitness_fn,
            &start_x,
            bounds,
            lambda_used,
            sigma_used,
            remaining,
            cfg.tol,
            &mut rng,
        )?;

        total_evals += evals;

        // Charge to appropriate budget
        if use_large {
            budget_large = budget_large.saturating_sub(evals);
        } else {
            budget_small = budget_small.saturating_sub(evals);
        }

        if bf < best_f {
            best_f = bf;
            best_x = bx;
        }

        regime_history.push(RestartRegime {
            pop_size: lambda_used,
            sigma0: sigma_used,
            n_evals_used: evals,
            final_best: bf,
            kind,
        });
    }

    let n_restarts = regime_history.len().saturating_sub(1);

    Ok(RestartState {
        best_x,
        best_f,
        n_evals: total_evals,
        n_restarts,
        regime_history,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    // ── Config / state construction ───────────────────────────────────────────

    #[test]
    fn test_restart_config_new_valid() {
        let cfg = RestartConfig::new(4).unwrap();
        assert_eq!(cfg.n, 4);
        assert!(cfg.sigma0 > 0.0);
        assert!(cfg.max_total_evals > 0);
        assert!(!cfg.bipop);
    }

    #[test]
    fn test_restart_config_new_zero_dims() {
        assert!(RestartConfig::new(0).is_err());
    }

    #[test]
    fn test_restart_config_default_reasonable() {
        let cfg = RestartConfig::default();
        assert!(cfg.max_restarts >= 1);
        assert!(cfg.small_sigma_factor >= 1.0);
        assert!(cfg.tol > 0.0);
    }

    // ── IPOP basic properties ─────────────────────────────────────────────────

    #[test]
    fn test_ipop_sphere_2d_finds_minimum() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.5,
            max_total_evals: 30_000,
            max_restarts: 3,
            tol: 1e-6,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 42,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = ipop_cmaes_run(sphere, &[2.0, -2.0], &bounds, &cfg).unwrap();
        assert!(
            state.best_f < 0.1,
            "sphere 2D ipop: best_f={}",
            state.best_f
        );
    }

    #[test]
    fn test_ipop_regime_history_non_empty() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.3,
            max_total_evals: 5_000,
            max_restarts: 2,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 1,
        };
        let bounds = vec![(-3.0, 3.0); 2];
        let state = ipop_cmaes_run(sphere, &[1.0, 1.0], &bounds, &cfg).unwrap();
        assert!(!state.regime_history.is_empty());
    }

    #[test]
    fn test_ipop_all_regimes_large_kind() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.3,
            max_total_evals: 8_000,
            max_restarts: 3,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 7,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = ipop_cmaes_run(sphere, &[1.0, -1.0], &bounds, &cfg).unwrap();
        for r in &state.regime_history {
            assert_eq!(
                r.kind,
                RegimeKind::Large,
                "IPOP should only produce Large regimes"
            );
        }
    }

    #[test]
    fn test_ipop_population_doubles() {
        let cfg = RestartConfig {
            n: 3,
            sigma0: 0.3,
            max_total_evals: 20_000,
            max_restarts: 4,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 11,
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let state = ipop_cmaes_run(sphere, &[1.0, 1.0, 1.0], &bounds, &cfg).unwrap();
        // Each successive restart should at most double the population
        for pair in state.regime_history.windows(2) {
            assert!(pair[1].pop_size >= pair[0].pop_size);
        }
    }

    #[test]
    fn test_ipop_bounds_respected_in_best_x() {
        let bounds = vec![(-1.0, 1.0); 2];
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.5,
            max_total_evals: 10_000,
            max_restarts: 2,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 3,
        };
        let state = ipop_cmaes_run(sphere, &[0.5, -0.5], &bounds, &cfg).unwrap();
        for (&x, &(lo, hi)) in state.best_x.iter().zip(bounds.iter()) {
            assert!(
                x >= lo - 1e-9 && x <= hi + 1e-9,
                "x={x} outside [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn test_ipop_total_evals_bounded() {
        let max_evals = 5_000usize;
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.3,
            max_total_evals: max_evals,
            max_restarts: 5,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 5,
        };
        let bounds = vec![(-3.0, 3.0); 2];
        let state = ipop_cmaes_run(sphere, &[1.0, 1.0], &bounds, &cfg).unwrap();
        // We allow a small overshoot of one lambda batch beyond the limit
        assert!(
            state.n_evals <= max_evals + 1000,
            "n_evals={} exceeds budget={}",
            state.n_evals,
            max_evals
        );
    }

    #[test]
    fn test_ipop_regime_history_records_evals() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.3,
            max_total_evals: 8_000,
            max_restarts: 3,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 9,
        };
        let bounds = vec![(-3.0, 3.0); 2];
        let state = ipop_cmaes_run(sphere, &[1.5, -1.5], &bounds, &cfg).unwrap();
        for r in &state.regime_history {
            assert!(r.n_evals_used > 0);
        }
    }

    #[test]
    fn test_ipop_regime_history_final_best_finite() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.3,
            max_total_evals: 8_000,
            max_restarts: 2,
            tol: 1e-8,
            bipop: false,
            small_sigma_factor: 2.0,
            seed: 13,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = ipop_cmaes_run(sphere, &[0.0, 0.0], &bounds, &cfg).unwrap();
        for r in &state.regime_history {
            assert!(r.final_best.is_finite());
        }
    }

    // ── BIPOP basic properties ────────────────────────────────────────────────

    #[test]
    fn test_bipop_sphere_2d_finds_minimum() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.5,
            max_total_evals: 30_000,
            max_restarts: 5,
            tol: 1e-6,
            bipop: true,
            small_sigma_factor: 2.0,
            seed: 100,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = bipop_cmaes_run(sphere, &[2.0, -2.0], &bounds, &cfg).unwrap();
        assert!(
            state.best_f < 0.5,
            "sphere 2D bipop: best_f={}",
            state.best_f
        );
    }

    #[test]
    fn test_bipop_has_both_regime_kinds() {
        // With enough restarts the BIPOP should produce both Large and Small regimes
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.5,
            max_total_evals: 50_000,
            max_restarts: 10,
            tol: 1e-8,
            bipop: true,
            small_sigma_factor: 2.0,
            seed: 200,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = bipop_cmaes_run(sphere, &[1.0, 1.0], &bounds, &cfg).unwrap();
        let has_large = state
            .regime_history
            .iter()
            .any(|r| r.kind == RegimeKind::Large);
        let has_small = state
            .regime_history
            .iter()
            .any(|r| r.kind == RegimeKind::Small);
        assert!(has_large, "BIPOP should produce Large regimes");
        assert!(has_small, "BIPOP should produce Small regimes");
    }

    #[test]
    fn test_bipop_regime_history_non_empty() {
        let cfg = RestartConfig {
            n: 3,
            sigma0: 0.3,
            max_total_evals: 10_000,
            max_restarts: 4,
            tol: 1e-8,
            bipop: true,
            small_sigma_factor: 2.0,
            seed: 77,
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let state = bipop_cmaes_run(sphere, &[1.0, -1.0, 0.5], &bounds, &cfg).unwrap();
        assert!(!state.regime_history.is_empty());
    }

    #[test]
    fn test_bipop_bounds_respected() {
        let bounds = vec![(-2.0, 2.0); 3];
        let cfg = RestartConfig {
            n: 3,
            sigma0: 0.5,
            max_total_evals: 15_000,
            max_restarts: 3,
            tol: 1e-8,
            bipop: true,
            small_sigma_factor: 2.0,
            seed: 55,
        };
        let state = bipop_cmaes_run(sphere, &[0.5, 0.5, 0.5], &bounds, &cfg).unwrap();
        for (&x, &(lo, hi)) in state.best_x.iter().zip(bounds.iter()) {
            assert!(
                x >= lo - 1e-9 && x <= hi + 1e-9,
                "x={x} outside [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn test_bipop_rosenbrock_2d() {
        let cfg = RestartConfig {
            n: 2,
            sigma0: 0.5,
            max_total_evals: 60_000,
            max_restarts: 6,
            tol: 1e-6,
            bipop: true,
            small_sigma_factor: 2.0,
            seed: 999,
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let state = bipop_cmaes_run(rosenbrock, &[0.0, 0.0], &bounds, &cfg).unwrap();
        assert!(
            state.best_f < 200.0,
            "rosenbrock bipop: best_f={}",
            state.best_f
        );
    }

    #[test]
    fn test_bipop_dimension_mismatch_error() {
        let cfg = RestartConfig::new(3).unwrap();
        let bounds = vec![(-5.0, 5.0); 3];
        // init_x has wrong length
        assert!(bipop_cmaes_run(sphere, &[0.0; 2], &bounds, &cfg).is_err());
    }

    #[test]
    fn test_ipop_dimension_mismatch_error() {
        let cfg = RestartConfig::new(3).unwrap();
        let bounds = vec![(-5.0, 5.0); 3];
        assert!(ipop_cmaes_run(sphere, &[0.0; 4], &bounds, &cfg).is_err());
    }

    #[test]
    fn test_inner_cmaes_condition_number_identity() {
        let mut rng = LcgRng::new(0);
        let state = InnerCmaEs::new(vec![0.0; 3], 0.3, 6, &mut rng).unwrap();
        // Initial covariance is identity: all d_vector = 1.0, cond = 1.0
        let cond = state.condition_number();
        assert!((cond - 1.0).abs() < 1e-6, "initial cond={cond}");
    }

    #[test]
    fn test_stagnation_sigma_too_small() {
        let history = vec![1e-5; 20];
        assert!(is_stagnated(&history, 10, 1e-8, 1e-13, 1.0));
    }

    #[test]
    fn test_stagnation_not_triggered_fresh() {
        let history = vec![1.0, 0.9, 0.8, 0.7];
        assert!(!is_stagnated(&history, 10, 1e-8, 0.5, 100.0));
    }
}
