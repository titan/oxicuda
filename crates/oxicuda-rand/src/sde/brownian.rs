//! Brownian motion and derived stochastic processes.
//!
//! Provides:
//! - [`BrownianMotion`] — standard Wiener process `dW = dW_t`.
//! - [`GeometricBrownianMotion`] — GBM `dS = μS dt + σS dW` (Black-Scholes model).
//! - [`OrnsteinUhlenbeck`] — mean-reverting OU `dX = θ(μ - X)dt + σ dW`.
//!
//! All three implement [`SdeProcess`] and can be passed directly to any solver.

use super::{PathMatrix, SdeConfig, SdeProcess, Xoshiro};
use crate::error::RandResult;

// ─── BrownianMotion ──────────────────────────────────────────────────────────

/// Standard Brownian motion (Wiener process).
///
/// Drift μ = 0, diffusion σ = 1. All paths start at `X_0 = x0`.
#[derive(Debug, Clone)]
pub struct BrownianMotion {
    n_paths: usize,
    x0: f64,
}

impl BrownianMotion {
    /// All paths start at `x0 = 0` (standard BM).
    #[must_use]
    pub fn standard(n_paths: usize) -> Self {
        Self { n_paths, x0: 0.0 }
    }

    /// All paths start at `x0`.
    #[must_use]
    pub fn with_start(n_paths: usize, x0: f64) -> Self {
        Self { n_paths, x0 }
    }
}

impl SdeProcess for BrownianMotion {
    fn drift(&self, _x: f64, _t: f64) -> f64 {
        0.0
    }
    fn diffusion(&self, _x: f64, _t: f64) -> f64 {
        1.0
    }
    fn diffusion_dx(&self, _x: f64, _t: f64) -> f64 {
        0.0
    }
    fn initial(&self, _path_idx: usize) -> f64 {
        self.x0
    }
    fn n_paths(&self) -> usize {
        self.n_paths
    }
}

/// Result of a Brownian motion simulation.
#[derive(Debug, Clone)]
pub struct BrownianPathResult {
    /// Simulated path values with shape `(n_steps + 1, n_paths)`.
    pub paths: PathMatrix,
    /// Time grid `[t0, t0 + dt, ..., t1]` with `n_steps + 1` entries.
    pub time_grid: Vec<f64>,
    /// Brownian increments `ΔW_t = W_{t+1} - W_t` for each step and path.
    pub increments: PathMatrix,
}

impl BrownianPathResult {
    /// Covariance E[W_s * W_t] = min(s, t) for standard BM.
    /// Approximated as sample covariance between two time indices.
    #[must_use]
    pub fn sample_covariance(&self, step_s: usize, step_t: usize) -> f64 {
        let xs = self.paths.step_slice(step_s);
        let xt = self.paths.step_slice(step_t);
        let n = xs.len() as f64;
        let mean_s = xs.iter().sum::<f64>() / n;
        let mean_t = xt.iter().sum::<f64>() / n;
        xs.iter()
            .zip(xt.iter())
            .map(|(&s, &t)| (s - mean_s) * (t - mean_t))
            .sum::<f64>()
            / n
    }

    /// Theoretical covariance E[W_s W_t] = min(t_s, t_t) for standard BM.
    #[must_use]
    pub fn theoretical_covariance(&self, step_s: usize, step_t: usize) -> f64 {
        let ts = self.time_grid[step_s];
        let tt = self.time_grid[step_t];
        ts.min(tt)
    }
}

/// Simulate standard Brownian motion using direct path construction.
///
/// `W_0 = x0`, `W_{t+Δ} = W_t + N(0, Δt)`.
pub fn simulate_brownian(cfg: SdeConfig) -> RandResult<BrownianPathResult> {
    cfg.validate()?;
    let dt = cfg.dt();
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro::new(cfg.seed);
    let mut paths = PathMatrix::new(cfg.n_steps, cfg.n_paths, 0.0);
    let mut increments = PathMatrix::new(cfg.n_steps, cfg.n_paths, 0.0);

    // Initial condition: W_0 = 0
    for p in 0..cfg.n_paths {
        paths.set(0, p, 0.0);
    }

    for step in 0..cfg.n_steps {
        for p in 0..cfg.n_paths {
            let z = rng.next_normal() * sqrt_dt;
            increments.set(step, p, z);
            let prev = paths.get(step, p);
            paths.set(step + 1, p, prev + z);
        }
    }

    let time_grid = cfg.time_grid();
    Ok(BrownianPathResult {
        paths,
        time_grid,
        increments,
    })
}

// ─── GeometricBrownianMotion ─────────────────────────────────────────────────

/// Geometric Brownian Motion: `dS = μ S dt + σ S dW`.
///
/// Models log-normally distributed asset prices (Black-Scholes framework).
/// Exact solution: `S_t = S_0 * exp((μ - σ²/2)t + σ W_t)`.
#[derive(Debug, Clone)]
pub struct GeometricBrownianMotion {
    /// Drift rate (annualised return) μ.
    pub mu: f64,
    /// Volatility σ.
    pub sigma: f64,
    /// Initial prices for each path.
    pub s0: Vec<f64>,
}

impl GeometricBrownianMotion {
    /// All paths start at `s0` with the same initial price.
    #[must_use]
    pub fn new(n_paths: usize, mu: f64, sigma: f64, s0: f64) -> Self {
        Self {
            mu,
            sigma,
            s0: vec![s0; n_paths],
        }
    }

    /// Each path has a distinct initial price.
    #[must_use]
    pub fn with_starts(mu: f64, sigma: f64, s0: Vec<f64>) -> Self {
        Self { mu, sigma, s0 }
    }

    /// Exact GBM path simulation via `S_t = S_0 * exp((μ - σ²/2)t + σ W_t)`.
    ///
    /// More accurate than Euler-Maruyama for GBM since it uses the closed-form
    /// solution.
    pub fn simulate_exact(&self, cfg: SdeConfig) -> RandResult<PathMatrix> {
        cfg.validate()?;
        let dt = cfg.dt();
        let sqrt_dt = dt.sqrt();
        let drift_adj = (self.mu - 0.5 * self.sigma * self.sigma) * dt;
        let mut rng = Xoshiro::new(cfg.seed);
        let mut paths = PathMatrix::new(cfg.n_steps, cfg.n_paths, 0.0);

        for p in 0..cfg.n_paths {
            paths.set(0, p, *self.s0.get(p).unwrap_or(&1.0));
        }

        for step in 0..cfg.n_steps {
            for p in 0..cfg.n_paths {
                let z = rng.next_normal() * sqrt_dt;
                let prev = paths.get(step, p);
                let next = prev * (drift_adj + self.sigma * z).exp();
                paths.set(step + 1, p, next);
            }
        }
        Ok(paths)
    }

    /// Expected price at time t: `E[S_t] = S_0 * exp(μ * t)`.
    #[must_use]
    pub fn expected_price(&self, s0: f64, t: f64) -> f64 {
        s0 * (self.mu * t).exp()
    }

    /// Variance of price: `Var[S_t] = S_0² * exp(2μt) * (exp(σ²t) - 1)`.
    #[must_use]
    pub fn price_variance(&self, s0: f64, t: f64) -> f64 {
        let e2mu = (2.0 * self.mu * t).exp();
        let esig2 = (self.sigma * self.sigma * t).exp();
        s0 * s0 * e2mu * (esig2 - 1.0)
    }
}

impl SdeProcess for GeometricBrownianMotion {
    fn drift(&self, x: f64, _t: f64) -> f64 {
        self.mu * x
    }
    fn diffusion(&self, x: f64, _t: f64) -> f64 {
        self.sigma * x
    }
    fn diffusion_dx(&self, _x: f64, _t: f64) -> f64 {
        self.sigma
    }
    fn initial(&self, path_idx: usize) -> f64 {
        *self.s0.get(path_idx).unwrap_or(&1.0)
    }
    fn n_paths(&self) -> usize {
        self.s0.len()
    }
}

// ─── OrnsteinUhlenbeck ────────────────────────────────────────────────────────

/// Ornstein-Uhlenbeck process: `dX = θ(μ - X)dt + σ dW`.
///
/// Mean-reverting process used in finance (Vasicek interest rate model),
/// physics (Langevin equation), and as a noise process for RL exploration.
///
/// Exact stationary distribution: `N(μ, σ²/(2θ))`.
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck {
    /// Mean-reversion speed θ (> 0).
    pub theta: f64,
    /// Long-run mean μ.
    pub mu: f64,
    /// Volatility σ.
    pub sigma: f64,
    /// Initial values (one per path).
    pub x0: Vec<f64>,
}

impl OrnsteinUhlenbeck {
    /// All paths start at `x0 = mu` (stationary initialization).
    #[must_use]
    pub fn new(n_paths: usize, theta: f64, mu: f64, sigma: f64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            x0: vec![mu; n_paths],
        }
    }

    /// Specify different starting values per path.
    #[must_use]
    pub fn with_starts(theta: f64, mu: f64, sigma: f64, x0: Vec<f64>) -> Self {
        Self {
            theta,
            mu,
            sigma,
            x0,
        }
    }

    /// Exact simulation using the conditional distribution:
    /// `X_{t+Δ} | X_t ~ N(μ + (X_t - μ)e^{-θΔ}, σ²(1 - e^{-2θΔ})/(2θ))`.
    pub fn simulate_exact(&self, cfg: SdeConfig) -> RandResult<PathMatrix> {
        cfg.validate()?;
        let dt = cfg.dt();
        let exp_neg_theta_dt = (-self.theta * dt).exp();
        let var = self.sigma * self.sigma * (1.0 - exp_neg_theta_dt * exp_neg_theta_dt)
            / (2.0 * self.theta);
        let std_dev = var.sqrt();
        let mut rng = Xoshiro::new(cfg.seed);
        let mut paths = PathMatrix::new(cfg.n_steps, cfg.n_paths, 0.0);

        for p in 0..cfg.n_paths {
            paths.set(0, p, *self.x0.get(p).unwrap_or(&self.mu));
        }

        for step in 0..cfg.n_steps {
            for p in 0..cfg.n_paths {
                let z = rng.next_normal();
                let prev = paths.get(step, p);
                let cond_mean = self.mu + (prev - self.mu) * exp_neg_theta_dt;
                let next = cond_mean + std_dev * z;
                paths.set(step + 1, p, next);
            }
        }
        Ok(paths)
    }

    /// Stationary mean: μ.
    #[must_use]
    pub fn stationary_mean(&self) -> f64 {
        self.mu
    }

    /// Stationary variance: σ²/(2θ).
    #[must_use]
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta)
    }

    /// Conditional mean E[X_t | X_0 = x0]: `μ + (x0 - μ) * e^{-θt}`.
    #[must_use]
    pub fn conditional_mean(&self, x0: f64, t: f64) -> f64 {
        self.mu + (x0 - self.mu) * (-self.theta * t).exp()
    }

    /// Conditional variance Var[X_t | X_0]: `σ²(1 - e^{-2θt}) / (2θ)`.
    #[must_use]
    pub fn conditional_variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma * (1.0 - (-2.0 * self.theta * t).exp()) / (2.0 * self.theta)
    }
}

impl SdeProcess for OrnsteinUhlenbeck {
    fn drift(&self, x: f64, _t: f64) -> f64 {
        self.theta * (self.mu - x)
    }
    fn diffusion(&self, _x: f64, _t: f64) -> f64 {
        self.sigma
    }
    fn diffusion_dx(&self, _x: f64, _t: f64) -> f64 {
        0.0
    }
    fn initial(&self, path_idx: usize) -> f64 {
        *self.x0.get(path_idx).unwrap_or(&self.mu)
    }
    fn n_paths(&self) -> usize {
        self.x0.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(n_paths: usize, n_steps: usize) -> SdeConfig {
        SdeConfig::new(0.0, 1.0, n_steps, n_paths, 42)
    }

    // ─── BrownianMotion ──────────────────────────────────────────────────────

    #[test]
    fn brownian_starts_at_zero() {
        let cfg = make_cfg(8, 100);
        let res = simulate_brownian(cfg).expect("BM simulation with valid config should not fail");
        for p in 0..8 {
            assert!((res.paths.get(0, p)).abs() < 1e-12, "W_0 != 0 for path {p}");
        }
    }

    #[test]
    fn brownian_shape() {
        let cfg = make_cfg(4, 50);
        let res = simulate_brownian(cfg).expect("BM simulation with valid config should not fail");
        assert_eq!(res.paths.shape(), (51, 4));
    }

    #[test]
    fn brownian_increments_are_finite() {
        let cfg = make_cfg(4, 100);
        let res = simulate_brownian(cfg).expect("BM simulation with valid config should not fail");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn brownian_mean_near_zero() {
        // E[W_t] = 0 for all t
        let cfg = SdeConfig::new(0.0, 1.0, 100, 10_000, 1);
        let res = simulate_brownian(cfg).expect("BM simulation with valid config should not fail");
        let means = res.paths.path_mean();
        // Final mean should be near 0 (within 3σ/√n)
        let final_mean = means[100];
        assert!(
            final_mean.abs() < 0.05,
            "final mean {final_mean} too far from 0"
        );
    }

    #[test]
    fn brownian_covariance_min_s_t() {
        // E[W_s W_t] ≈ min(s, t) for large n_paths
        let cfg = SdeConfig::new(0.0, 2.0, 200, 50_000, 7);
        let res = simulate_brownian(cfg).expect("BM simulation with valid config should not fail");
        // Check E[W_{0.5} W_{1.0}] ≈ 0.5
        let step_05 = 50; // t=0.5
        let step_10 = 100; // t=1.0
        let sample_cov = res.sample_covariance(step_05, step_10);
        let theory_cov = res.theoretical_covariance(step_05, step_10);
        assert!(
            (sample_cov - theory_cov).abs() < 0.05,
            "cov {sample_cov:.4} vs theory {theory_cov:.4}"
        );
    }

    #[test]
    fn brownian_bm_sde_process_impl() {
        let bm = BrownianMotion::standard(3);
        assert_eq!(bm.n_paths(), 3);
        assert!((bm.drift(1.0, 0.5)).abs() < 1e-12);
        assert!((bm.diffusion(1.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((bm.initial(0)).abs() < 1e-12);
    }

    // ─── GeometricBrownianMotion ─────────────────────────────────────────────

    #[test]
    fn gbm_starts_at_s0() {
        let gbm = GeometricBrownianMotion::new(4, 0.05, 0.2, 100.0);
        let cfg = make_cfg(4, 100);
        let paths = gbm
            .simulate_exact(cfg)
            .expect("GBM exact simulation with valid config should succeed");
        for p in 0..4 {
            assert!((paths.get(0, p) - 100.0).abs() < 1e-10, "S_0 != 100");
        }
    }

    #[test]
    fn gbm_prices_positive() {
        let gbm = GeometricBrownianMotion::new(8, 0.1, 0.3, 50.0);
        let cfg = make_cfg(8, 200);
        let paths = gbm
            .simulate_exact(cfg)
            .expect("GBM exact simulation with valid config should succeed");
        for &v in paths.as_slice() {
            assert!(v > 0.0, "GBM price must be positive");
        }
    }

    #[test]
    fn gbm_expected_mean() {
        // E[S_T] = S_0 * exp(μ T)
        let mu = 0.1_f64;
        let sigma = 0.2;
        let s0 = 100.0;
        let t = 1.0;
        let gbm = GeometricBrownianMotion::new(20_000, mu, sigma, s0);
        let cfg = SdeConfig::new(0.0, t, 252, 20_000, 3);
        let paths = gbm
            .simulate_exact(cfg)
            .expect("GBM exact simulation with valid config should succeed");
        let final_mean = paths.final_state().iter().sum::<f64>() / 20_000.0;
        let expected = gbm.expected_price(s0, t);
        assert!(
            (final_mean - expected).abs() / expected < 0.02,
            "mean {final_mean:.2} vs expected {expected:.2}"
        );
    }

    #[test]
    fn gbm_sde_process_impl() {
        let gbm = GeometricBrownianMotion::new(2, 0.05, 0.2, 1.0);
        assert_eq!(gbm.n_paths(), 2);
        assert!((gbm.drift(2.0, 0.0) - 0.1).abs() < 1e-12); // 0.05 * 2 = 0.1
        assert!((gbm.diffusion(3.0, 0.0) - 0.6).abs() < 1e-12); // 0.2 * 3 = 0.6
    }

    // ─── OrnsteinUhlenbeck ────────────────────────────────────────────────────

    #[test]
    fn ou_stationary_variance() {
        let ou = OrnsteinUhlenbeck::new(2, 2.0, 0.0, 1.0);
        // σ²/(2θ) = 1/(4) = 0.25
        assert!((ou.stationary_variance() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn ou_conditional_mean_reverts_to_mu() {
        let ou = OrnsteinUhlenbeck::new(1, 1.0, 3.0, 0.5);
        // As t → ∞, conditional_mean(x0, t) → mu for any x0
        let cond_mean = ou.conditional_mean(10.0, 100.0);
        assert!(
            (cond_mean - 3.0).abs() < 1e-6,
            "mean did not revert: {cond_mean}"
        );
    }

    #[test]
    fn ou_exact_sim_shape() {
        let ou = OrnsteinUhlenbeck::new(5, 0.5, 0.0, 1.0);
        let cfg = make_cfg(5, 100);
        let paths = ou
            .simulate_exact(cfg)
            .expect("OU exact simulation with valid config should succeed");
        assert_eq!(paths.shape(), (101, 5));
    }

    #[test]
    fn ou_exact_starts_at_mu() {
        let ou = OrnsteinUhlenbeck::new(3, 1.0, 2.5, 0.5);
        let cfg = make_cfg(3, 50);
        let paths = ou
            .simulate_exact(cfg)
            .expect("OU exact simulation with valid config should succeed");
        for p in 0..3 {
            assert!((paths.get(0, p) - 2.5).abs() < 1e-12, "start != mu");
        }
    }

    #[test]
    fn ou_stationary_mean_recovered() {
        // Long simulation, many paths: final mean ≈ μ
        let mu = 5.0_f64;
        let ou = OrnsteinUhlenbeck::new(10_000, 2.0, mu, 1.0);
        let cfg = SdeConfig::new(0.0, 10.0, 1000, 10_000, 11);
        let paths = ou
            .simulate_exact(cfg)
            .expect("OU exact simulation with valid config should succeed");
        let final_mean = paths.final_state().iter().sum::<f64>() / 10_000.0;
        assert!(
            (final_mean - mu).abs() < 0.1,
            "stationary mean {final_mean:.3} != {mu}"
        );
    }

    #[test]
    fn ou_sde_process_impl() {
        let ou = OrnsteinUhlenbeck::new(2, 1.5, 2.0, 0.5);
        // drift = θ(μ - x) = 1.5*(2-3) = -1.5
        assert!((ou.drift(3.0, 0.0) - (-1.5)).abs() < 1e-12);
        // diffusion = σ = 0.5
        assert!((ou.diffusion(3.0, 0.0) - 0.5).abs() < 1e-12);
        // diffusion_dx = 0 (additive noise)
        assert!(ou.diffusion_dx(3.0, 0.0).abs() < 1e-12);
    }

    #[test]
    fn ou_finite_paths() {
        let ou = OrnsteinUhlenbeck::new(4, 1.0, 0.0, 0.3);
        let cfg = make_cfg(4, 200);
        let paths = ou
            .simulate_exact(cfg)
            .expect("OU exact simulation with valid config should succeed");
        for &v in paths.as_slice() {
            assert!(v.is_finite(), "OU path value not finite: {v}");
        }
    }
}
