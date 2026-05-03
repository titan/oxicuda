//! Euler-Maruyama numerical scheme for Itô SDEs.
//!
//! The Euler-Maruyama method discretises `dX = μ(X,t)dt + σ(X,t)dW` as:
//! ```text
//! X_{t+Δ} = X_t + μ(X_t, t) Δt + σ(X_t, t) √Δt Z,   Z ~ N(0,1)
//! ```
//!
//! **Strong order**: 0.5 — the mean-square error E|X_T - X̂_T|² = O(Δt).
//! **Weak order**: 1.0 — distribution of X̂_T converges to distribution of X_T at O(Δt).
//!
//! This is the simplest and most general method, suitable for any Itô SDE.
//! For additive noise (σ independent of X), it achieves strong order 1.0.

use super::{PathMatrix, SdeConfig, SdeProcess, Xoshiro};
use crate::error::{RandError, RandResult};

/// Result of an Euler-Maruyama simulation.
#[derive(Debug, Clone)]
pub struct EulerMaruyamaResult {
    /// Path values: `paths[(step, path)]`.
    pub paths: PathMatrix,
    /// Time grid `[t0, t0+dt, ..., t1]`.
    pub time_grid: Vec<f64>,
    /// Step size used.
    pub dt: f64,
    /// Number of paths simulated.
    pub n_paths: usize,
}

impl EulerMaruyamaResult {
    /// Sample mean across all paths at the final time `t1`.
    #[must_use]
    pub fn final_mean(&self) -> f64 {
        let fs = self.paths.final_state();
        fs.iter().sum::<f64>() / fs.len() as f64
    }

    /// Sample standard deviation across all paths at the final time `t1`.
    #[must_use]
    pub fn final_std(&self) -> f64 {
        let fs = self.paths.final_state();
        let n = fs.len() as f64;
        let mean = fs.iter().sum::<f64>() / n;
        let var = fs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Sample mean at each time step.
    #[must_use]
    pub fn path_mean(&self) -> Vec<f64> {
        self.paths.path_mean()
    }

    /// Sample variance at each time step.
    #[must_use]
    pub fn path_variance(&self) -> Vec<f64> {
        self.paths.path_variance()
    }
}

/// Euler-Maruyama solver for Itô SDEs.
///
/// Applies the EM discretisation to any type implementing [`SdeProcess`].
#[derive(Debug, Clone)]
pub struct EulerMaruyama {
    cfg: SdeConfig,
}

impl EulerMaruyama {
    /// Create a solver with the given configuration.
    pub fn new(cfg: SdeConfig) -> Self {
        Self { cfg }
    }

    /// Solve the SDE over `[t0, t1]` and return all path values.
    ///
    /// The path matrix has shape `(n_steps + 1, n_paths)`.
    pub fn solve<P: SdeProcess>(&self, process: &P) -> RandResult<EulerMaruyamaResult> {
        self.cfg.validate()?;
        if process.n_paths() != self.cfg.n_paths {
            return Err(RandError::InvalidParameter(format!(
                "process has {} paths but config specifies {}",
                process.n_paths(),
                self.cfg.n_paths
            )));
        }

        let dt = self.cfg.dt();
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro::new(self.cfg.seed);
        let mut paths = PathMatrix::new(self.cfg.n_steps, self.cfg.n_paths, 0.0);

        // Set initial conditions
        for p in 0..self.cfg.n_paths {
            paths.set(0, p, process.initial(p));
        }

        // Time stepping
        for step in 0..self.cfg.n_steps {
            let t = self.cfg.t0 + step as f64 * dt;
            for p in 0..self.cfg.n_paths {
                let x = paths.get(step, p);
                let mu = process.drift(x, t);
                let sigma = process.diffusion(x, t);
                let z = rng.next_normal();
                let x_next = x + mu * dt + sigma * sqrt_dt * z;
                paths.set(step + 1, p, x_next);
            }
        }

        Ok(EulerMaruyamaResult {
            paths,
            time_grid: self.cfg.time_grid(),
            dt,
            n_paths: self.cfg.n_paths,
        })
    }

    /// Solve and return only the final states `X_{t1}` for all paths.
    pub fn solve_terminal<P: SdeProcess>(&self, process: &P) -> RandResult<Vec<f64>> {
        self.cfg.validate()?;
        if process.n_paths() != self.cfg.n_paths {
            return Err(RandError::InvalidParameter(format!(
                "process has {} paths but config specifies {}",
                process.n_paths(),
                self.cfg.n_paths
            )));
        }

        let dt = self.cfg.dt();
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro::new(self.cfg.seed);
        let mut x: Vec<f64> = (0..self.cfg.n_paths).map(|p| process.initial(p)).collect();

        for step in 0..self.cfg.n_steps {
            let t = self.cfg.t0 + step as f64 * dt;
            for x_val in &mut x {
                let mu = process.drift(*x_val, t);
                let sigma = process.diffusion(*x_val, t);
                let z = rng.next_normal();
                *x_val += mu * dt + sigma * sqrt_dt * z;
            }
        }
        Ok(x)
    }

    /// Estimate the strong error against a reference path matrix (same seed).
    /// Returns E[|X̂_T - X_ref_T|].
    pub fn strong_error(approx: &[f64], reference: &[f64]) -> RandResult<f64> {
        if approx.len() != reference.len() {
            return Err(RandError::InvalidParameter(format!(
                "approx len {} != reference len {}",
                approx.len(),
                reference.len()
            )));
        }
        let n = approx.len() as f64;
        let err = approx
            .iter()
            .zip(reference.iter())
            .map(|(&a, &r)| (a - r).abs())
            .sum::<f64>()
            / n;
        Ok(err)
    }
}

// ─── Convenience functions ────────────────────────────────────────────────────

/// Simulate geometric Brownian motion via Euler-Maruyama.
///
/// For comparison with [`super::brownian::GeometricBrownianMotion::simulate_exact`].
pub fn em_gbm(mu: f64, sigma: f64, s0: f64, cfg: SdeConfig) -> RandResult<EulerMaruyamaResult> {
    use super::brownian::GeometricBrownianMotion;
    let gbm = GeometricBrownianMotion::new(cfg.n_paths, mu, sigma, s0);
    EulerMaruyama::new(cfg).solve(&gbm)
}

/// Simulate Ornstein-Uhlenbeck process via Euler-Maruyama.
pub fn em_ou(theta: f64, mu: f64, sigma: f64, cfg: SdeConfig) -> RandResult<EulerMaruyamaResult> {
    use super::brownian::OrnsteinUhlenbeck;
    let ou = OrnsteinUhlenbeck::new(cfg.n_paths, theta, mu, sigma);
    EulerMaruyama::new(cfg).solve(&ou)
}

/// Simulate standard Brownian motion via Euler-Maruyama.
pub fn em_brownian(cfg: SdeConfig) -> RandResult<EulerMaruyamaResult> {
    use super::brownian::BrownianMotion;
    let bm = BrownianMotion::standard(cfg.n_paths);
    EulerMaruyama::new(cfg).solve(&bm)
}

#[cfg(test)]
mod tests {
    use super::super::brownian::{BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck};
    use super::*;

    fn make_cfg(n_paths: usize, n_steps: usize) -> SdeConfig {
        SdeConfig::new(0.0, 1.0, n_steps, n_paths, 42)
    }

    #[test]
    fn em_brownian_shape() {
        let cfg = make_cfg(4, 100);
        let res = em_brownian(cfg).expect("valid BM config should simulate without error");
        assert_eq!(res.paths.shape(), (101, 4));
        assert_eq!(res.n_paths, 4);
    }

    #[test]
    fn em_brownian_starts_correct() {
        let cfg = make_cfg(3, 50);
        let bm = BrownianMotion::with_start(3, 2.0);
        let res = EulerMaruyama::new(cfg)
            .solve(&bm)
            .expect("BM solve with matching n_paths should succeed");
        for p in 0..3 {
            assert!((res.paths.get(0, p) - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn em_brownian_mean_near_zero() {
        let cfg = SdeConfig::new(0.0, 1.0, 100, 20_000, 1);
        let res = em_brownian(cfg).expect("BM simulation with valid config should not fail");
        let mean = res.final_mean();
        assert!(mean.abs() < 0.05, "BM mean {mean:.4} too far from 0");
    }

    #[test]
    fn em_brownian_variance_near_t() {
        // Var[W_1] = 1
        let cfg = SdeConfig::new(0.0, 1.0, 200, 50_000, 2);
        let res = em_brownian(cfg).expect("BM simulation with valid config should not fail");
        let var = res.path_variance();
        let final_var = var[200];
        assert!((final_var - 1.0).abs() < 0.05, "var {final_var:.4} != 1.0");
    }

    #[test]
    fn em_gbm_positive_paths() {
        let cfg = make_cfg(8, 200);
        let res =
            em_gbm(0.05, 0.2, 100.0, cfg).expect("GBM simulation with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite() && v > 0.0, "GBM path must be positive: {v}");
        }
    }

    #[test]
    fn em_gbm_mean_close_to_exact() {
        // E[S_T] = S_0 * exp(μ T)
        let mu = 0.1_f64;
        let s0 = 100.0_f64;
        let t = 1.0;
        let cfg = SdeConfig::new(0.0, t, 500, 30_000, 3);
        let res =
            em_gbm(mu, 0.2, s0, cfg).expect("GBM simulation with valid params should succeed");
        let sample_mean = res.final_mean();
        let expected = s0 * (mu * t).exp();
        assert!(
            (sample_mean - expected).abs() / expected < 0.03,
            "GBM mean {sample_mean:.2} vs {expected:.2}"
        );
    }

    #[test]
    fn em_ou_mean_reverts() {
        // Long OU simulation should have final mean ≈ μ
        let mu = 3.0_f64;
        let cfg = SdeConfig::new(0.0, 10.0, 1000, 10_000, 4);
        let res = em_ou(2.0, mu, 0.5, cfg).expect("OU simulation with valid params should succeed");
        let final_mean = res.final_mean();
        assert!(
            (final_mean - mu).abs() < 0.1,
            "OU mean {final_mean:.3} != {mu}"
        );
    }

    #[test]
    fn em_path_n_mismatch_err() {
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(5); // mismatch: 5 != 4
        let result = EulerMaruyama::new(cfg).solve(&bm);
        assert!(result.is_err());
    }

    #[test]
    fn em_terminal_matches_full_final() {
        // terminal-only solve should give same final states as full solve (same seed)
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(4);
        let full = EulerMaruyama::new(cfg)
            .solve(&bm)
            .expect("full BM solve should succeed");
        let terminal = EulerMaruyama::new(cfg)
            .solve_terminal(&bm)
            .expect("terminal BM solve should succeed");
        let full_final = full.paths.final_state();
        for p in 0..4 {
            assert!(
                (full_final[p] - terminal[p]).abs() < 1e-12,
                "terminal mismatch at path {p}"
            );
        }
    }

    #[test]
    fn em_solve_all_finite() {
        let cfg = make_cfg(5, 50);
        let gbm = GeometricBrownianMotion::new(5, 0.08, 0.25, 50.0);
        let res = EulerMaruyama::new(cfg)
            .solve(&gbm)
            .expect("GBM solve with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn em_ou_finite() {
        let ou = OrnsteinUhlenbeck::new(4, 1.0, 0.0, 0.3);
        let cfg = make_cfg(4, 200);
        let res = EulerMaruyama::new(cfg)
            .solve(&ou)
            .expect("OU solve with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn em_strong_error_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let err = EulerMaruyama::strong_error(&a, &b)
            .expect("strong_error with equal-length slices should succeed");
        assert!(err.abs() < 1e-12);
    }

    #[test]
    fn em_strong_error_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 2.0];
        let err = EulerMaruyama::strong_error(&a, &b)
            .expect("strong_error with equal-length slices should succeed");
        assert!((err - 1.0).abs() < 1e-12);
    }

    #[test]
    fn em_strong_error_mismatch_err() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0];
        assert!(EulerMaruyama::strong_error(&a, &b).is_err());
    }
}
