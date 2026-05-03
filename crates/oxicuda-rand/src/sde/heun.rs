//! Stratonovich-Heun scheme for SDEs (predictor-corrector, strong order 1.0).
//!
//! The Heun method (also called the trapezoid method or improved Euler) is a
//! second-order predictor-corrector scheme for Stratonovich SDEs:
//!
//! ```text
//! dX = f(X,t) dt + g(X,t) ∘ dW    (Stratonovich)
//! ```
//!
//! **Algorithm** (one step from t to t+Δ, ΔW = √Δt · Z):
//! 1. Predictor:   X̃ = X_t + f(X_t, t)Δt + g(X_t, t)ΔW
//! 2. Corrector:   X_{t+Δ} = X_t + ½[f(X_t,t)+f(X̃,t+Δ)]Δt + ½[g(X_t,t)+g(X̃,t+Δ)]ΔW
//!
//! **Strong order**: 1.0 (in Stratonovich sense).
//!
//! ## Itô vs Stratonovich
//!
//! For an Itô SDE `dX = μ dt + σ dW`, the Stratonovich equivalent is:
//! `dX = (μ - ½σσ')dt + σ ∘ dW`. Thus when applying Heun to Itô processes,
//! pass the Stratonovich-corrected drift `f(x,t) = μ(x,t) - ½σ(x,t)σ'(x,t)`.
//!
//! This module provides both the raw Heun solver and an Itô-adapted version
//! that automatically performs the Stratonovich correction.

use super::{PathMatrix, SdeConfig, SdeProcess, Xoshiro};
use crate::error::{RandError, RandResult};

/// Result of a Stratonovich-Heun simulation.
#[derive(Debug, Clone)]
pub struct HeunResult {
    /// Path values with shape `(n_steps + 1, n_paths)`.
    pub paths: PathMatrix,
    /// Time grid `[t0, ..., t1]`.
    pub time_grid: Vec<f64>,
    /// Step size used.
    pub dt: f64,
    /// Number of paths.
    pub n_paths: usize,
}

impl HeunResult {
    /// Final-time mean across all paths.
    #[must_use]
    pub fn final_mean(&self) -> f64 {
        let fs = self.paths.final_state();
        fs.iter().sum::<f64>() / fs.len() as f64
    }

    /// Final-time standard deviation across all paths.
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

/// Stratonovich-Heun solver.
///
/// Treats the SDE as a **Stratonovich** equation. When the underlying
/// process is specified in Itô form (via [`SdeProcess`]), call
/// [`StratonovichHeun::solve_ito`] which applies the Itô-to-Stratonovich
/// conversion automatically.
#[derive(Debug, Clone)]
pub struct StratonovichHeun {
    cfg: SdeConfig,
}

impl StratonovichHeun {
    /// Create a Heun solver.
    pub fn new(cfg: SdeConfig) -> Self {
        Self { cfg }
    }

    /// Solve a **Stratonovich** SDE using the Heun predictor-corrector.
    ///
    /// Drift `f` and diffusion `g` are interpreted in the Stratonovich sense.
    /// For Itô processes use [`Self::solve_ito`].
    pub fn solve<P: SdeProcess>(&self, process: &P) -> RandResult<HeunResult> {
        self.cfg.validate()?;
        if process.n_paths() != self.cfg.n_paths {
            return Err(RandError::InvalidParameter(format!(
                "process n_paths {} != config n_paths {}",
                process.n_paths(),
                self.cfg.n_paths
            )));
        }

        let dt = self.cfg.dt();
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro::new(self.cfg.seed);
        let mut paths = PathMatrix::new(self.cfg.n_steps, self.cfg.n_paths, 0.0);

        for p in 0..self.cfg.n_paths {
            paths.set(0, p, process.initial(p));
        }

        for step in 0..self.cfg.n_steps {
            let t = self.cfg.t0 + step as f64 * dt;
            let t_next = t + dt;
            for p in 0..self.cfg.n_paths {
                let x = paths.get(step, p);
                let f0 = process.drift(x, t);
                let g0 = process.diffusion(x, t);
                let dw = rng.next_normal() * sqrt_dt;

                // Predictor (Euler step)
                let x_pred = x + f0 * dt + g0 * dw;

                // Corrector: trapezoid average
                let f1 = process.drift(x_pred, t_next);
                let g1 = process.diffusion(x_pred, t_next);
                let x_next = x + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dw;
                paths.set(step + 1, p, x_next);
            }
        }

        Ok(HeunResult {
            paths,
            time_grid: self.cfg.time_grid(),
            dt,
            n_paths: self.cfg.n_paths,
        })
    }

    /// Solve an **Itô** SDE using the Heun method.
    ///
    /// Converts from Itô to Stratonovich by adjusting the drift:
    /// `f_strat(x,t) = μ(x,t) - ½ σ(x,t) σ'(x,t)`
    /// then applies the standard Heun predictor-corrector.
    pub fn solve_ito<P: SdeProcess>(&self, process: &P) -> RandResult<HeunResult> {
        self.cfg.validate()?;
        if process.n_paths() != self.cfg.n_paths {
            return Err(RandError::InvalidParameter(format!(
                "process n_paths {} != config n_paths {}",
                process.n_paths(),
                self.cfg.n_paths
            )));
        }

        let dt = self.cfg.dt();
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro::new(self.cfg.seed);
        let mut paths = PathMatrix::new(self.cfg.n_steps, self.cfg.n_paths, 0.0);

        for p in 0..self.cfg.n_paths {
            paths.set(0, p, process.initial(p));
        }

        // Helper: Stratonovich drift = Itô drift - ½ σ σ'
        let strat_drift = |proc: &P, x: f64, t: f64| -> f64 {
            let mu = proc.drift(x, t);
            let sig = proc.diffusion(x, t);
            let sig_dx = proc.diffusion_dx(x, t);
            mu - 0.5 * sig * sig_dx
        };

        for step in 0..self.cfg.n_steps {
            let t = self.cfg.t0 + step as f64 * dt;
            let t_next = t + dt;
            for p in 0..self.cfg.n_paths {
                let x = paths.get(step, p);
                let f0 = strat_drift(process, x, t);
                let g0 = process.diffusion(x, t);
                let dw = rng.next_normal() * sqrt_dt;

                // Predictor
                let x_pred = x + f0 * dt + g0 * dw;

                // Corrector
                let f1 = strat_drift(process, x_pred, t_next);
                let g1 = process.diffusion(x_pred, t_next);
                let x_next = x + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dw;
                paths.set(step + 1, p, x_next);
            }
        }

        Ok(HeunResult {
            paths,
            time_grid: self.cfg.time_grid(),
            dt,
            n_paths: self.cfg.n_paths,
        })
    }

    /// Solve and return only the terminal values `X_{t1}`.
    pub fn solve_terminal<P: SdeProcess>(&self, process: &P) -> RandResult<Vec<f64>> {
        self.cfg.validate()?;
        if process.n_paths() != self.cfg.n_paths {
            return Err(RandError::InvalidParameter(format!(
                "process n_paths {} != config n_paths {}",
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
            let t_next = t + dt;
            for x_val in &mut x {
                let f0 = process.drift(*x_val, t);
                let g0 = process.diffusion(*x_val, t);
                let dw = rng.next_normal() * sqrt_dt;
                let x_pred = *x_val + f0 * dt + g0 * dw;
                let f1 = process.drift(x_pred, t_next);
                let g1 = process.diffusion(x_pred, t_next);
                *x_val += 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dw;
            }
        }
        Ok(x)
    }
}

// ─── Convenience functions ────────────────────────────────────────────────────

/// Heun simulation of standard Brownian motion (Stratonovich = Itô for additive).
pub fn heun_brownian(cfg: SdeConfig) -> RandResult<HeunResult> {
    use super::brownian::BrownianMotion;
    let bm = BrownianMotion::standard(cfg.n_paths);
    StratonovichHeun::new(cfg).solve(&bm)
}

/// Heun simulation of geometric Brownian motion in Itô form.
pub fn heun_gbm(mu: f64, sigma: f64, s0: f64, cfg: SdeConfig) -> RandResult<HeunResult> {
    use super::brownian::GeometricBrownianMotion;
    let gbm = GeometricBrownianMotion::new(cfg.n_paths, mu, sigma, s0);
    StratonovichHeun::new(cfg).solve_ito(&gbm)
}

/// Heun simulation of Ornstein-Uhlenbeck (Itô and Stratonovich coincide for additive noise).
pub fn heun_ou(theta: f64, mu: f64, sigma: f64, cfg: SdeConfig) -> RandResult<HeunResult> {
    use super::brownian::OrnsteinUhlenbeck;
    let ou = OrnsteinUhlenbeck::new(cfg.n_paths, theta, mu, sigma);
    StratonovichHeun::new(cfg).solve(&ou)
}

#[cfg(test)]
mod tests {
    use super::super::brownian::{BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck};
    use super::*;

    fn make_cfg(n_paths: usize, n_steps: usize) -> SdeConfig {
        SdeConfig::new(0.0, 1.0, n_steps, n_paths, 42)
    }

    #[test]
    fn heun_bm_shape() {
        let cfg = make_cfg(4, 100);
        let res = heun_brownian(cfg).expect("Heun BM simulation with valid config should succeed");
        assert_eq!(res.paths.shape(), (101, 4));
        assert_eq!(res.n_paths, 4);
    }

    #[test]
    fn heun_bm_starts_at_zero() {
        let cfg = make_cfg(3, 50);
        let bm = BrownianMotion::standard(3);
        let res = StratonovichHeun::new(cfg)
            .solve(&bm)
            .expect("Heun BM solve with matching n_paths should succeed");
        for p in 0..3 {
            assert!((res.paths.get(0, p)).abs() < 1e-12);
        }
    }

    #[test]
    fn heun_bm_mean_near_zero() {
        let cfg = SdeConfig::new(0.0, 1.0, 100, 20_000, 6);
        let res = heun_brownian(cfg).expect("Heun BM simulation with valid config should succeed");
        let mean = res.final_mean();
        assert!(mean.abs() < 0.05, "Heun BM mean {mean:.4} too far from 0");
    }

    #[test]
    fn heun_bm_variance_near_t() {
        let cfg = SdeConfig::new(0.0, 1.0, 200, 50_000, 9);
        let res = heun_brownian(cfg).expect("Heun BM simulation with valid config should succeed");
        let var = res.path_variance();
        let final_var = var[200];
        assert!(
            (final_var - 1.0).abs() < 0.05,
            "Heun BM var {final_var:.4} != 1"
        );
    }

    #[test]
    fn heun_gbm_positive() {
        let cfg = make_cfg(6, 200);
        let res =
            heun_gbm(0.05, 0.2, 100.0, cfg).expect("Heun GBM with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(
                v.is_finite() && v > 0.0,
                "GBM Heun path must be positive: {v}"
            );
        }
    }

    #[test]
    fn heun_gbm_mean_close_to_exact() {
        let mu = 0.1_f64;
        let s0 = 100.0_f64;
        let t = 1.0;
        let cfg = SdeConfig::new(0.0, t, 500, 30_000, 8);
        let res = heun_gbm(mu, 0.2, s0, cfg).expect("Heun GBM with valid params should succeed");
        let expected = s0 * (mu * t).exp();
        let sample_mean = res.final_mean();
        assert!(
            (sample_mean - expected).abs() / expected < 0.03,
            "Heun GBM mean {sample_mean:.2} vs expected {expected:.2}"
        );
    }

    #[test]
    fn heun_ou_mean_reverts() {
        let mu = 4.0_f64;
        let cfg = SdeConfig::new(0.0, 10.0, 1000, 10_000, 14);
        let res = heun_ou(2.0, mu, 0.5, cfg).expect("Heun OU with valid params should succeed");
        let final_mean = res.final_mean();
        assert!(
            (final_mean - mu).abs() < 0.15,
            "OU Heun mean {final_mean:.3} != {mu}"
        );
    }

    #[test]
    fn heun_n_paths_mismatch_err() {
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(5);
        assert!(StratonovichHeun::new(cfg).solve(&bm).is_err());
    }

    #[test]
    fn heun_terminal_matches_full() {
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(4);
        let full = StratonovichHeun::new(cfg)
            .solve(&bm)
            .expect("Heun BM full solve should succeed");
        let terminal = StratonovichHeun::new(cfg)
            .solve_terminal(&bm)
            .expect("Heun BM terminal solve should succeed");
        let full_final = full.paths.final_state();
        for p in 0..4 {
            assert!(
                (full_final[p] - terminal[p]).abs() < 1e-12,
                "terminal mismatch path {p}"
            );
        }
    }

    #[test]
    fn heun_solve_ito_gbm_positive() {
        // solve_ito with GBM: auto Itô-to-Strat conversion
        let cfg = make_cfg(4, 100);
        let gbm = GeometricBrownianMotion::new(4, 0.05, 0.2, 50.0);
        let res = StratonovichHeun::new(cfg)
            .solve_ito(&gbm)
            .expect("Heun GBM solve_ito with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[test]
    fn heun_ou_additive_solve_ito_equals_solve() {
        // For OU (additive noise, σ'=0): solve_ito == solve (same drift)
        let cfg = make_cfg(4, 50);
        let ou = OrnsteinUhlenbeck::new(4, 1.0, 0.0, 0.5);
        let strat = StratonovichHeun::new(cfg)
            .solve(&ou)
            .expect("Heun OU solve with valid params should succeed");
        let ito = StratonovichHeun::new(cfg)
            .solve_ito(&ou)
            .expect("Heun OU solve_ito with valid params should succeed");
        for i in 0..strat.paths.as_slice().len() {
            let diff = (strat.paths.as_slice()[i] - ito.paths.as_slice()[i]).abs();
            assert!(diff < 1e-12, "OU solve == solve_ito expected at idx {i}");
        }
    }

    #[test]
    fn heun_all_finite_ou() {
        let ou = OrnsteinUhlenbeck::new(4, 1.0, 0.0, 0.3);
        let cfg = make_cfg(4, 200);
        let res = StratonovichHeun::new(cfg)
            .solve(&ou)
            .expect("Heun OU solve with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn heun_time_grid_correct() {
        let cfg = SdeConfig::new(0.5, 1.5, 10, 2, 0);
        let res = heun_brownian(cfg).expect("Heun BM simulation with valid config should succeed");
        assert!((res.time_grid[0] - 0.5).abs() < 1e-12);
        assert!((res.time_grid[10] - 1.5).abs() < 1e-12);
        assert!((res.dt - 0.1).abs() < 1e-12);
    }

    #[test]
    fn heun_dt_field_consistent() {
        let cfg = SdeConfig::new(0.0, 2.0, 40, 2, 0);
        let res = heun_brownian(cfg).expect("Heun BM simulation with valid config should succeed");
        assert!((res.dt - 0.05).abs() < 1e-12);
    }
}
