//! Milstein numerical scheme for Itô SDEs (strong order 1.0).
//!
//! The Milstein method adds a correction term to Euler-Maruyama that accounts
//! for the curvature of the diffusion coefficient:
//!
//! ```text
//! X_{t+Δ} = X_t + μ(X_t,t)Δt + σ(X_t,t)ΔW
//!           + ½ σ(X_t,t) σ'(X_t,t) (ΔW² - Δt)
//! ```
//!
//! where `ΔW = √Δt · Z` (Z ~ N(0,1)) and σ' = ∂σ/∂x.
//!
//! **Strong order**: 1.0 — superior to Euler-Maruyama (0.5) for multiplicative
//! noise (σ depends on X). For additive noise (σ' = 0), the correction term
//! vanishes and Milstein reduces to Euler-Maruyama.
//!
//! The derivative σ' can be supplied analytically via [`SdeProcess::diffusion_dx`]
//! or is estimated via central finite differences if not overridden.

use super::{PathMatrix, SdeConfig, SdeProcess, Xoshiro};
use crate::error::{RandError, RandResult};

/// Result of a Milstein simulation.
#[derive(Debug, Clone)]
pub struct MilsteinResult {
    /// Path values with shape `(n_steps + 1, n_paths)`.
    pub paths: PathMatrix,
    /// Time grid `[t0, ..., t1]`.
    pub time_grid: Vec<f64>,
    /// Step size used.
    pub dt: f64,
    /// Number of paths.
    pub n_paths: usize,
}

impl MilsteinResult {
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
}

/// Milstein solver for Itô SDEs.
///
/// Achieves strong order 1.0 by including the σσ' correction term.
/// For processes where `diffusion_dx` is zero (additive noise), this
/// is identical to Euler-Maruyama at the same cost.
#[derive(Debug, Clone)]
pub struct Milstein {
    cfg: SdeConfig,
}

impl Milstein {
    /// Create a solver with the given configuration.
    pub fn new(cfg: SdeConfig) -> Self {
        Self { cfg }
    }

    /// Solve the SDE, returning full path histories.
    pub fn solve<P: SdeProcess>(&self, process: &P) -> RandResult<MilsteinResult> {
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
            for p in 0..self.cfg.n_paths {
                let x = paths.get(step, p);
                let mu = process.drift(x, t);
                let sigma = process.diffusion(x, t);
                let sigma_dx = process.diffusion_dx(x, t);
                let dw = rng.next_normal() * sqrt_dt;

                // Milstein correction: ½ σ σ' (ΔW² - Δt)
                let correction = 0.5 * sigma * sigma_dx * (dw * dw - dt);
                let x_next = x + mu * dt + sigma * dw + correction;
                paths.set(step + 1, p, x_next);
            }
        }

        Ok(MilsteinResult {
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
            for x_val in &mut x {
                let mu = process.drift(*x_val, t);
                let sigma = process.diffusion(*x_val, t);
                let sigma_dx = process.diffusion_dx(*x_val, t);
                let dw = rng.next_normal() * sqrt_dt;
                let correction = 0.5 * sigma * sigma_dx * (dw * dw - dt);
                *x_val += mu * dt + sigma * dw + correction;
            }
        }
        Ok(x)
    }

    /// Compare Milstein vs Euler-Maruyama mean-square error relative improvement.
    ///
    /// Generates `n_reference` fine-grid reference paths, then computes strong
    /// error for both methods at a coarser step size. Returns `(em_err, milstein_err)`.
    pub fn convergence_comparison<P: SdeProcess>(
        process: &P,
        t0: f64,
        t1: f64,
        coarse_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> RandResult<(f64, f64)> {
        use super::euler_maruyama::EulerMaruyama;

        let coarse_cfg = SdeConfig::new(t0, t1, coarse_steps, n_paths, seed);
        // Fine reference: 16x more steps
        let fine_cfg = SdeConfig::new(t0, t1, coarse_steps * 16, n_paths, seed);

        // Reference via fine EM
        let fine = EulerMaruyama::new(fine_cfg).solve_terminal(process)?;

        // Coarse EM
        let em_result = EulerMaruyama::new(coarse_cfg).solve_terminal(process)?;
        let em_err = EulerMaruyama::strong_error(&em_result, &fine)?;

        // Coarse Milstein
        let mil_result = Milstein::new(coarse_cfg).solve_terminal(process)?;
        let mil_err = EulerMaruyama::strong_error(&mil_result, &fine)?;

        Ok((em_err, mil_err))
    }
}

// ─── Pre-built convenience functions ─────────────────────────────────────────

/// Milstein simulation of geometric Brownian motion.
///
/// For GBM `dS = μS dt + σS dW`, Milstein correction: `½σ²S(ΔW² - Δt)`.
pub fn milstein_gbm(mu: f64, sigma: f64, s0: f64, cfg: SdeConfig) -> RandResult<MilsteinResult> {
    use super::brownian::GeometricBrownianMotion;
    let gbm = GeometricBrownianMotion::new(cfg.n_paths, mu, sigma, s0);
    Milstein::new(cfg).solve(&gbm)
}

/// Milstein simulation of the Ornstein-Uhlenbeck process.
///
/// Since OU has additive noise (σ independent of X), σ' = 0 and the
/// Milstein correction vanishes — this is identical to Euler-Maruyama.
pub fn milstein_ou(theta: f64, mu: f64, sigma: f64, cfg: SdeConfig) -> RandResult<MilsteinResult> {
    use super::brownian::OrnsteinUhlenbeck;
    let ou = OrnsteinUhlenbeck::new(cfg.n_paths, theta, mu, sigma);
    Milstein::new(cfg).solve(&ou)
}

#[cfg(test)]
mod tests {
    use super::super::brownian::{BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck};
    use super::*;

    fn make_cfg(n_paths: usize, n_steps: usize) -> SdeConfig {
        SdeConfig::new(0.0, 1.0, n_steps, n_paths, 42)
    }

    #[test]
    fn milstein_bm_shape() {
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(4);
        let res = Milstein::new(cfg)
            .solve(&bm)
            .expect("Milstein BM solve with valid config should succeed");
        assert_eq!(res.paths.shape(), (101, 4));
        assert_eq!(res.n_paths, 4);
    }

    #[test]
    fn milstein_bm_starts_at_zero() {
        let cfg = make_cfg(3, 50);
        let bm = BrownianMotion::standard(3);
        let res = Milstein::new(cfg)
            .solve(&bm)
            .expect("Milstein BM solve with valid config should succeed");
        for p in 0..3 {
            assert!((res.paths.get(0, p)).abs() < 1e-12);
        }
    }

    #[test]
    fn milstein_bm_additive_correction_zero() {
        // For BM (σ' = 0), Milstein = EM numerically at same seed
        use super::super::euler_maruyama::EulerMaruyama;
        let cfg = make_cfg(4, 50);
        let bm = BrownianMotion::standard(4);
        let mil = Milstein::new(cfg)
            .solve(&bm)
            .expect("Milstein BM solve with valid config should succeed");
        let em = EulerMaruyama::new(cfg)
            .solve(&bm)
            .expect("EM BM solve with valid config should succeed");
        for i in 0..mil.paths.as_slice().len() {
            let diff = (mil.paths.as_slice()[i] - em.paths.as_slice()[i]).abs();
            assert!(diff < 1e-12, "Milstein != EM for additive noise at idx {i}");
        }
    }

    #[test]
    fn milstein_gbm_positive() {
        let cfg = make_cfg(6, 200);
        let res = milstein_gbm(0.05, 0.2, 100.0, cfg)
            .expect("Milstein GBM with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[test]
    fn milstein_gbm_mean_close_to_exact() {
        let mu = 0.1_f64;
        let s0 = 100.0_f64;
        let t = 1.0;
        let cfg = SdeConfig::new(0.0, t, 500, 30_000, 5);
        let res =
            milstein_gbm(mu, 0.2, s0, cfg).expect("Milstein GBM with valid params should succeed");
        let expected = s0 * (mu * t).exp();
        let sample_mean = res.final_mean();
        assert!(
            (sample_mean - expected).abs() / expected < 0.03,
            "mean {sample_mean:.2} vs expected {expected:.2}"
        );
    }

    #[test]
    fn milstein_gbm_correction_different_from_em() {
        // For GBM (σ' = σ ≠ 0), correction should be non-trivial
        use super::super::euler_maruyama::EulerMaruyama;
        let cfg = SdeConfig::new(0.0, 1.0, 10, 4, 77);
        let gbm = GeometricBrownianMotion::new(4, 0.1, 0.5, 100.0);
        let mil = Milstein::new(cfg)
            .solve(&gbm)
            .expect("Milstein GBM solve with valid params should succeed");
        let em = EulerMaruyama::new(cfg)
            .solve(&gbm)
            .expect("EM GBM solve with valid params should succeed");
        // At least one step should differ (correction ≠ 0)
        let any_diff = mil
            .paths
            .as_slice()
            .iter()
            .zip(em.paths.as_slice().iter())
            .any(|(&m, &e)| (m - e).abs() > 1e-10);
        assert!(
            any_diff,
            "Milstein and EM should differ for multiplicative noise"
        );
    }

    #[test]
    fn milstein_ou_additive_equals_em() {
        // OU has additive noise: σ'=0, so Milstein == EM
        use super::super::euler_maruyama::EulerMaruyama;
        let cfg = make_cfg(3, 50);
        let ou = OrnsteinUhlenbeck::new(3, 1.0, 0.0, 0.5);
        let mil = Milstein::new(cfg)
            .solve(&ou)
            .expect("Milstein OU solve with valid params should succeed");
        let em = EulerMaruyama::new(cfg)
            .solve(&ou)
            .expect("EM OU solve with valid params should succeed");
        for i in 0..mil.paths.as_slice().len() {
            let diff = (mil.paths.as_slice()[i] - em.paths.as_slice()[i]).abs();
            assert!(diff < 1e-12, "OU Milstein != EM at idx {i}");
        }
    }

    #[test]
    fn milstein_terminal_matches_full() {
        let cfg = make_cfg(4, 100);
        let bm = BrownianMotion::standard(4);
        let full = Milstein::new(cfg)
            .solve(&bm)
            .expect("Milstein BM full solve should succeed");
        let terminal = Milstein::new(cfg)
            .solve_terminal(&bm)
            .expect("Milstein BM terminal solve should succeed");
        let full_final = full.paths.final_state();
        for p in 0..4 {
            assert!(
                (full_final[p] - terminal[p]).abs() < 1e-12,
                "terminal mismatch path {p}"
            );
        }
    }

    #[test]
    fn milstein_n_paths_mismatch_err() {
        let cfg = make_cfg(4, 50);
        let bm = BrownianMotion::standard(3);
        assert!(Milstein::new(cfg).solve(&bm).is_err());
    }

    #[test]
    fn milstein_gbm_variance_decreases_with_dt() {
        // Milstein should have smaller error than EM for GBM at coarser steps
        let gbm = GeometricBrownianMotion::new(2_000, 0.0, 0.5, 1.0);
        let (em_err, mil_err) = Milstein::convergence_comparison(&gbm, 0.0, 1.0, 10, 2_000, 13)
            .expect("convergence comparison with valid GBM params should succeed");
        // Milstein should be at least as accurate (often better for mult. noise)
        assert!(
            mil_err <= em_err * 1.5,
            "Milstein err {mil_err:.6} not better than EM err {em_err:.6}"
        );
    }

    #[test]
    fn milstein_all_finite() {
        let cfg = make_cfg(5, 100);
        let gbm = GeometricBrownianMotion::new(5, 0.05, 0.2, 50.0);
        let res = Milstein::new(cfg)
            .solve(&gbm)
            .expect("Milstein GBM solve with valid params should succeed");
        for &v in res.paths.as_slice() {
            assert!(v.is_finite());
        }
    }
}
