//! Stochastic Differential Equation (SDE) numerical samplers.
//!
//! Provides CPU-side solvers for SDEs of the Itô form:
//! ```text
//! dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
//! ```
//! where μ is the drift, σ is the diffusion coefficient, and W_t is a
//! standard Brownian motion (Wiener process).
//!
//! ## Implemented Methods
//!
//! | Method             | Strong order | Notes                              |
//! |--------------------|-------------|-------------------------------------|
//! | Euler-Maruyama     | 0.5         | Itô; simplest, most general         |
//! | Milstein           | 1.0         | Itô; improved with σσ' correction   |
//! | Stratonovich-Heun  | 1.0         | Stratonovich; predictor-corrector   |
//!
//! ## Pre-built Processes
//!
//! [`brownian`] provides standard Brownian motion, geometric Brownian motion
//! (GBM / Black-Scholes), and the Ornstein-Uhlenbeck (OU) mean-reverting process.
//!
//! ## Example
//!
//! ```rust
//! use oxicuda_rand::sde::{BrownianMotion, EulerMaruyama, SdeConfig};
//!
//! let cfg = SdeConfig { t0: 0.0, t1: 1.0, n_steps: 100, n_paths: 4, seed: 1 };
//! let bm = BrownianMotion::standard(cfg.n_paths);
//! let solver = EulerMaruyama::new(cfg);
//! let result = solver.solve(&bm).expect("valid config and matching n_paths");
//! assert_eq!(result.paths.shape(), (101, 4)); // (n_steps+1, n_paths)
//! ```

pub mod brownian;
pub mod euler_maruyama;
pub mod heun;
pub mod milstein;

pub use brownian::{
    BrownianMotion, BrownianPathResult, GeometricBrownianMotion, OrnsteinUhlenbeck,
};
pub use euler_maruyama::{EulerMaruyama, EulerMaruyamaResult};
pub use heun::{HeunResult, StratonovichHeun};
pub use milstein::{Milstein, MilsteinResult};

use crate::error::{RandError, RandResult};

// ─── Shared internal PRNG ────────────────────────────────────────────────────

/// Xoshiro256** PRNG used internally by all SDE solvers.
/// Provides fast, high-quality pseudo-random numbers without CUDA dependency.
#[derive(Clone, Debug)]
pub(crate) struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    /// Seed via SplitMix64 expansion for good initial state diversity.
    pub(crate) fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            *slot = z;
        }
        Self { s }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller transform.
    /// Loops until u1 > 0 to avoid log(0).
    pub(crate) fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }
}

// ─── SDE configuration ───────────────────────────────────────────────────────

/// Configuration for SDE numerical solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdeConfig {
    /// Start time of the simulation.
    pub t0: f64,
    /// End time of the simulation.
    pub t1: f64,
    /// Number of discrete time steps (t1 - t0 is divided equally).
    pub n_steps: usize,
    /// Number of independent sample paths to simulate.
    pub n_paths: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl SdeConfig {
    /// Create a standard config with sensible defaults.
    #[must_use]
    pub fn new(t0: f64, t1: f64, n_steps: usize, n_paths: usize, seed: u64) -> Self {
        Self {
            t0,
            t1,
            n_steps,
            n_paths,
            seed,
        }
    }

    /// Step size `Δt = (t1 - t0) / n_steps`.
    #[must_use]
    #[inline]
    pub fn dt(&self) -> f64 {
        (self.t1 - self.t0) / self.n_steps as f64
    }

    /// Time grid `[t0, t0+dt, ..., t1]` with `n_steps + 1` points.
    #[must_use]
    pub fn time_grid(&self) -> Vec<f64> {
        let dt = self.dt();
        (0..=self.n_steps)
            .map(|i| self.t0 + i as f64 * dt)
            .collect()
    }

    /// Validate that the config is internally consistent.
    pub fn validate(&self) -> RandResult<()> {
        if self.t1 <= self.t0 {
            return Err(RandError::InvalidParameter(format!(
                "t1 ({}) must be > t0 ({})",
                self.t1, self.t0
            )));
        }
        if self.n_steps == 0 {
            return Err(RandError::InvalidParameter(
                "n_steps must be >= 1".to_string(),
            ));
        }
        if self.n_paths == 0 {
            return Err(RandError::InvalidParameter(
                "n_paths must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ─── SdeProcess trait ────────────────────────────────────────────────────────

/// Trait representing an Itô SDE `dX = μ(X, t)dt + σ(X, t)dW`.
///
/// Implementors define the drift and diffusion functions and the initial
/// condition. All solvers operate on this trait.
pub trait SdeProcess: Send + Sync {
    /// Drift coefficient μ(x, t).
    fn drift(&self, x: f64, t: f64) -> f64;

    /// Diffusion coefficient σ(x, t).
    fn diffusion(&self, x: f64, t: f64) -> f64;

    /// Derivative of σ with respect to x: ∂σ/∂x(x, t).
    /// Needed by the Milstein correction term.
    /// Defaults to a finite-difference approximation with h = 1e-6.
    #[inline]
    fn diffusion_dx(&self, x: f64, t: f64) -> f64 {
        let h = 1e-6_f64;
        (self.diffusion(x + h, t) - self.diffusion(x - h, t)) / (2.0 * h)
    }

    /// Initial state X_0 for path index `path_idx`.
    fn initial(&self, path_idx: usize) -> f64;

    /// Number of paths (determines initial condition dimensionality).
    fn n_paths(&self) -> usize;
}

// ─── PathMatrix helper ────────────────────────────────────────────────────────

/// A 2-D matrix of SDE path values: `data[(step * n_paths) + path]`.
#[derive(Debug, Clone)]
pub struct PathMatrix {
    data: Vec<f64>,
    n_steps_plus_one: usize,
    n_paths: usize,
}

impl PathMatrix {
    pub(crate) fn new(n_steps: usize, n_paths: usize, fill: f64) -> Self {
        Self {
            data: vec![fill; (n_steps + 1) * n_paths],
            n_steps_plus_one: n_steps + 1,
            n_paths,
        }
    }

    /// Access value at `(step, path)`.
    #[inline]
    pub fn get(&self, step: usize, path: usize) -> f64 {
        self.data[step * self.n_paths + path]
    }

    /// Mutable access at `(step, path)`.
    #[inline]
    pub(crate) fn set(&mut self, step: usize, path: usize, val: f64) {
        self.data[step * self.n_paths + path] = val;
    }

    /// Returns `(n_steps + 1, n_paths)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_steps_plus_one, self.n_paths)
    }

    /// Flat slice of all data in row-major order `[step0_path0, step0_path1, ..., stepN_pathM]`.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Slice for a single step (all paths at that time).
    #[must_use]
    pub fn step_slice(&self, step: usize) -> &[f64] {
        let start = step * self.n_paths;
        &self.data[start..start + self.n_paths]
    }

    /// Final state of all paths `X_{t1}`.
    #[must_use]
    pub fn final_state(&self) -> &[f64] {
        self.step_slice(self.n_steps_plus_one - 1)
    }

    /// Sample mean across paths at each time step.
    #[must_use]
    pub fn path_mean(&self) -> Vec<f64> {
        (0..self.n_steps_plus_one)
            .map(|step| {
                let s = self.step_slice(step);
                s.iter().sum::<f64>() / self.n_paths as f64
            })
            .collect()
    }

    /// Sample variance across paths at each time step.
    #[must_use]
    pub fn path_variance(&self) -> Vec<f64> {
        (0..self.n_steps_plus_one)
            .map(|step| {
                let s = self.step_slice(step);
                let mean = s.iter().sum::<f64>() / self.n_paths as f64;
                s.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / self.n_paths as f64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro_produces_different_values() {
        let mut rng = Xoshiro::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn xoshiro_uniform_in_range() {
        let mut rng = Xoshiro::new(1234);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "uniform out of range: {v}");
        }
    }

    #[test]
    fn xoshiro_normal_finite() {
        let mut rng = Xoshiro::new(99);
        for _ in 0..1000 {
            let v = rng.next_normal();
            assert!(v.is_finite(), "normal was not finite: {v}");
        }
    }

    #[test]
    fn xoshiro_normal_approximate_moments() {
        let mut rng = Xoshiro::new(7);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_normal()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "mean too far from 0: {mean}");
    }

    #[test]
    fn sde_config_dt() {
        let cfg = SdeConfig::new(0.0, 1.0, 100, 4, 0);
        let dt = cfg.dt();
        assert!((dt - 0.01).abs() < 1e-12);
    }

    #[test]
    fn sde_config_time_grid_length() {
        let cfg = SdeConfig::new(0.0, 2.0, 50, 1, 0);
        assert_eq!(cfg.time_grid().len(), 51);
    }

    #[test]
    fn sde_config_time_grid_endpoints() {
        let cfg = SdeConfig::new(0.5, 1.5, 10, 1, 0);
        let grid = cfg.time_grid();
        assert!((grid[0] - 0.5).abs() < 1e-12);
        assert!((grid[10] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn sde_config_validate_ok() {
        let cfg = SdeConfig::new(0.0, 1.0, 10, 4, 0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn sde_config_validate_bad_t1() {
        let cfg = SdeConfig::new(1.0, 0.5, 10, 4, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sde_config_validate_zero_steps() {
        let cfg = SdeConfig::new(0.0, 1.0, 0, 4, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sde_config_validate_zero_paths() {
        let cfg = SdeConfig::new(0.0, 1.0, 10, 0, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn path_matrix_shape() {
        let m = PathMatrix::new(100, 4, 0.0);
        assert_eq!(m.shape(), (101, 4));
    }

    #[test]
    fn path_matrix_get_set() {
        let mut m = PathMatrix::new(10, 3, 0.0);
        m.set(5, 2, std::f64::consts::PI);
        assert!((m.get(5, 2) - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn path_matrix_step_slice() {
        let mut m = PathMatrix::new(2, 2, 0.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        let s = m.step_slice(1);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn path_matrix_final_state() {
        let mut m = PathMatrix::new(3, 2, 0.0);
        m.set(3, 0, 5.0);
        m.set(3, 1, 7.0);
        let fs = m.final_state();
        assert!((fs[0] - 5.0).abs() < 1e-12);
        assert!((fs[1] - 7.0).abs() < 1e-12);
    }
}
