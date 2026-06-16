//! W-cycle geometric multigrid solver for the 1D Poisson equation `-u'' = f`
//! on `[0, 1]` with homogeneous Dirichlet boundary conditions `u(0) = u(1) = 0`.
//!
//! # Algorithm
//!
//! The W-cycle differs from a V-cycle in that it applies **two recursive calls**
//! at each level instead of one.  This increases the work per cycle but gives
//! faster convergence for problems where a single V-cycle leaves significant
//! low-frequency error.
//!
//! The discrete operator at mesh spacing `h = 1/(n-1)` is
//!
//! ```text
//! (Au)_i = (2u_i − u_{i-1} − u_{i+1}) / h²   for i = 1..n-2
//! (Au)_0 = (Au)_{n-1} = 0   (Dirichlet BCs)
//! ```
//!
//! Pre- and post-smoothing use the weighted-Jacobi smoother from
//! [`crate::multigrid::smoother::weighted_jacobi_smooth`].  Restriction and
//! prolongation use the standard 1D full-weighting / linear operators from
//! [`crate::multigrid::restrict_prolong`].
//!
//! # Grid-size requirements
//!
//! `n` must satisfy `n = 2^k + 1` for some integer `k >= n_levels - 1`.
//! Equivalently, `(n - 1)` must be a power of 2 and at least `2^{n_levels-1}`.

use crate::error::{PdeError, PdeResult};
use crate::multigrid::restrict_prolong::{prolong_1d, restrict_1d};
use crate::multigrid::smoother::weighted_jacobi_smooth;

/// Configuration for the W-cycle multigrid solver.
#[derive(Debug, Clone)]
pub struct WcycleConfig {
    /// Number of multigrid levels (must be >= 2).
    pub n_levels: usize,
    /// Number of pre-smoothing Jacobi sweeps per level.
    pub nu1: usize,
    /// Number of post-smoothing Jacobi sweeps per level.
    pub nu2: usize,
    /// Weighted-Jacobi relaxation parameter (must be in (0, 1)).
    pub omega: f64,
    /// Convergence tolerance on the residual norm.
    pub coarse_tol: f64,
    /// Maximum number of outer W-cycle iterations.
    pub coarse_max_iter: usize,
}

/// W-cycle geometric multigrid solver for `-u'' = f` on `[0,1]`.
#[derive(Debug, Clone)]
pub struct WcycleSolver {
    config: WcycleConfig,
}

impl WcycleSolver {
    /// Construct a new solver, validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns `PdeError::InvalidParameter` if `omega` is not in `(0, 1)`.
    pub fn new(config: WcycleConfig) -> PdeResult<Self> {
        if config.omega <= 0.0 || config.omega >= 1.0 {
            return Err(PdeError::InvalidParameter {
                name: "omega".into(),
                reason: "must be strictly in (0, 1)".into(),
            });
        }
        if config.n_levels < 1 {
            return Err(PdeError::InvalidParameter {
                name: "n_levels".into(),
                reason: "must be >= 1".into(),
            });
        }
        Ok(Self { config })
    }

    /// Solve `Au = f` on a 1D grid of `n` points.
    ///
    /// `n` must equal `2^k + 1` for some `k >= n_levels - 1`.
    /// Returns the approximate solution vector of length `n` (with `u[0] = u[n-1] = 0`).
    ///
    /// # Errors
    ///
    /// Returns `PdeError::InvalidGrid` if `n` is too small or not of the required form.
    pub fn solve(&self, f: &[f64], n: usize) -> PdeResult<Vec<f64>> {
        validate_grid(n, self.config.n_levels)?;
        if f.len() != n {
            return Err(PdeError::DimensionMismatch { a: f.len(), b: n });
        }
        let h = 1.0 / (n - 1) as f64;
        let mut u = vec![0.0_f64; n];
        for _ in 0..self.config.coarse_max_iter {
            w_cycle_recursive(
                &mut u,
                f,
                h,
                0,
                self.config.n_levels,
                self.config.nu1,
                self.config.nu2,
                self.config.omega,
            )?;
            let res = residual_norm_impl(&u, f, n, h);
            if res < self.config.coarse_tol {
                break;
            }
        }
        Ok(u)
    }

    /// Compute the L2-norm of the residual `||f - Au||_2` for the 1D Poisson
    /// operator with mesh spacing `h = 1/(n-1)`.
    pub fn residual_norm(&self, u: &[f64], f: &[f64], n: usize) -> f64 {
        if u.len() != n || f.len() != n || n < 2 {
            return f64::INFINITY;
        }
        let h = 1.0 / (n - 1) as f64;
        residual_norm_impl(u, f, n, h)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check that `n = 2^k + 1` with `k >= n_levels - 1`.
fn validate_grid(n: usize, n_levels: usize) -> PdeResult<()> {
    if n < 3 {
        return Err(PdeError::InvalidGrid(format!("wcycle: n={n} must be >= 3")));
    }
    let nm1 = n - 1;
    // Check nm1 is a power of 2
    if nm1 & (nm1 - 1) != 0 {
        return Err(PdeError::InvalidGrid(format!(
            "wcycle: n-1={nm1} must be a power of 2"
        )));
    }
    // Check 2^(n_levels-1) <= n-1
    let min_nm1 = 1_usize << (n_levels.saturating_sub(1));
    if nm1 < min_nm1 {
        return Err(PdeError::InvalidGrid(format!(
            "wcycle: n-1={nm1} < 2^(n_levels-1)={min_nm1}; need more grid points or fewer levels"
        )));
    }
    Ok(())
}

/// Compute `||f - Au||_2` for the 1D Poisson operator.
fn residual_norm_impl(u: &[f64], f: &[f64], n: usize, h: f64) -> f64 {
    let inv_h2 = 1.0 / (h * h);
    let mut sum_sq = 0.0_f64;
    for i in 1..n - 1 {
        let lap = inv_h2 * (2.0 * u[i] - u[i - 1] - u[i + 1]);
        let r = f[i] - lap;
        sum_sq += r * r;
    }
    sum_sq.sqrt()
}

/// Compute the residual vector `r = f - Au`, with `r[0] = r[n-1] = 0`.
fn residual_vec(u: &[f64], f: &[f64], h: f64) -> Vec<f64> {
    let n = u.len();
    let inv_h2 = 1.0 / (h * h);
    let mut r = vec![0.0_f64; n];
    for i in 1..n - 1 {
        let lap = inv_h2 * (2.0 * u[i] - u[i - 1] - u[i + 1]);
        r[i] = f[i] - lap;
    }
    r
}

/// One W-cycle recursion.
///
/// - `level = 0` is the finest level.
/// - `level = n_levels - 1` is the coarsest level (direct solve via Jacobi).
/// - Two recursive calls are made at each intermediate level.
#[allow(clippy::too_many_arguments)]
fn w_cycle_recursive(
    u: &mut [f64],
    f: &[f64],
    h: f64,
    level: usize,
    n_levels: usize,
    nu1: usize,
    nu2: usize,
    omega: f64,
) -> PdeResult<()> {
    let n = u.len();

    // Coarsest level: solve directly with many Jacobi iterations.
    if level + 1 >= n_levels || n <= 3 {
        // Single interior unknown at n=3, or coarsest level: run Jacobi to convergence.
        let coarse_sweeps = if n <= 3 { 1 } else { 200 };
        weighted_jacobi_smooth(u, f, h, omega, coarse_sweeps)?;
        return Ok(());
    }

    // Pre-smooth
    weighted_jacobi_smooth(u, f, h, omega, nu1)?;

    // Compute residual and restrict to coarse grid
    let r_fine = residual_vec(u, f, h);
    let r_coarse = restrict_1d(&r_fine)?;
    let n_coarse = r_coarse.len();
    let h_coarse = h * 2.0;

    // ---- First W-cycle recursion ----
    let mut e_coarse = vec![0.0_f64; n_coarse];
    w_cycle_recursive(
        &mut e_coarse,
        &r_coarse,
        h_coarse,
        level + 1,
        n_levels,
        nu1,
        nu2,
        omega,
    )?;
    // Prolong and correct
    let e_fine = prolong_1d(&e_coarse)?;
    for i in 0..n {
        u[i] += e_fine[i];
    }

    // ---- Second W-cycle recursion (the defining feature of W-cycle) ----
    // Re-compute residual for the second coarse-grid correction.
    let r_fine2 = residual_vec(u, f, h);
    let r_coarse2 = restrict_1d(&r_fine2)?;
    let mut e_coarse2 = vec![0.0_f64; n_coarse];
    w_cycle_recursive(
        &mut e_coarse2,
        &r_coarse2,
        h_coarse,
        level + 1,
        n_levels,
        nu1,
        nu2,
        omega,
    )?;
    // Prolong and correct again
    let e_fine2 = prolong_1d(&e_coarse2)?;
    for i in 0..n {
        u[i] += e_fine2[i];
    }

    // Post-smooth
    weighted_jacobi_smooth(u, f, h, omega, nu2)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> WcycleConfig {
        WcycleConfig {
            n_levels: 3,
            nu1: 3,
            nu2: 3,
            omega: 0.667,
            coarse_tol: 1e-10,
            coarse_max_iter: 50,
        }
    }

    #[test]
    fn solve_n9_converges() {
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 9; // 2^3 + 1
        let f = vec![2.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        let res = solver.residual_norm(&u, &f, n);
        assert!(res < 1e-8, "residual {res} not < 1e-8");
    }

    #[test]
    fn solve_n17_converges() {
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 17;
        let f = vec![2.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        let res = solver.residual_norm(&u, &f, n);
        assert!(res < 1e-8, "residual {res} not < 1e-8");
    }

    #[test]
    fn solve_n33_converges() {
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 33;
        let f = vec![2.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        let res = solver.residual_norm(&u, &f, n);
        assert!(res < 1e-8, "residual {res} not < 1e-8");
    }

    #[test]
    fn residual_decreases() {
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg.clone()).expect("ok");
        let n = 17;
        // Use a sinusoidal RHS so the zero initial guess has a large residual.
        let h = 1.0 / (n - 1) as f64;
        let pi = std::f64::consts::PI;
        let f: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * h;
                pi * pi * (pi * x).sin()
            })
            .collect();
        // Start from zero — residual = ||f||_2, which is large
        let u0 = vec![0.0_f64; n];
        let res0 = solver.residual_norm(&u0, &f, n);
        // Run one W-cycle step
        let mut u = u0.clone();
        w_cycle_recursive(&mut u, &f, h, 0, cfg.n_levels, cfg.nu1, cfg.nu2, cfg.omega).expect("ok");
        let res1 = solver.residual_norm(&u, &f, n);
        assert!(
            res1 < res0,
            "residual did not decrease: {res0:.6e} -> {res1:.6e}"
        );
    }

    #[test]
    fn zero_rhs_zero_solution() {
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 17;
        let f = vec![0.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        for (i, &ui) in u.iter().enumerate() {
            assert!(ui.abs() < 1e-12, "u[{i}] = {ui} != 0 for zero rhs");
        }
    }

    #[test]
    fn constant_rhs_parabolic_profile() {
        // -u'' = 2 on [0,1] with u(0)=u(1)=0 => exact u(x) = x(1-x)
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 33;
        let f = vec![2.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        let h = 1.0 / (n - 1) as f64;
        for (i, &ui) in u.iter().enumerate().take(n - 1).skip(1) {
            let x = i as f64 * h;
            let exact = x * (1.0 - x);
            assert!((ui - exact).abs() < 1e-3, "u[{i}] = {ui} vs exact {exact}");
        }
    }

    #[test]
    fn n_levels_too_many_error() {
        let mut cfg = default_config();
        cfg.n_levels = 10; // requires n-1 >= 2^9 = 512, i.e. n >= 513
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 17; // n-1 = 16 = 2^4, need at least 2^9 = 512
        let f = vec![2.0_f64; n];
        let result = solver.solve(&f, n);
        assert!(
            matches!(result, Err(PdeError::InvalidGrid(_))),
            "expected InvalidGrid"
        );
    }

    #[test]
    fn omega_out_of_range_error() {
        let mut cfg = default_config();
        cfg.omega = 1.5;
        let result = WcycleSolver::new(cfg);
        assert!(
            matches!(result, Err(PdeError::InvalidParameter { .. })),
            "expected InvalidParameter for omega=1.5"
        );
    }

    #[test]
    fn omega_zero_error() {
        let mut cfg = default_config();
        cfg.omega = 0.0;
        let result = WcycleSolver::new(cfg);
        assert!(
            matches!(result, Err(PdeError::InvalidParameter { .. })),
            "expected InvalidParameter for omega=0.0"
        );
    }

    #[test]
    fn residual_norm_zero_for_exact() {
        // Exact solution u(x) = x(1-x) for -u''=2
        let n = 17;
        let h = 1.0 / (n - 1) as f64;
        let u: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * h;
                x * (1.0 - x)
            })
            .collect();
        let f = vec![2.0_f64; n];
        let cfg = default_config();
        let solver = WcycleSolver::new(cfg).expect("ok");
        let res = solver.residual_norm(&u, &f, n);
        // The FD discretisation of the exact parabola is exact (polynomial of degree 2),
        // so the residual should be machine-zero.
        assert!(res < 1e-10, "residual for exact solution = {res}");
    }

    #[test]
    fn solve_n65_converges() {
        let cfg = WcycleConfig {
            n_levels: 4,
            nu1: 3,
            nu2: 3,
            omega: 0.667,
            coarse_tol: 1e-9,
            coarse_max_iter: 60,
        };
        let solver = WcycleSolver::new(cfg).expect("ok");
        let n = 65; // 2^6 + 1
        let f = vec![2.0_f64; n];
        let u = solver.solve(&f, n).expect("ok");
        let res = solver.residual_norm(&u, &f, n);
        assert!(res < 1e-8, "residual {res} not < 1e-8");
    }
}
