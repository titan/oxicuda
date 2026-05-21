//! 1D viscous Burgers' equation: `u_t + u·u_x = ν·u_xx` on `[0, L]`.
//!
//! # Discretisation
//!
//! The PDE is written in conservation form
//!
//! ```text
//! u_t + (½ u²)_x = ν u_xx
//! ```
//!
//! and discretised with
//!
//! * **Conservative upwind / Lax-Friedrichs hybrid** for the advective flux
//!   `f(u) = ½ u²`. We use a local Lax-Friedrichs (Rusanov) numerical flux
//!
//!   ```text
//!   F̂_{i+½} = ½ ( f(u_i) + f(u_{i+1}) ) − ½ α_{i+½} ( u_{i+1} − u_i )
//!   ```
//!
//!   where the local viscosity coefficient is
//!   `α_{i+½} = max( |u_i|, |u_{i+1}| )`. This is the Godunov-type / Lax-Friedrichs
//!   hybrid recommended in LeVeque, *Finite Volume Methods for Hyperbolic Problems*
//!   (Cambridge, 2002, §12.2). It is monotone, total-variation-diminishing (TVD)
//!   on monotone data, and resolves shocks with at most a two-cell transition.
//!
//! * **Central second-order** for the viscous term: `(u_{i-1} − 2 u_i + u_{i+1}) / dx²`.
//!
//! * **Forward Euler** in time. Mention CFL constraint below.
//!
//! # CFL stability
//!
//! Combined advective + diffusive CFL:
//!
//! ```text
//! dt · ( max|u| + 2 ν / dx ) / dx  ≤  1
//! ```
//!
//! The solver rejects with `PdeError::CflViolation` if this is violated at
//! initialisation. With `ν = 0` the upwind flux reduces to the inviscid
//! Godunov / LLF scheme and reproduces the Rankine-Hugoniot shock speed
//! `s = ½ (u_L + u_R)`.
//!
//! # Boundary conditions
//!
//! Two choices: `Periodic` and `Dirichlet { left, right }`. Under Dirichlet the
//! boundary values are kept fixed; under periodic the ghost cells wrap around.
//!
//! # References
//!
//! * LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge 2002,
//!   chapters 12 and 15.

use crate::error::{PdeError, PdeResult};

/// Boundary condition variant for 1D Burgers' equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Burgers1dBc {
    /// Periodic boundary: `u_{nx} = u_0` (wrap-around).
    Periodic,
    /// Dirichlet boundary: fixed left and right values.
    Dirichlet {
        /// Value at the left boundary (cell index 0).
        left: f64,
        /// Value at the right boundary (cell index `nx-1`).
        right: f64,
    },
}

/// Configuration for the 1D Burgers' equation solver.
#[derive(Debug, Clone, Copy)]
pub struct Burgers1dConfig {
    /// Number of grid points (≥ 3).
    pub nx: usize,
    /// Spatial step (must be > 0).
    pub dx: f64,
    /// Time step (must be > 0).
    pub dt: f64,
    /// Kinematic viscosity (must be ≥ 0). `ν = 0` selects the inviscid path.
    pub nu: f64,
    /// Number of time steps (≥ 1).
    pub n_steps: usize,
    /// Boundary condition.
    pub bc: Burgers1dBc,
}

impl Default for Burgers1dConfig {
    fn default() -> Self {
        Self {
            nx: 101,
            dx: 1.0 / 100.0,
            dt: 1.0e-3,
            nu: 1.0e-2,
            n_steps: 100,
            bc: Burgers1dBc::Periodic,
        }
    }
}

/// Result of the 1D Burgers' equation solver.
#[derive(Debug, Clone)]
pub struct Burgers1dResult {
    /// Final solution vector of length `nx`.
    pub u: Vec<f64>,
    /// Final time `t = n_steps · dt`.
    pub t_final: f64,
}

/// Local Lax-Friedrichs numerical flux for `f(u) = ½ u²`.
///
/// ```text
/// F̂(u_l, u_r) = ½ ( ½ u_l² + ½ u_r² ) − ½ α ( u_r − u_l )
/// ```
///
/// with `α = max(|u_l|, |u_r|)`.
#[inline]
fn rusanov_flux(u_l: f64, u_r: f64) -> f64 {
    let f_l = 0.5 * u_l * u_l;
    let f_r = 0.5 * u_r * u_r;
    let alpha = u_l.abs().max(u_r.abs());
    0.5 * (f_l + f_r) - 0.5 * alpha * (u_r - u_l)
}

fn err_param(name: &str, reason: &str) -> PdeError {
    PdeError::InvalidParameter {
        name: name.into(),
        reason: reason.into(),
    }
}

fn validate_config(cfg: &Burgers1dConfig) -> PdeResult<()> {
    if cfg.nx < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "burgers_1d requires nx>=3, got nx={}",
            cfg.nx
        )));
    }
    if cfg.dx <= 0.0 || !cfg.dx.is_finite() {
        return Err(err_param("dx", "must be positive and finite"));
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(err_param("dt", "must be positive and finite"));
    }
    if cfg.nu < 0.0 || !cfg.nu.is_finite() {
        return Err(err_param("nu", "must be non-negative and finite"));
    }
    if cfg.n_steps == 0 {
        return Err(err_param("n_steps", "must be >= 1"));
    }
    Ok(())
}

/// Maximum |u| across the working array (used in CFL check).
fn umax_abs(u: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for &v in u {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// Take one explicit forward-Euler step using the conservative LLF flux plus
/// the central viscous stencil.
///
/// Returns the updated array in `next` (does not modify `u`).
fn step_once(u: &[f64], next: &mut [f64], cfg: &Burgers1dConfig) -> PdeResult<()> {
    let nx = cfg.nx;
    if u.len() != nx || next.len() != nx {
        return Err(PdeError::ShapeMismatch {
            expected: vec![nx],
            got: vec![u.len()],
        });
    }
    let dx = cfg.dx;
    let dt = cfg.dt;
    let nu = cfg.nu;
    let inv_dx = 1.0 / dx;
    let nu_over_dx2 = nu / (dx * dx);
    // Indices and boundary fluxes depend on BC.
    match cfg.bc {
        Burgers1dBc::Periodic => {
            // Compute fluxes at interfaces i+½ for i = 0..nx; F[i] sits between
            // cell i and cell (i+1) mod nx.
            // Then u_new[i] = u[i] − dt/dx · ( F[i] − F[i-1] ) + dt · ν · (u_{i-1} − 2 u_i + u_{i+1}) / dx²
            for i in 0..nx {
                let ip1 = if i + 1 == nx { 0 } else { i + 1 };
                let im1 = if i == 0 { nx - 1 } else { i - 1 };
                let f_right = rusanov_flux(u[i], u[ip1]);
                let f_left = rusanov_flux(u[im1], u[i]);
                let viscous = nu_over_dx2 * (u[im1] - 2.0 * u[i] + u[ip1]);
                next[i] = u[i] - dt * inv_dx * (f_right - f_left) + dt * viscous;
            }
        }
        Burgers1dBc::Dirichlet { left, right } => {
            // Pin endpoints, update interior with conservative LLF.
            next[0] = left;
            next[nx - 1] = right;
            for i in 1..nx - 1 {
                let f_right = rusanov_flux(u[i], u[i + 1]);
                let f_left = rusanov_flux(u[i - 1], u[i]);
                let viscous = nu_over_dx2 * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
                next[i] = u[i] - dt * inv_dx * (f_right - f_left) + dt * viscous;
            }
        }
    }
    Ok(())
}

/// Solve the 1D viscous (or inviscid) Burgers' equation by time-stepping `u0`
/// for `cfg.n_steps` explicit forward-Euler steps.
///
/// # Errors
///
/// Returns `PdeError::InvalidGrid` / `InvalidParameter` for invalid
/// configurations, `PdeError::ShapeMismatch` if `u0.len() != cfg.nx`, and
/// `PdeError::CflViolation` if the combined advective + diffusive CFL
/// constraint
///
/// ```text
/// dt · ( max|u| + 2 ν / dx ) / dx  ≤  1
/// ```
///
/// is violated by the initial data.
pub fn solve_burgers_1d(u0: &[f64], cfg: &Burgers1dConfig) -> PdeResult<Burgers1dResult> {
    validate_config(cfg)?;
    let nx = cfg.nx;
    if u0.len() != nx {
        return Err(PdeError::ShapeMismatch {
            expected: vec![nx],
            got: vec![u0.len()],
        });
    }
    // CFL on initial data. We are explicit, so the constraint is on the
    // *currently observed* max|u|; if the solution becomes more energetic
    // (rare for Burgers — it never amplifies in L^∞) the user must shrink dt.
    let umax = umax_abs(u0);
    let cfl = cfg.dt * (umax + 2.0 * cfg.nu / cfg.dx) / cfg.dx;
    if cfl > 1.0 + 1.0e-12 {
        let dt_max = cfg.dx / (umax + 2.0 * cfg.nu / cfg.dx).max(1.0e-300);
        return Err(PdeError::CflViolation { dt: cfg.dt, dt_max });
    }
    let mut u = u0.to_vec();
    let mut next = vec![0.0_f64; nx];
    // If Dirichlet, ensure endpoints of initial data match boundary values.
    if let Burgers1dBc::Dirichlet { left, right } = cfg.bc {
        u[0] = left;
        u[nx - 1] = right;
    }
    for _ in 0..cfg.n_steps {
        step_once(&u, &mut next, cfg)?;
        u.copy_from_slice(&next);
    }
    Ok(Burgers1dResult {
        u,
        t_final: cfg.dt * cfg.n_steps as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        let dx = (b - a) / n as f64;
        (0..n).map(|i| a + dx * (i as f64 + 0.5)).collect()
    }

    #[test]
    fn invalid_nx_rejected() {
        let cfg = Burgers1dConfig {
            nx: 2,
            ..Burgers1dConfig::default()
        };
        let res = solve_burgers_1d(&[0.0, 0.0], &cfg);
        assert!(matches!(res, Err(PdeError::InvalidGrid(_))));
    }

    fn base_cfg() -> Burgers1dConfig {
        Burgers1dConfig {
            nx: 10,
            dx: 0.1,
            dt: 1.0e-3,
            nu: 0.0,
            n_steps: 1,
            bc: Burgers1dBc::Periodic,
        }
    }

    #[test]
    fn invalid_dx_dt_rejected() {
        for cfg in [
            Burgers1dConfig {
                dx: -1.0,
                ..base_cfg()
            },
            Burgers1dConfig {
                dt: 0.0,
                ..base_cfg()
            },
            Burgers1dConfig {
                nu: -0.5,
                ..base_cfg()
            },
            Burgers1dConfig {
                n_steps: 0,
                ..base_cfg()
            },
        ] {
            let res = solve_burgers_1d(&[0.0_f64; 10], &cfg);
            assert!(
                matches!(res, Err(PdeError::InvalidParameter { .. })),
                "cfg={cfg:?}"
            );
        }
    }

    #[test]
    fn length_mismatch_rejected() {
        let cfg = Burgers1dConfig {
            nx: 16,
            dx: 1.0 / 16.0,
            dt: 1.0e-3,
            nu: 1.0e-3,
            n_steps: 1,
            bc: Burgers1dBc::Periodic,
        };
        let res = solve_burgers_1d(&[0.0_f64; 8], &cfg);
        assert!(matches!(res, Err(PdeError::ShapeMismatch { .. })));
    }

    #[test]
    fn cfl_violation_rejected() {
        // very small dx but very large dt
        let cfg = Burgers1dConfig {
            nx: 32,
            dx: 1.0 / 32.0,
            dt: 1.0,
            nu: 0.0,
            n_steps: 1,
            bc: Burgers1dBc::Periodic,
        };
        let u0 = vec![1.0_f64; 32];
        let res = solve_burgers_1d(&u0, &cfg);
        assert!(matches!(res, Err(PdeError::CflViolation { .. })));
    }

    #[test]
    fn zero_initial_stays_zero() {
        let cfg = Burgers1dConfig {
            nx: 64,
            dx: 1.0 / 64.0,
            dt: 1.0e-3,
            nu: 1.0e-2,
            n_steps: 200,
            bc: Burgers1dBc::Periodic,
        };
        let u0 = vec![0.0_f64; 64];
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        for &v in &r.u {
            assert!(v.abs() < 1.0e-15);
        }
        assert!((r.t_final - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn periodic_integral_conserved() {
        // ∫ u dx is exactly conserved by the conservative LLF flux on a periodic
        // domain (telescoping).
        let nx = 64;
        let dx = 1.0 / nx as f64;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt: 1.0e-3,
            nu: 1.0e-3,
            n_steps: 100,
            bc: Burgers1dBc::Periodic,
        };
        let xs = linspace(0.0, 1.0, nx);
        let u0: Vec<f64> = xs
            .iter()
            .map(|x| 0.5 + 0.3 * (2.0 * PI * x).sin())
            .collect();
        let m0: f64 = u0.iter().sum::<f64>() * dx;
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        let m1: f64 = r.u.iter().sum::<f64>() * dx;
        assert!((m1 - m0).abs() < 1.0e-10, "mass drift {m0} -> {m1}");
    }

    #[test]
    fn dirichlet_boundary_preserved() {
        let nx = 32;
        let cfg = Burgers1dConfig {
            nx,
            dx: 1.0 / nx as f64,
            dt: 5.0e-4,
            nu: 1.0e-3,
            n_steps: 50,
            bc: Burgers1dBc::Dirichlet {
                left: 1.0,
                right: -1.0,
            },
        };
        let mut u0 = vec![0.0_f64; nx];
        u0[0] = 1.0;
        u0[nx - 1] = -1.0;
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        assert!((r.u[0] - 1.0).abs() < 1.0e-15);
        assert!((r.u[nx - 1] - (-1.0)).abs() < 1.0e-15);
    }

    #[test]
    fn inviscid_shock_rankine_hugoniot() {
        // Initial step: u₀ = 1 if x < L/2, 0 otherwise, with Dirichlet BC
        // pinning u(0)=1 and u(L)=0. The shock speed by Rankine-Hugoniot is
        // s = ½ (1 + 0) = 0.5; after T = L/4 the shock midpoint is at L/2 + sT.
        let nx = 200;
        let l = 1.0_f64;
        let dx = l / nx as f64;
        let dt = 0.5 * dx;
        let t_final = l / 4.0;
        let n_steps = (t_final / dt).round() as usize;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt,
            nu: 0.0,
            n_steps,
            bc: Burgers1dBc::Dirichlet {
                left: 1.0,
                right: 0.0,
            },
        };
        let xs = linspace(0.0, l, nx);
        let u0: Vec<f64> = xs
            .iter()
            .map(|&x| if x < 0.5 * l { 1.0 } else { 0.0 })
            .collect();
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        // Locate the midshock: cell whose value is closest to 0.5.
        let (idx, _) =
            r.u.iter()
                .enumerate()
                .fold((0_usize, f64::INFINITY), |(i, m), (k, &v)| {
                    let d = (v - 0.5).abs();
                    if d < m { (k, d) } else { (i, m) }
                });
        let x_shock = xs[idx];
        let x_expected = 0.5 * l + 0.5 * t_final;
        assert!(
            (x_shock - x_expected).abs() <= 2.0 * dx,
            "shock at x={x_shock}, expected ~{x_expected}"
        );
    }

    #[test]
    fn inviscid_monotone_preservation() {
        // Monotone step initial data must remain monotone (no spurious
        // oscillations) — characteristic property of the LLF / Godunov scheme.
        let nx = 128;
        let l = 1.0_f64;
        let dx = l / nx as f64;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt: 0.4 * dx,
            nu: 0.0,
            n_steps: 50,
            bc: Burgers1dBc::Periodic,
        };
        let xs = linspace(0.0, l, nx);
        let u0: Vec<f64> = xs
            .iter()
            .map(|&x| if x < 0.5 * l { 1.0 } else { 0.0 })
            .collect();
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        let umax = r.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let umin = r.u.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(umax <= 1.0 + 1.0e-12, "overshoot: umax={umax}");
        assert!(umin >= 0.0 - 1.0e-12, "undershoot: umin={umin}");
    }

    #[test]
    fn viscous_smoothing_max_abs_decreases() {
        // High viscosity dissipates the initial bump; check that max|u| is
        // monotonically non-increasing over a sequence of intermediate snapshots.
        let nx = 64;
        let dx = 1.0 / nx as f64;
        let nu = 0.1_f64;
        // CFL: dt < dx² / (2ν) for the diffusive part with |u|=O(1).
        let dt = 0.4 * dx * dx / (2.0 * nu);
        let xs = linspace(0.0, 1.0, nx);
        let mut u: Vec<f64> = xs
            .iter()
            .map(|&x| (-((x - 0.5).powi(2)) / 0.01).exp())
            .collect();
        let mut prev_max = umax_abs(&u);
        for _ in 0..5 {
            let cfg = Burgers1dConfig {
                nx,
                dx,
                dt,
                nu,
                n_steps: 50,
                bc: Burgers1dBc::Periodic,
            };
            let r = solve_burgers_1d(&u, &cfg).expect("ok");
            u = r.u;
            let curr_max = umax_abs(&u);
            assert!(
                curr_max <= prev_max + 1.0e-12,
                "max grew {prev_max} -> {curr_max}"
            );
            prev_max = curr_max;
        }
    }

    #[test]
    fn self_similar_rarefaction_fan() {
        // Initial step u₀ = (0 if x<L/2 else 1) develops a rarefaction fan.
        // The expansion has u monotonically non-decreasing in x and the signal
        // speed is in [0, 1], so no over/undershoot and the fan is monotone.
        let nx = 200;
        let l = 1.0_f64;
        let dx = l / nx as f64;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt: 0.4 * dx,
            nu: 0.0,
            n_steps: 50,
            bc: Burgers1dBc::Periodic,
        };
        let xs = linspace(0.0, l, nx);
        let u0: Vec<f64> = xs
            .iter()
            .map(|&x| if x < 0.5 * l { 0.0 } else { 1.0 })
            .collect();
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        let umax = r.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let umin = r.u.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(umax <= 1.0 + 1.0e-12);
        assert!(umin >= 0.0 - 1.0e-12);
        let mid = nx / 2;
        for k in (mid - 5)..(mid + 5) {
            assert!(r.u[k + 1] >= r.u[k] - 1.0e-12, "fan not monotone at k={k}");
        }
    }

    #[test]
    fn linear_advection_limit_peak_shift() {
        // For tiny ν and a small-amplitude pulse on top of a constant mean
        // ∂_t u + ū u_x ≈ 0, so the peak moves at ū · t.
        let nx = 200;
        let dx = 1.0 / nx as f64;
        let u_mean = 0.5;
        let dt = 0.4 * dx / (u_mean + 0.1);
        let n_steps = 200;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt,
            nu: 1.0e-4,
            n_steps,
            bc: Burgers1dBc::Periodic,
        };
        let xs = linspace(0.0, 1.0, nx);
        let u0: Vec<f64> = xs
            .iter()
            .map(|&x| u_mean + 0.05 * (-((x - 0.3).powi(2)) / 0.005).exp())
            .collect();
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        let (idx, _) =
            r.u.iter()
                .enumerate()
                .fold((0_usize, f64::NEG_INFINITY), |(i, m), (k, &v)| {
                    if v > m { (k, v) } else { (i, m) }
                });
        let x_peak = xs[idx];
        let expected = (0.3 + u_mean * dt * n_steps as f64).rem_euclid(1.0);
        assert!(
            (x_peak - expected).abs() < 5.0 * dx,
            "peak at {x_peak}, expected {expected}"
        );
    }

    #[test]
    fn antisymmetric_initial_stays_antisymmetric() {
        // For Burgers, odd-about-x=½ initial data is preserved: the advective
        // flux ½(u²)_x and the viscous term u_xx both commute with the
        // involution (x → 1−x, u → −u). With cell centres x_i = (i + ½) dx the
        // reflection i → nx−1−i lands on x = 1 − x_i, so u(x,t) = −u(1−x, t).
        let nx = 64;
        let dx = 1.0 / nx as f64;
        let cfg = Burgers1dConfig {
            nx,
            dx,
            dt: 1.0e-3,
            nu: 1.0e-2,
            n_steps: 50,
            bc: Burgers1dBc::Periodic,
        };
        let xs = linspace(0.0, 1.0, nx);
        let u0: Vec<f64> = xs.iter().map(|&x| (2.0 * PI * x).sin()).collect();
        let r = solve_burgers_1d(&u0, &cfg).expect("ok");
        for i in 0..nx / 2 {
            let j = nx - 1 - i;
            assert!((r.u[i] + r.u[j]).abs() < 1.0e-10, "asymmetry at i={i}");
        }
    }

    #[test]
    fn default_config_round_trips() {
        let cfg = Burgers1dConfig::default();
        assert_eq!(cfg.nx, 101);
        assert!(cfg.dt > 0.0);
        assert!(cfg.nu >= 0.0);
        assert!(matches!(cfg.bc, Burgers1dBc::Periodic));
    }
}
