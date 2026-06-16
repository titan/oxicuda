//! 1D compressible Navier-Stokes solver on `[x_lo, x_hi]`.
//!
//! # Governing equations
//!
//! ```text
//! ∂ρ/∂t  + ∂(ρu)/∂x              = 0                         [mass]
//! ∂(ρu)/∂t + ∂(ρu² + p)/∂x      = ∂(μ ∂u/∂x)/∂x            [momentum]
//! ∂E/∂t  + ∂(u(E+p))/∂x         = ∂(κ ∂T/∂x)/∂x            [energy]
//! ```
//!
//! Closed by the ideal-gas EOS:
//!
//! ```text
//! p = (γ − 1)(E − ½ρu²/ρ)
//! T = p / (ρ R)
//! ```
//!
//! # Numerics
//!
//! The convective fluxes are computed with a **Lax-Friedrichs (Rusanov) scheme**:
//! the inter-cell numerical flux is
//!
//! ```text
//! F̂ = ½(F_L + F_R) − ½ α (U_R − U_L)
//! ```
//!
//! where `α = max(|u_L| + c_L, |u_R| + c_R)` is the local wave-speed estimate
//! and `c = sqrt(γ p / ρ)` is the sound speed.  Viscous/conductive terms are
//! discretised with second-order centred differences.  Time advancement uses
//! **RK2 (Heun's predictor-corrector)** with a CFL-based adaptive time step.
//!
//! # Boundary conditions
//!
//! Transmissive (zero-gradient) BCs: ghost cells on both ends copy the nearest
//! interior cell.

use crate::error::{PdeError, PdeResult};

/// Configuration for the 1D compressible Navier-Stokes solver.
#[derive(Debug, Clone)]
pub struct Ns1dConfig {
    /// Number of finite-volume cells (must be >= 2).
    pub nx: usize,
    /// Left domain boundary.
    pub x_lo: f64,
    /// Right domain boundary.
    pub x_hi: f64,
    /// Ratio of specific heats (γ, e.g. 1.4 for air; must be > 1).
    pub gamma: f64,
    /// Dynamic viscosity (μ ≥ 0).
    pub mu: f64,
    /// Thermal conductivity (κ ≥ 0).
    pub kappa: f64,
    /// Specific gas constant (R > 0).
    pub r_gas: f64,
    /// CFL number for adaptive time-step control (0 < cfl < 1, typical 0.4).
    pub cfl: f64,
    /// Maximum number of time steps (safety guard).
    pub max_steps: usize,
    /// End time (t_end > 0).
    pub t_end: f64,
}

/// State of the 1D compressible Navier-Stokes solution.
#[derive(Debug, Clone)]
pub struct Ns1dState {
    /// Cell-averaged density ρ (`nx` values).
    pub rho: Vec<f64>,
    /// Cell-averaged momentum density ρu (`nx` values).
    pub rho_u: Vec<f64>,
    /// Cell-averaged total energy E (`nx` values).
    pub energy: Vec<f64>,
    /// Current simulation time.
    pub t: f64,
    /// Number of completed time steps.
    pub step: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate the solver configuration.
fn validate_cfg(cfg: &Ns1dConfig) -> PdeResult<()> {
    if cfg.nx < 2 {
        return Err(PdeError::InvalidGrid(format!(
            "ns1d requires nx>=2, got nx={}",
            cfg.nx
        )));
    }
    if cfg.gamma <= 1.0 {
        return Err(PdeError::InvalidParameter {
            name: "gamma".into(),
            reason: "must be > 1.0".into(),
        });
    }
    if cfg.mu < 0.0 || !cfg.mu.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "mu".into(),
            reason: "must be >= 0 and finite".into(),
        });
    }
    if cfg.kappa < 0.0 || !cfg.kappa.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "kappa".into(),
            reason: "must be >= 0 and finite".into(),
        });
    }
    if cfg.r_gas <= 0.0 || !cfg.r_gas.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "r_gas".into(),
            reason: "must be > 0 and finite".into(),
        });
    }
    if cfg.cfl <= 0.0 || cfg.cfl >= 1.0 {
        return Err(PdeError::InvalidParameter {
            name: "cfl".into(),
            reason: "must be in (0, 1)".into(),
        });
    }
    if cfg.t_end <= 0.0 || !cfg.t_end.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "t_end".into(),
            reason: "must be > 0 and finite".into(),
        });
    }
    Ok(())
}

/// Compute pressure from conserved variables.
#[inline]
fn cell_pressure(rho: f64, rho_u: f64, energy: f64, gamma: f64) -> f64 {
    let ke = 0.5 * rho_u * rho_u / rho.max(1e-300);
    (gamma - 1.0) * (energy - ke)
}

/// Compute temperature T = p / (ρ R).
#[inline]
fn cell_temperature(p: f64, rho: f64, r_gas: f64) -> f64 {
    p / (rho.max(1e-300) * r_gas)
}

/// Lax-Friedrichs numerical flux for the Euler equations.
///
/// `(rho_l, rho_u_l, e_l)` and `(rho_r, rho_u_r, e_r)` are the conserved-variable
/// vectors on left/right of an interface.  Returns the numerical flux triplet.
#[inline]
fn llf_flux(
    rho_l: f64,
    rho_u_l: f64,
    e_l: f64,
    rho_r: f64,
    rho_u_r: f64,
    e_r: f64,
    gamma: f64,
) -> (f64, f64, f64) {
    let p_l = cell_pressure(rho_l, rho_u_l, e_l, gamma);
    let p_r = cell_pressure(rho_r, rho_u_r, e_r, gamma);
    let u_l = rho_u_l / rho_l.max(1e-300);
    let u_r = rho_u_r / rho_r.max(1e-300);
    let c_l = (gamma * p_l.max(0.0) / rho_l.max(1e-300)).sqrt();
    let c_r = (gamma * p_r.max(0.0) / rho_r.max(1e-300)).sqrt();
    let alpha = (u_l.abs() + c_l).max(u_r.abs() + c_r);
    // Physical fluxes F(U) = (ρu, ρu²+p, u(E+p))
    let f1_l = rho_u_l;
    let f1_r = rho_u_r;
    let f2_l = rho_u_l * u_l + p_l;
    let f2_r = rho_u_r * u_r + p_r;
    let f3_l = u_l * (e_l + p_l);
    let f3_r = u_r * (e_r + p_r);
    // LLF flux: ½(F_L + F_R) - ½ α (U_R - U_L)
    let fhat1 = 0.5 * (f1_l + f1_r) - 0.5 * alpha * (rho_r - rho_l);
    let fhat2 = 0.5 * (f2_l + f2_r) - 0.5 * alpha * (rho_u_r - rho_u_l);
    let fhat3 = 0.5 * (f3_l + f3_r) - 0.5 * alpha * (e_r - e_l);
    (fhat1, fhat2, fhat3)
}

/// Compute the RHS of the semi-discretised NS system (flux divergence + viscous).
///
/// Uses ghost cells (transmissive BCs) for boundary flux computation.
fn compute_rhs(state: &Ns1dState, cfg: &Ns1dConfig, dx: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = cfg.nx;
    let gamma = cfg.gamma;
    let mu = cfg.mu;
    let kappa = cfg.kappa;
    let r_gas = cfg.r_gas;
    let inv_dx = 1.0 / dx;

    // Build extended arrays with one ghost cell on each side (transmissive).
    let mut rho_ext = vec![0.0_f64; nx + 2];
    let mut rhu_ext = vec![0.0_f64; nx + 2];
    let mut ene_ext = vec![0.0_f64; nx + 2];
    rho_ext[1..nx + 1].copy_from_slice(&state.rho[..nx]);
    rhu_ext[1..nx + 1].copy_from_slice(&state.rho_u[..nx]);
    ene_ext[1..nx + 1].copy_from_slice(&state.energy[..nx]);
    // Ghost cells: copy nearest interior cell (transmissive).
    rho_ext[0] = rho_ext[1];
    rhu_ext[0] = rhu_ext[1];
    ene_ext[0] = ene_ext[1];
    rho_ext[nx + 1] = rho_ext[nx];
    rhu_ext[nx + 1] = rhu_ext[nx];
    ene_ext[nx + 1] = ene_ext[nx];

    let mut drho = vec![0.0_f64; nx];
    let mut drhu = vec![0.0_f64; nx];
    let mut dene = vec![0.0_f64; nx];

    // Inviscid (Euler) fluxes using LLF
    for i in 0..nx {
        // Right interface i+½: between extended[i+1] (cell i) and extended[i+2] (cell i+1)
        let (fhat1_r, fhat2_r, fhat3_r) = llf_flux(
            rho_ext[i + 1],
            rhu_ext[i + 1],
            ene_ext[i + 1],
            rho_ext[i + 2],
            rhu_ext[i + 2],
            ene_ext[i + 2],
            gamma,
        );
        // Left interface i-½: between extended[i] (cell i-1) and extended[i+1] (cell i)
        let (fhat1_l, fhat2_l, fhat3_l) = llf_flux(
            rho_ext[i],
            rhu_ext[i],
            ene_ext[i],
            rho_ext[i + 1],
            rhu_ext[i + 1],
            ene_ext[i + 1],
            gamma,
        );
        drho[i] = -inv_dx * (fhat1_r - fhat1_l);
        drhu[i] = -inv_dx * (fhat2_r - fhat2_l);
        dene[i] = -inv_dx * (fhat3_r - fhat3_l);
    }

    // Viscous + conductive terms (central differences)
    if mu > 0.0 || kappa > 0.0 {
        let mut u_ext = vec![0.0_f64; nx + 2];
        let mut t_ext = vec![0.0_f64; nx + 2];
        for i in 0..nx + 2 {
            let r = rho_ext[i].max(1e-300);
            let p = cell_pressure(rho_ext[i], rhu_ext[i], ene_ext[i], gamma);
            u_ext[i] = rhu_ext[i] / r;
            t_ext[i] = cell_temperature(p, r, r_gas);
        }
        let inv_dx2 = inv_dx * inv_dx;
        for i in 0..nx {
            let ie = i + 1; // index in extended array
            // Viscous momentum: μ ∂²u/∂x²
            drhu[i] += mu * inv_dx2 * (u_ext[ie - 1] - 2.0 * u_ext[ie] + u_ext[ie + 1]);
            // Conductive energy: κ ∂²T/∂x²
            dene[i] += kappa * inv_dx2 * (t_ext[ie - 1] - 2.0 * t_ext[ie] + t_ext[ie + 1]);
        }
    }

    (drho, drhu, dene)
}

/// Compute stable time step from CFL condition.
fn compute_dt(state: &Ns1dState, cfg: &Ns1dConfig, dx: f64) -> f64 {
    let gamma = cfg.gamma;
    let mu = cfg.mu;
    let r_gas = cfg.r_gas;
    let mut max_speed = 1e-300_f64;
    for i in 0..cfg.nx {
        let rho = state.rho[i].max(1e-300);
        let p = cell_pressure(rho, state.rho_u[i], state.energy[i], gamma);
        let p_pos = p.max(0.0);
        let u = state.rho_u[i] / rho;
        let c = (gamma * p_pos / rho).sqrt();
        let local = u.abs() + c;
        if local > max_speed {
            max_speed = local;
        }
    }
    let dt_conv = cfg.cfl * dx / max_speed;
    if mu > 0.0 {
        let rho_min = state
            .rho
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let nu = mu / rho_min;
        let dt_visc = cfg.cfl * dx * dx / (2.0 * nu);
        // Thermal diffusion limit
        let rho_min_cv = rho_min * r_gas / (gamma - 1.0);
        let alpha_therm = cfg.kappa / rho_min_cv.max(1e-300);
        let dt_therm = if alpha_therm > 0.0 {
            cfg.cfl * dx * dx / (2.0 * alpha_therm)
        } else {
            dt_conv
        };
        dt_conv.min(dt_visc).min(dt_therm)
    } else {
        dt_conv
    }
}

/// Apply a forward-Euler update: returns U + dt * L(U) as new state.
fn euler_update(
    state: &Ns1dState,
    drho: &[f64],
    drhu: &[f64],
    dene: &[f64],
    dt: f64,
    new_t: f64,
) -> Ns1dState {
    let nx = state.rho.len();
    let mut rho = Vec::with_capacity(nx);
    let mut rho_u = Vec::with_capacity(nx);
    let mut energy = Vec::with_capacity(nx);
    for i in 0..nx {
        rho.push(state.rho[i] + dt * drho[i]);
        rho_u.push(state.rho_u[i] + dt * drhu[i]);
        energy.push(state.energy[i] + dt * dene[i]);
    }
    Ns1dState {
        rho,
        rho_u,
        energy,
        t: new_t,
        step: state.step + 1,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialise the state with **Sod shock-tube** initial conditions.
///
/// The diaphragm is placed at the midpoint of `[x_lo, x_hi]`.
/// Left state: `(ρ, u, p) = (1, 0, 1)`, right state: `(0.125, 0, 0.1)`.
/// Total energy is `E = p/(γ−1)` since velocity is zero initially.
pub fn ns1d_init_sod(cfg: &Ns1dConfig) -> Ns1dState {
    let nx = cfg.nx;
    let dx = (cfg.x_hi - cfg.x_lo) / nx as f64;
    let x_mid = 0.5 * (cfg.x_lo + cfg.x_hi);
    let mut rho = Vec::with_capacity(nx);
    let mut rho_u = Vec::with_capacity(nx);
    let mut energy = Vec::with_capacity(nx);
    for i in 0..nx {
        // Cell centre at x_lo + (i + 0.5)*dx
        let x = cfg.x_lo + (i as f64 + 0.5) * dx;
        if x < x_mid {
            // Left state: ρ=1, u=0, p=1  →  E = p/(γ-1)
            rho.push(1.0_f64);
            rho_u.push(0.0_f64);
            energy.push(1.0 / (cfg.gamma - 1.0));
        } else {
            // Right state: ρ=0.125, u=0, p=0.1
            rho.push(0.125_f64);
            rho_u.push(0.0_f64);
            energy.push(0.1 / (cfg.gamma - 1.0));
        }
    }
    Ns1dState {
        rho,
        rho_u,
        energy,
        t: 0.0,
        step: 0,
    }
}

/// Advance the state by one adaptive time step using **Heun's RK2** method.
///
/// The time step `dt` is chosen from the CFL condition at the current state
/// and clipped so as not to exceed `t_end`.
///
/// # Errors
///
/// Returns `PdeError::NumericalInstability` if any updated value is non-finite,
/// `PdeError::InvalidGrid` for `nx < 2`, or `InvalidParameter` for bad config.
pub fn ns1d_step(state: &Ns1dState, cfg: &Ns1dConfig) -> PdeResult<Ns1dState> {
    validate_cfg(cfg)?;
    let nx = cfg.nx;
    if state.rho.len() != nx || state.rho_u.len() != nx || state.energy.len() != nx {
        return Err(PdeError::ShapeMismatch {
            expected: vec![nx],
            got: vec![state.rho.len()],
        });
    }
    let dx = (cfg.x_hi - cfg.x_lo) / nx as f64;
    // Adaptive time step clipped to not exceed t_end.
    let dt_raw = compute_dt(state, cfg, dx);
    let remaining = cfg.t_end - state.t;
    if remaining <= 0.0 {
        return Ok(state.clone());
    }
    let dt = dt_raw.min(remaining);

    // --- Stage 1: predictor (forward Euler) ---
    let (drho0, drhu0, dene0) = compute_rhs(state, cfg, dx);
    let state_star = euler_update(state, &drho0, &drhu0, &dene0, dt, state.t + dt);

    // --- Stage 2: corrector (Heun) ---
    let (drho1, drhu1, dene1) = compute_rhs(&state_star, cfg, dx);

    let mut rho_new = Vec::with_capacity(nx);
    let mut rhu_new = Vec::with_capacity(nx);
    let mut ene_new = Vec::with_capacity(nx);
    for i in 0..nx {
        // Heun: U^{n+1} = 0.5 * (U^n + U* + dt * L(U*))
        // which equals: U^{n+1} = U^n + 0.5*dt*(L(U^n) + L(U*))
        let r = 0.5 * (state.rho[i] + state_star.rho[i] + dt * drho1[i]);
        let m = 0.5 * (state.rho_u[i] + state_star.rho_u[i] + dt * drhu1[i]);
        let e = 0.5 * (state.energy[i] + state_star.energy[i] + dt * dene1[i]);
        if !r.is_finite() || !m.is_finite() || !e.is_finite() {
            return Err(PdeError::NumericalInstability(format!(
                "ns1d: non-finite value at cell {i} after RK2 step"
            )));
        }
        rho_new.push(r);
        rhu_new.push(m);
        ene_new.push(e);
    }

    Ok(Ns1dState {
        rho: rho_new,
        rho_u: rhu_new,
        energy: ene_new,
        t: state.t + dt,
        step: state.step + 1,
    })
}

/// Run the solver from Sod initial conditions until `t >= cfg.t_end`
/// or `cfg.max_steps` is exhausted.
///
/// # Errors
///
/// Propagates any error from [`ns1d_step`].
pub fn ns1d_solve(cfg: &Ns1dConfig) -> PdeResult<Ns1dState> {
    validate_cfg(cfg)?;
    let mut state = ns1d_init_sod(cfg);
    for _ in 0..cfg.max_steps {
        if state.t >= cfg.t_end - 1e-14 * cfg.t_end.abs() {
            break;
        }
        state = ns1d_step(&state, cfg)?;
    }
    Ok(state)
}

/// Compute the pressure field from a solution state.
///
/// Returns a `Vec<f64>` of length `nx` with
/// `p[i] = (γ−1)(E_i − ½(ρu)²_i / ρ_i)`.
pub fn ns1d_pressure(state: &Ns1dState, cfg: &Ns1dConfig) -> Vec<f64> {
    state
        .rho
        .iter()
        .zip(state.rho_u.iter())
        .zip(state.energy.iter())
        .map(|((&r, &m), &e)| cell_pressure(r, m, e, cfg.gamma))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sod_cfg() -> Ns1dConfig {
        Ns1dConfig {
            nx: 64,
            x_lo: 0.0,
            x_hi: 1.0,
            gamma: 1.4,
            mu: 0.0,
            kappa: 0.0,
            r_gas: 287.0,
            cfl: 0.4,
            max_steps: 10_000,
            t_end: 0.2,
        }
    }

    #[test]
    fn init_sod_density_positive() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        for &r in &state.rho {
            assert!(r > 0.0, "density must be positive, got {r}");
        }
    }

    #[test]
    fn init_sod_energy_positive() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        for &e in &state.energy {
            assert!(e > 0.0, "energy must be positive, got {e}");
        }
    }

    #[test]
    fn step_does_not_explode() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        let next = ns1d_step(&state, &cfg).expect("step ok");
        for i in 0..cfg.nx {
            assert!(next.rho[i] > 0.0, "density went non-positive at {i}");
            assert!(next.rho[i].is_finite(), "density non-finite at {i}");
            assert!(next.rho_u[i].is_finite(), "momentum non-finite at {i}");
            assert!(next.energy[i].is_finite(), "energy non-finite at {i}");
        }
    }

    #[test]
    fn solve_runs_without_nan() {
        let mut cfg = sod_cfg();
        cfg.t_end = 0.1;
        let state = ns1d_solve(&cfg).expect("solve ok");
        for i in 0..cfg.nx {
            assert!(state.rho[i].is_finite(), "rho[{i}] is nan/inf");
            assert!(state.rho_u[i].is_finite(), "rho_u[{i}] is nan/inf");
            assert!(state.energy[i].is_finite(), "energy[{i}] is nan/inf");
        }
    }

    #[test]
    fn pressure_positive() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        let p = ns1d_pressure(&state, &cfg);
        for (i, &pi) in p.iter().enumerate() {
            assert!(pi > 0.0, "pressure must be positive at {i}, got {pi}");
        }
    }

    #[test]
    fn sod_density_ratio() {
        // After t=0.2 the shock has moved right; left avg density > right avg density.
        let mut cfg = sod_cfg();
        cfg.t_end = 0.2;
        let state = ns1d_solve(&cfg).expect("solve ok");
        let left_avg: f64 = state.rho[..cfg.nx / 4].iter().sum::<f64>() / (cfg.nx / 4) as f64;
        let right_avg: f64 = state.rho[3 * cfg.nx / 4..].iter().sum::<f64>() / (cfg.nx / 4) as f64;
        assert!(
            left_avg > right_avg,
            "expected left density > right density, got {left_avg} vs {right_avg}"
        );
    }

    #[test]
    fn momentum_initially_zero() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        for &m in &state.rho_u {
            assert_eq!(m, 0.0, "initial momentum must be zero");
        }
    }

    #[test]
    fn nx_too_small_error() {
        let mut cfg = sod_cfg();
        cfg.nx = 1;
        // Build a 1-cell state manually
        let state = Ns1dState {
            rho: vec![1.0],
            rho_u: vec![0.0],
            energy: vec![2.5],
            t: 0.0,
            step: 0,
        };
        let result = ns1d_step(&state, &cfg);
        assert!(matches!(result, Err(PdeError::InvalidGrid(_))));
    }

    #[test]
    fn time_advances() {
        let cfg = sod_cfg();
        let mut state = ns1d_init_sod(&cfg);
        for _ in 0..5 {
            state = ns1d_step(&state, &cfg).expect("step ok");
        }
        assert!(state.t > 0.0, "time must advance, got t={}", state.t);
        assert_eq!(state.step, 5);
    }

    #[test]
    fn cfl_condition() {
        // Verify dt computed internally > 0 via t advancement in one step.
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        let next = ns1d_step(&state, &cfg).expect("step ok");
        assert!(next.t > 0.0, "time must advance: t={}", next.t);
    }

    #[test]
    fn energy_positive_after_step() {
        let cfg = sod_cfg();
        let state = ns1d_init_sod(&cfg);
        let next = ns1d_step(&state, &cfg).expect("step ok");
        for (i, &e) in next.energy.iter().enumerate() {
            assert!(e > 0.0, "energy non-positive at cell {i}: {e}");
        }
    }

    #[test]
    fn gamma_out_of_range_error() {
        let mut cfg = sod_cfg();
        cfg.gamma = 0.9;
        let state = ns1d_init_sod(&cfg);
        let result = ns1d_step(&state, &cfg);
        assert!(
            matches!(result, Err(PdeError::InvalidParameter { .. })),
            "expected InvalidParameter for gamma < 1.0"
        );
    }
}
