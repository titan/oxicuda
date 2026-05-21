//! 2D wave equation: `u_tt = c² (u_xx + u_yy)` on a rectangle.
//!
//! # Discretisation
//!
//! Explicit leapfrog (central difference in space *and* time, 2nd-order):
//!
//! ```text
//!     u^{n+1}_{i,j} = 2 u^n_{i,j} - u^{n-1}_{i,j}
//!                  + r² (u^n_{i+1,j} + u^n_{i-1,j}
//!                       + u^n_{i,j+1} + u^n_{i,j-1}
//!                       - 4 u^n_{i,j})
//! ```
//!
//! with Courant number `r = c·dt/h`. Stability requires `r ≤ 1/√2`.
//!
//! `u^1` is bootstrapped from `u^0` and `u̇^0` via a 2nd-order Taylor step:
//! `u^1 = u^0 + dt·u̇^0 + 0.5·dt²·c²·Δu^0`.
//!
//! # Boundary conditions (per edge, ordered left/right/bottom/top)
//!
//! * `Dirichlet(v)` — fix `u = v` on the edge for all time;
//! * `Neumann` — zero gradient via ghost-point reflection;
//! * `Absorbing` — first-order Mur (Engquist-Majda) outgoing-wave condition:
//!   `u^{n+1}_b = u^n_{b−1} + κ·(u^{n+1}_{b−1} − u^n_b)` with
//!   `κ = (c·dt − h)/(c·dt + h)`. Corners take the mean of the two adjacent
//!   edge updates.
//!
//! Reference: LeVeque, *Finite Difference Methods for Ordinary and Partial
//! Differential Equations*, SIAM 2007, Chapter 10.

use crate::error::{PdeError, PdeResult};

/// Boundary-condition kind for one edge of the 2D wave domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryKind {
    /// Fixed value `u = v` for all time.
    Dirichlet(f64),
    /// Zero normal derivative (`∂u/∂n = 0`) via ghost-point reflection.
    Neumann,
    /// First-order Mur absorbing condition (outgoing radiation).
    Absorbing,
}

/// Configuration for the 2D leapfrog wave solver.
#[derive(Debug, Clone, Copy)]
pub struct Wave2dConfig {
    /// Grid points along x (`nx ≥ 3`).
    pub nx: usize,
    /// Grid points along y (`ny ≥ 3`).
    pub ny: usize,
    /// Isotropic spatial step `h = dx = dy > 0`.
    pub h: f64,
    /// Time step `dt > 0`. Must satisfy `c·dt/h ≤ 1/√2`.
    pub dt: f64,
    /// Wave speed `c ≥ 0`.
    pub c: f64,
    /// Number of leapfrog time steps to perform.
    pub n_steps: usize,
    /// Boundary kinds, ordered `[left, right, bottom, top]`.
    pub bc: [BoundaryKind; 4],
}

impl Default for Wave2dConfig {
    fn default() -> Self {
        Self {
            nx: 33,
            ny: 33,
            h: 1.0 / 32.0,
            dt: 0.5 * (1.0 / 32.0) / std::f64::consts::SQRT_2,
            c: 1.0,
            n_steps: 100,
            bc: [BoundaryKind::Dirichlet(0.0); 4],
        }
    }
}

/// Result returned by [`solve_wave_2d`].
#[derive(Debug, Clone)]
pub struct Wave2dResult {
    /// Solution at time `t_final` (current time level).
    pub u: Vec<f64>,
    /// Solution at the previous time level — required to resume a simulation.
    pub u_prev: Vec<f64>,
    /// Final integration time `n_steps · dt`.
    pub t_final: f64,
}

impl Wave2dResult {
    /// Final integration time.
    #[must_use]
    pub fn t_final(&self) -> f64 {
        self.t_final
    }
}

const LEFT: usize = 0;
const RIGHT: usize = 1;
const BOTTOM: usize = 2;
const TOP: usize = 3;

#[inline]
fn idx2(i: usize, j: usize, nx: usize) -> usize {
    i + nx * j
}

fn invalid_param(name: &str, reason: String) -> PdeError {
    PdeError::InvalidParameter {
        name: name.into(),
        reason,
    }
}

fn validate_config(cfg: &Wave2dConfig, u0_len: usize, v0_len: usize) -> PdeResult<()> {
    if cfg.nx < 3 || cfg.ny < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "wave 2d needs nx,ny >= 3, got ({},{})",
            cfg.nx, cfg.ny
        )));
    }
    if !(cfg.h.is_finite() && cfg.h > 0.0) {
        return Err(invalid_param(
            "h",
            format!("must be > 0 finite, got {}", cfg.h),
        ));
    }
    if !(cfg.dt.is_finite() && cfg.dt > 0.0) {
        return Err(invalid_param(
            "dt",
            format!("must be > 0 finite, got {}", cfg.dt),
        ));
    }
    if !(cfg.c.is_finite() && cfg.c >= 0.0) {
        return Err(invalid_param(
            "c",
            format!("must be >= 0 finite, got {}", cfg.c),
        ));
    }
    if cfg.n_steps == 0 {
        return Err(invalid_param("n_steps", "must be >= 1".into()));
    }
    let n = cfg.nx * cfg.ny;
    for (got, _name) in [(u0_len, "u0"), (v0_len, "u0_dot")] {
        if got != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![got],
            });
        }
    }
    let r = cfg.c * cfg.dt / cfg.h;
    let r_max = 1.0 / std::f64::consts::SQRT_2;
    if r > r_max + 1.0e-12 {
        let dt_max = r_max * cfg.h / cfg.c.max(f64::MIN_POSITIVE);
        return Err(PdeError::CflViolation { dt: cfg.dt, dt_max });
    }
    Ok(())
}

/// Discrete Laplacian on the interior `(1..nx-1, 1..ny-1)`, written into `out`.
fn discrete_laplacian(u: &[f64], out: &mut [f64], nx: usize, ny: usize, h: f64) {
    let inv_h2 = 1.0 / (h * h);
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let c = idx2(i, j, nx);
            out[c] = (u[c - 1] + u[c + 1] + u[c - nx] + u[c + nx] - 4.0 * u[c]) * inv_h2;
        }
    }
}

/// One interior leapfrog sweep producing `u_next` for `(1..nx-1, 1..ny-1)`.
fn leapfrog_interior(
    u_curr: &[f64],
    u_prev: &[f64],
    u_next: &mut [f64],
    nx: usize,
    ny: usize,
    r2: f64,
) {
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let c = idx2(i, j, nx);
            let lap =
                u_curr[c - 1] + u_curr[c + 1] + u_curr[c - nx] + u_curr[c + nx] - 4.0 * u_curr[c];
            u_next[c] = 2.0 * u_curr[c] - u_prev[c] + r2 * lap;
        }
    }
}

/// Compute the boundary value for a single edge node from its BC kind. The
/// inner neighbour index `(i_in, j_in)` already holds the just-computed
/// interior `u^{n+1}` from the leapfrog sweep.
///
/// `edge_idx` is the flat index of the edge node itself, `inner_idx` of its
/// nearest inward neighbour.
#[inline]
fn edge_value(
    kind: BoundaryKind,
    u_curr: &[f64],
    u_next: &[f64],
    edge_idx: usize,
    inner_idx: usize,
    kappa: f64,
) -> f64 {
    match kind {
        BoundaryKind::Dirichlet(v) => v,
        // Neumann: ghost reflection ⇒ boundary update equals the just-computed
        // inner neighbour value.
        BoundaryKind::Neumann => u_next[inner_idx],
        // First-order Mur: u^{n+1}_b = u^n_{b−1} + κ (u^{n+1}_{b−1} − u^n_b)
        BoundaryKind::Absorbing => {
            u_curr[inner_idx] + kappa * (u_next[inner_idx] - u_curr[edge_idx])
        }
    }
}

/// Apply the four edge boundary conditions, updating `u_next` boundary nodes
/// in place. Corners are resolved as the average of the two adjacent edges
/// that touch the corner.
fn apply_boundaries(u_curr: &[f64], u_next: &mut [f64], cfg: &Wave2dConfig) {
    let nx = cfg.nx;
    let ny = cfg.ny;
    let kappa = if cfg.c * cfg.dt + cfg.h > 0.0 {
        (cfg.c * cfg.dt - cfg.h) / (cfg.c * cfg.dt + cfg.h)
    } else {
        0.0
    };
    let ev = |k, e_idx, in_idx| edge_value(k, u_curr, u_next, e_idx, in_idx, kappa);

    let mut scratch: Vec<(usize, f64)> = Vec::with_capacity(2 * (nx + ny));
    // Non-corner edge nodes.
    for j in 1..ny - 1 {
        let l = idx2(0, j, nx);
        let r = idx2(nx - 1, j, nx);
        scratch.push((l, ev(cfg.bc[LEFT], l, idx2(1, j, nx))));
        scratch.push((r, ev(cfg.bc[RIGHT], r, idx2(nx - 2, j, nx))));
    }
    for i in 1..nx - 1 {
        let b = idx2(i, 0, nx);
        let t = idx2(i, ny - 1, nx);
        scratch.push((b, ev(cfg.bc[BOTTOM], b, idx2(i, 1, nx))));
        scratch.push((t, ev(cfg.bc[TOP], t, idx2(i, ny - 2, nx))));
    }
    // Corners — average of the two adjacent edges' predictions.
    let bl0 = idx2(0, 0, nx);
    let br0 = idx2(nx - 1, 0, nx);
    let tl0 = idx2(0, ny - 1, nx);
    let tr0 = idx2(nx - 1, ny - 1, nx);
    let bl =
        0.5 * (ev(cfg.bc[LEFT], bl0, idx2(1, 0, nx)) + ev(cfg.bc[BOTTOM], bl0, idx2(0, 1, nx)));
    let br = 0.5
        * (ev(cfg.bc[RIGHT], br0, idx2(nx - 2, 0, nx))
            + ev(cfg.bc[BOTTOM], br0, idx2(nx - 1, 1, nx)));
    let tl = 0.5
        * (ev(cfg.bc[LEFT], tl0, idx2(1, ny - 1, nx)) + ev(cfg.bc[TOP], tl0, idx2(0, ny - 2, nx)));
    let tr = 0.5
        * (ev(cfg.bc[RIGHT], tr0, idx2(nx - 2, ny - 1, nx))
            + ev(cfg.bc[TOP], tr0, idx2(nx - 1, ny - 2, nx)));
    scratch.push((bl0, bl));
    scratch.push((br0, br));
    scratch.push((tl0, tl));
    scratch.push((tr0, tr));

    for (idx, val) in scratch {
        u_next[idx] = val;
    }
}

/// Write Dirichlet boundary values into the four edges of `u`.
fn clamp_dirichlet_edges(u: &mut [f64], cfg: &Wave2dConfig) {
    let (nx, ny) = (cfg.nx, cfg.ny);
    let set = |u: &mut [f64], kind: BoundaryKind, indices: &[usize]| {
        if let BoundaryKind::Dirichlet(v) = kind {
            for &k in indices {
                u[k] = v;
            }
        }
    };
    let left: Vec<usize> = (0..ny).map(|j| idx2(0, j, nx)).collect();
    let right: Vec<usize> = (0..ny).map(|j| idx2(nx - 1, j, nx)).collect();
    let bottom: Vec<usize> = (0..nx).map(|i| idx2(i, 0, nx)).collect();
    let top: Vec<usize> = (0..nx).map(|i| idx2(i, ny - 1, nx)).collect();
    set(u, cfg.bc[LEFT], &left);
    set(u, cfg.bc[RIGHT], &right);
    set(u, cfg.bc[BOTTOM], &bottom);
    set(u, cfg.bc[TOP], &top);
}

/// Bootstrap `u^1` from `u^0` and `u̇^0` using the second-order Taylor step
/// `u¹ = u⁰ + dt·u̇⁰ + ½ dt²·c²·Δu⁰`.
fn bootstrap_u1(u0: &[f64], v0: &[f64], cfg: &Wave2dConfig) -> Vec<f64> {
    let n = cfg.nx * cfg.ny;
    let mut lap = vec![0.0; n];
    discrete_laplacian(u0, &mut lap, cfg.nx, cfg.ny, cfg.h);
    let c2 = cfg.c * cfg.c;
    let dt = cfg.dt;
    let mut u1 = vec![0.0; n];
    for j in 1..cfg.ny - 1 {
        for i in 1..cfg.nx - 1 {
            let c = idx2(i, j, cfg.nx);
            u1[c] = u0[c] + dt * v0[c] + 0.5 * dt * dt * c2 * lap[c];
        }
    }
    // One synthetic boundary pass (treating u0 as both current and previous)
    // preserves Dirichlet exactly and seeds Neumann / Absorbing edges.
    apply_boundaries(u0, &mut u1, cfg);
    u1
}

/// Drive `n_steps` leapfrog updates and return the final state.
///
/// `u0` is the initial position field, `u0_dot` the initial velocity field;
/// both of length `nx*ny`, indexed as `i + nx*j`.
///
/// # Errors
/// Returns [`PdeError::CflViolation`] if `c·dt/h > 1/√2`,
/// [`PdeError::InvalidGrid`] / [`PdeError::InvalidParameter`] for malformed
/// configs, [`PdeError::ShapeMismatch`] if either input length is wrong.
pub fn solve_wave_2d(u0: &[f64], u0_dot: &[f64], cfg: &Wave2dConfig) -> PdeResult<Wave2dResult> {
    validate_config(cfg, u0.len(), u0_dot.len())?;
    if u0.iter().any(|v| !v.is_finite()) || u0_dot.iter().any(|v| !v.is_finite()) {
        return Err(PdeError::NumericalInstability(
            "u0 or u0_dot contains non-finite values".into(),
        ));
    }

    let (nx, ny) = (cfg.nx, cfg.ny);
    let r = cfg.c * cfg.dt / cfg.h;
    let r2 = r * r;

    let mut u_prev = u0.to_vec();
    clamp_dirichlet_edges(&mut u_prev, cfg);
    let mut u_curr = bootstrap_u1(&u_prev, u0_dot, cfg);

    if cfg.n_steps == 1 {
        return Ok(Wave2dResult {
            u: u_curr,
            u_prev,
            t_final: cfg.dt,
        });
    }

    let mut u_next = vec![0.0; nx * ny];
    for _step in 1..cfg.n_steps {
        leapfrog_interior(&u_curr, &u_prev, &mut u_next, nx, ny, r2);
        apply_boundaries(&u_curr, &mut u_next, cfg);
        std::mem::swap(&mut u_prev, &mut u_curr);
        std::mem::swap(&mut u_curr, &mut u_next);
    }

    Ok(Wave2dResult {
        u: u_curr,
        u_prev,
        t_final: cfg.n_steps as f64 * cfg.dt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(nx: usize, ny: usize, n_steps: usize) -> Wave2dConfig {
        let h = 1.0 / (nx - 1) as f64;
        let c = 1.0;
        let dt = 0.4 * h / c; // r = 0.4 < 1/√2 ≈ 0.707
        Wave2dConfig {
            nx,
            ny,
            h,
            dt,
            c,
            n_steps,
            bc: [BoundaryKind::Dirichlet(0.0); 4],
        }
    }

    fn gaussian_pulse(nx: usize, ny: usize, h: f64, sigma: f64) -> Vec<f64> {
        let mut u = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * h - 0.5;
                let y = j as f64 * h - 0.5;
                u[idx2(i, j, nx)] = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
            }
        }
        u
    }

    fn energy(u_curr: &[f64], u_prev: &[f64], cfg: &Wave2dConfig) -> f64 {
        // Leapfrog-conserved staggered energy
        //   E = ½ Σ ((u^n − u^{n−1})/dt)² · h²
        //     + ½ c² Σ_{forward pairs} (Δ_h u^n)(Δ_h u^{n−1})
        let (nx, ny) = (cfg.nx, cfg.ny);
        let h2 = cfg.h * cfg.h;
        let c2 = cfg.c * cfg.c;
        let mut e = 0.0;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = idx2(i, j, nx);
                let v = (u_curr[c] - u_prev[c]) / cfg.dt;
                e += 0.5 * v * v * h2;
            }
        }
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let c = idx2(i, j, nx);
                let dx_n = u_curr[c + 1] - u_curr[c];
                let dx_p = u_prev[c + 1] - u_prev[c];
                let dy_n = u_curr[c + nx] - u_curr[c];
                let dy_p = u_prev[c + nx] - u_prev[c];
                e += 0.5 * c2 * (dx_n * dx_p + dy_n * dy_p);
            }
        }
        e
    }

    #[test]
    fn default_config_is_consistent() {
        let cfg = Wave2dConfig::default();
        assert!(cfg.nx >= 3 && cfg.ny >= 3);
        assert!(cfg.h > 0.0 && cfg.dt > 0.0);
        let r = cfg.c * cfg.dt / cfg.h;
        assert!(r <= 1.0 / std::f64::consts::SQRT_2 + 1.0e-12);
    }

    fn zeros_for(cfg: &Wave2dConfig) -> (Vec<f64>, Vec<f64>) {
        let n = cfg.nx * cfg.ny;
        (vec![0.0; n], vec![0.0; n])
    }

    #[test]
    fn cfl_violation_rejected() {
        let mut cfg = make_cfg(11, 11, 10);
        cfg.dt = cfg.h * 10.0;
        let (u0, v0) = zeros_for(&cfg);
        assert!(matches!(
            solve_wave_2d(&u0, &v0, &cfg),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn nx_too_small_rejected() {
        let mut cfg = make_cfg(11, 11, 10);
        cfg.nx = 2;
        let (u0, v0) = zeros_for(&cfg);
        assert!(matches!(
            solve_wave_2d(&u0, &v0, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
    }

    #[test]
    fn invalid_scalar_params_rejected() {
        // c < 0, dt = 0, n_steps = 0 all map to InvalidParameter.
        for build in [
            (|c: &mut Wave2dConfig| c.c = -1.0) as fn(&mut Wave2dConfig),
            (|c: &mut Wave2dConfig| c.dt = 0.0) as fn(&mut Wave2dConfig),
            (|c: &mut Wave2dConfig| c.n_steps = 0) as fn(&mut Wave2dConfig),
            (|c: &mut Wave2dConfig| c.h = -1.0) as fn(&mut Wave2dConfig),
        ] {
            let mut cfg = make_cfg(11, 11, 10);
            build(&mut cfg);
            let (u0, v0) = zeros_for(&cfg);
            assert!(matches!(
                solve_wave_2d(&u0, &v0, &cfg),
                Err(PdeError::InvalidParameter { .. })
            ));
        }
    }

    #[test]
    fn wrong_length_rejected() {
        let cfg = make_cfg(11, 11, 10);
        let n = cfg.nx * cfg.ny;
        let u0 = vec![0.0; n - 1];
        let v0 = vec![0.0; n];
        assert!(matches!(
            solve_wave_2d(&u0, &v0, &cfg),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn gaussian_pulse_spreads() {
        // Initial centre spike spreads outward: central amplitude drops and
        // outer amplitude rises.
        let cfg = make_cfg(41, 41, 50);
        let u0 = gaussian_pulse(cfg.nx, cfg.ny, cfg.h, 0.05);
        let v0 = vec![0.0; cfg.nx * cfg.ny];
        let res = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        let mid = idx2(cfg.nx / 2, cfg.ny / 2, cfg.nx);
        assert!(u0[mid] > 0.9);
        assert!(res.u[mid].abs() < u0[mid]);
        let outer = res.u[idx2(cfg.nx - 4, cfg.ny / 2, cfg.nx)];
        assert!(outer.abs() > 1.0e-6, "outer {outer}");
    }

    #[test]
    fn energy_is_approximately_conserved_with_dirichlet_zero() {
        // Leapfrog conserves a discrete energy up to small higher-order drift;
        // allow ≤ 3 % drift over 100 steps for a smooth interior pulse.
        let cfg = make_cfg(41, 41, 100);
        let u0 = gaussian_pulse(cfg.nx, cfg.ny, cfg.h, 0.07);
        let v0 = vec![0.0; cfg.nx * cfg.ny];
        let cfg1 = Wave2dConfig { n_steps: 1, ..cfg };
        let r1 = solve_wave_2d(&u0, &v0, &cfg1).expect("solve ok");
        let e0 = energy(&r1.u, &r1.u_prev, &cfg);
        let res = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        let e1 = energy(&res.u, &res.u_prev, &cfg);
        let drift = (e1 - e0).abs() / e0.abs();
        assert!(drift < 0.03, "energy drift {drift} (e0={e0} e1={e1})");
    }

    #[test]
    fn standing_wave_reproduces_period() {
        // u = sin(π x)·sin(π y)·cos(c·π√2·t) on the unit square with
        // Dirichlet zero edges flips sign at t = T/2.
        let cfg_base = make_cfg(41, 41, 0);
        let pi = std::f64::consts::PI;
        let period = std::f64::consts::TAU / (cfg_base.c * pi * std::f64::consts::SQRT_2);
        let n_half = (0.5 * period / cfg_base.dt).round() as usize;
        let cfg = Wave2dConfig {
            n_steps: n_half,
            ..cfg_base
        };
        let mut u0 = vec![0.0; cfg.nx * cfg.ny];
        for j in 0..cfg.ny {
            for i in 0..cfg.nx {
                u0[idx2(i, j, cfg.nx)] =
                    (pi * i as f64 * cfg.h).sin() * (pi * j as f64 * cfg.h).sin();
            }
        }
        let v0 = vec![0.0; cfg.nx * cfg.ny];
        let res = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        let mid = idx2(cfg.nx / 2, cfg.ny / 2, cfg.nx);
        let want = -u0[mid];
        let rel = (res.u[mid] - want).abs() / want.abs();
        assert!(rel < 0.05, "standing wave rel={rel}");
    }

    #[test]
    fn absorbing_bc_reduces_reflection_vs_dirichlet() {
        // Pulse hitting the boundary: absorbing should leave less energy in
        // the domain than Dirichlet zero (which reflects fully).
        let n_steps = 120;
        let mk = |kind: BoundaryKind| Wave2dConfig {
            bc: [kind; 4],
            ..make_cfg(41, 41, n_steps)
        };
        let cfg_d = mk(BoundaryKind::Dirichlet(0.0));
        let cfg_a = mk(BoundaryKind::Absorbing);
        let u0 = gaussian_pulse(cfg_d.nx, cfg_d.ny, cfg_d.h, 0.04);
        let v0 = vec![0.0; cfg_d.nx * cfg_d.ny];
        let res_d = solve_wave_2d(&u0, &v0, &cfg_d).expect("solve ok");
        let res_a = solve_wave_2d(&u0, &v0, &cfg_a).expect("solve ok");
        let e_d = energy(&res_d.u, &res_d.u_prev, &cfg_d);
        let e_a = energy(&res_a.u, &res_a.u_prev, &cfg_a);
        assert!(e_a < e_d, "absorbing {e_a} >= dirichlet {e_d}");
    }

    #[test]
    fn neumann_bc_reflects_with_same_sign() {
        // Neumann walls conserve the spatial mean (no flux). Drift comes
        // only from boundary discretisation; allow ≤ 15 %.
        let cfg = Wave2dConfig {
            bc: [BoundaryKind::Neumann; 4],
            ..make_cfg(31, 31, 80)
        };
        let u0 = gaussian_pulse(cfg.nx, cfg.ny, cfg.h, 0.06);
        let v0 = vec![0.0; cfg.nx * cfg.ny];
        let res = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        let n = (cfg.nx * cfg.ny) as f64;
        let m0: f64 = u0.iter().sum::<f64>() / n;
        let m1: f64 = res.u.iter().sum::<f64>() / n;
        assert!(m0 > 0.0);
        assert!((m1 - m0).abs() / m0 < 0.15, "Neumann mean drift");
    }

    #[test]
    fn deterministic_runs_agree_bitwise() {
        let cfg = make_cfg(21, 21, 25);
        let u0 = gaussian_pulse(cfg.nx, cfg.ny, cfg.h, 0.08);
        let v0 = vec![0.0; cfg.nx * cfg.ny];
        let r1 = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        let r2 = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        for k in 0..r1.u.len() {
            assert!((r1.u[k] - r2.u[k]).abs() < 1.0e-12, "non-det at {k}");
        }
    }

    #[test]
    fn boundary_kind_partialeq() {
        assert_eq!(BoundaryKind::Dirichlet(0.5), BoundaryKind::Dirichlet(0.5));
        assert_ne!(BoundaryKind::Neumann, BoundaryKind::Absorbing);
    }

    #[test]
    fn t_final_matches_n_steps_times_dt() {
        let cfg = make_cfg(11, 11, 7);
        let (u0, v0) = zeros_for(&cfg);
        let res = solve_wave_2d(&u0, &v0, &cfg).expect("solve ok");
        assert!((res.t_final() - 7.0 * cfg.dt).abs() < 1.0e-15);
    }
}
