//! Maxwell's equations by the finite-difference time-domain (FDTD) method on a
//! Yee grid.
//!
//! # 1-D (`Ez`, `Hy`)
//!
//! Electric field `Ez` lives on the integer grid `x_i = i·Δx`; magnetic field
//! `Hy` lives on the staggered half-grid `x_{i+1/2}`, and the two are advanced
//! by a leapfrog half-step apart in time (Yee's scheme):
//!
//! ```text
//!   Hy_{i+1/2}^{n+1/2} = Hy_{i+1/2}^{n-1/2} + (Δt/μΔx)(Ez_{i+1}^n − Ez_i^n)
//!   Ez_i^{n+1}         = Ez_i^n            + (Δt/εΔx)(Hy_{i+1/2}^{n+1/2} − Hy_{i-1/2}^{n+1/2})
//! ```
//!
//! # 2-D transverse-magnetic (`Ez`, `Hx`, `Hy`)
//!
//! The TMz mode `(ε ∂Ez/∂t = ∂Hy/∂x − ∂Hx/∂y, μ ∂Hx/∂t = −∂Ez/∂y,
//! μ ∂Hy/∂t = ∂Ez/∂x)` is discretised on the standard 2-D Yee cell.
//!
//! # Courant stability
//!
//! Propagation is stable iff `c Δt √(Σ_d 1/Δx_d²) ≤ 1`, i.e. `c Δt/Δx ≤ 1` in 1-D
//! and `≤ 1/√2` for a square 2-D cell, where `c = 1/√(εμ)`. A larger step is
//! rejected with [`PdeError::CflViolation`]. At the 1-D "magic" step `cΔt = Δx`
//! the scheme is dispersion-free and translates a pulse exactly one cell per step.
//!
//! # Energy
//!
//! The lossless leapfrog conserves the staggered electromagnetic energy
//!
//! ```text
//!   U^n = ½ Δx Σ_i [ ε (Ez_i^n)² + μ Hy_{i+1/2}^{n-1/2} Hy_{i+1/2}^{n+1/2} ]
//! ```
//!
//! (and its 2-D analogue) — the discrete Hamiltonian — to round-off, exposed via
//! [`Maxwell1d::energy`] / [`Maxwell2dTm::energy`].
//!
//! Reference: Taflove & Hagness, *Computational Electrodynamics: The FDTD
//! Method*, 3rd ed., Artech House 2005.

use crate::bc::periodic::wrap_index;
use crate::error::{PdeError, PdeResult};

/// Relative tolerance applied to the Courant stability bound.
const COURANT_TOL: f64 = 1.0e-12;

/// Boundary condition for the 1-D FDTD solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxwellBoundary1d {
    /// Periodic torus.
    Periodic,
    /// Perfect electric conductor (PEC): `Ez = 0` at both end nodes.
    Pec,
}

/// 1-D Yee-grid FDTD solver for `(Ez, Hy)`.
#[derive(Debug, Clone)]
pub struct Maxwell1d {
    /// Electric permittivity `ε > 0`.
    pub epsilon: f64,
    /// Magnetic permeability `μ > 0`.
    pub mu: f64,
    /// Uniform grid spacing `Δx > 0`.
    pub dx: f64,
    /// Number of `Ez` nodes (`n ≥ 3`).
    pub n: usize,
    /// Boundary condition.
    pub boundary: MaxwellBoundary1d,
}

/// Leapfrog state: `Ez` at time level `n`, `Hy` at the half level `n−1/2`.
#[derive(Debug, Clone)]
pub struct MaxwellState1d {
    /// Electric field `Ez_i^n` (length `n`).
    pub ez: Vec<f64>,
    /// Magnetic field `Hy_{i+1/2}^{n−1/2}` (length `n`; slot `i` is the half-node `i+1/2`).
    pub hy: Vec<f64>,
    /// Current time `t = n·Δt`.
    pub t: f64,
    /// Time step that produced this state.
    pub dt: f64,
}

impl Maxwell1d {
    /// Build a solver, validating `ε, μ > 0`, `Δx > 0`, `n ≥ 3`.
    pub fn new(
        epsilon: f64,
        mu: f64,
        dx: f64,
        n: usize,
        boundary: MaxwellBoundary1d,
    ) -> PdeResult<Self> {
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "epsilon".into(),
                reason: format!("must be finite and > 0, got {epsilon}"),
            });
        }
        if !(mu.is_finite() && mu > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "mu".into(),
                reason: format!("must be finite and > 0, got {mu}"),
            });
        }
        if !(dx.is_finite() && dx > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dx".into(),
                reason: format!("must be finite and > 0, got {dx}"),
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "1-D FDTD requires n >= 3, got {n}"
            )));
        }
        Ok(Self {
            epsilon,
            mu,
            dx,
            n,
            boundary,
        })
    }

    /// Speed of light in the medium `c = 1/√(εμ)`.
    #[must_use]
    pub fn light_speed(&self) -> f64 {
        1.0 / (self.epsilon * self.mu).sqrt()
    }

    /// Wave impedance `η = √(μ/ε)`.
    #[must_use]
    pub fn impedance(&self) -> f64 {
        (self.mu / self.epsilon).sqrt()
    }

    /// Largest stable step from the Courant condition `cΔt/Δx ≤ 1`.
    #[must_use]
    pub fn courant_dt_max(&self) -> f64 {
        self.dx / self.light_speed()
    }

    fn check_step(&self, dt: f64) -> PdeResult<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be finite and > 0, got {dt}"),
            });
        }
        let dt_max = self.courant_dt_max();
        if dt > dt_max * (1.0 + COURANT_TOL) {
            return Err(PdeError::CflViolation { dt, dt_max });
        }
        Ok(())
    }

    /// Advance `Hy` by one half step from `Ez`, returning the new field.
    fn advance_h(&self, ez: &[f64], hy: &[f64], dt: f64) -> Vec<f64> {
        let n = self.n;
        let ch = dt / (self.mu * self.dx);
        let mut out = hy.to_vec();
        match self.boundary {
            MaxwellBoundary1d::Periodic => {
                for (i, out_i) in out.iter_mut().enumerate() {
                    let ip = wrap_index(i as isize + 1, n);
                    *out_i = hy[i] + ch * (ez[ip] - ez[i]);
                }
            }
            MaxwellBoundary1d::Pec => {
                // Hy slot i is the half-node i+1/2, defined for i = 0..n-1.
                for i in 0..n - 1 {
                    out[i] = hy[i] + ch * (ez[i + 1] - ez[i]);
                }
            }
        }
        out
    }

    /// Advance `Ez` by one step from `Hy`, returning the new field.
    fn advance_e(&self, hy: &[f64], ez: &[f64], dt: f64) -> Vec<f64> {
        let n = self.n;
        let ce = dt / (self.epsilon * self.dx);
        let mut out = ez.to_vec();
        match self.boundary {
            MaxwellBoundary1d::Periodic => {
                for (i, out_i) in out.iter_mut().enumerate() {
                    let im = wrap_index(i as isize - 1, n);
                    *out_i = ez[i] + ce * (hy[i] - hy[im]);
                }
            }
            MaxwellBoundary1d::Pec => {
                out[0] = 0.0;
                out[n - 1] = 0.0;
                for i in 1..n - 1 {
                    out[i] = ez[i] + ce * (hy[i] - hy[i - 1]);
                }
            }
        }
        out
    }

    /// Build a state from explicit `Ez` (level `n=0`) and `Hy` (level `−1/2`).
    pub fn init(&self, ez0: &[f64], hy0: &[f64], dt: f64) -> PdeResult<MaxwellState1d> {
        if ez0.len() != self.n || hy0.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![ez0.len()],
            });
        }
        self.check_step(dt)?;
        let mut ez = ez0.to_vec();
        if matches!(self.boundary, MaxwellBoundary1d::Pec) {
            ez[0] = 0.0;
            ez[self.n - 1] = 0.0;
        }
        Ok(MaxwellState1d {
            ez,
            hy: hy0.to_vec(),
            t: 0.0,
            dt,
        })
    }

    /// Build a right-moving plane wave from an `Ez` profile `f(x)` (sampled at
    /// `x = i·Δx` from the origin). The companion `Hy = −f/η` is sampled at the
    /// staggered, retarded position so the pulse cleanly translates at `+c`.
    pub fn init_right_moving<F>(&self, f: F, dt: f64) -> PdeResult<MaxwellState1d>
    where
        F: Fn(f64) -> f64,
    {
        self.check_step(dt)?;
        let c = self.light_speed();
        let inv_eta = 1.0 / self.impedance();
        let ez: Vec<f64> = (0..self.n).map(|i| f(i as f64 * self.dx)).collect();
        let hy: Vec<f64> = (0..self.n)
            .map(|i| {
                let x_half = (i as f64 + 0.5) * self.dx + 0.5 * c * dt;
                -inv_eta * f(x_half)
            })
            .collect();
        self.init(&ez, &hy, dt)
    }

    /// Advance the state by one leapfrog step of size `dt`.
    pub fn step(&self, state: &mut MaxwellState1d, dt: f64) -> PdeResult<()> {
        if state.ez.len() != self.n || state.hy.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![state.ez.len()],
            });
        }
        self.check_step(dt)?;
        let hy_next = self.advance_h(&state.ez, &state.hy, dt);
        let ez_next = self.advance_e(&hy_next, &state.ez, dt);
        state.hy = hy_next;
        state.ez = ez_next;
        state.t += dt;
        state.dt = dt;
        Ok(())
    }

    /// Integrate `n_steps` leapfrog steps from the initial state.
    pub fn solve(
        &self,
        init: &MaxwellState1d,
        dt: f64,
        n_steps: usize,
    ) -> PdeResult<MaxwellState1d> {
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut state = init.clone();
        for _ in 0..n_steps {
            self.step(&mut state, dt)?;
        }
        if state
            .ez
            .iter()
            .chain(state.hy.iter())
            .any(|v| !v.is_finite())
        {
            return Err(PdeError::NumericalInstability(
                "FDTD solution diverged to non-finite values".into(),
            ));
        }
        Ok(state)
    }

    /// Staggered electromagnetic energy `½Δx Σ (ε Ez² + μ Hy^{n−1/2} Hy^{n+1/2})`.
    pub fn energy(&self, state: &MaxwellState1d) -> PdeResult<f64> {
        if state.ez.len() != self.n || state.hy.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![state.ez.len()],
            });
        }
        if !(state.dt.is_finite() && state.dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: "state step must be finite and > 0".into(),
            });
        }
        let hy_next = self.advance_h(&state.ez, &state.hy, state.dt);
        let electric: f64 = state.ez.iter().map(|&e| self.epsilon * e * e).sum();
        let magnetic: f64 = state
            .hy
            .iter()
            .zip(hy_next.iter())
            .map(|(&h0, &h1)| self.mu * h0 * h1)
            .sum();
        Ok(0.5 * self.dx * (electric + magnetic))
    }
}

/// Boundary condition for the 2-D TM FDTD solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxwellBoundary2d {
    /// Doubly-periodic torus.
    Periodic,
    /// Perfect electric conductor on every wall (`Ez = 0` on the boundary ring).
    Pec,
}

/// 2-D transverse-magnetic Yee-grid FDTD solver for `(Ez, Hx, Hy)` on a
/// row-major (`i·ny + j`) grid.
#[derive(Debug, Clone)]
pub struct Maxwell2dTm {
    /// Permittivity `ε > 0`.
    pub epsilon: f64,
    /// Permeability `μ > 0`.
    pub mu: f64,
    /// Grid spacing `(Δx, Δy)`.
    pub spacing: (f64, f64),
    /// Grid resolution `(nx, ny)`.
    pub grid: (usize, usize),
    /// Boundary condition.
    pub boundary: MaxwellBoundary2d,
}

/// Leapfrog state for the 2-D TM solver.
#[derive(Debug, Clone)]
pub struct MaxwellState2dTm {
    /// `Ez_{i,j}^n` (length `nx·ny`).
    pub ez: Vec<f64>,
    /// `Hx_{i,j+1/2}^{n−1/2}` (length `nx·ny`).
    pub hx: Vec<f64>,
    /// `Hy_{i+1/2,j}^{n−1/2}` (length `nx·ny`).
    pub hy: Vec<f64>,
    /// Current time.
    pub t: f64,
    /// Time step that produced this state.
    pub dt: f64,
}

impl Maxwell2dTm {
    /// Build a 2-D TM solver, validating `ε, μ > 0`, spacings `> 0`, `nx, ny ≥ 3`.
    pub fn new(
        epsilon: f64,
        mu: f64,
        spacing: (f64, f64),
        grid: (usize, usize),
        boundary: MaxwellBoundary2d,
    ) -> PdeResult<Self> {
        let (dx, dy) = spacing;
        let (nx, ny) = grid;
        if !(epsilon.is_finite() && epsilon > 0.0 && mu.is_finite() && mu > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "material".into(),
                reason: format!("epsilon, mu must be finite and > 0, got ({epsilon}, {mu})"),
            });
        }
        if !(dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "spacing".into(),
                reason: format!("must be finite and > 0, got ({dx}, {dy})"),
            });
        }
        if nx < 3 || ny < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "2-D TM FDTD requires nx>=3 ny>=3, got ({nx}, {ny})"
            )));
        }
        Ok(Self {
            epsilon,
            mu,
            spacing,
            grid,
            boundary,
        })
    }

    /// Speed of light in the medium `c = 1/√(εμ)`.
    #[must_use]
    pub fn light_speed(&self) -> f64 {
        1.0 / (self.epsilon * self.mu).sqrt()
    }

    /// Largest stable step from the 2-D Courant condition `cΔt√(1/Δx²+1/Δy²) ≤ 1`.
    #[must_use]
    pub fn courant_dt_max(&self) -> f64 {
        let (dx, dy) = self.spacing;
        let diag = (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt();
        1.0 / (self.light_speed() * diag)
    }

    fn check_step(&self, dt: f64) -> PdeResult<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be finite and > 0, got {dt}"),
            });
        }
        let dt_max = self.courant_dt_max();
        if dt > dt_max * (1.0 + COURANT_TOL) {
            return Err(PdeError::CflViolation { dt, dt_max });
        }
        Ok(())
    }

    fn n_cells(&self) -> usize {
        self.grid.0 * self.grid.1
    }

    /// Advance `(Hx, Hy)` by one half step from `Ez`, returning the new fields.
    fn advance_h(&self, ez: &[f64], hx: &[f64], hy: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        let (dx, dy) = self.spacing;
        let (nx, ny) = self.grid;
        let cx = dt / (self.mu * dx);
        let cy = dt / (self.mu * dy);
        let mut hx_out = hx.to_vec();
        let mut hy_out = hy.to_vec();
        match self.boundary {
            MaxwellBoundary2d::Periodic => {
                for i in 0..nx {
                    let ip = wrap_index(i as isize + 1, nx);
                    for j in 0..ny {
                        let jp = wrap_index(j as isize + 1, ny);
                        let idx = i * ny + j;
                        // μ ∂Hx/∂t = −∂Ez/∂y ;  μ ∂Hy/∂t = +∂Ez/∂x
                        hx_out[idx] = hx[idx] - cy * (ez[i * ny + jp] - ez[idx]);
                        hy_out[idx] = hy[idx] + cx * (ez[ip * ny + j] - ez[idx]);
                    }
                }
            }
            MaxwellBoundary2d::Pec => {
                // Hx slot (i,j) is half-node (i, j+1/2): valid for j = 0..ny-1.
                for i in 0..nx {
                    for j in 0..ny - 1 {
                        let idx = i * ny + j;
                        hx_out[idx] = hx[idx] - cy * (ez[idx + 1] - ez[idx]);
                    }
                }
                // Hy slot (i,j) is half-node (i+1/2, j): valid for i = 0..nx-1.
                for i in 0..nx - 1 {
                    for j in 0..ny {
                        let idx = i * ny + j;
                        hy_out[idx] = hy[idx] + cx * (ez[(i + 1) * ny + j] - ez[idx]);
                    }
                }
            }
        }
        (hx_out, hy_out)
    }

    /// Advance `Ez` by one step from `(Hx, Hy)`, returning the new field.
    fn advance_e(&self, hx: &[f64], hy: &[f64], ez: &[f64], dt: f64) -> Vec<f64> {
        let (dx, dy) = self.spacing;
        let (nx, ny) = self.grid;
        let cex = dt / (self.epsilon * dx);
        let cey = dt / (self.epsilon * dy);
        let mut out = ez.to_vec();
        match self.boundary {
            MaxwellBoundary2d::Periodic => {
                for i in 0..nx {
                    let im = wrap_index(i as isize - 1, nx);
                    for j in 0..ny {
                        let jm = wrap_index(j as isize - 1, ny);
                        let idx = i * ny + j;
                        // ε ∂Ez/∂t = ∂Hy/∂x − ∂Hx/∂y
                        out[idx] = ez[idx] + cex * (hy[idx] - hy[im * ny + j])
                            - cey * (hx[idx] - hx[i * ny + jm]);
                    }
                }
            }
            MaxwellBoundary2d::Pec => {
                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        let idx = i * ny + j;
                        out[idx] = ez[idx] + cex * (hy[idx] - hy[(i - 1) * ny + j])
                            - cey * (hx[idx] - hx[idx - 1]);
                    }
                }
                // Boundary ring stays at the PEC value Ez = 0.
                for i in 0..nx {
                    out[i * ny] = 0.0;
                    out[i * ny + (ny - 1)] = 0.0;
                }
                for j in 0..ny {
                    out[j] = 0.0;
                    out[(nx - 1) * ny + j] = 0.0;
                }
            }
        }
        out
    }

    /// Build a state from explicit fields.
    pub fn init(
        &self,
        ez0: &[f64],
        hx0: &[f64],
        hy0: &[f64],
        dt: f64,
    ) -> PdeResult<MaxwellState2dTm> {
        let n = self.n_cells();
        if ez0.len() != n || hx0.len() != n || hy0.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![ez0.len()],
            });
        }
        self.check_step(dt)?;
        let mut ez = ez0.to_vec();
        if matches!(self.boundary, MaxwellBoundary2d::Pec) {
            let (nx, ny) = self.grid;
            for i in 0..nx {
                ez[i * ny] = 0.0;
                ez[i * ny + (ny - 1)] = 0.0;
            }
            for j in 0..ny {
                ez[j] = 0.0;
                ez[(nx - 1) * ny + j] = 0.0;
            }
        }
        Ok(MaxwellState2dTm {
            ez,
            hx: hx0.to_vec(),
            hy: hy0.to_vec(),
            t: 0.0,
            dt,
        })
    }

    /// Advance the state by one leapfrog step of size `dt`.
    pub fn step(&self, state: &mut MaxwellState2dTm, dt: f64) -> PdeResult<()> {
        let n = self.n_cells();
        if state.ez.len() != n || state.hx.len() != n || state.hy.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![state.ez.len()],
            });
        }
        self.check_step(dt)?;
        let (hx_next, hy_next) = self.advance_h(&state.ez, &state.hx, &state.hy, dt);
        let ez_next = self.advance_e(&hx_next, &hy_next, &state.ez, dt);
        state.hx = hx_next;
        state.hy = hy_next;
        state.ez = ez_next;
        state.t += dt;
        state.dt = dt;
        Ok(())
    }

    /// Integrate `n_steps` leapfrog steps from the initial state.
    pub fn solve(
        &self,
        init: &MaxwellState2dTm,
        dt: f64,
        n_steps: usize,
    ) -> PdeResult<MaxwellState2dTm> {
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut state = init.clone();
        for _ in 0..n_steps {
            self.step(&mut state, dt)?;
        }
        let finite = state
            .ez
            .iter()
            .chain(state.hx.iter())
            .chain(state.hy.iter())
            .all(|v| v.is_finite());
        if !finite {
            return Err(PdeError::NumericalInstability(
                "2-D TM FDTD solution diverged to non-finite values".into(),
            ));
        }
        Ok(state)
    }

    /// Staggered electromagnetic energy
    /// `½ΔxΔy Σ (ε Ez² + μ Hx^{n−1/2}Hx^{n+1/2} + μ Hy^{n−1/2}Hy^{n+1/2})`.
    pub fn energy(&self, state: &MaxwellState2dTm) -> PdeResult<f64> {
        let n = self.n_cells();
        if state.ez.len() != n || state.hx.len() != n || state.hy.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![state.ez.len()],
            });
        }
        if !(state.dt.is_finite() && state.dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: "state step must be finite and > 0".into(),
            });
        }
        let (hx_next, hy_next) = self.advance_h(&state.ez, &state.hx, &state.hy, state.dt);
        let electric: f64 = state.ez.iter().map(|&e| self.epsilon * e * e).sum();
        let mag_x: f64 = state
            .hx
            .iter()
            .zip(hx_next.iter())
            .map(|(&a, &b)| self.mu * a * b)
            .sum();
        let mag_y: f64 = state
            .hy
            .iter()
            .zip(hy_next.iter())
            .map(|(&a, &b)| self.mu * a * b)
            .sum();
        let (dx, dy) = self.spacing;
        Ok(0.5 * dx * dy * (electric + mag_x + mag_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(im, m), (k, &x)| {
                if x > m { (k, x) } else { (im, m) }
            })
            .0
    }

    #[test]
    fn right_moving_pulse_propagates_at_speed_c() {
        // Magic step cΔt = Δx: a +x plane-wave pulse translates exactly one cell
        // per step, so the Ez peak moves by c·t (k cells after k steps).
        let n = 200;
        let dx = 1.0 / n as f64;
        let solver = Maxwell1d::new(1.0, 1.0, dx, n, MaxwellBoundary1d::Periodic).expect("solver");
        let dt = solver.courant_dt_max(); // Courant = 1
        let x0 = 0.25;
        let sigma = 0.03;
        let profile = |x: f64| {
            let d = x - x0;
            (-d * d / (2.0 * sigma * sigma)).exp()
        };
        let init = solver.init_right_moving(profile, dt).expect("init");
        let p0 = argmax(&init.ez);
        let steps = 30usize;
        let state = solver.solve(&init, dt, steps).expect("solve");
        let p1 = argmax(&state.ez);
        assert!(
            (p1 as isize - (p0 + steps) as isize).abs() <= 1,
            "peak {p0} -> {p1}, expected ~{}",
            p0 + steps
        );
        // A clean traveling wave keeps (most of) its amplitude.
        let amp = state.ez.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(amp > 0.8, "traveling amplitude {amp}");
    }

    #[test]
    fn gaussian_pulse_splits_into_two_halves() {
        // Ez = Gaussian, Hy = 0: like d'Alembert the pulse splits into two
        // half-amplitude copies travelling at ±c. Magic step ⇒ exact.
        let n = 240;
        let dx = 1.0 / n as f64;
        let solver = Maxwell1d::new(1.0, 1.0, dx, n, MaxwellBoundary1d::Periodic).expect("solver");
        let dt = solver.courant_dt_max();
        let center = n / 2;
        let sigma = 4.0; // in cells
        let ez0: Vec<f64> = (0..n)
            .map(|i| {
                let d = (i as f64 - center as f64) / sigma;
                (-0.5 * d * d).exp()
            })
            .collect();
        let hy0 = vec![0.0; n];
        let init = solver.init(&ez0, &hy0, dt).expect("init");
        let shift = 20usize;
        let state = solver.solve(&init, dt, shift).expect("solve");
        let peak0 = ez0[center];
        let right = state.ez[center + shift];
        let left = state.ez[center - shift];
        assert!((right - 0.5 * peak0).abs() < 0.03, "right half {right}");
        assert!((left - 0.5 * peak0).abs() < 0.03, "left half {left}");
        assert!(
            state.ez[center] < 0.15 * peak0,
            "centre vacated {}",
            state.ez[center]
        );
    }

    #[test]
    fn energy_is_conserved_periodic() {
        let n = 128;
        let dx = 1.0 / n as f64;
        let solver = Maxwell1d::new(1.0, 1.0, dx, n, MaxwellBoundary1d::Periodic).expect("solver");
        let dt = 0.5 * solver.courant_dt_max();
        let ez0: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::TAU * i as f64 / n as f64).sin())
            .collect();
        let hy0 = vec![0.0; n];
        let init = solver.init(&ez0, &hy0, dt).expect("init");
        let e0 = solver.energy(&init).expect("energy");
        let state = solver.solve(&init, dt, 400).expect("solve");
        let e1 = solver.energy(&state).expect("energy");
        let drift = (e1 - e0).abs() / e0.abs();
        assert!(drift < 1.0e-6, "energy drift {drift} (e0={e0}, e1={e1})");
    }

    #[test]
    fn pec_cavity_energy_conserved() {
        let n = 80;
        let dx = 1.0 / (n - 1) as f64;
        let solver = Maxwell1d::new(1.0, 1.0, dx, n, MaxwellBoundary1d::Pec).expect("solver");
        let dt = 0.5 * solver.courant_dt_max();
        // Cavity mode sin(πx) vanishing at the PEC walls.
        let ez0: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::PI * i as f64 / (n - 1) as f64).sin())
            .collect();
        let hy0 = vec![0.0; n];
        let init = solver.init(&ez0, &hy0, dt).expect("init");
        let e0 = solver.energy(&init).expect("energy");
        let state = solver.solve(&init, dt, 500).expect("solve");
        let e1 = solver.energy(&state).expect("energy");
        assert!((e1 - e0).abs() / e0.abs() < 1.0e-6, "PEC energy drift");
        // PEC walls stay clamped.
        assert!(state.ez[0].abs() < 1.0e-300);
        assert!(state.ez[n - 1].abs() < 1.0e-300);
    }

    #[test]
    fn courant_violation_is_rejected() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let solver = Maxwell1d::new(1.0, 1.0, dx, n, MaxwellBoundary1d::Periodic).expect("solver");
        let dt_bad = 1.5 * solver.courant_dt_max();
        let init = solver
            .init(&vec![0.0; n], &vec![0.0; n], 0.5 * solver.courant_dt_max())
            .expect("init");
        assert!(matches!(
            solver.solve(&init, dt_bad, 3),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn solution_stays_finite() {
        let n = 64;
        let dx = 1.0 / n as f64;
        let solver = Maxwell1d::new(2.0, 1.5, dx, n, MaxwellBoundary1d::Periodic).expect("solver");
        let dt = 0.9 * solver.courant_dt_max();
        let ez0: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::TAU * i as f64 / n as f64).cos())
            .collect();
        let init = solver.init(&ez0, &vec![0.0; n], dt).expect("init");
        let state = solver.solve(&init, dt, 300).expect("solve");
        assert!(
            state
                .ez
                .iter()
                .chain(state.hy.iter())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn light_speed_and_impedance_track_material() {
        let solver = Maxwell1d::new(4.0, 1.0, 0.1, 8, MaxwellBoundary1d::Periodic).expect("solver");
        assert!((solver.light_speed() - 0.5).abs() < 1.0e-12); // 1/sqrt(4)
        assert!((solver.impedance() - 0.5).abs() < 1.0e-12); // sqrt(1/4)
    }

    #[test]
    fn invalid_construction_rejected() {
        assert!(Maxwell1d::new(0.0, 1.0, 0.1, 8, MaxwellBoundary1d::Periodic).is_err());
        assert!(Maxwell1d::new(1.0, -1.0, 0.1, 8, MaxwellBoundary1d::Periodic).is_err());
        assert!(Maxwell1d::new(1.0, 1.0, 0.0, 8, MaxwellBoundary1d::Periodic).is_err());
        assert!(Maxwell1d::new(1.0, 1.0, 0.1, 2, MaxwellBoundary1d::Periodic).is_err());
    }

    // ── 2-D TM ──────────────────────────────────────────────────────────────

    #[test]
    fn tm_energy_conserved_periodic() {
        let nx = 32;
        let ny = 32;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let solver = Maxwell2dTm::new(1.0, 1.0, (dx, dy), (nx, ny), MaxwellBoundary2d::Periodic)
            .expect("solver");
        let dt = 0.5 * solver.courant_dt_max();
        let mut ez0 = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                ez0[i * ny + j] =
                    (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            }
        }
        let init = solver
            .init(&ez0, &vec![0.0; nx * ny], &vec![0.0; nx * ny], dt)
            .expect("init");
        let e0 = solver.energy(&init).expect("energy");
        let state = solver.solve(&init, dt, 200).expect("solve");
        let e1 = solver.energy(&state).expect("energy");
        assert!(
            (e1 - e0).abs() / e0.abs() < 1.0e-5,
            "TM energy drift {e0}->{e1}"
        );
    }

    #[test]
    fn tm_finite_propagation_speed() {
        // A point Ez disturbance cannot reach beyond k cells after k steps
        // (Courant < 1): the far corner stays exactly zero — causality.
        let nx = 41;
        let ny = 41;
        let solver = Maxwell2dTm::new(
            1.0,
            1.0,
            (1.0 / nx as f64, 1.0 / ny as f64),
            (nx, ny),
            MaxwellBoundary2d::Periodic,
        )
        .expect("solver");
        let dt = 0.5 * solver.courant_dt_max();
        let mut ez0 = vec![0.0; nx * ny];
        let c = nx / 2;
        ez0[c * ny + c] = 1.0;
        let init = solver
            .init(&ez0, &vec![0.0; nx * ny], &vec![0.0; nx * ny], dt)
            .expect("init");
        let steps = 5usize;
        let state = solver.solve(&init, dt, steps).expect("solve");
        // A node 10 cells away in x is far outside the 5-step numerical cone.
        let far = state.ez[(c + 10) * ny + c];
        assert!(far.abs() < 1.0e-300, "signal leaked to far node: {far}");
        // Reflection symmetry across the source column.
        let a = state.ez[(c + 3) * ny + c];
        let b = state.ez[(c - 3) * ny + c];
        assert!((a - b).abs() < 1.0e-12, "asymmetric: {a} vs {b}");
    }

    #[test]
    fn tm_courant_violation_rejected() {
        let nx = 16;
        let ny = 16;
        let solver = Maxwell2dTm::new(
            1.0,
            1.0,
            (1.0 / nx as f64, 1.0 / ny as f64),
            (nx, ny),
            MaxwellBoundary2d::Periodic,
        )
        .expect("solver");
        let dt_bad = 1.2 * solver.courant_dt_max();
        let init = solver
            .init(
                &vec![0.0; nx * ny],
                &vec![0.0; nx * ny],
                &vec![0.0; nx * ny],
                0.5 * solver.courant_dt_max(),
            )
            .expect("init");
        assert!(matches!(
            solver.solve(&init, dt_bad, 2),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn tm_pec_cavity_finite_and_conserved() {
        let nx = 24;
        let ny = 24;
        let solver = Maxwell2dTm::new(
            1.0,
            1.0,
            (1.0 / (nx - 1) as f64, 1.0 / (ny - 1) as f64),
            (nx, ny),
            MaxwellBoundary2d::Pec,
        )
        .expect("solver");
        let dt = 0.5 * solver.courant_dt_max();
        let mut ez0 = vec![0.0; nx * ny];
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let sx = (std::f64::consts::PI * i as f64 / (nx - 1) as f64).sin();
                let sy = (std::f64::consts::PI * j as f64 / (ny - 1) as f64).sin();
                ez0[i * ny + j] = sx * sy;
            }
        }
        let init = solver
            .init(&ez0, &vec![0.0; nx * ny], &vec![0.0; nx * ny], dt)
            .expect("init");
        let e0 = solver.energy(&init).expect("energy");
        let state = solver.solve(&init, dt, 200).expect("solve");
        let e1 = solver.energy(&state).expect("energy");
        assert!((e1 - e0).abs() / e0.abs() < 1.0e-5, "PEC TM energy drift");
        assert!(state.ez.iter().all(|v| v.is_finite()));
    }
}
