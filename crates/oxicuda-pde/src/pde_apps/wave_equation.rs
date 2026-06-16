//! Second-order wave equation `∂²u/∂t² = c² ∇²u` via an explicit leapfrog scheme.
//!
//! # Discretisation (1-D)
//!
//! The classic three-level central-difference (leapfrog) stencil
//!
//! ```text
//! u^{n+1}_i = 2 u^n_i − u^{n−1}_i + r² (u^n_{i+1} − 2 u^n_i + u^n_{i−1}),   r = c·dt/dx
//! ```
//!
//! is bootstrapped from the initial displacement `u0` and velocity `u̇0` with the
//! second-order Taylor step `u^1 = u^0 + dt·u̇0 + ½ r² Δ_h u^0`.
//!
//! # Stability (CFL)
//!
//! The scheme is stable iff the Courant number `r = c·dt/dx ≤ 1`. A larger step is
//! rejected with [`PdeError::CflViolation`]. At exactly `r = 1` the 1-D scheme is
//! *nodally exact* (no numerical dispersion) and reproduces d'Alembert's solution
//! `u(x,t) = ½[u0(x − c t) + u0(x + c t)]` to round-off.
//!
//! # Boundary conditions
//!
//! * [`WaveBoundary::Dirichlet`] — fixed endpoint displacements;
//! * [`WaveBoundary::Periodic`] — torus topology, reusing
//!   [`crate::bc::periodic::periodic_laplacian_1d`].
//!
//! # Energy
//!
//! Leapfrog conserves the staggered discrete energy
//!
//! ```text
//! E = ½ dx Σ ((u^n − u^{n−1})/dt)² + ½ (c²/dx) Σ (Δ⁺u^n)(Δ⁺u^{n−1})
//! ```
//!
//! (the discrete Hamiltonian) to round-off; it is exposed via [`WaveEquation::energy`].
//!
//! Reference: LeVeque, *Finite Difference Methods for Ordinary and Partial
//! Differential Equations*, SIAM 2007, Chapter 10.

use crate::bc::periodic::periodic_laplacian_1d;
use crate::error::{PdeError, PdeResult};

/// Tolerance on the Courant number when checking the CFL condition.
const CFL_TOL: f64 = 1.0e-12;

/// Boundary condition for the 1-D wave solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveBoundary {
    /// Fixed endpoint displacements `u[0] = left`, `u[n−1] = right` for all time.
    Dirichlet {
        /// Displacement clamped at the left endpoint.
        left: f64,
        /// Displacement clamped at the right endpoint.
        right: f64,
    },
    /// Periodic domain: node `−1` wraps to `n−1` and node `n` wraps to `0`.
    Periodic,
}

/// Explicit second-order leapfrog solver for the 1-D wave equation.
#[derive(Debug, Clone)]
pub struct WaveEquation {
    /// Wave speed `c ≥ 0`.
    pub c: f64,
    /// Uniform grid spacing `dx > 0`.
    pub dx: f64,
    /// Number of grid nodes (`n ≥ 3`).
    pub n: usize,
    /// Boundary condition.
    pub boundary: WaveBoundary,
}

/// Two-level leapfrog state carrying both time levels and the step used.
#[derive(Debug, Clone)]
pub struct WaveState {
    /// Solution at the previous time level `u^{n−1}`.
    pub u_prev: Vec<f64>,
    /// Solution at the current time level `u^n`.
    pub u_curr: Vec<f64>,
    /// Current simulation time `t = n · dt`.
    pub t: f64,
    /// Time step used to advance the state (constant; required for energy diagnostics).
    pub dt: f64,
}

impl WaveEquation {
    /// Build a solver. Validates `c ≥ 0`, `dx > 0`, `n ≥ 3`, and finite Dirichlet data.
    pub fn new(c: f64, dx: f64, n: usize, boundary: WaveBoundary) -> PdeResult<Self> {
        if !(c.is_finite() && c >= 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "c".into(),
                reason: format!("wave speed must be finite and >= 0, got {c}"),
            });
        }
        if !(dx.is_finite() && dx > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dx".into(),
                reason: format!("grid spacing must be finite and > 0, got {dx}"),
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "wave equation requires n >= 3, got {n}"
            )));
        }
        if let WaveBoundary::Dirichlet { left, right } = boundary {
            if !(left.is_finite() && right.is_finite()) {
                return Err(PdeError::InvalidParameter {
                    name: "boundary".into(),
                    reason: "Dirichlet displacements must be finite".into(),
                });
            }
        }
        Ok(Self { c, dx, n, boundary })
    }

    /// Largest stable time step `dt_max = dx / c` (`+∞` when `c = 0`).
    #[must_use]
    pub fn cfl_dt_max(&self) -> f64 {
        if self.c > 0.0 {
            self.dx / self.c
        } else {
            f64::INFINITY
        }
    }

    /// Courant number `r = c·dt/dx` for a candidate step.
    fn courant(&self, dt: f64) -> f64 {
        self.c * dt / self.dx
    }

    /// Reject a step that violates the CFL condition `r ≤ 1`.
    fn check_step(&self, dt: f64) -> PdeResult<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("time step must be finite and > 0, got {dt}"),
            });
        }
        if self.courant(dt) > 1.0 + CFL_TOL {
            return Err(PdeError::CflViolation {
                dt,
                dt_max: self.cfl_dt_max(),
            });
        }
        Ok(())
    }

    /// Discrete Laplacian `∇²u` consistent with the active boundary condition.
    fn laplacian(&self, u: &[f64]) -> PdeResult<Vec<f64>> {
        match self.boundary {
            WaveBoundary::Periodic => periodic_laplacian_1d(u, self.dx),
            WaveBoundary::Dirichlet { .. } => {
                let n = self.n;
                let inv_h2 = 1.0 / (self.dx * self.dx);
                let mut lap = vec![0.0; n];
                for i in 1..n - 1 {
                    lap[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_h2;
                }
                Ok(lap)
            }
        }
    }

    /// Overwrite the boundary nodes of `u` with the prescribed Dirichlet data.
    fn apply_boundary(&self, u: &mut [f64]) {
        if let WaveBoundary::Dirichlet { left, right } = self.boundary {
            let n = self.n;
            u[0] = left;
            u[n - 1] = right;
        }
    }

    /// Validate an initial-data slice and ensure it is finite.
    fn check_field(&self, u: &[f64], name: &str) -> PdeResult<()> {
        if u.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![u.len()],
            });
        }
        if u.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(format!(
                "{name} contains non-finite values"
            )));
        }
        Ok(())
    }

    /// Bootstrap the two-level state from initial displacement `u0` and velocity `u0_dot`
    /// using the second-order Taylor step for `u^1`.
    pub fn init(&self, u0: &[f64], u0_dot: &[f64], dt: f64) -> PdeResult<WaveState> {
        self.check_field(u0, "u0")?;
        self.check_field(u0_dot, "u0_dot")?;
        self.check_step(dt)?;

        let n = self.n;
        let mut u_prev = u0.to_vec();
        self.apply_boundary(&mut u_prev);

        let lap = self.laplacian(&u_prev)?;
        let half_cdt2 = 0.5 * (self.c * dt) * (self.c * dt);
        let mut u_curr = vec![0.0; n];
        match self.boundary {
            WaveBoundary::Dirichlet { left, right } => {
                u_curr[0] = left;
                u_curr[n - 1] = right;
                for i in 1..n - 1 {
                    u_curr[i] = u_prev[i] + dt * u0_dot[i] + half_cdt2 * lap[i];
                }
            }
            WaveBoundary::Periodic => {
                for i in 0..n {
                    u_curr[i] = u_prev[i] + dt * u0_dot[i] + half_cdt2 * lap[i];
                }
            }
        }

        Ok(WaveState {
            u_prev,
            u_curr,
            t: dt,
            dt,
        })
    }

    /// Advance a state by one leapfrog step of size `dt` (rejecting CFL violations).
    pub fn step(&self, state: &mut WaveState, dt: f64) -> PdeResult<()> {
        if state.u_curr.len() != self.n || state.u_prev.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![state.u_curr.len()],
            });
        }
        self.check_step(dt)?;

        let n = self.n;
        let lap = self.laplacian(&state.u_curr)?;
        let cdt2 = (self.c * dt) * (self.c * dt);
        let mut u_next = vec![0.0; n];
        match self.boundary {
            WaveBoundary::Dirichlet { left, right } => {
                u_next[0] = left;
                u_next[n - 1] = right;
                for i in 1..n - 1 {
                    u_next[i] = 2.0 * state.u_curr[i] - state.u_prev[i] + cdt2 * lap[i];
                }
            }
            WaveBoundary::Periodic => {
                for i in 0..n {
                    u_next[i] = 2.0 * state.u_curr[i] - state.u_prev[i] + cdt2 * lap[i];
                }
            }
        }

        state.u_prev = std::mem::replace(&mut state.u_curr, u_next);
        state.t += dt;
        state.dt = dt;
        Ok(())
    }

    /// Integrate `n_steps` leapfrog steps from the initial data and return the final state.
    ///
    /// The bootstrap counts as the first step, so the returned state is `u^{n_steps}`
    /// at time `n_steps · dt`.
    pub fn solve(
        &self,
        u0: &[f64],
        u0_dot: &[f64],
        dt: f64,
        n_steps: usize,
    ) -> PdeResult<WaveState> {
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut state = self.init(u0, u0_dot, dt)?;
        for _ in 1..n_steps {
            self.step(&mut state, dt)?;
        }
        if state.u_curr.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "wave solution diverged to non-finite values".into(),
            ));
        }
        Ok(state)
    }

    /// Staggered discrete energy `E = ½ dx Σ u̇² + ½ (c²/dx) Σ (Δ⁺u^n)(Δ⁺u^{n−1})`.
    ///
    /// This is the discrete Hamiltonian conserved by the leapfrog scheme.
    pub fn energy(&self, state: &WaveState) -> PdeResult<f64> {
        if state.u_curr.len() != self.n || state.u_prev.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![state.u_curr.len()],
            });
        }
        if !(state.dt.is_finite() && state.dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: "state time step must be finite and > 0".into(),
            });
        }
        let n = self.n;
        let dx = self.dx;
        let inv_dt = 1.0 / state.dt;
        let mut kinetic = 0.0;
        for i in 0..n {
            let v = (state.u_curr[i] - state.u_prev[i]) * inv_dt;
            kinetic += v * v;
        }
        kinetic *= 0.5 * dx;

        let c2_over_dx = self.c * self.c / dx;
        let mut potential = 0.0;
        let pairs = match self.boundary {
            // Periodic: n forward pairs with wrap-around.
            WaveBoundary::Periodic => n,
            // Dirichlet: n−1 interior forward pairs.
            WaveBoundary::Dirichlet { .. } => n - 1,
        };
        for i in 0..pairs {
            let ip = if i + 1 == n { 0 } else { i + 1 };
            let dn = state.u_curr[ip] - state.u_curr[i];
            let dp = state.u_prev[ip] - state.u_prev[i];
            potential += dn * dp;
        }
        potential *= 0.5 * c2_over_dx;

        Ok(kinetic + potential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn standing_wave_dirichlet_quarter_period_is_zero() {
        // u(x,t) = sin(π x) cos(c π t) on [0,1] with fixed ends.
        // At t = T/4 = 0.5 (c=1) the displacement passes through zero.
        let n = 41;
        let dx = 1.0 / (n - 1) as f64;
        let c = 1.0;
        let eq = WaveEquation::new(
            c,
            dx,
            n,
            WaveBoundary::Dirichlet {
                left: 0.0,
                right: 0.0,
            },
        )
        .expect("solver");
        let dt = 0.5 * dx / c; // r = 0.5
        let u0: Vec<f64> = (0..n).map(|i| (PI * i as f64 * dx).sin()).collect();
        let v0 = vec![0.0; n];
        let nsteps = (0.5 / dt).round() as usize;
        let state = eq.solve(&u0, &v0, dt, nsteps).expect("solve");
        let max_u = state.u_curr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            max_u < 0.05,
            "max|u|={max_u} should vanish at quarter period"
        );
    }

    #[test]
    fn standing_wave_period_matches_dispersion() {
        // After one full period T = 2π/(c k) the solution returns to its initial shape.
        let n = 65;
        let dx = 1.0 / (n - 1) as f64;
        let c = 1.0;
        let k = PI; // fundamental Dirichlet mode on [0,1]
        let eq = WaveEquation::new(
            c,
            dx,
            n,
            WaveBoundary::Dirichlet {
                left: 0.0,
                right: 0.0,
            },
        )
        .expect("solver");
        let dt = 0.25 * dx / c;
        let period = std::f64::consts::TAU / (c * k);
        let nsteps = (period / dt).round() as usize;
        let u0: Vec<f64> = (0..n).map(|i| (k * i as f64 * dx).sin()).collect();
        let v0 = vec![0.0; n];
        let state = eq.solve(&u0, &v0, dt, nsteps).expect("solve");
        let mid = n / 2;
        let rel = (state.u_curr[mid] - u0[mid]).abs() / u0[mid].abs();
        assert!(
            rel < 0.05,
            "returned displacement rel err {rel} after one period"
        );
    }

    #[test]
    fn dalembert_pulse_splits_into_two_halves() {
        // Courant number 1 ⇒ nodally exact: a localised bump splits into two
        // half-amplitude copies translating at ±c.
        let n = 80;
        let dx = 1.0 / n as f64; // periodic: n independent nodes on [0,1)
        let c = 1.0;
        let eq = WaveEquation::new(c, dx, n, WaveBoundary::Periodic).expect("solver");
        let dt = dx / c; // r = 1
        let sigma = 0.04;
        let xc = 0.5;
        let u0: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx - xc;
                (-(x * x) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let v0 = vec![0.0; n];
        let shift = 8usize; // 8 cells each way
        let state = eq.solve(&u0, &v0, dt, shift).expect("solve");
        let center = (xc / dx).round() as usize;
        let peak0 = u0[center];
        // Right- and left-moving halves at center ± shift carry ≈ ½ the peak.
        let right = state.u_curr[center + shift];
        let left = state.u_curr[center - shift];
        assert!((right - 0.5 * peak0).abs() < 0.02, "right half {right}");
        assert!((left - 0.5 * peak0).abs() < 0.02, "left half {left}");
        // The original centre has been vacated.
        assert!(
            state.u_curr[center] < 0.1 * peak0,
            "centre {}",
            state.u_curr[center]
        );
    }

    #[test]
    fn energy_is_conserved_for_periodic_leapfrog() {
        // The staggered discrete energy is the leapfrog invariant: drift over a
        // couple of periods is at round-off level.
        let n = 64;
        let dx = 1.0 / n as f64;
        let c = 1.0;
        let eq = WaveEquation::new(c, dx, n, WaveBoundary::Periodic).expect("solver");
        let dt = 0.5 * dx / c;
        let u0: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::TAU * i as f64 * dx).sin())
            .collect();
        let v0 = vec![0.0; n];
        let state1 = eq.init(&u0, &v0, dt).expect("init");
        let e0 = eq.energy(&state1).expect("energy");
        // Two periods (T = 1/c) of integration.
        let nsteps = (2.0 / dt).round() as usize;
        let state = eq.solve(&u0, &v0, dt, nsteps).expect("solve");
        let e1 = eq.energy(&state).expect("energy");
        let drift = (e1 - e0).abs() / e0.abs();
        assert!(drift < 1.0e-6, "energy drift {drift} (e0={e0} e1={e1})");
    }

    #[test]
    fn cfl_violation_is_rejected() {
        let n = 16;
        let dx = 1.0 / (n - 1) as f64;
        let eq = WaveEquation::new(1.0, dx, n, WaveBoundary::Periodic).expect("solver");
        let u0 = vec![0.0; n];
        let v0 = vec![0.0; n];
        let dt = 2.0 * dx; // r = 2 > 1
        assert!(matches!(
            eq.solve(&u0, &v0, dt, 5),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn solution_stays_finite() {
        let n = 48;
        let dx = 1.0 / n as f64;
        let eq = WaveEquation::new(1.0, dx, n, WaveBoundary::Periodic).expect("solver");
        let u0: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::TAU * i as f64 * dx).cos())
            .collect();
        let v0 = vec![0.0; n];
        let state = eq.solve(&u0, &v0, 0.5 * dx, 200).expect("solve");
        assert!(state.u_curr.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        let n = 16;
        let dx = 1.0 / n as f64;
        let eq = WaveEquation::new(1.0, dx, n, WaveBoundary::Periodic).expect("solver");
        let u0 = vec![0.0; n - 1];
        let v0 = vec![0.0; n];
        assert!(matches!(
            eq.init(&u0, &v0, 0.5 * dx),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn invalid_construction_is_rejected() {
        let dx = 0.1;
        assert!(WaveEquation::new(-1.0, dx, 16, WaveBoundary::Periodic).is_err());
        assert!(WaveEquation::new(1.0, 0.0, 16, WaveBoundary::Periodic).is_err());
        assert!(WaveEquation::new(1.0, dx, 2, WaveBoundary::Periodic).is_err());
        assert!(matches!(
            WaveEquation::new(1.0, dx, 16, WaveBoundary::Periodic)
                .expect("solver")
                .solve(&[0.0; 16], &[0.0; 16], 0.05, 0),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn boundary_kind_equality() {
        assert_eq!(
            WaveBoundary::Dirichlet {
                left: 1.0,
                right: 2.0
            },
            WaveBoundary::Dirichlet {
                left: 1.0,
                right: 2.0
            }
        );
        assert_ne!(
            WaveBoundary::Periodic,
            WaveBoundary::Dirichlet {
                left: 0.0,
                right: 0.0
            }
        );
    }
}
