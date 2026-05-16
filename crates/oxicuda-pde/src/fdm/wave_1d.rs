//! 1D wave equation: `u_tt = c^2 * u_xx` with Dirichlet BCs (fixed ends).
//!
//! Uses the explicit leapfrog scheme:
//! `u^{n+1}_i = 2 u^n_i - u^{n-1}_i + (c dt/h)^2 * (u^n_{i+1} - 2 u^n_i + u^n_{i-1})`
//!
//! CFL: `c dt / h <= 1`.

use crate::error::{PdeError, PdeResult};
use crate::mesh::Mesh1d;

/// Internal state for a leapfrog wave simulation.
#[derive(Debug, Clone)]
pub struct WaveState1d {
    pub u_prev: Vec<f64>,
    pub u_curr: Vec<f64>,
}

impl WaveState1d {
    /// Construct from initial position `u0(x)` and initial velocity `v0(x)` using
    /// a second-order Taylor step:
    /// `u^1 = u^0 + dt v0 + 0.5 (c dt)^2 u0''`.
    pub fn from_initial(
        mesh: &Mesh1d,
        u0: &[f64],
        v0: &[f64],
        c: f64,
        dt: f64,
        ua: f64,
        ub: f64,
    ) -> PdeResult<Self> {
        let n = mesh.n;
        if u0.len() != n || v0.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![u0.len()],
            });
        }
        let h = mesh.h();
        if c * dt / h > 1.0 + 1.0e-12 {
            return Err(PdeError::CflViolation { dt, dt_max: h / c });
        }
        let r = c * dt / h;
        let r2 = r * r;
        let mut u_curr = vec![0.0; n];
        u_curr[0] = ua;
        u_curr[n - 1] = ub;
        for i in 1..n - 1 {
            let uxx = u0[i - 1] - 2.0 * u0[i] + u0[i + 1];
            u_curr[i] = u0[i] + dt * v0[i] + 0.5 * r2 * uxx;
        }
        Ok(Self {
            u_prev: u0.to_vec(),
            u_curr,
        })
    }
}

/// Leapfrog step in-place: updates `state` to the next time level.
pub fn leapfrog_step_1d(
    mesh: &Mesh1d,
    state: &mut WaveState1d,
    c: f64,
    dt: f64,
    ua: f64,
    ub: f64,
) -> PdeResult<()> {
    let n = mesh.n;
    if state.u_curr.len() != n || state.u_prev.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![state.u_curr.len()],
        });
    }
    let h = mesh.h();
    let r = c * dt / h;
    if r > 1.0 + 1.0e-12 {
        return Err(PdeError::CflViolation { dt, dt_max: h / c });
    }
    let r2 = r * r;
    let mut u_next = vec![0.0; n];
    u_next[0] = ua;
    u_next[n - 1] = ub;
    for (i, u_next_i) in u_next.iter_mut().enumerate().take(n - 1).skip(1) {
        let uxx = state.u_curr[i - 1] - 2.0 * state.u_curr[i] + state.u_curr[i + 1];
        *u_next_i = 2.0 * state.u_curr[i] - state.u_prev[i] + r2 * uxx;
    }
    state.u_prev = std::mem::take(&mut state.u_curr);
    state.u_curr = u_next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_1d_standing_wave_period() {
        // Standing wave u(x,t) = sin(pi x) cos(c pi t) on [0,1], c=1.
        // Should return to initial after t = 2 (one period).
        let pi = std::f64::consts::PI;
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let c = 1.0;
        let dt = 0.5 * mesh.h() / c;
        let u0: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
        let v0: Vec<f64> = vec![0.0; mesh.n];
        let mut state = WaveState1d::from_initial(&mesh, &u0, &v0, c, dt, 0.0, 0.0).expect("ok");
        let t_final = 0.5; // quarter period -> u should be approx zero
        let nsteps = (t_final / dt).round() as usize;
        for _ in 0..nsteps {
            leapfrog_step_1d(&mesh, &mut state, c, dt, 0.0, 0.0).expect("ok");
        }
        // At t=0.5 (T/2), cos(c*pi*t)=0 so u should be near 0.
        let max_u = state.u_curr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_u < 0.05, "max_u={max_u}");
    }

    #[test]
    fn wave_1d_cfl_violation() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let u0 = vec![0.0; mesh.n];
        let v0 = vec![0.0; mesh.n];
        let res = WaveState1d::from_initial(&mesh, &u0, &v0, 1.0, 1.0, 0.0, 0.0);
        assert!(res.is_err());
    }
}
