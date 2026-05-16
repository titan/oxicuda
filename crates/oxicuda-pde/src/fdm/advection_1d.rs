//! 1D linear advection: `u_t + c u_x = 0`.
//!
//! Schemes:
//! * First-order upwind (`c>0`):  `u^{n+1}_i = u^n_i - (c dt/h)(u^n_i - u^n_{i-1})`
//! * Lax-Wendroff:                  `u^{n+1}_i = u^n_i - (lambda/2)(u^n_{i+1} - u^n_{i-1})
//!                                                     + (lambda^2/2)(u^n_{i+1} - 2 u^n_i + u^n_{i-1})`
//!   where `lambda = c dt/h`.
//!
//! CFL stability: `|c| dt / h <= 1` for both schemes.

use crate::error::{PdeError, PdeResult};
use crate::mesh::Mesh1d;

/// First-order upwind step. Periodic BC: `u[n-1] = u[0]`-style assumed via `u_left`.
/// For an open Dirichlet inflow, set `u_left` to the upstream boundary value.
pub fn upwind_step_1d(
    mesh: &Mesh1d,
    u: &mut [f64],
    c: f64,
    dt: f64,
    u_left_bc: f64,
) -> PdeResult<()> {
    let n = mesh.n;
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let h = mesh.h();
    let lambda = c * dt / h;
    if lambda.abs() > 1.0 + 1.0e-12 {
        return Err(PdeError::CflViolation {
            dt,
            dt_max: h / c.abs().max(1.0e-300),
        });
    }
    let mut next = vec![0.0; n];
    if c >= 0.0 {
        next[0] = u_left_bc;
        for i in 1..n {
            next[i] = u[i] - lambda * (u[i] - u[i - 1]);
        }
    } else {
        next[n - 1] = u_left_bc;
        for i in 0..n - 1 {
            next[i] = u[i] - lambda * (u[i + 1] - u[i]);
        }
    }
    u.copy_from_slice(&next);
    Ok(())
}

/// Lax-Wendroff step. Periodic BCs (wraps `i-1` and `i+1`).
pub fn lax_wendroff_step_1d(mesh: &Mesh1d, u: &mut [f64], c: f64, dt: f64) -> PdeResult<()> {
    let n = mesh.n;
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let h = mesh.h();
    let lambda = c * dt / h;
    if lambda.abs() > 1.0 + 1.0e-12 {
        return Err(PdeError::CflViolation {
            dt,
            dt_max: h / c.abs().max(1.0e-300),
        });
    }
    let lam2 = lambda * lambda;
    let mut next = vec![0.0; n];
    for i in 0..n {
        let im1 = if i == 0 { n - 1 } else { i - 1 };
        let ip1 = if i + 1 == n { 0 } else { i + 1 };
        next[i] =
            u[i] - 0.5 * lambda * (u[ip1] - u[im1]) + 0.5 * lam2 * (u[ip1] - 2.0 * u[i] + u[im1]);
    }
    u.copy_from_slice(&next);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advection_upwind_transports_pulse() {
        // Initial Gaussian centered at x=0.3, c=1, after t=0.4 it should be at x=0.7.
        let mesh = Mesh1d::uniform(0.0, 1.0, 201).expect("ok");
        let c = 1.0;
        let dt = 0.5 * mesh.h() / c;
        let mut u: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|x| (-((x - 0.3) * (x - 0.3)) / 0.01).exp())
            .collect();
        let t_final = 0.4;
        let nsteps = (t_final / dt).round() as usize;
        for _ in 0..nsteps {
            upwind_step_1d(&mesh, &mut u, c, dt, 0.0).expect("ok");
        }
        // Find peak position
        let (idx_max, _) =
            u.iter()
                .enumerate()
                .fold((0_usize, f64::NEG_INFINITY), |(i, m), (k, &v)| {
                    if v > m { (k, v) } else { (i, m) }
                });
        let x_peak = mesh.nodes[idx_max];
        assert!((x_peak - 0.7).abs() < 0.05, "peak at x={x_peak}");
    }

    #[test]
    fn advection_lax_wendroff_periodic_conserves_mass() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 101).expect("ok");
        let c = 1.0;
        let dt = 0.5 * mesh.h() / c;
        let mut u: Vec<f64> = mesh
            .nodes
            .iter()
            .map(|x| (-((x - 0.5).powi(2)) / 0.005).exp())
            .collect();
        let initial_mass: f64 = u.iter().sum::<f64>() * mesh.h();
        for _ in 0..200 {
            lax_wendroff_step_1d(&mesh, &mut u, c, dt).expect("ok");
        }
        let final_mass: f64 = u.iter().sum::<f64>() * mesh.h();
        assert!((final_mass - initial_mass).abs() / initial_mass.abs() < 1.0e-6);
    }

    #[test]
    fn advection_upwind_cfl_violation() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let mut u = vec![1.0; mesh.n];
        let res = upwind_step_1d(&mesh, &mut u, 10.0, 1.0, 0.0);
        assert!(res.is_err());
    }
}
