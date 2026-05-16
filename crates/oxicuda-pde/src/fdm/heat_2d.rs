//! 2D heat equation: `u_t = alpha * (u_xx + u_yy)` with Dirichlet BCs.
//!
//! Forward Euler (explicit, conditionally stable).

use crate::error::{PdeError, PdeResult};
use crate::mesh::Mesh2d;

/// Forward Euler step for 2D heat equation on a uniform grid.
///
/// Stability: `alpha*dt*(1/hx^2 + 1/hy^2) <= 0.5`.
pub fn forward_euler_step_2d(
    mesh: &Mesh2d,
    u: &mut [f64],
    alpha: f64,
    dt: f64,
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
) -> PdeResult<()> {
    let n_nodes = mesh.n_nodes();
    if u.len() != n_nodes {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_nodes],
            got: vec![u.len()],
        });
    }
    let hx = mesh.hx();
    let hy = mesh.hy();
    let rx = alpha * dt / (hx * hx);
    let ry = alpha * dt / (hy * hy);
    let max_dt = 0.5 / (alpha * (1.0 / (hx * hx) + 1.0 / (hy * hy)));
    if dt > max_dt + 1.0e-12 {
        return Err(PdeError::CflViolation { dt, dt_max: max_dt });
    }
    let mut next = u.to_vec();
    // boundaries
    for j in 0..mesh.ny {
        next[j] = left;
        next[(mesh.nx - 1) * mesh.ny + j] = right;
    }
    for i in 0..mesh.nx {
        next[i * mesh.ny] = bottom;
        next[i * mesh.ny + mesh.ny - 1] = top;
    }
    for i in 1..mesh.nx - 1 {
        for j in 1..mesh.ny - 1 {
            let idx = i * mesh.ny + j;
            let lap_x = u[idx + mesh.ny] - 2.0 * u[idx] + u[idx - mesh.ny];
            let lap_y = u[idx + 1] - 2.0 * u[idx] + u[idx - 1];
            next[idx] = u[idx] + rx * lap_x + ry * lap_y;
        }
    }
    u.copy_from_slice(&next);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_2d_forward_euler_smooths() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let pi = std::f64::consts::PI;
        let mut u = vec![0.0; mesh.n_nodes()];
        for i in 0..mesh.nx {
            for j in 0..mesh.ny {
                u[i * mesh.ny + j] = (pi * mesh.x_nodes[i]).sin() * (pi * mesh.y_nodes[j]).sin();
            }
        }
        let alpha = 1.0;
        let dt_max =
            0.5 / (alpha * (1.0 / (mesh.hx() * mesh.hx()) + 1.0 / (mesh.hy() * mesh.hy())));
        let dt = 0.9 * dt_max;
        for _ in 0..30 {
            forward_euler_step_2d(&mesh, &mut u, alpha, dt, 0.0, 0.0, 0.0, 0.0).expect("ok");
        }
        // Solution should decay -> peak amplitude < initial
        let initial_peak = 1.0;
        let max_now = u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_now < initial_peak);
        assert!(max_now > 0.0);
    }

    #[test]
    fn heat_2d_cfl_violation() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let mut u = vec![0.5; mesh.n_nodes()];
        let res = forward_euler_step_2d(&mesh, &mut u, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(res.is_err());
    }
}
