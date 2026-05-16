//! 1D heat equation: `u_t = alpha * u_xx` with Dirichlet BCs.
//!
//! Schemes: forward Euler, backward Euler, Crank-Nicolson.

use crate::error::{PdeError, PdeResult};
use crate::fdm::poisson_1d::thomas_solve;
use crate::mesh::Mesh1d;

/// Forward Euler step: `u^{n+1}_i = u^n_i + (alpha*dt/h^2)*(u^n_{i-1} - 2 u^n_i + u^n_{i+1})`.
///
/// Stability: `alpha*dt/h^2 <= 0.5`.
pub fn forward_euler_step(
    mesh: &Mesh1d,
    u: &mut [f64],
    alpha: f64,
    dt: f64,
    ua: f64,
    ub: f64,
) -> PdeResult<()> {
    let n = mesh.n;
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let h = mesh.h();
    if h <= 0.0 {
        return Err(PdeError::InvalidGrid("non-positive h".into()));
    }
    let r = alpha * dt / (h * h);
    if r > 0.5 + 1.0e-12 {
        return Err(PdeError::CflViolation {
            dt,
            dt_max: 0.5 * h * h / alpha,
        });
    }
    let mut next = vec![0.0; n];
    next[0] = ua;
    next[n - 1] = ub;
    for i in 1..n - 1 {
        next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
    }
    u.copy_from_slice(&next);
    Ok(())
}

/// Backward Euler step: solve `(I - r*A) u^{n+1} = u^n + bc`.
///
/// Unconditionally stable.
pub fn backward_euler_step(
    mesh: &Mesh1d,
    u: &mut [f64],
    alpha: f64,
    dt: f64,
    ua: f64,
    ub: f64,
) -> PdeResult<()> {
    let n = mesh.n;
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let h = mesh.h();
    let r = alpha * dt / (h * h);
    let m = n - 2;
    let mut sub = vec![-r; m];
    let mut diag = vec![1.0 + 2.0 * r; m];
    let mut sup = vec![-r; m];
    let mut rhs = vec![0.0; m];
    rhs.copy_from_slice(&u[1..n - 1]);
    rhs[0] += r * ua;
    rhs[m - 1] += r * ub;
    sub[0] = 0.0;
    sup[m - 1] = 0.0;
    let x = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
    u[0] = ua;
    u[n - 1] = ub;
    u[1..n - 1].copy_from_slice(&x);
    Ok(())
}

/// Crank-Nicolson step: `(I - r/2 A) u^{n+1} = (I + r/2 A) u^n`.
///
/// Second-order in time, unconditionally stable.
pub fn crank_nicolson_step(
    mesh: &Mesh1d,
    u: &mut [f64],
    alpha: f64,
    dt: f64,
    ua: f64,
    ub: f64,
) -> PdeResult<()> {
    let n = mesh.n;
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let h = mesh.h();
    let r = alpha * dt / (h * h);
    let m = n - 2;
    // LHS: tridiag(-r/2, 1+r, -r/2)
    let mut sub = vec![-0.5 * r; m];
    let mut diag = vec![1.0 + r; m];
    let mut sup = vec![-0.5 * r; m];
    // RHS = (I + r/2 A) u^n
    let mut rhs = vec![0.0; m];
    for (i, rhs_i) in rhs.iter_mut().enumerate().take(m) {
        let gi = i + 1;
        *rhs_i = u[gi] + 0.5 * r * (u[gi - 1] - 2.0 * u[gi] + u[gi + 1]);
    }
    // boundary contributions for both old and new time levels (we assume BCs are constant)
    rhs[0] += 0.5 * r * ua + 0.5 * r * ua;
    rhs[m - 1] += 0.5 * r * ub + 0.5 * r * ub;
    sub[0] = 0.0;
    sup[m - 1] = 0.0;
    let x = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
    u[0] = ua;
    u[n - 1] = ub;
    u[1..n - 1].copy_from_slice(&x);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_1d_forward_euler_decay() {
        // u(x,0) = sin(pi x) decays like exp(-pi^2 alpha t)
        let pi = std::f64::consts::PI;
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let alpha = 1.0;
        let dt = 0.4 * (mesh.h() * mesh.h()) / alpha;
        let mut u: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
        let t_final = 0.01;
        let nsteps = (t_final / dt).ceil() as usize;
        let dt_used = t_final / nsteps as f64;
        for _ in 0..nsteps {
            forward_euler_step(&mesh, &mut u, alpha, dt_used, 0.0, 0.0).expect("ok");
        }
        let expected_amp = (-pi * pi * alpha * t_final).exp();
        let center = u[mesh.n / 2];
        let analytic_center = (pi * mesh.nodes[mesh.n / 2]).sin() * expected_amp;
        assert!(
            (center - analytic_center).abs() < 5.0e-3,
            "center={center} expected={analytic_center}"
        );
    }

    #[test]
    fn heat_1d_backward_euler_stable_large_dt() {
        let pi = std::f64::consts::PI;
        let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
        let alpha = 1.0;
        let dt = 10.0 * mesh.h() * mesh.h(); // very large -> would blow up FE
        let mut u: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
        for _ in 0..50 {
            backward_euler_step(&mesh, &mut u, alpha, dt, 0.0, 0.0).expect("ok");
        }
        // Solution should be (close to) zero (heat dissipates)
        for v in &u {
            assert!(v.abs() < 1.0e-3);
        }
    }

    #[test]
    fn heat_1d_crank_nicolson_decay() {
        let pi = std::f64::consts::PI;
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let alpha = 1.0_f64;
        let dt = 0.001_f64;
        let mut u: Vec<f64> = mesh.nodes.iter().map(|x| (pi * x).sin()).collect();
        let t_final = 0.05_f64;
        let nsteps = (t_final / dt).round() as usize;
        for _ in 0..nsteps {
            crank_nicolson_step(&mesh, &mut u, alpha, dt, 0.0, 0.0).expect("ok");
        }
        let expected_amp = (-pi * pi * alpha * t_final).exp();
        let center = u[mesh.n / 2];
        let analytic_center = (pi * mesh.nodes[mesh.n / 2]).sin() * expected_amp;
        assert!(
            (center - analytic_center).abs() < 1.0e-3,
            "center={center} expected={analytic_center}"
        );
    }

    #[test]
    fn heat_1d_cfl_violation_detected() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let alpha = 1.0;
        let dt = 1.0; // way too large
        let mut u: Vec<f64> = vec![0.5; mesh.n];
        let res = forward_euler_step(&mesh, &mut u, alpha, dt, 0.0, 0.0);
        assert!(res.is_err());
    }
}
