//! 1D Poisson solver: `-u''(x) = f(x)` with Dirichlet BCs.
//!
//! Uses second-order central differences on a uniform grid and the Thomas algorithm
//! to solve the resulting tridiagonal system in O(n).

use crate::error::{PdeError, PdeResult};
use crate::mesh::Mesh1d;

/// Boundary conditions for 1D Poisson: u(x0) = ua, u(x1) = ub.
#[derive(Debug, Clone, Copy)]
pub struct Dirichlet1d {
    pub ua: f64,
    pub ub: f64,
}

/// Solve `-u''(x) = f(x)` on the mesh with Dirichlet BCs.
///
/// Returns `u` of length `mesh.n` with `u[0]=ua, u[n-1]=ub`.
pub fn solve_poisson_1d(mesh: &Mesh1d, f_vals: &[f64], bc: Dirichlet1d) -> PdeResult<Vec<f64>> {
    let n = mesh.n;
    if f_vals.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![f_vals.len()],
        });
    }
    if n < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "poisson 1d requires n>=3, got {n}"
        )));
    }
    let h = mesh.h();
    if h <= 0.0 {
        return Err(PdeError::InvalidGrid(format!(
            "non-positive mesh spacing h={h}"
        )));
    }
    // Build tridiagonal system for interior nodes 1..n-1 (size m=n-2).
    let m = n - 2;
    let inv_h2 = 1.0 / (h * h);
    // sub = -inv_h2, diag = 2*inv_h2, sup = -inv_h2
    let mut sub = vec![-inv_h2; m];
    let mut diag = vec![2.0 * inv_h2; m];
    let mut sup = vec![-inv_h2; m];
    let mut rhs = vec![0.0; m];
    rhs.copy_from_slice(&f_vals[1..n - 1]);
    rhs[0] += inv_h2 * bc.ua;
    rhs[m - 1] += inv_h2 * bc.ub;
    sub[0] = 0.0;
    sup[m - 1] = 0.0;
    let x = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
    let mut u = vec![0.0; n];
    u[0] = bc.ua;
    u[n - 1] = bc.ub;
    u[1..n - 1].copy_from_slice(&x);
    Ok(u)
}

/// Thomas algorithm for tridiagonal systems: in-place, returns x in `rhs` (and copy).
///
/// `sub[0]` is ignored (no element below first row), `sup[m-1]` is ignored.
pub fn thomas_solve(
    sub: &[f64],
    diag: &mut [f64],
    sup: &mut [f64],
    rhs: &mut [f64],
) -> PdeResult<Vec<f64>> {
    let m = diag.len();
    if sub.len() != m || sup.len() != m || rhs.len() != m {
        return Err(PdeError::DimensionMismatch {
            a: diag.len(),
            b: sub.len(),
        });
    }
    if m == 0 {
        return Ok(Vec::new());
    }
    // Forward sweep
    for i in 1..m {
        if diag[i - 1].abs() < 1.0e-300 {
            return Err(PdeError::SingularMatrix(format!(
                "thomas: zero pivot at row {}",
                i - 1
            )));
        }
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    if diag[m - 1].abs() < 1.0e-300 {
        return Err(PdeError::SingularMatrix("thomas: last pivot zero".into()));
    }
    // Back substitution
    let mut x = vec![0.0; m];
    x[m - 1] = rhs[m - 1] / diag[m - 1];
    for i in (0..m - 1).rev() {
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_1d_constant_rhs() {
        // -u''(x) = 2 on [0,1], u(0)=u(1)=0 => u(x) = x*(1-x)
        let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
        let f: Vec<f64> = vec![2.0; mesh.n];
        let u = solve_poisson_1d(&mesh, &f, Dirichlet1d { ua: 0.0, ub: 0.0 }).expect("ok");
        for (i, &ui) in u.iter().enumerate() {
            let x = mesh.nodes[i];
            let expected = x * (1.0 - x);
            assert!(
                (ui - expected).abs() < 1.0e-3,
                "i={i} u={ui} expected={expected}"
            );
        }
    }

    #[test]
    fn poisson_1d_sine_convergence() {
        // -u''(x) = pi^2 sin(pi x) on [0,1], u(0)=u(1)=0 => u(x) = sin(pi x)
        let pi = std::f64::consts::PI;
        let ns = [21, 41, 81];
        let mut errs = Vec::new();
        for &n in &ns {
            let mesh = Mesh1d::uniform(0.0, 1.0, n).expect("ok");
            let f: Vec<f64> = mesh
                .nodes
                .iter()
                .map(|x| pi * pi * (pi * x).sin())
                .collect();
            let u = solve_poisson_1d(&mesh, &f, Dirichlet1d { ua: 0.0, ub: 0.0 }).expect("ok");
            let err: f64 = u
                .iter()
                .zip(mesh.nodes.iter())
                .map(|(ui, x)| (ui - (pi * x).sin()).abs())
                .fold(0.0_f64, |a, b| a.max(b));
            errs.push(err);
        }
        // Halving h should drop max-error ~ 4x (O(h^2)).
        let r1 = errs[0] / errs[1];
        let r2 = errs[1] / errs[2];
        assert!(r1 > 3.0 && r1 < 5.0, "r1={r1} errs={errs:?}");
        assert!(r2 > 3.0 && r2 < 5.0, "r2={r2} errs={errs:?}");
    }

    #[test]
    fn thomas_identity_system() {
        let sub = vec![0.0, 0.0, 0.0];
        let mut diag = vec![1.0, 1.0, 1.0];
        let mut sup = vec![0.0, 0.0, 0.0];
        let mut rhs = vec![3.5, -1.0, 2.0];
        let x = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs).expect("ok");
        assert!((x[0] - 3.5).abs() < 1.0e-12);
        assert!((x[1] + 1.0).abs() < 1.0e-12);
        assert!((x[2] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn thomas_simple_system() {
        // [[2,-1,0],[-1,2,-1],[0,-1,2]] x = [1,0,1] => x = [1,1,1]
        let sub = vec![0.0, -1.0, -1.0];
        let mut diag = vec![2.0, 2.0, 2.0];
        let mut sup = vec![-1.0, -1.0, 0.0];
        let mut rhs = vec![1.0, 0.0, 1.0];
        let x = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs).expect("ok");
        for v in &x {
            assert!((v - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn poisson_1d_non_homogeneous() {
        // -u''(x) = 0, u(0)=1, u(1)=3 => u(x)=1+2x
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let f = vec![0.0; mesh.n];
        let u = solve_poisson_1d(&mesh, &f, Dirichlet1d { ua: 1.0, ub: 3.0 }).expect("ok");
        for (i, &ui) in u.iter().enumerate() {
            let expected = 1.0 + 2.0 * mesh.nodes[i];
            assert!((ui - expected).abs() < 1.0e-10);
        }
    }
}
