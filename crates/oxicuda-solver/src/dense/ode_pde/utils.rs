//! Utility functions for ODE/PDE solvers.

use crate::error::{SolverError, SolverResult};

use super::pde::BoundaryCondition;
use super::types::OdeConfig;

/// Compute the numerical Jacobian of an ODE system by finite differences.
///
/// Returns an `n x n` matrix where `J[i][j] = df_i / dy_j`, approximated
/// by a first-order forward difference with perturbation `eps`.
pub fn numerical_jacobian(
    system: &dyn super::types::OdeSystem,
    t: f64,
    y: &[f64],
    eps: f64,
) -> SolverResult<Vec<Vec<f64>>> {
    let n = system.dim();
    let mut f0 = vec![0.0; n];
    system.rhs(t, y, &mut f0)?;

    let mut jac = vec![vec![0.0; n]; n];
    let mut y_pert = y.to_vec();
    let mut f_pert = vec![0.0; n];

    for j in 0..n {
        let h = eps * y[j].abs().max(1.0);
        y_pert[j] = y[j] + h;
        system.rhs(t, &y_pert, &mut f_pert)?;

        for i in 0..n {
            jac[i][j] = (f_pert[i] - f0[i]) / h;
        }
        y_pert[j] = y[j]; // restore
    }

    Ok(jac)
}

/// Solve a tridiagonal system using the Thomas algorithm.
///
/// `a` is the sub-diagonal (length n-1), `b` the main diagonal (length n),
/// `c` the super-diagonal (length n-1), and `d` the right-hand side (length n).
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> SolverResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        if b[0].abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        return Ok(vec![d[0] / b[0]]);
    }
    if a.len() < n - 1 || c.len() < n - 1 || d.len() < n {
        return Err(SolverError::DimensionMismatch(
            "solve_tridiagonal: inconsistent array lengths".to_string(),
        ));
    }

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    if b[0].abs() < 1e-300 {
        return Err(SolverError::SingularMatrix);
    }

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

// =========================================================================
// Internal helpers
// =========================================================================

/// Validate ODE input dimensions and config.
pub(super) fn validate_ode_inputs(dim: usize, y0: &[f64], config: &OdeConfig) -> SolverResult<()> {
    if dim == 0 {
        return Err(SolverError::DimensionMismatch(
            "ODE system dimension must be > 0".to_string(),
        ));
    }
    if y0.len() != dim {
        return Err(SolverError::DimensionMismatch(format!(
            "y0 length ({}) != system dimension ({dim})",
            y0.len()
        )));
    }
    if config.dt <= 0.0 {
        return Err(SolverError::InternalError(
            "step size dt must be positive".to_string(),
        ));
    }
    if config.t_end <= config.t_start {
        return Err(SolverError::InternalError(
            "t_end must be greater than t_start".to_string(),
        ));
    }
    Ok(())
}

/// Euclidean norm of a vector.
pub(super) fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Apply 1-D boundary conditions in-place.
pub(super) fn apply_bc_1d(
    u: &mut [f64],
    bc_left: &BoundaryCondition,
    bc_right: &BoundaryCondition,
    nx: usize,
) {
    if nx == 0 {
        return;
    }
    match *bc_left {
        BoundaryCondition::Dirichlet(val) => {
            u[0] = val;
        }
        BoundaryCondition::Neumann(val) => {
            // Forward difference: (u[1] - u[0]) / dx = val => u[0] = u[1] - dx*val
            // dx is not available here, so we use a simple ghost-point approach:
            // u[0] = u[1] (zero Neumann approximation when val = 0)
            if nx > 1 {
                u[0] = u[1] - val; // caller must scale val by dx if needed
            }
        }
        BoundaryCondition::Periodic => {
            if nx > 1 {
                u[0] = u[nx - 2];
            }
        }
    }
    match *bc_right {
        BoundaryCondition::Dirichlet(val) => {
            u[nx - 1] = val;
        }
        BoundaryCondition::Neumann(val) => {
            if nx > 1 {
                u[nx - 1] = u[nx - 2] + val;
            }
        }
        BoundaryCondition::Periodic => {
            if nx > 1 {
                u[nx - 1] = u[1];
            }
        }
    }
}

/// Solve a dense linear system A*x = b using Gaussian elimination with
/// partial pivoting. Used internally by the implicit ODE solvers for
/// Newton iteration.
pub(super) fn solve_dense_system(a: &[Vec<f64>], b: &[f64]) -> SolverResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        row.extend_from_slice(&a[i]);
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            let val = aug_row[col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            // Cannot borrow aug mutably and immutably at same time,
            // so copy the pivot row values we need.
            let pivot_row: Vec<f64> = aug[col][col..=n].to_vec();
            for (j, &pv) in (col..=n).zip(pivot_row.iter()) {
                aug[row][j] -= factor * pv;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::super::pde::BoundaryCondition;
    use super::super::types::OdeSystem;
    use super::*;
    use std::f64::consts::PI;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // ---------------------------------------------------------------------
    // Thomas algorithm (tridiagonal solver)
    // ---------------------------------------------------------------------

    #[test]
    fn tridiagonal_solves_known_system() {
        // 4x4 system with sub/super-diagonal 1 and main diagonal 4.
        // Constructed so that the exact solution is x = [1, 2, 3, 4].
        // Row products:
        //   4*1 + 1*2            = 6
        //   1*1 + 4*2 + 1*3      = 12
        //   1*2 + 4*3 + 1*4      = 18
        //   1*3 + 4*4            = 19
        let sub = [1.0, 1.0, 1.0];
        let main = [4.0, 4.0, 4.0, 4.0];
        let sup = [1.0, 1.0, 1.0];
        let rhs = [6.0, 12.0, 18.0, 19.0];
        let x = solve_tridiagonal(&sub, &main, &sup, &rhs).expect("tridiagonal solve");
        let expected = [1.0, 2.0, 3.0, 4.0];
        assert!(max_abs_diff(&x, &expected) < 1e-12);
    }

    #[test]
    fn tridiagonal_edge_cases_and_singular() {
        // Empty system => empty solution.
        let empty = solve_tridiagonal(&[], &[], &[], &[]).expect("empty system");
        assert!(empty.is_empty());

        // 1x1 system: 2 * x = 6 => x = 3.
        let one = solve_tridiagonal(&[], &[2.0], &[], &[6.0]).expect("1x1 system");
        assert_eq!(one.len(), 1);
        assert!((one[0] - 3.0).abs() < 1e-12);

        // 1x1 singular (zero pivot) is rejected.
        let singular = solve_tridiagonal(&[], &[0.0], &[], &[6.0]);
        assert!(matches!(singular, Err(SolverError::SingularMatrix)));

        // Inconsistent array lengths are rejected.
        let bad = solve_tridiagonal(&[1.0], &[1.0, 1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert!(matches!(bad, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn tridiagonal_laplacian_second_order_convergence() {
        // Solve the discrete 1-D Poisson problem -u'' = f directly through the
        // Thomas solver using the standard second-derivative stencil [-1, 2, -1].
        // With f = pi^2 sin(pi x) on [0,1] and homogeneous Dirichlet data the
        // exact solution is sin(pi x); the error must shrink as O(dx^2).
        let mut errors = Vec::new();
        for &nx in &[11usize, 21, 41] {
            let dx = 1.0 / (nx - 1) as f64;
            let m = nx - 2; // interior unknowns
            let sub = vec![-1.0; m - 1];
            let main = vec![2.0; m];
            let sup = vec![-1.0; m - 1];
            let rhs: Vec<f64> = (0..m)
                .map(|i| {
                    let x = (i + 1) as f64 * dx;
                    dx * dx * PI * PI * (PI * x).sin()
                })
                .collect();
            let interior = solve_tridiagonal(&sub, &main, &sup, &rhs).expect("laplacian solve");
            let exact: Vec<f64> = (0..m).map(|i| (PI * (i + 1) as f64 * dx).sin()).collect();
            errors.push(max_abs_diff(&interior, &exact));
        }
        for window in errors.windows(2) {
            let ratio = window[0] / window[1];
            assert!(
                ratio > 3.7 && ratio < 4.3,
                "expected ~4x error reduction, got ratio {ratio}"
            );
        }
        assert!(errors.last().expect("errors") < &1e-3);
    }

    // ---------------------------------------------------------------------
    // Numerical Jacobian (forward finite differences)
    // ---------------------------------------------------------------------

    /// Linear system f(y) = A y with constant Jacobian A = [[2, -1], [0, 3]].
    struct LinearSystem;
    impl OdeSystem for LinearSystem {
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = 2.0 * y[0] - y[1];
            dydt[1] = 3.0 * y[1];
            Ok(())
        }
        fn dim(&self) -> usize {
            2
        }
    }

    /// Scalar nonlinear system f(y) = y^2, exact Jacobian 2*y.
    struct SquareSystem;
    impl OdeSystem for SquareSystem {
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = y[0] * y[0];
            Ok(())
        }
        fn dim(&self) -> usize {
            1
        }
    }

    #[test]
    fn numerical_jacobian_linear_exact_and_quadratic_first_order() {
        // Forward differences are exact (to round-off) for a linear right-hand
        // side: J must equal the constant coefficient matrix.
        let jac = numerical_jacobian(&LinearSystem, 0.0, &[1.0, 2.0], 1e-6).expect("jacobian");
        let expected = [[2.0, -1.0], [0.0, 3.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((jac[i][j] - expected[i][j]).abs() < 1e-6);
            }
        }

        // For f(y) = y^2 the forward difference recovers 2*y to first order.
        let jac2 = numerical_jacobian(&SquareSystem, 0.0, &[1.5], 1e-6).expect("jacobian2");
        assert!((jac2[0][0] - 3.0).abs() < 1e-4);
    }

    // ---------------------------------------------------------------------
    // Vector norm
    // ---------------------------------------------------------------------

    #[test]
    fn vec_norm_euclidean() {
        // Pythagorean triples give exact integer norms.
        assert!((vec_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
        assert!((vec_norm(&[1.0, 2.0, 2.0]) - 3.0).abs() < 1e-15);
        // The empty vector has zero norm.
        assert_eq!(vec_norm(&[]), 0.0);
    }

    // ---------------------------------------------------------------------
    // Boundary-condition application
    // ---------------------------------------------------------------------

    #[test]
    fn apply_bc_1d_dirichlet_neumann_periodic() {
        // Dirichlet: end nodes are pinned to the prescribed values, interior
        // values are untouched.
        let mut u = [9.0; 5];
        apply_bc_1d(
            &mut u,
            &BoundaryCondition::Dirichlet(1.0),
            &BoundaryCondition::Dirichlet(2.0),
            5,
        );
        assert_eq!(u[0], 1.0);
        assert_eq!(u[4], 2.0);
        assert_eq!(&u[1..4], &[9.0, 9.0, 9.0]);

        // Neumann: ghost-node update u[0] = u[1] - val, u[n-1] = u[n-2] + val.
        let mut un = [5.0, 3.0, 7.0, 4.0, 9.0];
        apply_bc_1d(
            &mut un,
            &BoundaryCondition::Neumann(0.5),
            &BoundaryCondition::Neumann(0.2),
            5,
        );
        assert!((un[0] - 2.5).abs() < 1e-15); // 3.0 - 0.5
        assert!((un[4] - 4.2).abs() < 1e-15); // 4.0 + 0.2

        // Periodic: u[0] = u[n-2], u[n-1] = u[1].
        let mut up = [10.0, 20.0, 30.0, 40.0, 50.0];
        apply_bc_1d(
            &mut up,
            &BoundaryCondition::Periodic,
            &BoundaryCondition::Periodic,
            5,
        );
        assert_eq!(up[0], 40.0);
        assert_eq!(up[4], 20.0);
    }

    // ---------------------------------------------------------------------
    // Dense Gaussian elimination with partial pivoting
    // ---------------------------------------------------------------------

    #[test]
    fn solve_dense_system_with_pivoting_and_singular() {
        // A x = b with known x = [1, 2, 3].
        let a = vec![
            vec![2.0, 1.0, 1.0],
            vec![4.0, 1.0, 0.0],
            vec![-1.0, 2.0, 1.0],
        ];
        let b = [7.0, 6.0, 6.0];
        let x = solve_dense_system(&a, &b).expect("dense solve");
        assert!(max_abs_diff(&x, &[1.0, 2.0, 3.0]) < 1e-10);

        // A zero leading pivot forces a row swap; solution is still exact.
        let a_pivot = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let b_pivot = [6.0, 5.0, 14.0];
        let x_pivot = solve_dense_system(&a_pivot, &b_pivot).expect("pivot solve");
        assert!(max_abs_diff(&x_pivot, &[5.0, 6.0, 7.0]) < 1e-10);

        // A rank-deficient matrix is reported as singular.
        let a_sing = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b_sing = [3.0, 6.0];
        assert!(matches!(
            solve_dense_system(&a_sing, &b_sing),
            Err(SolverError::SingularMatrix)
        ));
    }
}
