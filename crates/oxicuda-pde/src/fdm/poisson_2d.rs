//! 2D Poisson solver: `-Δu = f` on a rectangle with Dirichlet BCs (5-point stencil).
//!
//! Uses red-black Gauss-Seidel by default; for higher accuracy use CG on the assembled
//! CSR system (see `solver::cg`).

use crate::error::{PdeError, PdeResult};
use crate::mesh::Mesh2d;
use crate::solver::sparse::SparseCsr;

/// Boundary condition: constant value on each of 4 sides (left, right, bottom, top).
#[derive(Debug, Clone, Copy)]
pub struct DirichletRect {
    pub left: f64,
    pub right: f64,
    pub bottom: f64,
    pub top: f64,
}

/// Build the 5-point Laplacian CSR matrix on the interior (`(nx-2)*(ny-2)` unknowns).
///
/// The system: `(1/hx^2)*(2u - u_E - u_W) + (1/hy^2)*(2u - u_N - u_S) = f` for interior.
pub fn assemble_poisson_2d_csr(mesh: &Mesh2d) -> PdeResult<SparseCsr> {
    if mesh.nx < 3 || mesh.ny < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "poisson 2d csr requires nx>=3 ny>=3, got nx={} ny={}",
            mesh.nx, mesh.ny
        )));
    }
    let mx = mesh.nx - 2;
    let my = mesh.ny - 2;
    let m = mx * my;
    let hx2 = mesh.hx() * mesh.hx();
    let hy2 = mesh.hy() * mesh.hy();
    let inv_hx2 = 1.0 / hx2;
    let inv_hy2 = 1.0 / hy2;
    let diag_val = 2.0 * (inv_hx2 + inv_hy2);
    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for r in 0..m {
        let i = r / my; // 0..mx
        let j = r % my; // 0..my
        // south (j-1)
        if j > 0 {
            cols.push(r - 1);
            vals.push(-inv_hy2);
        }
        // west (i-1)
        if i > 0 {
            cols.push(r - my);
            vals.push(-inv_hx2);
        }
        // diagonal
        cols.push(r);
        vals.push(diag_val);
        // east (i+1)
        if i + 1 < mx {
            cols.push(r + my);
            vals.push(-inv_hx2);
        }
        // north (j+1)
        if j + 1 < my {
            cols.push(r + 1);
            vals.push(-inv_hy2);
        }
        row_ptr.push(cols.len());
    }
    SparseCsr::new(m, m, row_ptr, cols, vals)
}

/// Build the RHS vector accounting for Dirichlet boundary contributions.
pub fn build_poisson_2d_rhs(
    mesh: &Mesh2d,
    f_grid: &[f64],
    bc: DirichletRect,
) -> PdeResult<Vec<f64>> {
    let n_nodes = mesh.n_nodes();
    if f_grid.len() != n_nodes {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_nodes],
            got: vec![f_grid.len()],
        });
    }
    let mx = mesh.nx - 2;
    let my = mesh.ny - 2;
    let m = mx * my;
    let inv_hx2 = 1.0 / (mesh.hx() * mesh.hx());
    let inv_hy2 = 1.0 / (mesh.hy() * mesh.hy());
    let mut rhs = vec![0.0; m];
    for (r, rhs_r) in rhs.iter_mut().enumerate().take(m) {
        let i = r / my;
        let j = r % my;
        let gi = i + 1; // global x-index
        let gj = j + 1; // global y-index
        *rhs_r = f_grid[gi * mesh.ny + gj];
        if i == 0 {
            *rhs_r += inv_hx2 * bc.left;
        }
        if i + 1 == mx {
            *rhs_r += inv_hx2 * bc.right;
        }
        if j == 0 {
            *rhs_r += inv_hy2 * bc.bottom;
        }
        if j + 1 == my {
            *rhs_r += inv_hy2 * bc.top;
        }
    }
    Ok(rhs)
}

/// Solve `-Δu = f` on the 2D rectangular mesh using red-black Gauss-Seidel.
///
/// Returns `u` of length `nx*ny` with boundary set to BC values.
pub fn solve_poisson_2d_gs(
    mesh: &Mesh2d,
    f_grid: &[f64],
    bc: DirichletRect,
    max_iter: usize,
    tol: f64,
) -> PdeResult<(Vec<f64>, usize, f64)> {
    let n_nodes = mesh.n_nodes();
    if f_grid.len() != n_nodes {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_nodes],
            got: vec![f_grid.len()],
        });
    }
    if mesh.nx < 3 || mesh.ny < 3 {
        return Err(PdeError::InvalidGrid("nx,ny must be >=3".into()));
    }
    let mut u = vec![0.0; n_nodes];
    // Initialize boundaries
    for j in 0..mesh.ny {
        u[j] = bc.left;
        u[(mesh.nx - 1) * mesh.ny + j] = bc.right;
    }
    for i in 0..mesh.nx {
        u[i * mesh.ny] = bc.bottom;
        u[i * mesh.ny + mesh.ny - 1] = bc.top;
    }
    let hx2 = mesh.hx() * mesh.hx();
    let hy2 = mesh.hy() * mesh.hy();
    let inv_hx2 = 1.0 / hx2;
    let inv_hy2 = 1.0 / hy2;
    let diag_val = 2.0 * (inv_hx2 + inv_hy2);
    let mut last_res = f64::INFINITY;
    for it in 0..max_iter {
        for color in 0..2 {
            for i in 1..mesh.nx - 1 {
                for j in 1..mesh.ny - 1 {
                    if (i + j) % 2 != color {
                        continue;
                    }
                    let idx = i * mesh.ny + j;
                    let rhs = f_grid[idx]
                        + inv_hx2 * (u[idx - mesh.ny] + u[idx + mesh.ny])
                        + inv_hy2 * (u[idx - 1] + u[idx + 1]);
                    u[idx] = rhs / diag_val;
                }
            }
        }
        // residual L2 over interior
        let mut acc = 0.0;
        let mut cnt = 0;
        for i in 1..mesh.nx - 1 {
            for j in 1..mesh.ny - 1 {
                let idx = i * mesh.ny + j;
                let lap = diag_val * u[idx]
                    - inv_hx2 * (u[idx - mesh.ny] + u[idx + mesh.ny])
                    - inv_hy2 * (u[idx - 1] + u[idx + 1]);
                let r = lap - f_grid[idx];
                acc += r * r;
                cnt += 1;
            }
        }
        let res = if cnt == 0 {
            0.0
        } else {
            (acc / cnt as f64).sqrt()
        };
        last_res = res;
        if res < tol {
            return Ok((u, it + 1, res));
        }
    }
    Ok((u, max_iter, last_res))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_poisson_2d_dimensions() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 5, 5).expect("ok");
        let a = assemble_poisson_2d_csr(&mesh).expect("ok");
        assert_eq!(a.n_rows, 9);
        assert_eq!(a.n_cols, 9);
    }

    #[test]
    fn poisson_2d_gs_constant_rhs() {
        // -Δu = 2 with u=0 on boundary of [0,1]^2 should give a positive symmetric solution.
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 21, 21).expect("ok");
        let f = vec![2.0; mesh.n_nodes()];
        let (u, _, _) = solve_poisson_2d_gs(
            &mesh,
            &f,
            DirichletRect {
                left: 0.0,
                right: 0.0,
                bottom: 0.0,
                top: 0.0,
            },
            5000,
            1e-6,
        )
        .expect("ok");
        // center value of u for -Δu=2 with u=0 on unit-square boundary is ≈ 0.295 (analytic series).
        let center = u[10 * mesh.ny + 10];
        assert!(center > 0.0 && center < 0.6, "center={center}");
    }

    #[test]
    fn poisson_2d_zero_rhs_constant_bc() {
        // u=1 on boundary => u=1 everywhere.
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 7, 7).expect("ok");
        let f = vec![0.0; mesh.n_nodes()];
        let (u, _, _) = solve_poisson_2d_gs(
            &mesh,
            &f,
            DirichletRect {
                left: 1.0,
                right: 1.0,
                bottom: 1.0,
                top: 1.0,
            },
            2000,
            1e-10,
        )
        .expect("ok");
        for v in u.iter() {
            assert!((v - 1.0).abs() < 1.0e-4);
        }
    }
}
