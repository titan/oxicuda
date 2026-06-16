//! Tensor-product Chebyshev collocation for the 2D Poisson equation.
//!
//! Solves `-Δu = f` on a rectangle `[ax,bx] × [ay,by]` with Dirichlet boundary
//! conditions by Chebyshev-Gauss-Lobatto collocation in each coordinate.
//!
//! # Method
//!
//! The 1D Chebyshev second-derivative operator `D2 = D1·D1` (built from the
//! Trefethen differentiation matrix in [`crate::spectral::chebyshev`]) is formed
//! on `Nx+1` and `Ny+1` nodes. After the affine map `[-1,1] → [a,b]` the physical
//! operator carries the chain-rule factor `(2/(b-a))²`.
//!
//! The 2D negative Laplacian acts on the tensor grid through the Kronecker
//! structure
//!
//! ```text
//! L = (I_y ⊗ D2x) + (D2y ⊗ I_x).
//! ```
//!
//! Dirichlet boundary values are imposed strongly: only the interior nodes are
//! unknown, and the known boundary contributions are moved to the right-hand
//! side. The resulting dense interior system is solved by Gaussian elimination
//! with partial pivoting (reusing [`crate::spectral::chebyshev::gauss_solve_dense`]).
//!
//! Because the basis is global and the operator spectral, the nodal error decays
//! super-algebraically (faster than any power of `h`) for smooth data.
//!
//! Reference: Trefethen, *Spectral Methods in MATLAB* (2000), Chapters 6 & 7.

use crate::error::{PdeError, PdeResult};
use crate::spectral::chebyshev::{cheb_diff_matrix, gauss_solve_dense};

/// Axis-aligned rectangular domain `[ax,bx] × [ay,by]`.
#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    /// Lower x bound.
    pub ax: f64,
    /// Upper x bound.
    pub bx: f64,
    /// Lower y bound.
    pub ay: f64,
    /// Upper y bound.
    pub by: f64,
}

impl Rectangle {
    /// Construct a rectangle, validating `bx>ax` and `by>ay`.
    pub fn new(ax: f64, bx: f64, ay: f64, by: f64) -> PdeResult<Self> {
        if bx <= ax || by <= ay {
            return Err(PdeError::InvalidGrid(format!(
                "rectangle requires bx>ax and by>ay, got ax={ax} bx={bx} ay={ay} by={by}"
            )));
        }
        Ok(Self { ax, bx, ay, by })
    }
}

/// Chebyshev-Gauss-Lobatto nodes mapped to a physical interval `[a, b]`.
///
/// The reference nodes are `ξ_i = cos(iπ/n)` (so `ξ_0=1`, `ξ_n=-1`). They are
/// mapped by `x_i = a + (b-a)·(1-ξ_i)/2`, giving an **increasing** sequence with
/// `x_0 = a` and `x_n = b`.
pub fn cheb_nodes_mapped(n: usize, a: f64, b: f64) -> Vec<f64> {
    if n == 0 {
        return vec![0.5 * (a + b)];
    }
    let half = 0.5 * (b - a);
    (0..=n)
        .map(|j| {
            let xi = (std::f64::consts::PI * j as f64 / n as f64).cos();
            a + half * (1.0 - xi)
        })
        .collect()
}

/// Second-derivative Chebyshev operator `D2` on the physical interval `[a,b]`.
///
/// Returns the `(n+1)×(n+1)` row-major matrix `(2/(b-a))² · (D1·D1)` where `D1`
/// is the Trefethen differentiation matrix. The reflection of the node ordering
/// performed by [`cheb_nodes_mapped`] leaves the *second* derivative invariant,
/// so only the squared length scale enters.
fn cheb_d2_mapped(n: usize, a: f64, b: f64) -> PdeResult<Vec<f64>> {
    if n < 2 {
        return Err(PdeError::InvalidOrder {
            order: n,
            reason: "n>=2 required for a 2D Chebyshev Laplacian".into(),
        });
    }
    let (_xi, d1) = cheb_diff_matrix(n)?;
    let n1 = n + 1;
    // D2 = D1 · D1.
    let mut d2 = vec![0.0_f64; n1 * n1];
    for i in 0..n1 {
        for j in 0..n1 {
            let mut s = 0.0;
            for k in 0..n1 {
                s += d1[i * n1 + k] * d1[k * n1 + j];
            }
            d2[i * n1 + j] = s;
        }
    }
    let scale = (2.0 / (b - a)).powi(2);
    for v in d2.iter_mut() {
        *v *= scale;
    }
    Ok(d2)
}

/// Generate the full physical tensor grid (size `(nx+1)*(ny+1)`).
///
/// The returned vectors are the mapped 1D nodes in each direction; the global
/// node `(ix, iy)` has coordinates `(x[ix], y[iy])` and flat index
/// `iy*(nx+1) + ix` (x is the fastest-varying / row-major inner index).
pub fn chebyshev_2d_grid(nx: usize, ny: usize, domain: &Rectangle) -> (Vec<f64>, Vec<f64>) {
    let x = cheb_nodes_mapped(nx, domain.ax, domain.bx);
    let y = cheb_nodes_mapped(ny, domain.ay, domain.by);
    (x, y)
}

/// Solve `-Δu = f` on `domain` by tensor-product Chebyshev collocation with
/// Dirichlet boundary conditions.
///
/// # Arguments
/// * `nx`, `ny` — spectral orders (`nx+1` × `ny+1` collocation nodes).
/// * `domain` — the rectangle `[ax,bx] × [ay,by]`.
/// * `f_at_nodes` — the forcing sampled on the full tensor grid in the layout
///   `iy*(nx+1) + ix` (see [`chebyshev_2d_grid`]); length `(nx+1)*(ny+1)`.
/// * `boundary_values` — prescribed `u` on the full tensor grid, same layout and
///   length. Only entries on the rectangle boundary are read.
///
/// # Returns
/// The solution `u` on the full `(nx+1)*(ny+1)` tensor grid (interior values
/// computed, boundary values copied from `boundary_values`).
pub fn chebyshev_2d_poisson(
    nx: usize,
    ny: usize,
    domain: &Rectangle,
    f_at_nodes: &[f64],
    boundary_values: &[f64],
) -> PdeResult<Vec<f64>> {
    if nx < 2 || ny < 2 {
        return Err(PdeError::InvalidOrder {
            order: nx.min(ny),
            reason: "nx>=2 and ny>=2 required".into(),
        });
    }
    let nx1 = nx + 1;
    let ny1 = ny + 1;
    let n_total = nx1 * ny1;
    if f_at_nodes.len() != n_total {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_total],
            got: vec![f_at_nodes.len()],
        });
    }
    if boundary_values.len() != n_total {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_total],
            got: vec![boundary_values.len()],
        });
    }

    let d2x = cheb_d2_mapped(nx, domain.ax, domain.bx)?;
    let d2y = cheb_d2_mapped(ny, domain.ay, domain.by)?;

    // Interior index sets (strip the two endpoints in each direction).
    let nxi = nx - 1; // interior count in x
    let nyi = ny - 1; // interior count in y
    let m = nxi * nyi; // total interior unknowns

    // Map an interior pair (ax_i in 0..nxi, ay_i in 0..nyi) to a dense row.
    let lin = |ax_i: usize, ay_i: usize| ay_i * nxi + ax_i;

    let mut a = vec![0.0_f64; m * m];
    let mut rhs = vec![0.0_f64; m];

    // Build -L on interior nodes, moving boundary contributions to the RHS.
    // For interior node (ix, iy) with ix=ax_i+1, iy=ay_i+1:
    //   (-L u)(ix,iy) = -sum_k D2x[ix,k] u(k,iy) - sum_k D2y[iy,k] u(ix,k) = f(ix,iy)
    for ay_i in 0..nyi {
        let iy = ay_i + 1;
        for ax_i in 0..nxi {
            let ix = ax_i + 1;
            let row = lin(ax_i, ay_i);
            // x-direction coupling (vary kx, hold iy).
            for kx in 0..nx1 {
                let coef = -d2x[ix * nx1 + kx];
                if kx == 0 || kx == nx {
                    // boundary node (kx, iy)
                    rhs[row] -= coef * boundary_values[iy * nx1 + kx];
                } else {
                    let col = lin(kx - 1, ay_i);
                    a[row * m + col] += coef;
                }
            }
            // y-direction coupling (vary ky, hold ix).
            for ky in 0..ny1 {
                let coef = -d2y[iy * ny1 + ky];
                if ky == 0 || ky == ny {
                    rhs[row] -= coef * boundary_values[ky * nx1 + ix];
                } else {
                    let col = lin(ax_i, ky - 1);
                    a[row * m + col] += coef;
                }
            }
            // forcing
            rhs[row] += f_at_nodes[iy * nx1 + ix];
        }
    }

    let u_int = gauss_solve_dense(&mut a, &mut rhs, m)?;

    // Assemble full grid: boundary copied, interior filled.
    let mut u = boundary_values.to_vec();
    for ay_i in 0..nyi {
        let iy = ay_i + 1;
        for ax_i in 0..nxi {
            let ix = ax_i + 1;
            u[iy * nx1 + ix] = u_int[lin(ax_i, ay_i)];
        }
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn max_abs_err(u: &[f64], exact: &[f64]) -> f64 {
        u.iter()
            .zip(exact)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
    }

    #[test]
    fn mapped_nodes_span_interval() {
        let x = cheb_nodes_mapped(8, -2.0, 3.0);
        assert_eq!(x.len(), 9);
        assert!((x[0] - (-2.0)).abs() < 1e-12);
        assert!((x[8] - 3.0).abs() < 1e-12);
        // strictly increasing
        for w in x.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn d2_differentiates_quadratic_exactly() {
        // d^2/dx^2 (x^2) = 2 on [0,1] mapped Chebyshev nodes.
        let n = 10;
        let a = 0.0;
        let b = 1.0;
        let x = cheb_nodes_mapped(n, a, b);
        let d2 = cheb_d2_mapped(n, a, b).expect("ok");
        let v: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let n1 = n + 1;
        for i in 0..n1 {
            let mut s = 0.0;
            for j in 0..n1 {
                s += d2[i * n1 + j] * v[j];
            }
            assert!((s - 2.0).abs() < 1e-7, "row {i}: {s}");
        }
    }

    #[test]
    fn manufactured_sine_spectral_accuracy() {
        // u = sin(pi x) sin(pi y), -Δu = 2π² u, homogeneous Dirichlet on [0,1]².
        let domain = Rectangle::new(0.0, 1.0, 0.0, 1.0).expect("ok");
        let n = 20;
        let (x, y) = chebyshev_2d_grid(n, n, &domain);
        let n1 = n + 1;
        let mut f = vec![0.0; n1 * n1];
        let mut exact = vec![0.0; n1 * n1];
        for (iy, &yy) in y.iter().enumerate() {
            for (ix, &xx) in x.iter().enumerate() {
                let u = (PI * xx).sin() * (PI * yy).sin();
                exact[iy * n1 + ix] = u;
                f[iy * n1 + ix] = 2.0 * PI * PI * u;
            }
        }
        let bc = vec![0.0; n1 * n1];
        let u = chebyshev_2d_poisson(n, n, &domain, &f, &bc).expect("ok");
        let err = max_abs_err(&u, &exact);
        assert!(err < 1e-8, "spectral error too large: {err}");
    }

    #[test]
    fn polynomial_recovered_to_machine_precision() {
        // u = x^2 y - is harmonic? -Δ(x^2 y) = -(2y + 0) = -2y, so f = 2y.
        // Dirichlet data = exact u on the boundary.
        let domain = Rectangle::new(0.0, 1.0, 0.0, 1.0).expect("ok");
        let nx = 8;
        let ny = 6;
        let (x, y) = chebyshev_2d_grid(nx, ny, &domain);
        let nx1 = nx + 1;
        let ny1 = ny + 1;
        let mut f = vec![0.0; nx1 * ny1];
        let mut exact = vec![0.0; nx1 * ny1];
        let mut bc = vec![0.0; nx1 * ny1];
        // u = x^2 y  =>  Δu = d2u/dx2 + d2u/dy2 = 2y + 0,  so -Δu = -2y, f = -2y.
        for (iy, &yy) in y.iter().enumerate() {
            for (ix, &xx) in x.iter().enumerate() {
                let g = iy * nx1 + ix;
                exact[g] = xx * xx * yy;
                f[g] = -2.0 * yy;
                if ix == 0 || ix == nx || iy == 0 || iy == ny {
                    bc[g] = exact[g];
                }
            }
        }
        let u = chebyshev_2d_poisson(nx, ny, &domain, &f, &bc).expect("ok");
        let err = max_abs_err(&u, &exact);
        assert!(err < 1e-10, "polynomial not exact: {err}");
    }

    #[test]
    fn super_algebraic_convergence() {
        // Error must drop by orders of magnitude from N=8 to N=16 (spectral),
        // not the ~4x of a 2nd-order scheme.
        let domain = Rectangle::new(0.0, 1.0, 0.0, 1.0).expect("ok");
        let err_at = |n: usize| -> f64 {
            let (x, y) = chebyshev_2d_grid(n, n, &domain);
            let n1 = n + 1;
            let mut f = vec![0.0; n1 * n1];
            let mut exact = vec![0.0; n1 * n1];
            for (iy, &yy) in y.iter().enumerate() {
                for (ix, &xx) in x.iter().enumerate() {
                    let u = (PI * xx).sin() * (PI * yy).sin();
                    exact[iy * n1 + ix] = u;
                    f[iy * n1 + ix] = 2.0 * PI * PI * u;
                }
            }
            let bc = vec![0.0; n1 * n1];
            let u = chebyshev_2d_poisson(n, n, &domain, &f, &bc).expect("ok");
            max_abs_err(&u, &exact)
        };
        let e8 = err_at(8);
        let e16 = err_at(16);
        // super-algebraic: ratio far exceeds (16/8)^2 = 4.
        assert!(e8 / e16 > 1.0e3, "not spectral: e8={e8} e16={e16}");
    }

    #[test]
    fn rejects_bad_orders_and_shapes() {
        let domain = Rectangle::new(0.0, 1.0, 0.0, 1.0).expect("ok");
        assert!(chebyshev_2d_poisson(1, 4, &domain, &[], &[]).is_err());
        let n = 4;
        let n1 = n + 1;
        let good = vec![0.0; n1 * n1];
        let bad = vec![0.0; n1 * n1 - 1];
        assert!(chebyshev_2d_poisson(n, n, &domain, &bad, &good).is_err());
        assert!(chebyshev_2d_poisson(n, n, &domain, &good, &bad).is_err());
    }
}
