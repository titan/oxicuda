//! Crank-Nicolson 2D heat equation via the Peaceman-Rachford ADI scheme.
//!
//! Solves `u_t = α·(u_xx + u_yy)` on a rectangle with Dirichlet boundary
//! conditions. A direct Crank-Nicolson discretisation in 2D couples every
//! interior node and yields a large banded system; the **Alternating Direction
//! Implicit** (ADI) method of Peaceman & Rachford (1955) splits each time step
//! into two half-steps, each of which is implicit in only one coordinate
//! direction and therefore reduces to a set of independent **tridiagonal**
//! solves:
//!
//! ```text
//! Half-step 1 (implicit in x, explicit in y):
//!   (I − (r_x/2) δ_xx) u* = (I + (r_y/2) δ_yy) u^n
//! Half-step 2 (implicit in y, explicit in x):
//!   (I − (r_y/2) δ_yy) u^{n+1} = (I + (r_x/2) δ_xx) u*
//! ```
//!
//! with `r_x = α·Δt/h_x²` and `r_y = α·Δt/h_y²`. Each half-step performs one
//! tridiagonal solve per grid line, giving `O(N)` work per time step for an
//! `N`-node grid. The method is second-order accurate in both space and time
//! and unconditionally stable.
//!
//! The grid uses the row-major layout `idx(i, j) = i·ny + j` matching
//! [`crate::mesh::Mesh2d`].

use crate::error::{PdeError, PdeResult};
use crate::fdm::poisson_1d::thomas_solve;
use crate::mesh::Mesh2d;

/// Peaceman-Rachford ADI solver for the 2D heat equation on a fixed mesh.
#[derive(Debug, Clone)]
pub struct CrankNicolson2d {
    /// Diffusion coefficient `α > 0`.
    pub alpha: f64,
    /// Time step `Δt > 0`.
    pub dt: f64,
}

impl CrankNicolson2d {
    /// Build an ADI solver.
    ///
    /// Returns [`PdeError::InvalidParameter`] for non-positive `alpha` or `dt`.
    pub fn new(alpha: f64, dt: f64) -> PdeResult<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "alpha".into(),
                reason: format!("must be a finite value > 0, got {alpha}"),
            });
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be a finite value > 0, got {dt}"),
            });
        }
        Ok(Self { alpha, dt })
    }

    /// Advance one full ADI time step in place.
    ///
    /// * `mesh` — uniform 2D mesh.
    /// * `u` — current field of length `mesh.n_nodes()` (row-major `i·ny + j`),
    ///   overwritten with the next time level. Boundary entries are treated as
    ///   fixed Dirichlet data and are preserved.
    pub fn step(&self, mesh: &Mesh2d, u: &mut [f64]) -> PdeResult<()> {
        let nx = mesh.nx;
        let ny = mesh.ny;
        if u.len() != nx * ny {
            return Err(PdeError::ShapeMismatch {
                expected: vec![nx * ny],
                got: vec![u.len()],
            });
        }
        if nx < 3 || ny < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "ADI requires nx>=3 and ny>=3, got nx={nx} ny={ny}"
            )));
        }
        let hx = mesh.hx();
        let hy = mesh.hy();
        if hx <= 0.0 || hy <= 0.0 {
            return Err(PdeError::InvalidGrid(format!(
                "non-positive spacing hx={hx} hy={hy}"
            )));
        }
        let rx = self.alpha * self.dt / (hx * hx);
        let ry = self.alpha * self.dt / (hy * hy);

        // ── Half-step 1: implicit in x, explicit in y → u_star ──────────────
        // For each interior row index j, solve along i = 1..nx-1.
        let mut u_star = u.to_vec();
        let mx = nx - 2;
        for j in 1..ny - 1 {
            let mut sub = vec![-0.5 * rx; mx];
            let mut diag = vec![1.0 + rx; mx];
            let mut sup = vec![-0.5 * rx; mx];
            let mut rhs = vec![0.0; mx];
            for (k, rhs_k) in rhs.iter_mut().enumerate().take(mx) {
                let i = k + 1;
                let c = u[i * ny + j];
                let yp = u[i * ny + (j + 1)];
                let ym = u[i * ny + (j - 1)];
                // explicit y-Laplacian term
                *rhs_k = c + 0.5 * ry * (yp - 2.0 * c + ym);
            }
            // Dirichlet contributions from the x-boundaries (i=0 and i=nx-1),
            // which keep their values across the half-step.
            rhs[0] += 0.5 * rx * u[j];
            rhs[mx - 1] += 0.5 * rx * u[(nx - 1) * ny + j];
            sub[0] = 0.0;
            sup[mx - 1] = 0.0;
            let line = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
            for (k, &val) in line.iter().enumerate() {
                u_star[(k + 1) * ny + j] = val;
            }
        }

        // ── Half-step 2: implicit in y, explicit in x → u^{n+1} ─────────────
        let my = ny - 2;
        let mut u_next = u_star.clone();
        for i in 1..nx - 1 {
            let mut sub = vec![-0.5 * ry; my];
            let mut diag = vec![1.0 + ry; my];
            let mut sup = vec![-0.5 * ry; my];
            let mut rhs = vec![0.0; my];
            for (k, rhs_k) in rhs.iter_mut().enumerate().take(my) {
                let j = k + 1;
                let c = u_star[i * ny + j];
                let xp = u_star[(i + 1) * ny + j];
                let xm = u_star[(i - 1) * ny + j];
                *rhs_k = c + 0.5 * rx * (xp - 2.0 * c + xm);
            }
            // Dirichlet contributions from the y-boundaries (j=0 and j=ny-1).
            rhs[0] += 0.5 * ry * u_star[i * ny];
            rhs[my - 1] += 0.5 * ry * u_star[i * ny + (ny - 1)];
            sub[0] = 0.0;
            sup[my - 1] = 0.0;
            let line = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
            for (k, &val) in line.iter().enumerate() {
                u_next[i * ny + (k + 1)] = val;
            }
        }

        u.copy_from_slice(&u_next);
        Ok(())
    }

    /// Integrate `n_steps` ADI steps, returning the final field.
    pub fn solve(&self, mesh: &Mesh2d, u0: &[f64], n_steps: usize) -> PdeResult<Vec<f64>> {
        if u0.len() != mesh.n_nodes() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![mesh.n_nodes()],
                got: vec![u0.len()],
            });
        }
        let mut u = u0.to_vec();
        for _ in 0..n_steps {
            self.step(mesh, &mut u)?;
        }
        Ok(u)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Separable eigenmode sin(πx)·sin(πy), zero on the boundary of [0,1]².
    fn mode(mesh: &Mesh2d) -> Vec<f64> {
        let mut u = vec![0.0; mesh.n_nodes()];
        for i in 0..mesh.nx {
            for j in 0..mesh.ny {
                let x = mesh.x_nodes[i];
                let y = mesh.y_nodes[j];
                u[i * mesh.ny + j] = (PI * x).sin() * (PI * y).sin();
            }
        }
        u
    }

    fn max_abs(u: &[f64]) -> f64 {
        u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    #[test]
    fn constructor_rejects_bad_parameters() {
        assert!(CrankNicolson2d::new(0.0, 0.1).is_err());
        assert!(CrankNicolson2d::new(1.0, -0.1).is_err());
        assert!(CrankNicolson2d::new(1.0, 0.1).is_ok());
    }

    #[test]
    fn eigenmode_decays_at_correct_rate() {
        // sin(πx)sin(πy) decays like exp(−2π²αt) on [0,1]².
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 33, 33).expect("ok");
        let alpha = 1.0;
        let dt = 0.001;
        let adi = CrankNicolson2d::new(alpha, dt).expect("ok");
        let u0 = mode(&mesh);
        let t_final = 0.02;
        let n_steps = (t_final / dt).round() as usize;
        let u = adi.solve(&mesh, &u0, n_steps).expect("ok");
        let amp = (-2.0 * PI * PI * alpha * t_final).exp();
        let ci = mesh.nx / 2;
        let cj = mesh.ny / 2;
        let analytic = (PI * mesh.x_nodes[ci]).sin() * (PI * mesh.y_nodes[cj]).sin() * amp;
        let got = u[ci * mesh.ny + cj];
        assert!(
            (got - analytic).abs() < 2e-3,
            "got={got} analytic={analytic}"
        );
    }

    #[test]
    fn solution_amplitude_decreases_monotonically() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 21, 21).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.002).expect("ok");
        let mut u = mode(&mesh);
        let mut prev = max_abs(&u);
        for _ in 0..20 {
            adi.step(&mesh, &mut u).expect("ok");
            let cur = max_abs(&u);
            assert!(cur <= prev + 1e-12, "amplitude grew: {prev} -> {cur}");
            prev = cur;
        }
    }

    #[test]
    fn unconditionally_stable_large_dt() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 17, 17).expect("ok");
        let dt = 20.0 * mesh.hx() * mesh.hx();
        let adi = CrankNicolson2d::new(1.0, dt).expect("ok");
        let u0 = mode(&mesh);
        let u = adi.solve(&mesh, &u0, 40).expect("ok");
        assert!(u.iter().all(|v| v.is_finite()));
        assert!(max_abs(&u) < 0.1, "did not decay: {}", max_abs(&u));
    }

    #[test]
    fn boundary_values_preserved() {
        // Constant boundary 5.0 with constant interior 5.0 must remain 5.0.
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.01).expect("ok");
        let mut u = vec![5.0; mesh.n_nodes()];
        adi.step(&mesh, &mut u).expect("ok");
        for v in &u {
            assert!((v - 5.0).abs() < 1e-9, "v={v}");
        }
    }

    #[test]
    fn nonsquare_grid_works() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 25, 15).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.001).expect("ok");
        let u0 = mode(&mesh);
        let u = adi.solve(&mesh, &u0, 10).expect("ok");
        assert!(u.iter().all(|v| v.is_finite()));
        // Amplitude should have decreased.
        assert!(max_abs(&u) < max_abs(&u0));
    }

    #[test]
    fn symmetric_initial_stays_symmetric() {
        // The eigenmode is symmetric under (x,y)->(y,x) on a square grid;
        // ADI should preserve that symmetry to round-off.
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 21, 21).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.002).expect("ok");
        let mut u = mode(&mesh);
        for _ in 0..15 {
            adi.step(&mesh, &mut u).expect("ok");
        }
        for i in 0..mesh.nx {
            for j in 0..mesh.ny {
                let a = u[i * mesh.ny + j];
                let b = u[j * mesh.ny + i];
                assert!((a - b).abs() < 1e-9, "asymmetry at ({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n_nodes() - 1];
        assert!(adi.step(&mesh, &mut u).is_err());
    }

    #[test]
    fn too_small_grid_errors() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 2, 5).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n_nodes()];
        assert!(adi.step(&mesh, &mut u).is_err());
    }

    #[test]
    fn solve_shape_mismatch_errors() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.01).expect("ok");
        let u0 = vec![0.0; 5];
        assert!(adi.solve(&mesh, &u0, 1).is_err());
    }

    #[test]
    fn zero_steps_returns_initial() {
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 11, 11).expect("ok");
        let adi = CrankNicolson2d::new(1.0, 0.01).expect("ok");
        let u0 = mode(&mesh);
        let u = adi.solve(&mesh, &u0, 0).expect("ok");
        assert_eq!(u, u0);
    }

    #[test]
    fn second_order_in_time() {
        // Self-convergence: isolate the temporal error with a tiny-dt reference
        // on a fixed grid; halving dt should shrink it by ~4× (ADI is O(dt²)).
        let mesh = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 25, 25).expect("ok");
        let alpha = 1.0;
        let t_final = 0.02;
        let ci = mesh.nx / 2;
        let cj = mesh.ny / 2;
        let solve = |dt: f64| {
            let adi = CrankNicolson2d::new(alpha, dt).expect("ok");
            let u0 = mode(&mesh);
            let n = (t_final / dt).round() as usize;
            let u = adi.solve(&mesh, &u0, n).expect("ok");
            u[ci * mesh.ny + cj]
        };
        let reference = solve(t_final / 1024.0);
        let e_coarse = (solve(t_final / 8.0) - reference).abs();
        let e_fine = (solve(t_final / 16.0) - reference).abs();
        let ratio = e_coarse / e_fine.max(1e-15);
        assert!(ratio > 3.0, "second-order ratio too low: {ratio}");
    }
}
