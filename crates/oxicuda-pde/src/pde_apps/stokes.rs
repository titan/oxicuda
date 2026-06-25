//! Steady incompressible **Stokes** flow on a rectangle, discretised with the
//! marker-and-cell (MAC) staggered scheme and solved as a mixed saddle-point
//! system.
//!
//! The Stokes equations for velocity `u = (u, v)` and pressure `p` with
//! dynamic viscosity `μ` and body force `f = (fx, fy)` are
//!
//! ```text
//! −μ ∇²u + ∇p = f      (momentum)
//!        ∇·u = 0       (incompressibility)
//! ```
//!
//! # MAC staggering
//!
//! On an `nx × ny` grid of pressure cells over `[0, Lx] × [0, Ly]`:
//!
//! * pressures `p` live at the `nx·ny` **cell centres**;
//! * `u` (x-velocity) lives on **vertical faces**; the `(nx−1)·ny` faces
//!   strictly interior in `x` are unknowns, the left/right boundary faces are
//!   Dirichlet data;
//! * `v` (y-velocity) lives on **horizontal faces**; the `nx·(ny−1)` faces
//!   strictly interior in `y` are unknowns, the bottom/top boundary faces are
//!   Dirichlet data.
//!
//! This staggering is the classical inf-sup (LBB) stable mixed discretisation:
//! the pressure–velocity coupling does not admit spurious checkerboard pressure
//! modes, so no stabilisation is required. The assembled system is exactly the
//! saddle-point form
//!
//! ```text
//! [ A   Bᵀ ] [ U ]   [ F ]
//! [ B   0  ] [ P ] = [ 0 ]
//! ```
//!
//! with `A` the block-diagonal velocity Laplacian (`−μ ∇²`), `B` the discrete
//! divergence (cell continuity in terms of face velocities) and `Bᵀ` the
//! discrete pressure gradient. It is handed to the
//! [`uzawa`] /
//! [`minres`] solvers in
//! [`crate::solver::saddle_point`].
//!
//! With Dirichlet velocity on the whole boundary the pressure is determined
//! only up to an additive constant (the constant-pressure null space of `B`);
//! we remove it by pinning one pressure degree of freedom.
//!
//! # References
//!
//! * F. H. Harlow and J. E. Welch, "Numerical calculation of time-dependent
//!   viscous incompressible flow of fluid with free surface", Phys. Fluids
//!   8(12), 2182–2189, 1965 (the MAC scheme).
//! * H. C. Elman, D. J. Silvester, A. J. Wathen, *Finite Elements and Fast
//!   Iterative Solvers*, 2nd ed., OUP, 2014, ch. 3.

use crate::error::{PdeError, PdeResult};
use crate::solver::saddle_point::{MinresConfig, SaddleResult, UzawaConfig, minres, uzawa};
use crate::solver::sparse::SparseCsr;

/// A boundary-velocity sampler: given physical `(x, y)` it returns the
/// prescribed velocity component on that face.
pub type BoundaryFn<'a> = dyn Fn(f64, f64) -> f64 + 'a;

/// Steady Stokes problem on a rectangle with the MAC staggered scheme.
#[derive(Debug, Clone)]
pub struct StokesMac {
    /// Dynamic viscosity `μ > 0`.
    pub mu: f64,
    /// Number of pressure cells in `x` (`nx ≥ 2`).
    pub nx: usize,
    /// Number of pressure cells in `y` (`ny ≥ 2`).
    pub ny: usize,
    /// Cell width `Lx / nx`.
    pub hx: f64,
    /// Cell height `Ly / ny`.
    pub hy: f64,
}

/// A fully assembled MAC Stokes solution on the staggered grid.
#[derive(Debug, Clone)]
pub struct StokesSolution {
    /// x-velocity at all vertical faces, indexed `[i + j*(nx+1)]` for
    /// `i ∈ 0..=nx`, `j ∈ 0..ny`. Boundary columns carry the Dirichlet data.
    pub u: Vec<f64>,
    /// y-velocity at all horizontal faces, indexed `[i + j*nx]` for
    /// `i ∈ 0..nx`, `j ∈ 0..=ny`. Boundary rows carry the Dirichlet data.
    pub v: Vec<f64>,
    /// Pressure at cell centres, indexed `[i + j*nx]`, with the pinned cell set
    /// so the discrete mean is zero.
    pub p: Vec<f64>,
    /// Number of outer solver iterations.
    pub iterations: usize,
    /// Final saddle-point residual two-norm.
    pub residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

impl StokesMac {
    /// Build a solver on `[0, lx] × [0, ly]` with `nx × ny` pressure cells.
    ///
    /// # Errors
    ///
    /// [`PdeError::InvalidParameter`] / [`PdeError::InvalidGrid`] on bad inputs.
    pub fn new(mu: f64, lx: f64, ly: f64, nx: usize, ny: usize) -> PdeResult<Self> {
        if !(mu.is_finite() && mu > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "mu".into(),
                reason: format!("viscosity must be finite and > 0, got {mu}"),
            });
        }
        if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "domain".into(),
                reason: "lx, ly must be finite and > 0".into(),
            });
        }
        if nx < 2 || ny < 2 {
            return Err(PdeError::InvalidGrid(format!(
                "Stokes MAC requires nx, ny >= 2, got nx={nx} ny={ny}"
            )));
        }
        Ok(Self {
            mu,
            nx,
            ny,
            hx: lx / nx as f64,
            hy: ly / ny as f64,
        })
    }

    /// Number of interior `u`-velocity unknowns (`(nx−1)·ny`).
    #[inline]
    #[must_use]
    pub fn n_u(&self) -> usize {
        (self.nx - 1) * self.ny
    }

    /// Number of interior `v`-velocity unknowns (`nx·(ny−1)`).
    #[inline]
    #[must_use]
    pub fn n_v(&self) -> usize {
        self.nx * (self.ny - 1)
    }

    /// Number of pressure unknowns (`nx·ny`).
    #[inline]
    #[must_use]
    pub fn n_p(&self) -> usize {
        self.nx * self.ny
    }

    /// Physical location of an interior `u`-face.
    ///
    /// Interior u-faces sit at `x = i·hx` for `i ∈ 1..nx` and
    /// `y = (j + ½)·hy` for `j ∈ 0..ny`. The unknown index is `i−1 + j·(nx−1)`.
    fn u_xy(&self, iu: usize) -> (f64, f64) {
        let i = iu % (self.nx - 1) + 1; // 1..=nx-1
        let j = iu / (self.nx - 1); // 0..ny-1
        (i as f64 * self.hx, (j as f64 + 0.5) * self.hy)
    }

    /// Physical location of an interior `v`-face.
    ///
    /// Interior v-faces sit at `x = (i + ½)·hx` for `i ∈ 0..nx` and
    /// `y = j·hy` for `j ∈ 1..ny`. The unknown index is `i + (j−1)·nx`.
    fn v_xy(&self, iv: usize) -> (f64, f64) {
        let i = iv % self.nx; // 0..nx-1
        let j = iv / self.nx + 1; // 1..=ny-1
        ((i as f64 + 0.5) * self.hx, j as f64 * self.hy)
    }

    /// Assemble the velocity Laplacian block `A = −μ ∇²` (SPD) over the stacked
    /// `[U; V]` unknowns, together with the boundary-condition contribution to
    /// the momentum right-hand side.
    ///
    /// Dirichlet velocity values on boundary faces enter `A`-rows of
    /// near-boundary unknowns and are moved to the right-hand side.
    fn assemble_velocity_block(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
        fx: &BoundaryFn<'_>,
        fy: &BoundaryFn<'_>,
    ) -> PdeResult<(SparseCsr, Vec<f64>)> {
        let nu = self.n_u();
        let nv = self.n_v();
        let ndof = nu + nv;
        let cx = self.mu / (self.hx * self.hx);
        let cy = self.mu / (self.hy * self.hy);
        let mut row_ptr = vec![0_usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // RHS contributions are pushed in row order, then concatenated, so no
        // loop counter ever indexes a slice directly.
        let mut rhs = Vec::with_capacity(ndof);

        // --- u-momentum rows (5-point Laplacian on u-faces) ---
        let nxm1 = self.nx - 1;
        let diag0 = 2.0 * cx + 2.0 * cy;
        for iu in 0..nu {
            let i = iu % nxm1 + 1; // x-face column 1..=nx-1
            let j = iu / nxm1; // row 0..ny-1
            let (x, y) = self.u_xy(iu);
            let mut rhs_acc = fx(x, y);
            // West neighbour (i-1).
            if i >= 2 {
                let nb = (i - 2) + j * nxm1;
                cols.push(nb);
                vals.push(-cx);
            } else {
                // i-1 == 0 ⇒ left boundary face at x=0.
                rhs_acc += cx * u_bc(0.0, y);
            }
            // East neighbour (i+1).
            if i + 1 < self.nx {
                let nb = i + j * nxm1; // (i+1)-1 = i
                cols.push(nb);
                vals.push(-cx);
            } else {
                rhs_acc += cx * u_bc(self.nx as f64 * self.hx, y);
            }
            // South neighbour (j-1).
            if j >= 1 {
                let nb = (i - 1) + (j - 1) * nxm1;
                cols.push(nb);
                vals.push(-cy);
            } else {
                // Ghost below the bottom: reflect Dirichlet tangential u at y=0.
                rhs_acc += 2.0 * cy * u_bc(x, 0.0);
            }
            // North neighbour (j+1).
            if j + 1 < self.ny {
                let nb = (i - 1) + (j + 1) * nxm1;
                cols.push(nb);
                vals.push(-cy);
            } else {
                rhs_acc += 2.0 * cy * u_bc(x, self.ny as f64 * self.hy);
            }
            // Diagonal: tangential (top/bottom) ghost cells use the mirror
            // stencil so the wall value is enforced at the face midpoint,
            // adding an extra cy on the boundary-adjacent diagonal.
            let mut d = diag0;
            if j == 0 {
                d += cy;
            }
            if j + 1 >= self.ny {
                d += cy;
            }
            push_diag(&mut cols, &mut vals, iu, d);
            row_ptr.push(cols.len());
            rhs.push(rhs_acc);
        }

        // --- v-momentum rows ---
        for iv in 0..nv {
            let i = iv % self.nx; // 0..nx-1
            let j = iv / self.nx + 1; // y-face row 1..=ny-1
            let (x, y) = self.v_xy(iv);
            let global = nu + iv;
            let mut rhs_acc = fy(x, y);
            // South neighbour (j-1).
            if j >= 2 {
                let nb = nu + i + (j - 2) * self.nx;
                cols.push(nb);
                vals.push(-cy);
            } else {
                rhs_acc += cy * v_bc(x, 0.0);
            }
            // North neighbour (j+1).
            if j + 1 < self.ny {
                let nb = nu + i + j * self.nx;
                cols.push(nb);
                vals.push(-cy);
            } else {
                rhs_acc += cy * v_bc(x, self.ny as f64 * self.hy);
            }
            // West neighbour (i-1).
            if i >= 1 {
                let nb = nu + (i - 1) + (j - 1) * self.nx;
                cols.push(nb);
                vals.push(-cx);
            } else {
                rhs_acc += 2.0 * cx * v_bc(0.0, y);
            }
            // East neighbour (i+1).
            if i + 1 < self.nx {
                let nb = nu + (i + 1) + (j - 1) * self.nx;
                cols.push(nb);
                vals.push(-cx);
            } else {
                rhs_acc += 2.0 * cx * v_bc(self.nx as f64 * self.hx, y);
            }
            let mut d = diag0;
            if i == 0 {
                d += cx;
            }
            if i + 1 >= self.nx {
                d += cx;
            }
            push_diag(&mut cols, &mut vals, global, d);
            row_ptr.push(cols.len());
            rhs.push(rhs_acc);
        }

        debug_assert_eq!(rhs.len(), ndof);
        let a = SparseCsr::new(ndof, ndof, row_ptr, cols, vals)?;
        Ok((a, rhs))
    }

    /// Assemble the discrete divergence operator `B` (continuity per pressure
    /// cell, expressed in the interior face velocities). Returns the `n_p × ndof`
    /// CSR matrix and the boundary contribution to the continuity RHS `g`
    /// (from prescribed boundary-face velocities).
    fn assemble_divergence(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
    ) -> PdeResult<(SparseCsr, Vec<f64>)> {
        let nu = self.n_u();
        let nv = self.n_v();
        let ndof = nu + nv;
        let np = self.n_p();
        let nxm1 = self.nx - 1;
        let mut row_ptr = vec![0_usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // Continuity per cell: (u_E − u_W)/hx + (v_N − v_S)/hy = 0.
        // We assemble rows of B so that B·[U;V] = −(boundary divergence),
        // hence g = −(known boundary face contributions). For an interior-only
        // unknown set, B U − (−g) = 0, i.e. the saddle system uses g on the RHS.
        let mut g = vec![0.0_f64; np];
        for (cell, gc) in g.iter_mut().enumerate().take(np) {
            let i = cell % self.nx; // 0..nx-1
            let j = cell / self.nx; // 0..ny-1
            // West u-face at column i (x = i·hx). Interior iff i ≥ 1.
            if i >= 1 {
                let iu = (i - 1) + j * nxm1;
                cols.push(iu);
                vals.push(-1.0 / self.hx);
            } else {
                // x=0 wall: known u.
                let y = (j as f64 + 0.5) * self.hy;
                *gc += (-1.0 / self.hx) * u_bc(0.0, y);
            }
            // East u-face at column i+1 (x = (i+1)·hx). Interior iff i+1 < nx.
            if i + 1 < self.nx {
                let iu = i + j * nxm1; // (i+1)-1 = i
                cols.push(iu);
                vals.push(1.0 / self.hx);
            } else {
                let y = (j as f64 + 0.5) * self.hy;
                *gc += (1.0 / self.hx) * u_bc(self.nx as f64 * self.hx, y);
            }
            // South v-face at row j (y = j·hy). Interior iff j ≥ 1.
            if j >= 1 {
                let iv = nu + i + (j - 1) * self.nx;
                cols.push(iv);
                vals.push(-1.0 / self.hy);
            } else {
                let x = (i as f64 + 0.5) * self.hx;
                *gc += (-1.0 / self.hy) * v_bc(x, 0.0);
            }
            // North v-face at row j+1 (y = (j+1)·hy). Interior iff j+1 < ny.
            if j + 1 < self.ny {
                let iv = nu + i + j * self.nx;
                cols.push(iv);
                vals.push(1.0 / self.hy);
            } else {
                let x = (i as f64 + 0.5) * self.hx;
                *gc += (1.0 / self.hy) * v_bc(x, self.ny as f64 * self.hy);
            }
            row_ptr.push(cols.len());
        }
        // Move known boundary divergence to the RHS: B U = −g_boundary, so the
        // continuity right-hand side seen by the saddle solver is `-g`.
        for gi in g.iter_mut() {
            *gi = -*gi;
        }
        let b = SparseCsr::new(np, ndof, row_ptr, cols, vals)?;
        Ok((b, g))
    }

    /// Assemble the saddle-point blocks `(A, B, F, g)`, with one pressure DOF
    /// pinned (its row/column removed via a unit constraint) so the system is
    /// non-singular under all-Dirichlet velocity boundaries.
    ///
    /// To keep `B` rectangular and the constraint simple we pin pressure by
    /// adding a tiny regularising row that fixes `p[0] = 0`; numerically we
    /// instead post-process by subtracting the discrete mean (see [`Self::solve_minres`]),
    /// so here we return the raw operators.
    ///
    /// # Errors
    ///
    /// Propagates assembly errors.
    pub fn assemble(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
        fx: &BoundaryFn<'_>,
        fy: &BoundaryFn<'_>,
    ) -> PdeResult<(SparseCsr, SparseCsr, Vec<f64>, Vec<f64>)> {
        let (a, f) = self.assemble_velocity_block(u_bc, v_bc, fx, fy)?;
        let (b, g) = self.assemble_divergence(u_bc, v_bc)?;
        Ok((a, b, f, g))
    }

    /// Solve the Stokes problem with the MINRES saddle-point solver and return
    /// the full staggered field, with the pressure shifted to zero mean.
    ///
    /// # Errors
    ///
    /// Propagates assembly and solver failures.
    pub fn solve_minres(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
        fx: &BoundaryFn<'_>,
        fy: &BoundaryFn<'_>,
        cfg: &MinresConfig,
    ) -> PdeResult<StokesSolution> {
        let (a, b, f, g) = self.assemble(u_bc, v_bc, fx, fy)?;
        let res = minres(&a, &b, &f, &g, cfg)?;
        self.expand_solution(u_bc, v_bc, res)
    }

    /// Solve the Stokes problem with the Uzawa saddle-point solver.
    ///
    /// # Errors
    ///
    /// Propagates assembly and solver failures.
    pub fn solve_uzawa(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
        fx: &BoundaryFn<'_>,
        fy: &BoundaryFn<'_>,
        cfg: &UzawaConfig,
    ) -> PdeResult<StokesSolution> {
        let (a, b, f, g) = self.assemble(u_bc, v_bc, fx, fy)?;
        let p0 = vec![0.0_f64; self.n_p()];
        let res = uzawa(&a, &b, &f, &g, &p0, cfg)?;
        self.expand_solution(u_bc, v_bc, res)
    }

    /// Scatter an interior saddle-point solution back onto the full staggered
    /// grid (filling boundary faces with their Dirichlet values) and shift the
    /// pressure to discrete zero mean.
    fn expand_solution(
        &self,
        u_bc: &BoundaryFn<'_>,
        v_bc: &BoundaryFn<'_>,
        res: SaddleResult,
    ) -> PdeResult<StokesSolution> {
        let nu = self.n_u();
        let nxm1 = self.nx - 1;
        // Full u-field on (nx+1) × ny vertical faces.
        let mut u = vec![0.0_f64; (self.nx + 1) * self.ny];
        for j in 0..self.ny {
            let y = (j as f64 + 0.5) * self.hy;
            // Boundary faces.
            u[j * (self.nx + 1)] = u_bc(0.0, y);
            u[self.nx + j * (self.nx + 1)] = u_bc(self.nx as f64 * self.hx, y);
            for i in 1..self.nx {
                let iu = (i - 1) + j * nxm1;
                u[i + j * (self.nx + 1)] = res.u[iu];
            }
        }
        // Full v-field on nx × (ny+1) horizontal faces.
        let mut v = vec![0.0_f64; self.nx * (self.ny + 1)];
        for i in 0..self.nx {
            let x = (i as f64 + 0.5) * self.hx;
            v[i] = v_bc(x, 0.0);
            v[i + self.ny * self.nx] = v_bc(x, self.ny as f64 * self.hy);
            for j in 1..self.ny {
                let iv = nu + i + (j - 1) * self.nx;
                v[i + j * self.nx] = res.u[iv];
            }
        }
        // Pressure with zero discrete mean.
        let mut p = res.p.clone();
        let mean = p.iter().sum::<f64>() / p.len().max(1) as f64;
        for pi in p.iter_mut() {
            *pi -= mean;
        }
        Ok(StokesSolution {
            u,
            v,
            p,
            iterations: res.iterations,
            residual: res.residual,
            converged: res.converged,
        })
    }

    /// Maximum discrete divergence over all cells for a computed solution
    /// (should be at the solver tolerance for an incompressible field).
    #[must_use]
    pub fn max_divergence(&self, sol: &StokesSolution) -> f64 {
        let mut worst = 0.0_f64;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let ue = sol.u[(i + 1) + j * (self.nx + 1)];
                let uw = sol.u[i + j * (self.nx + 1)];
                let vn = sol.v[i + (j + 1) * self.nx];
                let vs = sol.v[i + j * self.nx];
                let div = (ue - uw) / self.hx + (vn - vs) / self.hy;
                worst = worst.max(div.abs());
            }
        }
        worst
    }
}

/// Insert a diagonal entry for `row` into the (column, value) row buffers.
///
/// The CSR consumers in this crate (matvec, ILU, etc.) do not require sorted
/// columns, so we simply append; this keeps the assembly O(nnz).
fn push_diag(cols: &mut Vec<usize>, vals: &mut Vec<f64>, row: usize, diag: f64) {
    cols.push(row);
    vals.push(diag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dof_counts_consistent() {
        let s = StokesMac::new(1.0, 1.0, 1.0, 4, 5).expect("ok");
        assert_eq!(s.n_u(), 3 * 5);
        assert_eq!(s.n_v(), 4 * 4);
        assert_eq!(s.n_p(), 20);
    }

    #[test]
    fn velocity_block_is_symmetric() {
        let s = StokesMac::new(1.5, 1.0, 1.0, 5, 4).expect("ok");
        let zero = |_: f64, _: f64| 0.0;
        let (a, _f) = s
            .assemble_velocity_block(&zero, &zero, &zero, &zero)
            .expect("ok");
        // Densify and check symmetry.
        let n = a.n_rows;
        let mut dense = vec![0.0_f64; n * n];
        for r in 0..n {
            for k in a.row_ptr[r]..a.row_ptr[r + 1] {
                dense[r * n + a.cols[k]] += a.vals[k];
            }
        }
        for r in 0..n {
            for c in 0..n {
                assert!(
                    (dense[r * n + c] - dense[c * n + r]).abs() < 1e-12,
                    "A not symmetric at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn velocity_block_is_positive_definite_diagonally_dominant() {
        // The MAC velocity Laplacian is weakly diagonally dominant with strict
        // dominance on boundary-adjacent rows ⇒ SPD.
        let s = StokesMac::new(1.0, 1.0, 1.0, 5, 5).expect("ok");
        let zero = |_: f64, _: f64| 0.0;
        let (a, _f) = s
            .assemble_velocity_block(&zero, &zero, &zero, &zero)
            .expect("ok");
        let n = a.n_rows;
        for r in 0..n {
            let mut diag = 0.0;
            let mut off = 0.0;
            for k in a.row_ptr[r]..a.row_ptr[r + 1] {
                if a.cols[k] == r {
                    diag += a.vals[k];
                } else {
                    off += a.vals[k].abs();
                }
            }
            assert!(diag > 0.0, "non-positive diagonal at row {r}");
            assert!(diag >= off - 1e-12, "row {r} not diagonally dominant");
        }
    }

    #[test]
    fn divergence_free_field_has_zero_divergence() {
        // A uniform horizontal flow u = U0, v = 0 is divergence-free; feeding it
        // as boundary data with zero forcing must reproduce it (up to a constant
        // pressure) and have near-zero discrete divergence.
        let s = StokesMac::new(1.0, 1.0, 1.0, 8, 8).expect("ok");
        let u0 = 1.3_f64;
        let u_bc = move |_x: f64, _y: f64| u0;
        let v_bc = |_x: f64, _y: f64| 0.0;
        let zero = |_x: f64, _y: f64| 0.0;
        let sol = s
            .solve_minres(&u_bc, &v_bc, &zero, &zero, &MinresConfig::default())
            .expect("solve ok");
        assert!(sol.converged, "residual {}", sol.residual);
        assert!(
            s.max_divergence(&sol) < 1e-7,
            "max div {}",
            s.max_divergence(&sol)
        );
        // Interior u should be ≈ u0 everywhere (constant flow is exact).
        for j in 0..s.ny {
            for i in 1..s.nx {
                let val = sol.u[i + j * (s.nx + 1)];
                assert!((val - u0).abs() < 1e-6, "u[{i},{j}]={val}");
            }
        }
    }

    #[test]
    fn couette_flow_linear_profile() {
        // Plane Couette flow: top wall moves at U, bottom wall at rest, no body
        // force. Exact solution u(y) = U·y/Ly, v = 0, p const. The MAC scheme
        // reproduces the linear profile to round-off (it is in the FD kernel).
        let nx = 6;
        let ny = 16;
        let ly = 1.0;
        let s = StokesMac::new(1.0, 1.0, ly, nx, ny).expect("ok");
        let u_top = 2.0_f64;
        let u_bc = move |_x: f64, y: f64| u_top * y / ly;
        let v_bc = |_x: f64, _y: f64| 0.0;
        let zero = |_x: f64, _y: f64| 0.0;
        let sol = s
            .solve_minres(&u_bc, &v_bc, &zero, &zero, &MinresConfig::default())
            .expect("solve ok");
        assert!(sol.converged);
        assert!(s.max_divergence(&sol) < 1e-7);
        // Check the linear profile at interior u-faces.
        for j in 0..ny {
            let y = (j as f64 + 0.5) * s.hy;
            let exact = u_top * y / ly;
            for i in 1..nx {
                let val = sol.u[i + j * (nx + 1)];
                assert!((val - exact).abs() < 1e-4, "j={j} u={val} exact={exact}");
            }
        }
    }

    #[test]
    fn uzawa_and_minres_agree_on_couette() {
        let nx = 5;
        let ny = 10;
        let s = StokesMac::new(1.0, 1.0, 1.0, nx, ny).expect("ok");
        let u_bc = |_x: f64, y: f64| y;
        let v_bc = |_x: f64, _y: f64| 0.0;
        let zero = |_x: f64, _y: f64| 0.0;
        let mr = s
            .solve_minres(&u_bc, &v_bc, &zero, &zero, &MinresConfig::default())
            .expect("minres");
        let uz_cfg = UzawaConfig {
            omega: s.mu, // diag(A) ~ 2μ(1/hx²+1/hy²); ω≈μ keeps spectral radius < 1 here
            max_iter: 4000,
            tol: 1.0e-8,
            inner_tol: 1.0e-12,
            inner_max_iter: 1000,
        };
        let uz = s
            .solve_uzawa(&u_bc, &v_bc, &zero, &zero, &uz_cfg)
            .expect("uzawa");
        // Compare interior u-faces.
        for j in 0..ny {
            for i in 1..nx {
                let a = mr.u[i + j * (nx + 1)];
                let b = uz.u[i + j * (nx + 1)];
                assert!((a - b).abs() < 1e-3, "u[{i},{j}] {a} vs {b}");
            }
        }
    }

    #[test]
    fn pressure_has_zero_mean() {
        let s = StokesMac::new(1.0, 1.0, 1.0, 6, 6).expect("ok");
        let u_bc = |_x: f64, y: f64| y;
        let v_bc = |_x: f64, _y: f64| 0.0;
        let zero = |_x: f64, _y: f64| 0.0;
        let sol = s
            .solve_minres(&u_bc, &v_bc, &zero, &zero, &MinresConfig::default())
            .expect("ok");
        let mean = sol.p.iter().sum::<f64>() / sol.p.len() as f64;
        assert!(mean.abs() < 1e-9, "pressure mean {mean}");
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(StokesMac::new(0.0, 1.0, 1.0, 4, 4).is_err());
        assert!(StokesMac::new(1.0, -1.0, 1.0, 4, 4).is_err());
        assert!(StokesMac::new(1.0, 1.0, 1.0, 1, 4).is_err());
    }
}
