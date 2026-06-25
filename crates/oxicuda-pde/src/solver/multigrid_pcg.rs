//! Geometric-multigrid-preconditioned conjugate gradient (MG-PCG) for the
//! 1-D Poisson operator on a uniform grid.
//!
//! This complements [`crate::solver::preconditioner_amg`] (which wraps the
//! *algebraic* multigrid hierarchy) with a *geometric* V-cycle preconditioner
//! built directly from the structured-grid restriction / prolongation
//! operators in [`crate::multigrid`]. For a structured Poisson problem the
//! geometric V-cycle is cheaper to set up (no aggregation) and gives the
//! textbook mesh-independent convergence: the number of PCG iterations stays
//! essentially constant as the grid is refined.
//!
//! # Why the V-cycle is a valid PCG preconditioner
//!
//! A V-cycle started from a **zero** initial guess and applied to a vector `r`
//! is a fixed linear operator `M⁻¹ r`. With
//!
//! * weighted-Jacobi pre- and post-smoothing applied the same number of times,
//! * full-weighting restriction `R` and linear prolongation `P = c·Rᵀ`, and
//! * an exact one-unknown coarse solve,
//!
//! `M⁻¹` is symmetric positive definite, so it is admissible inside the
//! preconditioned conjugate-gradient recurrence. We verify this symmetry
//! numerically in the unit tests.
//!
//! # Operator convention
//!
//! The system matrix is the interior 1-D Dirichlet Laplacian scaled as
//! `A = (1/h²)·tridiag(−1, 2, −1)` of size `m = n_grid − 2`, matching the
//! V-cycle smoother which solves `(1/h²)(2u_i − u_{i−1} − u_{i+1}) = f_i`.
//! Applying the preconditioner to a residual `r` therefore amounts to running
//! the V-cycle with right-hand side `r` on the full grid (interior loaded with
//! `r`, boundaries zero) and reading back the interior correction.
//!
//! # References
//!
//! * W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial*,
//!   2nd ed., SIAM, 2000, ch. 4–5.
//! * Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM,
//!   2003, §13.

use crate::error::{PdeError, PdeResult};
use crate::multigrid::vcycle::v_cycle_1d;
use crate::solver::preconditioner_amg::pcg_generic;
use crate::solver::sparse::SparseCsr;

/// Configuration for the 1-D geometric MG-PCG solver.
#[derive(Debug, Clone)]
pub struct MgPcgConfig {
    /// Maximum number of outer PCG iterations.
    pub max_iter: usize,
    /// Relative residual tolerance `‖b − A x‖ / ‖b‖`.
    pub tol: f64,
    /// Number of weighted-Jacobi pre-smoothing sweeps per level.
    pub n_pre: usize,
    /// Number of weighted-Jacobi post-smoothing sweeps per level.
    pub n_post: usize,
    /// Number of V-cycles per preconditioner application (1 is standard).
    pub n_cycles: usize,
}

impl Default for MgPcgConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1.0e-9,
            n_pre: 2,
            n_post: 2,
            n_cycles: 1,
        }
    }
}

/// Result of a geometric MG-PCG solve.
#[derive(Debug, Clone)]
pub struct MgPcgResult {
    /// Approximate solution on the interior unknowns (length `n_grid − 2`).
    pub x: Vec<f64>,
    /// Number of PCG iterations performed.
    pub iterations: usize,
    /// Final residual two-norm `‖b − A x‖`.
    pub residual: f64,
    /// Whether the relative tolerance was reached.
    pub converged: bool,
}

/// A reusable geometric-multigrid preconditioner for the 1-D Poisson operator.
///
/// Holds the mesh spacing and the full-grid size; each [`Self::apply`] runs the
/// configured number of V-cycles from a zero start.
#[derive(Debug, Clone)]
pub struct GeometricMgPreconditioner {
    /// Number of interior unknowns.
    m: usize,
    /// Full-grid node count (`m + 2`, must be odd).
    n_grid: usize,
    /// Uniform mesh spacing.
    h: f64,
    n_pre: usize,
    n_post: usize,
    n_cycles: usize,
}

impl GeometricMgPreconditioner {
    /// Build the preconditioner for `m` interior unknowns with spacing `h`.
    ///
    /// The full grid has `m + 2` nodes and must have an odd node count for the
    /// structured coarsening to bisect cleanly, i.e. `m` must be **odd**.
    ///
    /// # Errors
    ///
    /// * [`PdeError::InvalidParameter`] when `m == 0`, `h ≤ 0`, or `m` is even.
    pub fn new(m: usize, h: f64, cfg: &MgPcgConfig) -> PdeResult<Self> {
        if m == 0 {
            return Err(PdeError::InvalidParameter {
                name: "m".into(),
                reason: "must be ≥ 1".into(),
            });
        }
        if !(h > 0.0 && h.is_finite()) {
            return Err(PdeError::InvalidParameter {
                name: "h".into(),
                reason: "must be positive and finite".into(),
            });
        }
        let n_grid = m + 2;
        if n_grid % 2 == 0 {
            return Err(PdeError::InvalidParameter {
                name: "m".into(),
                reason: "m+2 (full grid) must be odd for geometric coarsening".into(),
            });
        }
        Ok(Self {
            m,
            n_grid,
            h,
            n_pre: cfg.n_pre,
            n_post: cfg.n_post,
            n_cycles: cfg.n_cycles.max(1),
        })
    }

    /// Number of interior unknowns this preconditioner acts on.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Apply the preconditioner: `z = M⁻¹ r` via geometric V-cycles.
    ///
    /// # Errors
    ///
    /// [`PdeError::DimensionMismatch`] when `r.len() != dim()`.
    pub fn apply(&self, r: &[f64]) -> PdeResult<Vec<f64>> {
        if r.len() != self.m {
            return Err(PdeError::DimensionMismatch {
                a: r.len(),
                b: self.m,
            });
        }
        // Embed the residual into a full grid with zero boundaries.
        let mut u = vec![0.0_f64; self.n_grid];
        let mut f = vec![0.0_f64; self.n_grid];
        f[1..=self.m].copy_from_slice(r);
        for _ in 0..self.n_cycles {
            v_cycle_1d(&mut u, &f, self.h, self.n_pre, self.n_post)?;
        }
        Ok(u[1..=self.m].to_vec())
    }
}

/// Build the interior 1-D Dirichlet Laplacian `(1/h²)·tridiag(−1, 2, −1)` as
/// CSR, of size `m × m`.
///
/// This is the canonical SPD matrix whose preconditioner is the geometric
/// V-cycle; exposed so callers can solve the same operator with [`mg_pcg`].
///
/// # Errors
///
/// [`PdeError::InvalidParameter`] when `m == 0` or `h ≤ 0`.
pub fn poisson_1d_interior_csr(m: usize, h: f64) -> PdeResult<SparseCsr> {
    if m == 0 {
        return Err(PdeError::InvalidParameter {
            name: "m".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if !(h > 0.0 && h.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: "must be positive and finite".into(),
        });
    }
    let ih2 = 1.0 / (h * h);
    let mut row_ptr = vec![0_usize];
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        if i > 0 {
            cols.push(i - 1);
            vals.push(-ih2);
        }
        cols.push(i);
        vals.push(2.0 * ih2);
        if i + 1 < m {
            cols.push(i + 1);
            vals.push(-ih2);
        }
        row_ptr.push(cols.len());
    }
    SparseCsr::new(m, m, row_ptr, cols, vals)
}

/// Solve the SPD 1-D Poisson system `A x = b` with geometric-multigrid PCG.
///
/// `A` must be the interior 1-D Laplacian of size `m × m` for some uniform
/// spacing `h` (see [`poisson_1d_interior_csr`]); `x0` is the initial guess
/// (zeros if unsure).
///
/// # Errors
///
/// * [`PdeError::DimensionMismatch`] when `A` is not square or the vectors are
///   inconsistent with `A`.
/// * [`PdeError::InvalidParameter`] when `h` is invalid or the grid size is
///   incompatible with geometric coarsening.
/// * [`PdeError::NumericalInstability`] on a non-SPD breakdown.
pub fn mg_pcg(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    h: f64,
    cfg: &MgPcgConfig,
) -> PdeResult<MgPcgResult> {
    if a.n_rows != a.n_cols {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    let m = a.n_rows;
    if b.len() != m || x0.len() != m {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: m });
    }
    let precond = GeometricMgPreconditioner::new(m, h, cfg)?;
    let (x, iterations, residual, converged) =
        pcg_generic(a, b, x0, |r| precond.apply(r), cfg.max_iter, cfg.tol)?;
    Ok(MgPcgResult {
        x,
        iterations,
        residual,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::sparse::{dot, norm2};

    fn grid_spacing(m: usize) -> f64 {
        1.0 / (m as f64 + 1.0)
    }

    #[test]
    fn solves_constant_rhs_to_tolerance() {
        // -u'' = 2 on (0,1), Dirichlet 0 ⇒ u(x) = x(1−x).
        let m = 31; // grid = 33 (odd)
        let h = grid_spacing(m);
        let a = poisson_1d_interior_csr(m, h).expect("csr");
        let b = vec![2.0_f64; m];
        let cfg = MgPcgConfig {
            tol: 1.0e-10,
            ..MgPcgConfig::default()
        };
        let res = mg_pcg(&a, &b, &vec![0.0; m], h, &cfg).expect("mg_pcg");
        assert!(res.converged, "residual {}", res.residual);
        for i in 0..m {
            let x = (i as f64 + 1.0) * h;
            let exact = x * (1.0 - x);
            assert!(
                (res.x[i] - exact).abs() < 1e-4,
                "i={i} got {} exact {exact}",
                res.x[i]
            );
        }
    }

    #[test]
    fn mesh_independent_iteration_count() {
        // The hallmark of multigrid preconditioning: iteration count stays
        // (nearly) constant under refinement, unlike plain CG.
        let cfg = MgPcgConfig {
            tol: 1.0e-8,
            ..MgPcgConfig::default()
        };
        let mut counts = Vec::new();
        for &m in &[31_usize, 63, 127] {
            let h = grid_spacing(m);
            let a = poisson_1d_interior_csr(m, h).expect("csr");
            let b = vec![1.0_f64; m];
            let res = mg_pcg(&a, &b, &vec![0.0; m], h, &cfg).expect("mg_pcg");
            assert!(res.converged);
            counts.push(res.iterations);
        }
        // No more than a couple of extra iterations across an 4× refinement.
        let spread = counts.iter().max().expect("max") - counts.iter().min().expect("min");
        assert!(
            spread <= 4,
            "iteration counts not mesh-independent: {counts:?}"
        );
    }

    #[test]
    fn fewer_iterations_than_plain_cg() {
        let m = 127;
        let h = grid_spacing(m);
        let a = poisson_1d_interior_csr(m, h).expect("csr");
        let b = vec![1.0_f64; m];
        let x0 = vec![0.0_f64; m];
        let tol = 1.0e-8;
        let cfg = MgPcgConfig {
            tol,
            ..MgPcgConfig::default()
        };
        let mg = mg_pcg(&a, &b, &x0, h, &cfg).expect("mg");
        // Identity preconditioner ⇒ unpreconditioned CG via the same engine.
        let (_, cg_iters, _, cg_conv) =
            pcg_generic(&a, &b, &x0, |r| Ok(r.to_vec()), 2000, tol).expect("cg");
        assert!(mg.converged && cg_conv);
        assert!(
            mg.iterations < cg_iters,
            "MG-PCG iters {} not fewer than CG iters {cg_iters}",
            mg.iterations
        );
    }

    #[test]
    fn preconditioner_is_symmetric() {
        let m = 31;
        let h = grid_spacing(m);
        let cfg = MgPcgConfig::default();
        let precond = GeometricMgPreconditioner::new(m, h, &cfg).expect("setup");
        let v: Vec<f64> = (0..m).map(|i| ((i as f64) * 0.41).sin()).collect();
        let w: Vec<f64> = (0..m).map(|i| ((i as f64) * 0.83 + 0.2).cos()).collect();
        let mv = precond.apply(&v).expect("apply");
        let mw = precond.apply(&w).expect("apply");
        let vmw = dot(&v, &mw).expect("dot");
        let wmv = dot(&w, &mv).expect("dot");
        let scale = vmw.abs().max(wmv.abs()).max(1.0);
        assert!(
            (vmw - wmv).abs() < 1e-9 * scale,
            "preconditioner not symmetric: {vmw} vs {wmv}"
        );
    }

    #[test]
    fn matches_direct_residual() {
        let m = 63;
        let h = grid_spacing(m);
        let a = poisson_1d_interior_csr(m, h).expect("csr");
        let b: Vec<f64> = (0..m).map(|i| ((i as f64) * 0.1).sin() + 0.5).collect();
        let res = mg_pcg(&a, &b, &vec![0.0; m], h, &MgPcgConfig::default()).expect("ok");
        let ax = a.matvec(&res.x).expect("matvec");
        let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        assert!(norm2(&r) < 1e-6, "residual {}", norm2(&r));
    }

    #[test]
    fn rejects_even_grid() {
        // m even ⇒ full grid m+2 even ⇒ coarsening rejected.
        let cfg = MgPcgConfig::default();
        assert!(matches!(
            GeometricMgPreconditioner::new(32, 0.1, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_bad_inputs() {
        let cfg = MgPcgConfig::default();
        assert!(matches!(
            GeometricMgPreconditioner::new(0, 0.1, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
        assert!(matches!(
            GeometricMgPreconditioner::new(31, -1.0, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
        assert!(matches!(
            poisson_1d_interior_csr(0, 0.1),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn apply_dim_mismatch() {
        let m = 15;
        let h = grid_spacing(m);
        let precond = GeometricMgPreconditioner::new(m, h, &MgPcgConfig::default()).expect("setup");
        assert_eq!(precond.dim(), m);
        assert!(matches!(
            precond.apply(&[1.0, 2.0]),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_square() {
        let rect = SparseCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("ok");
        assert!(matches!(
            mg_pcg(
                &rect,
                &[0.0, 0.0],
                &[0.0, 0.0],
                0.1,
                &MgPcgConfig::default()
            ),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }
}
