//! Algebraic-multigrid (AMG) preconditioner for the conjugate-gradient method.
//!
//! This module wraps the existing smoothed-aggregation AMG hierarchy
//! ([`crate::multigrid::amg::AmgSolver`]) and uses a **fixed** number of its
//! V-cycles (started from zero) as the preconditioner application
//! `z = M⁻¹ r` inside preconditioned conjugate gradient (PCG).
//!
//! # Why a fixed number of V-cycles
//!
//! For PCG to be a valid Krylov method the preconditioner `M⁻¹` must be a
//! *fixed* symmetric positive-definite linear operator. A multigrid V-cycle
//! built from
//!
//! * a symmetric smoother applied the same number of times for pre- and
//!   post-smoothing (here weighted Jacobi, which from a zero start is the
//!   symmetric operator `B = ω D^{-1/2}(Σ_j G̃ʲ) D^{-1/2}`),
//! * Galerkin coarsening with restriction `R = Pᵀ`, and
//! * a symmetric coarse solve,
//!
//! is itself a fixed symmetric linear operator, so `vᵀ M⁻¹ w = wᵀ M⁻¹ v`. We
//! obtain it without modifying the AMG code by configuring the underlying
//! [`AmgSolver`] with `tol = 0` and `max_outer_iter = n_precond_cycles`, so each
//! [`AmgSolver::solve`] call performs exactly that many V-cycles from a zero
//! initial guess.
//!
//! # References
//!
//! * Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003,
//!   §9.6 (preconditioned CG) and ch. 13 (multigrid preconditioning).
//! * Trottenberg, Oosterlee & Schüller, *Multigrid*, Academic Press 2001.

use crate::error::{PdeError, PdeResult};
use crate::multigrid::amg::{AmgConfig, AmgSolver};
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Configuration for the AMG-preconditioned conjugate-gradient solver.
#[derive(Debug, Clone)]
pub struct AmgPcgConfig {
    /// Maximum number of (outer) PCG iterations.
    pub max_iter: usize,
    /// Relative residual tolerance `‖b − A x‖ / ‖b‖`.
    pub tol: f64,
    /// Maximum number of AMG levels in the hierarchy.
    pub amg_max_levels: usize,
    /// Strength-of-connection threshold for aggregation.
    pub amg_agg_threshold: f64,
    /// Number of pre/post weighted-Jacobi smoothing sweeps per level.
    pub amg_nu_smooth: usize,
    /// Number of V-cycles per preconditioner application (1 is standard).
    pub n_precond_cycles: usize,
}

impl Default for AmgPcgConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1.0e-8,
            amg_max_levels: 6,
            amg_agg_threshold: 0.08,
            amg_nu_smooth: 2,
            n_precond_cycles: 1,
        }
    }
}

/// Result of an AMG-preconditioned CG solve.
#[derive(Debug, Clone)]
pub struct AmgPcgResult {
    /// Approximate solution vector.
    pub x: Vec<f64>,
    /// Number of PCG iterations performed.
    pub iterations: usize,
    /// Final residual norm `‖b − A x‖`.
    pub residual: f64,
    /// Whether the relative tolerance was reached.
    pub converged: bool,
}

/// A reusable AMG preconditioner: applies a fixed number of V-cycles as `M⁻¹`.
#[derive(Debug, Clone)]
pub struct AmgPreconditioner {
    solver: AmgSolver,
    n: usize,
}

impl AmgPreconditioner {
    /// Build the preconditioner from a dense row-major `n × n` SPD matrix.
    ///
    /// # Errors
    ///
    /// Propagates [`AmgSolver::setup`] errors (e.g. `n == 0` or a length
    /// mismatch).
    pub fn from_dense(a: &[f64], n: usize, cfg: &AmgPcgConfig) -> PdeResult<Self> {
        let amg_cfg = AmgConfig {
            max_levels: cfg.amg_max_levels.max(1),
            agg_threshold: cfg.amg_agg_threshold,
            nu_smooth: cfg.amg_nu_smooth,
            // tol = 0 ⇒ the outer residual check never short-circuits, so each
            // `solve` runs exactly `max_outer_iter` V-cycles: a fixed operator.
            tol: 0.0,
            max_outer_iter: cfg.n_precond_cycles.max(1),
        };
        let solver = AmgSolver::setup(a, n, amg_cfg)?;
        Ok(Self { solver, n })
    }

    /// Build the preconditioner from a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`PdeError::DimensionMismatch`] if `a` is not square.
    pub fn from_csr(a: &SparseCsr, cfg: &AmgPcgConfig) -> PdeResult<Self> {
        let (dense, n) = csr_to_dense(a)?;
        Self::from_dense(&dense, n, cfg)
    }

    /// Problem dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Apply the preconditioner: `z = M⁻¹ r` via a fixed number of V-cycles.
    ///
    /// # Errors
    ///
    /// Returns [`PdeError::DimensionMismatch`] if `r.len() != dim()`.
    pub fn apply(&self, r: &[f64]) -> PdeResult<Vec<f64>> {
        if r.len() != self.n {
            return Err(PdeError::DimensionMismatch {
                a: r.len(),
                b: self.n,
            });
        }
        self.solver.solve(r)
    }
}

/// Convert a square CSR matrix to dense row-major storage.
fn csr_to_dense(a: &SparseCsr) -> PdeResult<(Vec<f64>, usize)> {
    if a.n_rows != a.n_cols {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    let n = a.n_rows;
    let mut dense = vec![0.0_f64; n * n];
    for i in 0..n {
        let lo = a.row_ptr[i];
        let hi = a.row_ptr[i + 1];
        for k in lo..hi {
            dense[i * n + a.cols[k]] += a.vals[k];
        }
    }
    Ok((dense, n))
}

/// Generic preconditioned conjugate gradient returning the iteration count.
///
/// `apply_precond` computes `z = M⁻¹ r`. Passing the identity recovers plain
/// (unpreconditioned) CG, which makes the iteration counts directly comparable.
///
/// Returns `(x, iterations, residual_norm, converged)`.
pub(crate) fn pcg_generic<P>(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    apply_precond: P,
    max_iter: usize,
    tol: f64,
) -> PdeResult<(Vec<f64>, usize, f64, bool)>
where
    P: Fn(&[f64]) -> PdeResult<Vec<f64>>,
{
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    if b.len() != n || x0.len() != n {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let b_norm = norm2(b).max(1.0);
    let mut res_norm = norm2(&r);
    if res_norm / b_norm < tol {
        return Ok((x, 0, res_norm, true));
    }
    let mut z = apply_precond(&r)?;
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z)?;
    let mut iterations = 0_usize;
    let mut converged = false;
    for it in 0..max_iter {
        let ap = a.matvec(&p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(PdeError::NumericalInstability(
                "amg_pcg: zero pᵀAp (matrix may not be SPD)".into(),
            ));
        }
        let alpha = rz_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        iterations = it + 1;
        res_norm = norm2(&r);
        if res_norm / b_norm < tol {
            converged = true;
            break;
        }
        z = apply_precond(&r)?;
        let rz_new = dot(&r, &z)?;
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }
    Ok((x, iterations, res_norm, converged))
}

/// Solve the SPD system `A x = b` with AMG-preconditioned conjugate gradient.
///
/// `A` must be a square symmetric positive-definite CSR matrix and `x0` is the
/// initial guess (use zeros if unsure).
///
/// # Errors
///
/// * [`PdeError::DimensionMismatch`] if `A` is not square or the vector lengths
///   are inconsistent.
/// * [`PdeError::NumericalInstability`] if a non-SPD breakdown occurs.
/// * Setup errors from the underlying AMG hierarchy.
pub fn amg_pcg(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    cfg: &AmgPcgConfig,
) -> PdeResult<AmgPcgResult> {
    let precond = AmgPreconditioner::from_csr(a, cfg)?;
    let (x, iterations, residual, converged) =
        pcg_generic(a, b, x0, |r| precond.apply(r), cfg.max_iter, cfg.tol)?;
    Ok(AmgPcgResult {
        x,
        iterations,
        residual,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-D interior Dirichlet Laplacian `(1/h²)·tridiag(-1, 2, -1)` as CSR —
    /// symmetric positive definite.
    fn laplacian_1d_csr(m: usize, h: f64) -> SparseCsr {
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
        SparseCsr::new(m, m, row_ptr, cols, vals).expect("valid csr")
    }

    /// Dense Gaussian elimination with partial pivoting (reference solver).
    fn dense_solve(a_in: &[f64], b_in: &[f64], n: usize) -> Vec<f64> {
        let mut a = a_in.to_vec();
        let mut b = b_in.to_vec();
        for col in 0..n {
            // Partial pivot.
            let mut piv = col;
            let mut best = a[col * n + col].abs();
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > best {
                    best = v;
                    piv = row;
                }
            }
            if piv != col {
                for c in 0..n {
                    a.swap(col * n + c, piv * n + c);
                }
                b.swap(col, piv);
            }
            let diag = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / diag;
                if factor == 0.0 {
                    continue;
                }
                for c in col..n {
                    a[row * n + c] -= factor * a[col * n + c];
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = vec![0.0_f64; n];
        for col in (0..n).rev() {
            let mut s = b[col];
            for c in (col + 1)..n {
                s -= a[col * n + c] * x[c];
            }
            x[col] = s / a[col * n + col];
        }
        x
    }

    #[test]
    fn amg_pcg_solves_to_tolerance() {
        let m = 31;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let b = vec![1.0_f64; m];
        let cfg = AmgPcgConfig {
            tol: 1.0e-9,
            ..AmgPcgConfig::default()
        };
        let res = amg_pcg(&a, &b, &vec![0.0; m], &cfg).expect("amg_pcg ok");
        assert!(res.converged, "did not converge, residual {}", res.residual);
        let ax = a.matvec(&res.x).expect("matvec ok");
        let abs_res: f64 = (0..m).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(abs_res < 1.0e-6, "absolute residual {abs_res}");
    }

    #[test]
    fn amg_pcg_fewer_iterations_than_plain_cg() {
        let m = 63;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let b = vec![1.0_f64; m];
        let x0 = vec![0.0_f64; m];
        let tol = 1.0e-8;
        let cfg = AmgPcgConfig {
            max_iter: 500,
            tol,
            ..AmgPcgConfig::default()
        };
        let pcg = amg_pcg(&a, &b, &x0, &cfg).expect("amg_pcg ok");
        // Same PCG engine with the identity preconditioner = unpreconditioned CG.
        let (_, cg_iters, _, cg_conv) =
            pcg_generic(&a, &b, &x0, |r| Ok(r.to_vec()), 500, tol).expect("cg ok");
        assert!(pcg.converged && cg_conv, "both solvers must converge");
        assert!(
            pcg.iterations < cg_iters,
            "AMG-PCG iters {} not fewer than CG iters {cg_iters}",
            pcg.iterations
        );
    }

    #[test]
    fn preconditioner_is_symmetric() {
        let m = 31;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let cfg = AmgPcgConfig::default();
        let precond = AmgPreconditioner::from_csr(&a, &cfg).expect("setup ok");
        // Two deterministic, distinct test vectors.
        let v: Vec<f64> = (0..m).map(|i| ((i as f64) * 0.37).sin()).collect();
        let w: Vec<f64> = (0..m).map(|i| ((i as f64) * 0.91 + 0.5).cos()).collect();
        let mv = precond.apply(&v).expect("apply ok");
        let mw = precond.apply(&w).expect("apply ok");
        let vmw = dot(&v, &mw).expect("dot ok");
        let wmv = dot(&w, &mv).expect("dot ok");
        let scale = vmw.abs().max(wmv.abs()).max(1.0);
        assert!(
            (vmw - wmv).abs() < 1.0e-9 * scale,
            "preconditioner not symmetric: vᵀM⁻¹w {vmw} vs wᵀM⁻¹v {wmv}"
        );
    }

    #[test]
    fn matches_dense_reference() {
        let m = 20;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let b: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let cfg = AmgPcgConfig {
            tol: 1.0e-10,
            max_iter: 500,
            ..AmgPcgConfig::default()
        };
        let res = amg_pcg(&a, &b, &vec![0.0; m], &cfg).expect("amg_pcg ok");
        let (dense, n) = csr_to_dense(&a).expect("dense ok");
        let x_ref = dense_solve(&dense, &b, n);
        let max_err = res
            .x
            .iter()
            .zip(&x_ref)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1.0e-6, "max err vs dense reference {max_err}");
    }

    #[test]
    fn solution_is_finite() {
        let m = 16;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let b = vec![1.0_f64; m];
        let res = amg_pcg(&a, &b, &vec![0.0; m], &AmgPcgConfig::default()).expect("ok");
        assert!(res.x.iter().all(|v| v.is_finite()));
        assert!(res.residual.is_finite());
    }

    #[test]
    fn rejects_non_square() {
        let rect = SparseCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("ok");
        let cfg = AmgPcgConfig::default();
        assert!(matches!(
            AmgPreconditioner::from_csr(&rect, &cfg),
            Err(PdeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            amg_pcg(&rect, &[0.0, 0.0], &[0.0, 0.0, 0.0], &cfg),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn preconditioner_dim_and_apply_mismatch() {
        let m = 8;
        let h = 1.0 / (m as f64 + 1.0);
        let a = laplacian_1d_csr(m, h);
        let precond = AmgPreconditioner::from_csr(&a, &AmgPcgConfig::default()).expect("ok");
        assert_eq!(precond.dim(), m);
        assert!(matches!(
            precond.apply(&[1.0, 2.0]),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }
}
