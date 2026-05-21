//! Bi-Conjugate Gradient Stabilized (Bi-CGSTAB) solver for general
//! (non-symmetric) linear systems `A x = b`.
//!
//! # References
//!
//! - H. A. van der Vorst, "Bi-CGSTAB: A Fast and Smoothly Converging
//!   Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems",
//!   SIAM J. Sci. Stat. Comput., 13(2), 631-644, 1992.
//! - Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
//!   SIAM, 2003, §7.4.
//!
//! # Algorithm
//!
//! Bi-CGSTAB combines a Bi-CG-style step (without requiring the
//! transposed matrix-vector product) with a one-dimensional GMRES
//! stabilisation sweep per iteration. Compared to Bi-CG it produces
//! much smoother convergence histories. Each iteration costs two
//! matrix-vector products with `A` and four `axpy`-like updates.
//!
//! The iteration may break down when either
//! `<r̃0, v> ≈ 0` (the Bi-CG component stalls) or `<t, t> ≈ 0`
//! (the stabilisation step is undefined). Both events are detected,
//! and the routine returns `converged = false` with the best residual
//! observed so far rather than panicking.

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Configuration for Bi-CGSTAB.
#[derive(Debug, Clone, Copy)]
pub struct BicgstabConfig {
    /// Maximum number of Bi-CGSTAB iterations; must be ≥ 1.
    pub max_iter: usize,
    /// Relative residual tolerance: stop when `||r|| / ||b|| < tol`.
    pub tol: f64,
}

impl Default for BicgstabConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1.0e-10,
        }
    }
}

/// Outcome of a Bi-CGSTAB solve.
#[derive(Debug, Clone)]
pub struct BicgstabResult {
    /// Final approximate solution.
    pub x: Vec<f64>,
    /// Number of Bi-CGSTAB iterations performed.
    pub iterations: usize,
    /// `true` if the relative residual fell below `cfg.tol`.
    pub converged: bool,
    /// Final relative residual `||b - A x|| / ||b||`.
    pub final_residual: f64,
}

fn validate_config(cfg: &BicgstabConfig) -> PdeResult<()> {
    if cfg.max_iter == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_iter".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: "must be positive and finite".into(),
        });
    }
    Ok(())
}

/// Threshold for declaring a near-zero scalar in inner-product breakdowns.
const BREAKDOWN_EPS: f64 = 1.0e-300;

/// Bi-CGSTAB iteration for a general linear system `A x = b`.
///
/// `x0` is the initial guess. The shadow residual `r̃_0` is fixed
/// equal to the initial residual `r_0`, which is the standard choice.
/// When `b == 0`, the routine returns `x = x0` and a relative residual
/// computed against `max(||b||, 1)`.
pub fn bicgstab(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    cfg: &BicgstabConfig,
) -> PdeResult<BicgstabResult> {
    validate_config(cfg)?;
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    if b.len() != n {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
    }
    if x0.len() != n {
        return Err(PdeError::DimensionMismatch { a: x0.len(), b: n });
    }

    let mut x = x0.to_vec();
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let r_tilde = r.clone();
    let b_norm = norm2(b).max(1.0);

    let mut residual_norm = norm2(&r);
    let mut final_rel = residual_norm / b_norm;
    if final_rel < cfg.tol {
        return Ok(BicgstabResult {
            x,
            iterations: 0,
            converged: true,
            final_residual: final_rel,
        });
    }

    let mut rho_prev = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut p = vec![0.0_f64; n];
    let mut v = vec![0.0_f64; n];

    let mut iters_done = 0usize;
    let mut converged = false;

    for it in 1..=cfg.max_iter {
        iters_done = it;
        let rho = dot(&r_tilde, &r)?;
        if rho.abs() < BREAKDOWN_EPS {
            // Bi-CGSTAB breakdown: shadow inner product vanished.
            final_rel = residual_norm / b_norm;
            break;
        }
        // beta = (rho / rho_prev) * (alpha / omega)
        if rho_prev.abs() < BREAKDOWN_EPS || omega.abs() < BREAKDOWN_EPS {
            final_rel = residual_norm / b_norm;
            break;
        }
        let beta = (rho / rho_prev) * (alpha / omega);
        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        // v = A p
        v = a.matvec(&p)?;
        let r_tilde_v = dot(&r_tilde, &v)?;
        if r_tilde_v.abs() < BREAKDOWN_EPS {
            final_rel = residual_norm / b_norm;
            break;
        }
        alpha = rho / r_tilde_v;
        // s = r - alpha * v
        let mut s = vec![0.0_f64; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let s_norm = norm2(&s);
        if s_norm / b_norm < cfg.tol {
            // Convergence in the half-step: x += alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            // Recompute true residual for reporting.
            let ax_final = a.matvec(&x)?;
            let r_final: Vec<f64> = b.iter().zip(&ax_final).map(|(bi, axi)| bi - axi).collect();
            final_rel = norm2(&r_final) / b_norm;
            converged = true;
            break;
        }
        // t = A s
        let t = a.matvec(&s)?;
        let tt = dot(&t, &t)?;
        if tt.abs() < BREAKDOWN_EPS {
            // Stabilisation step undefined; take half-step and stop.
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            let ax_final = a.matvec(&x)?;
            let r_final: Vec<f64> = b.iter().zip(&ax_final).map(|(bi, axi)| bi - axi).collect();
            final_rel = norm2(&r_final) / b_norm;
            break;
        }
        omega = dot(&t, &s)? / tt;
        // x = x + alpha p + omega s
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }
        // r = s - omega t
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        residual_norm = norm2(&r);
        final_rel = residual_norm / b_norm;
        if final_rel < cfg.tol {
            converged = true;
            break;
        }
        if omega.abs() < BREAKDOWN_EPS {
            // omega == 0 prevents continuation.
            break;
        }
        rho_prev = rho;
    }

    Ok(BicgstabResult {
        x,
        iterations: iters_done,
        converged,
        final_residual: final_rel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> BicgstabConfig {
        BicgstabConfig {
            max_iter: 200,
            tol: 1.0e-10,
        }
    }

    fn identity(n: usize) -> SparseCsr {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = vec![1.0; n];
        SparseCsr::new(n, n, row_ptr, cols, vals).expect("ok")
    }

    fn spd_tridiag5() -> SparseCsr {
        SparseCsr::new(
            5,
            5,
            vec![0, 2, 5, 8, 11, 13],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
            vec![
                2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0,
            ],
        )
        .expect("ok")
    }

    fn nonsym_tridiag5() -> SparseCsr {
        SparseCsr::new(
            5,
            5,
            vec![0, 2, 5, 8, 11, 13],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
            vec![
                2.1, -0.9, -1.0, 2.1, -0.9, -1.0, 2.1, -0.9, -1.0, 2.1, -0.9, -1.0, 2.1,
            ],
        )
        .expect("ok")
    }

    fn residual(a: &SparseCsr, x: &[f64], b: &[f64]) -> f64 {
        let ax = a.matvec(x).expect("ok");
        ((0..b.len()).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>()).sqrt()
    }

    #[test]
    fn bicgstab_identity_converges_in_one_iter() {
        let a = identity(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = bicgstab(&a, &b, &[0.0; 4], &cfg_default()).expect("ok");
        assert!(res.converged);
        assert!(res.iterations <= 1);
        for (i, &bi) in b.iter().enumerate() {
            assert!((res.x[i] - bi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn bicgstab_spd_tridiag_converges() {
        let a = spd_tridiag5();
        let b = vec![1.0; 5];
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged, "rel resid {}", res.final_residual);
        assert!(residual(&a, &res.x, &b) < 1.0e-9);
    }

    #[test]
    fn bicgstab_nonsymmetric_converges() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged, "rel resid {}", res.final_residual);
        assert!(residual(&a, &res.x, &b) < 1.0e-9);
    }

    #[test]
    fn bicgstab_zero_rhs_yields_x0() {
        let a = nonsym_tridiag5();
        let x0 = vec![0.0; 5];
        let res = bicgstab(&a, &[0.0; 5], &x0, &cfg_default()).expect("ok");
        assert!(res.converged);
        for (i, &x0i) in x0.iter().enumerate() {
            assert!((res.x[i] - x0i).abs() < 1.0e-14);
        }
    }

    #[test]
    fn bicgstab_b_is_column_of_a() {
        let a = nonsym_tridiag5();
        let mut e3 = vec![0.0; 5];
        e3[3] = 1.0;
        let b = a.matvec(&e3).expect("ok");
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged);
        for (i, &e3i) in e3.iter().enumerate() {
            assert!((res.x[i] - e3i).abs() < 1.0e-8);
        }
    }

    #[test]
    fn bicgstab_breakdown_zero_matrix_no_panic() {
        // A is the zero matrix on a 2x2 grid (degenerate). The first
        // matvec gives r = b, but <r̃0, v> = <r̃0, A p> = 0. Should
        // gracefully return non-converged.
        let a = SparseCsr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0, 0.0]).expect("ok");
        let b = vec![1.0, 1.0];
        let res = bicgstab(&a, &b, &[0.0, 0.0], &cfg_default()).expect("ok");
        assert!(!res.converged);
        assert!(res.final_residual.is_finite());
    }

    #[test]
    fn bicgstab_large_iter_limit() {
        let a = nonsym_tridiag5();
        let b = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let cfg = BicgstabConfig {
            max_iter: 5000,
            tol: 1.0e-12,
        };
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-10);
    }

    #[test]
    fn bicgstab_tight_tol_respected() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let cfg = BicgstabConfig {
            max_iter: 2000,
            tol: 1.0e-12,
        };
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-11);
    }

    #[test]
    fn bicgstab_max_iter_zero_invalid() {
        let a = identity(3);
        let cfg = BicgstabConfig {
            max_iter: 0,
            tol: 1.0e-10,
        };
        let r = bicgstab(&a, &[1.0; 3], &[0.0; 3], &cfg);
        assert!(matches!(r, Err(PdeError::InvalidParameter { .. })));
    }

    #[test]
    fn bicgstab_tol_nonpositive_invalid() {
        let a = identity(3);
        let cfg = BicgstabConfig {
            max_iter: 10,
            tol: 0.0,
        };
        let r = bicgstab(&a, &[1.0; 3], &[0.0; 3], &cfg);
        assert!(matches!(r, Err(PdeError::InvalidParameter { .. })));
        let cfg_neg = BicgstabConfig {
            max_iter: 10,
            tol: -1.0,
        };
        let r2 = bicgstab(&a, &[1.0; 3], &[0.0; 3], &cfg_neg);
        assert!(matches!(r2, Err(PdeError::InvalidParameter { .. })));
    }

    #[test]
    fn bicgstab_dim_mismatch_b() {
        let a = identity(3);
        let r = bicgstab(&a, &[1.0; 4], &[0.0; 3], &cfg_default());
        assert!(matches!(r, Err(PdeError::DimensionMismatch { .. })));
    }

    #[test]
    fn bicgstab_dim_mismatch_x0() {
        let a = identity(3);
        let r = bicgstab(&a, &[1.0; 3], &[0.0; 2], &cfg_default());
        assert!(matches!(r, Err(PdeError::DimensionMismatch { .. })));
    }

    #[test]
    fn bicgstab_initial_guess_close_to_solution() {
        let a = spd_tridiag5();
        let b = vec![1.0; 5];
        let res1 = bicgstab(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        let res2 = bicgstab(&a, &b, &res1.x, &cfg_default()).expect("ok");
        assert!(res2.converged);
        // Already converged → no further iterations needed.
        assert!(res2.iterations <= 1);
    }

    #[test]
    fn bicgstab_low_iter_truncation_reports_residual() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let cfg = BicgstabConfig {
            max_iter: 1,
            tol: 1.0e-14,
        };
        let res = bicgstab(&a, &b, &[0.0; 5], &cfg).expect("ok");
        // 1 iteration is far too few; should not converge.
        assert_eq!(res.iterations, 1);
        assert!(res.final_residual.is_finite());
    }
}
