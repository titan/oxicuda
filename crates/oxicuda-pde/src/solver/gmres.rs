//! Restarted Generalized Minimal Residual (GMRES(m)) solver for
//! non-symmetric linear systems `A x = b`.
//!
//! # References
//!
//! - Y. Saad and M. H. Schultz, "GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems", SIAM J. Sci.
//!   Stat. Comput., 7(3), 856-869, 1986.
//! - Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
//!   SIAM, 2003, Chapter 6.
//!
//! # Algorithm
//!
//! Restarted GMRES(m) builds an `m`-dimensional Krylov subspace
//! `K_m(A, r_0) = span{r_0, A r_0, ..., A^{m-1} r_0}` via the Arnoldi
//! process and finds the minimizer of `||b - A x||_2` over `x_0 + K_m`.
//! The Hessenberg matrix produced by Arnoldi is reduced to upper
//! triangular form by Givens rotations applied on-the-fly, so the
//! least-squares problem is solved cheaply via back substitution.
//! When the inner loop reaches `m` steps without converging, the
//! algorithm restarts from the latest approximation, bounding the
//! storage to `(m + 1) n` doubles.
//!
//! All arithmetic is pure Rust with `Vec<f64>` storage and no
//! `unsafe`. The CSR matrix type is [`crate::solver::sparse::SparseCsr`].

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Configuration for restarted GMRES(m).
#[derive(Debug, Clone, Copy)]
pub struct GmresConfig {
    /// Krylov subspace dimension between restarts (`m`); must be ≥ 1.
    pub restart: usize,
    /// Maximum number of restart cycles; must be ≥ 1.
    pub max_restarts: usize,
    /// Relative residual tolerance: stop when `||r|| / ||b|| < tol`.
    pub tol: f64,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            restart: 30,
            max_restarts: 50,
            tol: 1.0e-10,
        }
    }
}

/// Outcome of a restarted GMRES(m) solve.
#[derive(Debug, Clone)]
pub struct GmresResult {
    /// Final approximate solution.
    pub x: Vec<f64>,
    /// Total number of Arnoldi steps performed across all restart cycles.
    pub iterations: usize,
    /// Number of restart cycles entered (including the one that converged).
    pub restarts: usize,
    /// `true` if the relative residual fell below `cfg.tol`.
    pub converged: bool,
    /// Final achieved relative residual `||b - A x|| / ||b||`.
    pub final_residual: f64,
}

fn validate_config(cfg: &GmresConfig) -> PdeResult<()> {
    if cfg.restart == 0 {
        return Err(PdeError::InvalidParameter {
            name: "restart".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if cfg.max_restarts == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_restarts".into(),
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

/// Restarted GMRES(m) for a general (possibly non-symmetric) linear system
/// `A x = b`.
///
/// `x0` is the initial guess. The algorithm restarts every `cfg.restart`
/// Arnoldi steps and performs at most `cfg.max_restarts` restart cycles.
/// Convergence is declared when the relative residual `||r|| / ||b||`
/// drops below `cfg.tol`. When `b == 0`, the returned solution equals
/// `x0` and the residual is reported relative to `max(||b||, 1)`.
pub fn gmres(a: &SparseCsr, b: &[f64], x0: &[f64], cfg: &GmresConfig) -> PdeResult<GmresResult> {
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

    let m = cfg.restart;
    let b_norm = norm2(b).max(1.0);

    let mut x = x0.to_vec();

    // Compute initial residual r0 = b - A x0
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut beta = norm2(&r);
    let mut final_rel = beta / b_norm;

    if final_rel < cfg.tol {
        return Ok(GmresResult {
            x,
            iterations: 0,
            restarts: 0,
            converged: true,
            final_residual: final_rel,
        });
    }

    // Pre-allocate Arnoldi data structures, reused across restarts.
    // V holds the orthonormal Krylov basis vectors as flat Vec<f64> of length n.
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    for _ in 0..=m {
        v.push(vec![0.0; n]);
    }
    // H is stored column-major as a Vec<Vec<f64>> of length m,
    // where h[k] has length k + 2 (entries 0..=k of upper part plus subdiagonal at index k+1).
    let mut h: Vec<Vec<f64>> = Vec::with_capacity(m);
    // Givens rotation cosines/sines stored per column index k.
    let mut cs: Vec<f64> = Vec::with_capacity(m);
    let mut sn: Vec<f64> = Vec::with_capacity(m);
    // g is the right-hand side for the reduced least-squares problem.
    let mut g: Vec<f64> = Vec::with_capacity(m + 1);

    let mut total_iters: usize = 0;
    let mut converged = false;
    let mut restart_count: usize = 0;

    'outer: for _ in 0..cfg.max_restarts {
        restart_count += 1;
        // V[0] = r / beta
        for i in 0..n {
            v[0][i] = r[i] / beta;
        }
        h.clear();
        cs.clear();
        sn.clear();
        g.clear();
        g.resize(m + 1, 0.0);
        g[0] = beta;

        let mut inner_k_used = 0;
        for k in 0..m {
            // w = A * V[k]
            let mut w = a.matvec(&v[k])?;
            let mut h_col = vec![0.0_f64; k + 2];
            // Modified Gram-Schmidt orthogonalisation against V[0..=k].
            for (i, h_ik) in h_col.iter_mut().enumerate().take(k + 1) {
                let dot_iw = dot(&v[i], &w)?;
                *h_ik = dot_iw;
                for j in 0..n {
                    w[j] -= dot_iw * v[i][j];
                }
            }
            let w_norm = norm2(&w);
            h_col[k + 1] = w_norm;
            total_iters += 1;
            inner_k_used = k + 1;
            // Build next basis vector if w is non-degenerate.
            // If w_norm == 0 we have an "happy breakdown": A invariant subspace
            // reached, the iteration should converge exactly at this step.
            let breakdown = w_norm < 1.0e-300;
            if !breakdown {
                for j in 0..n {
                    v[k + 1][j] = w[j] / w_norm;
                }
            }

            // Apply previous Givens rotations to column k of H.
            for i in 0..k {
                let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
                h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1];
                h_col[i] = temp;
            }
            // Compute new Givens rotation to zero h_col[k+1].
            let (ck, sk) = givens(h_col[k], h_col[k + 1]);
            cs.push(ck);
            sn.push(sk);
            // Apply to column k.
            h_col[k] = ck * h_col[k] + sk * h_col[k + 1];
            h_col[k + 1] = 0.0;
            // Apply rotation to g.
            let temp = ck * g[k] + sk * g[k + 1];
            g[k + 1] = -sk * g[k] + ck * g[k + 1];
            g[k] = temp;

            h.push(h_col);

            let residual_estimate = g[k + 1].abs();
            final_rel = residual_estimate / b_norm;
            if final_rel < cfg.tol || breakdown {
                converged = breakdown || final_rel < cfg.tol;
                // Solve upper triangular system H[:k+1,:k+1] y = g[:k+1].
                let y = back_solve(&h, &g, k + 1)?;
                // x = x + V[:,:k+1] y
                for (i, &yi) in y.iter().enumerate().take(k + 1) {
                    for j in 0..n {
                        x[j] += yi * v[i][j];
                    }
                }
                break 'outer;
            }
        }

        // Did not converge within m steps: form approximation and restart.
        if inner_k_used == 0 {
            // Defensive: should not be hit since restart >= 1 is enforced.
            break;
        }
        let y = back_solve(&h, &g, inner_k_used)?;
        for (i, &yi) in y.iter().enumerate().take(inner_k_used) {
            for j in 0..n {
                x[j] += yi * v[i][j];
            }
        }
        // Recompute true residual to avoid drift.
        let ax_new = a.matvec(&x)?;
        for j in 0..n {
            r[j] = b[j] - ax_new[j];
        }
        beta = norm2(&r);
        final_rel = beta / b_norm;
        if final_rel < cfg.tol {
            converged = true;
            break;
        }
    }

    Ok(GmresResult {
        x,
        iterations: total_iters,
        restarts: restart_count,
        converged,
        final_residual: final_rel,
    })
}

/// Compute a Givens rotation `(c, s)` such that
/// `[[c, s], [-s, c]] * [a; b] = [r; 0]` with `r = sqrt(a^2 + b^2)`.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        if a >= 0.0 { (1.0, 0.0) } else { (-1.0, 0.0) }
    } else if a.abs() < b.abs() {
        let tau = a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt() * b.signum();
        let c = s * tau;
        (c, s)
    } else {
        let tau = b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt() * a.signum();
        let s = c * tau;
        (c, s)
    }
}

/// Back-substitute the upper triangular `k x k` block of `H`
/// (with `H[i,j] = h[j][i]`) against the first `k` entries of `g`.
fn back_solve(h: &[Vec<f64>], g: &[f64], k: usize) -> PdeResult<Vec<f64>> {
    let mut y = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let diag = h[i][i];
        if diag.abs() < 1.0e-300 {
            return Err(PdeError::NumericalInstability(
                "gmres: zero diagonal in Hessenberg back-solve".into(),
            ));
        }
        let mut s = g[i];
        for j in (i + 1)..k {
            s -= h[j][i] * y[j];
        }
        y[i] = s / diag;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> GmresConfig {
        GmresConfig {
            restart: 20,
            max_restarts: 20,
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
        // tridiag(-1, 2.1, -0.9)
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

    fn vec_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn gmres_identity_converges_immediately() {
        let a = identity(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = gmres(&a, &b, &[0.0; 4], &cfg_default()).expect("ok");
        assert!(res.converged);
        assert!(vec_close(&res.x, &b, 1.0e-10));
        // Solved in a single Arnoldi step.
        assert_eq!(res.iterations, 1);
    }

    #[test]
    fn gmres_spd_5x5_converges() {
        let a = spd_tridiag5();
        let b = vec![1.0; 5];
        let res = gmres(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged, "residual = {}", res.final_residual);
        let ax = a.matvec(&res.x).expect("ok");
        let r: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1.0e-9);
        // Should converge in at most n Arnoldi steps.
        assert!(res.iterations <= 5);
    }

    #[test]
    fn gmres_nonsymmetric_5x5_converges() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let res = gmres(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged, "residual = {}", res.final_residual);
        let ax = a.matvec(&res.x).expect("ok");
        let r: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1.0e-9);
    }

    #[test]
    fn gmres_zero_rhs_returns_x0() {
        let a = nonsym_tridiag5();
        let x0 = vec![0.0; 5];
        let res = gmres(&a, &[0.0; 5], &x0, &cfg_default()).expect("ok");
        assert!(res.converged);
        assert!(vec_close(&res.x, &x0, 1.0e-14));
    }

    #[test]
    fn gmres_b_is_column_of_a_recovers_unit_vector() {
        let a = nonsym_tridiag5();
        // b = A * e_2 (third column).
        let mut e2 = vec![0.0; 5];
        e2[2] = 1.0;
        let b = a.matvec(&e2).expect("ok");
        let res = gmres(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        assert!(res.converged);
        assert!(vec_close(&res.x, &e2, 1.0e-9));
    }

    #[test]
    fn gmres_respects_tol() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let cfg = GmresConfig {
            restart: 20,
            max_restarts: 20,
            tol: 1.0e-6,
        };
        let res = gmres(&a, &b, &[0.0; 5], &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-6);
    }

    #[test]
    fn gmres_max_restarts_truncates() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let cfg = GmresConfig {
            restart: 1,
            max_restarts: 1,
            tol: 1.0e-14,
        };
        let res = gmres(&a, &b, &[0.0; 5], &cfg).expect("ok");
        // 1 Arnoldi step with restart=1, then truncate
        assert!(!res.converged);
        assert_eq!(res.iterations, 1);
        assert_eq!(res.restarts, 1);
    }

    #[test]
    fn gmres_more_restarts_does_not_exceed_iterations_of_few_restarts() {
        let a = nonsym_tridiag5();
        let b = vec![1.0, 2.0, 0.5, -0.25, 1.5];
        let cfg_small = GmresConfig {
            restart: 2,
            max_restarts: 50,
            tol: 1.0e-10,
        };
        let cfg_large = GmresConfig {
            restart: 20,
            max_restarts: 50,
            tol: 1.0e-10,
        };
        let res_small = gmres(&a, &b, &[0.0; 5], &cfg_small).expect("ok");
        let res_large = gmres(&a, &b, &[0.0; 5], &cfg_large).expect("ok");
        assert!(res_small.converged);
        assert!(res_large.converged);
        // Larger Krylov space should converge in no more Arnoldi steps.
        assert!(res_large.iterations <= res_small.iterations);
    }

    #[test]
    fn gmres_invalid_restart_zero() {
        let a = identity(3);
        let cfg = GmresConfig {
            restart: 0,
            max_restarts: 1,
            tol: 1.0e-10,
        };
        let r = gmres(&a, &[1.0; 3], &[0.0; 3], &cfg);
        assert!(matches!(r, Err(PdeError::InvalidParameter { .. })));
    }

    #[test]
    fn gmres_invalid_max_restarts_zero() {
        let a = identity(3);
        let cfg = GmresConfig {
            restart: 5,
            max_restarts: 0,
            tol: 1.0e-10,
        };
        let r = gmres(&a, &[1.0; 3], &[0.0; 3], &cfg);
        assert!(matches!(r, Err(PdeError::InvalidParameter { .. })));
    }

    #[test]
    fn gmres_invalid_tol_nonpositive() {
        let a = identity(3);
        let cfg = GmresConfig {
            restart: 5,
            max_restarts: 5,
            tol: 0.0,
        };
        let r = gmres(&a, &[1.0; 3], &[0.0; 3], &cfg);
        assert!(matches!(r, Err(PdeError::InvalidParameter { .. })));
        let cfg_neg = GmresConfig {
            restart: 5,
            max_restarts: 5,
            tol: -1.0e-6,
        };
        let r2 = gmres(&a, &[1.0; 3], &[0.0; 3], &cfg_neg);
        assert!(matches!(r2, Err(PdeError::InvalidParameter { .. })));
    }

    #[test]
    fn gmres_dim_mismatch_b() {
        let a = identity(3);
        let r = gmres(&a, &[1.0; 4], &[0.0; 3], &cfg_default());
        assert!(matches!(r, Err(PdeError::DimensionMismatch { .. })));
    }

    #[test]
    fn gmres_dim_mismatch_x0() {
        let a = identity(3);
        let r = gmres(&a, &[1.0; 3], &[0.0; 2], &cfg_default());
        assert!(matches!(r, Err(PdeError::DimensionMismatch { .. })));
    }

    #[test]
    fn gmres_initial_guess_close_to_solution() {
        let a = spd_tridiag5();
        let b = vec![1.0; 5];
        // First solve, then use the solution as the initial guess for a second solve.
        let res1 = gmres(&a, &b, &[0.0; 5], &cfg_default()).expect("ok");
        let res2 = gmres(&a, &b, &res1.x, &cfg_default()).expect("ok");
        assert!(res2.converged);
        // Should require zero or very few additional Arnoldi steps.
        assert!(res2.iterations <= 1);
    }
}
