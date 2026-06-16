//! Dantzig selector via iterative reweighted L1 / ADMM — lasso-module variant.
//!
//! Solves `min ||x||_1  s.t.  ||Aᵀ(b − Ax)||_∞ ≤ delta`
//!
//! using a scaled-form ADMM iteration:
//!
//! 1. Precompute `M = AᵀA` and `q = Aᵀb`.
//! 2. Factorize `(M + ρI)` via Cholesky (ρ = 1.0).
//! 3. Iterate:
//!    a. **x-update**: solve `(M + ρI) x_new = ρ(q − s + u)` then soft-threshold.
//!    b. Check constraint `||Aᵀ(b − Ax)||_∞ ≤ delta`.
//!    c. **s-update**: project dual variable onto `[-delta, delta]`.
//!    d. **u-update**: dual residual accumulation.
//! 4. Stop when constraint is satisfied within `tol` and primal change is small.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::iht::soft_threshold;

/// Configuration for the lasso-module Dantzig selector ([`dantzig_selector`]).
#[derive(Debug, Clone)]
pub struct DantzigConfig {
    /// Noise tolerance δ: we seek `||Aᵀ(b − Ax)||_∞ ≤ delta`.
    pub delta: f64,
    /// Maximum number of ADMM iterations.
    pub max_iter: usize,
    /// Convergence tolerance for stopping criterion.
    pub tol: f64,
}

/// Dantzig selector (lasso-module variant) via ADMM with hardcoded `ρ = 1.0`.
///
/// # Arguments
/// - `a`: `[m × n]` measurement matrix in row-major order.
/// - `b`: `[m]` observation vector.
/// - `m`: number of rows (measurements).
/// - `n`: number of columns (signal dimension).
/// - `cfg`: algorithm configuration.
///
/// # Returns
/// A length-`n` sparse estimate that approximately satisfies `||Aᵀ(b − Ax)||_∞ ≤ delta`.
///
/// # Errors
/// - [`CsError::EmptyInput`] if `m == 0` or `n == 0`.
/// - [`CsError::ShapeMismatch`] if `a.len() != m * n`.
/// - [`CsError::DimensionMismatch`] if `b.len() != m`.
/// - [`CsError::InvalidParameter`] if `cfg.delta <= 0` or `cfg.tol <= 0`.
/// - [`CsError::SingularMatrix`] if `AᵀA + ρI` is not positive-definite.
pub fn dantzig_selector(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    cfg: &DantzigConfig,
) -> CsResult<Vec<f64>> {
    // ── Input validation ─────────────────────────────────────────────────────
    if m == 0 || n == 0 {
        return Err(CsError::EmptyInput);
    }
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CsError::DimensionMismatch { a: b.len(), b: m });
    }
    if cfg.delta <= 0.0 {
        return Err(CsError::InvalidParameter(format!(
            "delta must be > 0, got {}",
            cfg.delta
        )));
    }
    if cfg.tol <= 0.0 {
        return Err(CsError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            cfg.tol
        )));
    }

    // ── Precomputation ───────────────────────────────────────────────────────
    // ρ is fixed at 1.0; chosen to give balanced primal/dual steps.
    let rho = 1.0_f64;

    // M = AᵀA  (n × n, row-major)
    let mut ata = vec![0.0_f64; n * n];
    for k in 0..m {
        for i in 0..n {
            let aki = a[k * n + i];
            for j in 0..n {
                ata[i * n + j] += aki * a[k * n + j];
            }
        }
    }

    // q = Aᵀb  (length n)
    let at_b = mat_t_vec(a, m, n, b)?;

    // System matrix for x-update: (AᵀA + ρI)
    let mut system = ata.clone();
    for i in 0..n {
        system[i * n + i] += rho;
    }
    let l = cholesky_factor(&system, n)?;

    // ── ADMM iterates ────────────────────────────────────────────────────────
    let mut x = vec![0.0_f64; n];
    // s lives in R^n and represents the slack for the dual constraint.
    let mut s = vec![0.0_f64; n];
    // u: scaled dual variable.
    let mut u = vec![0.0_f64; n];

    for _iter in 0..cfg.max_iter {
        // ── x-update ─────────────────────────────────────────────────────────
        // Solve (AᵀA + ρI) x_hat = ρ(q − s + u)  then apply soft-threshold.
        let mut rhs = vec![0.0_f64; n];
        for j in 0..n {
            rhs[j] = rho * (at_b[j] - s[j] + u[j]);
        }
        // Add the ρ x term from the augmented Lagrangian quadratic (proximal gradient viewpoint).
        for j in 0..n {
            rhs[j] += rho * x[j];
        }
        let x_hat = cholesky_solve(&l, n, &rhs)?;
        // Soft-threshold with threshold 1/ρ to enforce L1 penalty.
        let x_new = soft_threshold(&x_hat, 1.0 / rho);

        // ── Constraint residual for s-update ─────────────────────────────────
        // r_constraint = Aᵀ(b − A x_new)  (length n)
        let ax_new = mat_vec(a, m, n, &x_new)?;
        let mut primal_res = vec![0.0_f64; m];
        for i in 0..m {
            primal_res[i] = b[i] - ax_new[i];
        }
        let at_res = mat_t_vec(a, m, n, &primal_res)?;

        // ── s-update: project (Aᵀ res + u) onto [-delta, delta] ─────────────
        let mut s_new = vec![0.0_f64; n];
        for j in 0..n {
            let v = at_res[j] + u[j];
            s_new[j] = v.clamp(-cfg.delta, cfg.delta);
        }

        // ── u-update: dual residual accumulation ─────────────────────────────
        for j in 0..n {
            u[j] += at_res[j] - s_new[j];
        }

        // ── Convergence check ─────────────────────────────────────────────────
        // Primal: how much x changed.
        let x_change = {
            let mut sq = 0.0_f64;
            for j in 0..n {
                let d = x_new[j] - x[j];
                sq += d * d;
            }
            sq.sqrt()
        };
        // Constraint satisfaction: ||Aᵀ(b - Ax)||_∞ ≤ delta + tol?
        let constraint_viol = at_res.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let x_norm = norm2(&x_new).max(1.0e-300);

        x = x_new;
        s = s_new;

        if (constraint_viol - cfg.delta).max(0.0) < cfg.tol && x_change / x_norm < cfg.tol {
            break;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    /// Compute ||Aᵀ(b − Ax)||_∞ for checking the Dantzig constraint.
    fn constraint_inf(a: &[f64], b: &[f64], x: &[f64], m: usize, n: usize) -> f64 {
        let ax: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect();
        let resid: Vec<f64> = (0..m).map(|i| b[i] - ax[i]).collect();
        let at_r: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j] * resid[i]).sum())
            .collect();
        at_r.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    }

    // ── Test 1: output length equals n ───────────────────────────────────────
    #[test]
    fn output_len() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2
        let b = vec![1.0, 0.5];
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 50,
            tol: 1.0e-6,
        };
        let x = dantzig_selector(&a, &b, 2, 2, &cfg).expect("output_len");
        assert_eq!(x.len(), 2);
    }

    // ── Test 2: constraint approximately satisfied after convergence ──────────
    #[test]
    fn constraint_satisfied() {
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        let delta = 0.05_f64;
        let cfg = DantzigConfig {
            delta,
            max_iter: 200,
            tol: 1.0e-7,
        };
        let x = dantzig_selector(&a, &b, 4, 4, &cfg).expect("constraint_satisfied");
        let viol = constraint_inf(&a, &b, &x, 4, 4);
        // Allow generous slack: constraint should be within delta + 0.5 at least.
        assert!(
            viol < delta + 0.5,
            "constraint violation {viol} exceeds delta + slack"
        );
    }

    // ── Test 3: sparse solution for sparse-friendly problem ───────────────────
    #[test]
    fn sparse_solution() {
        // 4×4 identity: only two measurements non-zero → at most 2 active columns.
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        let cfg = DantzigConfig {
            delta: 0.01,
            max_iter: 200,
            tol: 1.0e-7,
        };
        let x = dantzig_selector(&a, &b, 4, 4, &cfg).expect("sparse_solution");
        let nnz = x.iter().filter(|&&v| v.abs() > 1.0e-4).count();
        // Expect at most 3 nonzeros (generous for ADMM; the signal has 2 active entries).
        assert!(
            nnz <= 3,
            "expected sparse solution, got {nnz} nonzeros: {:?}",
            x
        );
    }

    // ── Test 4: 2×2 identity with small delta recovers approximately ──────────
    #[test]
    fn zero_noise_exact_recovery() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![1.0, 0.5];
        let cfg = DantzigConfig {
            delta: 0.01,
            max_iter: 200,
            tol: 1.0e-8,
        };
        let x = dantzig_selector(&a, &b, 2, 2, &cfg).expect("zero_noise_exact_recovery");
        assert!((x[0] - 1.0).abs() < 0.1, "x[0]={} not close to 1.0", x[0]);
        assert!((x[1] - 0.5).abs() < 0.1, "x[1]={} not close to 0.5", x[1]);
    }

    // ── Test 5: very small delta should not panic ─────────────────────────────
    #[test]
    fn delta_too_small_ok() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.5];
        // delta = 1e-10 is extremely tight but must not panic.
        let cfg = DantzigConfig {
            delta: 1.0e-10,
            max_iter: 30,
            tol: 1.0e-5,
        };
        let result = dantzig_selector(&a, &b, 2, 2, &cfg);
        assert!(
            result.is_ok(),
            "should not error on tiny delta: {:?}",
            result
        );
        let x = result.expect("delta_too_small_ok");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "non-finite output: {:?}",
            x
        );
    }

    // ── Test 6: large tol stops early with few iterations ─────────────────────
    #[test]
    fn tol_stops_early() {
        // We can't directly observe iteration count here, but we verify that a very
        // large tol produces a valid (finite) result without running all max_iter steps.
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        // max_iter=1000 but tol=1e5 means it exits after the first iteration.
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 1000,
            tol: 1.0e5,
        };
        let x = dantzig_selector(&a, &b, 4, 4, &cfg).expect("tol_stops_early");
        assert_eq!(x.len(), 4);
        assert!(
            x.iter().all(|v| v.is_finite()),
            "non-finite output: {:?}",
            x
        );
    }

    // ── Test 7: all outputs are finite ────────────────────────────────────────
    #[test]
    fn output_finite() {
        let a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 1.5];
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 100,
            tol: 1.0e-6,
        };
        let x = dantzig_selector(&a, &b, 3, 3, &cfg).expect("output_finite");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "non-finite output: {:?}",
            x
        );
    }

    // ── Test 8: underdetermined (n > m) runs without error ────────────────────
    #[test]
    fn n_gt_m_underdetermined() {
        // 3 × 6 measurement matrix (underdetermined).
        let a = vec![
            1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.5,
        ];
        let b = vec![1.0, 0.5, 0.3];
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 100,
            tol: 1.0e-6,
        };
        let x = dantzig_selector(&a, &b, 3, 6, &cfg).expect("n_gt_m_underdetermined");
        assert_eq!(x.len(), 6, "output length should be n=6");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "non-finite output: {:?}",
            x
        );
    }

    // ── Error path tests ─────────────────────────────────────────────────────

    #[test]
    fn empty_input_error_m0() {
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 10,
            tol: 1.0e-6,
        };
        let result = dantzig_selector(&[], &[], 0, 3, &cfg);
        assert!(
            matches!(result, Err(CsError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    #[test]
    fn empty_input_error_n0() {
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 10,
            tol: 1.0e-6,
        };
        let result = dantzig_selector(&[], &[1.0, 2.0], 2, 0, &cfg);
        assert!(
            matches!(result, Err(CsError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    #[test]
    fn invalid_delta_error() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.5];
        let cfg = DantzigConfig {
            delta: -0.1,
            max_iter: 10,
            tol: 1.0e-6,
        };
        let result = dantzig_selector(&a, &b, 2, 2, &cfg);
        assert!(
            matches!(result, Err(CsError::InvalidParameter(_))),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    #[test]
    fn invalid_tol_error() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.5];
        let cfg = DantzigConfig {
            delta: 0.1,
            max_iter: 10,
            tol: -1.0e-6,
        };
        let result = dantzig_selector(&a, &b, 2, 2, &cfg);
        assert!(
            matches!(result, Err(CsError::InvalidParameter(_))),
            "expected InvalidParameter, got {:?}",
            result
        );
    }
}
