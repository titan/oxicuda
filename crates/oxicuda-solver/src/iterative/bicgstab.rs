//! Dense BiCGSTAB — Biconjugate Gradient Stabilized method.
//!
//! Solves a general nonsymmetric linear system `A x = b` for a *dense*
//! row-major matrix `A ∈ ℝ^{n×n}`. BiCGSTAB is a Krylov subspace method that
//! smooths the irregular convergence of plain BiCG by combining each BiCG step
//! with a one-dimensional GMRES(1) (steepest-descent) minimisation, giving
//! smoother, often faster convergence without the growing storage of GMRES.
//!
//! # Algorithm (van der Vorst 1992)
//!
//! With shadow residual `r̂₀ = r₀` the iteration maintains scalars
//! `ρ, α, ω` and search directions `p, s, v, t`:
//!
//! ```text
//! ρ_i      = ⟨r̂₀, r⟩
//! β        = (ρ_i / ρ_{i-1}) · (α / ω)
//! p        = r + β (p − ω v)
//! v        = A p
//! α        = ρ_i / ⟨r̂₀, v⟩
//! s        = r − α v
//! t        = A s
//! ω        = ⟨t, s⟩ / ⟨t, t⟩
//! x        = x + α p + ω s
//! r        = s − ω t
//! ```
//!
//! Breakdowns (`ρ → 0` or `ω → 0`) are detected and reported via the result's
//! `converged` flag rather than panicking.
//!
//! # Reference
//! - van der Vorst, H. A. (1992) "Bi-CGSTAB: A Fast and Smoothly Converging
//!   Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems."
//!   SIAM J. Sci. Stat. Comput. 13(2), 631–644.

use crate::error::{SolverError, SolverResult};
use crate::iterative::gmres::{GmresResult, matvec};

/// Configuration for [`bicgstab`].
#[derive(Debug, Clone, Copy)]
pub struct BicgstabConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative residual `‖b − A x‖ / ‖b‖`.
    pub tol: f64,
}

impl Default for BicgstabConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
        }
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Solve `A x = b` with BiCGSTAB.
///
/// `a` is the dense row-major coefficient matrix (`n × n`), `b` the right-hand
/// side (length `n`). The initial guess is the zero vector. Reuses
/// [`GmresResult`] as the result type for symmetry with [`mod@crate::iterative::gmres`].
///
/// # Errors
///
/// * [`SolverError::DimensionMismatch`] if `n == 0` or the slice lengths do not
///   match `n`.
///
/// Breakdown and non-convergence are reported through
/// [`GmresResult::converged`], not as errors.
pub fn bicgstab(a: &[f64], b: &[f64], n: usize, cfg: &BicgstabConfig) -> SolverResult<GmresResult> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "bicgstab: n must be ≥ 1".into(),
        ));
    }
    if a.len() != n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "bicgstab: A has {} elements, expected n*n = {}",
            a.len(),
            n * n
        )));
    }
    if b.len() != n {
        return Err(SolverError::DimensionMismatch(format!(
            "bicgstab: b has {} elements, expected n = {n}",
            b.len()
        )));
    }

    let b_norm = norm2(b);
    let mut x = vec![0.0_f64; n];

    if b_norm == 0.0 {
        return Ok(GmresResult {
            x,
            iter: 0,
            residual: 0.0,
            converged: true,
        });
    }

    // r = b − A x  (x = 0 → r = b).
    let mut r = b.to_vec();
    let r_hat = r.clone(); // fixed shadow residual r̂₀
    let mut rho_prev = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut p = vec![0.0_f64; n];
    let mut v = vec![0.0_f64; n];

    let mut rel_residual = norm2(&r) / b_norm;
    let mut iter = 0usize;
    let mut converged = rel_residual <= cfg.tol;

    // Threshold under which a scalar is treated as a breakdown.
    let breakdown_eps = 1e-300_f64;

    while iter < cfg.max_iter && !converged {
        iter += 1;

        let rho = dot(&r_hat, &r);
        if rho.abs() < breakdown_eps {
            // BiCG breakdown: ⟨r̂₀, r⟩ ≈ 0. Stop (report current residual).
            break;
        }

        let beta = (rho / rho_prev) * (alpha / omega);
        // p = r + β (p − ω v).
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A p.
        v = matvec(a, &p, n);
        let r_hat_v = dot(&r_hat, &v);
        if r_hat_v.abs() < breakdown_eps {
            break;
        }
        alpha = rho / r_hat_v;

        // s = r − α v.
        let mut s = vec![0.0_f64; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        // Early exit: if ‖s‖ already small, x += α p is the solution. The
        // true residual is recomputed from `x` after the loop, so no need to
        // update the bookkeeping scalars here.
        let s_norm = norm2(&s);
        if s_norm / b_norm <= cfg.tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }

        // t = A s.
        let t = matvec(a, &s, n);
        let tt = dot(&t, &t);
        if tt < breakdown_eps {
            // t ≈ 0: take the BiCG-only update and stop.
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }
        omega = dot(&t, &s) / tt;
        if omega.abs() < breakdown_eps {
            // Stabiliser collapsed — take the BiCG-only update and stop.
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }

        // x = x + α p + ω s.
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }
        // r = s − ω t.
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }

        rel_residual = norm2(&r) / b_norm;
        converged = rel_residual <= cfg.tol;
        rho_prev = rho;
    }

    // Honest final residual.
    let ax = matvec(a, &x, n);
    let res_vec: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
    let true_rel = norm2(&res_vec) / b_norm;

    Ok(GmresResult {
        x,
        iter,
        residual: true_rel,
        converged: true_rel <= cfg.tol,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BicgstabConfig {
        BicgstabConfig {
            max_iter: 500,
            tol: 1e-10,
        }
    }

    #[test]
    fn identity_solves() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for (xi, bi) in res.x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn diagonal() {
        let n = 3;
        let a = vec![2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let b = vec![2.0, 4.0, 8.0];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for xi in &res.x {
            assert!((xi - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn known_system() {
        // [[4,1],[1,3]] x = [1,2] → x = [1/11, 7/11].
        let n = 2;
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 1.0 / 11.0).abs() < 1e-9);
        assert!((res.x[1] - 7.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn residual_decreases() {
        let n = 5;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if i == j {
                    (i as f64) + 6.0
                } else {
                    0.4 / ((i as f64 - j as f64).abs() + 1.0)
                }
            })
            .collect();
        let b = vec![1.0; n];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        // Initial relative residual is ‖b‖/‖b‖ = 1; converged should be << 1.
        assert!(res.residual < 1.0);
        assert!(res.converged, "residual={}", res.residual);
    }

    #[test]
    fn spd_converges() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 4.0 } else { 1.0 };
            }
        }
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        let ax = matvec(&a, &res.x, n);
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - bi).abs() < 1e-8);
        }
    }

    #[test]
    fn nonsymmetric_converges() {
        let n = 3;
        let a = vec![10.0, 2.0, 1.0, -1.0, 8.0, 3.0, 2.0, -2.0, 9.0];
        let b = vec![13.0, 10.0, 9.0];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged, "residual={}", res.residual);
    }

    #[test]
    fn max_iter_bound() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![1.0; n];
        let capped = BicgstabConfig {
            max_iter: 1,
            tol: 1e-15,
        };
        let res = bicgstab(&a, &b, n, &capped).expect("solve");
        assert!(res.iter <= 1);
    }

    #[test]
    fn breakdown_handled() {
        // Singular matrix: BiCGSTAB must not panic and should report failure
        // (or accidental convergence on the zero RHS-projection) gracefully.
        let n = 2;
        let a = vec![1.0, 1.0, 1.0, 1.0]; // rank-1, singular
        let b = vec![1.0, 2.0]; // inconsistent
        let res = bicgstab(&a, &b, n, &cfg()).expect("must not panic");
        // It will not converge to a true solution; just ensure finiteness.
        for xi in &res.x {
            assert!(xi.is_finite());
        }
    }

    #[test]
    fn n_0_error() {
        let err = bicgstab(&[], &[], 0, &cfg());
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn solution_finite() {
        let n = 6;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if i == j {
                    (i as f64) + 10.0
                } else {
                    ((i + j) as f64).sin() * 0.1
                }
            })
            .collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        for xi in &res.x {
            assert!(xi.is_finite());
        }
    }

    #[test]
    fn dim_mismatch_a_error() {
        let n = 3;
        let a = vec![1.0; 8]; // should be 9
        let b = vec![1.0; 3];
        let err = bicgstab(&a, &b, n, &cfg());
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn zero_rhs_trivial() {
        let n = 3;
        let a = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let b = vec![0.0; n];
        let res = bicgstab(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for xi in &res.x {
            assert_eq!(*xi, 0.0);
        }
    }
}
