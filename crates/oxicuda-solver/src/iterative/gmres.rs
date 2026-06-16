//! Dense restarted GMRES(m) — Generalized Minimal Residual method.
//!
//! Solves a general (possibly nonsymmetric, nonsingular) linear system
//! `A x = b` for a *dense* row-major matrix `A ∈ ℝ^{n×n}`.
//!
//! # Algorithm (Saad & Schultz 1986)
//!
//! GMRES minimises the residual `‖b − A x‖₂` over the affine Krylov subspace
//! `x₀ + K_m(A, r₀)` where `K_m = span{r₀, A r₀, …, A^{m-1} r₀}`. The method:
//!
//! 1. Builds an orthonormal basis `V_m` of `K_m` with the **Arnoldi** process
//!    (modified Gram–Schmidt), producing an upper-Hessenberg `H̄ ∈ ℝ^{(m+1)×m}`.
//! 2. Incrementally triangularises `H̄` with **Givens rotations**, so the
//!    least-squares problem `min_y ‖β e₁ − H̄ y‖₂` reduces to back-substitution
//!    on a small upper-triangular system.
//! 3. **Restarts** every `restart` steps using the current iterate `x` as the
//!    new initial guess, bounding memory to `O(n · restart)`.
//!
//! The running residual is the magnitude of the rotated right-hand side, so
//! convergence is detected without forming `A x` each inner step.
//!
//! # Reference
//! - Saad, Y. & Schultz, M. H. (1986) "GMRES: A Generalized Minimal Residual
//!   Algorithm for Solving Nonsymmetric Linear Systems." SIAM J. Sci. Stat.
//!   Comput. 7(3), 856–869.

use crate::error::{SolverError, SolverResult};

/// Configuration for [`gmres`].
#[derive(Debug, Clone, Copy)]
pub struct GmresConfig {
    /// Maximum total number of Arnoldi iterations across all restart cycles.
    pub max_iter: usize,
    /// Restart parameter `m`: Arnoldi steps before restarting.
    pub restart: usize,
    /// Convergence tolerance on the relative residual `‖b − A x‖ / ‖b‖`.
    pub tol: f64,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            tol: 1e-10,
        }
    }
}

/// Result of an iterative solve.
#[derive(Debug, Clone)]
pub struct GmresResult {
    /// Approximate solution vector, length `n`.
    pub x: Vec<f64>,
    /// Total number of inner iterations performed.
    pub iter: usize,
    /// Final relative residual `‖b − A x‖ / ‖b‖`.
    pub residual: f64,
    /// Whether the relative residual fell below `cfg.tol`.
    pub converged: bool,
}

/// Dense matrix-vector product `y = A x` for row-major `A ∈ ℝ^{n×n}`.
#[inline]
pub(crate) fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for (i, yi) in y.iter_mut().enumerate() {
        let row = &a[i * n..i * n + n];
        let mut acc = 0.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            acc += row[j] * xj;
        }
        *yi = acc;
    }
    y
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Solve `A x = b` with restarted GMRES(m).
///
/// `a` is the dense row-major coefficient matrix (`n × n`), `b` the right-hand
/// side (length `n`). The initial guess is the zero vector.
///
/// # Errors
///
/// * [`SolverError::DimensionMismatch`] if `n == 0` or the slice lengths do not
///   match `n`.
/// * [`SolverError::InternalError`] if `cfg.restart == 0`.
///
/// Non-convergence is *not* an error: inspect [`GmresResult::converged`].
pub fn gmres(a: &[f64], b: &[f64], n: usize, cfg: &GmresConfig) -> SolverResult<GmresResult> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "gmres: n must be ≥ 1".into(),
        ));
    }
    if a.len() != n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "gmres: A has {} elements, expected n*n = {}",
            a.len(),
            n * n
        )));
    }
    if b.len() != n {
        return Err(SolverError::DimensionMismatch(format!(
            "gmres: b has {} elements, expected n = {n}",
            b.len()
        )));
    }
    if cfg.restart == 0 {
        return Err(SolverError::InternalError(
            "gmres: restart parameter must be ≥ 1".into(),
        ));
    }

    let m = cfg.restart.min(n);
    let b_norm = norm2(b);
    let mut x = vec![0.0_f64; n];

    // Degenerate RHS: x = 0 is exact.
    if b_norm == 0.0 {
        return Ok(GmresResult {
            x,
            iter: 0,
            residual: 0.0,
            converged: true,
        });
    }

    let mut total_iter = 0usize;

    // Outer restart loop.
    'restart: while total_iter < cfg.max_iter {
        // r₀ = b − A x.
        let ax = matvec(a, &x, n);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
        let beta = norm2(&r);
        let mut rel_residual = beta / b_norm;
        if rel_residual <= cfg.tol {
            break;
        }

        // Krylov basis V (m+1 vectors of length n), Hessenberg H ((m+1) × m).
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        for ri in r.iter_mut() {
            *ri /= beta;
        }
        v.push(r);

        // Hessenberg stored column-major-ish as h[col][row]; we keep a dense
        // (m+1) × m buffer indexed h[col * (m+1) + row].
        let mut h = vec![0.0_f64; (m + 1) * m];
        // Givens rotation coefficients per column.
        let mut cs = vec![0.0_f64; m];
        let mut sn = vec![0.0_f64; m];
        // Rotated RHS g = β e₁.
        let mut g = vec![0.0_f64; m + 1];
        g[0] = beta;

        let mut k_used = 0usize; // number of completed Arnoldi columns

        for k in 0..m {
            if total_iter >= cfg.max_iter {
                break;
            }
            total_iter += 1;

            // w = A v_k.
            let mut w = matvec(a, &v[k], n);

            // Modified Gram–Schmidt against v_0..v_k.
            for (i, vi) in v.iter().enumerate().take(k + 1) {
                let hik = dot(&w, vi);
                h[k * (m + 1) + i] = hik;
                for (wj, &vij) in w.iter_mut().zip(vi.iter()) {
                    *wj -= hik * vij;
                }
            }
            let h_next = norm2(&w);
            h[k * (m + 1) + (k + 1)] = h_next;

            // New basis vector (guard against lucky/exact breakdown).
            if h_next > 1e-300 {
                for wj in w.iter_mut() {
                    *wj /= h_next;
                }
            }
            v.push(w);

            // Apply previous Givens rotations to column k of H.
            for i in 0..k {
                let temp = cs[i] * h[k * (m + 1) + i] + sn[i] * h[k * (m + 1) + (i + 1)];
                h[k * (m + 1) + (i + 1)] =
                    -sn[i] * h[k * (m + 1) + i] + cs[i] * h[k * (m + 1) + (i + 1)];
                h[k * (m + 1) + i] = temp;
            }

            // Compute and apply the new rotation eliminating H[k+1, k].
            let hk = h[k * (m + 1) + k];
            let hk1 = h[k * (m + 1) + (k + 1)];
            let denom = (hk * hk + hk1 * hk1).sqrt();
            if denom > 0.0 {
                cs[k] = hk / denom;
                sn[k] = hk1 / denom;
            } else {
                cs[k] = 1.0;
                sn[k] = 0.0;
            }
            h[k * (m + 1) + k] = cs[k] * hk + sn[k] * hk1;
            h[k * (m + 1) + (k + 1)] = 0.0;

            // Rotate the RHS; |g[k+1]| is the current residual norm.
            let g_k = g[k];
            g[k] = cs[k] * g_k;
            g[k + 1] = -sn[k] * g_k;

            k_used = k + 1;
            rel_residual = g[k + 1].abs() / b_norm;
            if rel_residual <= cfg.tol {
                break;
            }
        }

        // Back-substitution: solve upper-triangular H[0..k_used, 0..k_used] y = g.
        if k_used == 0 {
            // No progress possible (immediate breakdown) — stop.
            break 'restart;
        }
        let mut y = vec![0.0_f64; k_used];
        for i in (0..k_used).rev() {
            let mut acc = g[i];
            for j in (i + 1)..k_used {
                acc -= h[j * (m + 1) + i] * y[j];
            }
            let diag = h[i * (m + 1) + i];
            y[i] = if diag.abs() > 1e-300 { acc / diag } else { 0.0 };
        }

        // x += V_{k_used} y.
        for (j, &yj) in y.iter().enumerate() {
            for (xi, &vij) in x.iter_mut().zip(v[j].iter()) {
                *xi += yj * vij;
            }
        }

        if rel_residual <= cfg.tol {
            break;
        }
        // If we made a full cycle without converging, loop again (restart).
        if total_iter >= cfg.max_iter {
            break;
        }
    }

    // Recompute the true residual for an honest report (the rotated-RHS
    // estimate can drift from the actual residual after restarts).
    let ax = matvec(a, &x, n);
    let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
    let true_rel = norm2(&r) / b_norm;

    Ok(GmresResult {
        x,
        iter: total_iter,
        residual: true_rel,
        converged: true_rel <= cfg.tol,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GmresConfig {
        GmresConfig {
            max_iter: 500,
            restart: 30,
            tol: 1e-10,
        }
    }

    #[test]
    fn identity_solves_trivially() {
        // I x = b → x = b.
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for (xi, bi) in res.x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-9, "x={xi} expected {bi}");
        }
    }

    #[test]
    fn diagonal_system() {
        let n = 3;
        let a = vec![2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let b = vec![2.0, 4.0, 8.0];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for xi in &res.x {
            assert!((xi - 1.0).abs() < 1e-9, "expected 1.0, got {xi}");
        }
    }

    #[test]
    fn known_2x2() {
        // [[4,1],[1,3]] x = [1,2] → x = [1/11, 7/11].
        let n = 2;
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 1.0 / 11.0).abs() < 1e-9);
        assert!((res.x[1] - 7.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn residual_lt_tol() {
        let n = 5;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if i == j {
                    (i as f64) + 5.0
                } else {
                    0.3 / ((i as f64 - j as f64).abs() + 1.0)
                }
            })
            .collect();
        let b = vec![1.0; n];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(
            res.residual < cfg().tol,
            "residual {} too large",
            res.residual
        );
    }

    #[test]
    fn spd_system_converges() {
        // SPD: A = M^T M + I, well-conditioned.
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 4.0 } else { 1.0 };
            }
        }
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        // Cross-check residual.
        let ax = matvec(&a, &res.x, n);
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - bi).abs() < 1e-8);
        }
    }

    #[test]
    fn nonsymmetric_system() {
        // Strongly nonsymmetric but diagonally dominant.
        let n = 3;
        let a = vec![10.0, 2.0, 1.0, -1.0, 8.0, 3.0, 2.0, -2.0, 9.0];
        let b = vec![13.0, 10.0, 9.0];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged, "residual={}", res.residual);
        let ax = matvec(&a, &res.x, n);
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - bi).abs() < 1e-7);
        }
    }

    #[test]
    fn restart_respected() {
        // Tiny restart must still converge (more outer cycles).
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 5.0 } else { 0.5 };
            }
        }
        let b = vec![1.0; n];
        let small = GmresConfig {
            max_iter: 500,
            restart: 1,
            tol: 1e-10,
        };
        let res = gmres(&a, &b, n, &small).expect("solve");
        assert!(res.converged, "GMRES(1) failed, residual={}", res.residual);
    }

    #[test]
    fn max_iter_bound() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![1.0; n];
        let capped = GmresConfig {
            max_iter: 2,
            restart: 30,
            tol: 1e-15,
        };
        let res = gmres(&a, &b, n, &capped).expect("solve");
        assert!(res.iter <= 2, "iter {} exceeded max_iter", res.iter);
    }

    #[test]
    fn n_0_error() {
        let err = gmres(&[], &[], 0, &cfg());
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
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        for xi in &res.x {
            assert!(xi.is_finite(), "non-finite solution component {xi}");
        }
    }

    #[test]
    fn dim_mismatch_b_error() {
        let n = 3;
        let a = vec![1.0; n * n];
        let b = vec![1.0; 2]; // wrong length
        let err = gmres(&a, &b, n, &cfg());
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn restart_0_error() {
        let n = 2;
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 1.0];
        let bad = GmresConfig {
            max_iter: 10,
            restart: 0,
            tol: 1e-10,
        };
        let err = gmres(&a, &b, n, &bad);
        assert!(matches!(err, Err(SolverError::InternalError(_))));
    }

    #[test]
    fn zero_rhs_trivial() {
        let n = 3;
        let a = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let b = vec![0.0; n];
        let res = gmres(&a, &b, n, &cfg()).expect("solve");
        assert!(res.converged);
        for xi in &res.x {
            assert_eq!(*xi, 0.0);
        }
    }
}
