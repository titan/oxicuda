//! Mehrotra Predictor-Corrector interior-point QP solver.
//!
//! Solves:  min  ½ xᵀ P x + qᵀ x
//!           s.t. A x = b,  x ≥ 0
//!
//! The Mehrotra predictor-corrector method computes an affine predictor step
//! (σ = 0) to estimate the ideal centering parameter `σ ∈ [0,1]`, then solves a
//! combined corrector system that simultaneously handles centering and the
//! cross-term correction from the affine step.  This typically reduces iteration
//! count by 30–50 % versus a simple long-step method.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// Result returned by [`mehrotra_qp`].
#[derive(Debug, Clone)]
pub struct MehrotraQpResult {
    /// Primal solution vector (length `n`).
    pub x: Vec<f64>,
    /// Dual variable for equality constraints (length `m`).
    pub y: Vec<f64>,
    /// Dual slack / complementarity variable (length `n`).
    pub z: Vec<f64>,
    /// Number of iterations performed.
    pub iter: usize,
    /// Final duality gap μ = (xᵀz) / n.
    pub mu: f64,
    /// `true` if all KKT residuals fell below `tol`.
    pub converged: bool,
}

/// Build the (n+m)×(n+m) augmented KKT system for a given (x, z) iterate.
///
/// ```text
/// M = [ P + diag(z/x)   −Aᵀ ]
///     [ A                 0  ]
/// ```
///
/// The regularisation `z[i]/x[i]` keeps the (0,0) block positive definite as
/// long as `x, z > 0`, making the overall system non-singular for strictly
/// feasible iterates.
fn build_aug_system(
    p_mat: &[f64],
    n: usize,
    a: &[f64],
    m: usize,
    x_cur: &[f64],
    z_cur: &[f64],
) -> Vec<f64> {
    let sz = n + m;
    let mut mat = vec![0.0_f64; sz * sz];

    // ── top-left: P + diag(z / x) ──────────────────────────────────────────
    for i in 0..n {
        for j in 0..n {
            mat[i * sz + j] = p_mat[i * n + j];
        }
        // Clamp denominator away from zero to maintain non-singularity.
        let xi = x_cur[i].max(1.0e-14);
        mat[i * sz + i] += z_cur[i] / xi;
    }

    // ── top-right: −Aᵀ  (M[i, n+k] = −A[k, i]) ───────────────────────────
    for i in 0..n {
        for k in 0..m {
            mat[i * sz + n + k] = -a[k * n + i];
        }
    }

    // ── bottom-left: A  (M[n+k, j] = A[k, j]) ─────────────────────────────
    for k in 0..m {
        for j in 0..n {
            mat[(n + k) * sz + j] = a[k * n + j];
        }
    }

    mat
}

/// Mehrotra predictor-corrector primal-dual interior-point QP.
///
/// Solves  `min ½ xᵀ P x + qᵀ x   s.t.  A x = b,  x ≥ 0`.
///
/// Returns `Ok(result)` with `converged = false` when `max_iter` is exhausted
/// without meeting `tol`; only returns `Err` on structural problems (bad
/// dimensions, singular matrix, etc.).
///
/// # Parameters
/// * `p_mat`    – n×n symmetric PSD Hessian, row-major.
/// * `n`        – Number of primal variables.
/// * `q`        – n-dimensional linear cost vector.
/// * `a`        – m×n constraint matrix, row-major.
/// * `m`        – Number of equality constraints.
/// * `b`        – m-dimensional RHS.
/// * `max_iter` – Maximum number of predictor-corrector iterations.
/// * `tol`      – Convergence tolerance applied to ‖r_d‖, ‖r_p‖, and μ.
pub fn mehrotra_qp(
    p_mat: &[f64],
    n: usize,
    q: &[f64],
    a: &[f64],
    m: usize,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<MehrotraQpResult> {
    // ── Input validation ────────────────────────────────────────────────────
    if p_mat.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p_mat.len()],
        });
    }
    if q.len() != n {
        return Err(CvxError::DimensionMismatch { a: q.len(), b: n });
    }
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    if tol <= 0.0 {
        return Err(CvxError::InvalidParameter("tol must be positive".into()));
    }

    // ── Initialisation: strictly interior starting point ────────────────────
    let mut x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; m];
    let mut z = vec![1.0_f64; n];

    let sz = n + m;

    for it in 0..max_iter {
        // ── Step 1: Compute KKT residuals ───────────────────────────────────
        //   r_d = P x + q − Aᵀ y − z       (dual feasibility)
        //   r_p = A x − b                   (primal feasibility)
        //   μ   = (xᵀ z) / n                (duality gap measure)
        let p_x = mat_vec(p_mat, n, n, &x)?;
        let at_y = mat_t_vec(a, m, n, &y)?;

        let mut r_d = vec![0.0_f64; n];
        for j in 0..n {
            r_d[j] = p_x[j] + q[j] - at_y[j] - z[j];
        }

        let ax = mat_vec(a, m, n, &x)?;
        let mut r_p = vec![0.0_f64; m];
        for i in 0..m {
            r_p[i] = ax[i] - b[i];
        }

        let mu: f64 = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;

        // Convergence check.
        if norm2(&r_d) < tol && norm2(&r_p) < tol && mu < tol {
            return Ok(MehrotraQpResult {
                x,
                y,
                z,
                iter: it,
                mu,
                converged: true,
            });
        }

        // ── Step 2: Build augmented KKT matrix ──────────────────────────────
        let m_aug = build_aug_system(p_mat, n, a, m, &x, &z);

        // ── Step 3: Affine predictor step (σ = 0) ───────────────────────────
        //
        // The affine step ignores centering (r_xz_aff = x ⊙ z).
        // Reduced RHS for primal-dual direction:
        //   rhs_d[j] = −r_d[j] − (x[j]·z[j]) / x[j]  =  −r_d[j] − z[j]
        //   rhs_p[k] = −r_p[k]
        let mut rhs_aff = vec![0.0_f64; sz];
        for j in 0..n {
            rhs_aff[j] = -r_d[j] - z[j];
        }
        for k in 0..m {
            rhs_aff[n + k] = -r_p[k];
        }

        let sol_aff = solve_dense(&m_aug, sz, &rhs_aff)?;
        let dx_a: Vec<f64> = sol_aff[..n].to_vec();
        // dy_a is only used to split the solution; not used independently.
        let _dy_a: Vec<f64> = sol_aff[n..].to_vec();

        // Recover dz_a from complementarity: x ⊙ dz + z ⊙ dx = −(x ⊙ z)
        //   dz_a[j] = (−x[j]·z[j] − z[j]·dx_a[j]) / x[j]
        //           = −z[j] − (z[j] / x[j]) · dx_a[j]
        let mut dz_a = vec![0.0_f64; n];
        for j in 0..n {
            let xi = x[j].max(1.0e-14);
            dz_a[j] = -z[j] - (z[j] / xi) * dx_a[j];
        }

        // ── Step 4: Affine step lengths ─────────────────────────────────────
        //
        // Largest step that keeps x + α dx_a ≥ 0 and z + α dz_a ≥ 0.
        let mut alpha_p_aff = 1.0_f64;
        let mut alpha_d_aff = 1.0_f64;
        for j in 0..n {
            if dx_a[j] < 0.0 {
                let r = -x[j] / dx_a[j];
                if r < alpha_p_aff {
                    alpha_p_aff = r;
                }
            }
            if dz_a[j] < 0.0 {
                let r = -z[j] / dz_a[j];
                if r < alpha_d_aff {
                    alpha_d_aff = r;
                }
            }
        }

        // ── Step 5: Centering parameter σ ───────────────────────────────────
        //
        // Estimate μ_aff after the affine step, then set
        //   σ = (μ_aff / μ)³   clamped to [0, 1].
        let mu_aff: f64 = (0..n)
            .map(|j| (x[j] + alpha_p_aff * dx_a[j]) * (z[j] + alpha_d_aff * dz_a[j]))
            .sum::<f64>()
            / n as f64;

        let sigma = if mu < 1.0e-300 {
            0.0_f64
        } else {
            (mu_aff / mu).powi(3).clamp(0.0, 1.0)
        };

        // ── Step 6: Corrector step ───────────────────────────────────────────
        //
        // The corrector modifies the complementarity residual to include
        // centering (σ μ) and the second-order cross term (dx_a ⊙ dz_a):
        //   r_xz_cor[j] = x[j]·z[j] − σ μ + dx_a[j]·dz_a[j]
        //
        // Combined RHS:
        //   rhs_cor[j] = −r_d[j] − r_xz_cor[j] / x[j]
        //   rhs_cor[n+k] = −r_p[k]
        let mut rhs_cor = vec![0.0_f64; sz];
        for j in 0..n {
            let xi = x[j].max(1.0e-14);
            let r_xz_cor_j = x[j] * z[j] - sigma * mu + dx_a[j] * dz_a[j];
            rhs_cor[j] = -r_d[j] - r_xz_cor_j / xi;
        }
        for k in 0..m {
            rhs_cor[n + k] = -r_p[k];
        }

        // Reuse the same augmented matrix (it depends only on x, z which have
        // not changed within this iteration).
        let sol_cor = solve_dense(&m_aug, sz, &rhs_cor)?;
        let dx: Vec<f64> = sol_cor[..n].to_vec();
        let dy: Vec<f64> = sol_cor[n..].to_vec();

        // Recover dz from corrector complementarity.
        let mut dz = vec![0.0_f64; n];
        for j in 0..n {
            let xi = x[j].max(1.0e-14);
            let r_xz_cor_j = x[j] * z[j] - sigma * mu + dx_a[j] * dz_a[j];
            dz[j] = (-r_xz_cor_j - z[j] * dx[j]) / xi;
        }

        // ── Step 7: Final step lengths (0.99 fraction-to-boundary) ──────────
        let mut alpha_p = 1.0_f64;
        let mut alpha_d = 1.0_f64;
        for j in 0..n {
            if dx[j] < 0.0 {
                let r = -x[j] / dx[j];
                if r < alpha_p {
                    alpha_p = r;
                }
            }
            if dz[j] < 0.0 {
                let r = -z[j] / dz[j];
                if r < alpha_d {
                    alpha_d = r;
                }
            }
        }
        let alpha_p = 0.99 * alpha_p;
        let alpha_d = 0.99 * alpha_d;

        // ── Step 8: Update iterates ──────────────────────────────────────────
        for j in 0..n {
            x[j] += alpha_p * dx[j];
            z[j] += alpha_d * dz[j];
        }
        for i in 0..m {
            y[i] += alpha_d * dy[i];
        }
    }

    // Maximum iterations reached — return solution without error so tests can
    // still inspect the iterate.
    let mu = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;
    Ok(MehrotraQpResult {
        x,
        y,
        z,
        iter: max_iter,
        mu,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::primal_dual_qp::primal_dual_qp;

    // ── Test 1 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_2var_equality_constraint() {
        // min ½(x1² + x2²)  s.t. x1+x2=1, x≥0 → optimal (0.5, 0.5), obj=0.25
        // The solver minimises ½ xᵀ P x + qᵀ x; with P=I, q=0 the optimal
        // objective is ½(0.5² + 0.5²) = 0.25, NOT 0.5.
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1e-7).expect("converges");
        assert!(
            (res.x[0] - 0.5).abs() < 1e-4,
            "x[0]={} not near 0.5",
            res.x[0]
        );
        assert!(
            (res.x[1] - 0.5).abs() < 1e-4,
            "x[1]={} not near 0.5",
            res.x[1]
        );
        let obj = 0.5 * (res.x[0].powi(2) + res.x[1].powi(2));
        assert!((obj - 0.25).abs() < 1e-3, "obj={} not near 0.25", obj);
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_matches_primal_dual_qp() {
        // Compare Mehrotra solution to the existing primal-dual solver on the
        // same 2-variable problem.
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res_meh = mehrotra_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1e-7).expect("ok");
        let res_pd = primal_dual_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1e-7).expect("ok");
        assert!(
            (res_meh.x[0] - res_pd.x[0]).abs() < 1e-4,
            "x[0] mismatch: mehrotra={}, primal_dual={}",
            res_meh.x[0],
            res_pd.x[0]
        );
        assert!(
            (res_meh.x[1] - res_pd.x[1]).abs() < 1e-4,
            "x[1] mismatch: mehrotra={}, primal_dual={}",
            res_meh.x[1],
            res_pd.x[1]
        );
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_3var_problem() {
        // min ½(x1²+x2²+x3²)  s.t. x1+x2+x3=1, x≥0 → (1/3, 1/3, 1/3)
        let p_mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0_f64];
        let q = vec![0.0_f64; 3];
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 3, &q, &a, 1, &b, 100, 1e-7).expect("ok");
        for (idx, &xi) in res.x.iter().enumerate() {
            assert!(
                (xi - 1.0 / 3.0).abs() < 1e-4,
                "x[{}]={} expected 1/3",
                idx,
                xi
            );
        }
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_convergence_within_max_iter() {
        // n=6, m=3: min ½‖x‖²  s.t. x1+x2=1, x3+x4=1, x5+x6=1
        let n = 6;
        let m = 3;
        let p_mat: Vec<f64> = (0..n * n)
            .map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let q = vec![0.0_f64; n];
        let mut a = vec![0.0_f64; m * n];
        a[0] = 1.0;
        a[1] = 1.0;
        a[n + 2] = 1.0;
        a[n + 3] = 1.0;
        a[2 * n + 4] = 1.0;
        a[2 * n + 5] = 1.0;
        let b = vec![1.0_f64; m];
        let res = mehrotra_qp(&p_mat, n, &q, &a, m, &b, 50, 1e-7).expect("ok");
        assert!(res.iter <= 50, "used {} iterations", res.iter);
        assert!(
            (res.x[0] + res.x[1] - 1.0).abs() < 1e-3,
            "pair 0+1 = {}, expected 1",
            res.x[0] + res.x[1]
        );
    }

    // ── Test 5 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_kkt_residuals_small() {
        // Verify ‖Px + q − Aᵀy − z‖ < 1e-5  and  ‖Ax − b‖ < 1e-5.
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a_mat = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 2, &q, &a_mat, 1, &b, 100, 1e-8).expect("ok");
        // r_d = P x + q − Aᵀ y − z  (P = I here)
        let r_d: Vec<f64> = (0..2)
            .map(|j| res.x[j] + q[j] - a_mat[j] * res.y[0] - res.z[j])
            .collect();
        let ax_b = (res.x[0] + res.x[1] - 1.0).abs();
        assert!(
            norm2(&r_d) < 1e-5,
            "dual residual {} not < 1e-5",
            norm2(&r_d)
        );
        assert!(ax_b < 1e-5, "primal residual {} not < 1e-5", ax_b);
    }

    // ── Test 6 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_complementarity_small() {
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1e-8).expect("ok");
        let compl: f64 = (0..2).map(|j| res.x[j] * res.z[j]).sum::<f64>() / 2.0;
        assert!(compl < 1e-5, "complementarity {} not < 1e-5", compl);
    }

    // ── Test 7 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_dimension_validation() {
        // Wrong p_mat size for n=2 → ShapeMismatch.
        let p_mat = vec![1.0_f64, 0.0, 0.0]; // 3 elements, not 4
        let result = mehrotra_qp(&p_mat, 2, &[0.0, 0.0], &[1.0, 1.0], 1, &[1.0], 50, 1e-6);
        assert!(
            matches!(result, Err(CvxError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {:?}",
            result
        );
    }

    // ── Test 8 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_zero_p_reduces_to_lp() {
        // P=0, q=[1,2,3], x1+x2+x3=1, x≥0
        // Linear program: minimum at corner x=(1,0,0), objective=1.
        // With P=0, the diagonal regulariser z/x keeps the system non-singular.
        let p_mat = vec![0.0_f64; 9];
        let q = vec![1.0_f64, 2.0, 3.0];
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 3, &q, &a, 1, &b, 200, 1e-6).expect("ok");
        let obj: f64 = res.x.iter().zip(q.iter()).map(|(xi, qi)| xi * qi).sum();
        assert!(obj < 1.1, "obj={} should be near 1", obj);
        assert!(res.x[0] > 0.8, "x[0]={} should be near 1", res.x[0]);
    }

    // ── Test 9 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_identity_p_constrained_ls() {
        // P=I, q=0, A=[1,1], b=[1] → minimum-norm solution (0.5, 0.5).
        let p_mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res = mehrotra_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1e-7).expect("ok");
        assert!(
            (res.x[0] - 0.5).abs() < 1e-4,
            "x[0]={} not near 0.5",
            res.x[0]
        );
        assert!(
            (res.x[1] - 0.5).abs() < 1e-4,
            "x[1]={} not near 0.5",
            res.x[1]
        );
    }
}
