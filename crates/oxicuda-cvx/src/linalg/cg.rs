//! Conjugate Gradient on dense SPD systems (plain and preconditioned).

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{dot, mat_vec, norm2};

// ── Plain (unpreconditioned) CG ───────────────────────────────────────────────

/// Internal CG driver that also tracks the number of iterations performed.
///
/// Returns `(solution, iterations)` where `iterations == 0` means the initial
/// guess already satisfied the tolerance.
fn cg_solve_impl(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<(Vec<f64>, usize)> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if b.len() != n || x0.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax0 = mat_vec(a, n, n, &x)?;
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax0[i]).collect();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r)?;
    let b_norm = norm2(b).max(1.0);
    if rs_old.sqrt() / b_norm < tol {
        return Ok((x, 0));
    }
    for iter in 0..max_iter {
        let ap = mat_vec(a, n, n, &p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(CvxError::NumericalInstability(
                "cg: zero p·A·p (A may not be SPD)".into(),
            ));
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r)?;
        if rs_new.sqrt() / b_norm < tol {
            return Ok((x, iter + 1));
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Err(CvxError::NotConverged {
        iter: max_iter,
        residual: rs_old.sqrt(),
    })
}

/// Hestenes-Stiefel CG.  Solves `A x = b` with SPD `A` (dense row-major).
/// Returns `x` of length n.
pub fn cg_solve(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>> {
    cg_solve_impl(a, n, b, x0, max_iter, tol).map(|(x, _)| x)
}

/// Like [`cg_solve`] but also returns the number of CG iterations performed.
///
/// An iteration count of `0` means the initial guess already satisfied the
/// tolerance.  Useful for benchmarking convergence rates against preconditioned
/// variants.
pub fn cg_solve_counted(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<(Vec<f64>, usize)> {
    cg_solve_impl(a, n, b, x0, max_iter, tol)
}

// ── Preconditioner trait and concrete implementations ─────────────────────────

/// Applies the approximate inverse `M⁻¹` to a residual vector in a PCG solve.
///
/// Implementors represent a symmetric positive-definite preconditioner `M ≈ A`.
/// The only required operation is `z = M⁻¹ r`; the result must have the same
/// length as `r`.
///
/// A blanket implementation is provided for any `Fn(&[f64]) -> Vec<f64>` closure,
/// so anonymous functions can be passed directly to [`pcg_solve`].
pub trait Preconditioner {
    /// Apply `M⁻¹` to `r`, returning `z = M⁻¹ r`.
    fn apply_m_inv(&self, r: &[f64]) -> Vec<f64>;
}

/// Blanket implementation: any `Fn(&[f64]) -> Vec<f64>` is a valid preconditioner.
///
/// This lets callers pass an anonymous closure directly:
///
/// ```rust,ignore
/// let d_inv = vec![1.0, 0.5, 0.25];
/// pcg_solve(&a, 3, &b, &x0, 50, 1e-10,
///     &|r: &[f64]| r.iter().zip(d_inv.iter()).map(|(ri, di)| ri * di).collect::<Vec<_>>())
/// ```
impl<F: Fn(&[f64]) -> Vec<f64>> Preconditioner for F {
    #[inline]
    fn apply_m_inv(&self, r: &[f64]) -> Vec<f64> {
        self(r)
    }
}

/// Identity preconditioner `M = I` (no preconditioning).
///
/// When used with [`pcg_solve`], the PCG iterates are algebraically — and in
/// IEEE 754 arithmetic, numerically — identical to those of [`cg_solve`].
/// The substitution `z = M⁻¹ r = r` reduces every PCG recurrence to its
/// standard unpreconditioned form.
pub struct IdentityPrecond;

impl Preconditioner for IdentityPrecond {
    #[inline]
    fn apply_m_inv(&self, r: &[f64]) -> Vec<f64> {
        r.to_vec()
    }
}

/// Jacobi (diagonal) preconditioner `M = diag(A)`.
///
/// Applies `z_i = r_i / a_{ii}`, scaling each component by the reciprocal of
/// the corresponding diagonal entry.  For a diagonally-scaled SPD system
/// (large variation in `a_{ii}`) this dramatically reduces the condition number:
/// if `A = diag(λ₁, …, λₙ)` then `M⁻¹A = I` so PCG converges in **one** step.
///
/// Construct via [`JacobiPrecond::new`]; an error is returned if any diagonal
/// entry is zero (which would violate SPD).
pub struct JacobiPrecond {
    d_inv: Vec<f64>,
}

impl JacobiPrecond {
    /// Build the Jacobi preconditioner from the diagonal of the row-major `n × n`
    /// SPD matrix `a`.
    ///
    /// # Errors
    /// Returns [`CvxError::SingularMatrix`] if any diagonal entry has absolute
    /// value below `1e-300`.
    pub fn new(a: &[f64], n: usize) -> CvxResult<Self> {
        if a.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![a.len()],
            });
        }
        let d_inv: CvxResult<Vec<f64>> = (0..n)
            .map(|i| {
                let d = a[i * n + i];
                if d.abs() < 1.0e-300 {
                    Err(CvxError::SingularMatrix(format!(
                        "Jacobi preconditioner: zero diagonal at index {i}"
                    )))
                } else {
                    Ok(1.0 / d)
                }
            })
            .collect();
        Ok(Self { d_inv: d_inv? })
    }
}

impl Preconditioner for JacobiPrecond {
    #[inline]
    fn apply_m_inv(&self, r: &[f64]) -> Vec<f64> {
        r.iter()
            .zip(self.d_inv.iter())
            .map(|(ri, di)| ri * di)
            .collect()
    }
}

// ── Preconditioned CG ─────────────────────────────────────────────────────────

/// Internal PCG driver returning `(solution, iterations_used)`.
///
/// The standard PCG recurrence (Hestenes-Stiefel variant) is:
/// ```text
/// r₀  = b − A x₀
/// z₀  = M⁻¹ r₀,  p₀ = z₀,  ρ₀ = r₀ᵀ z₀
/// for k = 0, 1, …:
///   αₖ    = ρₖ / (pₖᵀ A pₖ)
///   xₖ₊₁ = xₖ + αₖ pₖ
///   rₖ₊₁ = rₖ − αₖ A pₖ
///   zₖ₊₁ = M⁻¹ rₖ₊₁
///   ρₖ₊₁ = rₖ₊₁ᵀ zₖ₊₁
///   βₖ   = ρₖ₊₁ / ρₖ
///   pₖ₊₁ = zₖ₊₁ + βₖ pₖ
/// ```
/// Convergence is declared when `‖rₖ‖₂ / max(‖b‖₂, 1) < tol`, matching
/// the criterion used by [`cg_solve`].
fn pcg_solve_impl<P: Preconditioner>(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
    precond: &P,
) -> CvxResult<(Vec<f64>, usize)> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if b.len() != n || x0.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax0 = mat_vec(a, n, n, &x)?;
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax0[i]).collect();
    let b_norm = norm2(b).max(1.0);

    // Convergence is measured in the Euclidean norm of r, consistent with cg_solve.
    let mut r_sq = dot(&r, &r)?;
    if r_sq.sqrt() / b_norm < tol {
        return Ok((x, 0));
    }

    // z₀ = M⁻¹ r₀;  p₀ = z₀;  ρ₀ = r₀ᵀ z₀.
    let mut z = precond.apply_m_inv(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z)?;

    for iter in 0..max_iter {
        // αₖ = ρₖ / (pₖᵀ A pₖ)
        let ap = mat_vec(a, n, n, &p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(CvxError::NumericalInstability(
                "pcg: zero p·A·p (A may not be SPD or preconditioner is indefinite)".into(),
            ));
        }
        let alpha = rz / pap;

        // xₖ₊₁ = xₖ + αₖ pₖ;  rₖ₊₁ = rₖ − αₖ A pₖ
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        // Euclidean convergence test (same metric as cg_solve).
        r_sq = dot(&r, &r)?;
        if r_sq.sqrt() / b_norm < tol {
            return Ok((x, iter + 1));
        }

        // zₖ₊₁ = M⁻¹ rₖ₊₁;  ρₖ₊₁ = rₖ₊₁ᵀ zₖ₊₁;  βₖ = ρₖ₊₁ / ρₖ
        z = precond.apply_m_inv(&r);
        let rz_new = dot(&r, &z)?;
        let beta = rz_new / rz;

        // pₖ₊₁ = zₖ₊₁ + βₖ pₖ
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    Err(CvxError::NotConverged {
        iter: max_iter,
        residual: r_sq.sqrt(),
    })
}

/// Preconditioned Conjugate Gradient (PCG).  Solves `A x = b` with SPD `A`
/// (dense, row-major) using a symmetric positive-definite preconditioner `M ≈ A`.
///
/// See `pcg_solve_impl` for the full PCG recurrence.  When `precond` is
/// [`IdentityPrecond`], the iterates are numerically identical to those of
/// [`cg_solve`] in IEEE 754 arithmetic.
///
/// # Preconditioner choices
///
/// | Type | Description |
/// |---|---|
/// | [`IdentityPrecond`] | No preconditioning (PCG ≡ CG) |
/// | [`JacobiPrecond`] | Diagonal scaling; ideal for diagonally-dominant systems |
/// | `Fn(&[f64]) -> Vec<f64>` | Custom closure (blanket impl) |
///
/// # Errors
/// [`CvxError::NotConverged`] if `max_iter` is exhausted without reaching `tol`.
/// [`CvxError::NumericalInstability`] if `p·Ap ≈ 0` (non-SPD matrix or degenerate
/// preconditioner).
pub fn pcg_solve<P: Preconditioner>(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
    precond: &P,
) -> CvxResult<Vec<f64>> {
    pcg_solve_impl(a, n, b, x0, max_iter, tol, precond).map(|(x, _)| x)
}

/// Like [`pcg_solve`] but also returns the number of PCG iterations performed.
///
/// Each iteration corresponds to one matrix-vector product `A p`.  A count of `0`
/// means the initial guess already satisfied the tolerance.  Useful for comparing
/// convergence rates between preconditioners.
pub fn pcg_solve_counted<P: Preconditioner>(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
    precond: &P,
) -> CvxResult<(Vec<f64>, usize)> {
    pcg_solve_impl(a, n, b, x0, max_iter, tol, precond)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::solve_dense;

    // ── Existing CG tests (unchanged) ─────────────────────────────────────────

    #[test]
    fn cg_identity_solve() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0, 5.0];
        let x = cg_solve(&a, 3, &b, &[0.0; 3], 50, 1.0e-12).expect("ok");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cg_spd_solve() {
        // A = [[4, 1], [1, 3]]; b = [1, 2]; x = [1/11, 7/11].
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = cg_solve(&a, 2, &b, &[0.0, 0.0], 50, 1.0e-12).expect("ok");
        assert!((x[0] - 1.0 / 11.0).abs() < 1.0e-9);
        assert!((x[1] - 7.0 / 11.0).abs() < 1.0e-9);
    }

    // ── PCG test 1: Identity preconditioner collapses PCG to CG ──────────────

    /// When M = I, the PCG recurrence reduces algebraically to the unpreconditioned
    /// CG recurrence.  In IEEE 754 arithmetic the two algorithms compute the same
    /// multiplications in the same order, so the solutions agree to within 1e-12.
    #[test]
    fn pcg_identity_precond_matches_cg() {
        // 4×4 tridiagonal SPD: diag 4, off-diag −1.
        #[rustfmt::skip]
        let a = vec![
             4.0, -1.0,  0.0,  0.0,
            -1.0,  4.0, -1.0,  0.0,
             0.0, -1.0,  4.0, -1.0,
             0.0,  0.0, -1.0,  4.0,
        ];
        let n = 4_usize;
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x0 = vec![0.0_f64; n];
        let tol = 1.0e-13;
        let max_iter = 200_usize;

        let x_cg = cg_solve(&a, n, &b, &x0, max_iter, tol).expect("cg");
        let x_pcg =
            pcg_solve(&a, n, &b, &x0, max_iter, tol, &IdentityPrecond).expect("pcg_identity");

        for (i, (cg_i, pcg_i)) in x_cg.iter().zip(x_pcg.iter()).enumerate() {
            assert!(
                (cg_i - pcg_i).abs() < 1.0e-12,
                "component {i}: cg={cg_i:.16e}  pcg-identity={pcg_i:.16e}"
            );
        }
    }

    // ── PCG test 2: Accuracy vs. direct dense solver ──────────────────────────

    /// On several small SPD systems the PCG solution must agree with the direct
    /// dense LU solver to within 1e-8, regardless of which preconditioner is used.
    #[test]
    fn pcg_reference_identity() {
        // System 1: 2×2  A = [[4, 1], [1, 3]], b = [1, 2].  Jacobi precond.
        {
            let a = vec![4.0, 1.0, 1.0, 3.0];
            let b = vec![1.0, 2.0];
            let n = 2_usize;
            let x_ref = solve_dense(&a, n, &b).expect("lu ref s1");
            let jac = JacobiPrecond::new(&a, n).expect("jacobi s1");
            let x_pcg = pcg_solve(&a, n, &b, &vec![0.0; n], 50, 1.0e-12, &jac).expect("pcg s1");
            for (i, (r, p)) in x_ref.iter().zip(x_pcg.iter()).enumerate() {
                assert!((r - p).abs() < 1.0e-8, "s1[{i}] ref={r:.10e} pcg={p:.10e}");
            }
        }

        // System 2: 3×3 tridiagonal, b = [0.5, 0.5, 0.5].  Identity precond.
        {
            #[rustfmt::skip]
            let a = vec![
                 2.0, -1.0,  0.0,
                -1.0,  2.0, -1.0,
                 0.0, -1.0,  2.0,
            ];
            let b = vec![0.5, 0.5, 0.5];
            let n = 3_usize;
            let x_ref = solve_dense(&a, n, &b).expect("lu ref s2");
            let x_pcg =
                pcg_solve(&a, n, &b, &vec![0.0; n], 50, 1.0e-12, &IdentityPrecond).expect("pcg s2");
            for (i, (r, p)) in x_ref.iter().zip(x_pcg.iter()).enumerate() {
                assert!((r - p).abs() < 1.0e-8, "s2[{i}] ref={r:.10e} pcg={p:.10e}");
            }
        }

        // System 3: 4×4 banded SPD, b = [1, 2, 3, 4].  Closure-based precond.
        {
            #[rustfmt::skip]
            let a = vec![
                 5.0, 1.0, 0.0, 0.0,
                 1.0, 5.0, 1.0, 0.0,
                 0.0, 1.0, 5.0, 1.0,
                 0.0, 0.0, 1.0, 5.0,
            ];
            let b = vec![1.0, 2.0, 3.0, 4.0];
            let n = 4_usize;
            // Build diagonal-inverse vector for a closure-based Jacobi preconditioner.
            let d_inv: Vec<f64> = (0..n).map(|i| 1.0 / a[i * n + i]).collect();
            let x_ref = solve_dense(&a, n, &b).expect("lu ref s3");
            let precond_fn = move |r: &[f64]| -> Vec<f64> {
                r.iter().zip(d_inv.iter()).map(|(ri, di)| ri * di).collect()
            };
            let x_pcg = pcg_solve(&a, n, &b, &vec![0.0; n], 100, 1.0e-12, &precond_fn)
                .expect("pcg s3 closure");
            for (i, (r, p)) in x_ref.iter().zip(x_pcg.iter()).enumerate() {
                assert!((r - p).abs() < 1.0e-8, "s3[{i}] ref={r:.10e} pcg={p:.10e}");
            }
        }
    }

    // ── PCG test 3: Jacobi preconditioner reduces iteration count ─────────────

    /// On a deliberately ill-conditioned diagonal SPD matrix the Jacobi
    /// preconditioner must deliver STRICTLY FEWER iterations than unpreconditioned CG.
    ///
    /// Matrix: A = diag(1, 10, 100, 1000, 10000) → κ(A) = 10 000.
    ///
    /// Analytical proof that Jacobi PCG converges in EXACTLY 1 iteration:
    ///   M = diag(A), so M⁻¹A = I.
    ///   z₀ = M⁻¹b = [1, 0.1, 0.01, 0.001, 0.0001] = x*.
    ///   p₀ = z₀.   Ap₀ = A x* = b.   α = (r₀·z₀)/(p₀·Ap₀) = (b·x*)/(x*·b) = 1.
    ///   x₁ = 0 + 1·z₀ = x*.  r₁ = b − A x* = 0.  Done.
    ///
    /// Unpreconditioned CG on a 5-eigenvalue system with all-ones RHS needs
    /// ≥ 2 iterations (CG cannot solve in 1 step when α ≠ 1 in iteration 0).
    #[test]
    fn pcg_jacobi_speedup() {
        const N: usize = 5;
        let diag = [1.0_f64, 10.0, 100.0, 1_000.0, 10_000.0];
        let mut a = vec![0.0_f64; N * N];
        for i in 0..N {
            a[i * N + i] = diag[i];
        }
        let b = vec![1.0_f64; N];
        let x0 = vec![0.0_f64; N];
        let tol = 1.0e-10;
        let max_iter = 1_000_usize;

        // Unpreconditioned CG.
        let (x_cg, cg_iters) = cg_solve_counted(&a, N, &b, &x0, max_iter, tol).expect("cg counted");

        // Jacobi-preconditioned PCG.
        let jac = JacobiPrecond::new(&a, N).expect("jacobi build");
        let (x_pcg, pcg_iters) =
            pcg_solve_counted(&a, N, &b, &x0, max_iter, tol, &jac).expect("pcg counted");

        // True solution: x_i = 1 / diag[i].
        let x_true: Vec<f64> = diag.iter().map(|d| 1.0 / d).collect();

        // Both must recover the true solution within 1e-8.
        for i in 0..N {
            assert!(
                (x_cg[i] - x_true[i]).abs() < 1.0e-8,
                "CG  x[{i}]={:.10e}  expected {:.10e}",
                x_cg[i],
                x_true[i]
            );
            assert!(
                (x_pcg[i] - x_true[i]).abs() < 1.0e-8,
                "PCG x[{i}]={:.10e}  expected {:.10e}",
                x_pcg[i],
                x_true[i]
            );
        }

        // Core property: Jacobi PCG uses strictly fewer iterations than plain CG.
        assert!(
            pcg_iters < cg_iters,
            "Jacobi PCG ({pcg_iters} iters) must beat unpreconditioned CG ({cg_iters} iters) \
             on κ=10000 diagonal system"
        );
    }
}
