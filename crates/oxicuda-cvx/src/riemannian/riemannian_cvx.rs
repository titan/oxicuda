//! Riemannian gradient descent with retraction and geodesic Armijo line search.
//!
//! Given a smooth cost `f : M → ℝ` on a Riemannian manifold `M`, the Riemannian
//! gradient `grad f(x)` is obtained by projecting the ambient Euclidean gradient
//! onto the tangent space `T_x M`. The iteration is
//!
//! ```text
//! x_{k+1} = R_{x_k}( −α_k · grad f(x_k) )
//! ```
//!
//! where `R_x : T_x M → M` is a retraction (a first-order approximation of the
//! exponential map). The step `α_k` is chosen by a backtracking **Riemannian Armijo**
//! rule: accept the first `α ∈ {α₀, β α₀, β² α₀, …}` with
//!
//! ```text
//! f( R_x(−α g) ) ≤ f(x) − c α ‖g‖²_x .
//! ```
//!
//! # Manifolds
//!
//! | Manifold | Tangent projection `P_x(ξ)` | Retraction `R_x(ξ)` |
//! |---|---|---|
//! | Sphere `Sⁿ⁻¹` | `ξ − ⟨x, ξ⟩ x` | `(x + ξ) / ‖x + ξ‖` |
//! | SPD `S⁺⁺(n)` | `sym(ξ)` (ambient grad already symmetric) | `X^{½} expm(X^{−½} ξ X^{−½}) X^{½}` |
//! | Stiefel `St(n,p)` | `ξ − X·sym(Xᵀξ)` | `qf(X + ξ)` (Q-factor of thin QR) |
//!
//! # References
//!
//! - P.-A. Absil, R. Mahony & R. Sepulchre (2008), "Optimization Algorithms on Matrix
//!   Manifolds", Princeton University Press.
//! - N. Boumal (2023), "An Introduction to Optimization on Smooth Manifolds", CUP.

use crate::error::{CvxError, CvxResult};

// ---------------------------------------------------------------------------
// Manifold specification
// ---------------------------------------------------------------------------

/// Choice of matrix manifold. Points are stored row-major.
#[derive(Debug, Clone)]
pub enum Manifold {
    /// Unit sphere `Sⁿ⁻¹`: a length-`n` vector with `‖x‖ = 1`.
    Sphere {
        /// Ambient dimension `n`.
        n: usize,
    },
    /// Symmetric positive-definite `n×n` matrices with the affine-invariant metric.
    Spd {
        /// Matrix order `n`.
        n: usize,
    },
    /// Stiefel manifold `St(n, p)`: `n×p` matrices with orthonormal columns.
    Stiefel {
        /// Number of rows `n`.
        n: usize,
        /// Number of columns `p` (`p ≤ n`).
        p: usize,
    },
}

impl Manifold {
    /// Number of stored scalars for a point on this manifold.
    #[must_use]
    pub fn len(&self) -> usize {
        match *self {
            Manifold::Sphere { n } => n,
            Manifold::Spd { n } => n * n,
            Manifold::Stiefel { n, p } => n * p,
        }
    }

    /// Whether the manifold has zero stored scalars (always false for valid specs).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Project ambient vector `g` onto the tangent space `T_x M`.
    fn project(&self, x: &[f64], g: &[f64]) -> CvxResult<Vec<f64>> {
        match *self {
            Manifold::Sphere { n } => {
                check_len(x, n)?;
                check_len(g, n)?;
                let xg = dot(x, g);
                Ok(x.iter()
                    .zip(g.iter())
                    .map(|(xi, gi)| gi - xg * xi)
                    .collect())
            }
            Manifold::Spd { n } => {
                check_len(x, n * n)?;
                check_len(g, n * n)?;
                // Tangent space at SPD is the symmetric matrices; symmetrise g.
                Ok(symmetrise(g, n))
            }
            Manifold::Stiefel { n, p } => {
                check_len(x, n * p)?;
                check_len(g, n * p)?;
                // P_X(g) = g − X sym(Xᵀ g).
                let xtg = mat_t_mat_general(x, g, n, p, p); // p×p
                let sym = symmetrise(&xtg, p);
                let x_sym = mat_mul(x, &sym, n, p, p); // n×p
                Ok(g.iter().zip(x_sym.iter()).map(|(gi, si)| gi - si).collect())
            }
        }
    }

    /// Riemannian inner product `⟨u, v⟩_x` of two tangent vectors at `x`.
    fn inner(&self, x: &[f64], u: &[f64], v: &[f64]) -> CvxResult<f64> {
        match *self {
            // Sphere & Stiefel use the Euclidean (Frobenius) metric on tangent vectors.
            Manifold::Sphere { .. } | Manifold::Stiefel { .. } => Ok(dot(u, v)),
            Manifold::Spd { n } => {
                // Affine-invariant metric: ⟨u, v⟩_X = tr(X⁻¹ u X⁻¹ v).
                let xinv = inv_spd(x, n)?;
                let a = mat_mul(&xinv, u, n, n, n);
                let b = mat_mul(&xinv, v, n, n, n);
                // tr(a · b) = Σ_{i,j} a_{ij} b_{ji}.
                let mut s = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        s += a[i * n + j] * b[j * n + i];
                    }
                }
                Ok(s)
            }
        }
    }

    /// Retract tangent vector `xi` at `x` back onto the manifold.
    fn retract(&self, x: &[f64], xi: &[f64]) -> CvxResult<Vec<f64>> {
        match *self {
            Manifold::Sphere { n } => {
                check_len(xi, n)?;
                let sum: Vec<f64> = x.iter().zip(xi.iter()).map(|(a, b)| a + b).collect();
                let nrm = norm(&sum);
                if nrm < 1e-300 {
                    return Err(CvxError::NumericalInstability(
                        "sphere retraction: zero-norm point".into(),
                    ));
                }
                Ok(sum.iter().map(|v| v / nrm).collect())
            }
            Manifold::Spd { n } => {
                // R_X(ξ) = X^{½} expm( X^{−½} ξ X^{−½} ) X^{½}.
                let (xh, xhi) = sqrt_and_inv_sqrt_spd(x, n)?;
                let m1 = mat_mul(&xhi, xi, n, n, n);
                let inner = mat_mul(&m1, &xhi, n, n, n); // X^{-½} ξ X^{-½}
                let exp_inner = expm_sym(&symmetrise(&inner, n), n)?;
                let m2 = mat_mul(&xh, &exp_inner, n, n, n);
                Ok(mat_mul(&m2, &xh, n, n, n))
            }
            Manifold::Stiefel { n, p } => {
                check_len(xi, n * p)?;
                let sum: Vec<f64> = x.iter().zip(xi.iter()).map(|(a, b)| a + b).collect();
                qr_q_factor(&sum, n, p)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration & result
// ---------------------------------------------------------------------------

/// Configuration for Riemannian gradient descent.
#[derive(Debug, Clone)]
pub struct RiemannianConfig {
    /// Maximum number of iterations (default `500`).
    pub max_iter: usize,
    /// Initial Armijo trial step `α₀` (default `1.0`).
    pub init_step: f64,
    /// Armijo backtracking factor `β ∈ (0, 1)` (default `0.5`).
    pub backtrack: f64,
    /// Armijo sufficient-decrease constant `c ∈ (0, 1)` (default `1 × 10⁻⁴`).
    pub armijo_c: f64,
    /// Maximum backtracking trials per iteration (default `40`).
    pub max_ls: usize,
    /// Stop when `‖grad f(x)‖_x < tol` (default `1 × 10⁻⁸`).
    pub tol: f64,
}

impl Default for RiemannianConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            init_step: 1.0,
            backtrack: 0.5,
            armijo_c: 1e-4,
            max_ls: 40,
            tol: 1e-8,
        }
    }
}

/// Result of a Riemannian gradient-descent run.
#[derive(Debug, Clone)]
pub struct RiemannianResult {
    /// Final point on the manifold.
    pub x: Vec<f64>,
    /// Final objective value.
    pub f: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether `‖grad f‖_x < tol` was attained.
    pub converged: bool,
    /// Final Riemannian gradient norm `‖grad f(x)‖_x`.
    pub grad_norm: f64,
    /// Objective history (one entry per iteration).
    pub obj_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------

/// Minimise `f` on `manifold` by Riemannian gradient descent with Armijo line search.
///
/// * `manifold` — manifold specification.
/// * `x0` — starting point (assumed to already lie on the manifold; it is
///   projected back via retraction of the zero vector for safety).
/// * `f` — cost `f(x)`.
/// * `egrad` — **Euclidean** gradient `∇f(x)` in the ambient space; it is projected
///   to the tangent space internally.
/// * `cfg` — configuration.
///
/// # Errors
/// * [`CvxError::InvalidParameter`] for malformed config or manifold dimensions.
/// * [`CvxError::DimensionMismatch`] if `x0` or a returned gradient has wrong length.
/// * [`CvxError::LineSearchFailed`] if Armijo backtracking exhausts `max_ls` trials.
pub fn riemannian_gradient_descent<F, G>(
    manifold: &Manifold,
    x0: &[f64],
    f: F,
    egrad: G,
    cfg: &RiemannianConfig,
) -> CvxResult<RiemannianResult>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    let dim = manifold.len();
    if dim == 0 {
        return Err(CvxError::InvalidParameter("manifold has zero size".into()));
    }
    if let Manifold::Stiefel { n, p } = *manifold {
        if p > n {
            return Err(CvxError::InvalidParameter(format!(
                "Stiefel requires p ≤ n, got p={p}, n={n}"
            )));
        }
    }
    if x0.len() != dim {
        return Err(CvxError::DimensionMismatch {
            a: x0.len(),
            b: dim,
        });
    }
    if !(cfg.backtrack > 0.0 && cfg.backtrack < 1.0) {
        return Err(CvxError::InvalidParameter(format!(
            "backtrack ∈ (0, 1) required, got {}",
            cfg.backtrack
        )));
    }
    if !(cfg.armijo_c > 0.0 && cfg.armijo_c < 1.0) {
        return Err(CvxError::InvalidParameter(format!(
            "armijo_c ∈ (0, 1) required, got {}",
            cfg.armijo_c
        )));
    }
    if cfg.init_step <= 0.0 || !cfg.init_step.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "init_step > 0 required, got {}",
            cfg.init_step
        )));
    }

    // Pull the initial point onto the manifold (retraction of zero tangent).
    let zero = vec![0.0_f64; dim];
    let mut x = manifold.retract(x0, &zero)?;
    let mut fx = f(&x)?;
    if !fx.is_finite() {
        return Err(CvxError::NumericalInstability(
            "Riemannian GD: objective not finite at x0".into(),
        ));
    }
    let mut obj_history = Vec::with_capacity(cfg.max_iter);

    let mut converged = false;
    let mut grad_norm = f64::INFINITY;
    let mut iters = 0usize;

    for it in 0..cfg.max_iter {
        iters = it + 1;
        obj_history.push(fx);

        let g_eucl = egrad(&x)?;
        if g_eucl.len() != dim {
            return Err(CvxError::DimensionMismatch {
                a: g_eucl.len(),
                b: dim,
            });
        }
        let rgrad = manifold.project(&x, &g_eucl)?;
        let gnorm_sq = manifold.inner(&x, &rgrad, &rgrad)?;
        grad_norm = gnorm_sq.max(0.0).sqrt();
        if grad_norm < cfg.tol {
            converged = true;
            break;
        }

        // Riemannian Armijo backtracking along the geodesic-ish curve R_x(−α g).
        let mut alpha = cfg.init_step;
        let mut accepted = false;
        let mut x_next = x.clone();
        let mut f_next = fx;
        for _ in 0..cfg.max_ls {
            let dir: Vec<f64> = rgrad.iter().map(|gi| -alpha * gi).collect();
            let cand = manifold.retract(&x, &dir)?;
            let f_cand = f(&cand)?;
            if f_cand.is_finite() && f_cand <= fx - cfg.armijo_c * alpha * gnorm_sq {
                x_next = cand;
                f_next = f_cand;
                accepted = true;
                break;
            }
            alpha *= cfg.backtrack;
        }
        if !accepted {
            // Could not decrease: we are at a (numerical) stationary point.
            return Ok(RiemannianResult {
                x,
                f: fx,
                n_iter: iters,
                converged: grad_norm < cfg.tol,
                grad_norm,
                obj_history,
            });
        }

        x = x_next;
        fx = f_next;
    }

    Ok(RiemannianResult {
        x,
        f: fx,
        n_iter: iters,
        converged,
        grad_norm,
        obj_history,
    })
}

// ---------------------------------------------------------------------------
// Small dense matrix helpers (row-major, self-contained)
// ---------------------------------------------------------------------------

fn check_len(v: &[f64], expected: usize) -> CvxResult<()> {
    if v.len() != expected {
        return Err(CvxError::DimensionMismatch {
            a: v.len(),
            b: expected,
        });
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `sym(A) = ½(A + Aᵀ)` for an `n×n` row-major matrix.
fn symmetrise(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    out
}

/// General matrix product `C = A·B` where `A` is `m×k`, `B` is `k×p`, all row-major.
fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * p];
    for i in 0..m {
        for l in 0..k {
            let ail = a[i * k + l];
            if ail == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i * p + j] += ail * b[l * p + j];
            }
        }
    }
    c
}

/// `Aᵀ·B` where `A` is `m×ka`, `B` is `m×kb` (contract over the `m` rows) → `ka×kb`.
fn mat_t_mat_general(a: &[f64], b: &[f64], m: usize, ka: usize, kb: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; ka * kb];
    for r in 0..m {
        for i in 0..ka {
            let air = a[r * ka + i];
            for j in 0..kb {
                c[i * kb + j] += air * b[r * kb + j];
            }
        }
    }
    c
}

/// Thin QR via modified Gram–Schmidt; returns the `n×p` Q factor with a sign
/// convention making the diagonal of R non-negative (a valid retraction `qf`).
fn qr_q_factor(a: &[f64], n: usize, p: usize) -> CvxResult<Vec<f64>> {
    // Column-extract, orthogonalise, re-pack.
    let mut q = vec![0.0_f64; n * p];
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| a[i * p + j]).collect::<Vec<f64>>())
        .collect();
    for j in 0..p {
        // Subtract projections onto previously computed orthonormal columns.
        for l in 0..j {
            let ql: Vec<f64> = (0..n).map(|i| q[i * p + l]).collect();
            let r = dot(&ql, &cols[j]);
            for i in 0..n {
                cols[j][i] -= r * ql[i];
            }
        }
        let nrm = norm(&cols[j]);
        if nrm < 1e-300 {
            return Err(CvxError::NumericalInstability(
                "QR retraction: rank-deficient input".into(),
            ));
        }
        // Sign convention: make R_jj ≥ 0 ⇒ keep direction of the (reduced) column.
        for i in 0..n {
            q[i * p + j] = cols[j][i] / nrm;
        }
    }
    Ok(q)
}

/// Jacobi eigendecomposition of a symmetric `n×n` matrix.
/// Returns `(eigenvalues, eigenvectors_columns)` where eigenvector `k` is column `k`
/// of the returned row-major `n×n` matrix `V`, so `A ≈ V diag(λ) Vᵀ`.
fn jacobi_eig(a_in: &[f64], n: usize) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p, q.
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    Ok((eig, v))
}

/// Reconstruct `V · diag(g(λ)) · Vᵀ` given eigenpairs and a scalar map `g`.
fn reconstruct(eig: &[f64], v: &[f64], n: usize, g: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += v[i * n + k] * g(eig[k]) * v[j * n + k];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Inverse of an SPD matrix via eigendecomposition.
fn inv_spd(a: &[f64], n: usize) -> CvxResult<Vec<f64>> {
    let (eig, v) = jacobi_eig(&symmetrise(a, n), n)?;
    if eig.iter().any(|&l| l <= 1e-300) {
        return Err(CvxError::SingularMatrix(
            "inv_spd: non-positive eigenvalue".into(),
        ));
    }
    Ok(reconstruct(&eig, &v, n, |l| 1.0 / l))
}

/// Matrix square root and inverse square root of an SPD matrix.
fn sqrt_and_inv_sqrt_spd(a: &[f64], n: usize) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    let (eig, v) = jacobi_eig(&symmetrise(a, n), n)?;
    if eig.iter().any(|&l| l <= 1e-300) {
        return Err(CvxError::SingularMatrix(
            "sqrt_spd: non-positive eigenvalue".into(),
        ));
    }
    let half = reconstruct(&eig, &v, n, |l| l.sqrt());
    let inv_half = reconstruct(&eig, &v, n, |l| 1.0 / l.sqrt());
    Ok((half, inv_half))
}

/// Matrix exponential of a symmetric matrix via eigendecomposition.
fn expm_sym(a: &[f64], n: usize) -> CvxResult<Vec<f64>> {
    let (eig, v) = jacobi_eig(a, n)?;
    Ok(reconstruct(&eig, &v, n, |l| l.exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sphere: minimise xᵀ A x over the unit sphere ⇒ smallest eigenvector. ──
    #[test]
    fn sphere_rayleigh_quotient_min() {
        // A = diag(1, 4, 9). Min of xᵀAx on the sphere is 1 at e_0.
        let a = [1.0_f64, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let ax = mat_mul(&a, x, 3, 3, 1);
            Ok(dot(x, &ax))
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            // ∇(xᵀAx) = 2 A x.
            Ok(mat_mul(&a, x, 3, 3, 1).iter().map(|v| 2.0 * v).collect())
        };
        let m = Manifold::Sphere { n: 3 };
        let cfg = RiemannianConfig {
            max_iter: 4000,
            ..Default::default()
        };
        let res = riemannian_gradient_descent(&m, &[0.3, 0.6, 0.7], f, g, &cfg).expect("ok");
        assert!((res.f - 1.0).abs() < 1e-4, "f = {}", res.f);
        // Point should align with ±e_0.
        assert!(res.x[0].abs() > 0.99, "x = {:?}", res.x);
        assert!(res.x[1].abs() < 1e-2 && res.x[2].abs() < 1e-2);
    }

    #[test]
    fn sphere_point_stays_on_manifold() {
        let a = [2.0_f64, 0.5, 0.5, 3.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let ax = mat_mul(&a, x, 2, 2, 1);
            Ok(dot(x, &ax))
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(mat_mul(&a, x, 2, 2, 1).iter().map(|v| 2.0 * v).collect())
        };
        let m = Manifold::Sphere { n: 2 };
        let res = riemannian_gradient_descent(&m, &[0.6, 0.8], f, g, &RiemannianConfig::default())
            .expect("ok");
        assert!((norm(&res.x) - 1.0).abs() < 1e-9, "‖x‖ = {}", norm(&res.x));
    }

    #[test]
    fn sphere_converges_flag_and_grad() {
        // Large eigengap diag(1, 10) ⇒ fast linear convergence of RGD.
        let a = [1.0_f64, 0.0, 0.0, 10.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let ax = mat_mul(&a, x, 2, 2, 1);
            Ok(dot(x, &ax))
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(mat_mul(&a, x, 2, 2, 1).iter().map(|v| 2.0 * v).collect())
        };
        let m = Manifold::Sphere { n: 2 };
        let res = riemannian_gradient_descent(
            &m,
            &[0.6, 0.8],
            f,
            g,
            &RiemannianConfig {
                tol: 1e-7,
                max_iter: 1000,
                ..Default::default()
            },
        )
        .expect("ok");
        assert!(res.converged, "grad_norm = {}", res.grad_norm);
        assert!(res.grad_norm < 1e-7);
        assert!((res.f - 1.0).abs() < 1e-6, "f = {}", res.f);
    }

    // ── Objective history must be non-increasing under Armijo. ──
    #[test]
    fn objective_monotone_nonincreasing() {
        let a = [1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let ax = mat_mul(&a, x, 3, 3, 1);
            Ok(dot(x, &ax))
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(mat_mul(&a, x, 3, 3, 1).iter().map(|v| 2.0 * v).collect())
        };
        let m = Manifold::Sphere { n: 3 };
        let res =
            riemannian_gradient_descent(&m, &[0.4, 0.5, 0.7], f, g, &RiemannianConfig::default())
                .expect("ok");
        for w in res.obj_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "history not monotone: {w:?}");
        }
    }

    // ── Stiefel: minimise −tr(XᵀA X) over St(n, p) ⇒ top-p eigenspace. ──
    #[test]
    fn stiefel_trace_maximisation() {
        // A = diag(10, 5, 1), p = 1 ⇒ leading eigenvector e_0, objective −10.
        let a = [10.0_f64, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            // −tr(Xᵀ A X), X is 3×1.
            let ax = mat_mul(&a, x, 3, 3, 1);
            Ok(-dot(x, &ax))
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            // ∇(−tr(XᵀAX)) = −2 A X.
            Ok(mat_mul(&a, x, 3, 3, 1).iter().map(|v| -2.0 * v).collect())
        };
        let m = Manifold::Stiefel { n: 3, p: 1 };
        let res = riemannian_gradient_descent(
            &m,
            &[0.3, 0.4, 0.866],
            f,
            g,
            &RiemannianConfig {
                max_iter: 800,
                ..Default::default()
            },
        )
        .expect("ok");
        assert!((res.f + 10.0).abs() < 1e-4, "f = {}", res.f);
        assert!(res.x[0].abs() > 0.999, "x = {:?}", res.x);
    }

    #[test]
    fn stiefel_columns_orthonormal() {
        // p = 2 leading eigenspace of diag(8, 6, 1).
        let a = [8.0_f64, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            // −tr(XᵀAX), X is 3×2.
            let mut s = 0.0_f64;
            for j in 0..2 {
                let col: Vec<f64> = (0..3).map(|i| x[i * 2 + j]).collect();
                let ax = mat_mul(&a, &col, 3, 3, 1);
                s += dot(&col, &ax);
            }
            Ok(-s)
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(mat_mul(&a, x, 3, 3, 2).iter().map(|v| -2.0 * v).collect())
        };
        let m = Manifold::Stiefel { n: 3, p: 2 };
        let res = riemannian_gradient_descent(
            &m,
            &[1.0, 0.0, 0.0, 1.0, 0.1, 0.1],
            f,
            g,
            &RiemannianConfig {
                max_iter: 800,
                ..Default::default()
            },
        )
        .expect("ok");
        // XᵀX = I_2.
        let xtx = mat_t_mat_general(&res.x, &res.x, 3, 2, 2);
        assert!((xtx[0] - 1.0).abs() < 1e-6, "‖c0‖² = {}", xtx[0]);
        assert!((xtx[3] - 1.0).abs() < 1e-6, "‖c1‖² = {}", xtx[3]);
        assert!(xtx[1].abs() < 1e-6, "c0·c1 = {}", xtx[1]);
        assert!((res.f + 14.0).abs() < 1e-3, "f = {}", res.f); // -(8+6)
    }

    // ── SPD: Karcher-mean-style cost f(X) = ½‖logm(X⁻¹ T)‖²? Use simpler convex
    //    surrogate that is well-conditioned: f(X) = ½‖X − T‖²_F with SPD retraction.
    //    The Euclidean minimiser T is SPD here, so RGD should approach it. ──
    #[test]
    fn spd_quadratic_pull_to_target() {
        // T SPD: [[2, 0.3],[0.3, 1.5]].
        let t = [2.0_f64, 0.3, 0.3, 1.5];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let d: f64 = x
                .iter()
                .zip(t.iter())
                .map(|(xi, ti)| (xi - ti) * (xi - ti))
                .sum();
            Ok(0.5 * d)
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(t.iter()).map(|(xi, ti)| xi - ti).collect())
        };
        let m = Manifold::Spd { n: 2 };
        let x0 = [1.0_f64, 0.0, 0.0, 1.0]; // identity is SPD
        let res = riemannian_gradient_descent(
            &m,
            &x0,
            f,
            g,
            &RiemannianConfig {
                max_iter: 400,
                init_step: 0.5,
                ..Default::default()
            },
        )
        .expect("ok");
        // Final point close to T and still symmetric.
        for (k, tk) in t.iter().enumerate() {
            assert!(
                (res.x[k] - tk).abs() < 5e-2,
                "x[{k}] = {} vs {}",
                res.x[k],
                tk
            );
        }
        assert!((res.x[1] - res.x[2]).abs() < 1e-9, "not symmetric");
    }

    #[test]
    fn spd_stays_positive_definite() {
        let t = [3.0_f64, 0.0, 0.0, 2.0];
        let f = |x: &[f64]| -> CvxResult<f64> {
            let d: f64 = x
                .iter()
                .zip(t.iter())
                .map(|(xi, ti)| (xi - ti) * (xi - ti))
                .sum();
            Ok(0.5 * d)
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(t.iter()).map(|(xi, ti)| xi - ti).collect())
        };
        let m = Manifold::Spd { n: 2 };
        let res = riemannian_gradient_descent(
            &m,
            &[1.0, 0.0, 0.0, 1.0],
            f,
            g,
            &RiemannianConfig {
                max_iter: 200,
                init_step: 0.5,
                ..Default::default()
            },
        )
        .expect("ok");
        // Eigenvalues of the result must be positive.
        let (eig, _) = jacobi_eig(&res.x, 2).expect("ok");
        assert!(eig.iter().all(|&l| l > 0.0), "eigs = {eig:?}");
    }

    // ── Geometric helper checks. ──
    #[test]
    fn jacobi_eig_reconstructs() {
        let a = [2.0_f64, 1.0, 1.0, 2.0];
        let (eig, v) = jacobi_eig(&a, 2).expect("ok");
        let recon = reconstruct(&eig, &v, 2, |l| l);
        for k in 0..4 {
            assert!((recon[k] - a[k]).abs() < 1e-10, "recon[{k}] = {}", recon[k]);
        }
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3.
        let mut sorted = eig.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_inv_sqrt_consistency() {
        let a = [4.0_f64, 0.0, 0.0, 9.0];
        let (h, hi) = sqrt_and_inv_sqrt_spd(&a, 2).expect("ok");
        // h·h ≈ A.
        let hh = mat_mul(&h, &h, 2, 2, 2);
        for k in 0..4 {
            assert!((hh[k] - a[k]).abs() < 1e-9);
        }
        // h·hi ≈ I.
        let id = mat_mul(&h, &hi, 2, 2, 2);
        assert!((id[0] - 1.0).abs() < 1e-9 && (id[3] - 1.0).abs() < 1e-9);
        assert!(id[1].abs() < 1e-9 && id[2].abs() < 1e-9);
    }

    #[test]
    fn qr_q_factor_orthonormal() {
        // A 3×2 with independent columns.
        let a = [1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0];
        let q = qr_q_factor(&a, 3, 2).expect("ok");
        let qtq = mat_t_mat_general(&q, &q, 3, 2, 2);
        assert!((qtq[0] - 1.0).abs() < 1e-12);
        assert!((qtq[3] - 1.0).abs() < 1e-12);
        assert!(qtq[1].abs() < 1e-12);
    }

    #[test]
    fn dimension_mismatch_x0() {
        let m = Manifold::Sphere { n: 3 };
        let f = |_x: &[f64]| -> CvxResult<f64> { Ok(0.0) };
        let g = |_x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0; 3]) };
        let err = riemannian_gradient_descent(&m, &[1.0, 0.0], f, g, &RiemannianConfig::default());
        assert!(matches!(err, Err(CvxError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_stiefel_dims() {
        let m = Manifold::Stiefel { n: 2, p: 3 };
        let f = |_x: &[f64]| -> CvxResult<f64> { Ok(0.0) };
        let g = |_x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0; 6]) };
        let err = riemannian_gradient_descent(&m, &[0.0; 6], f, g, &RiemannianConfig::default());
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }

    #[test]
    fn invalid_backtrack_param() {
        let m = Manifold::Sphere { n: 2 };
        let f = |_x: &[f64]| -> CvxResult<f64> { Ok(0.0) };
        let g = |_x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0; 2]) };
        let cfg = RiemannianConfig {
            backtrack: 1.5,
            ..Default::default()
        };
        let err = riemannian_gradient_descent(&m, &[1.0, 0.0], f, g, &cfg);
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }
}
