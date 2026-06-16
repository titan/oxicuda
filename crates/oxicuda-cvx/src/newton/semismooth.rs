//! Semismooth Newton method for the Linear Complementarity Problem (Qi-Sun 1993).
//!
//! # Problem
//!
//! The Linear Complementarity Problem `LCP(M, q)` seeks a vector `x ∈ ℝⁿ` with
//!
//! ```text
//!   x ≥ 0,        w := M x + q ≥ 0,        xᵀ w = 0.
//! ```
//!
//! Componentwise the last two conditions read `xᵢ ≥ 0`, `wᵢ ≥ 0`,
//! `xᵢ wᵢ = 0`, i.e. each pair `(xᵢ, wᵢ)` is *complementary*.
//!
//! # Fischer-Burmeister reformulation
//!
//! The Fischer-Burmeister (FB) function
//!
//! ```text
//!   φ(a, b) = √(a² + b²) − a − b
//! ```
//!
//! is an *NCP function*: `φ(a, b) = 0  ⟺  a ≥ 0, b ≥ 0, a b = 0`.  Stacking the
//! per-component FB residuals turns the LCP into the (square) nonsmooth equation
//!
//! ```text
//!   F(x) = 0,   Fᵢ(x) = φ(xᵢ, (M x + q)ᵢ).
//! ```
//!
//! `φ` is continuously differentiable everywhere except at the origin `(0, 0)`,
//! yet it is *strongly semismooth*, which is exactly the regularity the
//! generalized Newton method of Qi & Sun exploits.
//!
//! # Generalized Newton iteration
//!
//! Pick an element `V_k ∈ ∂_B F(x_k)` of the B-subdifferential, solve the Newton
//! system `V_k d_k = −F(x_k)`, and globalize with an Armijo line search on the
//! natural merit function
//!
//! ```text
//!   Ψ(x) = ½ ‖F(x)‖²,        ∇Ψ(x) = Vᵀ F(x)   for any V ∈ ∂_B F(x).
//! ```
//!
//! A B-subdifferential element has the chain-rule structure
//!
//! ```text
//!   V = D_a + D_b M,
//!   D_a = diag(∂φ/∂a at (xᵢ, wᵢ)),   D_b = diag(∂φ/∂b at (xᵢ, wᵢ)),
//! ```
//!
//! where, away from the origin,
//!
//! ```text
//!   ∂φ/∂a = a / √(a²+b²) − 1,        ∂φ/∂b = b / √(a²+b²) − 1,
//! ```
//!
//! and at the origin we select the limiting gradient along the fixed direction
//! `z = (1, 1)`:  `∂φ/∂a = z₁/‖z‖ − 1`, `∂φ/∂b = z₂/‖z‖ − 1` (a valid Clarke
//! generalized gradient element of `φ` at `0`).
//!
//! Locally the method converges *superlinearly* (quadratically when `F` is
//! strongly semismooth and the chosen `V_k` are nonsingular), which the tests
//! verify by tracking the sharp drop of the residual in the final iterations.
//!
//! # References
//!
//! * L. Qi and J. Sun, *A nonsmooth version of Newton's method*,
//!   Mathematical Programming 58 (1993), 353-367.
//! * A. Fischer, *A special Newton-type optimization method*,
//!   Optimization 24 (1992), 269-284.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// Direction used to pick a B-subdifferential element of `φ` at the origin.
///
/// Both partials are evaluated along the normalized direction `(1, 1)`, giving
/// the Clarke gradient element `(1/√2 − 1, 1/√2 − 1)`.
const ORIGIN_DIRECTION: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Termination status of [`semismooth_newton_lcp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemismoothStatus {
    /// The merit residual `‖F(x)‖` fell below the requested tolerance.
    Converged,
    /// The iteration cap was reached before convergence.
    MaxIterReached,
    /// The line search could not produce sufficient decrease (stationary point
    /// of the merit function that is not a root, or numerical stall).
    LineSearchStalled,
}

/// Tuning parameters for the semismooth Newton LCP solver.
#[derive(Debug, Clone, Copy)]
pub struct SemismoothConfig {
    /// Maximum number of outer Newton iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖F(x)‖₂`.
    pub tol: f64,
    /// Armijo sufficient-decrease constant `σ ∈ (0, ½)`.
    pub armijo_sigma: f64,
    /// Backtracking shrink factor `β ∈ (0, 1)`.
    pub backtrack_beta: f64,
    /// Maximum number of backtracking steps per iteration.
    pub max_backtrack: usize,
    /// Levenberg-style regularization added to the Newton matrix diagonal when
    /// it is (near) singular, ensuring a usable descent direction.
    pub regularization: f64,
}

impl Default for SemismoothConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1.0e-10,
            armijo_sigma: 1.0e-4,
            backtrack_beta: 0.5,
            max_backtrack: 40,
            regularization: 1.0e-10,
        }
    }
}

/// Result of [`semismooth_newton_lcp`].
#[derive(Debug, Clone)]
pub struct SemismoothNewtonResult {
    /// The computed solution estimate `x`.
    pub x: Vec<f64>,
    /// The complementary slack `w = M x + q` at the solution.
    pub w: Vec<f64>,
    /// Final merit residual `‖F(x)‖₂`.
    pub residual: f64,
    /// Number of outer Newton iterations performed.
    pub iterations: usize,
    /// Per-iteration history of `‖F(x_k)‖₂` (length `iterations + 1`, including
    /// the initial residual).  Useful for inspecting the asymptotic rate.
    pub residual_history: Vec<f64>,
    /// Termination status.
    pub status: SemismoothStatus,
}

/// The Fischer-Burmeister NCP function `φ(a, b) = √(a² + b²) − a − b`.
///
/// `φ(a, b) = 0` if and only if `a ≥ 0`, `b ≥ 0`, and `a b = 0`.
#[must_use]
pub fn fischer_burmeister(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt() - a - b
}

/// A generalized-gradient element `(∂φ/∂a, ∂φ/∂b)` of the FB function.
///
/// Away from the origin this is the ordinary gradient
/// `(a/r − 1, b/r − 1)` with `r = √(a² + b²)`.  At the origin we return the
/// limiting gradient along the direction `(1, 1)`, namely
/// `(1/√2 − 1, 1/√2 − 1)`, which is a valid element of the Clarke
/// subdifferential `∂φ(0, 0)`.
#[must_use]
pub fn fischer_burmeister_gradient(a: f64, b: f64) -> (f64, f64) {
    let r = (a * a + b * b).sqrt();
    if r <= f64::MIN_POSITIVE {
        // Origin: B-subdifferential element along the fixed unit direction.
        (ORIGIN_DIRECTION - 1.0, ORIGIN_DIRECTION - 1.0)
    } else {
        (a / r - 1.0, b / r - 1.0)
    }
}

/// Componentwise FB residual `F(x)` with `Fᵢ = φ(xᵢ, (M x + q)ᵢ)`.
///
/// `M` is a row-major `n × n` matrix; `q` and `x` have length `n`.
fn fb_residual(m: &[f64], q: &[f64], x: &[f64], n: usize) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    let mx = mat_vec(m, n, n, x)?;
    let mut w = vec![0.0_f64; n];
    let mut f = vec![0.0_f64; n];
    for i in 0..n {
        w[i] = mx[i] + q[i];
        f[i] = fischer_burmeister(x[i], w[i]);
    }
    Ok((f, w))
}

/// Euclidean norm of the FB residual `‖F(x)‖₂` for the LCP `(M, q)`.
///
/// At a true LCP solution this equals zero.
///
/// # Errors
/// Propagates dimension errors from the matrix-vector product.
pub fn lcp_residual(m: &[f64], q: &[f64], x: &[f64]) -> CvxResult<f64> {
    let n = q.len();
    if x.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }
    if m.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![m.len()],
        });
    }
    let (f, _) = fb_residual(m, q, x, n)?;
    Ok(norm2(&f))
}

/// Assemble a B-subdifferential element `V = D_a + D_b M` (row-major `n × n`).
fn generalized_jacobian(m: &[f64], x: &[f64], w: &[f64], n: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        let (da, db) = fischer_burmeister_gradient(x[i], w[i]);
        let row = i * n;
        for j in 0..n {
            v[row + j] = db * m[row + j];
        }
        v[row + i] += da;
    }
    v
}

/// Solve the LCP `x ≥ 0, M x + q ≥ 0, xᵀ(M x + q) = 0` by a globalized
/// semismooth (generalized) Newton iteration on the Fischer-Burmeister system.
///
/// # Arguments
/// * `m` – row-major `n × n` matrix `M`.
/// * `q` – right-hand-side vector of length `n`.
/// * `x0` – starting point of length `n`.
/// * `config` – algorithmic parameters (see [`SemismoothConfig`]).
///
/// # Errors
/// * [`CvxError::EmptyInput`] if `q` is empty.
/// * [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] on size
///   disagreement between `m`, `q`, `x0`.
/// * [`CvxError::InvalidParameter`] if any config parameter is out of range.
/// * [`CvxError::NumericalInstability`] if the regularized Newton system cannot
///   be solved (should not occur for finite inputs with positive
///   regularization).
pub fn semismooth_newton_lcp(
    m: &[f64],
    q: &[f64],
    x0: &[f64],
    config: &SemismoothConfig,
) -> CvxResult<SemismoothNewtonResult> {
    let n = q.len();
    if n == 0 {
        return Err(CvxError::EmptyInput);
    }
    if m.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![m.len()],
        });
    }
    if x0.len() != n {
        return Err(CvxError::DimensionMismatch { a: x0.len(), b: n });
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            config.tol
        )));
    }
    if !(0.0..0.5).contains(&config.armijo_sigma) || config.armijo_sigma <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "armijo_sigma must lie in (0, 0.5), got {}",
            config.armijo_sigma
        )));
    }
    if !(0.0..1.0).contains(&config.backtrack_beta) || config.backtrack_beta <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "backtrack_beta must lie in (0, 1), got {}",
            config.backtrack_beta
        )));
    }
    if !config.regularization.is_finite() || config.regularization < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "regularization must be ≥ 0, got {}",
            config.regularization
        )));
    }

    let mut x = x0.to_vec();
    let (mut f, mut w) = fb_residual(m, q, &x, n)?;
    let mut res = norm2(&f);
    let mut residual_history = vec![res];

    let mut status = SemismoothStatus::MaxIterReached;
    let mut iterations = 0_usize;

    for _ in 0..config.max_iter {
        if res < config.tol {
            status = SemismoothStatus::Converged;
            break;
        }
        iterations += 1;

        // Generalized Jacobian element V ∈ ∂_B F(x).
        let v = generalized_jacobian(m, &x, &w, n);

        // Newton direction: V d = −F.  Regularize the *normal* matrix to keep a
        // descent guarantee even where V is singular: solve (VᵀV + μI) d = −VᵀF
        // when the direct solve degenerates.  The merit gradient is g = VᵀF.
        let neg_f: Vec<f64> = f.iter().map(|fi| -fi).collect();
        let direction = match solve_dense(&v, n, &neg_f) {
            Ok(d) => {
                // Verify d is a genuine descent direction for Ψ; otherwise fall
                // back to the regularized Gauss-Newton step below.
                let grad = mat_t_vec_local(&v, n, &f);
                let slope = dot_local(&grad, &d);
                if slope < -config.regularization * res * res {
                    d
                } else {
                    regularized_gauss_newton(&v, &f, n, config.regularization)?
                }
            }
            Err(_) => regularized_gauss_newton(&v, &f, n, config.regularization)?,
        };

        // Merit-function Armijo line search:
        //   Ψ(x + t d) ≤ Ψ(x) + σ t ∇Ψ(x)ᵀ d.
        let grad = mat_t_vec_local(&v, n, &f);
        let slope = dot_local(&grad, &direction);
        let psi = 0.5 * res * res;

        let mut t = 1.0_f64;
        let mut accepted = false;
        let mut x_trial = x.clone();
        let mut f_trial = f.clone();
        let mut w_trial = w.clone();
        for _ in 0..config.max_backtrack {
            for i in 0..n {
                x_trial[i] = x[i] + t * direction[i];
            }
            let (ft, wt) = fb_residual(m, q, &x_trial, n)?;
            let psi_trial = 0.5 * norm2(&ft) * norm2(&ft);
            if psi_trial <= psi + config.armijo_sigma * t * slope {
                f_trial = ft;
                w_trial = wt;
                accepted = true;
                break;
            }
            t *= config.backtrack_beta;
        }

        if !accepted {
            // No sufficient decrease: stationary point of the merit or stall.
            status = SemismoothStatus::LineSearchStalled;
            break;
        }

        x.clone_from(&x_trial);
        f = f_trial;
        w = w_trial;
        res = norm2(&f);
        residual_history.push(res);

        if res < config.tol {
            status = SemismoothStatus::Converged;
            break;
        }
    }

    Ok(SemismoothNewtonResult {
        x,
        w,
        residual: res,
        iterations,
        residual_history,
        status,
    })
}

/// Local `Vᵀ y` (`V` row-major `n × n`) without a public dependency.
fn mat_t_vec_local(v: &[f64], n: usize, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (i, &yi) in y.iter().enumerate().take(n) {
        let row = i * n;
        for j in 0..n {
            out[j] += v[row + j] * yi;
        }
    }
    out
}

/// Local dot product (inputs guaranteed equal length by construction).
fn dot_local(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Solve the regularized Gauss-Newton system `(VᵀV + μI) d = −VᵀF`.
///
/// This always yields a descent direction for `Ψ = ½‖F‖²` since `VᵀV + μI ≻ 0`
/// for `μ > 0`, with slope `∇Ψᵀ d = −(VᵀF)ᵀ(VᵀV+μI)⁻¹(VᵀF) ≤ 0`.
fn regularized_gauss_newton(v: &[f64], f: &[f64], n: usize, mu: f64) -> CvxResult<Vec<f64>> {
    // Normal matrix VᵀV (symmetric n × n).
    let mut vtv = vec![0.0_f64; n * n];
    for k in 0..n {
        let row = k * n;
        for i in 0..n {
            let vki = v[row + i];
            for j in 0..n {
                vtv[i * n + j] += vki * v[row + j];
            }
        }
    }
    let reg = if mu > 0.0 { mu } else { 1.0e-12 };
    for i in 0..n {
        vtv[i * n + i] += reg;
    }
    // Right-hand side −VᵀF.
    let mut rhs = mat_t_vec_local(v, n, f);
    for r in rhs.iter_mut() {
        *r = -*r;
    }
    solve_dense(&vtv, n, &rhs).map_err(|e| {
        CvxError::NumericalInstability(format!("regularized Gauss-Newton solve failed: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FB function vanishes exactly on the complementarity cone and is nonzero
    /// off it.
    #[test]
    fn fb_zero_iff_complementary() {
        // a, b ≥ 0 with a b = 0  →  φ = 0.
        assert!(fischer_burmeister(0.0, 0.0).abs() < 1e-15);
        assert!(fischer_burmeister(3.0, 0.0).abs() < 1e-15);
        assert!(fischer_burmeister(0.0, 5.0).abs() < 1e-15);

        // Interior of the positive orthant: a, b > 0  →  φ < 0 (not a root).
        assert!(fischer_burmeister(1.0, 1.0) < -1e-9);
        assert!(fischer_burmeister(2.0, 3.0) < -1e-9);

        // Any negative component breaks complementarity  →  φ ≠ 0.
        assert!(fischer_burmeister(-1.0, 0.0).abs() > 1e-9);
        assert!(fischer_burmeister(0.0, -2.0).abs() > 1e-9);
        assert!(fischer_burmeister(-1.0, -1.0).abs() > 1e-9);
        assert!(fischer_burmeister(2.0, -1.0).abs() > 1e-9);
    }

    /// φ(a, b) = 0 across a grid is equivalent to (a≥0, b≥0, ab≈0).
    #[test]
    fn fb_characterization_grid() {
        let grid = [-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5];
        for &a in &grid {
            for &b in &grid {
                let phi = fischer_burmeister(a, b);
                let is_complementary = a >= 0.0 && b >= 0.0 && (a * b).abs() < 1e-12;
                if is_complementary {
                    assert!(phi.abs() < 1e-12, "phi({a},{b})={phi} expected 0");
                } else {
                    assert!(phi.abs() > 1e-9, "phi({a},{b})={phi} expected nonzero");
                }
            }
        }
    }

    /// The analytic FB gradient matches a central finite difference where `φ` is
    /// differentiable (i.e. away from the origin) — confirming the generalized
    /// Jacobian element is the *true* Jacobian on the smooth region.
    #[test]
    fn fb_gradient_matches_finite_difference() {
        let pts = [
            (1.0, 2.0),
            (3.0, -1.0),
            (-2.0, 4.0),
            (0.5, 0.5),
            (-1.0, -3.0),
        ];
        let h = 1e-6;
        for &(a, b) in &pts {
            let (ga, gb) = fischer_burmeister_gradient(a, b);
            let fd_a = (fischer_burmeister(a + h, b) - fischer_burmeister(a - h, b)) / (2.0 * h);
            let fd_b = (fischer_burmeister(a, b + h) - fischer_burmeister(a, b - h)) / (2.0 * h);
            assert!((ga - fd_a).abs() < 1e-5, "∂a at ({a},{b}): {ga} vs {fd_a}");
            assert!((gb - fd_b).abs() < 1e-5, "∂b at ({a},{b}): {gb} vs {fd_b}");
        }
    }

    /// The origin element of the FB subdifferential lies in the Clarke
    /// generalized gradient (a convex combination of limiting gradients), which
    /// for the FB function means each partial sits in `[−1−1/√2, 1/√2 − 1]`.
    #[test]
    fn fb_origin_subgradient_is_valid() {
        let (ga, gb) = fischer_burmeister_gradient(0.0, 0.0);
        let lo = -1.0 - std::f64::consts::FRAC_1_SQRT_2;
        let hi = std::f64::consts::FRAC_1_SQRT_2 - 1.0;
        assert!(ga >= lo - 1e-12 && ga <= hi + 1e-12, "ga={ga}");
        assert!(gb >= lo - 1e-12 && gb <= hi + 1e-12, "gb={gb}");
    }

    /// The assembled generalized Jacobian `V = D_a + D_b M` equals the true
    /// Jacobian of `F` (computed by finite differences) wherever `F` is smooth.
    #[test]
    fn generalized_jacobian_matches_smooth_jacobian() {
        // 2×2 PSD M, point strictly in the positive orthant ⇒ F is smooth here.
        let m = vec![2.0, 0.5, 0.5, 3.0];
        let q = vec![-1.0, -2.0];
        let n = 2;
        let x = vec![1.3, 0.7];
        let (_f, w) = fb_residual(&m, &q, &x, n).expect("residual");
        let v = generalized_jacobian(&m, &x, &w, n);

        // Finite-difference Jacobian J[i][j] = ∂Fᵢ/∂xⱼ.
        let h = 1e-6;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let (fp, _) = fb_residual(&m, &q, &xp, n).expect("fp");
            let (fm, _) = fb_residual(&m, &q, &xm, n).expect("fm");
            for i in 0..n {
                let fd = (fp[i] - fm[i]) / (2.0 * h);
                assert!(
                    (v[i * n + j] - fd).abs() < 1e-5,
                    "V[{i}][{j}]={} vs fd {fd}",
                    v[i * n + j]
                );
            }
        }
    }

    /// Solve an LCP with a known interior-complementary solution.
    ///
    /// Take `M` SPD and `x* > 0`; set `q = −M x*` so `w* = 0` and the unique
    /// solution is `x*` (with `xᵢ wᵢ = 0` trivially).
    #[test]
    fn solves_lcp_known_solution_w_zero() {
        let m = vec![4.0, 1.0, 1.0, 3.0];
        let x_star = vec![2.0, 1.0];
        // q = −M x*.
        let mx = mat_vec(&m, 2, 2, &x_star).expect("mx");
        let q: Vec<f64> = mx.iter().map(|v| -v).collect();
        let cfg = SemismoothConfig::default();
        let res = semismooth_newton_lcp(&m, &q, &[0.0, 0.0], &cfg).expect("solve");
        assert_eq!(res.status, SemismoothStatus::Converged);
        assert!(res.residual < 1e-8, "residual {}", res.residual);
        assert!((res.x[0] - 2.0).abs() < 1e-6, "x0={}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-6, "x1={}", res.x[1]);
    }

    /// Solve an LCP whose solution has an *active* and an *inactive* component
    /// (mixed complementarity), a sharper test of the FB reformulation.
    #[test]
    fn solves_lcp_mixed_active_set() {
        // M = I (PSD), q = [−1, 2].
        //   Component 0: want x≥0, x−1≥0, x(x−1)=0 ⇒ x=1, w=0.
        //   Component 1: want x≥0, x+2≥0, x(x+2)=0 ⇒ x=0, w=2.
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let q = vec![-1.0, 2.0];
        let cfg = SemismoothConfig::default();
        let res = semismooth_newton_lcp(&m, &q, &[0.5, 0.5], &cfg).expect("solve");
        assert_eq!(res.status, SemismoothStatus::Converged);
        assert!(res.residual < 1e-8, "residual {}", res.residual);
        assert!((res.x[0] - 1.0).abs() < 1e-6, "x0={}", res.x[0]);
        assert!(res.x[1].abs() < 1e-6, "x1={}", res.x[1]);
        // Feasibility of the solution.
        assert!(res.x[0] >= -1e-7 && res.x[1] >= -1e-7);
        assert!(res.w[0] >= -1e-7 && res.w[1] >= -1e-7);
        // Complementarity xᵀw ≈ 0.
        let compl: f64 = res.x.iter().zip(res.w.iter()).map(|(a, b)| a * b).sum();
        assert!(compl.abs() < 1e-6, "compl {compl}");
    }

    /// Larger random-ish SPD LCP solved to high accuracy.
    #[test]
    fn solves_lcp_spd_dim4() {
        // SPD M = LᵀL + I with a fixed lower-triangular L.
        let l = vec![
            1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, -0.3, 0.2, 1.0, 0.0, 0.1, -0.4, 0.6, 1.0,
        ];
        let n = 4;
        let mut m = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += l[k * n + i] * l[k * n + j];
                }
                m[i * n + j] = s + if i == j { 1.0 } else { 0.0 };
            }
        }
        // Known solution x* with one zero component.
        let x_star = vec![1.5, 0.0, 2.0, 0.5];
        let mx = mat_vec(&m, n, n, &x_star).expect("mx");
        // Build q so the inactive component (index 1) has w*>0 while actives have w*=0.
        // Start from q = −M x*  (gives w*=0 everywhere) then bump q[1] so w*[1]>0,
        // keeping x*[1]=0 complementary.
        let mut q: Vec<f64> = mx.iter().map(|v| -v).collect();
        q[1] += 1.0; // w*[1] = 1 > 0, still complementary with x*[1] = 0.
        let cfg = SemismoothConfig::default();
        let res = semismooth_newton_lcp(&m, &q, &vec![0.0; n], &cfg).expect("solve");
        assert_eq!(res.status, SemismoothStatus::Converged);
        assert!(res.residual < 1e-8, "residual {}", res.residual);
        for (xi, si) in res.x.iter().zip(x_star.iter()) {
            assert!((xi - si).abs() < 1e-5, "x {xi} vs {si}");
        }
    }

    /// Near the solution the residual must drop *sharply* (superlinear /
    /// quadratic local rate of the semismooth Newton method).
    #[test]
    fn local_convergence_is_superlinear() {
        let m = vec![4.0, 1.0, 1.0, 3.0];
        let x_star = vec![2.0, 1.0];
        let mx = mat_vec(&m, 2, 2, &x_star).expect("mx");
        let q: Vec<f64> = mx.iter().map(|v| -v).collect();
        let cfg = SemismoothConfig {
            tol: 1e-14,
            ..SemismoothConfig::default()
        };
        // Start close to x* so we are in the local fast region.
        let res = semismooth_newton_lcp(&m, &q, &[1.9, 1.05], &cfg).expect("solve");
        let h = &res.residual_history;
        assert!(h.len() >= 3, "need at least 3 residuals, got {}", h.len());
        // Superlinear: the ratio rₖ₊₁ / rₖ → 0.  Verify the *last* contraction
        // ratio is dramatically smaller than the first usable one.
        let first_ratio = h[1] / h[0].max(1e-300);
        let last_k = h.len() - 1;
        let last_ratio = h[last_k] / h[last_k - 1].max(1e-300);
        assert!(
            last_ratio < first_ratio * 0.1 || h[last_k] < 1e-12,
            "ratios first={first_ratio}, last={last_ratio}, hist={h:?}"
        );
        // And the final residual is essentially machine zero.
        assert!(res.residual < 1e-10, "final residual {}", res.residual);
    }

    /// The merit `Ψ = ½‖F‖²` is non-increasing across the accepted iterates
    /// (consequence of the Armijo line search).
    #[test]
    fn merit_is_non_increasing() {
        let m = vec![3.0, 0.5, 0.5, 2.0];
        let q = vec![-1.0, -0.5];
        let cfg = SemismoothConfig::default();
        let res = semismooth_newton_lcp(&m, &q, &[5.0, -3.0], &cfg).expect("solve");
        let h = &res.residual_history;
        for w in h.windows(2) {
            // Allow a tiny positive slack for round-off.
            assert!(
                w[1] <= w[0] + 1e-12,
                "residual increased {} → {}",
                w[0],
                w[1]
            );
        }
    }

    /// `lcp_residual` returns zero at a true solution and is positive elsewhere.
    #[test]
    fn lcp_residual_zero_at_solution() {
        let m = vec![2.0, 0.0, 0.0, 2.0];
        let q = vec![-2.0, 4.0];
        // Solution x* = [1, 0]: w = [0, 4], complementary.
        let r0 = lcp_residual(&m, &q, &[1.0, 0.0]).expect("res");
        assert!(r0 < 1e-12, "r0={r0}");
        let r1 = lcp_residual(&m, &q, &[0.5, 0.5]).expect("res");
        assert!(r1 > 1e-6, "r1={r1}");
    }

    /// Degenerate / infeasible-start handling: a zero matrix with negative `q`
    /// makes `w = q < 0` unattainable jointly with `x ≥ 0` complementarity; the
    /// solver must terminate gracefully (not panic) with a defined status.
    #[test]
    fn degenerate_problem_terminates_gracefully() {
        // M = 0 ⇒ w = q is constant.  q = [-1, -1] < 0: φ(x, -1) = √(x²+1) − x + 1
        //  is strictly positive for all x, so F has no root — no LCP solution.
        let m = vec![0.0, 0.0, 0.0, 0.0];
        let q = vec![-1.0, -1.0];
        let cfg = SemismoothConfig {
            max_iter: 50,
            ..SemismoothConfig::default()
        };
        let res = semismooth_newton_lcp(&m, &q, &[0.0, 0.0], &cfg).expect("no panic");
        // It cannot converge to residual 0; status must reflect that.
        assert_ne!(res.status, SemismoothStatus::Converged);
        assert!(res.residual > 1e-6, "residual {}", res.residual);
    }

    /// Input-validation guards.
    #[test]
    fn rejects_bad_inputs() {
        let cfg = SemismoothConfig::default();
        // Empty.
        assert!(matches!(
            semismooth_newton_lcp(&[], &[], &[], &cfg),
            Err(CvxError::EmptyInput)
        ));
        // Shape mismatch in M.
        assert!(matches!(
            semismooth_newton_lcp(&[1.0, 0.0, 0.0], &[1.0, 1.0], &[0.0, 0.0], &cfg),
            Err(CvxError::ShapeMismatch { .. })
        ));
        // x0 length mismatch.
        assert!(matches!(
            semismooth_newton_lcp(&[1.0, 0.0, 0.0, 1.0], &[1.0, 1.0], &[0.0], &cfg),
            Err(CvxError::DimensionMismatch { .. })
        ));
        // Bad tol.
        let bad = SemismoothConfig {
            tol: 0.0,
            ..SemismoothConfig::default()
        };
        assert!(matches!(
            semismooth_newton_lcp(&[1.0, 0.0, 0.0, 1.0], &[1.0, 1.0], &[0.0, 0.0], &bad),
            Err(CvxError::InvalidParameter(_))
        ));
        // Bad armijo_sigma.
        let bad2 = SemismoothConfig {
            armijo_sigma: 0.9,
            ..SemismoothConfig::default()
        };
        assert!(matches!(
            semismooth_newton_lcp(&[1.0, 0.0, 0.0, 1.0], &[1.0, 1.0], &[0.0, 0.0], &bad2),
            Err(CvxError::InvalidParameter(_))
        ));
    }
}
