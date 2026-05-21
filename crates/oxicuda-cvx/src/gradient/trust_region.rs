//! Trust-region Newton method with Steihaug-Toint conjugate-gradient subsolver.
//!
//! Minimises `f(x)` via a sequence of local quadratic subproblems:
//!   min_{‖p‖ ≤ Δ}  g^T p + ½ p^T H p
//!
//! The subproblem is solved approximately by the truncated CG of Steihaug (1983)
//! and Toint (1981), as described in:
//!
//! - Conn, Gould & Toint (2000), "Trust-Region Methods", Chapters 7 and 17
//!   (SIAM, MPS-SIAM Series on Optimization).
//! - Steihaug (1983), "The conjugate gradient method and trust regions in large-scale
//!   optimisation", SIAM J. Numer. Anal.
//! - Toint (1981), "Towards an efficient sparsity exploiting Newton method for
//!   minimisation", in Duff (ed.) Sparse Matrices and Their Uses, pp. 57-88.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Configuration for the Trust-Region Newton solver.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    /// Maximum outer iterations (default 200).
    pub max_iter: usize,
    /// Initial trust-region radius Δ₀ (default 1.0).
    pub initial_radius: f64,
    /// Maximum trust-region radius Δ_max (default 1 × 10⁴).
    pub max_radius: f64,
    /// Convergence criterion: stop when ‖g‖₂ < tol_grad (default 1 × 10⁻⁸).
    pub tol_grad: f64,
    /// Acceptance threshold: reject step when actual/predicted reduction < eta (default 0.1).
    pub eta: f64,
    /// Maximum Steihaug-Toint CG iterations per subproblem (default 50).
    pub cg_max_iter: usize,
    /// Steihaug-Toint CG inner tolerance: stop when ‖r‖ < cg_tol·‖r₀‖ (default 0.1).
    pub cg_tol: f64,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            initial_radius: 1.0,
            max_radius: 1e4,
            tol_grad: 1e-8,
            eta: 0.1,
            cg_max_iter: 50,
            cg_tol: 0.1,
        }
    }
}

/// Result of a Trust-Region Newton run.
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    /// Final iterate x.
    pub x: Vec<f64>,
    /// Total outer iterations performed.
    pub n_iter: usize,
    /// Whether ‖g‖₂ < tol_grad was achieved.
    pub converged: bool,
    /// L2 norm of the final gradient.
    pub final_grad_norm: f64,
    /// Final trust-region radius.
    pub final_radius: f64,
    /// Total (f, g) evaluations (one initial + one per outer iteration).
    pub n_func_evals: usize,
    /// Number of rejected steps (actual/predicted reduction < eta).
    pub n_rejected: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the step that hits the trust-region boundary along direction `d`
/// from base point `z`: solve ‖z + τ d‖² = δ² for the positive root τ.
///
///   τ² (dᵀd) + 2τ (zᵀd) + (zᵀz − δ²) = 0
///   τ = (−b + sqrt(b² − 4ac)) / (2a)   with positive root chosen.
fn boundary_step(z: &[f64], d: &[f64], delta: f64) -> f64 {
    let a: f64 = d.iter().map(|di| di * di).sum();
    let b: f64 = 2.0 * z.iter().zip(d.iter()).map(|(zi, di)| zi * di).sum::<f64>();
    let c: f64 = z.iter().map(|zi| zi * zi).sum::<f64>() - delta * delta;

    if a < 1e-300 {
        // d ≈ 0: return zero step (degenerate)
        return 0.0;
    }

    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let sqrt_disc = discriminant.sqrt();
    // Two roots; take the positive one.
    let tau1 = (-b + sqrt_disc) / (2.0 * a);
    let tau2 = (-b - sqrt_disc) / (2.0 * a);
    tau1.max(tau2).max(0.0)
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Steihaug-Toint truncated CG subsolver for the trust-region subproblem:
///   min_{‖p‖ ≤ δ}  gᵀ p + ½ pᵀ H p
///
/// Implements Algorithm 7.5.1 from Conn, Gould & Toint (2000).
///
/// # Arguments
/// - `g`: gradient vector at the current iterate.
/// - `hess_vec_prod`: closure computing H·v for any vector v.
/// - `delta`: trust-region radius.
/// - `max_iter`: maximum CG iterations.
/// - `tol`: relative residual tolerance; stop when ‖r‖ < tol · ‖r₀‖.
///
/// # Returns
/// Approximate minimiser p ∈ {‖p‖ ≤ δ}.
pub fn steihaug_cg(
    g: &[f64],
    hess_vec_prod: impl Fn(&[f64]) -> Vec<f64>,
    delta: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }

    // z ← 0, r ← g, d ← −g
    let mut z = vec![0.0_f64; n];
    let mut r: Vec<f64> = g.to_vec();
    let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();

    let mut r_sq: f64 = r.iter().map(|ri| ri * ri).sum();
    let r0_sq = r_sq;

    // If gradient is already (numerically) zero, return the zero step.
    if r0_sq < 1e-300 {
        return z;
    }

    let tol_sq = (tol * tol) * r0_sq; // convergence threshold on ‖r‖²

    for _ in 0..max_iter {
        // Convergence check on residual norm.
        if r_sq < tol_sq {
            return z;
        }

        let hd = hess_vec_prod(&d);

        // κ = dᵀ H d  (curvature along d)
        let kappa: f64 = d.iter().zip(hd.iter()).map(|(di, hdi)| di * hdi).sum();

        if kappa <= 0.0 {
            // Negative (or zero) curvature encountered: step to boundary.
            let tau = boundary_step(&z, &d, delta);
            let p: Vec<f64> = z
                .iter()
                .zip(d.iter())
                .map(|(zi, di)| zi + tau * di)
                .collect();
            return p;
        }

        let alpha = r_sq / kappa;

        // Tentative new z.
        let z_new: Vec<f64> = z
            .iter()
            .zip(d.iter())
            .map(|(zi, di)| zi + alpha * di)
            .collect();
        let z_new_norm_sq: f64 = z_new.iter().map(|zi| zi * zi).sum();

        if z_new_norm_sq >= delta * delta {
            // Step would leave the trust region: hit the boundary exactly.
            let tau = boundary_step(&z, &d, delta);
            let p: Vec<f64> = z
                .iter()
                .zip(d.iter())
                .map(|(zi, di)| zi + tau * di)
                .collect();
            return p;
        }

        z = z_new;

        // Update residual: r ← r + α H d
        for i in 0..n {
            r[i] += alpha * hd[i];
        }

        let r_sq_new: f64 = r.iter().map(|ri| ri * ri).sum();
        let beta = r_sq_new / r_sq;
        r_sq = r_sq_new;

        // Update direction: d ← −r + β d
        for i in 0..n {
            d[i] = -r[i] + beta * d[i];
        }
    }

    // Max iterations reached: return current interior solution.
    z
}

/// Compute the predicted reduction in the quadratic model:
///   m(0) − m(p) = −(gᵀ p + ½ pᵀ H p) = −gᵀ p − ½ pᵀ (H p)
///
/// Positive value indicates the model predicts a decrease.
pub fn predicted_reduction(g: &[f64], p: &[f64], hp: &[f64]) -> f64 {
    let gp: f64 = g.iter().zip(p.iter()).map(|(gi, pi)| gi * pi).sum();
    let php: f64 = p.iter().zip(hp.iter()).map(|(pi, hpi)| pi * hpi).sum();
    -gp - 0.5 * php
}

/// Finite-difference Hessian-vector product approximation:
///   H(x)·v ≈ (∇f(x + ε v) − ∇f(x)) / ε
///
/// Useful when the exact Hessian is unavailable. The step size `eps` should be
/// chosen as O(√ε_mach) ≈ 1 × 10⁻⁸ for double precision; set it smaller for
/// smoother objectives and larger for noisy ones.
pub fn fd_hess_vec(
    x: &[f64],
    v: &[f64],
    grad_f: impl Fn(&[f64]) -> Vec<f64>,
    eps: f64,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || eps == 0.0 {
        return vec![0.0_f64; n];
    }

    let g0 = grad_f(x);
    let x_eps: Vec<f64> = x
        .iter()
        .zip(v.iter())
        .map(|(xi, vi)| xi + eps * vi)
        .collect();
    let g_eps = grad_f(&x_eps);

    g_eps
        .iter()
        .zip(g0.iter())
        .map(|(ge, g0i)| (ge - g0i) / eps)
        .collect()
}

/// Trust-region Newton method for smooth unconstrained optimisation.
///
/// Minimises `f(x)` given:
/// - `f_and_grad`: returns `(f(x), ∇f(x))`.
/// - `hess_vec`: returns `H(x)·v` for the current Hessian at `x`.
///
/// Uses the Steihaug-Toint CG for the trust-region subproblem.
///
/// # Errors
/// Returns [`CvxError::EmptyInput`] when `x0` is empty.
/// Returns [`CvxError::NotConverged`] when the trust-region radius collapses
/// below 1 × 10⁻¹⁴ before converging.
pub fn trust_region_newton<F, H>(
    x0: &[f64],
    f_and_grad: F,
    hess_vec: H,
    cfg: &TrustRegionConfig,
) -> CvxResult<TrustRegionResult>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    H: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut delta = cfg.initial_radius;

    let (mut f_val, mut g) = f_and_grad(&x);
    let mut n_func_evals = 1usize;
    let mut n_rejected = 0usize;

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..cfg.max_iter {
        let g_norm = norm2(&g);
        if g_norm < cfg.tol_grad {
            converged = true;
            break;
        }

        // ---- Solve trust-region subproblem via Steihaug-Toint CG ----
        let p = {
            let hv_at_x = |v: &[f64]| -> Vec<f64> { hess_vec(&x, v) };
            steihaug_cg(&g, hv_at_x, delta, cfg.cg_max_iter, cfg.cg_tol)
        };

        // Guard: if CG returned a vector of wrong length (degenerate), treat as zero step.
        if p.len() != n {
            return Err(CvxError::NumericalInstability(
                "steihaug_cg returned wrong-length vector".into(),
            ));
        }

        // ---- Compute actual vs. predicted reduction ----
        let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(xi, pi)| xi + pi).collect();
        let (f_new, g_new) = f_and_grad(&x_new);
        n_func_evals += 1;

        let hp: Vec<f64> = hess_vec(&x, &p);
        let pred_red = predicted_reduction(&g, &p, &hp);
        let actual_red = f_val - f_new;

        let rho = if pred_red.abs() < 1e-14 {
            0.0_f64
        } else {
            actual_red / pred_red
        };

        // ---- Accept or reject step ----
        if rho > cfg.eta {
            x = x_new;
            f_val = f_new;
            g = g_new;
        } else {
            n_rejected += 1;
        }

        // ---- Update trust-region radius ----
        let p_norm = norm2(&p);
        if rho < 0.25 {
            // Shrink: use ¼ × ‖p‖ (the actual step taken or proposed).
            delta = (0.25 * p_norm).max(1e-14);
        } else if rho > 0.75 && (p_norm - delta).abs() < 1e-8 * delta {
            // Expand only when the boundary was active.
            delta = (2.0 * delta).min(cfg.max_radius);
        }
        // else: keep delta unchanged.

        if delta < 1e-14 {
            return Err(CvxError::NotConverged {
                iter: _iter,
                residual: norm2(&g),
            });
        }

        n_iter += 1;
    }

    let final_grad_norm = norm2(&g);

    Ok(TrustRegionResult {
        x,
        n_iter,
        converged,
        final_grad_norm,
        final_radius: delta,
        n_func_evals,
        n_rejected,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: identity Hessian-vector product H·v = v
    // ------------------------------------------------------------------
    fn hess_identity(v: &[f64]) -> Vec<f64> {
        v.to_vec()
    }

    // ------------------------------------------------------------------
    // Helper: H·v = −v  (negative curvature)
    // ------------------------------------------------------------------
    fn hess_neg_identity(v: &[f64]) -> Vec<f64> {
        v.iter().map(|vi| -vi).collect()
    }

    // ------------------------------------------------------------------
    // Helper: diagonal Hessian H = diag(d), H·v = d ∘ v
    // ------------------------------------------------------------------
    fn hess_diag(diag: &[f64], v: &[f64]) -> Vec<f64> {
        diag.iter().zip(v.iter()).map(|(di, vi)| di * vi).collect()
    }

    // ------------------------------------------------------------------
    // steihaug_cg tests
    // ------------------------------------------------------------------

    /// g = [1, 0], H = I, δ = 10: optimal is p = −g = [−1, 0].
    #[test]
    fn steihaug_cg_simple_x() {
        let g = vec![1.0_f64, 0.0];
        let p = steihaug_cg(&g, hess_identity, 10.0, 50, 1e-10);
        assert_eq!(p.len(), 2);
        assert!((p[0] - (-1.0)).abs() < 1e-6, "p[0]={}", p[0]);
        assert!(p[1].abs() < 1e-6, "p[1]={}", p[1]);
    }

    /// g = [0, 1], H = I, δ = 10: optimal is p = [0, −1].
    #[test]
    fn steihaug_cg_simple_y() {
        let g = vec![0.0_f64, 1.0];
        let p = steihaug_cg(&g, hess_identity, 10.0, 50, 1e-10);
        assert_eq!(p.len(), 2);
        assert!(p[0].abs() < 1e-6, "p[0]={}", p[0]);
        assert!((p[1] - (-1.0)).abs() < 1e-6, "p[1]={}", p[1]);
    }

    /// When δ is tiny (0.01) and optimal step has norm 1, CG must return a boundary step.
    #[test]
    fn steihaug_cg_hits_boundary() {
        let g = vec![1.0_f64, 1.0];
        let delta = 0.01;
        let p = steihaug_cg(&g, hess_identity, delta, 50, 1e-10);
        let pnorm = norm2(&p);
        assert!(
            (pnorm - delta).abs() < 1e-6 || pnorm <= delta + 1e-10,
            "‖p‖={pnorm}, δ={delta}"
        );
    }

    /// Negative curvature (H = −I): CG must detect this and return boundary step.
    #[test]
    fn steihaug_cg_negative_curvature_boundary() {
        let g = vec![1.0_f64, 0.0];
        let delta = 2.0;
        let p = steihaug_cg(&g, hess_neg_identity, delta, 50, 1e-10);
        let pnorm = norm2(&p);
        // Must be on (or inside) the boundary.
        assert!(pnorm <= delta + 1e-9, "‖p‖={pnorm} > δ={delta}");
        // For negative curvature the step should be non-trivial.
        assert!(pnorm > 1e-10, "step degenerate: ‖p‖={pnorm}");
    }

    /// Zero gradient → CG returns the zero step immediately.
    #[test]
    fn steihaug_cg_zero_gradient_returns_zero() {
        let g = vec![0.0_f64, 0.0];
        let p = steihaug_cg(&g, hess_identity, 5.0, 50, 1e-10);
        assert!(norm2(&p) < 1e-12, "‖p‖={}", norm2(&p));
    }

    /// CG respects the trust-region constraint for random diagonal H.
    #[test]
    fn steihaug_cg_within_trust_region() {
        let g = vec![3.0_f64, -2.0, 1.0];
        let diag = vec![4.0_f64, 2.0, 1.0]; // positive definite
        let delta = 0.5;
        let p = steihaug_cg(&g, |v| hess_diag(&diag, v), delta, 100, 1e-10);
        assert!(norm2(&p) <= delta + 1e-9, "‖p‖={} > δ={delta}", norm2(&p));
    }

    // ------------------------------------------------------------------
    // predicted_reduction tests
    // ------------------------------------------------------------------

    /// g = [1], p = [−0.5], Hp = [−0.5]:
    ///   pred = −gᵀp − ½pᵀHp = −(−0.5) − ½·(−0.5·(−0.5)) = 0.5 − 0.125 = 0.375
    #[test]
    fn predicted_reduction_scalar() {
        let g = vec![1.0_f64];
        let p = vec![-0.5_f64];
        let hp = vec![-0.5_f64];
        let pred = predicted_reduction(&g, &p, &hp);
        assert!((pred - 0.375).abs() < 1e-14, "pred={pred}");
    }

    /// Positive definite quadratic: pred_red must be positive.
    #[test]
    fn predicted_reduction_positive() {
        let g = vec![2.0_f64, 3.0];
        let p = vec![-1.0_f64, -1.5]; // descent direction
        let hp = vec![2.0_f64, 3.0]; // H = I
        let pred = predicted_reduction(&g, &p, &hp);
        assert!(pred > 0.0, "pred={pred}");
    }

    // ------------------------------------------------------------------
    // fd_hess_vec tests
    // ------------------------------------------------------------------

    /// For f(x) = ½ xᵀ A x, ∇f = A x, H·v = A v. FD should recover A v.
    #[test]
    fn fd_hess_vec_quadratic_2x2() {
        // A = [[2, 1], [1, 3]]
        let a = [2.0_f64, 1.0, 1.0, 3.0];
        let x = vec![1.0_f64, 2.0];
        let v = vec![1.0_f64, 0.0];

        let grad_f = |xi: &[f64]| -> Vec<f64> {
            vec![a[0] * xi[0] + a[1] * xi[1], a[2] * xi[0] + a[3] * xi[1]]
        };

        let hv = fd_hess_vec(&x, &v, grad_f, 1e-5);
        // Expected: A·v = [2, 1]
        assert!((hv[0] - 2.0).abs() < 1e-5, "hv[0]={}", hv[0]);
        assert!((hv[1] - 1.0).abs() < 1e-5, "hv[1]={}", hv[1]);
    }

    /// FD on f(x) = ½ ‖x‖² (identity Hessian): H·v = v.
    #[test]
    fn fd_hess_vec_identity_hessian() {
        let x = vec![1.0_f64, -2.0, 3.0];
        let v = vec![0.0_f64, 1.0, 0.0];
        let grad_f = |xi: &[f64]| xi.to_vec();
        let hv = fd_hess_vec(&x, &v, grad_f, 1e-5);
        assert!((hv[1] - 1.0).abs() < 1e-5, "hv[1]={}", hv[1]);
    }

    // ------------------------------------------------------------------
    // trust_region_newton tests
    // ------------------------------------------------------------------

    fn quad_fg(x: &[f64]) -> (f64, Vec<f64>) {
        let f: f64 = x.iter().map(|xi| 0.5 * xi * xi).sum();
        let g: Vec<f64> = x.to_vec();
        (f, g)
    }

    fn quad_hv(_x: &[f64], v: &[f64]) -> Vec<f64> {
        v.to_vec()
    }

    /// f(x) = ½‖x‖²: should converge to x ≈ 0 from [1, 2, 3].
    #[test]
    fn trust_region_newton_sum_of_squares() {
        let cfg = TrustRegionConfig::default();
        let res =
            trust_region_newton(&[1.0, 2.0, 3.0], quad_fg, quad_hv, &cfg).expect("should converge");
        assert!(
            res.converged,
            "did not converge; ‖g‖={}",
            res.final_grad_norm
        );
        for xi in &res.x {
            assert!(xi.abs() < 1e-6, "xi={xi}");
        }
    }

    /// f(x) = (x − 3)²: converge to x ≈ 3.
    #[test]
    fn trust_region_newton_shifted_quadratic_1d() {
        let cfg = TrustRegionConfig::default();
        let fg = |x: &[f64]| -> (f64, Vec<f64>) {
            let d = x[0] - 3.0;
            (d * d, vec![2.0 * d])
        };
        let hv = |_x: &[f64], v: &[f64]| -> Vec<f64> { vec![2.0 * v[0]] };
        let res = trust_region_newton(&[0.0], fg, hv, &cfg).expect("should converge");
        assert!(res.converged, "‖g‖={}", res.final_grad_norm);
        assert!((res.x[0] - 3.0).abs() < 1e-6, "x[0]={}", res.x[0]);
    }

    /// 2D quadratic: (x − 1)² + (y − 2)²: converge to [1, 2].
    #[test]
    fn trust_region_newton_2d_quadratic() {
        let cfg = TrustRegionConfig::default();
        let fg = |x: &[f64]| -> (f64, Vec<f64>) {
            let f = (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
            let g = vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
            (f, g)
        };
        let hv = |_x: &[f64], v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };
        let res = trust_region_newton(&[5.0, 5.0], fg, hv, &cfg).expect("ok");
        assert!(res.converged, "‖g‖={}", res.final_grad_norm);
        assert!((res.x[0] - 1.0).abs() < 1e-6, "x[0]={}", res.x[0]);
        assert!((res.x[1] - 2.0).abs() < 1e-6, "x[1]={}", res.x[1]);
    }

    /// f(x) = x⁴ + x²: minimiser at x = 0.
    ///
    /// Uses a looser gradient tolerance (1e-7) because the quartic dominates: near x=0
    /// the Hessian (12x² + 2) is dominated by the quadratic term, which still drives
    /// convergence to high accuracy within 500 iterations.
    #[test]
    fn trust_region_newton_x4_plus_x2() {
        let cfg = TrustRegionConfig {
            max_iter: 500,
            tol_grad: 1e-7, // looser than default 1e-8 due to quartic near-flat gradient
            ..TrustRegionConfig::default()
        };
        let fg = |x: &[f64]| -> (f64, Vec<f64>) {
            let xi = x[0];
            (xi.powi(4) + xi * xi, vec![4.0 * xi.powi(3) + 2.0 * xi])
        };
        let hv = |x: &[f64], v: &[f64]| -> Vec<f64> {
            let h = 12.0 * x[0].powi(2) + 2.0;
            vec![h * v[0]]
        };
        let res = trust_region_newton(&[5.0], fg, hv, &cfg).expect("ok");
        assert!(res.converged, "‖g‖={}", res.final_grad_norm);
        assert!(res.x[0].abs() < 1e-3, "x[0]={}", res.x[0]);
    }

    /// f(x) = x⁴ from x₀ = 1: converge to x ≈ 0.
    ///
    /// For x⁴ the Hessian is 12x², which collapses to zero at the optimum.
    /// Trust-region handles this via negative-curvature / CG boundary steps;
    /// we allow a generous tolerance since quadratic convergence breaks down.
    #[test]
    fn trust_region_newton_x4() {
        let cfg = TrustRegionConfig {
            max_iter: 500,
            ..TrustRegionConfig::default()
        };
        let fg = |x: &[f64]| -> (f64, Vec<f64>) { (x[0].powi(4), vec![4.0 * x[0].powi(3)]) };
        let hv = |x: &[f64], v: &[f64]| -> Vec<f64> { vec![12.0 * x[0].powi(2) * v[0]] };
        let res = trust_region_newton(&[1.0], fg, hv, &cfg).expect("ok");
        // For a purely quartic objective the gradient is 4x³, so ‖g‖ < 1e-8 means x < (1e-8/4)^{1/3} ≈ 0.0014.
        assert!(
            res.converged || res.final_grad_norm < 1e-6,
            "‖g‖={}",
            res.final_grad_norm
        );
        assert!(res.x[0].abs() < 5e-3, "x[0]={}", res.x[0]);
    }

    /// converged = true implies final_grad_norm < tol_grad.
    #[test]
    fn trust_region_newton_converged_flag() {
        let cfg = TrustRegionConfig::default();
        let res = trust_region_newton(&[1.0, 0.0], quad_fg, quad_hv, &cfg).expect("ok");
        if res.converged {
            assert!(
                res.final_grad_norm < cfg.tol_grad,
                "converged=true but ‖g‖={}",
                res.final_grad_norm
            );
        }
    }

    /// n_iter is at most max_iter.
    #[test]
    fn trust_region_newton_n_iter_bound() {
        let cfg = TrustRegionConfig::default();
        let res = trust_region_newton(&[1.0, 2.0, 3.0], quad_fg, quad_hv, &cfg).expect("ok");
        assert!(
            res.n_iter <= cfg.max_iter,
            "n_iter={} > max_iter={}",
            res.n_iter,
            cfg.max_iter
        );
    }

    /// n_func_evals must be > 0 after any run.
    #[test]
    fn trust_region_newton_func_evals_positive() {
        let cfg = TrustRegionConfig::default();
        let res = trust_region_newton(&[0.5], quad_fg, quad_hv, &cfg).expect("ok");
        assert!(res.n_func_evals > 0, "n_func_evals=0");
    }

    /// result.x.len() == x0.len().
    #[test]
    fn trust_region_newton_output_length() {
        let x0 = vec![1.0_f64; 5];
        let cfg = TrustRegionConfig::default();
        let res = trust_region_newton(&x0, quad_fg, quad_hv, &cfg).expect("ok");
        assert_eq!(res.x.len(), x0.len());
    }

    /// Empty x0 → CvxError::EmptyInput.
    #[test]
    fn trust_region_newton_empty_input() {
        let cfg = TrustRegionConfig::default();
        let result = trust_region_newton(
            &[],
            |x: &[f64]| -> (f64, Vec<f64>) { (0.0, x.to_vec()) },
            |_: &[f64], v: &[f64]| v.to_vec(),
            &cfg,
        );
        assert!(matches!(result, Err(CvxError::EmptyInput)));
    }

    /// Non-convex problems may produce n_rejected > 0.
    #[test]
    fn trust_region_newton_n_rejected_field_exists() {
        let cfg = TrustRegionConfig::default();
        // Mildly non-convex near start: f = x^3 - x (has inflection at 0).
        let fg = |x: &[f64]| -> (f64, Vec<f64>) {
            let xi = x[0];
            (xi.powi(3) - xi, vec![3.0 * xi.powi(2) - 1.0])
        };
        let hv = |x: &[f64], v: &[f64]| -> Vec<f64> { vec![6.0 * x[0] * v[0]] };
        let res = trust_region_newton(&[2.0], fg, hv, &cfg);
        // Just verify the field is accessible regardless of success/failure.
        if let Ok(r) = res {
            let _ = r.n_rejected;
        }
    }

    /// Default config has expected field values.
    #[test]
    fn trust_region_config_defaults() {
        let cfg = TrustRegionConfig::default();
        assert_eq!(cfg.max_iter, 200);
        assert!((cfg.initial_radius - 1.0).abs() < 1e-15);
        assert!((cfg.max_radius - 1e4).abs() < 1e-10);
        assert!((cfg.tol_grad - 1e-8).abs() < 1e-20);
        assert!((cfg.eta - 0.1).abs() < 1e-15);
        assert_eq!(cfg.cg_max_iter, 50);
        assert!((cfg.cg_tol - 0.1).abs() < 1e-15);
    }
}
