//! Tseng's forward-backward-forward (FBF) splitting (Tseng, 2000).
//!
//! # Problem
//!
//! Find a zero of the sum of two maximal-monotone operators
//!
//! ```text
//!   0 ∈ A x + B x,
//! ```
//!
//! where `A` is maximal monotone and accessed through its *resolvent*
//! `J_{γA} = (I + γA)⁻¹` (for `A = ∂f` this is the proximal operator
//! `prox_{γf}`), and `B` is monotone and `L`-Lipschitz but accessed only through
//! forward evaluation `x ↦ B x`.
//!
//! # Iteration
//!
//! Plain forward-backward `x_{k+1} = J_{γA}(x_k − γ B x_k)` requires `B` to be
//! *cocoercive* to converge; for a merely monotone-Lipschitz `B` (e.g. a
//! skew-symmetric / saddle-point operator) it can diverge.  Tseng's method adds
//! a second forward step that restores convergence for any `γ ∈ (0, 1/L)`:
//!
//! ```text
//!   y_k     = J_{γA}(x_k − γ B x_k),         (backward / resolvent step)
//!   x_{k+1} = y_k − γ (B y_k − B x_k).       (extra forward correction)
//! ```
//!
//! Both `B x_k` and `B y_k` are needed per iteration (two forward evaluations,
//! one resolvent).  The fixed points of the map are exactly the zeros of
//! `A + B`.
//!
//! # Why the extra step matters
//!
//! For a skew operator `B = [[0, 1], [-1, 0]]` (monotone, `L = 1`, **not**
//! cocoercive) and `A = 0`, plain forward-backward has iteration matrix
//! `I − γB` with eigenvalues `1 ∓ iγ` of modulus `√(1 + γ²) > 1`, so it spirals
//! outward for every `γ > 0`.  Tseng's correction contracts toward the unique
//! solution `x = 0`.  The tests exercise exactly this contrast.
//!
//! # Application to convex-concave saddle points
//!
//! For `min_x max_y Φ(x, y)` with `Φ` convex-concave, the variational operator
//! `B(x, y) = (∇_x Φ, −∇_y Φ)` is monotone; when `Φ` is bilinear it is skew and
//! not cocoercive, the canonical setting where FBF beats forward-backward.
//!
//! # References
//!
//! * P. Tseng, *A modified forward-backward splitting method for maximal
//!   monotone mappings*, SIAM Journal on Control and Optimization 38 (2000),
//!   431-446.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Termination status of [`tseng_fbf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsengStatus {
    /// The step `‖x_{k+1} − x_k‖` fell below the tolerance.
    Converged,
    /// The iteration cap was reached first.
    MaxIterReached,
}

/// Tuning parameters for Tseng's FBF splitting.
#[derive(Debug, Clone, Copy)]
pub struct TsengConfig {
    /// Step size `γ`.  Convergence requires `γ ∈ (0, 1/L)` for an `L`-Lipschitz
    /// operator `B`.
    pub gamma: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖x_{k+1} − x_k‖₂`.
    pub tol: f64,
}

impl Default for TsengConfig {
    fn default() -> Self {
        Self {
            gamma: 0.5,
            max_iter: 5000,
            tol: 1.0e-10,
        }
    }
}

/// Result of [`tseng_fbf`].
#[derive(Debug, Clone)]
pub struct TsengResult {
    /// The solution estimate `x` (a zero of `A + B`).
    pub x: Vec<f64>,
    /// Final step residual `‖x_{k+1} − x_k‖₂`.
    pub residual: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Termination status.
    pub status: TsengStatus,
}

/// Tseng's forward-backward-forward splitting for `0 ∈ A x + B x`.
///
/// # Arguments
/// * `x0` – starting point (length `n`).
/// * `resolvent_a` – resolvent `J_{γA}`: `(point, γ) ↦ (I + γA)⁻¹(point)`.
///   For `A = ∂f` pass `prox_{γf}`; for `A = 0` pass the identity map.
/// * `op_b` – the single-valued monotone-Lipschitz operator `B`: `x ↦ B x`.
/// * `config` – algorithmic parameters (see [`TsengConfig`]).
///
/// # Errors
/// * [`CvxError::EmptyInput`] if `x0` is empty.
/// * [`CvxError::InvalidParameter`] if `γ ≤ 0` or `tol ≤ 0`.
/// * [`CvxError::DimensionMismatch`] if any operator returns a wrong-length
///   vector.
pub fn tseng_fbf<RA, OB>(
    x0: &[f64],
    resolvent_a: RA,
    op_b: OB,
    config: &TsengConfig,
) -> CvxResult<TsengResult>
where
    RA: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    OB: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !config.gamma.is_finite() || config.gamma <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "gamma must be > 0, got {}",
            config.gamma
        )));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            config.tol
        )));
    }

    let n = x0.len();
    let gamma = config.gamma;
    let mut x = x0.to_vec();
    let mut residual = f64::INFINITY;
    let mut status = TsengStatus::MaxIterReached;
    let mut iterations = 0_usize;

    for _ in 0..config.max_iter {
        iterations += 1;

        // Forward evaluation B x_k.
        let bx = op_b(&x)?;
        if bx.len() != n {
            return Err(CvxError::DimensionMismatch { a: bx.len(), b: n });
        }

        // Backward step: y = J_{γA}(x − γ B x).
        let mut arg = vec![0.0_f64; n];
        for i in 0..n {
            arg[i] = x[i] - gamma * bx[i];
        }
        let y = resolvent_a(&arg, gamma)?;
        if y.len() != n {
            return Err(CvxError::DimensionMismatch { a: y.len(), b: n });
        }

        // Second forward evaluation B y.
        let by = op_b(&y)?;
        if by.len() != n {
            return Err(CvxError::DimensionMismatch { a: by.len(), b: n });
        }

        // Correction: x_{k+1} = y − γ (B y − B x).
        let mut x_new = vec![0.0_f64; n];
        let mut step = vec![0.0_f64; n];
        for i in 0..n {
            x_new[i] = y[i] - gamma * (by[i] - bx[i]);
            step[i] = x_new[i] - x[i];
        }

        residual = norm2(&step);
        x = x_new;

        if residual < config.tol {
            status = TsengStatus::Converged;
            break;
        }
    }

    Ok(TsengResult {
        x,
        residual,
        iterations,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l2::prox_l2;

    /// Identity resolvent (for `A = 0`).
    fn identity_resolvent(v: &[f64], _g: f64) -> CvxResult<Vec<f64>> {
        Ok(v.to_vec())
    }

    /// Skew operator `B(x) = [x₁, −x₀]` (rotation generator): monotone,
    /// `L = 1`, **not** cocoercive.  Its only zero is the origin.
    fn skew_op(x: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(vec![x[1], -x[0]])
    }

    /// One step of *plain* forward-backward with the identity resolvent, used to
    /// demonstrate divergence on the skew operator.
    fn plain_fb_step(x: &[f64], gamma: f64) -> Vec<f64> {
        // x ← x − γ B x  (no correction).
        let b = [x[1], -x[0]];
        vec![x[0] - gamma * b[0], x[1] - gamma * b[1]]
    }

    /// Tseng FBF converges to the zero of a skew (non-cocoercive) operator.
    #[test]
    fn solves_skew_monotone_inclusion() {
        let cfg = TsengConfig {
            gamma: 0.5,
            max_iter: 20000,
            tol: 1e-11,
        };
        let res = tseng_fbf(&[3.0, -2.0], identity_resolvent, skew_op, &cfg).expect("fbf");
        assert_eq!(res.status, TsengStatus::Converged);
        assert!(norm2(&res.x) < 1e-9, "‖x‖ = {}", norm2(&res.x));
    }

    /// On the *same* skew operator, plain forward-backward diverges (its iterate
    /// norm strictly grows) for every positive step — the property Tseng's extra
    /// forward step repairs.
    #[test]
    fn plain_forward_backward_diverges_where_tseng_converges() {
        let gamma = 0.5_f64;
        // Plain FB: norm grows by factor √(1+γ²) each step.
        let mut x = vec![1.0_f64, 0.0];
        let n0 = norm2(&x);
        for _ in 0..50 {
            x = plain_fb_step(&x, gamma);
        }
        let n_plain = norm2(&x);
        assert!(
            n_plain > 10.0 * n0,
            "plain FB should diverge, ‖x‖ grew {n0} → {n_plain}"
        );

        // Tseng FBF on the identical problem from the identical start: converges.
        let cfg = TsengConfig {
            gamma,
            max_iter: 20000,
            tol: 1e-11,
        };
        let res = tseng_fbf(&[1.0, 0.0], identity_resolvent, skew_op, &cfg).expect("fbf");
        assert_eq!(res.status, TsengStatus::Converged);
        assert!(norm2(&res.x) < 1e-9, "Tseng ‖x‖ = {}", norm2(&res.x));
    }

    /// Bilinear saddle point `min_x max_y x·y` has the unique saddle `(0, 0)`.
    /// The variational operator `B(x, y) = (∂_x, −∂_y)(x y) = (y, −x)` is skew.
    #[test]
    fn solves_bilinear_saddle_point() {
        let saddle_op = |z: &[f64]| -> CvxResult<Vec<f64>> {
            // z = (x, y); B = (∂Φ/∂x, −∂Φ/∂y) = (y, −x).
            Ok(vec![z[1], -z[0]])
        };
        let cfg = TsengConfig {
            gamma: 0.4,
            max_iter: 20000,
            tol: 1e-11,
        };
        let res = tseng_fbf(&[5.0, -4.0], identity_resolvent, saddle_op, &cfg).expect("fbf");
        assert_eq!(res.status, TsengStatus::Converged);
        assert!(res.x[0].abs() < 1e-9, "x = {}", res.x[0]);
        assert!(res.x[1].abs() < 1e-9, "y = {}", res.x[1]);
    }

    /// FBF with a *non-trivial* resolvent: `A = ∂(½β‖·‖²)` (so
    /// `J_{γA}(v) = v/(1+γβ) = prox_{γ·(β/2)‖·‖²}`) plus the skew `B`.
    ///
    /// The inclusion `0 ∈ β x + B x` with skew `B` has matrix `βI + B`, which is
    /// nonsingular (its symmetric part `βI ≻ 0`), so the unique zero is `x = 0`.
    #[test]
    fn solves_inclusion_with_resolvent_and_skew() {
        let beta = 1.5_f64;
        let resolvent = move |v: &[f64], g: f64| prox_l2(v, g * beta);
        let cfg = TsengConfig {
            gamma: 0.5,
            max_iter: 20000,
            tol: 1e-11,
        };
        let res = tseng_fbf(&[2.0, 3.0], &resolvent, skew_op, &cfg).expect("fbf");
        assert_eq!(res.status, TsengStatus::Converged);
        assert!(norm2(&res.x) < 1e-9, "‖x‖ = {}", norm2(&res.x));
    }

    /// An affine monotone operator with a *nonzero* solution.
    ///
    /// `B(x) = G x + c` with `G` positive-definite-plus-skew
    /// `G = [[1, 2], [−2, 1]]` and `c = [−3, 1]`.  `B` is monotone (symmetric
    /// part `I ≻ 0`) and Lipschitz.  The zero solves `G x = −c`, i.e.
    /// `x* = G⁻¹(−c)`.  With `det G = 1·1 − 2·(−2) = 5`,
    /// `G⁻¹ = (1/5)[[1, −2], [2, 1]]`, so `x* = (1/5)[[1,−2],[2,1]]·[3,−1]
    ///        = (1/5)[3+2, 6−1] = [1, 1]`.
    #[test]
    fn solves_affine_monotone_nonzero_solution() {
        let g_op = |x: &[f64]| -> CvxResult<Vec<f64>> {
            // G x + c.
            let gx0 = 1.0 * x[0] + 2.0 * x[1];
            let gx1 = -2.0 * x[0] + 1.0 * x[1];
            Ok(vec![gx0 - 3.0, gx1 + 1.0])
        };
        // ‖G‖₂ ≤ √(‖G‖₁‖G‖∞) = √(3·3) = 3 ⇒ pick γ < 1/3.
        let cfg = TsengConfig {
            gamma: 0.3,
            max_iter: 50000,
            tol: 1e-12,
        };
        let res = tseng_fbf(&[0.0, 0.0], identity_resolvent, g_op, &cfg).expect("fbf");
        assert_eq!(res.status, TsengStatus::Converged);
        assert!((res.x[0] - 1.0).abs() < 1e-7, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-7, "x1 = {}", res.x[1]);
        // Verify the residual ‖B x*‖ ≈ 0.
        let b = g_op(&res.x).expect("b");
        assert!(norm2(&b) < 1e-6, "‖B x*‖ = {}", norm2(&b));
    }

    /// The fixed point genuinely satisfies `0 ∈ A x + B x`.  With `A = ∂ι_{≥0}`
    /// (projection onto the non-negative orthant) and a strongly-monotone
    /// `B(x) = x − a`, the solution is `max(a, 0)` componentwise.
    #[test]
    fn fixed_point_is_zero_of_sum() {
        let a = vec![2.0_f64, -1.0, 0.5];
        // Resolvent of A = ∂ι_{≥0} is projection onto the non-negative orthant.
        let proj_nn = |v: &[f64], _g: f64| -> CvxResult<Vec<f64>> {
            Ok(v.iter().map(|vi| vi.max(0.0)).collect())
        };
        // B(x) = x − a (cocoercive here, but FBF must still be correct).
        let b_op = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(a.iter()).map(|(xi, ai)| xi - ai).collect())
        };
        let cfg = TsengConfig {
            gamma: 0.5,
            max_iter: 20000,
            tol: 1e-12,
        };
        let res = tseng_fbf(&[0.0, 0.0, 0.0], &proj_nn, &b_op, &cfg).expect("fbf");
        // Expected componentwise max(a, 0) = [2, 0, 0.5].
        let expected = [2.0_f64, 0.0, 0.5];
        for (xi, ei) in res.x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < 1e-6, "x {xi} vs {ei}");
        }
    }

    /// Input-validation guards.
    #[test]
    fn rejects_bad_inputs() {
        let cfg = TsengConfig::default();
        // Empty.
        assert!(matches!(
            tseng_fbf(&[], identity_resolvent, skew_op, &cfg),
            Err(CvxError::EmptyInput)
        ));
        // Bad gamma.
        let bad = TsengConfig {
            gamma: -1.0,
            ..TsengConfig::default()
        };
        assert!(matches!(
            tseng_fbf(&[1.0, 1.0], identity_resolvent, skew_op, &bad),
            Err(CvxError::InvalidParameter(_))
        ));
        // Wrong-length op output.
        let bad_b = |_x: &[f64]| Ok(vec![0.0, 0.0, 0.0]);
        assert!(matches!(
            tseng_fbf(&[1.0, 1.0], identity_resolvent, bad_b, &cfg),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }
}
