//! KKT, duality-gap, and convergence-rate metrics.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};

/// Primal residual `||A x − b||₂` for equality `A x = b`.
pub fn primal_residual(a: &[f64], m: usize, n: usize, x: &[f64], b: &[f64]) -> CvxResult<f64> {
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    let ax = mat_vec(a, m, n, x)?;
    let mut r = vec![0.0_f64; m];
    for i in 0..m {
        r[i] = ax[i] - b[i];
    }
    Ok(norm2(&r))
}

/// Dual residual `||∇f(x) + A^T λ||₂` (stationarity for `min f(x) s.t. Ax = b`).
///
/// `grad_fx` is the gradient of the smooth objective at x.  Convention matches Lagrangian
/// `L = f(x) + λ^T (Ax − b)`, so stationarity is `∇f + A^T λ = 0`.
pub fn dual_residual(
    a: &[f64],
    m: usize,
    n: usize,
    lambda: &[f64],
    grad_fx: &[f64],
) -> CvxResult<f64> {
    if grad_fx.len() != n || lambda.len() != m {
        return Err(CvxError::DimensionMismatch {
            a: grad_fx.len(),
            b: n,
        });
    }
    let at_lam = mat_t_vec(a, m, n, lambda)?;
    let mut r = vec![0.0_f64; n];
    for j in 0..n {
        r[j] = grad_fx[j] + at_lam[j];
    }
    Ok(norm2(&r))
}

/// Duality gap `f(x_pri) − g(λ_dual)` (assumes user-supplied primal/dual values).
#[must_use]
pub fn duality_gap(primal_value: f64, dual_value: f64) -> f64 {
    (primal_value - dual_value).abs()
}

/// KKT residual for `min f(x) s.t. A x = b, x ≥ 0` (LP/QP form).
///
/// Sums (in L2-norm) the:
///  - stationarity: ||∇f(x) − A^T λ − μ||
///  - primal:        ||A x − b||
///  - non-negativity: || max(0, -x) ||
///  - non-negativity: || max(0, -μ) ||
///  - complementarity: ||x ⊙ μ||
pub fn kkt_residual(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    x: &[f64],
    lambda: &[f64],
    mu: &[f64],
    grad_fx: &[f64],
) -> CvxResult<f64> {
    if x.len() != n || mu.len() != n || grad_fx.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }
    if lambda.len() != m || b.len() != m {
        return Err(CvxError::DimensionMismatch {
            a: lambda.len(),
            b: m,
        });
    }
    // Stationarity: ∇f + A^T λ − μ = 0 with Lagrangian L = f + λ^T (Ax−b) − μ^T x.
    let at_lam = mat_t_vec(a, m, n, lambda)?;
    let mut stationarity = vec![0.0_f64; n];
    for j in 0..n {
        stationarity[j] = grad_fx[j] + at_lam[j] - mu[j];
    }
    let pr = primal_residual(a, m, n, x, b)?;
    // Non-negativity violation.
    let neg_x: Vec<f64> = x.iter().map(|&xi| (-xi).max(0.0)).collect();
    let neg_mu: Vec<f64> = mu.iter().map(|&mi| (-mi).max(0.0)).collect();
    let comp: Vec<f64> = x.iter().zip(mu.iter()).map(|(xi, mi)| xi * mi).collect();
    let r_sq = norm2(&stationarity).powi(2)
        + pr * pr
        + norm2(&neg_x).powi(2)
        + norm2(&neg_mu).powi(2)
        + norm2(&comp).powi(2);
    Ok(r_sq.sqrt())
}

/// Estimate convergence rate from two consecutive residuals.
///
/// For sequence `r_k` with `r_{k+1} ≤ C r_k^p`, returns p ≈ log(r_{k+1}) / log(r_k).
pub fn convergence_rate(rk: f64, rkp1: f64) -> CvxResult<f64> {
    if !rk.is_finite() || !rkp1.is_finite() || rk <= 0.0 || rkp1 <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "convergence rate requires positive finite residuals, got rk={rk}, rkp1={rkp1}"
        )));
    }
    Ok(rkp1.ln() / rk.ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primal_residual_satisfied() {
        let a = vec![1.0, 1.0];
        let x = vec![0.5, 0.5];
        let b = vec![1.0];
        let r = primal_residual(&a, 1, 2, &x, &b).expect("ok");
        assert!(r.abs() < 1.0e-12);
    }

    #[test]
    fn duality_gap_basic() {
        assert!((duality_gap(5.0, 4.0) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn kkt_zero_at_optimum() {
        // min 0.5 ||x||² s.t. x_1 + x_2 = 1, x ≥ 0.  x* = [0.5, 0.5], λ* = -0.5, μ = 0.
        let a = vec![1.0, 1.0];
        let b = vec![1.0];
        let x = vec![0.5, 0.5];
        let lambda = vec![-0.5];
        let mu = vec![0.0, 0.0];
        let grad = x.clone();
        let r = kkt_residual(&a, 1, 2, &b, &x, &lambda, &mu, &grad).expect("ok");
        assert!(r < 1.0e-9);
    }

    #[test]
    fn convergence_rate_basic() {
        let p = convergence_rate(0.1, 0.01).expect("ok");
        assert!((p - 2.0).abs() < 1.0e-9);
    }
}
