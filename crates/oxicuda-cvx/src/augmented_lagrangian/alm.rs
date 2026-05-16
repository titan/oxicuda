//! Augmented Lagrangian / Method of Multipliers.
//!
//! For `min f(x)  s.t. A x = b`, iterates:
//!   x_{k+1} = argmin_x  f(x) + λ_k · (Ax − b) + ρ_k/2 ||Ax − b||²
//!   λ_{k+1} = λ_k + ρ_k (A x_{k+1} − b)
//!   (optionally) ρ_{k+1} = γ · ρ_k.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_vec, norm2};

/// ALM solver. `inner_solve(lambda, rho)` produces the new x for given multiplier/penalty.
pub fn augmented_lagrangian<S>(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    inner_solve: S,
    rho0: f64,
    rho_growth: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<AlmResult>
where
    S: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    if rho0 <= 0.0 || !rho0.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "ALM rho0 > 0, got {rho0}"
        )));
    }
    if rho_growth < 1.0 || !rho_growth.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "ALM rho_growth ≥ 1, got {rho_growth}"
        )));
    }
    let mut lambda = vec![0.0_f64; m];
    let mut rho = rho0;
    let mut x = vec![0.0_f64; n];
    let mut iters = 0usize;
    let mut residual = 0.0_f64;
    for it in 0..max_iter {
        x = inner_solve(&lambda, rho)?;
        if x.len() != n {
            return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
        }
        let ax = mat_vec(a, m, n, &x)?;
        let mut r = vec![0.0_f64; m];
        for i in 0..m {
            r[i] = ax[i] - b[i];
        }
        for i in 0..m {
            lambda[i] += rho * r[i];
        }
        residual = norm2(&r);
        iters = it + 1;
        if residual < tol {
            break;
        }
        rho *= rho_growth;
    }
    Ok(AlmResult {
        x,
        lambda,
        rho,
        iter: iters,
        residual,
    })
}

/// ALM result.
#[derive(Debug, Clone)]
pub struct AlmResult {
    pub x: Vec<f64>,
    pub lambda: Vec<f64>,
    pub rho: f64,
    pub iter: usize,
    pub residual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alm_quadratic_equality() {
        // min 0.5 ||x||² s.t. x_1 + x_2 = 1.  Analytic: x* = [0.5, 0.5], λ* = -0.5.
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let inner = |lambda: &[f64], rho: f64| -> CvxResult<Vec<f64>> {
            // grad of augmented Lagrangian:
            //   x_i + λ * a_i + rho * (a·x − b) a_i = 0 ∀ i.
            // → x_i (1 + rho a_i²) = - a_i (λ + rho · (a·x_- b))
            // For a = [1,1], system: (1 + rho) x + rho 1 1^T x = -(λ + rho·something), simplest:
            // overall linear: M x = -a (λ - rho b)/(1 - simplified).  We solve via direct 2x2.
            // M = I + rho a a^T → invert.  a a^T = ones(2,2).  M = (1+rho) I except off-diag = rho.
            // Inverse via Sherman-Morrison: M^{-1} = I - rho/(1 + 2 rho) * a a^T.
            // RHS = -(λ + rho * 0) * a = -λ a + rho b a = (rho b − λ) a.
            // Wait: x − a^T λ - rho a (a^T x − b) = 0 (Lagrangian gradient with augmented).
            // Solve as: (I + rho a a^T) x = a (rho b - λ).
            // Then SM: x = a (rho b - λ) - rho a (a^T a)(rho b - λ)/(1 + rho a^T a)
            //          = a (rho b - λ) (1 - 2 rho/(1+2 rho))
            //          = a (rho b - λ) / (1 + 2 rho).
            let s = (rho * b[0] - lambda[0]) / (1.0 + 2.0 * rho);
            Ok(vec![s * a[0], s * a[1]])
        };
        let res = augmented_lagrangian(&a, 1, 2, &b, inner, 1.0, 2.0, 30, 1.0e-8).expect("ok");
        assert!((res.x[0] - 0.5).abs() < 1.0e-5);
        assert!((res.x[1] - 0.5).abs() < 1.0e-5);
        // Check feasibility.
        assert!((res.x[0] + res.x[1] - 1.0).abs() < 1.0e-5);
    }
}
