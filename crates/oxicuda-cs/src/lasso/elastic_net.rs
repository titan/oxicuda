//! Elastic Net (Zou-Hastie 2005).
//!
//! Objective: `½ ||Φ x − y||² + λ_1 ||x||_1 + (λ_2/2) ||x||²`.
//! Solved by coordinate descent: per-coordinate prox is
//!   `x_j ← soft_threshold(z_j, λ_1) / (||Φ_j||²/n + λ_2)` after centering / standardising.

use crate::error::{CsError, CsResult};

/// Cyclic coordinate descent for elastic net.
pub fn elastic_net(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    lambda_1: f64,
    lambda_2: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<Vec<f64>> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if lambda_1 < 0.0 || lambda_2 < 0.0 {
        return Err(CsError::InvalidParameter("lambdas must be ≥ 0".into()));
    }
    let mut col_norm_sq = vec![0.0_f64; n];
    for i in 0..m {
        for j in 0..n {
            let v = phi[i * n + j];
            col_norm_sq[j] += v * v;
        }
    }
    let mut x = vec![0.0_f64; n];
    let mut r = y.to_vec();
    let m_f = m as f64;
    for _ in 0..max_iter {
        let mut max_delta = 0.0_f64;
        for j in 0..n {
            let cn = col_norm_sq[j];
            if cn < 1.0e-300 {
                continue;
            }
            let xj_old = x[j];
            for i in 0..m {
                r[i] += phi[i * n + j] * xj_old;
            }
            let mut zj = 0.0_f64;
            for i in 0..m {
                zj += phi[i * n + j] * r[i];
            }
            zj /= m_f;
            let denom = cn / m_f + lambda_2;
            let xj_new = if zj > lambda_1 {
                (zj - lambda_1) / denom
            } else if zj < -lambda_1 {
                (zj + lambda_1) / denom
            } else {
                0.0
            };
            x[j] = xj_new;
            for i in 0..m {
                r[i] -= phi[i * n + j] * xj_new;
            }
            let d = (xj_new - xj_old).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < tol {
            break;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_net_runs() {
        let phi = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![1.0, 1.0];
        let x = elastic_net(&phi, 2, 2, &y, 0.05, 0.1, 200, 1.0e-9).expect("ok");
        assert!(x[0] > 0.0 && x[1] > 0.0);
    }
}
