//! LSQR iterative method for least-squares problems `min ||Ax - b||`.
//!
//! Paige and Saunders 1982. Numerically equivalent to applying conjugate gradients to
//! the normal equations but more stable. This implementation follows the standard
//! Golub-Kahan bidiagonalisation with QR-style Givens updates.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};

/// Solve `min ||A x - b||₂` via LSQR.
pub fn lsqr(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CsError::DimensionMismatch { a: b.len(), b: m });
    }
    if max_iter == 0 {
        return Err(CsError::InvalidParameter("max_iter == 0".into()));
    }
    let mut x = vec![0.0_f64; n];

    // Initial Golub-Kahan bidiagonalisation step.
    let beta1 = norm2(b);
    if beta1 < 1.0e-300 {
        return Ok(x);
    }
    let mut u: Vec<f64> = b.iter().map(|v| v / beta1).collect();
    let mut v = mat_t_vec(a, m, n, &u)?;
    let mut alpha = norm2(&v);
    if alpha < 1.0e-300 {
        return Ok(x);
    }
    for vi in v.iter_mut() {
        *vi /= alpha;
    }
    let mut w = v.clone();
    let mut phi_bar = beta1;
    let mut rho_bar = alpha;
    for _iter in 0..max_iter {
        // Continue bidiagonalisation:
        //  beta_new * u_new = A v - alpha * u
        let av = mat_vec(a, m, n, &v)?;
        let mut u_next = vec![0.0_f64; m];
        for i in 0..m {
            u_next[i] = av[i] - alpha * u[i];
        }
        let beta = norm2(&u_next);
        if beta > 1.0e-300 {
            for ui in u_next.iter_mut() {
                *ui /= beta;
            }
        }
        //  alpha_new * v_new = A^T u_new - beta * v
        let mut v_next = mat_t_vec(a, m, n, &u_next)?;
        for j in 0..n {
            v_next[j] -= beta * v[j];
        }
        let alpha_new = norm2(&v_next);
        if alpha_new > 1.0e-300 {
            for vi in v_next.iter_mut() {
                *vi /= alpha_new;
            }
        }

        // Givens rotation:
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        if rho < 1.0e-300 {
            return Err(CsError::NumericalInstability("LSQR rho underflow".into()));
        }
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha_new;
        rho_bar = -c * alpha_new;
        let phi = c * phi_bar;
        phi_bar *= s;

        let scale_w = phi / rho;
        for j in 0..n {
            x[j] += scale_w * w[j];
            w[j] = v_next[j] - (theta / rho) * w[j];
        }
        u = u_next;
        v = v_next;
        alpha = alpha_new;
        if phi_bar.abs() / beta1 < tol {
            return Ok(x);
        }
        if beta < 1.0e-300 || alpha_new < 1.0e-300 {
            return Ok(x);
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsqr_solves_square() {
        // A = [[3, 2], [1, 4]], b = [7, 9]; x = [1, 2]
        let a = vec![3.0, 2.0, 1.0, 4.0];
        let b = vec![7.0, 9.0];
        let x = lsqr(&a, 2, 2, &b, 200, 1.0e-12).expect("ok");
        let ax = mat_vec(&a, 2, 2, &x).expect("ok");
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1.0e-6,
                "ax[{i}]={}, b[{i}]={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn lsqr_least_squares() {
        // Overdetermined: A 3×2 of rank 2; x∗ = [1, 1].
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 2.0];
        let x = lsqr(&a, 3, 2, &b, 200, 1.0e-12).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-4);
        assert!((x[1] - 1.0).abs() < 1.0e-4);
    }
}
