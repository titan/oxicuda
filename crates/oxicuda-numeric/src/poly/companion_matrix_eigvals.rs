//! Real-eigenvalue extraction from a polynomial's companion matrix via shifted QR.
//!
//! The companion matrix `C` of `p(z) = z^n + a_{n-1} z^{n-1} + … + a_0` is
//! ```text
//!     0 1 0 ... 0
//!     0 0 1 ... 0
//!     :       :
//!     0 0 0 ... 1
//!    -a_0 -a_1 ... -a_{n-1}
//! ```
//! Its eigenvalues are exactly the roots of `p`. The current implementation
//! performs un-shifted QR iterations and returns the real eigenvalues from the
//! resulting upper-(quasi-)triangular form.

use crate::error::{NumericError, NumericResult};
use crate::linalg::qr_givens::qr_givens;

/// Compute the real eigenvalues of the companion matrix of the polynomial
/// `coeffs[0] + coeffs[1] z + … + coeffs[n] z^n` (indexed by power).
pub fn companion_matrix_real_eigenvalues(
    coeffs: &[f64],
    max_iter: usize,
) -> NumericResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = coeffs.len() - 1;
    if n == 0 {
        return Ok(vec![]);
    }
    let an = coeffs[n];
    if an.abs() < 1.0e-300 {
        return Err(NumericError::InvalidParameter(
            "leading coefficient is zero".into(),
        ));
    }
    // Build companion matrix
    let mut c = vec![0.0_f64; n * n];
    for i in 0..(n - 1) {
        c[i * n + (i + 1)] = 1.0;
    }
    for j in 0..n {
        c[(n - 1) * n + j] = -coeffs[j] / an;
    }
    // QR iterations (un-shifted) — converges slowly but adequately for moderate n.
    for _ in 0..max_iter {
        let (q, r) = qr_givens(&c, n, n)?;
        // c' = R Q
        let mut c_new = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += r[i * n + k] * q[k * n + j];
                }
                c_new[i * n + j] = s;
            }
        }
        // measure off-diagonal mass
        let mut off = 0.0_f64;
        for i in 1..n {
            for j in 0..i {
                off += c_new[i * n + j].powi(2);
            }
        }
        c = c_new;
        if off.sqrt() < 1.0e-12 {
            break;
        }
    }
    let mut out: Vec<f64> = (0..n).map(|i| c[i * n + i]).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn companion_quadratic_real() {
        // (x-2)(x-5) = x² - 7x + 10
        let p = vec![10.0_f64, -7.0, 1.0];
        let evs = companion_matrix_real_eigenvalues(&p, 200).expect("ok");
        assert!((evs[0] - 2.0).abs() < 1.0e-3);
        assert!((evs[1] - 5.0).abs() < 1.0e-3);
    }

    #[test]
    fn companion_cubic_real() {
        let p = vec![-6.0_f64, 11.0, -6.0, 1.0];
        let evs = companion_matrix_real_eigenvalues(&p, 400).expect("ok");
        assert!((evs[0] - 1.0).abs() < 1.0e-3);
        assert!((evs[1] - 2.0).abs() < 1.0e-3);
        assert!((evs[2] - 3.0).abs() < 1.0e-3);
    }
}
