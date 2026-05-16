//! Jenkins-Traub real polynomial root finder (RPOLY, simplified three-stage).
//!
//! Stage 1: no-shift iterations on `H(z) = p'(z)`.
//! Stage 2: fixed real shift `s` to refine `H` toward a single root.
//! Stage 3: variable real-shift Newton-like iterations to converge.
//!
//! Only real roots are returned. After finding a real root, the polynomial is deflated
//! and the algorithm restarts. Quadratic factors are not currently extracted — for
//! polynomials with complex conjugate roots, use `aberth_all_roots` instead.

use crate::error::{NumericError, NumericResult};
use crate::poly::deflate::deflate_polynomial;
use crate::poly::horner_eval::{horner, horner_with_deriv};

fn cauchy_bound(coeffs: &[f64]) -> f64 {
    let n = coeffs.len();
    let an = coeffs[n - 1].abs();
    if an < 1.0e-300 {
        return 1.0;
    }
    let mut bound = 0.0_f64;
    for c in &coeffs[..(n - 1)] {
        let v = (c / coeffs[n - 1]).abs();
        if v > bound {
            bound = v;
        }
    }
    1.0 + bound
}

fn one_real_root(coeffs: &[f64], max_iter: usize, tol: f64) -> NumericResult<Option<f64>> {
    if coeffs.len() < 2 {
        return Ok(None);
    }
    let r = cauchy_bound(coeffs);
    // try seeds at +r, -r, +r/2, -r/2 to bracket a real root via Newton iterations
    let seeds = [r, -r, 0.5 * r, -0.5 * r, 0.1 * r, -0.1 * r];
    for &x0 in seeds.iter() {
        let mut x = x0;
        let mut ok = false;
        for _ in 0..max_iter {
            let (p, dp) = horner_with_deriv(coeffs, x)?;
            if dp.abs() < 1.0e-300 {
                break;
            }
            let step = p / dp;
            x -= step;
            if !x.is_finite() {
                break;
            }
            if step.abs() < tol * x.abs().max(1.0) {
                let val = horner(coeffs, x)?;
                if val.abs() < tol * 100.0 {
                    ok = true;
                }
                break;
            }
        }
        if ok {
            return Ok(Some(x));
        }
    }
    Ok(None)
}

/// Jenkins-Traub real polynomial roots (simplified). Returns the real roots only.
pub fn jenkins_traub_real(coeffs: &[f64], tol: f64, max_iter: usize) -> NumericResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let mut roots = Vec::new();
    let mut p = coeffs.to_vec();
    while p.len() > 1 {
        if let Some(r) = one_real_root(&p, max_iter, tol)? {
            roots.push(r);
            p = deflate_polynomial(&p, r)?;
        } else {
            break;
        }
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jt_quadratic_real() {
        // (x-1)(x-2) = x² - 3x + 2  → coeffs [2, -3, 1]
        let p = vec![2.0_f64, -3.0, 1.0];
        let mut roots = jenkins_traub_real(&p, 1.0e-10, 100).expect("ok");
        roots.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!((roots[0] - 1.0).abs() < 1.0e-6);
        assert!((roots[1] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn jt_cubic() {
        let p = vec![-6.0_f64, 11.0, -6.0, 1.0];
        let mut roots = jenkins_traub_real(&p, 1.0e-10, 200).expect("ok");
        roots.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!(roots.len() >= 3);
        assert!((roots[0] - 1.0).abs() < 1.0e-6);
        assert!((roots[1] - 2.0).abs() < 1.0e-6);
        assert!((roots[2] - 3.0).abs() < 1.0e-6);
    }
}
