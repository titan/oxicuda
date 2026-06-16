//! Broyden's method — quasi-Newton root finding with rank-1 updates.
//!
//! Solves `F(x) = 0` without computing the Jacobian at every step.  Instead it
//! maintains an approximation to the **inverse** Jacobian `B ≈ J⁻¹` and updates
//! it after each step with Broyden's "good" rank-1 formula via the
//! Sherman-Morrison identity:
//!
//! ```text
//! s = x_{k+1} − x_k          (step)
//! y = F_{k+1} − F_k          (residual change)
//! B ← B + ((s − B y) / (sᵀ B y)) · (sᵀ B)
//! ```
//!
//! The Newton-like step is then `x_{k+1} = x_k − B · F_k`.  The inverse update
//! keeps every iteration at `O(n²)` cost with no linear solves, giving
//! super-linear convergence near a simple root.

use crate::error::{NumericError, NumericResult};

/// Solves `F(x) = 0` by Broyden's method (good update, inverse-Jacobian form).
///
/// `f` evaluates the residual vector.  `x0` is the initial guess; the result
/// has the same length.  The inverse Jacobian is initialised to the identity
/// (i.e. the first step is a damped steepest-descent-like move).
///
/// # Errors
///
/// * [`NumericError::EmptyInput`] if `x0` is empty.
/// * [`NumericError::InvalidParameter`] if `tol` is not positive finite, `x0`
///   is non-finite, or `f` returns a vector of the wrong length.
/// * [`NumericError::NotConverged`] if `‖F‖` stays above `tol` for `max_iter`
///   iterations.
pub fn broyden<F>(f: F, x0: &[f64], max_iter: usize, tol: f64) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    if !(tol > 0.0 && tol.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "tol must be positive finite, got {tol}"
        )));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(NumericError::InvalidParameter(
            "x0 has non-finite entries".into(),
        ));
    }

    let mut x = x0.to_vec();
    let mut fx = eval(&f, &x, n)?;
    if norm2(&fx) <= tol {
        return Ok(x);
    }

    // Inverse-Jacobian approximation, row-major n×n, initialised to identity.
    let mut binv = vec![0.0_f64; n * n];
    for i in 0..n {
        binv[i * n + i] = 1.0;
    }

    for _ in 0..max_iter {
        // Newton-like direction: p = −B · F.
        let bf = matvec(&binv, &fx, n);
        let p: Vec<f64> = bf.iter().map(|v| -v).collect();

        // Backtracking line search to guarantee residual decrease.
        let r_old = norm2(&fx);
        let mut lambda = 1.0_f64;
        let mut x_new = x.clone();
        let mut f_new = fx.clone();
        let mut accepted = false;
        for _ in 0..25 {
            let trial: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + lambda * pi).collect();
            if trial.iter().all(|v| v.is_finite()) {
                let ftrial = eval(&f, &trial, n)?;
                if norm2(&ftrial) < r_old {
                    x_new = trial;
                    f_new = ftrial;
                    accepted = true;
                    break;
                }
            }
            lambda *= 0.5;
        }
        if !accepted {
            // No decrease found: take the full step to keep moving.
            x_new = x.iter().zip(&p).map(|(xi, pi)| xi + pi).collect();
            if x_new.iter().any(|v| !v.is_finite()) {
                return Err(NumericError::NumericalInstability(
                    "iterate became non-finite".into(),
                ));
            }
            f_new = eval(&f, &x_new, n)?;
            lambda = 1.0;
        }

        // s = x_new − x  ;  y = f_new − f.
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = f_new.iter().zip(&fx).map(|(a, b)| a - b).collect();

        x = x_new;
        fx = f_new;
        let residual = norm2(&fx);
        if residual <= tol {
            return Ok(x);
        }

        // Sherman-Morrison rank-1 update of B ≈ J⁻¹:
        //   B ← B + ((s − B y) / (sᵀ B y)) (sᵀ B)
        let by = matvec(&binv, &y, n); // B y
        let s_minus_by: Vec<f64> = s.iter().zip(&by).map(|(a, b)| a - b).collect();
        let stb = vec_mat(&s, &binv, n); // sᵀ B  (row vector, length n)
        let denom = dot(&stb, &y); // sᵀ B y
        if denom.abs() > 1.0e-300 {
            let inv_denom = 1.0 / denom;
            for i in 0..n {
                let factor = s_minus_by[i] * inv_denom;
                if factor != 0.0 {
                    for j in 0..n {
                        binv[i * n + j] += factor * stb[j];
                    }
                }
            }
        }
        let _ = lambda; // step length used implicitly via s
    }

    if norm2(&fx) <= tol {
        return Ok(x);
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: norm2(&fx),
    })
}

/// `y = A · v` for row-major `n×n` `a`.
fn matvec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (i, oi) in out.iter_mut().enumerate() {
        let row = &a[i * n..i * n + n];
        *oi = dot(row, v);
    }
    out
}

/// `r = vᵀ · A` (row vector times matrix) for row-major `n×n` `a`.
fn vec_mat(v: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (i, &vi) in v.iter().enumerate() {
        if vi != 0.0 {
            let row = &a[i * n..i * n + n];
            for (oj, &aij) in out.iter_mut().zip(row) {
                *oj += vi * aij;
            }
        }
    }
    out
}

fn eval<F>(f: &F, x: &[f64], n: usize) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let y = f(x);
    if y.len() != n {
        return Err(NumericError::InvalidParameter(format!(
            "f returned length {}, expected {n}",
            y.len()
        )));
    }
    Ok(y)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_root() {
        let f = |x: &[f64]| vec![x[0] * x[0] - 2.0];
        let r = broyden(f, &[1.5], 100, 1e-10).expect("ok");
        assert!((r[0] - 2.0_f64.sqrt()).abs() < 1e-8, "{}", r[0]);
    }

    #[test]
    fn system_root() {
        // x + y = 3 ; x·y = 2 → (1, 2) or (2, 1).
        let f = |v: &[f64]| vec![v[0] + v[1] - 3.0, v[0] * v[1] - 2.0];
        let r = broyden(f, &[0.5, 2.5], 200, 1e-10).expect("ok");
        let res = f(&r);
        assert!(res[0].hypot(res[1]) < 1e-8, "residual");
    }

    #[test]
    fn converges() {
        let f = |v: &[f64]| vec![3.0 * v[0] - v[1].cos() - 1.5, v[1] - v[0].sin() - 0.5];
        let r = broyden(f, &[0.5, 0.5], 200, 1e-11).expect("ok");
        let res = f(&r);
        assert!(res[0].hypot(res[1]) < 1e-9);
    }

    #[test]
    fn max_iter_bound() {
        let f = |x: &[f64]| vec![x[0].exp() - 100.0];
        let res = broyden(f, &[0.0], 1, 1e-14);
        match res {
            Err(NumericError::NotConverged { iter, .. }) => assert_eq!(iter, 1),
            other => panic!("expected NotConverged, got {other:?}"),
        }
    }

    #[test]
    fn at_root() {
        let f = |v: &[f64]| vec![v[0] - 5.0, v[1] + 3.0];
        let r = broyden(f, &[5.0, -3.0], 50, 1e-12).expect("ok");
        assert!((r[0] - 5.0).abs() < 1e-12);
        assert!((r[1] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn output_len() {
        let f = |v: &[f64]| v.iter().map(|x| x - 2.0).collect::<Vec<_>>();
        let r = broyden(f, &[0.0; 6], 100, 1e-10).expect("ok");
        assert_eq!(r.len(), 6);
    }

    #[test]
    fn linear_system_exact() {
        // For a linear system Broyden converges to the exact solution.
        // A = [[2,1],[1,3]], b = [3,5] → x = [4/5, 7/5].
        let f = |v: &[f64]| vec![2.0 * v[0] + v[1] - 3.0, v[0] + 3.0 * v[1] - 5.0];
        let r = broyden(f, &[0.0, 0.0], 200, 1e-12).expect("ok");
        assert!((r[0] - 0.8).abs() < 1e-8, "x={}", r[0]);
        assert!((r[1] - 1.4).abs() < 1e-8, "y={}", r[1]);
    }

    #[test]
    fn finite() {
        let f = |v: &[f64]| vec![v[0] * v[0] + v[1] - 3.0, v[0] + v[1] * v[1] - 5.0];
        let r = broyden(f, &[1.0, 1.0], 200, 1e-9).expect("ok");
        for v in &r {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_bad_input() {
        let f = |v: &[f64]| v.to_vec();
        assert!(broyden(f, &[], 10, 1e-10).is_err());
        assert!(broyden(|v: &[f64]| v.to_vec(), &[1.0], 10, 0.0).is_err());
        assert!(broyden(|v: &[f64]| v.to_vec(), &[f64::NAN], 10, 1e-10).is_err());
    }
}
