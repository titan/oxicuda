//! Brent's method — robust hybrid of bisection, secant, and inverse quadratic interpolation.

use crate::error::{NumericError, NumericResult};

/// Brent's method on `[a, b]` with `f(a) f(b) < 0`.
pub fn brent<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    let mut fa = f(a)?;
    let mut fb = f(b)?;
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(NumericError::RootNotBracketed { a, b, fa, fb });
    }
    // ensure |f(a)| ≥ |f(b)| (b is the "better" estimate).
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for _ in 0..max_iter {
        if fb.abs() < tol || (b - a).abs() < tol {
            return Ok(b);
        }
        let s_iqi = if fa != fc && fb != fc {
            // inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // secant
            b - fb * (b - a) / (fb - fa)
        };

        // conditions for accepting s
        let between = {
            let lo = (3.0 * a + b) * 0.25;
            let hi = b;
            (s_iqi - lo) * (s_iqi - hi) < 0.0
        };
        let prev_step = if mflag { (b - c).abs() } else { (c - d).abs() };
        let s_use_bisect = !between
            || (mflag && (s_iqi - b).abs() >= 0.5 * (b - c).abs())
            || (!mflag && (s_iqi - b).abs() >= 0.5 * prev_step)
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol);
        let s = if s_use_bisect {
            mflag = true;
            0.5 * (a + b)
        } else {
            mflag = false;
            s_iqi
        };

        let fs = f(s)?;
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: fb.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn brent_sin_pi() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.sin()) };
        let r = brent(f, 3.0, 4.0, 1.0e-12, 100).expect("ok");
        assert!((r - PI).abs() < 1.0e-10);
    }

    #[test]
    fn brent_quadratic() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x - 5.0) };
        let r = brent(f, 0.0, 10.0, 1.0e-12, 100).expect("ok");
        assert!((r - 5.0_f64.sqrt()).abs() < 1.0e-10);
    }

    #[test]
    fn brent_not_bracketed_err() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x + 1.0) };
        let res = brent(f, -1.0, 1.0, 1.0e-10, 50);
        assert!(matches!(res, Err(NumericError::RootNotBracketed { .. })));
    }
}
