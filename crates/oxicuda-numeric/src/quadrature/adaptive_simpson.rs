//! Adaptive Simpson's rule with recursive subdivision.

use crate::error::{NumericError, NumericResult};

fn simpson(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    let h = b - a;
    h / 6.0 * (fa + 4.0 * fm + fb)
}

fn aux<F>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    s_whole: f64,
    eps: f64,
    depth: usize,
    max_depth: usize,
) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    let c = 0.5 * (a + b);
    let lm = 0.5 * (a + c);
    let rm = 0.5 * (c + b);
    let flm = f(lm)?;
    let frm = f(rm)?;
    let s_left = simpson(a, c, fa, fm, flm);
    let s_right = simpson(c, b, fm, fb, frm);
    let s2 = s_left + s_right;
    let err = (s2 - s_whole).abs();
    if err < 15.0 * eps || depth >= max_depth {
        return Ok(s2 + (s2 - s_whole) / 15.0);
    }
    let left = aux(
        f,
        a,
        c,
        fa,
        fm,
        flm,
        s_left,
        eps * 0.5,
        depth + 1,
        max_depth,
    )?;
    let right = aux(
        f,
        c,
        b,
        fm,
        fb,
        frm,
        s_right,
        eps * 0.5,
        depth + 1,
        max_depth,
    )?;
    Ok(left + right)
}

/// Adaptive Simpson integration of `f` over `[a, b]`.
pub fn adaptive_simpson<F>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter("non-finite limits".into()));
    }
    let fa = f(a)?;
    let fb = f(b)?;
    let m = 0.5 * (a + b);
    let fm = f(m)?;
    let s_whole = simpson(a, b, fa, fb, fm);
    aux(&f, a, b, fa, fb, fm, s_whole, tol, 0, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn simpson_arctan() {
        let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
        let r = adaptive_simpson(f, 0.0, 1.0, 1.0e-10, 30).expect("ok");
        assert!((r - PI / 4.0).abs() < 1.0e-8);
    }

    #[test]
    fn simpson_sqrt_singular() {
        // ∫_0^1 1/√x dx = 2 (mild endpoint singularity)
        let f = |x: f64| -> NumericResult<f64> { Ok(if x > 0.0 { 1.0 / x.sqrt() } else { 1.0e6 }) };
        let r = adaptive_simpson(f, 1.0e-12, 1.0, 1.0e-6, 50).expect("ok");
        assert!((r - 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn simpson_polynomial_exact() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(3)) };
        let r = adaptive_simpson(f, 0.0, 2.0, 1.0e-10, 30).expect("ok");
        assert!((r - 4.0).abs() < 1.0e-10);
    }
}
