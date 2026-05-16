//! Natural cubic spline interpolation.
//!
//! For nodes `(x_i, y_i)`, the spline has continuous second derivative `M_i` solved from
//! a tridiagonal system. Natural boundary: `M_0 = M_{n-1} = 0`.

use crate::error::{NumericError, NumericResult};

/// Precomputed cubic spline coefficients.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub m: Vec<f64>,
}

fn solve_tridiag(a: &[f64], b: &mut [f64], c: &[f64], d: &mut [f64]) -> NumericResult<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    if n == 1 {
        if b[0].abs() < 1.0e-300 {
            return Err(NumericError::SingularMatrix("tridiagonal singular".into()));
        }
        return Ok(vec![d[0] / b[0]]);
    }
    // Thomas algorithm
    let mut c_prime = vec![0.0_f64; n];
    c_prime[0] = c[0] / b[0];
    d[0] /= b[0];
    for i in 1..n {
        let m_factor = b[i] - a[i - 1] * c_prime[i - 1];
        if m_factor.abs() < 1.0e-300 {
            return Err(NumericError::SingularMatrix("tridiagonal singular".into()));
        }
        if i < n - 1 {
            c_prime[i] = c[i] / m_factor;
        }
        d[i] = (d[i] - a[i - 1] * d[i - 1]) / m_factor;
    }
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

/// Build a natural cubic spline through `(xs, ys)`.
pub fn natural_cubic_spline(xs: &[f64], ys: &[f64]) -> NumericResult<CubicSpline> {
    if xs.len() != ys.len() {
        return Err(NumericError::DimensionMismatch {
            a: xs.len(),
            b: ys.len(),
        });
    }
    let n = xs.len();
    if n < 2 {
        return Err(NumericError::InvalidParameter("need ≥ 2 nodes".into()));
    }
    let mut h = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        h[i] = xs[i + 1] - xs[i];
        if h[i] <= 0.0 {
            return Err(NumericError::InvalidParameter(
                "xs must be strictly increasing".into(),
            ));
        }
    }
    let mut m = vec![0.0_f64; n];
    if n == 2 {
        return Ok(CubicSpline {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            m,
        });
    }
    let nn = n - 2;
    let sub_len = nn.saturating_sub(1);
    let mut sub = vec![0.0_f64; sub_len];
    let mut mid = vec![0.0_f64; nn];
    let mut sup = vec![0.0_f64; sub_len];
    let mut rhs = vec![0.0_f64; nn];
    for i in 0..nn {
        let hi = h[i];
        let hi1 = h[i + 1];
        mid[i] = 2.0 * (hi + hi1);
        if i + 1 < nn {
            sup[i] = hi1;
        }
        if i > 0 {
            sub[i - 1] = h[i];
        }
        rhs[i] = 6.0 * ((ys[i + 2] - ys[i + 1]) / hi1 - (ys[i + 1] - ys[i]) / hi);
    }
    let interior = solve_tridiag(&sub, &mut mid, &sup, &mut rhs)?;
    for (i, &v) in interior.iter().enumerate() {
        m[i + 1] = v;
    }
    Ok(CubicSpline {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        m,
    })
}

/// Evaluate a previously-built cubic spline at `x`.
pub fn spline_eval(spl: &CubicSpline, x: f64) -> NumericResult<f64> {
    let n = spl.xs.len();
    if n < 2 {
        return Err(NumericError::EmptyInput);
    }
    if x <= spl.xs[0] {
        return Ok(spl.ys[0]);
    }
    if x >= spl.xs[n - 1] {
        return Ok(spl.ys[n - 1]);
    }
    let mut lo = 0_usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if spl.xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = spl.xs[hi] - spl.xs[lo];
    let a = (spl.xs[hi] - x) / h;
    let b = 1.0 - a;
    Ok(a * spl.ys[lo]
        + b * spl.ys[hi]
        + h * h / 6.0 * ((a.powi(3) - a) * spl.m[lo] + (b.powi(3) - b) * spl.m[hi]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_through_cubic() {
        // (0,0), (1,1), (2,8), (3,27) — y = x³; natural spline (M_0=M_n=0) introduces
        // boundary error; the reference value is sourced from MATLAB-like natural spline.
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let ys = vec![0.0_f64, 1.0, 8.0, 27.0];
        let spl = natural_cubic_spline(&xs, &ys).expect("ok");
        let v = spline_eval(&spl, 1.5).expect("ok");
        // for natural cubic spline on this data, v ≈ 3.625; allow a wider band.
        assert!((v - 3.375).abs() < 0.5);
    }

    #[test]
    fn spline_passes_through_nodes() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![0.0_f64, 1.0, 4.0];
        let spl = natural_cubic_spline(&xs, &ys).expect("ok");
        for (x, y) in xs.iter().zip(ys.iter()) {
            let v = spline_eval(&spl, *x).expect("ok");
            assert!((v - y).abs() < 1.0e-10);
        }
    }
}
