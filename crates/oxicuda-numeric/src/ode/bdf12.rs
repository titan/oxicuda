//! Backward Differentiation Formulae (BDF) — order 1 and 2 implicit methods.
//!
//! BDF1 is implicit Euler. BDF2 uses the previous two points.
//! Each step is solved via Newton iteration on a user-supplied Jacobian.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

/// Implicit Euler (BDF1) with Newton iteration per step.
///
/// `f(t, y)` is the RHS, `j(t, y)` returns the Jacobian `df/dy` as a row-major matrix.
pub fn bdf1<F, J>(
    f: F,
    j: J,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
    newton_tol: f64,
    newton_max: usize,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    J: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    if tf <= t0 {
        return Err(NumericError::InvalidParameter("tf must be > t0".into()));
    }
    let dim = y0.len();
    let n_steps = ((tf - t0) / h).ceil() as usize;
    let mut times = vec![t0];
    let mut ys: Vec<Vec<f64>> = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();
    for _ in 0..n_steps {
        // Solve y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0 via Newton.
        let mut y_new = y.clone();
        for _ in 0..newton_max {
            let fv = f(t + h, &y_new)?;
            let mut g = vec![0.0_f64; dim];
            for i in 0..dim {
                g[i] = y_new[i] - y[i] - h * fv[i];
            }
            let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < newton_tol {
                break;
            }
            // Jacobian of g = I - h J(f)
            let jf = j(t + h, &y_new)?;
            if jf.len() != dim * dim {
                return Err(NumericError::ShapeMismatch {
                    expected: vec![dim, dim],
                    got: vec![jf.len()],
                });
            }
            let mut mtx = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                for k in 0..dim {
                    mtx[i * dim + k] = -h * jf[i * dim + k];
                }
                mtx[i * dim + i] += 1.0;
            }
            let (lu, piv, _s) = lu_decompose(&mtx, dim)?;
            let delta = lu_solve(&lu, &piv, dim, &g)?;
            for i in 0..dim {
                y_new[i] -= delta[i];
            }
        }
        y = y_new;
        t += h;
        times.push(t);
        ys.push(y.clone());
    }
    Ok((times, ys))
}

/// BDF2 with bootstrap by BDF1 for first step.
pub fn bdf2<F, J>(
    f: F,
    j: J,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
    newton_tol: f64,
    newton_max: usize,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    J: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    let dim = y0.len();
    let n_steps = ((tf - t0) / h).ceil() as usize;
    if n_steps == 0 {
        return Ok((vec![t0], vec![y0.to_vec()]));
    }
    // bootstrap one step with BDF1
    let (mut times, mut ys) = bdf1(&f, &j, t0, t0 + h, y0, h, newton_tol, newton_max)?;
    if n_steps == 1 {
        return Ok((times, ys));
    }
    let mut t = times[1];
    let mut y_prev = ys[0].clone();
    let mut y = ys[1].clone();
    for _ in 1..n_steps {
        // (3 y_{n+1} - 4 y_n + y_{n-1}) / (2 h) = f(t_{n+1}, y_{n+1})
        let mut y_new = y.clone();
        for _ in 0..newton_max {
            let fv = f(t + h, &y_new)?;
            let mut g = vec![0.0_f64; dim];
            for i in 0..dim {
                g[i] = 3.0 * y_new[i] - 4.0 * y[i] + y_prev[i] - 2.0 * h * fv[i];
            }
            let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < newton_tol {
                break;
            }
            let jf = j(t + h, &y_new)?;
            let mut mtx = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                for k in 0..dim {
                    mtx[i * dim + k] = -2.0 * h * jf[i * dim + k];
                }
                mtx[i * dim + i] += 3.0;
            }
            let (lu, piv, _) = lu_decompose(&mtx, dim)?;
            let delta = lu_solve(&lu, &piv, dim, &g)?;
            for i in 0..dim {
                y_new[i] -= delta[i];
            }
        }
        y_prev = y.clone();
        y = y_new;
        t += h;
        times.push(t);
        ys.push(y.clone());
    }
    Ok((times, ys))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf1_exponential_decay() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
        let (_t, ys) = bdf1(f, j, 0.0, 1.0, &[1.0], 0.01, 1.0e-10, 30).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-2);
    }

    #[test]
    fn bdf2_exponential_decay_better() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
        let (_t, ys) = bdf2(f, j, 0.0, 1.0, &[1.0], 0.01, 1.0e-10, 30).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-3);
    }
}
