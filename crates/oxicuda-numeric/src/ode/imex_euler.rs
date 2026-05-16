//! IMEX Euler (1st-order Implicit-Explicit splitting).
//!
//! Decompose RHS `f(t, y) = f_stiff(t, y) + f_nonstiff(t, y)`. The stiff part is treated
//! implicitly (backward Euler) and the non-stiff part explicitly (forward Euler):
//!     `y_{n+1} = y_n + h f_nonstiff(t_n, y_n) + h f_stiff(t_{n+1}, y_{n+1})`.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

/// IMEX Euler integrator. The user supplies the stiff RHS, non-stiff RHS, and the Jacobian of
/// the stiff part with respect to `y`.
pub fn imex_euler<S, N, J>(
    f_stiff: S,
    f_nonstiff: N,
    j_stiff: J,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
    newton_tol: f64,
    newton_max: usize,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    S: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    N: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    J: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    let dim = y0.len();
    let n_steps = ((tf - t0) / h).ceil() as usize;
    let mut times = vec![t0];
    let mut ys: Vec<Vec<f64>> = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();
    for _ in 0..n_steps {
        let fn_v = f_nonstiff(t, &y)?;
        let mut y_pred = vec![0.0_f64; dim];
        for i in 0..dim {
            y_pred[i] = y[i] + h * fn_v[i];
        }
        // Newton on g(y_{n+1}) = y_{n+1} - y_pred - h f_stiff(t_{n+1}, y_{n+1}) = 0
        let mut y_new = y_pred.clone();
        for _ in 0..newton_max {
            let fs = f_stiff(t + h, &y_new)?;
            let mut g = vec![0.0_f64; dim];
            for i in 0..dim {
                g[i] = y_new[i] - y_pred[i] - h * fs[i];
            }
            let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < newton_tol {
                break;
            }
            let jf = j_stiff(t + h, &y_new)?;
            let mut m = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                for k in 0..dim {
                    m[i * dim + k] = -h * jf[i * dim + k];
                }
                m[i * dim + i] += 1.0;
            }
            let (lu, piv, _) = lu_decompose(&m, dim)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imex_pure_stiff() {
        // y' = -y treated as fully stiff
        let s = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let n = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![0.0]) };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
        let (_t, ys) = imex_euler(s, n, j, 0.0, 1.0, &[1.0], 0.01, 1.0e-10, 30).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-2);
    }

    #[test]
    fn imex_split() {
        // y' = -y - y², split: f_stiff = -y, f_nonstiff = -y²
        let s = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let n = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0] * y[0]]) };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
        let (_t, ys) = imex_euler(s, n, j, 0.0, 1.0, &[1.0], 0.001, 1.0e-10, 30).expect("ok");
        let last = ys.last().expect("non-empty");
        // analytical: dy/dt = -y - y² → 1/y - 1/(y+1) etc.; we just check sanity
        assert!(last[0] > 0.0 && last[0] < 1.0);
    }
}
