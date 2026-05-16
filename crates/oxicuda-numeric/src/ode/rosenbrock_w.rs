//! Rosenbrock-W method (linearly implicit, 2-stage, order 2).
//!
//! `(I - γ h J) k1 = f(t, y)`
//! `(I - γ h J) k2 = f(t + h, y + h k1) - 2 γ h J k1`  (simplified form)
//! `y_{n+1} = y + (h/2)(k1 + k2)`
//! with `γ = 1 / (2 + √2)` for order-2 L-stability.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

const GAMMA: f64 = 0.292_893_218_813_452_4; // 1 / (2 + sqrt(2))

/// Rosenbrock-W step-by-step integrator.
pub fn rosenbrock_w<F, J>(
    f: F,
    j: J,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
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
    let mut times = vec![t0];
    let mut ys: Vec<Vec<f64>> = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();
    for _ in 0..n_steps {
        let jv = j(t, &y)?;
        if jv.len() != dim * dim {
            return Err(NumericError::ShapeMismatch {
                expected: vec![dim, dim],
                got: vec![jv.len()],
            });
        }
        let mut m = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                m[i * dim + k] = -GAMMA * h * jv[i * dim + k];
            }
            m[i * dim + i] += 1.0;
        }
        let (lu, piv, _) = lu_decompose(&m, dim)?;
        let fv = f(t, &y)?;
        let k1 = lu_solve(&lu, &piv, dim, &fv)?;
        // y + h k1
        let mut ytmp = vec![0.0_f64; dim];
        for i in 0..dim {
            ytmp[i] = y[i] + h * k1[i];
        }
        let fv2 = f(t + h, &ytmp)?;
        // RHS for k2: f(t+h, y + h k1) - 2 γ h J k1
        let mut rhs = vec![0.0_f64; dim];
        for i in 0..dim {
            let mut jk1 = 0.0_f64;
            for kk in 0..dim {
                jk1 += jv[i * dim + kk] * k1[kk];
            }
            rhs[i] = fv2[i] - 2.0 * GAMMA * h * jk1;
        }
        let k2 = lu_solve(&lu, &piv, dim, &rhs)?;
        for i in 0..dim {
            y[i] += 0.5 * h * (k1[i] + k2[i]);
        }
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
    fn rw_exponential_decay() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-1.0]) };
        let (_t, ys) = rosenbrock_w(f, j, 0.0, 1.0, &[1.0], 0.01).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-3);
    }

    #[test]
    fn rw_stiff_robertson_small() {
        // small Robertson problem (toy)
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![
                -0.04 * y[0] + 1.0e4 * y[1],
                0.04 * y[0] - 1.0e4 * y[1],
            ])
        };
        let j = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![-0.04, 1.0e4, 0.04, -1.0e4])
        };
        let (_t, ys) = rosenbrock_w(f, j, 0.0, 0.1, &[1.0, 0.0], 0.001).expect("ok");
        // mass-conservation: y[0] + y[1] should remain ≈ 1
        for yvec in ys.iter() {
            assert!((yvec[0] + yvec[1] - 1.0).abs() < 1.0e-6);
        }
    }
}
