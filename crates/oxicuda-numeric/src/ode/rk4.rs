//! Classical 4th-order Runge-Kutta (RK4) method.
//!
//! `k1 = f(t, y)`,
//! `k2 = f(t + h/2, y + h k1 / 2)`,
//! `k3 = f(t + h/2, y + h k2 / 2)`,
//! `k4 = f(t + h, y + h k3)`,
//! `y_{k+1} = y + (h/6)(k1 + 2 k2 + 2 k3 + k4)`.

use crate::error::{NumericError, NumericResult};

/// RK4 with fixed step `h`.
pub fn rk4<F>(
    f: F,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    if tf <= t0 {
        return Err(NumericError::InvalidParameter("tf must be > t0".into()));
    }
    let n_steps = ((tf - t0) / h).ceil() as usize;
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);
    let mut t = t0;
    let mut y = y0.to_vec();
    times.push(t);
    ys.push(y.clone());
    let dim = y.len();
    let mut ytmp = vec![0.0_f64; dim];
    for _ in 0..n_steps {
        let k1 = f(t, &y)?;
        for i in 0..dim {
            ytmp[i] = y[i] + 0.5 * h * k1[i];
        }
        let k2 = f(t + 0.5 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] = y[i] + 0.5 * h * k2[i];
        }
        let k3 = f(t + 0.5 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] = y[i] + h * k3[i];
        }
        let k4 = f(t + h, &ytmp)?;
        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
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
    fn rk4_exponential_decay() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let (t_arr, ys) = rk4(f, 0.0, 1.0, &[1.0], 0.01).expect("ok");
        for (t, yvec) in t_arr.iter().zip(ys.iter()) {
            let exact = (-t).exp();
            assert!((yvec[0] - exact).abs() < 1.0e-4);
        }
    }

    #[test]
    fn rk4_oscillator() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![y[1], -y[0]]) };
        let (_t, ys) = rk4(f, 0.0, std::f64::consts::PI, &[1.0, 0.0], 0.001).expect("ok");
        let last = ys.last().expect("non-empty");
        // cos(π) = -1
        assert!((last[0] + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rk4_constant_derivative() {
        let f = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![2.0]) };
        let (_t, ys) = rk4(f, 0.0, 1.0, &[0.0], 0.1).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - 2.0).abs() < 1.0e-12);
    }
}
