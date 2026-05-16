//! Heun's method (improved Euler / 2-stage RK2).
//!
//! `k1 = f(t, y)`, `k2 = f(t + h, y + h k1)`, `y_{k+1} = y + h/2 (k1 + k2)`.

use crate::error::{NumericError, NumericResult};

/// Heun method with fixed step `h`.
pub fn heun<F>(
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
            ytmp[i] = y[i] + h * k1[i];
        }
        let k2 = f(t + h, &ytmp)?;
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
    fn heun_exponential_decay() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let (_t, ys) = heun(f, 0.0, 1.0, &[1.0], 0.01).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-3);
    }

    #[test]
    fn heun_oscillator() {
        // y'' + y = 0 with y(0)=1, y'(0)=0 → cos(t)
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![y[1], -y[0]]) };
        let (_t, ys) = heun(f, 0.0, 1.0, &[1.0, 0.0], 0.001).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - 1.0_f64.cos()).abs() < 1.0e-3);
    }
}
