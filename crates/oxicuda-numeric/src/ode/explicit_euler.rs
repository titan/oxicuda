//! Explicit (forward) Euler method: `y_{k+1} = y_k + h f(t_k, y_k)`.

use crate::error::{NumericError, NumericResult};

/// Integrate `dy/dt = f(t, y)` with fixed step `h` from `t0` to `tf`.
///
/// `f` receives `(t, y)` and returns `dy/dt`. Returns the trajectory
/// `(times, ys)` where `ys` is row-major `(n_steps, dim)`.
pub fn explicit_euler<F>(
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
    for _ in 0..n_steps {
        let dy = f(t, &y)?;
        if dy.len() != y.len() {
            return Err(NumericError::DimensionMismatch {
                a: dy.len(),
                b: y.len(),
            });
        }
        for i in 0..y.len() {
            y[i] += h * dy[i];
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
    fn euler_exponential_decay() {
        // y' = -y, y(0) = 1 → y(1) = e^{-1}
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let (_t, ys) = explicit_euler(f, 0.0, 1.0, &[1.0], 0.001).expect("ok");
        let last = ys.last().expect("non-empty");
        // Euler O(h) — fairly imprecise but should be close
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-2);
    }

    #[test]
    fn euler_simple_const() {
        let f = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let (_t, ys) = explicit_euler(f, 0.0, 1.0, &[0.0], 0.01).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn euler_bad_step_err() {
        let f = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![1.0]) };
        let r = explicit_euler(f, 0.0, 1.0, &[0.0], -1.0);
        assert!(matches!(r, Err(NumericError::InvalidStepSize { .. })));
    }
}
