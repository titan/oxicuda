//! Forward (explicit) Euler scheme: `u^{n+1} = u^n + dt * F(t^n, u^n)`.

use crate::error::PdeResult;

/// One forward-Euler step of a generic ODE system `du/dt = F(t, u)`.
pub fn forward_euler_step<F>(u: &mut [f64], t: f64, dt: f64, f: F) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let k = f(t, u);
    for i in 0..u.len() {
        u[i] += dt * k[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_euler_linear_decay() {
        // du/dt = -u, u(0) = 1 => u(t) = exp(-t)
        let mut u = vec![1.0];
        let dt = 0.001;
        let n = 1000;
        for k in 0..n {
            let t = k as f64 * dt;
            forward_euler_step(&mut u, t, dt, |_, x| vec![-x[0]]).expect("ok");
        }
        let expected = (-1.0_f64).exp();
        assert!((u[0] - expected).abs() < 1.0e-3);
    }

    #[test]
    fn forward_euler_quadratic_growth() {
        // du/dt = 2t, u(0) = 0 => u(t) = t^2
        let mut u = vec![0.0];
        let dt = 0.0001;
        let n = 10_000;
        for k in 0..n {
            let t = k as f64 * dt;
            forward_euler_step(&mut u, t, dt, |t, _| vec![2.0 * t]).expect("ok");
        }
        let expected = 1.0;
        assert!((u[0] - expected).abs() < 1.0e-3);
    }
}
