//! Classical 4th-order Runge-Kutta method.

use crate::error::PdeResult;

/// Classical RK4 step:
/// k1 = F(t, u)
/// k2 = F(t + dt/2, u + dt/2 * k1)
/// k3 = F(t + dt/2, u + dt/2 * k2)
/// k4 = F(t + dt, u + dt * k3)
/// u^{n+1} = u^n + dt/6 * (k1 + 2 k2 + 2 k3 + k4)
pub fn rk4_step<F>(u: &mut [f64], t: f64, dt: f64, f: F) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = u.len();
    let k1 = f(t, u);
    let mut u_tmp = vec![0.0; n];
    for i in 0..n {
        u_tmp[i] = u[i] + 0.5 * dt * k1[i];
    }
    let k2 = f(t + 0.5 * dt, &u_tmp);
    for i in 0..n {
        u_tmp[i] = u[i] + 0.5 * dt * k2[i];
    }
    let k3 = f(t + 0.5 * dt, &u_tmp);
    for i in 0..n {
        u_tmp[i] = u[i] + dt * k3[i];
    }
    let k4 = f(t + dt, &u_tmp);
    for i in 0..n {
        u[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_exponential_decay() {
        // du/dt = -u, u(0)=1 => u(1) = exp(-1)
        let mut u = vec![1.0];
        let dt = 0.01;
        let n = 100;
        for k in 0..n {
            let t = k as f64 * dt;
            rk4_step(&mut u, t, dt, |_, x| vec![-x[0]]).expect("ok");
        }
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1.0e-9,
            "u={} expected={}",
            u[0],
            expected
        );
    }

    #[test]
    fn rk4_harmonic_oscillator() {
        // du/dt = v, dv/dt = -u; with u(0)=1, v(0)=0 => u(t)=cos(t), v(t)=-sin(t)
        let mut x = vec![1.0, 0.0];
        let dt = 0.01;
        let nsteps = 1000;
        for k in 0..nsteps {
            let t = k as f64 * dt;
            rk4_step(&mut x, t, dt, |_, s| vec![s[1], -s[0]]).expect("ok");
        }
        let t_final = nsteps as f64 * dt;
        let u_expected = t_final.cos();
        let v_expected = -t_final.sin();
        assert!((x[0] - u_expected).abs() < 1.0e-7);
        assert!((x[1] - v_expected).abs() < 1.0e-7);
    }
}
