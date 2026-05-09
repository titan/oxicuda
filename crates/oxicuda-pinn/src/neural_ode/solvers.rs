//! ODE integrators: Euler, Heun (RK2), RK4, Dormand-Prince RK4(5).

use crate::error::{PinnError, PinnResult};

/// ODE right-hand side function signature: `f(t, y, dydt)`.
pub type OdeRhsFn<'a> = &'a dyn Fn(f32, &[f32], &mut [f32]);

// ─── Single-step methods ──────────────────────────────────────────────────────

/// Explicit Euler step: `y_new = y + h * f(t, y)`.
pub fn euler_step(rhs: OdeRhsFn, t: f32, y: &[f32], h: f32) -> Vec<f32> {
    let dim = y.len();
    let mut k = vec![0.0_f32; dim];
    rhs(t, y, &mut k);
    y.iter()
        .zip(k.iter())
        .map(|(&yi, &ki)| yi + h * ki)
        .collect()
}

/// Heun (RK2) step: `k1 = f(t, y)`, `k2 = f(t+h, y+h*k1)`,
/// `y_new = y + h/2*(k1 + k2)`.
pub fn heun_step(rhs: OdeRhsFn, t: f32, y: &[f32], h: f32) -> Vec<f32> {
    let dim = y.len();
    let mut k1 = vec![0.0_f32; dim];
    let mut k2 = vec![0.0_f32; dim];
    rhs(t, y, &mut k1);
    let y_euler: Vec<f32> = y
        .iter()
        .zip(k1.iter())
        .map(|(&yi, &ki)| yi + h * ki)
        .collect();
    rhs(t + h, &y_euler, &mut k2);
    y.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .map(|((&yi, &k1i), &k2i)| yi + h * 0.5 * (k1i + k2i))
        .collect()
}

/// Classic RK4 step.
pub fn rk4_step(rhs: OdeRhsFn, t: f32, y: &[f32], h: f32) -> Vec<f32> {
    let dim = y.len();
    let mut k1 = vec![0.0_f32; dim];
    let mut k2 = vec![0.0_f32; dim];
    let mut k3 = vec![0.0_f32; dim];
    let mut k4 = vec![0.0_f32; dim];

    rhs(t, y, &mut k1);

    let y2: Vec<f32> = y
        .iter()
        .zip(k1.iter())
        .map(|(&yi, &ki)| yi + 0.5 * h * ki)
        .collect();
    rhs(t + 0.5 * h, &y2, &mut k2);

    let y3: Vec<f32> = y
        .iter()
        .zip(k2.iter())
        .map(|(&yi, &ki)| yi + 0.5 * h * ki)
        .collect();
    rhs(t + 0.5 * h, &y3, &mut k3);

    let y4: Vec<f32> = y
        .iter()
        .zip(k3.iter())
        .map(|(&yi, &ki)| yi + h * ki)
        .collect();
    rhs(t + h, &y4, &mut k4);

    y.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .zip(k3.iter())
        .zip(k4.iter())
        .map(|((((yi, k1i), k2i), k3i), k4i)| yi + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i))
        .collect()
}

// ─── Dormand-Prince RK4(5) ────────────────────────────────────────────────────

/// Butcher tableau constants for Dormand-Prince (DOPRI5).
mod dopri_coeff {
    // Node values
    pub const C2: f32 = 1.0 / 5.0;
    pub const C3: f32 = 3.0 / 10.0;
    pub const C4: f32 = 4.0 / 5.0;
    pub const C5: f32 = 8.0 / 9.0;

    // Runge-Kutta matrix a_ij
    pub const A21: f32 = 1.0 / 5.0;

    pub const A31: f32 = 3.0 / 40.0;
    pub const A32: f32 = 9.0 / 40.0;

    pub const A41: f32 = 44.0 / 45.0;
    pub const A42: f32 = -56.0 / 15.0;
    pub const A43: f32 = 32.0 / 9.0;

    pub const A51: f32 = 19372.0 / 6561.0;
    pub const A52: f32 = -25360.0 / 2187.0;
    pub const A53: f32 = 64448.0 / 6561.0;
    pub const A54: f32 = -212.0 / 729.0;

    pub const A61: f32 = 9017.0 / 3168.0;
    pub const A62: f32 = -355.0 / 33.0;
    pub const A63: f32 = 46732.0 / 5247.0;
    pub const A64: f32 = 49.0 / 176.0;
    pub const A65: f32 = -5103.0 / 18656.0;

    // 5th-order weights (b)
    pub const B1: f32 = 35.0 / 384.0;
    // B2 = 0
    pub const B3: f32 = 500.0 / 1113.0;
    pub const B4: f32 = 125.0 / 192.0;
    pub const B5: f32 = -2187.0 / 6784.0;
    pub const B6: f32 = 11.0 / 84.0;

    // Error coefficients (difference between 5th and 4th order)
    pub const E1: f32 = 71.0 / 57600.0;
    // E2 = 0
    pub const E3: f32 = -71.0 / 16695.0;
    pub const E4: f32 = 71.0 / 1920.0;
    pub const E5: f32 = -17253.0 / 339200.0;
    pub const E6: f32 = 22.0 / 525.0;
    pub const E7: f32 = -1.0 / 40.0;
}

/// Dormand-Prince RK4(5) adaptive step.
///
/// Returns `(y_new, error_estimate)` where the error is the per-component
/// difference between the 5th and 4th order solutions.
pub fn dopri45_step(rhs: OdeRhsFn, t: f32, y: &[f32], h: f32) -> (Vec<f32>, Vec<f32>) {
    use dopri_coeff::*;
    let dim = y.len();
    let mut k1 = vec![0.0_f32; dim];
    let mut k2 = vec![0.0_f32; dim];
    let mut k3 = vec![0.0_f32; dim];
    let mut k4 = vec![0.0_f32; dim];
    let mut k5 = vec![0.0_f32; dim];
    let mut k6 = vec![0.0_f32; dim];
    let mut k7 = vec![0.0_f32; dim];

    rhs(t, y, &mut k1);

    let y2: Vec<f32> = (0..dim).map(|i| y[i] + h * A21 * k1[i]).collect();
    rhs(t + C2 * h, &y2, &mut k2);

    let y3: Vec<f32> = (0..dim)
        .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
        .collect();
    rhs(t + C3 * h, &y3, &mut k3);

    let y4: Vec<f32> = (0..dim)
        .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
        .collect();
    rhs(t + C4 * h, &y4, &mut k4);

    let y5: Vec<f32> = (0..dim)
        .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
        .collect();
    rhs(t + C5 * h, &y5, &mut k5);

    let y6: Vec<f32> = (0..dim)
        .map(|i| y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
        .collect();
    rhs(t + h, &y6, &mut k6);

    // 5th-order solution (also used for next step via FSAL property)
    let y_new: Vec<f32> = (0..dim)
        .map(|i| y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]))
        .collect();

    rhs(t + h, &y_new, &mut k7);

    // Error estimate (difference of 5th and 4th order)
    let error: Vec<f32> = (0..dim)
        .map(|i| h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]))
        .collect();

    (y_new, error)
}

// ─── Fixed-step integration ───────────────────────────────────────────────────

/// Integrate an ODE with fixed step size using RK4.
///
/// Returns `(times, states)` where `states[k]` is the solution at `times[k]`.
pub fn integrate_fixed(
    rhs: OdeRhsFn,
    t0: f32,
    t1: f32,
    y0: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<Vec<f32>>)> {
    if h <= 0.0 || !h.is_finite() {
        return Err(PinnError::InvalidStepSize { h });
    }
    if t1 <= t0 {
        return Err(PinnError::InvalidTimeInterval { t0, t1 });
    }

    let mut times = vec![t0];
    let mut states = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();

    while t < t1 {
        let h_eff = h.min(t1 - t);
        let y_new = rk4_step(rhs, t, &y, h_eff);
        if y_new.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::SolverDivergence {
                reason: "NaN/Inf in RK4 state",
            });
        }
        t += h_eff;
        times.push(t);
        states.push(y_new.clone());
        y = y_new;
    }
    Ok((times, states))
}

// ─── Adaptive-step integration (Dormand-Prince) ───────────────────────────────

/// Integrate using Dormand-Prince RK4(5) with adaptive step size control.
///
/// Returns `(times, states)`.
pub fn integrate_adaptive(
    rhs: OdeRhsFn,
    t0: f32,
    t1: f32,
    y0: &[f32],
    atol: f32,
    rtol: f32,
    h_init: f32,
) -> PinnResult<(Vec<f32>, Vec<Vec<f32>>)> {
    if h_init <= 0.0 || !h_init.is_finite() {
        return Err(PinnError::InvalidStepSize { h: h_init });
    }
    if t1 <= t0 {
        return Err(PinnError::InvalidTimeInterval { t0, t1 });
    }

    let mut times = vec![t0];
    let mut states = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h_init;
    let max_steps = 100_000_usize;

    for _ in 0..max_steps {
        if t >= t1 {
            break;
        }
        let h_eff = h.min(t1 - t);
        let (y_new, err) = dopri45_step(rhs, t, &y, h_eff);

        if y_new.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::SolverDivergence {
                reason: "NaN/Inf in DOPRI45 state",
            });
        }

        // Compute error norm: max |e_i / (atol + rtol * max(|y_i|, |y_new_i|))|
        let err_norm = err
            .iter()
            .zip(y.iter())
            .zip(y_new.iter())
            .map(|((&ei, &yi), &yni)| {
                let scale = atol + rtol * yi.abs().max(yni.abs());
                (ei / scale).abs()
            })
            .fold(0.0_f32, f32::max);

        if err_norm <= 1.0 || h_eff <= 1e-10 {
            // Accept step
            t += h_eff;
            times.push(t);
            states.push(y_new.clone());
            y = y_new;
        }

        // Step size update
        let factor = if err_norm > 1e-10 {
            0.9 * err_norm.powf(-0.2)
        } else {
            5.0
        };
        h = h_eff * factor.clamp(0.2, 5.0);
    }

    Ok((times, states))
}

#[cfg(test)]
mod tests {
    use super::*;

    // dy/dt = -y, y(0) = 1 → y(t) = exp(-t)
    fn exp_decay(t: f32, y: &[f32], dydt: &mut [f32]) {
        let _ = t;
        dydt[0] = -y[0];
    }

    // Spring oscillator: dy/dt = v, dv/dt = -y
    fn spring(t: f32, y: &[f32], dydt: &mut [f32]) {
        let _ = t;
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }

    #[test]
    fn euler_exp_decay_rough() {
        let y_final = euler_step(&exp_decay, 0.0, &[1.0], 0.01);
        assert!((y_final[0] - 0.99).abs() < 0.01);
    }

    #[test]
    fn heun_exp_decay() {
        let mut y = vec![1.0_f32];
        let h = 0.01;
        for step in 0..100 {
            let t = step as f32 * h;
            y = heun_step(&exp_decay, t, &y, h);
        }
        let expected = (-1.0_f32).exp();
        assert!(
            (y[0] - expected).abs() < 1e-3,
            "Heun y(1)={} expected {}",
            y[0],
            expected
        );
    }

    #[test]
    fn rk4_exp_decay_accurate() {
        let mut y = vec![1.0_f32];
        let h = 0.01;
        for step in 0..100 {
            let t = step as f32 * h;
            y = rk4_step(&exp_decay, t, &y, h);
        }
        let expected = (-1.0_f32).exp();
        assert!(
            (y[0] - expected).abs() < 1e-5,
            "RK4 y(1)={} expected {}",
            y[0],
            expected
        );
    }

    #[test]
    fn integrate_fixed_exp_decay() {
        let (times, states) = integrate_fixed(&exp_decay, 0.0, 1.0, &[1.0], 0.01).unwrap();
        assert!(!times.is_empty());
        let y_final = states.last().unwrap()[0];
        let expected = (-1.0_f32).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "fixed y(1)={} expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn integrate_adaptive_exp_decay() {
        let (_, states) =
            integrate_adaptive(&exp_decay, 0.0, 1.0, &[1.0], 1e-6, 1e-4, 0.1).unwrap();
        let y_final = states.last().unwrap()[0];
        let expected = (-1.0_f32).exp();
        assert!(
            (y_final - expected).abs() < 1e-3,
            "adaptive y(1)={} expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn dopri45_step_single_finite() {
        let (y_new, err) = dopri45_step(&exp_decay, 0.0, &[1.0], 0.1);
        assert!(y_new[0].is_finite());
        assert!(err[0].is_finite());
    }

    #[test]
    fn spring_energy_conservation() {
        // Total energy E = y[0]^2 + y[1]^2 should be ~conserved for harmonic oscillator
        let y0 = vec![1.0_f32, 0.0_f32];
        let period = 2.0 * std::f32::consts::PI;
        let h = 0.001;
        let (_, states) = integrate_fixed(&spring, 0.0, 10.0 * period, &y0, h).unwrap();
        let e0 = 1.0_f32; // initial energy = 1^2 + 0^2
        let e_final = {
            let s = states.last().unwrap();
            s[0] * s[0] + s[1] * s[1]
        };
        assert!(
            (e_final - e0).abs() < 0.01 * e0,
            "Energy not conserved: {} vs {}",
            e_final,
            e0
        );
    }

    #[test]
    fn invalid_step_size_error() {
        let result = integrate_fixed(&exp_decay, 0.0, 1.0, &[1.0], -0.1);
        assert!(matches!(result, Err(PinnError::InvalidStepSize { .. })));
    }

    #[test]
    fn invalid_time_interval_error() {
        let result = integrate_fixed(&exp_decay, 1.0, 0.5, &[1.0], 0.1);
        assert!(matches!(result, Err(PinnError::InvalidTimeInterval { .. })));
    }

    #[test]
    fn adaptive_invalid_h_error() {
        let result = integrate_adaptive(&exp_decay, 0.0, 1.0, &[1.0], 1e-6, 1e-4, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn rk4_step_dim4_finite() {
        fn linear_4d(t: f32, y: &[f32], dy: &mut [f32]) {
            let _ = t;
            for (dyi, &yi) in dy.iter_mut().zip(y.iter()) {
                *dyi = -yi;
            }
        }
        let y0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let y1 = rk4_step(&linear_4d, 0.0, &y0, 0.1);
        assert!(y1.iter().all(|v| v.is_finite()));
    }
}
