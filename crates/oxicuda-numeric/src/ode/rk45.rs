//! Dormand-Prince RK45 — adaptive step ODE integrator.
//!
//! Implements the classic Dormand-Prince 1980 embedded Runge-Kutta pair of orders 4 and 5.
//! The 5th-order solution is used for propagation (local extrapolation); the 4th-order
//! solution provides the error estimate.
//!
//! # Butcher Tableau (Dormand-Prince)
//!
//! ```text
//!  0   |
//!  1/5 | 1/5
//!  3/10| 3/40      9/40
//!  4/5 | 44/45    -56/15     32/9
//!  8/9 | 19372/6561 -25360/2187 64448/6561 -212/729
//!  1   | 9017/3168  -355/33   46732/5247  49/176  -5103/18656
//!  1   | 35/384     0        500/1113   125/192  -2187/6784   11/84
//! -----+-----------------------------------------------------------
//! b5   | 35/384     0        500/1113   125/192  -2187/6784   11/84    0
//! b4   | 5179/57600 0        7571/16695 393/640  -92097/339200 187/2100 1/40
//! ```
//!
//! Step-size control uses an I-controller:
//! `h_new = h * safety * (tol / err)^(1/5)`, clipped to `[min_shrink, max_grow]`.

use crate::error::{NumericError, NumericResult};

// ---------------------------------------------------------------------------
// Butcher tableau constants
// ---------------------------------------------------------------------------

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
// C6 = C7 = 1.0 (endpoint evaluations)

const A21: f64 = 1.0 / 5.0;

const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;

const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;

const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;

const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order weights (b5 = row 7 of the tableau)
const B1: f64 = 35.0 / 384.0;
// B2 = 0
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;
// B7 = 0

// Error coefficients: E = b5 - b4
const E1: f64 = 71.0 / 57600.0;
// E2 = 0
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the RK45 adaptive integrator.
#[derive(Debug, Clone)]
pub struct Rk45Config {
    /// Absolute tolerance for step-size control.
    pub atol: f64,
    /// Relative tolerance for step-size control.
    pub rtol: f64,
    /// Initial step size.
    pub h_init: f64,
    /// Minimum allowed step size (error if step shrinks below this).
    pub h_min: f64,
    /// Maximum allowed step size.
    pub h_max: f64,
    /// Maximum number of accepted steps.
    pub max_steps: usize,
}

impl Default for Rk45Config {
    fn default() -> Self {
        Self {
            atol: 1.0e-6,
            rtol: 1.0e-3,
            h_init: 0.1,
            h_min: 1.0e-12,
            h_max: 1.0,
            max_steps: 100_000,
        }
    }
}

/// Output of a successful RK45 integration.
#[derive(Debug, Clone)]
pub struct Rk45Result {
    /// Accepted time points (length = `n_steps + 1`; includes `t0`).
    pub t: Vec<f64>,
    /// State at each accepted time point: `y[k]` corresponds to `t[k]`.
    pub y: Vec<Vec<f64>>,
    /// Number of accepted steps.
    pub n_steps: usize,
    /// Total number of right-hand-side function evaluations.
    pub n_func_evals: usize,
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/// Integrate `dy/dt = f(t, y)` from `t_span.0` to `t_span.1` using Dormand-Prince RK45.
///
/// # Arguments
/// - `f`: the right-hand side `f(t, y) → dy/dt`.
/// - `t_span`: `(t0, tf)` with `tf > t0`.
/// - `y0`: initial condition.
/// - `cfg`: integration parameters.
///
/// # Errors
/// - [`NumericError::InvalidParameter`] if `t_span` is invalid or `y0` is empty.
/// - [`NumericError::InvalidStepSize`] if `h_init` is non-positive.
/// - [`NumericError::NumericalInstability`] if step shrinks below `h_min`.
/// - [`NumericError::NotConverged`] if `max_steps` is reached before `tf`.
pub fn rk45_integrate<F>(
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    cfg: &Rk45Config,
) -> NumericResult<Rk45Result>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = t_span;
    if !t0.is_finite() || !tf.is_finite() || tf <= t0 {
        return Err(NumericError::InvalidParameter(format!(
            "t_span must satisfy t0 < tf (finite); got ({t0}, {tf})"
        )));
    }
    if y0.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if cfg.h_init <= 0.0 || !cfg.h_init.is_finite() {
        return Err(NumericError::InvalidStepSize { step: cfg.h_init });
    }

    let dim = y0.len();
    let mut t_vec = vec![t0];
    let mut y_vec: Vec<Vec<f64>> = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = cfg.h_init.min(cfg.h_max).min(tf - t0);
    let mut n_func_evals: usize = 0;

    // Reusable stage buffers
    let mut ytmp = vec![0.0_f64; dim];
    let mut yerr = vec![0.0_f64; dim];
    let mut y5 = vec![0.0_f64; dim];

    let safety = 0.9_f64;
    let max_grow = 5.0_f64;
    let min_shrink = 0.1_f64;

    let mut n_steps: usize = 0;

    loop {
        if (tf - t).abs() < 1.0e-14 * (1.0 + t.abs()) {
            break;
        }
        // Clip step to not overshoot tf
        if t + h > tf {
            h = tf - t;
        }
        if h <= 0.0 {
            break;
        }

        // --- Stage evaluations ---
        let k1 = f(t, &y);
        n_func_evals += 1;

        for i in 0..dim {
            ytmp[i] = y[i] + h * A21 * k1[i];
        }
        let k2 = f(t + C2 * h, &ytmp);
        n_func_evals += 1;

        for i in 0..dim {
            ytmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        let k3 = f(t + C3 * h, &ytmp);
        n_func_evals += 1;

        for i in 0..dim {
            ytmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        let k4 = f(t + C4 * h, &ytmp);
        n_func_evals += 1;

        for i in 0..dim {
            ytmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        let k5 = f(t + C5 * h, &ytmp);
        n_func_evals += 1;

        for i in 0..dim {
            ytmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        let k6 = f(t + h, &ytmp);
        n_func_evals += 1;

        // 5th-order solution
        for i in 0..dim {
            y5[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }

        // 7th stage (FSAL — first same as last for next step, but here we use it for error)
        let k7 = f(t + h, &y5);
        n_func_evals += 1;

        // Error estimate: E = h * (b5 - b4) · k_i
        for i in 0..dim {
            yerr[i] =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        }

        // Compute mixed tolerance-normalised error norm
        let mut err_sq_sum = 0.0_f64;
        for i in 0..dim {
            let sc = cfg.atol + cfg.rtol * y[i].abs().max(y5[i].abs());
            err_sq_sum += (yerr[i] / sc).powi(2);
        }
        let err_norm = (err_sq_sum / dim as f64).sqrt();

        if err_norm <= 1.0 {
            // Accept the step
            t += h;
            y[..dim].copy_from_slice(&y5[..dim]);
            t_vec.push(t);
            y_vec.push(y.clone());
            n_steps += 1;

            if n_steps >= cfg.max_steps {
                return Err(NumericError::NotConverged {
                    iter: n_steps,
                    residual: tf - t,
                });
            }
        }

        // Adapt step size
        let factor = if err_norm == 0.0 {
            max_grow
        } else {
            (safety * (1.0 / err_norm).powf(0.2))
                .max(min_shrink)
                .min(max_grow)
        };
        h = (h * factor).min(cfg.h_max);

        if h < cfg.h_min {
            return Err(NumericError::NumericalInstability(format!(
                "RK45 step size {h:.3e} fell below h_min={:.3e}",
                cfg.h_min
            )));
        }
    }

    Ok(Rk45Result {
        t: t_vec,
        y: y_vec,
        n_steps,
        n_func_evals,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> Rk45Config {
        Rk45Config {
            atol: 1.0e-8,
            rtol: 1.0e-6,
            h_init: 0.1,
            h_min: 1.0e-12,
            h_max: 1.0,
            max_steps: 50_000,
        }
    }

    #[test]
    fn constant_ode_exact() {
        // dy/dt = 0, y(0) = 5  → y(1) = 5
        let f = |_t: f64, y: &[f64]| vec![0.0_f64 * y[0]];
        let res = rk45_integrate(f, (0.0, 1.0), &[5.0], &default_cfg()).expect("ok");
        let last = res.y.last().expect("non-empty");
        assert!(
            (last[0] - 5.0).abs() < 1.0e-10,
            "constant ODE: got {}",
            last[0]
        );
    }

    #[test]
    fn exponential_decay_accurate() {
        // dy/dt = -y, y(0)=1 → y(1)=e^{-1}
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let res = rk45_integrate(f, (0.0, 1.0), &[1.0], &default_cfg()).expect("ok");
        let last = res.y.last().expect("non-empty");
        let expected = (-1.0_f64).exp();
        assert!(
            (last[0] - expected).abs() < 1.0e-6,
            "exp decay: got {}, expected {}",
            last[0],
            expected
        );
    }

    #[test]
    fn output_len() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let res = rk45_integrate(f, (0.0, 0.5), &[1.0], &default_cfg()).expect("ok");
        assert_eq!(
            res.t.len(),
            res.y.len(),
            "t and y vectors must have same length"
        );
        assert_eq!(res.t.len(), res.n_steps + 1, "length = n_steps + 1");
    }

    #[test]
    fn h_init_respected() {
        // With h_init=0.5 the first step should be at most 0.5 wide
        let cfg = Rk45Config {
            h_init: 0.5,
            ..default_cfg()
        };
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let res = rk45_integrate(f, (0.0, 2.0), &[1.0], &cfg).expect("ok");
        // Second time point ≤ t0 + 0.5 = 0.5
        assert!(res.t[1] <= 0.5 + 1.0e-10, "first step ≤ h_init");
    }

    #[test]
    fn t_monotone() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let res = rk45_integrate(f, (0.0, 3.0), &[1.0], &default_cfg()).expect("ok");
        for w in res.t.windows(2) {
            assert!(w[1] > w[0], "t must be strictly increasing");
        }
    }

    #[test]
    fn y_shape() {
        let f = |_t: f64, y: &[f64]| vec![-y[0], -y[1]];
        let res = rk45_integrate(f, (0.0, 1.0), &[1.0, 2.0], &default_cfg()).expect("ok");
        for yv in &res.y {
            assert_eq!(yv.len(), 2, "each y must have dim=2");
        }
    }

    #[test]
    fn atol_affects_steps() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let cfg_loose = Rk45Config {
            atol: 1.0e-3,
            rtol: 1.0e-2,
            ..default_cfg()
        };
        let cfg_tight = Rk45Config {
            atol: 1.0e-10,
            rtol: 1.0e-9,
            ..default_cfg()
        };
        let res_loose = rk45_integrate(f, (0.0, 2.0), &[1.0], &cfg_loose).expect("ok");
        let res_tight = rk45_integrate(f, (0.0, 2.0), &[1.0], &cfg_tight).expect("ok");
        assert!(
            res_tight.n_steps >= res_loose.n_steps,
            "tighter tolerance needs more steps"
        );
    }

    #[test]
    fn rtol_affects_accuracy() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let cfg = Rk45Config {
            atol: 1.0e-12,
            rtol: 1.0e-10,
            ..default_cfg()
        };
        let res = rk45_integrate(f, (0.0, 1.0), &[1.0], &cfg).expect("ok");
        let last = res.y.last().expect("non-empty");
        let expected = (-1.0_f64).exp();
        assert!(
            (last[0] - expected).abs() < 1.0e-8,
            "tight rtol: error = {}",
            (last[0] - expected).abs()
        );
    }

    #[test]
    fn max_steps_error() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let cfg = Rk45Config {
            max_steps: 2,
            ..default_cfg()
        };
        // With only 2 accepted steps, should fail to reach t=100
        let result = rk45_integrate(f, (0.0, 100.0), &[1.0], &cfg);
        assert!(
            matches!(result, Err(NumericError::NotConverged { .. })),
            "should hit max_steps"
        );
    }

    #[test]
    fn oscillator_bounded() {
        // y'' + y = 0 ↔ [y, y']' = [y', -y]; energy = y² + y'² = 1
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let cfg = Rk45Config {
            atol: 1.0e-9,
            rtol: 1.0e-8,
            h_max: 0.5,
            max_steps: 20_000,
            ..default_cfg()
        };
        let res = rk45_integrate(f, (0.0, 20.0), &[1.0, 0.0], &cfg).expect("ok");
        for yv in &res.y {
            let energy = yv[0] * yv[0] + yv[1] * yv[1];
            assert!(
                (energy - 1.0).abs() < 1.0e-4,
                "oscillator energy drift: |E-1|={}",
                (energy - 1.0).abs()
            );
        }
    }
}
