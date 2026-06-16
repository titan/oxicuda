//! Adams-Bashforth-Moulton (ABM) predictor-corrector methods, orders 1–4.
//!
//! Uses the PECE (Predict–Evaluate–Correct–Evaluate) mode.
//! The first `order - 1` steps are bootstrapped with RK4.

use crate::error::{NumericError, NumericResult};

// ── Adams-Bashforth predictor coefficients (multiplied by 1/denom) ──────────
// Order 1:  y_pred = y + h * f_n
// Order 2:  y_pred = y + h * (3/2 f_n - 1/2 f_{n-1})
// Order 3:  y_pred = y + h * (23/12 f_n - 16/12 f_{n-1} + 5/12 f_{n-2})
// Order 4:  y_pred = y + h * (55/24 f_n - 59/24 f_{n-1} + 37/24 f_{n-2} - 9/24 f_{n-3})
//
// The history slice is ordered oldest-first: hist[0] = f_{n-p+1}, hist[last] = f_n.

fn ab_predict(y: &[f64], hist: &[Vec<f64>], h: f64, order: usize) -> Vec<f64> {
    let dim = y.len();
    let mut y_pred = vec![0.0_f64; dim];
    let n = hist.len(); // == order, newest is hist[n-1]
    match order {
        1 => {
            // y_pred = y + h * f_n
            let fn0 = &hist[n - 1];
            for i in 0..dim {
                y_pred[i] = y[i] + h * fn0[i];
            }
        }
        2 => {
            let fn0 = &hist[n - 1]; // f_n
            let fn1 = &hist[n - 2]; // f_{n-1}
            for i in 0..dim {
                y_pred[i] = y[i] + h * (1.5 * fn0[i] - 0.5 * fn1[i]);
            }
        }
        3 => {
            let fn0 = &hist[n - 1]; // f_n
            let fn1 = &hist[n - 2]; // f_{n-1}
            let fn2 = &hist[n - 3]; // f_{n-2}
            for i in 0..dim {
                y_pred[i] = y[i]
                    + h * ((23.0 / 12.0) * fn0[i] - (16.0 / 12.0) * fn1[i] + (5.0 / 12.0) * fn2[i]);
            }
        }
        4 => {
            let fn0 = &hist[n - 1]; // f_n
            let fn1 = &hist[n - 2]; // f_{n-1}
            let fn2 = &hist[n - 3]; // f_{n-2}
            let fn3 = &hist[n - 4]; // f_{n-3}
            for i in 0..dim {
                y_pred[i] = y[i]
                    + h * ((55.0 / 24.0) * fn0[i] - (59.0 / 24.0) * fn1[i]
                        + (37.0 / 24.0) * fn2[i]
                        - (9.0 / 24.0) * fn3[i]);
            }
        }
        _ => unreachable!("order already validated"),
    }
    y_pred
}

// ── Adams-Moulton corrector coefficients ────────────────────────────────────
// Order 1:  y = y + h * f_pred
// Order 2:  y = y + h * (1/2 f_pred + 1/2 f_n)
// Order 3:  y = y + h * (5/12 f_pred + 8/12 f_n - 1/12 f_{n-1})
// Order 4:  y = y + h * (9/24 f_pred + 19/24 f_n - 5/24 f_{n-1} + 1/24 f_{n-2})
//
// `f_pred` is the evaluation at (t+h, y_pred).  hist newest is f_n.

fn am_correct(y: &[f64], hist: &[Vec<f64>], f_pred: &[f64], h: f64, order: usize) -> Vec<f64> {
    let dim = y.len();
    let mut y_new = vec![0.0_f64; dim];
    let n = hist.len();
    match order {
        1 => {
            for i in 0..dim {
                y_new[i] = y[i] + h * f_pred[i];
            }
        }
        2 => {
            let fn0 = &hist[n - 1]; // f_n
            for i in 0..dim {
                y_new[i] = y[i] + h * (0.5 * f_pred[i] + 0.5 * fn0[i]);
            }
        }
        3 => {
            let fn0 = &hist[n - 1]; // f_n
            let fn1 = &hist[n - 2]; // f_{n-1}
            for i in 0..dim {
                y_new[i] = y[i]
                    + h * ((5.0 / 12.0) * f_pred[i] + (8.0 / 12.0) * fn0[i]
                        - (1.0 / 12.0) * fn1[i]);
            }
        }
        4 => {
            let fn0 = &hist[n - 1]; // f_n
            let fn1 = &hist[n - 2]; // f_{n-1}
            let fn2 = &hist[n - 3]; // f_{n-2}
            for i in 0..dim {
                y_new[i] = y[i]
                    + h * ((9.0 / 24.0) * f_pred[i] + (19.0 / 24.0) * fn0[i]
                        - (5.0 / 24.0) * fn1[i]
                        + (1.0 / 24.0) * fn2[i]);
            }
        }
        _ => unreachable!("order already validated"),
    }
    y_new
}

// ── Public interface ─────────────────────────────────────────────────────────

/// Adams-Bashforth-Moulton PECE integrator, orders 1–4.
///
/// # Parameters
/// - `f`     — RHS of `y' = f(t, y)` (fallible)
/// - `t0`    — initial time
/// - `tf`    — final time (must be > t0)
/// - `y0`    — initial state
/// - `h`     — fixed step size (must be finite and positive)
/// - `order` — method order in `1..=4`
///
/// # Returns
/// `(times, ys)` where `times[0] == t0` and `ys[0] == y0`.
pub fn adams_bashforth_moulton<F>(
    f: F,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h: f64,
    order: usize,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    // ── Input validation ────────────────────────────────────────────────────
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    if tf <= t0 {
        return Err(NumericError::InvalidParameter("tf must be > t0".into()));
    }
    if order == 0 || order > 4 {
        return Err(NumericError::InvalidParameter(
            "order must be in 1..=4".into(),
        ));
    }

    let dim = y0.len();
    let n_steps = ((tf - t0) / h).ceil() as usize;

    let mut times: Vec<f64> = Vec::with_capacity(n_steps + 1);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);

    // Push initial state.
    times.push(t0);
    ys.push(y0.to_vec());

    // ── Bootstrap using RK4 ─────────────────────────────────────────────────
    // We need `order - 1` additional steps from RK4 before starting ABM.
    // For order == 1 this is zero bootstrap steps.
    let boot_steps = order - 1;

    // Derivative history (oldest first, newest last), capacity = order.
    let mut hist: Vec<Vec<f64>> = Vec::with_capacity(order);

    // Evaluate f at the initial point.
    let f0 = f(t0, y0)?;
    if f0.len() != dim {
        return Err(NumericError::DimensionMismatch {
            a: f0.len(),
            b: dim,
        });
    }
    hist.push(f0);

    // Run RK4 for boot_steps steps, collecting intermediate (t, y) pairs.
    let mut current_y = y0.to_vec();
    let mut current_t = t0;

    for _bs in 0..boot_steps {
        if _bs >= n_steps {
            // The total integration range is shorter than bootstrap would need.
            // Return what we have so far.
            return Ok((times, ys));
        }
        // One RK4 step manually (avoids calling the rk4 function which does its own init push).
        let k1 = f(current_t, &current_y)?;
        if k1.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: k1.len(),
                b: dim,
            });
        }

        let mut ytmp = vec![0.0_f64; dim];
        for i in 0..dim {
            ytmp[i] = current_y[i] + 0.5 * h * k1[i];
        }
        let k2 = f(current_t + 0.5 * h, &ytmp)?;
        if k2.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: k2.len(),
                b: dim,
            });
        }

        for i in 0..dim {
            ytmp[i] = current_y[i] + 0.5 * h * k2[i];
        }
        let k3 = f(current_t + 0.5 * h, &ytmp)?;
        if k3.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: k3.len(),
                b: dim,
            });
        }

        for i in 0..dim {
            ytmp[i] = current_y[i] + h * k3[i];
        }
        let k4 = f(current_t + h, &ytmp)?;
        if k4.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: k4.len(),
                b: dim,
            });
        }

        let mut y_next = vec![0.0_f64; dim];
        for i in 0..dim {
            y_next[i] = current_y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        current_t += h;
        current_y = y_next;

        times.push(current_t);
        ys.push(current_y.clone());

        // Evaluate f at the new bootstrap point and push into history.
        let f_bs = f(current_t, &current_y)?;
        if f_bs.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: f_bs.len(),
                b: dim,
            });
        }
        hist.push(f_bs);
    }

    // ── Main ABM-PECE loop ───────────────────────────────────────────────────
    // At this point: `current_t` and `current_y` hold the last bootstrapped state;
    // `hist` contains f values at [t0, t1, ..., t_{order-1}] (length == order).
    let steps_done = boot_steps; // steps already taken
    let remaining = n_steps.saturating_sub(steps_done);

    for _ in 0..remaining {
        // P — Adams-Bashforth predictor
        let y_pred = ab_predict(&current_y, &hist, h, order);

        // E — Evaluate f at predicted point
        let f_pred = f(current_t + h, &y_pred)?;
        if f_pred.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: f_pred.len(),
                b: dim,
            });
        }

        // C — Adams-Moulton corrector
        let y_new = am_correct(&current_y, &hist, &f_pred, h, order);

        // E — Evaluate f at corrected point (becomes new f_n for next step)
        let f_new = f(current_t + h, &y_new)?;
        if f_new.len() != dim {
            return Err(NumericError::DimensionMismatch {
                a: f_new.len(),
                b: dim,
            });
        }

        // Advance state
        current_t += h;
        current_y = y_new;

        times.push(current_t);
        ys.push(current_y.clone());

        // Slide derivative history: drop oldest, push newest.
        if hist.len() >= order {
            hist.remove(0);
        }
        hist.push(f_new);
    }

    Ok((times, ys))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    type F64RhsFn = fn(f64, &[f64]) -> NumericResult<Vec<f64>>;

    fn exp_decay(t: f64, y: &[f64]) -> NumericResult<Vec<f64>> {
        let _ = t;
        Ok(vec![-y[0]])
    }

    fn exp_growth(t: f64, y: &[f64]) -> NumericResult<Vec<f64>> {
        let _ = t;
        Ok(vec![y[0]])
    }

    fn harmonic(_t: f64, y: &[f64]) -> NumericResult<Vec<f64>> {
        Ok(vec![y[1], -y[0]])
    }

    // ── Test 1: exponential decay, order=4 ──────────────────────────────────
    #[test]
    fn abm4_exponential_decay() {
        let (ts, ys) = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 2.0, &[1.0], 0.01, 4)
            .expect("abm4 decay");
        for (t, yv) in ts.iter().zip(ys.iter()) {
            let exact = (-t).exp();
            assert!(
                (yv[0] - exact).abs() < 1.0e-6,
                "t={t:.3} exact={exact:.8} got={:.8}",
                yv[0]
            );
        }
    }

    // ── Test 2: exponential growth, order=4 ─────────────────────────────────
    #[test]
    fn abm4_exponential_growth() {
        let (ts, ys) = adams_bashforth_moulton(exp_growth as F64RhsFn, 0.0, 1.0, &[1.0], 0.01, 4)
            .expect("abm4 growth");
        for (t, yv) in ts.iter().zip(ys.iter()) {
            let exact = t.exp();
            assert!(
                (yv[0] - exact).abs() < 1.0e-6,
                "t={t:.3} exact={exact:.8} got={:.8}",
                yv[0]
            );
        }
    }

    // ── Test 3: harmonic oscillator, energy conservation ────────────────────
    #[test]
    fn abm4_harmonic_oscillator_energy() {
        let t_end = 10.0 * 2.0 * PI; // 10 full periods
        let (_, ys) =
            adams_bashforth_moulton(harmonic as F64RhsFn, 0.0, t_end, &[1.0, 0.0], 0.01, 4)
                .expect("harmonic");
        let last = ys.last().expect("non-empty");
        let norm_sq = last[0] * last[0] + last[1] * last[1];
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 1.0e-4,
            "energy norm deviation: {}",
            norm_sq.sqrt()
        );
    }

    // ── Test 4: order-2 empirical convergence — ratio ≈ 4 ──────────────────
    #[test]
    fn abm2_order_convergence() {
        let exact = (-1.0_f64).exp(); // y(1) for y'=-y, y(0)=1
        let (ts_h, ys_h) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.05, 2)
                .expect("abm2 h");
        let err_h = (ys_h.last().expect("ne")[0] - exact).abs();

        let (ts_h2, ys_h2) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.025, 2)
                .expect("abm2 h/2");
        let err_h2 = (ys_h2.last().expect("ne")[0] - exact).abs();
        let _ = (ts_h.len(), ts_h2.len());

        let ratio = err_h / err_h2;
        assert!(
            (3.0..=5.0).contains(&ratio),
            "order-2 convergence ratio = {ratio:.4} (expected ≈4)"
        );
    }

    // ── Test 5: order-3 empirical convergence — ratio ≈ 8 ──────────────────
    #[test]
    fn abm3_order_convergence() {
        let exact = (-1.0_f64).exp();
        let (ts_h, ys_h) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.05, 3)
                .expect("abm3 h");
        let err_h = (ys_h.last().expect("ne")[0] - exact).abs();

        let (ts_h2, ys_h2) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.025, 3)
                .expect("abm3 h/2");
        let err_h2 = (ys_h2.last().expect("ne")[0] - exact).abs();
        let _ = (ts_h.len(), ts_h2.len());

        let ratio = err_h / err_h2;
        assert!(
            (6.0..=10.0).contains(&ratio),
            "order-3 convergence ratio = {ratio:.4} (expected ≈8)"
        );
    }

    // ── Test 6: order-4 empirical convergence — ratio ≈ 16 ─────────────────
    #[test]
    fn abm4_order_convergence() {
        let exact = (-1.0_f64).exp();
        let (ts_h, ys_h) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.05, 4)
                .expect("abm4 h");
        let err_h = (ys_h.last().expect("ne")[0] - exact).abs();

        let (ts_h2, ys_h2) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.025, 4)
                .expect("abm4 h/2");
        let err_h2 = (ys_h2.last().expect("ne")[0] - exact).abs();
        let _ = (ts_h.len(), ts_h2.len());

        let ratio = err_h / err_h2;
        assert!(
            (12.0..=20.0).contains(&ratio),
            "order-4 convergence ratio = {ratio:.4} (expected ≈16)"
        );
    }

    // ── Test 7: ABM4 more accurate than explicit Euler at same h ────────────
    #[test]
    fn abm4_beats_euler_accuracy() {
        let h = 0.1;
        let exact = (-1.0_f64).exp();

        // Explicit Euler
        let mut y_eu = 1.0_f64;
        let mut t_eu = 0.0_f64;
        while t_eu + h <= 1.0 + 1.0e-12 {
            y_eu += h * (-y_eu);
            t_eu += h;
        }
        let err_euler = (y_eu - exact).abs();

        let (_, ys_abm) =
            adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], h, 4).expect("abm4");
        let err_abm = (ys_abm.last().expect("ne")[0] - exact).abs();

        assert!(
            err_abm < err_euler,
            "ABM4 error {err_abm:.2e} not < Euler error {err_euler:.2e}"
        );
    }

    // ── Test 8: order=1 (backward Euler corrector) close to exact ───────────
    #[test]
    fn abm1_close_to_exact() {
        // Order-1 ABM uses AB1 predictor (forward Euler) + AM1 corrector (backward Euler).
        // The PECE mode applies one corrector evaluation, giving O(h) global error ≈ 1e-3 at t=1.
        // Use a smaller h to bring the error down.
        let (ts, ys) = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.0001, 1)
            .expect("abm1");
        let last_t = *ts.last().expect("ne");
        let exact = (-last_t).exp();
        let err = (ys.last().expect("ne")[0] - exact).abs();
        assert!(err < 1.0e-3, "order-1 error = {err:.2e}");
    }

    // ── Test 9: order=0 → InvalidParameter ──────────────────────────────────
    #[test]
    fn abm_order_zero_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.01, 0);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    // ── Test 10: order=5 → InvalidParameter ─────────────────────────────────
    #[test]
    fn abm_order_five_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.01, 5);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    // ── Test 11: h=0 → InvalidStepSize ──────────────────────────────────────
    #[test]
    fn abm_h_zero_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], 0.0, 4);
        assert!(matches!(res, Err(NumericError::InvalidStepSize { .. })));
    }

    // ── Test 12: h negative → InvalidStepSize ───────────────────────────────
    #[test]
    fn abm_h_negative_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], -0.1, 4);
        assert!(matches!(res, Err(NumericError::InvalidStepSize { .. })));
    }

    // ── Test 13: tf < t0 → InvalidParameter ─────────────────────────────────
    #[test]
    fn abm_tf_lt_t0_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 1.0, 0.0, &[1.0], 0.01, 4);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    // ── Test 14: tf == t0 → InvalidParameter ────────────────────────────────
    #[test]
    fn abm_tf_eq_t0_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 1.0, 1.0, &[1.0], 0.01, 4);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    // ── Test 15: RHS dimension mismatch → DimensionMismatch ─────────────────
    #[test]
    fn abm_dim_mismatch_err() {
        let bad_f = |_t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![1.0, 2.0]) // returns dim 2 but y0 has dim 1
        };
        let res = adams_bashforth_moulton(bad_f, 0.0, 1.0, &[1.0], 0.01, 4);
        assert!(matches!(res, Err(NumericError::DimensionMismatch { .. })));
    }

    // ── Test 16: matches RK4 within 1% on 2-D linear system ─────────────────
    #[test]
    fn abm4_matches_rk4_2d() {
        // y' = [-0.5*y0 + 0.1*y1, 0.1*y0 - 0.5*y1]
        let linear_2d = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![-0.5 * y[0] + 0.1 * y[1], 0.1 * y[0] - 0.5 * y[1]])
        };
        let (_, ys_abm) =
            adams_bashforth_moulton(linear_2d, 0.0, 2.0, &[1.0, 0.5], 0.01, 4).expect("abm4 2d");
        let (_, ys_rk4) =
            crate::ode::rk4::rk4(linear_2d, 0.0, 2.0, &[1.0, 0.5], 0.01).expect("rk4 2d");

        let abm_last = ys_abm.last().expect("ne");
        let rk4_last = ys_rk4.last().expect("ne");
        for k in 0..2 {
            let rel_err = (abm_last[k] - rk4_last[k]).abs() / (rk4_last[k].abs() + 1.0e-12);
            assert!(
                rel_err < 0.01,
                "comp {k}: abm={:.8} rk4={:.8} rel_err={rel_err:.4}",
                abm_last[k],
                rk4_last[k]
            );
        }
    }

    // ── Test 17: 3-D diagonal system, check each component ──────────────────
    #[test]
    fn abm4_three_dim_diagonal() {
        // y' = diag(-1, -2, -3) * y
        let diag_3d = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> {
            Ok(vec![-y[0], -2.0 * y[1], -3.0 * y[2]])
        };
        let y0 = [1.0, 1.0, 1.0];
        let t_end = 1.0;
        let (ts, ys) = adams_bashforth_moulton(diag_3d, 0.0, t_end, &y0, 0.01, 4).expect("3d");
        for (t, yv) in ts.iter().zip(ys.iter()) {
            let exact = [(-t).exp(), (-2.0 * t).exp(), (-3.0 * t).exp()];
            for k in 0..3 {
                assert!(
                    (yv[k] - exact[k]).abs() < 1.0e-5,
                    "comp {k} t={t:.3} exact={:.8} got={:.8}",
                    exact[k],
                    yv[k]
                );
            }
        }
    }

    // ── Test 18: order=2 on y'=cos(t) → y≈sin(t) ───────────────────────────
    #[test]
    fn abm2_cosine_rhs() {
        let cos_f = |t: f64, _y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![t.cos()]) };
        let (ts, ys) = adams_bashforth_moulton(cos_f, 0.0, PI, &[0.0], 0.001, 2).expect("abm2 cos");
        for (t, yv) in ts.iter().zip(ys.iter()) {
            let exact = t.sin();
            assert!(
                (yv[0] - exact).abs() < 1.0e-5,
                "t={t:.4} exact={exact:.8} got={:.8}",
                yv[0]
            );
        }
    }

    // ── Test 19: large t range, oscillator stays bounded ────────────────────
    #[test]
    fn abm4_large_t_oscillator_bounded() {
        let (_, ys) =
            adams_bashforth_moulton(harmonic as F64RhsFn, 0.0, 10.0, &[1.0, 0.0], 0.01, 4)
                .expect("long harmonic");
        for yv in &ys {
            let norm = (yv[0] * yv[0] + yv[1] * yv[1]).sqrt();
            assert!(norm < 2.0, "solution blew up: norm={norm}");
        }
    }

    // ── Test 20: h=NAN → InvalidStepSize ────────────────────────────────────
    #[test]
    fn abm_h_nan_err() {
        let res = adams_bashforth_moulton(exp_decay as F64RhsFn, 0.0, 1.0, &[1.0], f64::NAN, 4);
        assert!(matches!(res, Err(NumericError::InvalidStepSize { .. })));
    }
}
