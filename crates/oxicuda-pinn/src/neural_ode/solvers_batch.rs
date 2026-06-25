//! Batched ODE integrators: integrate `B` independent IVP systems in one call.
//!
//! These mirror the single-system steppers in [`crate::neural_ode::solvers`]
//! ([`euler_step`](crate::neural_ode::solvers::euler_step),
//! [`heun_step`](crate::neural_ode::solvers::heun_step),
//! [`rk4_step`](crate::neural_ode::solvers::rk4_step)) exactly: the same
//! Butcher-tableau coefficients and the same per-stage floating-point
//! expressions are reused, so a batch of identical IVPs reproduces the scalar
//! trajectory bit-for-bit. The only difference is that each of the `B` systems
//! may carry its *own* dynamics, selected through the leading system index in
//! the right-hand-side closure.
//!
//! The state buffer `y_batch` is laid out row-major / system-major as
//! `[B * dim]`: system `i` occupies the contiguous slice
//! `y_batch[i * dim .. (i + 1) * dim]`.

use crate::error::{PinnError, PinnResult};

/// Batched ODE right-hand side: `f(system_index, t, y_i, dydt_i)`.
///
/// `system_index` selects which of the `B` independent systems is being
/// evaluated, allowing every system to have distinct dynamics; `y_i` and
/// `dydt_i` are the `dim`-length state / derivative slices for that system.
pub type OdeRhsFnBatch<'a> = &'a dyn Fn(usize, f32, &[f32], &mut [f32]);

// ─── Single-step methods ──────────────────────────────────────────────────────

/// Batched explicit Euler step: for each system `i`,
/// `y_new_i = y_i + h * f(i, t, y_i)`.
///
/// Mirrors [`crate::neural_ode::solvers::euler_step`] for every system and
/// returns the stacked `[B * dim]` next state.
///
/// # Panics
/// Panics (in debug builds, via `debug_assert`) if `y_batch.len() != batch * dim`.
pub fn euler_step_batch(
    rhs: OdeRhsFnBatch,
    t: f32,
    y_batch: &[f32],
    batch: usize,
    dim: usize,
    h: f32,
) -> Vec<f32> {
    debug_assert_eq!(y_batch.len(), batch * dim, "y_batch must be batch * dim");
    let mut out = vec![0.0_f32; batch * dim];
    let mut k = vec![0.0_f32; dim];
    for i in 0..batch {
        let base = i * dim;
        let y_i = &y_batch[base..base + dim];
        rhs(i, t, y_i, &mut k);
        for (o, (&yi, &ki)) in out[base..base + dim]
            .iter_mut()
            .zip(y_i.iter().zip(k.iter()))
        {
            *o = yi + h * ki;
        }
    }
    out
}

/// Batched Heun (RK2) step: for each system `i`, `k1 = f(i, t, y_i)`,
/// `k2 = f(i, t+h, y_i + h*k1)`, `y_new_i = y_i + h/2*(k1 + k2)`.
///
/// Mirrors [`crate::neural_ode::solvers::heun_step`] for every system.
///
/// # Panics
/// Panics (in debug builds, via `debug_assert`) if `y_batch.len() != batch * dim`.
pub fn heun_step_batch(
    rhs: OdeRhsFnBatch,
    t: f32,
    y_batch: &[f32],
    batch: usize,
    dim: usize,
    h: f32,
) -> Vec<f32> {
    debug_assert_eq!(y_batch.len(), batch * dim, "y_batch must be batch * dim");
    let mut out = vec![0.0_f32; batch * dim];
    let mut k1 = vec![0.0_f32; dim];
    let mut k2 = vec![0.0_f32; dim];
    for i in 0..batch {
        let base = i * dim;
        let y_i = &y_batch[base..base + dim];
        rhs(i, t, y_i, &mut k1);
        let y_euler: Vec<f32> = y_i
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &ki)| yi + h * ki)
            .collect();
        rhs(i, t + h, &y_euler, &mut k2);
        for (o, ((&yi, &k1i), &k2i)) in out[base..base + dim]
            .iter_mut()
            .zip(y_i.iter().zip(k1.iter()).zip(k2.iter()))
        {
            *o = yi + h * 0.5 * (k1i + k2i);
        }
    }
    out
}

/// Batched classic RK4 step: for each system `i`, the standard four-stage
/// Runge-Kutta update reusing the exact tableau of
/// [`crate::neural_ode::solvers::rk4_step`].
///
/// # Panics
/// Panics (in debug builds, via `debug_assert`) if `y_batch.len() != batch * dim`.
pub fn rk4_step_batch(
    rhs: OdeRhsFnBatch,
    t: f32,
    y_batch: &[f32],
    batch: usize,
    dim: usize,
    h: f32,
) -> Vec<f32> {
    debug_assert_eq!(y_batch.len(), batch * dim, "y_batch must be batch * dim");
    let mut out = vec![0.0_f32; batch * dim];
    let mut k1 = vec![0.0_f32; dim];
    let mut k2 = vec![0.0_f32; dim];
    let mut k3 = vec![0.0_f32; dim];
    let mut k4 = vec![0.0_f32; dim];

    for i in 0..batch {
        let base = i * dim;
        let y_i = &y_batch[base..base + dim];

        rhs(i, t, y_i, &mut k1);

        let y2: Vec<f32> = y_i
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &ki)| yi + 0.5 * h * ki)
            .collect();
        rhs(i, t + 0.5 * h, &y2, &mut k2);

        let y3: Vec<f32> = y_i
            .iter()
            .zip(k2.iter())
            .map(|(&yi, &ki)| yi + 0.5 * h * ki)
            .collect();
        rhs(i, t + 0.5 * h, &y3, &mut k3);

        let y4: Vec<f32> = y_i
            .iter()
            .zip(k3.iter())
            .map(|(&yi, &ki)| yi + h * ki)
            .collect();
        rhs(i, t + h, &y4, &mut k4);

        for (o, ((((&yi, &k1i), &k2i), &k3i), &k4i)) in out[base..base + dim].iter_mut().zip(
            y_i.iter()
                .zip(k1.iter())
                .zip(k2.iter())
                .zip(k3.iter())
                .zip(k4.iter()),
        ) {
            *o = yi + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i);
        }
    }
    out
}

// ─── Fixed-step batched integration ───────────────────────────────────────────

/// Integrate `batch` independent IVP systems with a fixed step size using the
/// batched RK4 stepper.
///
/// Each system carries its own dynamics through the `rhs` system index. The
/// state layout is `[batch * dim]` (system-major). Returns `(times, states)`
/// where `states[k]` is the stacked `[batch * dim]` solution at `times[k]`;
/// this mirrors [`crate::neural_ode::solvers::integrate_fixed`].
pub fn integrate_batch(
    rhs: OdeRhsFnBatch,
    t0: f32,
    t1: f32,
    y0_batch: &[f32],
    batch: usize,
    dim: usize,
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<Vec<f32>>)> {
    if batch == 0 || dim == 0 {
        return Err(PinnError::EmptyInput);
    }
    if y0_batch.len() != batch * dim {
        return Err(PinnError::DimensionMismatch {
            expected: batch * dim,
            got: y0_batch.len(),
        });
    }
    if h <= 0.0 || !h.is_finite() {
        return Err(PinnError::InvalidStepSize { h });
    }
    if t1 <= t0 {
        return Err(PinnError::InvalidTimeInterval { t0, t1 });
    }

    let mut times = vec![t0];
    let mut states = vec![y0_batch.to_vec()];
    let mut t = t0;
    let mut y = y0_batch.to_vec();

    while t < t1 {
        let h_eff = h.min(t1 - t);
        let y_new = rk4_step_batch(rhs, t, &y, batch, dim, h_eff);
        if y_new.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::SolverDivergence {
                reason: "NaN/Inf in batched RK4 state",
            });
        }
        t += h_eff;
        times.push(t);
        states.push(y_new.clone());
        y = y_new;
    }
    Ok((times, states))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_ode::solvers::{euler_step, heun_step, integrate_fixed, rk4_step};

    // ── Reference dynamics ──────────────────────────────────────────────────

    // Scalar dy/dt = -y, y(0) = 1 → y(t) = exp(-t).
    fn exp_decay(_t: f32, y: &[f32], dydt: &mut [f32]) {
        dydt[0] = -y[0];
    }

    // Batched dy/dt = -y for every system (identical dynamics).
    fn exp_decay_batch(_i: usize, _t: f32, y: &[f32], dydt: &mut [f32]) {
        dydt[0] = -y[0];
    }

    // Scalar harmonic oscillator: dy/dt = v, dv/dt = -y.
    fn spring(_t: f32, y: &[f32], dydt: &mut [f32]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }

    // Batched harmonic oscillator for every system.
    fn spring_batch(_i: usize, _t: f32, y: &[f32], dydt: &mut [f32]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }

    // ── EQUIVALENCE: batch of identical IVPs == repeated scalar solve ───────

    #[test]
    fn euler_step_batch_matches_scalar_bit_exact() {
        let batch = 4;
        let dim = 1;
        let y_batch = vec![1.0_f32; batch * dim];
        let h = 0.05_f32;
        let got = euler_step_batch(&exp_decay_batch, 0.0, &y_batch, batch, dim, h);
        let expect = euler_step(&exp_decay, 0.0, &[1.0], h);
        for i in 0..batch {
            assert_eq!(
                &got[i * dim..(i + 1) * dim],
                expect.as_slice(),
                "euler batch system {i} must equal scalar euler bit-for-bit"
            );
        }
    }

    #[test]
    fn heun_step_batch_matches_scalar_bit_exact() {
        let batch = 3;
        let dim = 1;
        let y_batch = vec![1.0_f32; batch * dim];
        let h = 0.05_f32;
        let got = heun_step_batch(&exp_decay_batch, 0.0, &y_batch, batch, dim, h);
        let expect = heun_step(&exp_decay, 0.0, &[1.0], h);
        for i in 0..batch {
            assert_eq!(&got[i * dim..(i + 1) * dim], expect.as_slice());
        }
    }

    #[test]
    fn rk4_step_batch_matches_scalar_bit_exact() {
        let batch = 5;
        let dim = 1;
        let y_batch = vec![1.0_f32; batch * dim];
        let h = 0.1_f32;
        let got = rk4_step_batch(&exp_decay_batch, 0.0, &y_batch, batch, dim, h);
        let expect = rk4_step(&exp_decay, 0.0, &[1.0], h);
        for i in 0..batch {
            assert_eq!(&got[i * dim..(i + 1) * dim], expect.as_slice());
        }
    }

    #[test]
    fn rk4_step_batch_matches_scalar_multidim_bit_exact() {
        // dim = 2 harmonic oscillator, B = 3 identical systems.
        let batch = 3;
        let dim = 2;
        let y0 = [1.0_f32, 0.3];
        let mut y_batch = Vec::with_capacity(batch * dim);
        for _ in 0..batch {
            y_batch.extend_from_slice(&y0);
        }
        let h = 0.07_f32;
        let got = rk4_step_batch(&spring_batch, 0.0, &y_batch, batch, dim, h);
        let expect = rk4_step(&spring, 0.0, &y0, h);
        for i in 0..batch {
            assert_eq!(
                &got[i * dim..(i + 1) * dim],
                expect.as_slice(),
                "multidim RK4 batch system {i} must equal scalar bit-for-bit"
            );
        }
    }

    #[test]
    fn integrate_batch_matches_integrate_fixed_bit_exact() {
        let batch = 4;
        let dim = 1;
        let y0_batch = vec![1.0_f32; batch * dim];
        let (times_b, states_b) =
            integrate_batch(&exp_decay_batch, 0.0, 1.0, &y0_batch, batch, dim, 0.01)
                .expect("batched fixed-step integration of exp-decay should succeed");
        let (times_s, states_s) = integrate_fixed(&exp_decay, 0.0, 1.0, &[1.0], 0.01)
            .expect("scalar fixed-step integration of exp-decay should succeed");

        assert_eq!(times_b, times_s, "time grids must match bit-for-bit");
        assert_eq!(states_b.len(), states_s.len());
        for (sb, ss) in states_b.iter().zip(states_s.iter()) {
            for i in 0..batch {
                assert_eq!(
                    &sb[i * dim..(i + 1) * dim],
                    ss.as_slice(),
                    "every batched system must equal the scalar trajectory bit-for-bit"
                );
            }
        }
    }

    // ── INDEPENDENCE: distinct linear IVPs y' = λ_i y match e^{λ_i t} ───────

    #[test]
    fn integrate_batch_distinct_lambda_closed_form() {
        // y'_i = λ_i y_i, y_i(0) = 1 → y_i(t) = e^{λ_i t}.
        let lambdas = [-2.0_f32, -0.5, 0.5, 1.0];
        let batch = lambdas.len();
        let dim = 1;
        let rhs = |i: usize, _t: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = lambdas[i] * y[0];
        };
        let y0_batch = vec![1.0_f32; batch * dim];
        let t_final = 0.5_f32;
        let (_, states) = integrate_batch(&rhs, 0.0, t_final, &y0_batch, batch, dim, 0.01)
            .expect("batched integration of distinct linear IVPs should succeed");
        let final_state = states
            .last()
            .expect("batched integration produced no states");
        for (i, &lam) in lambdas.iter().enumerate() {
            let got = final_state[i * dim];
            let exact = (lam * t_final).exp();
            assert!(
                (got - exact).abs() < 1e-5,
                "system {i} (λ={lam}): RK4 y={got} vs exact e^(λt)={exact}"
            );
        }
    }

    #[test]
    fn integrate_batch_independence_no_cross_talk() {
        // Two runs that differ ONLY in system 1's dynamics must produce
        // bit-identical trajectories for every other system.
        let dim = 1;
        let batch = 4;
        let y0_batch = vec![1.0_f32; batch * dim];

        let lambdas_a = [-2.0_f32, -0.5, 0.5, 1.0];
        let lambdas_b = [-2.0_f32, 3.7, 0.5, 1.0]; // only index 1 changed
        let rhs_a = |i: usize, _t: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = lambdas_a[i] * y[0];
        };
        let rhs_b = |i: usize, _t: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = lambdas_b[i] * y[0];
        };

        let (_, states_a) = integrate_batch(&rhs_a, 0.0, 0.5, &y0_batch, batch, dim, 0.01)
            .expect("run A should succeed");
        let (_, states_b) = integrate_batch(&rhs_b, 0.0, 0.5, &y0_batch, batch, dim, 0.01)
            .expect("run B should succeed");

        assert_eq!(states_a.len(), states_b.len());
        for (sa, sb) in states_a.iter().zip(states_b.iter()) {
            for i in 0..batch {
                if i == 1 {
                    continue; // system 1 is intentionally different
                }
                assert_eq!(
                    &sa[i * dim..(i + 1) * dim],
                    &sb[i * dim..(i + 1) * dim],
                    "system {i} must be unaffected by changing system 1 (no cross-talk)"
                );
            }
        }
        // And the perturbed system actually differs at the final time.
        let fa = states_a.last().expect("A non-empty")[1];
        let fb = states_b.last().expect("B non-empty")[1];
        assert!(
            (fa - fb).abs() > 1e-3,
            "system 1 should respond to its own changed dynamics: {fa} vs {fb}"
        );
    }

    #[test]
    fn euler_step_batch_distinct_lambda_direction() {
        // Looser (1st-order) check that each Euler system tracks its own e^{λt}.
        let lambdas = [-1.0_f32, -0.25, 0.25, 1.0];
        let batch = lambdas.len();
        let dim = 1;
        let rhs = |i: usize, _t: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = lambdas[i] * y[0];
        };
        let mut y = vec![1.0_f32; batch * dim];
        let h = 0.001_f32;
        let n_steps = 500;
        let mut t = 0.0_f32;
        for _ in 0..n_steps {
            y = euler_step_batch(&rhs, t, &y, batch, dim, h);
            t += h;
        }
        let t_final = h * n_steps as f32;
        for (i, &lam) in lambdas.iter().enumerate() {
            let got = y[i * dim];
            let exact = (lam * t_final).exp();
            assert!(got.is_finite(), "Euler system {i} not finite");
            assert!(
                (got - exact).abs() < 1e-2,
                "Euler system {i} (λ={lam}): y={got} vs e^(λt)={exact}"
            );
            if lam < 0.0 {
                assert!(got < 1.0, "decaying system {i} should drop below 1");
            } else {
                assert!(got > 1.0, "growing system {i} should rise above 1");
            }
        }
    }

    // ── ORDER / DETERMINISM ─────────────────────────────────────────────────

    // Integrate one exp-decay system through the BATCH RK4 stepper from 0 to
    // `t_final` with fixed step `h`; return |y(t_final) - e^{-t_final}|.
    fn rk4_batch_exp_decay_error(h: f32, t_final: f32) -> f32 {
        let batch = 1;
        let dim = 1;
        let mut y = vec![1.0_f32];
        let n = (t_final / h).round() as usize;
        let mut t = 0.0_f32;
        for _ in 0..n {
            y = rk4_step_batch(&exp_decay_batch, t, &y, batch, dim, h);
            t += h;
        }
        (y[0] - (-t_final).exp()).abs()
    }

    #[test]
    fn rk4_batch_fourth_order_convergence() {
        // RK4 is 4th order: halving h cuts the global error by ≈ 2^4 = 16.
        let t_final = 2.0_f32;
        let err_coarse = rk4_batch_exp_decay_error(0.5, t_final);
        let err_fine = rk4_batch_exp_decay_error(0.25, t_final);
        assert!(err_coarse > 0.0 && err_fine > 0.0);
        let ratio = err_coarse / err_fine;
        // ≈ 16 for 4th order; band excludes 1st (2), 2nd (4), 3rd (8) order.
        assert!(
            (10.0..28.0).contains(&ratio),
            "RK4 batch 4th-order error ratio ≈16 expected, got {ratio} \
             (err_coarse={err_coarse:e}, err_fine={err_fine:e})"
        );
    }

    #[test]
    fn integrate_batch_is_deterministic() {
        let lambdas = [-1.5_f32, 0.0, 0.7];
        let batch = lambdas.len();
        let dim = 1;
        let rhs = |i: usize, _t: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = lambdas[i] * y[0];
        };
        let y0 = vec![1.0_f32; batch * dim];
        let run = || {
            integrate_batch(&rhs, 0.0, 1.0, &y0, batch, dim, 0.02)
                .expect("deterministic batched run should succeed")
        };
        let (t1, s1) = run();
        let (t2, s2) = run();
        assert_eq!(t1, t2, "times must be reproducible bit-for-bit");
        assert_eq!(s1, s2, "states must be reproducible bit-for-bit");
    }

    // ── Error-path validation (mirrors scalar integrate_fixed) ──────────────

    #[test]
    fn integrate_batch_invalid_step_size_error() {
        let r = integrate_batch(&exp_decay_batch, 0.0, 1.0, &[1.0], 1, 1, -0.1);
        assert!(matches!(r, Err(PinnError::InvalidStepSize { .. })));
    }

    #[test]
    fn integrate_batch_invalid_time_interval_error() {
        let r = integrate_batch(&exp_decay_batch, 1.0, 0.5, &[1.0], 1, 1, 0.1);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
    }

    #[test]
    fn integrate_batch_dimension_mismatch_error() {
        // batch * dim = 6 but only 4 values supplied.
        let r = integrate_batch(&exp_decay_batch, 0.0, 1.0, &[1.0, 1.0, 1.0, 1.0], 3, 2, 0.1);
        assert!(matches!(
            r,
            Err(PinnError::DimensionMismatch {
                expected: 6,
                got: 4
            })
        ));
    }

    #[test]
    fn integrate_batch_empty_input_error() {
        let r = integrate_batch(&exp_decay_batch, 0.0, 1.0, &[], 0, 1, 0.1);
        assert!(matches!(r, Err(PinnError::EmptyInput)));
    }
}
