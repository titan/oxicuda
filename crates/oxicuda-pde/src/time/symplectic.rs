//! Symplectic integrators for Hamiltonian dynamical systems.
//!
//! These methods preserve the symplectic structure (phase-space volume) exactly
//! and exhibit near-conservation of energy over long time horizons.
//!
//! Two methods are provided:
//! - **Velocity Verlet** (Störmer-Verlet 1907) — 2nd-order symplectic, O(dt²) error per step.
//! - **Forest-Ruth** (Forest-Ruth 1990 / Yoshida 1990) — 4th-order symplectic, three sub-steps.

use crate::error::{PdeError, PdeResult};

// ── internal helpers ───────────────────────────────────────────────────────────

/// Validate common preconditions shared by every exported function.
fn validate_qv_dt(q: &[f64], v: &[f64], dt: f64) -> PdeResult<()> {
    if dt <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "dt".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if q.is_empty() {
        return Err(PdeError::InvalidParameter {
            name: "q".to_string(),
            reason: "must be non-empty".to_string(),
        });
    }
    if q.len() != v.len() {
        return Err(PdeError::DimensionMismatch {
            a: q.len(),
            b: v.len(),
        });
    }
    Ok(())
}

/// Validate that an acceleration vector returned by the user's closure has the
/// correct length.
fn check_accel_len(a: &[f64], expected: usize) -> PdeResult<()> {
    if a.len() != expected {
        return Err(PdeError::DimensionMismatch {
            a: a.len(),
            b: expected,
        });
    }
    Ok(())
}

// ── Velocity Verlet ────────────────────────────────────────────────────────────

/// A single velocity Verlet (Störmer-Verlet) step for Hamiltonian systems.
///
/// Updates `(q, v)` by one step of size `dt`:
/// 1. `v_{1/2} = v + (dt/2) * a(q)`
/// 2. `q_new   = q + dt * v_{1/2}`
/// 3. `v_new   = v_{1/2} + (dt/2) * a(q_new)`
///
/// where `a(q) = force / mass` (acceleration from positions).
///
/// # Arguments
/// * `q`     — positions (length d), modified in place.
/// * `v`     — velocities (length d), modified in place.
/// * `dt`    — time step (must be > 0).
/// * `accel` — acceleration function `a(q)`, returning `Vec<f64>` of the same length as `q`.
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] if `dt ≤ 0` or `q` is empty,
/// [`PdeError::DimensionMismatch`] if `q.len() != v.len()` or the closure
/// returns a vector of the wrong length.
pub fn velocity_verlet_step<F>(q: &mut [f64], v: &mut [f64], dt: f64, accel: F) -> PdeResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_qv_dt(q, v, dt)?;
    let d = q.len();
    let half_dt = 0.5 * dt;

    // half-kick: v_{1/2} = v + (dt/2) * a(q)
    let a0 = accel(q);
    check_accel_len(&a0, d)?;
    for i in 0..d {
        v[i] += half_dt * a0[i];
    }

    // full drift: q = q + dt * v_{1/2}
    for i in 0..d {
        q[i] += dt * v[i];
    }

    // half-kick with new acceleration: v = v_{1/2} + (dt/2) * a(q_new)
    let a1 = accel(q);
    check_accel_len(&a1, d)?;
    for i in 0..d {
        v[i] += half_dt * a1[i];
    }

    Ok(())
}

/// Multiple velocity Verlet steps.
///
/// Equivalent to calling [`velocity_verlet_step`] `n_steps` times in sequence.
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] if `n_steps == 0`, or any error
/// from an individual step.
pub fn velocity_verlet<F>(
    q: &mut [f64],
    v: &mut [f64],
    dt: f64,
    n_steps: usize,
    accel: F,
) -> PdeResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_qv_dt(q, v, dt)?;
    if n_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_steps".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    let d = q.len();
    let half_dt = 0.5 * dt;

    for _ in 0..n_steps {
        // half-kick
        let a0 = accel(q);
        check_accel_len(&a0, d)?;
        for i in 0..d {
            v[i] += half_dt * a0[i];
        }
        // full drift
        for i in 0..d {
            q[i] += dt * v[i];
        }
        // half-kick with new acceleration
        let a1 = accel(q);
        check_accel_len(&a1, d)?;
        for i in 0..d {
            v[i] += half_dt * a1[i];
        }
    }

    Ok(())
}

// ── Forest-Ruth 4th-order ─────────────────────────────────────────────────────

/// Yoshida (1990) composition coefficients for the Forest-Ruth integrator.
///
/// `θ = 1 / (2 − 2^{1/3})`
/// ```text
/// c1 = c4 = θ/2
/// c2 = c3 = (1−θ)/2
/// d1 = d3 = θ
/// d2      = 1 − 2θ
/// ```
#[inline]
fn forest_ruth_coefficients() -> (f64, f64, f64, f64) {
    let theta = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
    let c1 = theta / 2.0;
    let c2 = (1.0 - theta) / 2.0;
    let d1 = theta;
    let d2 = 1.0 - 2.0 * theta;
    (c1, c2, d1, d2)
}

/// A single Forest-Ruth 4th-order symplectic integrator step.
///
/// Uses the Yoshida (1990) composition with coefficients:
/// ```text
/// θ = 1 / (2 − 2^{1/3}),  c1 = c4 = θ/2,  c2 = c3 = (1−θ)/2,
/// d1 = d3 = θ,              d2 = 1 − 2θ
/// ```
/// Sub-step sequence (4 position updates, 3 acceleration evaluations):
/// ```text
/// q += c1·dt·v;  v += d1·dt·a(q)
/// q += c2·dt·v;  v += d2·dt·a(q)
/// q += c2·dt·v;  v += d1·dt·a(q)   (c3=c2, d3=d1)
/// q += c1·dt·v                       (c4=c1)
/// ```
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] if `dt ≤ 0` or `q` is empty,
/// [`PdeError::DimensionMismatch`] if dimensions mismatch.
pub fn forest_ruth_step<F>(q: &mut [f64], v: &mut [f64], dt: f64, accel: F) -> PdeResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_qv_dt(q, v, dt)?;
    let d = q.len();
    let (c1, c2, d1, d2) = forest_ruth_coefficients();

    // sub-step 1
    for i in 0..d {
        q[i] += c1 * dt * v[i];
    }
    let a0 = accel(q);
    check_accel_len(&a0, d)?;
    for i in 0..d {
        v[i] += d1 * dt * a0[i];
    }

    // sub-step 2
    for i in 0..d {
        q[i] += c2 * dt * v[i];
    }
    let a1 = accel(q);
    check_accel_len(&a1, d)?;
    for i in 0..d {
        v[i] += d2 * dt * a1[i];
    }

    // sub-step 3  (c3=c2, d3=d1)
    for i in 0..d {
        q[i] += c2 * dt * v[i];
    }
    let a2 = accel(q);
    check_accel_len(&a2, d)?;
    for i in 0..d {
        v[i] += d1 * dt * a2[i];
    }

    // final drift  (c4=c1)
    for i in 0..d {
        q[i] += c1 * dt * v[i];
    }

    Ok(())
}

/// Multiple Forest-Ruth steps.
///
/// Equivalent to calling [`forest_ruth_step`] `n_steps` times in sequence.
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] if `n_steps == 0`, or any error
/// from an individual step.
pub fn forest_ruth<F>(
    q: &mut [f64],
    v: &mut [f64],
    dt: f64,
    n_steps: usize,
    accel: F,
) -> PdeResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_qv_dt(q, v, dt)?;
    if n_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_steps".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    let d = q.len();
    let (c1, c2, d1, d2) = forest_ruth_coefficients();

    for _ in 0..n_steps {
        // sub-step 1
        for i in 0..d {
            q[i] += c1 * dt * v[i];
        }
        let a0 = accel(q);
        check_accel_len(&a0, d)?;
        for i in 0..d {
            v[i] += d1 * dt * a0[i];
        }

        // sub-step 2
        for i in 0..d {
            q[i] += c2 * dt * v[i];
        }
        let a1 = accel(q);
        check_accel_len(&a1, d)?;
        for i in 0..d {
            v[i] += d2 * dt * a1[i];
        }

        // sub-step 3  (c3=c2, d3=d1)
        for i in 0..d {
            q[i] += c2 * dt * v[i];
        }
        let a2 = accel(q);
        check_accel_len(&a2, d)?;
        for i in 0..d {
            v[i] += d1 * dt * a2[i];
        }

        // final drift  (c4=c1)
        for i in 0..d {
            q[i] += c1 * dt * v[i];
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Velocity Verlet error cases ──────────────────────────────────────────

    #[test]
    fn vv_err_dt_zero() {
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let result = velocity_verlet_step(&mut q, &mut v, 0.0, |x| vec![-x[0]]);
        assert!(result.is_err());
    }

    #[test]
    fn vv_err_dt_neg() {
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let result = velocity_verlet_step(&mut q, &mut v, -1.0, |x| vec![-x[0]]);
        assert!(result.is_err());
    }

    #[test]
    fn vv_err_dim_mismatch() {
        let mut q = vec![1.0, 0.0];
        let mut v = vec![0.0, 0.0, 0.0];
        let result = velocity_verlet_step(&mut q, &mut v, 0.01, |_| vec![0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn vv_err_empty() {
        let mut q: Vec<f64> = vec![];
        let mut v: Vec<f64> = vec![];
        let result = velocity_verlet_step(&mut q, &mut v, 0.01, |_| vec![]);
        assert!(result.is_err());
    }

    // ── Velocity Verlet correctness ──────────────────────────────────────────

    #[test]
    fn vv_harmonic_oscillator_energy() {
        // H = 0.5*(q^2 + v^2);  a(q) = -q
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let h0 = 0.5 * (q[0] * q[0] + v[0] * v[0]);
        let dt = 0.01;
        velocity_verlet(&mut q, &mut v, dt, 1000, |x| vec![-x[0]])
            .expect("velocity_verlet should succeed");
        let h_final = 0.5 * (q[0] * q[0] + v[0] * v[0]);
        assert!(
            (h_final - h0).abs() < 0.01,
            "energy drift too large: |H_final - H0| = {}",
            (h_final - h0).abs()
        );
    }

    #[test]
    fn vv_harmonic_exact() {
        // q(0)=1, v(0)=0, ω=1 → period = 2π  ⟹  q(2π) ≈ 1.0
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let dt = 2.0 * PI / 1000.0;
        velocity_verlet(&mut q, &mut v, dt, 1000, |x| vec![-x[0]])
            .expect("velocity_verlet should succeed");
        assert!(
            (q[0] - 1.0).abs() < 1.0e-4,
            "q after one period = {}, expected ≈ 1.0",
            q[0]
        );
    }

    #[test]
    fn vv_free_particle() {
        // a = 0 → q grows linearly
        let q0 = 2.0;
        let v0 = 3.0;
        let mut q = vec![q0];
        let mut v = vec![v0];
        let dt = 0.1;
        let n = 50usize;
        velocity_verlet(&mut q, &mut v, dt, n, |_| vec![0.0])
            .expect("velocity_verlet should succeed");
        let t_final = dt * n as f64;
        let expected = q0 + v0 * t_final;
        assert!(
            (q[0] - expected).abs() < 1.0e-12,
            "free-particle: q = {}, expected = {}",
            q[0],
            expected
        );
    }

    #[test]
    fn vv_multi_step_same_as_loop() {
        // velocity_verlet(..., n, ..) must equal n sequential velocity_verlet_step calls
        let q_init = [1.0f64, 0.5];
        let v_init = [0.0f64, 1.0];
        let dt = 0.05;
        let n = 20usize;
        let accel = |x: &[f64]| vec![-x[0], -x[1]];

        let mut q1 = q_init.to_vec();
        let mut v1 = v_init.to_vec();
        velocity_verlet(&mut q1, &mut v1, dt, n, accel).expect("ok");

        let mut q2 = q_init.to_vec();
        let mut v2 = v_init.to_vec();
        for _ in 0..n {
            velocity_verlet_step(&mut q2, &mut v2, dt, accel).expect("ok");
        }

        for i in 0..2 {
            assert!((q1[i] - q2[i]).abs() < 1.0e-14, "q mismatch at index {i}");
            assert!((v1[i] - v2[i]).abs() < 1.0e-14, "v mismatch at index {i}");
        }
    }

    #[test]
    fn vv_1d_vs_2d() {
        // Both 1-D and 2-D should succeed without error.
        let mut q1 = vec![1.0];
        let mut v1 = vec![0.0];
        velocity_verlet_step(&mut q1, &mut v1, 0.01, |x| vec![-x[0]]).expect("1-D ok");

        let mut q2 = vec![1.0, 0.0];
        let mut v2 = vec![0.0, 1.0];
        velocity_verlet_step(&mut q2, &mut v2, 0.01, |x| vec![-x[0], -x[1]]).expect("2-D ok");
    }

    // ── Forest-Ruth error cases ──────────────────────────────────────────────

    #[test]
    fn fr_err_dt_zero() {
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let result = forest_ruth_step(&mut q, &mut v, 0.0, |x| vec![-x[0]]);
        assert!(result.is_err());
    }

    #[test]
    fn fr_err_dim_mismatch() {
        let mut q = vec![1.0, 0.0];
        let mut v = vec![0.0, 0.0, 0.0];
        let result = forest_ruth_step(&mut q, &mut v, 0.01, |_| vec![0.0, 0.0]);
        assert!(result.is_err());
    }

    // ── Forest-Ruth correctness ──────────────────────────────────────────────

    #[test]
    fn fr_harmonic_oscillator_energy() {
        // Larger dt=0.1 compared to VV; FR has better energy conservation.
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        let h0 = 0.5 * (q[0] * q[0] + v[0] * v[0]);
        let dt = 0.1;
        let n = 200usize; // simulate for 20 time units
        forest_ruth(&mut q, &mut v, dt, n, |x| vec![-x[0]]).expect("ok");
        let h_final = 0.5 * (q[0] * q[0] + v[0] * v[0]);
        // FR is 4th order so energy drift is small over this interval
        assert!(
            (h_final - h0).abs() < 1.0e-5,
            "|H_final - H0| = {}",
            (h_final - h0).abs()
        );
    }

    #[test]
    fn fr_harmonic_exact_large_dt() {
        // q(0)=1, v(0)=0, ω=1; integrate for exactly 1000 steps of dt=2π/1000.
        // This guarantees the total time is exactly 2π (one full period).
        // FR 4th-order accuracy means |q-1| should be very small.
        let two_pi = 2.0 * PI;
        let n = 1000usize;
        let dt = two_pi / n as f64; // ≈ 0.006283
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        forest_ruth(&mut q, &mut v, dt, n, |x| vec![-x[0]]).expect("ok");
        assert!(
            (q[0] - 1.0).abs() < 1.0e-10,
            "FR after one period (exact): q = {}, |q-1| = {}",
            q[0],
            (q[0] - 1.0).abs()
        );
    }

    #[test]
    fn fr_free_particle() {
        // a = 0 → q grows linearly, same as VV
        let q0 = 2.0;
        let v0 = 3.0;
        let mut q = vec![q0];
        let mut v = vec![v0];
        let dt = 0.1;
        let n = 50usize;
        forest_ruth(&mut q, &mut v, dt, n, |_| vec![0.0]).expect("ok");
        let expected = q0 + v0 * dt * n as f64;
        assert!(
            (q[0] - expected).abs() < 1.0e-12,
            "free-particle FR: q = {}, expected = {}",
            q[0],
            expected
        );
    }

    #[test]
    fn fr_4th_order() {
        // FR energy deviation after 1 period should be less than VV with same step size.
        let dt = 0.3;
        let two_pi = 2.0 * PI;
        let n = (two_pi / dt).round() as usize;

        let mut q_vv = vec![1.0];
        let mut v_vv = vec![0.0];
        let h0 = 0.5_f64;
        velocity_verlet(&mut q_vv, &mut v_vv, dt, n, |x| vec![-x[0]]).expect("ok");
        let err_vv = (0.5 * (q_vv[0] * q_vv[0] + v_vv[0] * v_vv[0]) - h0).abs();

        let mut q_fr = vec![1.0];
        let mut v_fr = vec![0.0];
        forest_ruth(&mut q_fr, &mut v_fr, dt, n, |x| vec![-x[0]]).expect("ok");
        let err_fr = (0.5 * (q_fr[0] * q_fr[0] + v_fr[0] * v_fr[0]) - h0).abs();

        assert!(
            err_fr < err_vv,
            "FR energy error ({err_fr}) should be less than VV ({err_vv})"
        );
    }

    #[test]
    fn fr_multi_step_ok() {
        let mut q = vec![1.0];
        let mut v = vec![0.0];
        forest_ruth(&mut q, &mut v, 0.01, 100, |x| vec![-x[0]]).expect("100 steps ok");
    }

    #[test]
    fn fr_coefficients_sum() {
        // c1 + c2 + c2 + c1 == 1.0  (total drift)
        // d1 + d2 + d1 == 1.0       (total kick)
        let (c1, c2, d1, d2) = forest_ruth_coefficients();
        let drift_sum = c1 + c2 + c2 + c1;
        let kick_sum = d1 + d2 + d1;
        assert!(
            (drift_sum - 1.0).abs() < 1.0e-14,
            "drift sum = {drift_sum}, expected 1.0"
        );
        assert!(
            (kick_sum - 1.0).abs() < 1.0e-14,
            "kick sum = {kick_sum}, expected 1.0"
        );
    }

    #[test]
    fn fr_vs_vv_accuracy() {
        // For harmonic oscillator with dt=0.5, compare energy error after 10 steps.
        let dt = 0.5;
        let h0 = 0.5_f64;

        let mut q_vv = vec![1.0];
        let mut v_vv = vec![0.0];
        velocity_verlet(&mut q_vv, &mut v_vv, dt, 10, |x| vec![-x[0]]).expect("ok");
        let err_vv = (0.5 * (q_vv[0] * q_vv[0] + v_vv[0] * v_vv[0]) - h0).abs();

        let mut q_fr = vec![1.0];
        let mut v_fr = vec![0.0];
        forest_ruth(&mut q_fr, &mut v_fr, dt, 10, |x| vec![-x[0]]).expect("ok");
        let err_fr = (0.5 * (q_fr[0] * q_fr[0] + v_fr[0] * v_fr[0]) - h0).abs();

        assert!(
            err_fr < err_vv,
            "FR energy error ({err_fr}) should be less than VV ({err_vv}) for dt=0.5"
        );
    }
}
