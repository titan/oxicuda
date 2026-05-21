//! Exponential integrators for stiff ODEs of the form `du/dt = L·u + N(t, u)`
//! where `L` is a diagonal linear operator.
//!
//! Three methods are provided:
//!
//! - **Lawson-Euler** — first-order, based on the variation-of-constants formula
//!   with a single forward-Euler stage for the nonlinear part.
//! - **Lawson-RK4** (Lawson 1967) — fourth-order, classical RK4 in the
//!   "interaction picture" (the state is transported along the linear flow before
//!   each stage evaluation).
//! - **ETD-RK4** (Cox–Matthews 2002) — fourth-order exponential time-differencing
//!   RK4 using φ-functions for exact treatment of the linear part.
//!
//! All methods operate componentwise on the diagonal of `L`; no matrix
//! factorisation is needed.

use crate::error::{PdeError, PdeResult};

// ── φ-functions ───────────────────────────────────────────────────────────────

/// φ₁(z) = (e^z − 1) / z, with the L'Hôpital limit φ₁(0) = 1.
#[inline]
fn phi1(z: f64) -> f64 {
    if z.abs() < 1.0e-14 {
        1.0
    } else {
        (z.exp() - 1.0) / z
    }
}

/// φ₂(z) = (e^z − 1 − z) / z², with the limit φ₂(0) = 1/2.
#[inline]
fn phi2(z: f64) -> f64 {
    if z.abs() < 1.0e-14 {
        0.5
    } else {
        (z.exp() - 1.0 - z) / (z * z)
    }
}

/// φ₃(z) = (e^z − 1 − z − z²/2) / z³, with the limit φ₃(0) = 1/6.
#[inline]
fn phi3(z: f64) -> f64 {
    if z.abs() < 1.0e-14 {
        1.0 / 6.0
    } else {
        (z.exp() - 1.0 - z - 0.5 * z * z) / (z * z * z)
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Shared validation: u and l_diag must have the same length, dt must be > 0.
fn validate_inputs(u: &[f64], l_diag: &[f64], dt: f64) -> PdeResult<()> {
    if u.len() != l_diag.len() {
        return Err(PdeError::DimensionMismatch {
            a: u.len(),
            b: l_diag.len(),
        });
    }
    if dt <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "dt".to_string(),
            reason: "must be strictly positive".to_string(),
        });
    }
    Ok(())
}

/// Validate also that n_steps ≥ 1.
fn validate_integrate(u: &[f64], l_diag: &[f64], dt: f64, n_steps: usize) -> PdeResult<()> {
    validate_inputs(u, l_diag, dt)?;
    if n_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_steps".to_string(),
            reason: "must be at least 1".to_string(),
        });
    }
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the elementwise exponential `exp(l[i] * dt)` for each diagonal entry.
pub fn exp_diag(l_diag: &[f64], dt: f64) -> Vec<f64> {
    l_diag.iter().map(|&li| (li * dt).exp()).collect()
}

/// Perform one Lawson-Euler step.
///
/// `u_{n+1}[i] = exp(l[i]·dt)·u[i] + dt·exp(l[i]·dt)·N[i]`
///
/// where `n_fn` is the nonlinear term `N(t, u)` already evaluated at the
/// current state.
///
/// # Errors
/// Returns `Err` if `u.len() != l_diag.len()` or `dt <= 0`.
pub fn lawson_euler_step(u: &[f64], l_diag: &[f64], n_fn: &[f64], dt: f64) -> PdeResult<Vec<f64>> {
    validate_inputs(u, l_diag, dt)?;
    if n_fn.len() != u.len() {
        return Err(PdeError::DimensionMismatch {
            a: n_fn.len(),
            b: u.len(),
        });
    }
    let n = u.len();
    let mut u_new = vec![0.0_f64; n];
    for i in 0..n {
        let e = (l_diag[i] * dt).exp();
        u_new[i] = e * u[i] + dt * e * n_fn[i];
    }
    Ok(u_new)
}

/// Perform one Lawson-RK4 step (Lawson 1967 interaction-picture RK4).
///
/// For each component `i` with `E = exp(l·dt)`, `E2 = exp(l·dt/2)`:
///
/// ```text
/// k1 = N(t, u)
/// u2[i] = E2·u[i] + (dt/2)·E2·k1[i]
/// k2 = N(t+dt/2, u2)
/// u3[i] = E2·u[i] + (dt/2)·k2[i]
/// k3 = N(t+dt/2, u3)
/// u4[i] = E·u[i] + dt·E2·k3[i]
/// k4 = N(t+dt, u4)
/// u_new[i] = E·u[i] + (dt/6)·(E·k1[i] + 2·E2·k2[i] + 2·E2·k3[i] + k4[i])
/// ```
///
/// # Errors
/// Returns `Err` if `u.len() != l_diag.len()` or `dt <= 0`.
pub fn lawson_rk4_step<F>(
    u: &[f64],
    l_diag: &[f64],
    nonlinear: F,
    t: f64,
    dt: f64,
) -> PdeResult<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_inputs(u, l_diag, dt)?;
    let n = u.len();
    let half = dt * 0.5;

    // Precompute E and E2 per component
    let mut e = vec![0.0_f64; n];
    let mut e2 = vec![0.0_f64; n];
    for i in 0..n {
        e[i] = (l_diag[i] * dt).exp();
        e2[i] = (l_diag[i] * half).exp();
    }

    // Stage 1
    let k1 = nonlinear(t, u);

    // Stage 2
    let mut u2 = vec![0.0_f64; n];
    for i in 0..n {
        u2[i] = e2[i] * u[i] + half * e2[i] * k1[i];
    }
    let k2 = nonlinear(t + half, &u2);

    // Stage 3 (note: u3 uses E2·u[i], not E2·u2[i])
    let mut u3 = vec![0.0_f64; n];
    for i in 0..n {
        u3[i] = e2[i] * u[i] + half * k2[i];
    }
    let k3 = nonlinear(t + half, &u3);

    // Stage 4
    let mut u4 = vec![0.0_f64; n];
    for i in 0..n {
        u4[i] = e[i] * u[i] + dt * e2[i] * k3[i];
    }
    let k4 = nonlinear(t + dt, &u4);

    // Combine
    let mut u_new = vec![0.0_f64; n];
    for i in 0..n {
        u_new[i] = e[i] * u[i]
            + (dt / 6.0) * (e[i] * k1[i] + 2.0 * e2[i] * k2[i] + 2.0 * e2[i] * k3[i] + k4[i]);
    }
    Ok(u_new)
}

/// Perform one ETD-RK4 step (Cox–Matthews 2002).
///
/// Uses φ-functions for exact treatment of the linear part:
///
/// ```text
/// z = l[i]·dt,  E = exp(z),  E2 = exp(z/2)
/// p1h = φ₁(z/2)·(dt/2)
///
/// a[i] = E2·u[i] + p1h·N0[i]
/// b[i] = E2·u[i] + p1h·Na[i]
/// c[i] = E2·a[i] + p1h·(2·Nb[i] − N0[i])
///
/// u_new[i] = E·u[i] + dt·(
///     (φ₁(z) − 3φ₂(z) + 4φ₃(z))·N0[i]
///   + 2(φ₂(z) − 2φ₃(z))·(Na[i] + Nb[i])
///   + (−φ₂(z) + 4φ₃(z))·Nc[i]
/// )
/// ```
///
/// where `N0=N(t,u)`, `Na=N(t+dt/2, a)`, `Nb=N(t+dt/2, b)`, `Nc=N(t+dt, c)`.
///
/// For `L=0` this reduces to the standard RK4 formula.
///
/// # Errors
/// Returns `Err` if `u.len() != l_diag.len()` or `dt <= 0`.
pub fn etd_rk4_step<F>(
    u: &[f64],
    l_diag: &[f64],
    nonlinear: F,
    t: f64,
    dt: f64,
) -> PdeResult<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_inputs(u, l_diag, dt)?;
    let n = u.len();
    let half = dt * 0.5;

    let n0 = nonlinear(t, u);

    // Precompute per-component exponentials and phi coefficients
    let mut e_full = vec![0.0_f64; n];
    let mut e_half = vec![0.0_f64; n];
    let mut p1h = vec![0.0_f64; n];
    let mut c1 = vec![0.0_f64; n]; // φ₁(z) − 3φ₂(z) + 4φ₃(z)
    let mut c2 = vec![0.0_f64; n]; // 2(φ₂(z) − 2φ₃(z))
    let mut c3 = vec![0.0_f64; n]; // −φ₂(z) + 4φ₃(z)
    for i in 0..n {
        let z = l_diag[i] * dt;
        let zh = z * 0.5;
        e_full[i] = z.exp();
        e_half[i] = zh.exp();
        p1h[i] = phi1(zh) * half;
        c1[i] = phi1(z) - 3.0 * phi2(z) + 4.0 * phi3(z);
        c2[i] = 2.0 * (phi2(z) - 2.0 * phi3(z));
        c3[i] = -phi2(z) + 4.0 * phi3(z);
    }

    // a = E2·u + p1h·N0
    let mut a = vec![0.0_f64; n];
    for i in 0..n {
        a[i] = e_half[i] * u[i] + p1h[i] * n0[i];
    }
    let na = nonlinear(t + half, &a);

    // b = E2·u + p1h·Na
    let mut b = vec![0.0_f64; n];
    for i in 0..n {
        b[i] = e_half[i] * u[i] + p1h[i] * na[i];
    }
    let nb = nonlinear(t + half, &b);

    // c = E2·a + p1h·(2·Nb − N0)
    let mut c = vec![0.0_f64; n];
    for i in 0..n {
        c[i] = e_half[i] * a[i] + p1h[i] * (2.0 * nb[i] - n0[i]);
    }
    let nc = nonlinear(t + dt, &c);

    // Combine with ETD-RK4 coefficients
    let mut u_new = vec![0.0_f64; n];
    for i in 0..n {
        u_new[i] =
            e_full[i] * u[i] + dt * (c1[i] * n0[i] + c2[i] * (na[i] + nb[i]) + c3[i] * nc[i]);
    }
    Ok(u_new)
}

// ── Integration wrappers ──────────────────────────────────────────────────────

/// Integrate `n_steps` steps with the Lawson-Euler method.
///
/// Returns a `Vec` of `n_steps + 1` state vectors: `[u0, u1, …, u_{n_steps}]`.
///
/// # Errors
/// Returns `Err` if `u0.len() != l_diag.len()`, `dt <= 0`, or `n_steps < 1`.
pub fn lawson_euler_integrate<F>(
    u0: &[f64],
    l_diag: &[f64],
    nonlinear: F,
    t0: f64,
    dt: f64,
    n_steps: usize,
) -> PdeResult<Vec<Vec<f64>>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_integrate(u0, l_diag, dt, n_steps)?;
    let mut states = Vec::with_capacity(n_steps + 1);
    let mut u = u0.to_vec();
    states.push(u.clone());
    for k in 0..n_steps {
        let t = t0 + k as f64 * dt;
        let n_fn = nonlinear(t, &u);
        u = lawson_euler_step(&u, l_diag, &n_fn, dt)?;
        states.push(u.clone());
    }
    Ok(states)
}

/// Integrate `n_steps` steps with Lawson-RK4.
///
/// Returns a `Vec` of `n_steps + 1` state vectors.
///
/// # Errors
/// Returns `Err` if `u0.len() != l_diag.len()`, `dt <= 0`, or `n_steps < 1`.
pub fn lawson_rk4_integrate<F>(
    u0: &[f64],
    l_diag: &[f64],
    nonlinear: F,
    t0: f64,
    dt: f64,
    n_steps: usize,
) -> PdeResult<Vec<Vec<f64>>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_integrate(u0, l_diag, dt, n_steps)?;
    let mut states = Vec::with_capacity(n_steps + 1);
    let mut u = u0.to_vec();
    states.push(u.clone());
    for k in 0..n_steps {
        let t = t0 + k as f64 * dt;
        u = lawson_rk4_step(&u, l_diag, &nonlinear, t, dt)?;
        states.push(u.clone());
    }
    Ok(states)
}

/// Integrate `n_steps` steps with ETD-RK4.
///
/// Returns a `Vec` of `n_steps + 1` state vectors.
///
/// # Errors
/// Returns `Err` if `u0.len() != l_diag.len()`, `dt <= 0`, or `n_steps < 1`.
pub fn etd_rk4_integrate<F>(
    u0: &[f64],
    l_diag: &[f64],
    nonlinear: F,
    t0: f64,
    dt: f64,
    n_steps: usize,
) -> PdeResult<Vec<Vec<f64>>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_integrate(u0, l_diag, dt, n_steps)?;
    let mut states = Vec::with_capacity(n_steps + 1);
    let mut u = u0.to_vec();
    states.push(u.clone());
    for k in 0..n_steps {
        let t = t0 + k as f64 * dt;
        u = etd_rk4_step(&u, l_diag, &nonlinear, t, dt)?;
        states.push(u.clone());
    }
    Ok(states)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-12;

    // ── φ-functions at zero ────────────────────────────────────────────────

    #[test]
    fn phi1_at_zero() {
        assert!((phi1(0.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn phi1_at_one() {
        // φ₁(1) = e − 1
        let expected = std::f64::consts::E - 1.0;
        assert!((phi1(1.0) - expected).abs() < TOL, "φ₁(1)={}", phi1(1.0));
    }

    #[test]
    fn phi2_at_zero() {
        assert!((phi2(0.0) - 0.5).abs() < TOL);
    }

    #[test]
    fn phi3_at_zero() {
        assert!((phi3(0.0) - 1.0 / 6.0).abs() < TOL);
    }

    // ── exp_diag ───────────────────────────────────────────────────────────

    #[test]
    fn exp_diag_shape() {
        let l = vec![-1.0, -2.0, -3.0];
        let e = exp_diag(&l, 0.1);
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn exp_diag_zeros_give_ones() {
        let l = vec![0.0; 5];
        let e = exp_diag(&l, 1.0);
        for &v in &e {
            assert!((v - 1.0).abs() < TOL);
        }
    }

    // ── Lawson-Euler step ──────────────────────────────────────────────────

    #[test]
    fn lawson_euler_step_shape() {
        let u = vec![1.0, 2.0, 3.0];
        let l = vec![-1.0, -1.0, -1.0];
        let n_fn = vec![0.0, 0.0, 0.0];
        let u_new = lawson_euler_step(&u, &l, &n_fn, 0.1).expect("ok");
        assert_eq!(u_new.len(), 3);
    }

    #[test]
    fn lawson_euler_step_n_zero_exact() {
        // With N=0: u_{n+1}[i] = exp(l·dt)·u[i]
        let u = vec![2.0, -1.0];
        let l = vec![-0.5, -2.0];
        let n_fn = vec![0.0, 0.0];
        let dt = 0.3;
        let u_new = lawson_euler_step(&u, &l, &n_fn, dt).expect("ok");
        for i in 0..2 {
            let expected = (l[i] * dt).exp() * u[i];
            assert!((u_new[i] - expected).abs() < TOL);
        }
    }

    // ── Lawson-RK4 step ────────────────────────────────────────────────────

    #[test]
    fn lawson_rk4_step_n_zero_exact() {
        // With N=0: u_{n+1}[i] = exp(l·dt)·u[i] exactly
        let u = vec![1.5, 3.0];
        let l = vec![-1.0, -2.0];
        let dt = 0.2;
        let u_new = lawson_rk4_step(&u, &l, |_, _| vec![0.0, 0.0], 0.0, dt).expect("ok");
        for i in 0..2 {
            let expected = (l[i] * dt).exp() * u[i];
            assert!(
                (u_new[i] - expected).abs() < 1.0e-14,
                "component {i}: {} != {}",
                u_new[i],
                expected
            );
        }
    }

    #[test]
    fn lawson_rk4_l_zero_reduces_to_rk4() {
        // With L=0, Lawson-RK4 should agree with classical RK4 for du/dt = N(u).
        // Test: du/dt = -u, u(0)=1 → u(1)=exp(-1)
        let u0 = vec![1.0];
        let l = vec![0.0];
        let dt = 0.01;
        let n_steps = 100;
        let states =
            lawson_rk4_integrate(&u0, &l, |_, x| vec![-x[0]], 0.0, dt, n_steps).expect("ok");
        let u_final = &states[n_steps];
        let expected = (-1.0_f64).exp();
        assert!(
            (u_final[0] - expected).abs() < 1.0e-9,
            "u={} expected={}",
            u_final[0],
            expected
        );
    }

    // ── ETD-RK4 step ───────────────────────────────────────────────────────

    #[test]
    fn etd_rk4_step_n_zero_exact() {
        // With N=0: u_{n+1}[i] = exp(l·dt)·u[i]  (ETD-RK4 is exact for linear problems)
        let u = vec![2.0, -0.5];
        let l = vec![-3.0, -0.1];
        let dt = 0.05;
        let u_new = etd_rk4_step(&u, &l, |_, _| vec![0.0, 0.0], 0.0, dt).expect("ok");
        for i in 0..2 {
            let expected = (l[i] * dt).exp() * u[i];
            assert!(
                (u_new[i] - expected).abs() < 1.0e-13,
                "component {i}: {} != {}",
                u_new[i],
                expected
            );
        }
    }

    #[test]
    fn etd_rk4_l_zero_reduces_to_rk4() {
        // With L=0, ETD-RK4 must recover the standard RK4 formula.
        // phi1→1, phi2→1/2, phi3→1/6 → weights: 1/6, 2·(1/2−2/6)=1/3, 1/3, (−1/2+4/6)=1/6.
        let u0 = vec![1.0];
        let l = vec![0.0];
        let dt = 0.01;
        let n_steps = 100;
        let states = etd_rk4_integrate(&u0, &l, |_, x| vec![-x[0]], 0.0, dt, n_steps).expect("ok");
        let u_final = &states[n_steps];
        let expected = (-1.0_f64).exp();
        assert!(
            (u_final[0] - expected).abs() < 1.0e-9,
            "u={} expected={}",
            u_final[0],
            expected
        );
    }

    // ── Exponential decay accuracy ─────────────────────────────────────────

    #[test]
    fn lawson_euler_exponential_decay_ten_steps() {
        // du/dt = -u + 0, exact solution u(t)=e^{-t}.
        // With Lawson-Euler and N=0, each step gives u_{n+1}=exp(-dt)·u_n, so
        // after k steps u_k = exp(-k·dt)·u_0, which is exact.
        let u0 = vec![1.0];
        let l = vec![-1.0];
        let dt = 0.1;
        let n_steps = 10;
        let states =
            lawson_euler_integrate(&u0, &l, |_, _| vec![0.0], 0.0, dt, n_steps).expect("ok");
        let u_final = &states[n_steps];
        let t_final = dt * n_steps as f64;
        let expected = (-t_final).exp();
        assert!(
            (u_final[0] - expected).abs() < 1.0e-14,
            "Lawson-Euler (N=0): u={} expected={}",
            u_final[0],
            expected
        );
    }

    #[test]
    fn lawson_rk4_exponential_decay_small_dt() {
        // du/dt = -u, u(0)=1. Lawson-RK4 should achieve < 1e-10 error at t=1.
        let u0 = vec![1.0];
        let l = vec![-1.0];
        let dt = 0.001;
        let n_steps = 1000;
        let states = lawson_rk4_integrate(&u0, &l, |_, _| vec![0.0], 0.0, dt, n_steps).expect("ok");
        let u_final = &states[n_steps];
        let expected = (-1.0_f64).exp();
        assert!(
            (u_final[0] - expected).abs() < 1.0e-10,
            "Lawson-RK4: u={} expected={}",
            u_final[0],
            expected
        );
    }

    #[test]
    fn etd_rk4_exponential_decay_small_dt() {
        // du/dt = -u, u(0)=1. ETD-RK4 should achieve < 1e-10 error at t=1.
        let u0 = vec![1.0];
        let l = vec![-1.0];
        let dt = 0.001;
        let n_steps = 1000;
        let states = etd_rk4_integrate(&u0, &l, |_, _| vec![0.0], 0.0, dt, n_steps).expect("ok");
        let u_final = &states[n_steps];
        let expected = (-1.0_f64).exp();
        assert!(
            (u_final[0] - expected).abs() < 1.0e-10,
            "ETD-RK4: u={} expected={}",
            u_final[0],
            expected
        );
    }

    // ── Integration: output length ────────────────────────────────────────

    #[test]
    fn integrate_output_length() {
        let u0 = vec![1.0, 2.0];
        let l = vec![-1.0, -1.0];
        let n_steps = 5;

        let le =
            lawson_euler_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");
        let lrk =
            lawson_rk4_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");
        let etd = etd_rk4_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");

        assert_eq!(le.len(), n_steps + 1);
        assert_eq!(lrk.len(), n_steps + 1);
        assert_eq!(etd.len(), n_steps + 1);
    }

    #[test]
    fn integrate_constant_solution() {
        // du/dt = 0·u + 0, u stays constant.
        let u0 = vec![3.0, -2.0];
        let l = vec![0.0, 0.0];
        let n_steps = 10;

        let le =
            lawson_euler_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");
        for state in &le {
            assert!((state[0] - 3.0).abs() < 1.0e-14);
            assert!((state[1] - (-2.0)).abs() < 1.0e-14);
        }

        let lrk =
            lawson_rk4_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");
        for state in &lrk {
            assert!((state[0] - 3.0).abs() < 1.0e-14);
            assert!((state[1] - (-2.0)).abs() < 1.0e-14);
        }

        let etd = etd_rk4_integrate(&u0, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1, n_steps).expect("ok");
        for state in &etd {
            assert!((state[0] - 3.0).abs() < 1.0e-14);
            assert!((state[1] - (-2.0)).abs() < 1.0e-14);
        }
    }

    // ── Error conditions ──────────────────────────────────────────────────

    #[test]
    fn dt_nonpositive_returns_error() {
        let u = vec![1.0];
        let l = vec![-1.0];
        let n_fn = vec![0.0];

        assert!(lawson_euler_step(&u, &l, &n_fn, 0.0).is_err());
        assert!(lawson_euler_step(&u, &l, &n_fn, -0.1).is_err());
        assert!(lawson_rk4_step(&u, &l, |_, _| vec![0.0], 0.0, 0.0).is_err());
        assert!(etd_rk4_step(&u, &l, |_, _| vec![0.0], 0.0, -1.0).is_err());
    }

    #[test]
    fn dim_mismatch_l_vs_u_returns_error() {
        let u = vec![1.0, 2.0];
        let l = vec![-1.0]; // wrong length
        let n_fn = vec![0.0, 0.0];
        assert!(lawson_euler_step(&u, &l, &n_fn, 0.1).is_err());
        assert!(lawson_rk4_step(&u, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1).is_err());
        assert!(etd_rk4_step(&u, &l, |_, _| vec![0.0, 0.0], 0.0, 0.1).is_err());
    }
}
