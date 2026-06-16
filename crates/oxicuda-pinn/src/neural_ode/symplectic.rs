//! Symplectic integrators for Hamiltonian systems.
//!
//! For a **separable** Hamiltonian `H(q, p) = T(p) + V(q)` (kinetic + potential),
//! Hamilton's equations are
//!
//! ```text
//! dq/dt =  ∂H/∂p =  ∇T(p)
//! dp/dt = −∂H/∂q = −∇V(q) =: F(q)   (the force).
//! ```
//!
//! With unit masses (`T(p) = ½|p|²`, so `∇T(p) = p`) this is the classical
//! second-order system `q̈ = F(q)`. Unlike generic Runge-Kutta methods,
//! **symplectic** integrators preserve the canonical two-form `dq ∧ dp`, so the
//! discrete flow conserves a *shadow* Hamiltonian and the energy error stays
//! bounded for exponentially long times rather than drifting secularly. This makes
//! them the integrators of choice for Hamiltonian / Lagrangian Neural Networks and
//! long-horizon molecular / orbital dynamics.
//!
//! Implemented here, all expressed through a user-supplied force `F(q)`:
//!
//! - [`leapfrog_step`] / [`velocity_verlet_step`] — the kick-drift-kick form,
//!   2nd-order accurate and time-reversible.
//! - [`stormer_verlet_step`] — the position (drift-kick-drift) variant.
//! - [`symplectic_euler_step`] — 1st-order semi-implicit Euler (`p` then `q`).
//! - [`integrate_symplectic`] — fixed-step trajectory rollout with any method.
//!
//! These complement the dissipative RK solvers in [`crate::neural_ode::solvers`];
//! use a symplectic method whenever the dynamics are conservative.

use crate::error::{PinnError, PinnResult};

/// Force-field signature `F(q) = −∇V(q)`: writes the force into `force`.
pub type ForceFn<'a> = &'a dyn Fn(&[f32], &mut [f32]);

/// Choice of symplectic integration scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymplecticMethod {
    /// 1st-order semi-implicit (symplectic) Euler: update `p` then `q`.
    SymplecticEuler,
    /// 2nd-order velocity Verlet (kick-drift-kick); identical to leapfrog
    /// when positions and momenta are reported at the same time level.
    VelocityVerlet,
    /// 2nd-order Störmer-Verlet (drift-kick-drift) position form.
    StormerVerlet,
}

/// One **velocity Verlet** (kick-drift-kick) step for unit-mass dynamics
/// `q̈ = F(q)`.
///
/// ```text
/// p_{1/2} = p + (h/2) F(q)
/// q'      = q + h p_{1/2}
/// p'      = p_{1/2} + (h/2) F(q')
/// ```
///
/// Returns `(q', p')`. Time-reversible and symplectic (2nd order).
///
/// # Errors
/// - [`PinnError::DimensionMismatch`] if `q.len() != p.len()`.
/// - [`PinnError::InvalidStepSize`] if `h` is non-finite.
/// - [`PinnError::NanEncountered`] if the force or state becomes non-finite.
pub fn velocity_verlet_step(
    force: ForceFn,
    q: &[f32],
    p: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    check_state(q, p, h)?;
    let dim = q.len();
    let mut f0 = vec![0.0_f32; dim];
    force(q, &mut f0);
    finite(&f0, "velocity_verlet_step::F(q)")?;

    // Half kick.
    let p_half: Vec<f32> = p
        .iter()
        .zip(f0.iter())
        .map(|(&pi, &fi)| pi + 0.5 * h * fi)
        .collect();
    // Drift.
    let q_new: Vec<f32> = q
        .iter()
        .zip(p_half.iter())
        .map(|(&qi, &phi)| qi + h * phi)
        .collect();

    let mut f1 = vec![0.0_f32; dim];
    force(&q_new, &mut f1);
    finite(&f1, "velocity_verlet_step::F(q')")?;

    // Second half kick.
    let p_new: Vec<f32> = p_half
        .iter()
        .zip(f1.iter())
        .map(|(&phi, &fi)| phi + 0.5 * h * fi)
        .collect();

    finite(&q_new, "velocity_verlet_step::q")?;
    finite(&p_new, "velocity_verlet_step::p")?;
    Ok((q_new, p_new))
}

/// Alias for the kick-drift-kick **leapfrog** step (synchronised form), which
/// equals [`velocity_verlet_step`].
///
/// # Errors
/// See [`velocity_verlet_step`].
#[inline]
pub fn leapfrog_step(
    force: ForceFn,
    q: &[f32],
    p: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    velocity_verlet_step(force, q, p, h)
}

/// One **Störmer-Verlet** (drift-kick-drift) step.
///
/// ```text
/// q_{1/2} = q + (h/2) p
/// p'      = p + h F(q_{1/2})
/// q'      = q_{1/2} + (h/2) p'
/// ```
///
/// Returns `(q', p')`. Symplectic and 2nd order; the positional dual of
/// velocity Verlet.
///
/// # Errors
/// See [`velocity_verlet_step`].
pub fn stormer_verlet_step(
    force: ForceFn,
    q: &[f32],
    p: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    check_state(q, p, h)?;
    let dim = q.len();
    // Half drift.
    let q_half: Vec<f32> = q
        .iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| qi + 0.5 * h * pi)
        .collect();
    let mut f = vec![0.0_f32; dim];
    force(&q_half, &mut f);
    finite(&f, "stormer_verlet_step::F(q_half)")?;
    // Full kick.
    let p_new: Vec<f32> = p
        .iter()
        .zip(f.iter())
        .map(|(&pi, &fi)| pi + h * fi)
        .collect();
    // Half drift.
    let q_new: Vec<f32> = q_half
        .iter()
        .zip(p_new.iter())
        .map(|(&qhi, &pi)| qhi + 0.5 * h * pi)
        .collect();
    finite(&q_new, "stormer_verlet_step::q")?;
    finite(&p_new, "stormer_verlet_step::p")?;
    Ok((q_new, p_new))
}

/// One **symplectic (semi-implicit) Euler** step: update momentum with the old
/// position, then position with the new momentum.
///
/// ```text
/// p' = p + h F(q)
/// q' = q + h p'
/// ```
///
/// 1st-order accurate but symplectic (energy-bounded), unlike explicit Euler.
///
/// # Errors
/// See [`velocity_verlet_step`].
pub fn symplectic_euler_step(
    force: ForceFn,
    q: &[f32],
    p: &[f32],
    h: f32,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    check_state(q, p, h)?;
    let dim = q.len();
    let mut f = vec![0.0_f32; dim];
    force(q, &mut f);
    finite(&f, "symplectic_euler_step::F(q)")?;
    let p_new: Vec<f32> = p
        .iter()
        .zip(f.iter())
        .map(|(&pi, &fi)| pi + h * fi)
        .collect();
    let q_new: Vec<f32> = q
        .iter()
        .zip(p_new.iter())
        .map(|(&qi, &pi)| qi + h * pi)
        .collect();
    finite(&q_new, "symplectic_euler_step::q")?;
    finite(&p_new, "symplectic_euler_step::p")?;
    Ok((q_new, p_new))
}

/// Roll out a fixed-step symplectic trajectory of `n_steps` steps with the
/// chosen `method`.
///
/// Returns the stored trajectory `[(q_0, p_0), (q_1, p_1), …, (q_n, p_n)]`
/// (length `n_steps + 1`, including the initial state).
///
/// # Errors
/// - [`PinnError::InvalidStepSize`] if `h <= 0` or non-finite.
/// - Propagates step errors.
#[allow(clippy::type_complexity)]
pub fn integrate_symplectic(
    force: ForceFn,
    q0: &[f32],
    p0: &[f32],
    h: f32,
    n_steps: usize,
    method: SymplecticMethod,
) -> PinnResult<Vec<(Vec<f32>, Vec<f32>)>> {
    if !h.is_finite() || h <= 0.0 {
        return Err(PinnError::InvalidStepSize { h });
    }
    check_state(q0, p0, h)?;
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();
    traj.push((q.clone(), p.clone()));
    for _ in 0..n_steps {
        let (qn, pn) = match method {
            SymplecticMethod::SymplecticEuler => symplectic_euler_step(force, &q, &p, h)?,
            SymplecticMethod::VelocityVerlet => velocity_verlet_step(force, &q, &p, h)?,
            SymplecticMethod::StormerVerlet => stormer_verlet_step(force, &q, &p, h)?,
        };
        q = qn;
        p = pn;
        traj.push((q.clone(), p.clone()));
    }
    Ok(traj)
}

/// Total energy `H = ½|p|² + V(q)` of a unit-mass separable system given a
/// potential `V`.
///
/// # Errors
/// - [`PinnError::EmptyInput`] if `p` is empty.
/// - [`PinnError::NanEncountered`] if the result is non-finite.
pub fn hamiltonian_energy<V>(q: &[f32], p: &[f32], potential: V) -> PinnResult<f32>
where
    V: Fn(&[f32]) -> f32,
{
    if p.is_empty() {
        return Err(PinnError::EmptyInput);
    }
    let kinetic = 0.5 * p.iter().map(|&pi| pi * pi).sum::<f32>();
    let energy = kinetic + potential(q);
    if !energy.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "hamiltonian_energy",
        });
    }
    Ok(energy)
}

// ────────────────────────────── helpers ──────────────────────────────────────

#[inline]
fn check_state(q: &[f32], p: &[f32], h: f32) -> PinnResult<()> {
    if q.len() != p.len() {
        return Err(PinnError::DimensionMismatch {
            expected: q.len(),
            got: p.len(),
        });
    }
    if q.is_empty() {
        return Err(PinnError::EmptyInput);
    }
    if !h.is_finite() {
        return Err(PinnError::InvalidStepSize { h });
    }
    Ok(())
}

#[inline]
fn finite(v: &[f32], location: &'static str) -> PinnResult<()> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(PinnError::NanEncountered { location });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Harmonic oscillator: V(q) = ½ω²q², F(q) = −ω²q. Energy H = ½p² + ½ω²q².
    fn sho_force(omega2: f32) -> impl Fn(&[f32], &mut [f32]) {
        move |q: &[f32], f: &mut [f32]| {
            for (fi, &qi) in f.iter_mut().zip(q.iter()) {
                *fi = -omega2 * qi;
            }
        }
    }

    fn sho_potential(omega2: f32) -> impl Fn(&[f32]) -> f32 {
        move |q: &[f32]| 0.5 * omega2 * q.iter().map(|&qi| qi * qi).sum::<f32>()
    }

    #[test]
    fn velocity_verlet_single_step_shapes() {
        let f = sho_force(1.0);
        let (q, p) = velocity_verlet_step(&f, &[1.0], &[0.0], 0.01)
            .expect("velocity_verlet_step with valid SHO params should succeed");
        assert_eq!(q.len(), 1);
        assert_eq!(p.len(), 1);
        assert!(q[0].is_finite() && p[0].is_finite());
    }

    #[test]
    fn leapfrog_equals_velocity_verlet() {
        let f = sho_force(2.0);
        let (q1, p1) = leapfrog_step(&f, &[0.5], &[0.3], 0.02)
            .expect("leapfrog_step with valid SHO params should succeed");
        let (q2, p2) = velocity_verlet_step(&f, &[0.5], &[0.3], 0.02)
            .expect("velocity_verlet_step for leapfrog equivalence comparison should succeed");
        assert!((q1[0] - q2[0]).abs() < 1e-9);
        assert!((p1[0] - p2[0]).abs() < 1e-9);
    }

    #[test]
    fn sho_energy_conserved_velocity_verlet() {
        // ω = 1, one full period T = 2π. Symplectic energy error must be tiny & bounded.
        let omega2 = 1.0_f32;
        let f = sho_force(omega2);
        let v = sho_potential(omega2);
        let h = 0.001_f32;
        let n = (2.0 * std::f32::consts::PI / h) as usize;
        let traj = integrate_symplectic(&f, &[1.0], &[0.0], h, n, SymplecticMethod::VelocityVerlet)
            .expect("symplectic integration of SHO over one period should succeed");
        let e0 = hamiltonian_energy(&traj[0].0, &traj[0].1, &v)
            .expect("initial hamiltonian energy of SHO should be finite");
        let mut max_err = 0.0_f32;
        for (q, p) in &traj {
            let e = hamiltonian_energy(q, p, &v)
                .expect("hamiltonian energy should be computable at each trajectory point");
            max_err = max_err.max((e - e0).abs() / e0);
        }
        assert!(max_err < 1e-3, "energy drift too large: {max_err}");
    }

    #[test]
    fn sho_period_returns_to_start() {
        // After one full period the harmonic oscillator returns near (q0, p0).
        let omega2 = 1.0_f32;
        let f = sho_force(omega2);
        let h = 0.0005_f32;
        let n = (2.0 * std::f32::consts::PI / h).round() as usize;
        let traj = integrate_symplectic(&f, &[1.0], &[0.0], h, n, SymplecticMethod::VelocityVerlet)
            .expect("symplectic integration over one SHO period should succeed");
        let (qf, pf) = traj
            .last()
            .expect("trajectory should be non-empty after integration");
        assert!((qf[0] - 1.0).abs() < 1e-2, "q final = {}", qf[0]);
        assert!(pf[0].abs() < 1e-2, "p final = {}", pf[0]);
    }

    #[test]
    fn stormer_verlet_energy_bounded() {
        let omega2 = 1.0_f32;
        let f = sho_force(omega2);
        let v = sho_potential(omega2);
        let h = 0.001_f32;
        let n = 2000;
        let traj = integrate_symplectic(&f, &[1.0], &[0.0], h, n, SymplecticMethod::StormerVerlet)
            .expect("StormerVerlet integration of SHO should succeed");
        let e0 = hamiltonian_energy(&traj[0].0, &traj[0].1, &v)
            .expect("initial hamiltonian energy for StormerVerlet trajectory should be finite");
        for (q, p) in &traj {
            let e = hamiltonian_energy(q, p, &v)
                .expect("hamiltonian energy should remain finite at each StormerVerlet step");
            assert!((e - e0).abs() / e0 < 1e-3);
        }
    }

    #[test]
    fn symplectic_euler_first_order_step() {
        // p' = p + h F(q); q' = q + h p'. Check exact arithmetic on one step.
        let f = sho_force(1.0);
        let (q, p) = symplectic_euler_step(&f, &[2.0], &[1.0], 0.1)
            .expect("symplectic_euler_step with known values should succeed");
        // F(2) = -2 → p' = 1 + 0.1*(-2) = 0.8 ; q' = 2 + 0.1*0.8 = 2.08
        assert!((p[0] - 0.8).abs() < 1e-6, "p = {}", p[0]);
        assert!((q[0] - 2.08).abs() < 1e-6, "q = {}", q[0]);
    }

    #[test]
    fn time_reversibility_velocity_verlet() {
        // Step forward then step backward (negate p, step, negate p) → original.
        let f = sho_force(1.3);
        let q0 = [0.7_f32];
        let p0 = [0.2_f32];
        let h = 0.01_f32;
        let (q1, p1) = velocity_verlet_step(&f, &q0, &p0, h)
            .expect("forward velocity_verlet_step for time-reversibility test should succeed");
        let p1_rev = [-p1[0]];
        let (q2, p2) = velocity_verlet_step(&f, &q1, &p1_rev, h)
            .expect("backward velocity_verlet_step for time-reversibility test should succeed");
        assert!((q2[0] - q0[0]).abs() < 1e-5, "q reversed = {}", q2[0]);
        assert!((-p2[0] - p0[0]).abs() < 1e-5, "p reversed = {}", -p2[0]);
    }

    #[test]
    fn multi_dim_state() {
        // 2-D isotropic oscillator.
        let f = sho_force(1.0);
        let v = sho_potential(1.0);
        let traj = integrate_symplectic(
            &f,
            &[1.0, 0.0],
            &[0.0, 1.0],
            0.001,
            500,
            SymplecticMethod::VelocityVerlet,
        )
        .expect("2D isotropic oscillator symplectic integration should succeed");
        let e0 = hamiltonian_energy(&traj[0].0, &traj[0].1, &v)
            .expect("initial hamiltonian energy of 2D oscillator should be finite");
        let ef = hamiltonian_energy(
            &traj
                .last()
                .expect("trajectory should be non-empty after 2D oscillator integration")
                .0,
            &traj
                .last()
                .expect("trajectory should be non-empty after 2D oscillator integration")
                .1,
            &v,
        )
        .expect("final hamiltonian energy of 2D oscillator should be finite");
        assert!((ef - e0).abs() / e0 < 1e-3);
    }

    #[test]
    fn integrate_trajectory_length() {
        let f = sho_force(1.0);
        let traj = integrate_symplectic(
            &f,
            &[1.0],
            &[0.0],
            0.01,
            50,
            SymplecticMethod::SymplecticEuler,
        )
        .expect("symplectic integration trajectory length test should succeed");
        assert_eq!(traj.len(), 51);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let f = sho_force(1.0);
        assert!(matches!(
            velocity_verlet_step(&f, &[1.0, 2.0], &[0.0], 0.01),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_step_size_errors() {
        let f = sho_force(1.0);
        assert!(matches!(
            integrate_symplectic(
                &f,
                &[1.0],
                &[0.0],
                0.0,
                10,
                SymplecticMethod::VelocityVerlet
            ),
            Err(PinnError::InvalidStepSize { .. })
        ));
        assert!(velocity_verlet_step(&f, &[1.0], &[0.0], f32::NAN).is_err());
    }

    #[test]
    fn energy_empty_errors() {
        assert!(matches!(
            hamiltonian_energy(&[], &[], |_q| 0.0),
            Err(PinnError::EmptyInput)
        ));
    }
}
