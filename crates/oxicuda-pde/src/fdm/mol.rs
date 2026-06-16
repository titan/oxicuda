//! Method of Lines (MOL) for the 1D heat equation  u_t = α u_xx.
//!
//! Spatial discretisation uses a second-order centred finite-difference
//! stencil; time integration uses explicit Forward Euler. The approach
//! is due to Schiesser (1991), "The Numerical Method of Lines".
//!
//! Stability constraint (CFL-type): `α dt / dx² ≤ 0.5`.

use crate::error::{PdeError, PdeResult};

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for the Method-of-Lines 1D heat equation solver.
///
/// The spatial domain is the interior of `[0, L]` represented by
/// `n_x` equally-spaced interior points with spacing `dx`.
/// The full grid (including the two boundary ghost points) therefore has
/// `n_x + 2` points, but only the `n_x` interior values are tracked.
#[derive(Debug, Clone)]
pub struct MolConfig {
    /// Number of **interior** spatial points (excluding the two BC nodes).
    pub n_x: usize,
    /// Spatial step h = L / (n_x + 1).
    pub dx: f64,
    /// Thermal diffusivity α > 0.
    pub alpha: f64,
    /// Final time T ≥ 0.
    pub t_end: f64,
    /// Time step Δt used in the Forward Euler integrator.
    pub dt: f64,
    /// Dirichlet BC at the left wall: u(0, t) = bc_left.
    pub bc_left: f64,
    /// Dirichlet BC at the right wall: u(L, t) = bc_right.
    pub bc_right: f64,
}

// ─── Solver ────────────────────────────────────────────────────────────────

/// Integrate the 1D heat equation `u_t = α u_xx` from t=0 to t=t_end using
/// the Method of Lines with a Forward Euler time integrator.
///
/// # Arguments
/// * `u0`     – Initial condition at the `n_x` interior points.
/// * `config` – Solver parameters (see [`MolConfig`]).
///
/// # Returns
/// `Vec<f64>` of length `n_x` holding the interior solution at `t = t_end`.
///
/// # Errors
/// * [`PdeError::ShapeMismatch`] – `u0.len() != n_x`.
/// * [`PdeError::CflViolation`]  – stability condition `α dt / dx² > 0.5`.
/// * [`PdeError::NumericalInstability`] – any output value is not finite.
///
/// # Notes
/// If `t_end == 0.0` the initial condition is returned immediately without
/// any time stepping, so the CFL condition is still validated.
pub fn mol_heat_1d(u0: &[f64], config: &MolConfig) -> PdeResult<Vec<f64>> {
    let n_x = config.n_x;
    let dx = config.dx;
    let alpha = config.alpha;
    let dt = config.dt;
    let t_end = config.t_end;

    // ── Input validation ────────────────────────────────────────────────
    if u0.len() != n_x {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n_x],
            got: vec![u0.len()],
        });
    }

    // CFL / von-Neumann stability for Forward Euler on the heat equation:
    //   r = α dt / dx² ≤ 0.5
    // We allow alpha == 0 (r == 0 is trivially stable).
    let r = if dx > 0.0 {
        alpha * dt / (dx * dx)
    } else {
        0.0
    };
    if r > 0.5 + 1.0e-12 {
        let dt_max = 0.5 * dx * dx / alpha;
        return Err(PdeError::CflViolation { dt, dt_max });
    }

    // ── t_end == 0: return initial condition unchanged ──────────────────
    if t_end == 0.0 {
        return Ok(u0.to_vec());
    }

    // ── Time integration (Forward Euler) ────────────────────────────────
    let mut u = u0.to_vec();
    let mut t = 0.0_f64;

    // Number of full steps and the possible remainder step.
    let n_steps = (t_end / dt).floor() as usize;
    let dt_rem = t_end - n_steps as f64 * dt;

    // Helper closure: advance by one step of size h.
    // Borrows bc_left/bc_right from config.
    let step = |u: &mut Vec<f64>, h: f64| {
        let coeff = alpha * h / (dx * dx);
        let mut u_new = vec![0.0_f64; n_x];
        for i in 0..n_x {
            let u_left = if i == 0 { config.bc_left } else { u[i - 1] };
            let u_right = if i == n_x - 1 {
                config.bc_right
            } else {
                u[i + 1]
            };
            let laplacian = u_left - 2.0 * u[i] + u_right;
            u_new[i] = u[i] + coeff * laplacian;
        }
        *u = u_new;
    };

    for _ in 0..n_steps {
        step(&mut u, dt);
        t += dt;
    }

    // Final (possibly fractional) step to reach exactly t_end.
    if dt_rem > 1.0e-15 {
        step(&mut u, dt_rem);
        t += dt_rem;
    }

    // Silence unused-variable warning (t is used only for tracking).
    let _ = t;

    // ── Validate output ─────────────────────────────────────────────────
    for (i, &v) in u.iter().enumerate() {
        if !v.is_finite() {
            return Err(PdeError::NumericalInstability(format!(
                "mol_heat_1d: u[{i}] = {v} is not finite at t={t_end:.4e}"
            )));
        }
    }

    Ok(u)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience constructor for a default stable config.
    fn stable_config(n_x: usize, alpha: f64, t_end: f64) -> MolConfig {
        let dx = 1.0 / (n_x + 1) as f64;
        // Use r = 0.4 < 0.5 for stability margin.
        let dt = 0.4 * dx * dx / alpha.max(1.0e-15);
        MolConfig {
            n_x,
            dx,
            alpha,
            t_end,
            dt,
            bc_left: 0.0,
            bc_right: 0.0,
        }
    }

    // ── Basic shape / identity tests ────────────────────────────────────

    #[test]
    fn output_len() {
        let n_x = 10_usize;
        let cfg = stable_config(n_x, 1.0, 0.01);
        let u0 = vec![0.5_f64; n_x];
        let result = mol_heat_1d(&u0, &cfg).expect("should succeed");
        assert_eq!(result.len(), n_x, "output length must equal n_x");
    }

    #[test]
    fn t_end_0_returns_u0() {
        let n_x = 8_usize;
        let u0: Vec<f64> = (0..n_x).map(|i| i as f64 * 0.1).collect();
        let cfg = stable_config(n_x, 1.0, 0.0);
        let result = mol_heat_1d(&u0, &cfg).expect("t_end=0 should succeed");
        assert_eq!(
            result, u0,
            "t_end=0 must return the initial condition unchanged"
        );
    }

    // ── Error conditions ────────────────────────────────────────────────

    #[test]
    fn shape_mismatch_error() {
        let n_x = 5_usize;
        let u0 = vec![0.0_f64; n_x + 1]; // wrong length
        let cfg = stable_config(n_x, 1.0, 0.01);
        let result = mol_heat_1d(&u0, &cfg);
        assert!(result.is_err(), "wrong u0 length should return Err");
        match result {
            Err(PdeError::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, vec![n_x]);
                assert_eq!(got, vec![n_x + 1]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn stability_check_error() {
        let n_x = 10_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        // r = alpha * dt / dx^2 = 1.0 * 1.0 / dx^2 >> 0.5  → CFL violation.
        let cfg = MolConfig {
            n_x,
            dx,
            alpha: 1.0,
            t_end: 0.1,
            dt: 1.0, // grossly unstable
            bc_left: 0.0,
            bc_right: 0.0,
        };
        let u0 = vec![0.5_f64; n_x];
        let result = mol_heat_1d(&u0, &cfg);
        assert!(result.is_err(), "CFL violation should return Err");
        match result {
            Err(PdeError::CflViolation { .. }) => {}
            other => panic!("expected CflViolation, got {other:?}"),
        }
    }

    // ── Physics / accuracy tests ────────────────────────────────────────

    #[test]
    fn finite_output() {
        let n_x = 20_usize;
        let cfg = stable_config(n_x, 0.5, 0.02);
        let pi = std::f64::consts::PI;
        let dx = 1.0 / (n_x + 1) as f64;
        let u0: Vec<f64> = (0..n_x).map(|i| ((i + 1) as f64 * dx * pi).sin()).collect();
        let result = mol_heat_1d(&u0, &cfg).expect("should not blow up");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "u[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn zero_diffusion_no_change() {
        // With alpha=0 the heat equation reduces to u_t = 0 → u stays constant.
        let n_x = 10_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        let cfg = MolConfig {
            n_x,
            dx,
            alpha: 0.0,
            t_end: 1.0,
            dt: 0.1, // r = 0 * 0.1 / dx^2 = 0 ≤ 0.5 → stable
            bc_left: 0.0,
            bc_right: 0.0,
        };
        let u0: Vec<f64> = (0..n_x).map(|i| (i + 1) as f64 * 0.05).collect();
        let result = mol_heat_1d(&u0, &cfg).expect("alpha=0 should succeed");
        for (i, (&u_out, &u_in)) in result.iter().zip(u0.iter()).enumerate() {
            assert!(
                (u_out - u_in).abs() < 1.0e-12,
                "u[{i}]: got {u_out} expected {u_in} (alpha=0)"
            );
        }
    }

    #[test]
    fn steady_state_linear_bc() {
        // bc_left=0, bc_right=1, u0=zeros.  After sufficient time the
        // solution should be close to the linear steady state u(x) = x / L.
        let n_x = 20_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        let alpha = 1.0_f64;
        // Use a stable time step.
        let dt = 0.4 * dx * dx / alpha;
        let cfg = MolConfig {
            n_x,
            dx,
            alpha,
            t_end: 1.0, // long enough to approach steady state for these params
            dt,
            bc_left: 0.0,
            bc_right: 1.0,
        };
        let u0 = vec![0.0_f64; n_x];
        let result = mol_heat_1d(&u0, &cfg).expect("should converge");

        // The steady state is u_ss(x_i) = x_i / L where x_i = (i+1)*dx
        // and L = (n_x+1)*dx = 1.0.
        for (i, &v) in result.iter().enumerate() {
            let x_i = (i + 1) as f64 * dx;
            let u_ss = x_i; // L = 1
            assert!(
                (v - u_ss).abs() < 0.02,
                "node {i}: got {v:.4} expected ~ {u_ss:.4} (steady state)"
            );
        }
    }

    #[test]
    fn bc_applied() {
        // bc_right=2.0 should cause rightmost interior values to be pulled
        // toward 2.0, while bc_left=0.0 keeps the left end near 0.
        let n_x = 10_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        let alpha = 0.5_f64;
        let dt = 0.4 * dx * dx / alpha;
        let cfg = MolConfig {
            n_x,
            dx,
            alpha,
            t_end: 0.5,
            dt,
            bc_left: 0.0,
            bc_right: 2.0,
        };
        let u0 = vec![0.0_f64; n_x];
        let result = mol_heat_1d(&u0, &cfg).expect("ok");
        // The solution should be monotonically increasing from left to right.
        for i in 1..n_x {
            assert!(
                result[i] >= result[i - 1] - 1.0e-10,
                "result not monotone at i={i}: {} < {}",
                result[i],
                result[i - 1]
            );
        }
        // The rightmost value should be noticeably larger than the leftmost.
        assert!(
            result[n_x - 1] > result[0] + 0.1,
            "rightmost should be much larger than leftmost"
        );
    }

    #[test]
    fn gaussian_pulse_spreads() {
        // A Gaussian IC centred in the domain should spread: the maximum
        // value decreases over time (energy is conserved but concentrated
        // mass diffuses away).
        let n_x = 50_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        let alpha = 0.5_f64;
        let dt = 0.4 * dx * dx / alpha;
        let cfg = MolConfig {
            n_x,
            dx,
            alpha,
            t_end: 0.02,
            dt,
            bc_left: 0.0,
            bc_right: 0.0,
        };
        let mu = 0.5_f64;
        let sigma = 0.05_f64;
        let u0: Vec<f64> = (0..n_x)
            .map(|i| {
                let x = (i + 1) as f64 * dx;
                (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let max_u0 = u0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let result = mol_heat_1d(&u0, &cfg).expect("ok");
        let max_result = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_result < max_u0,
            "Gaussian peak should decrease: max_u0={max_u0:.4} max_result={max_result:.4}"
        );
    }

    #[test]
    fn alpha_affects_speed() {
        // A larger diffusivity should produce a lower peak after the same time
        // (faster diffusion ⇒ more spreading of the same Gaussian IC).
        let n_x = 50_usize;
        let dx = 1.0 / (n_x + 1) as f64;
        let mu = 0.5_f64;
        let sigma = 0.05_f64;
        let u0: Vec<f64> = (0..n_x)
            .map(|i| {
                let x = (i + 1) as f64 * dx;
                (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp()
            })
            .collect();

        let run_with_alpha = |alpha: f64| -> f64 {
            let dt = 0.4 * dx * dx / alpha;
            let cfg = MolConfig {
                n_x,
                dx,
                alpha,
                t_end: 0.01,
                dt,
                bc_left: 0.0,
                bc_right: 0.0,
            };
            let res = mol_heat_1d(&u0, &cfg).expect("ok");
            res.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        let peak_slow = run_with_alpha(0.1);
        let peak_fast = run_with_alpha(1.0);

        assert!(
            peak_fast < peak_slow,
            "larger alpha should spread faster: peak_slow={peak_slow:.4} peak_fast={peak_fast:.4}"
        );
    }
}
