//! Navier-Stokes vorticity equation PDE templates.
//!
//! Vorticity form: `∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y - ν·(∂²ω/∂x² + ∂²ω/∂y²) = 0`

use crate::error::{PinnError, PinnResult};

/// Compute NS vorticity equation residual.
///
/// `R = omega_t + u*omega_x + v*omega_y - nu*(omega_xx + omega_yy)`
pub fn ns_vorticity_residual(
    omega_t: f32,
    u: f32,
    v: f32,
    omega_x: f32,
    omega_y: f32,
    omega_xx: f32,
    omega_yy: f32,
    nu: f32,
) -> PinnResult<f32> {
    if nu <= 0.0 {
        return Err(PinnError::InvalidPdeCoefficient {
            name: "nu",
            value: nu,
        });
    }
    let r = omega_t + u * omega_x + v * omega_y - nu * (omega_xx + omega_yy);
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "ns_vorticity_residual",
        });
    }
    Ok(r)
}

/// Taylor-Green vortex analytic solution.
///
/// `ω(x, y, t) = 2·cos(x)·cos(y)·exp(-2νt)`
///
/// Velocity field: `u = cos(x)·sin(y)·exp(-2νt)`, `v = -sin(x)·cos(y)·exp(-2νt)`.
pub fn taylor_green_vortex(x: f32, y: f32, t: f32, nu: f32) -> f32 {
    2.0 * x.cos() * y.cos() * (-2.0 * nu * t).exp()
}

/// Taylor-Green velocity field.
pub fn taylor_green_velocity(x: f32, y: f32, t: f32, nu: f32) -> (f32, f32) {
    let decay = (-2.0 * nu * t).exp();
    let u = x.cos() * y.sin() * decay;
    let v = -x.sin() * y.cos() * decay;
    (u, v)
}

/// Verify Taylor-Green satisfies vorticity equation.
pub fn taylor_green_residual_check(x: f32, y: f32, t: f32, nu: f32, tol: f32) -> PinnResult<bool> {
    let decay = (-2.0 * nu * t).exp();
    let omega = taylor_green_vortex(x, y, t, nu);
    let omega_t = -2.0 * nu * omega; // ∂ω/∂t = -2ν·ω
    let omega_x = -2.0 * x.sin() * y.cos() * decay; // ∂ω/∂x
    let omega_y = -2.0 * x.cos() * y.sin() * decay; // ∂ω/∂y
    let omega_xx = -2.0 * x.cos() * y.cos() * decay; // ∂²ω/∂x²
    let omega_yy = -2.0 * x.cos() * y.cos() * decay; // ∂²ω/∂y²
    let (u, v) = taylor_green_velocity(x, y, t, nu);
    let r = ns_vorticity_residual(omega_t, u, v, omega_x, omega_y, omega_xx, omega_yy, nu)?;
    Ok(r.abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NU: f32 = 0.1;

    #[test]
    fn taylor_green_ic_at_t0() {
        // ω(x, y, 0) = 2cos(x)cos(y)
        let x = 0.5_f32;
        let y = 0.7_f32;
        let omega = taylor_green_vortex(x, y, 0.0, NU);
        let expected = 2.0 * x.cos() * y.cos();
        assert!((omega - expected).abs() < 1e-6);
    }

    #[test]
    fn taylor_green_decay() {
        // ω should decay with time
        let x = 0.3_f32;
        let y = 0.4_f32;
        let omega0 = taylor_green_vortex(x, y, 0.0, NU).abs();
        let omega1 = taylor_green_vortex(x, y, 1.0, NU).abs();
        assert!(
            omega1 < omega0,
            "Taylor-Green should decay: |ω(t=1)|={omega1} >= |ω(t=0)|={omega0}"
        );
    }

    #[test]
    fn ns_vorticity_residual_formula() {
        let r = ns_vorticity_residual(1.0, 0.5, 0.3, 2.0, 1.5, 0.5, 0.5, NU).unwrap();
        let expected = 1.0 + 0.5 * 2.0 + 0.3 * 1.5 - NU * (0.5 + 0.5);
        assert!((r - expected).abs() < 1e-5);
    }

    #[test]
    fn taylor_green_residual_check_passes() {
        let ok = taylor_green_residual_check(0.5, 0.5, 0.1, NU, 0.05).unwrap();
        assert!(ok, "Taylor-Green residual check should pass");
    }

    #[test]
    fn ns_invalid_nu_error() {
        let result = ns_vorticity_residual(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(matches!(
            result,
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn taylor_green_velocity_divergence_free() {
        // ∂u/∂x + ∂v/∂y should be ~0 (incompressible)
        let eps = 1e-4_f32;
        let x = 1.0_f32;
        let y = 0.5_f32;
        let t = 0.2_f32;
        let (u_plus, _) = taylor_green_velocity(x + eps, y, t, NU);
        let (u_minus, _) = taylor_green_velocity(x - eps, y, t, NU);
        let (_, v_plus) = taylor_green_velocity(x, y + eps, t, NU);
        let (_, v_minus) = taylor_green_velocity(x, y - eps, t, NU);
        let div = (u_plus - u_minus) / (2.0 * eps) + (v_plus - v_minus) / (2.0 * eps);
        assert!(div.abs() < 1e-3, "Divergence should be ~0, got {div}");
    }

    #[test]
    fn taylor_green_vortex_finite() {
        for i in 0..5 {
            for j in 0..5 {
                let x = i as f32 * 0.5;
                let y = j as f32 * 0.5;
                let omega = taylor_green_vortex(x, y, 0.5, NU);
                assert!(omega.is_finite());
            }
        }
    }
}
