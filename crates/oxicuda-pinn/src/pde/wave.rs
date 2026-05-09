//! Wave equation PDE templates and D'Alembert analytic solution.
//!
//! PDE: `∂²u/∂t² - c²·∂²u/∂x² = 0`

use crate::error::{PinnError, PinnResult};

/// Compute wave equation PDE residual.
///
/// `R = u_tt - c² * u_xx`
pub fn wave_residual(u_tt: f32, u_xx: f32, c: f32) -> PinnResult<f32> {
    if c <= 0.0 {
        return Err(PinnError::InvalidPdeCoefficient {
            name: "c",
            value: c,
        });
    }
    let r = u_tt - c * c * u_xx;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "wave_residual",
        });
    }
    Ok(r)
}

/// D'Alembert analytic solution.
///
/// For initial condition `u₀(x) = sin(π·x)`, `u_t(x, 0) = 0`:
/// `u(x, t) = 0.5·[sin(π·(x-c·t)) + sin(π·(x+c·t))]`.
pub fn wave_analytic(x: f32, t: f32, c: f32) -> f32 {
    let pi = std::f32::consts::PI;
    0.5 * ((pi * (x - c * t)).sin() + (pi * (x + c * t)).sin())
}

/// Verify analytic satisfies wave PDE within tolerance.
pub fn wave_residual_check(x: f32, t: f32, c: f32, tol: f32) -> PinnResult<bool> {
    let pi = std::f32::consts::PI;
    let u = wave_analytic(x, t, c);
    // ∂²u/∂t² = -c²·π²·0.5·[sin(π(x-ct)) + sin(π(x+ct))] = -c²·π²·u
    let u_tt = -c * c * pi * pi * u;
    // ∂²u/∂x² = -π²·u
    let u_xx = -pi * pi * u;
    let r = wave_residual(u_tt, u_xx, c)?;
    Ok(r.abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: f32 = 1.0;

    #[test]
    fn wave_analytic_ic_displacement() {
        // u(x, 0) = sin(πx)
        let pi = std::f32::consts::PI;
        for i in 0..8 {
            let x = i as f32 / 8.0;
            let u = wave_analytic(x, 0.0, C);
            assert!(
                (u - (pi * x).sin()).abs() < 1e-5,
                "IC failed at x={x}: u={u}"
            );
        }
    }

    #[test]
    fn wave_analytic_ic_velocity() {
        // u_t(x, 0) ≈ 0 (via finite diff)
        let eps = 1e-5_f32;
        for i in 0..5 {
            let x = (i + 1) as f32 * 0.2;
            let u_t = (wave_analytic(x, eps, C) - wave_analytic(x, -eps, C)) / (2.0 * eps);
            assert!(u_t.abs() < 1e-3, "u_t(x,0) should be 0 at x={x}, got {u_t}");
        }
    }

    #[test]
    fn wave_residual_analytic_near_zero() {
        let pi = std::f32::consts::PI;
        let x = 0.5_f32;
        let t = 0.3_f32;
        let u = wave_analytic(x, t, C);
        let u_tt = -C * C * pi * pi * u;
        let u_xx = -pi * pi * u;
        let r = wave_residual(u_tt, u_xx, C).unwrap();
        assert!(r.abs() < 1e-3, "Wave residual on analytic solution: {r}");
    }

    #[test]
    fn wave_residual_check_passes() {
        let ok = wave_residual_check(0.4, 0.2, C, 1e-3).unwrap();
        assert!(ok);
    }

    #[test]
    fn wave_invalid_c_error() {
        let result = wave_residual(0.0, 0.0, 0.0);
        assert!(matches!(
            result,
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn wave_negative_c_error() {
        let result = wave_residual(0.0, 0.0, -1.0);
        assert!(matches!(
            result,
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }
}
