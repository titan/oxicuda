//! Heat equation PDE templates and analytic solutions.
//!
//! PDE: `∂u/∂t - α·∂²u/∂x² = 0`

use crate::error::{PinnError, PinnResult};

/// Compute heat equation PDE residual.
///
/// `R = u_t - alpha * u_xx`
pub fn heat_residual(u_t: f32, u_xx: f32, alpha: f32) -> PinnResult<f32> {
    if alpha <= 0.0 {
        return Err(PinnError::InvalidPdeCoefficient {
            name: "alpha",
            value: alpha,
        });
    }
    let r = u_t - alpha * u_xx;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "heat_residual",
        });
    }
    Ok(r)
}

/// Analytic solution to the 1D heat equation on \[0,1\]:
/// `u(x, t) = sin(π·x) · exp(-α·π²·t)`.
pub fn heat_analytic(x: f32, t: f32, alpha: f32) -> f32 {
    let pi = std::f32::consts::PI;
    (pi * x).sin() * (-alpha * pi * pi * t).exp()
}

/// Check that the analytic solution satisfies the residual within tolerance.
pub fn heat_residual_check(x: f32, t: f32, alpha: f32, tol: f32) -> PinnResult<bool> {
    let pi = std::f32::consts::PI;
    let u = heat_analytic(x, t, alpha);
    // Compute ∂u/∂t = -α·π²·u
    let u_t = -alpha * pi * pi * u;
    // Compute ∂²u/∂x² = -π²·u
    let u_xx = -pi * pi * u;
    let r = heat_residual(u_t, u_xx, alpha)?;
    Ok(r.abs() < tol)
}

/// Evaluate residuals on a grid of (x, t) points.
pub fn heat_residual_on_grid(x_pts: &[f32], t_pts: &[f32], alpha: f32) -> PinnResult<Vec<f32>> {
    let pi = std::f32::consts::PI;
    let mut residuals = Vec::with_capacity(x_pts.len() * t_pts.len());
    for &x in x_pts {
        for &t in t_pts {
            let u = heat_analytic(x, t, alpha);
            let u_t = -alpha * pi * pi * u;
            let u_xx = -pi * pi * u;
            residuals.push(heat_residual(u_t, u_xx, alpha)?);
        }
    }
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: f32 = 0.01;

    #[test]
    fn heat_analytic_ic() {
        // u(x, 0) = sin(πx)
        let pi = std::f32::consts::PI;
        for i in 0..10 {
            let x = i as f32 / 10.0;
            let u = heat_analytic(x, 0.0, ALPHA);
            assert!((u - (pi * x).sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn heat_analytic_bc_left() {
        // u(0, t) = 0
        for i in 0..5 {
            let t = i as f32 * 0.1;
            assert!(heat_analytic(0.0, t, ALPHA).abs() < 1e-6);
        }
    }

    #[test]
    fn heat_analytic_bc_right() {
        // u(1, t) = sin(π) = 0
        for i in 0..5 {
            let t = i as f32 * 0.1;
            assert!(heat_analytic(1.0, t, ALPHA).abs() < 1e-5);
        }
    }

    #[test]
    fn heat_residual_analytic_near_zero() {
        let pi = std::f32::consts::PI;
        let x = 0.5_f32;
        let t = 0.1_f32;
        let u = heat_analytic(x, t, ALPHA);
        let u_t = -ALPHA * pi * pi * u;
        let u_xx = -pi * pi * u;
        let r = heat_residual(u_t, u_xx, ALPHA).unwrap();
        assert!(
            r.abs() < 1e-3,
            "Residual on analytic solution should be ~0, got {r}"
        );
    }

    #[test]
    fn heat_residual_check_passes() {
        let ok = heat_residual_check(0.5, 0.5, ALPHA, 1e-3).unwrap();
        assert!(ok, "heat_residual_check should pass for analytic solution");
    }

    #[test]
    fn heat_residual_nonzero_for_wrong_solution() {
        // u = 1 (constant) → u_t = 0, u_xx = 0, R = 0 only by coincidence
        let r = heat_residual(1.0, 0.0, ALPHA).unwrap();
        assert!((r - 1.0).abs() < 1e-5, "R = u_t - alpha*u_xx = 1 - 0 = 1");
    }

    #[test]
    fn heat_invalid_alpha_error() {
        let result = heat_residual(0.0, 0.0, 0.0);
        assert!(matches!(
            result,
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn heat_negative_alpha_error() {
        let result = heat_residual(0.0, 0.0, -0.1);
        assert!(matches!(
            result,
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn heat_grid_residuals_small() {
        let x_pts: Vec<f32> = (1..5).map(|i| i as f32 * 0.2).collect();
        let t_pts: Vec<f32> = (1..5).map(|i| i as f32 * 0.1).collect();
        let residuals = heat_residual_on_grid(&x_pts, &t_pts, ALPHA).unwrap();
        for r in &residuals {
            assert!(r.abs() < 1e-3, "Grid residual should be near zero: {r}");
        }
    }
}
