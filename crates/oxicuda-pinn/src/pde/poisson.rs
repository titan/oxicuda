//! Poisson equation PDE templates and analytic solutions.
//!
//! PDE: `∇²u = f` on \[0,1\]²

use crate::error::{PinnError, PinnResult};

/// Compute Poisson equation PDE residual.
///
/// `R = u_xx + u_yy - f`
pub fn poisson_residual(u_xx: f32, u_yy: f32, f: f32) -> PinnResult<f32> {
    let r = u_xx + u_yy - f;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "poisson_residual",
        });
    }
    Ok(r)
}

/// Analytic Poisson solution.
///
/// For `f = -2π²·sin(πx)·sin(πy)`, the solution is `u = sin(πx)·sin(πy)`.
pub fn poisson_analytic(x: f32, y: f32) -> f32 {
    let pi = std::f32::consts::PI;
    (pi * x).sin() * (pi * y).sin()
}

/// Source term corresponding to the analytic solution.
pub fn poisson_source(x: f32, y: f32) -> f32 {
    let pi = std::f32::consts::PI;
    -2.0 * pi * pi * (pi * x).sin() * (pi * y).sin()
}

/// Verify analytic satisfies residual within tolerance.
pub fn poisson_residual_check(x: f32, y: f32, tol: f32) -> PinnResult<bool> {
    let pi = std::f32::consts::PI;
    let u = poisson_analytic(x, y);
    let u_xx = -pi * pi * u;
    let u_yy = -pi * pi * u;
    let f = poisson_source(x, y);
    let r = poisson_residual(u_xx, u_yy, f)?;
    Ok(r.abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_analytic_bc_on_boundaries() {
        // u(0, y) = u(1, y) = u(x, 0) = u(x, 1) = 0
        for i in 0..=5 {
            let s = i as f32 / 5.0;
            assert!(poisson_analytic(0.0, s).abs() < 1e-6);
            assert!(poisson_analytic(1.0, s).abs() < 1e-5);
            assert!(poisson_analytic(s, 0.0).abs() < 1e-6);
            assert!(poisson_analytic(s, 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn poisson_analytic_peak() {
        // Peak at (0.5, 0.5)
        let u_peak = poisson_analytic(0.5, 0.5);
        assert!(
            (u_peak - 1.0).abs() < 1e-5,
            "Peak should be 1, got {u_peak}"
        );
    }

    #[test]
    fn poisson_residual_analytic_near_zero() {
        let ok = poisson_residual_check(0.5, 0.5, 1e-3)
            .expect("poisson_residual_check should succeed for interior point (0.5, 0.5)");
        assert!(
            ok,
            "Poisson residual on analytic solution should be near zero"
        );
    }

    #[test]
    fn poisson_residual_on_grid() {
        for i in 1..5 {
            for j in 1..5 {
                let x = i as f32 / 5.0;
                let y = j as f32 / 5.0;
                let ok = poisson_residual_check(x, y, 1e-3)
                    .expect("poisson_residual_check should succeed for interior grid point");
                assert!(ok, "Poisson grid residual failed at ({x}, {y})");
            }
        }
    }

    #[test]
    fn poisson_residual_formula() {
        let r = poisson_residual(1.0, -2.0, 3.0)
            .expect("poisson_residual should succeed for finite inputs u_xx=1.0, u_yy=-2.0, f=3.0");
        assert!((r - (1.0 + -2.0 - 3.0)).abs() < 1e-6);
    }

    #[test]
    fn poisson_source_negative() {
        // Source f = -2π²·sin(πx)·sin(πy) is negative in interior
        let f = poisson_source(0.5, 0.5);
        assert!(f < 0.0, "Source at interior should be negative: {f}");
    }
}
