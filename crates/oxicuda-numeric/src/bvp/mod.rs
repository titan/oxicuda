//! Two-point boundary-value problem (BVP) solvers for second-order ODEs
//!
//! ```text
//! y'' = f(x, y, y'),     x ∈ [a, b],     y(a) = α,   y(b) = β.
//! ```
//!
//! Two independent methods are provided:
//!
//! * [`shooting`] — **single shooting**: reduce the BVP to a root-find on the
//!   unknown initial slope `s = y'(a)`, integrating the resulting IVP with RK4 and
//!   driving the boundary residual `φ(s) = y(b; s) − β` to zero by the secant
//!   method;
//! * [`finite_difference`] — **central finite differences** on a uniform grid,
//!   giving a (non)linear system solved by Newton with an analytic *tridiagonal*
//!   Jacobian and the Thomas algorithm.
//!
//! Both accept the right-hand side as a closure `f(x, y, y') → y''` and return the
//! solution sampled on a uniform grid. Single shooting is spectrally accurate in
//! the slope for linear problems (secant converges in two evaluations) and inherits
//! RK4's `𝒪(h⁴)` IVP accuracy; the finite-difference scheme is globally `𝒪(h²)`.
//!
//! Reference: H. B. Keller, *Numerical Methods for Two-Point Boundary-Value
//! Problems*, Blaisdell (1968); U. M. Ascher, R. M. R. Mattheij and R. D. Russell,
//! *Numerical Solution of Boundary Value Problems for Ordinary Differential
//! Equations*, SIAM Classics in Applied Mathematics (1995).

pub mod finite_difference;
pub mod shooting;

pub use finite_difference::{
    FiniteDifferenceConfig, FiniteDifferenceSolution, solve_finite_difference,
};
pub use shooting::{ShootingConfig, ShootingSolution, solve_shooting};

#[cfg(test)]
mod tests {
    use super::finite_difference::{FiniteDifferenceConfig, solve_finite_difference};
    use super::shooting::{ShootingConfig, solve_shooting};

    /// Shooting and finite differences must agree (to within FD truncation error)
    /// on the same linear BVP `y'' = y, y(0)=0, y(1)=1`.
    #[test]
    fn shooting_and_fd_agree_linear() {
        let f = |_x: f64, y: f64, _yp: f64| y;
        let n = 100_usize;
        let shoot = solve_shooting(
            f,
            0.0,
            1.0,
            0.0,
            1.0,
            &ShootingConfig {
                n_steps: n,
                ..ShootingConfig::default()
            },
        )
        .expect("shooting ok");
        let fd = solve_finite_difference(
            f,
            0.0,
            1.0,
            0.0,
            1.0,
            &FiniteDifferenceConfig {
                n_intervals: n,
                ..FiniteDifferenceConfig::default()
            },
        )
        .expect("fd ok");

        // Same grid; compare node-by-node. FD truncation at N=100 is ~1e-5.
        assert_eq!(shoot.x.len(), fd.x.len());
        let mut max_diff = 0.0_f64;
        for ((&xs, &ys), (&xf, &yf)) in shoot
            .x
            .iter()
            .zip(shoot.y.iter())
            .zip(fd.x.iter().zip(fd.y.iter()))
        {
            assert!((xs - xf).abs() < 1.0e-13, "grids differ: {xs} vs {xf}");
            max_diff = max_diff.max((ys - yf).abs());
        }
        // Difference is dominated by the O(h²) FD error.
        assert!(max_diff < 1.0e-3, "max |shoot - fd| = {max_diff:e}");
    }

    /// Both methods reproduce the same nonlinear solution `y = 4/(1+x)²`.
    #[test]
    fn shooting_and_fd_agree_nonlinear() {
        let f = |_x: f64, y: f64, _yp: f64| 1.5 * y * y;
        let shoot = solve_shooting(
            f,
            0.0,
            1.0,
            4.0,
            1.0,
            &ShootingConfig {
                slope0: -10.0,
                slope1: -6.0,
                ..ShootingConfig::default()
            },
        )
        .expect("shooting ok");
        let fd = solve_finite_difference(
            f,
            0.0,
            1.0,
            4.0,
            1.0,
            &FiniteDifferenceConfig {
                n_intervals: 200,
                ..FiniteDifferenceConfig::default()
            },
        )
        .expect("fd ok");
        // Compare both against the exact solution at the shared midpoint x=0.5.
        let exact_mid = 4.0 / (1.5_f64).powi(2);
        let ys_mid = shoot.y[shoot.y.len() / 2];
        let yf_mid = fd.y[fd.y.len() / 2];
        assert!((ys_mid - exact_mid).abs() < 1.0e-3, "shoot mid {ys_mid}");
        assert!((yf_mid - exact_mid).abs() < 1.0e-3, "fd mid {yf_mid}");
    }
}
