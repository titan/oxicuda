//! Fifth-order WENO (Jiang–Shu 1996) finite-volume scheme for 1D advection.
//!
//! Solves the linear advection equation
//!
//! ```text
//!     u_t + a u_x = 0,   x ∈ [0, L),   periodic,
//! ```
//!
//! by a conservative finite-volume method. Cell averages `ū_i` are evolved; at
//! each interface the **left** (and, by symmetry, right) reconstructed state is
//! built from a weighted combination of three candidate third-order stencils
//! whose nonlinear weights are driven by the Jiang–Shu smoothness indicators.
//! The numerical flux is the upwind / local Lax–Friedrichs flux, and time is
//! advanced with the third-order strong-stability-preserving Runge–Kutta scheme
//! of Shu & Osher (SSP-RK3).
//!
//! # WENO5 reconstruction
//!
//! For the left-biased reconstruction `u_{i+1/2}^-` the three candidate stencils
//! (using cell averages) are
//!
//! ```text
//!   p0 =  (1/3)  ū_{i-2} − (7/6) ū_{i-1} + (11/6) ū_i
//!   p1 = −(1/6)  ū_{i-1} + (5/6) ū_i     + (1/3)  ū_{i+1}
//!   p2 =  (1/3)  ū_i     + (5/6) ū_{i+1} − (1/6)  ū_{i+2}
//! ```
//!
//! with **ideal linear weights** `d = (1/10, 6/10, 3/10)` reproducing the
//! 5th-order central reconstruction on smooth data. The Jiang–Shu smoothness
//! indicators are
//!
//! ```text
//!   β0 = 13/12 (ū_{i-2} − 2ū_{i-1} + ū_i)²   + 1/4 (ū_{i-2} − 4ū_{i-1} + 3ū_i)²
//!   β1 = 13/12 (ū_{i-1} − 2ū_i + ū_{i+1})²   + 1/4 (ū_{i-1} − ū_{i+1})²
//!   β2 = 13/12 (ū_i − 2ū_{i+1} + ū_{i+2})²   + 1/4 (3ū_i − 4ū_{i+1} + ū_{i+2})²
//! ```
//!
//! and the nonlinear weights `ω_k = α_k / Σ α`, `α_k = d_k / (ε + β_k)²` with a
//! small `ε` (here `1e-6`) avoiding division by zero. On smooth data `ω_k → d_k`
//! (5th order); near a discontinuity the weight on the rough stencil collapses,
//! giving the essentially-non-oscillatory property.

use crate::error::{PdeError, PdeResult};

/// Ideal linear weights `d = (1/10, 6/10, 3/10)` for left-biased WENO5.
pub const WENO5_IDEAL_WEIGHTS: [f64; 3] = [0.1, 0.6, 0.3];

/// Jiang–Shu regularisation `ε` preventing zero denominators.
pub const WENO5_EPS: f64 = 1.0e-6;

/// The three nonlinear WENO5 weights `ω = (ω0, ω1, ω2)` for a left-biased
/// reconstruction, given the five cell averages straddling interface `i+1/2`.
///
/// `um2..up2` are `ū_{i-2}, ū_{i-1}, ū_i, ū_{i+1}, ū_{i+2}`.
///
/// The returned weights are nonnegative and **sum to 1**.
#[must_use]
pub fn weno5_weights(um2: f64, um1: f64, u0: f64, up1: f64, up2: f64) -> [f64; 3] {
    // Jiang–Shu smoothness indicators.
    let c = 13.0 / 12.0;
    let beta0 = c * (um2 - 2.0 * um1 + u0).powi(2) + 0.25 * (um2 - 4.0 * um1 + 3.0 * u0).powi(2);
    let beta1 = c * (um1 - 2.0 * u0 + up1).powi(2) + 0.25 * (um1 - up1).powi(2);
    let beta2 = c * (u0 - 2.0 * up1 + up2).powi(2) + 0.25 * (3.0 * u0 - 4.0 * up1 + up2).powi(2);

    let a0 = WENO5_IDEAL_WEIGHTS[0] / (WENO5_EPS + beta0).powi(2);
    let a1 = WENO5_IDEAL_WEIGHTS[1] / (WENO5_EPS + beta1).powi(2);
    let a2 = WENO5_IDEAL_WEIGHTS[2] / (WENO5_EPS + beta2).powi(2);
    let sum = a0 + a1 + a2;
    [a0 / sum, a1 / sum, a2 / sum]
}

/// Left-biased WENO5 reconstruction of the interface value `u_{i+1/2}^-`.
///
/// `um2..up2` are `ū_{i-2} … ū_{i+2}`. Combines the three candidate stencils
/// with the nonlinear weights from [`weno5_weights`].
#[must_use]
pub fn weno5_reconstruct_left(um2: f64, um1: f64, u0: f64, up1: f64, up2: f64) -> f64 {
    let w = weno5_weights(um2, um1, u0, up1, up2);
    let p0 = (1.0 / 3.0) * um2 - (7.0 / 6.0) * um1 + (11.0 / 6.0) * u0;
    let p1 = -(1.0 / 6.0) * um1 + (5.0 / 6.0) * u0 + (1.0 / 3.0) * up1;
    let p2 = (1.0 / 3.0) * u0 + (5.0 / 6.0) * up1 - (1.0 / 6.0) * up2;
    w[0] * p0 + w[1] * p1 + w[2] * p2
}

/// A periodic 1D WENO5 advection solver for `u_t + a u_x = 0`.
#[derive(Debug, Clone)]
pub struct Weno5Advection {
    /// Number of finite-volume cells.
    pub n: usize,
    /// Domain length `L` (periodic over `[0, L)`).
    pub length: f64,
    /// Constant advection speed `a`.
    pub speed: f64,
    /// Cell width `dx = L / n`.
    pub dx: f64,
}

impl Weno5Advection {
    /// Construct a WENO5 advection solver on `n` periodic cells over `[0, length)`.
    ///
    /// # Errors
    /// * [`PdeError::InvalidGrid`] if `n < 5` (WENO5 needs a 5-cell stencil) or
    ///   `length <= 0`.
    pub fn new(n: usize, length: f64, speed: f64) -> PdeResult<Self> {
        if n < 5 {
            return Err(PdeError::InvalidGrid(
                "weno5: need at least 5 cells for the stencil".into(),
            ));
        }
        if length <= 0.0 {
            return Err(PdeError::InvalidGrid("weno5: length must be > 0".into()));
        }
        Ok(Self {
            n,
            length,
            speed,
            dx: length / n as f64,
        })
    }

    /// Cell-centre coordinate of cell `i` (`x_i = (i + ½) dx`).
    #[must_use]
    pub fn cell_center(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.dx
    }

    /// All cell-centre coordinates.
    #[must_use]
    pub fn cell_centers(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.cell_center(i)).collect()
    }

    /// Periodic index helper (wraps `i` into `[0, n)`).
    fn wrap(&self, i: isize) -> usize {
        let n = self.n as isize;
        (((i % n) + n) % n) as usize
    }

    /// Compute the conservative spatial residual `L(u) = −a u_x` discretised as
    /// `−(F_{i+1/2} − F_{i-1/2}) / dx`, where `F` is the upwind WENO5 flux.
    ///
    /// For `a ≥ 0` the upwind flux uses the **left**-biased reconstruction at
    /// each interface; for `a < 0` it uses the mirror (right-biased) stencil,
    /// obtained by reflecting the cell-average window.
    #[must_use]
    pub fn spatial_residual(&self, u: &[f64]) -> Vec<f64> {
        let n = self.n;
        // Interface flux F_{i+1/2} for i = 0..n (with F_{n-1/2} = F_{-1/2} by
        // periodicity). We store flux[i] = F_{i+1/2}.
        let mut flux = vec![0.0; n];
        for i in 0..n {
            // Reconstruct the upwind state at interface i+1/2.
            let face = if self.speed >= 0.0 {
                // Left-biased: window centred so cell i is the upwind cell.
                let um2 = u[self.wrap(i as isize - 2)];
                let um1 = u[self.wrap(i as isize - 1)];
                let u0 = u[self.wrap(i as isize)];
                let up1 = u[self.wrap(i as isize + 1)];
                let up2 = u[self.wrap(i as isize + 2)];
                weno5_reconstruct_left(um2, um1, u0, up1, up2)
            } else {
                // Right-biased: reflect the window about the interface. The state
                // u_{i+1/2}^+ is the left reconstruction of the mirrored averages
                // centred on cell i+1.
                let um2 = u[self.wrap(i as isize + 3)];
                let um1 = u[self.wrap(i as isize + 2)];
                let u0 = u[self.wrap(i as isize + 1)];
                let up1 = u[self.wrap(i as isize)];
                let up2 = u[self.wrap(i as isize - 1)];
                weno5_reconstruct_left(um2, um1, u0, up1, up2)
            };
            flux[i] = self.speed * face;
        }
        // Residual: −(F_{i+1/2} − F_{i-1/2}) / dx.
        let mut res = vec![0.0; n];
        for i in 0..n {
            let f_plus = flux[i];
            let f_minus = flux[self.wrap(i as isize - 1)];
            res[i] = -(f_plus - f_minus) / self.dx;
        }
        res
    }

    /// Advance the solution by one SSP-RK3 (Shu–Osher) step of size `dt`.
    ///
    /// ```text
    ///   u^(1) = u^n + dt L(u^n)
    ///   u^(2) = ¾ u^n + ¼ (u^(1) + dt L(u^(1)))
    ///   u^{n+1} = ⅓ u^n + ⅔ (u^(2) + dt L(u^(2)))
    /// ```
    ///
    /// # Errors
    /// * [`PdeError::DimensionMismatch`] if `u.len() != n`.
    /// * [`PdeError::CflViolation`] if the CFL number `|a| dt / dx` exceeds `1`.
    pub fn step(&self, u: &[f64], dt: f64) -> PdeResult<Vec<f64>> {
        if u.len() != self.n {
            return Err(PdeError::DimensionMismatch {
                a: u.len(),
                b: self.n,
            });
        }
        let cfl = self.speed.abs() * dt / self.dx;
        if cfl > 1.0 + 1.0e-12 {
            return Err(PdeError::CflViolation {
                dt,
                dt_max: self.dx / self.speed.abs().max(1.0e-300),
            });
        }
        let l0 = self.spatial_residual(u);
        let u1: Vec<f64> = (0..self.n).map(|i| u[i] + dt * l0[i]).collect();

        let l1 = self.spatial_residual(&u1);
        let u2: Vec<f64> = (0..self.n)
            .map(|i| 0.75 * u[i] + 0.25 * (u1[i] + dt * l1[i]))
            .collect();

        let l2 = self.spatial_residual(&u2);
        let un: Vec<f64> = (0..self.n)
            .map(|i| (1.0 / 3.0) * u[i] + (2.0 / 3.0) * (u2[i] + dt * l2[i]))
            .collect();
        Ok(un)
    }

    /// Integrate from `u0` to final time `t_final` with CFL number `cfl_target`.
    ///
    /// The time step is `dt = cfl_target · dx / |a|`, adjusted on the final step
    /// to land exactly on `t_final`.
    ///
    /// # Errors
    /// As [`Weno5Advection::step`]; also [`PdeError::InvalidParameter`] if
    /// `cfl_target ∉ (0, 1]` or `t_final < 0`.
    pub fn integrate(&self, u0: &[f64], t_final: f64, cfl_target: f64) -> PdeResult<Vec<f64>> {
        if !(cfl_target > 0.0 && cfl_target <= 1.0) {
            return Err(PdeError::InvalidParameter {
                name: "cfl_target".into(),
                reason: "must lie in (0, 1]".into(),
            });
        }
        if t_final < 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "t_final".into(),
                reason: "must be non-negative".into(),
            });
        }
        if u0.len() != self.n {
            return Err(PdeError::DimensionMismatch {
                a: u0.len(),
                b: self.n,
            });
        }
        let speed = self.speed.abs().max(1.0e-300);
        let dt_full = cfl_target * self.dx / speed;
        let mut u = u0.to_vec();
        let mut t = 0.0;
        while t < t_final - 1.0e-14 {
            let dt = dt_full.min(t_final - t);
            u = self.step(&u, dt)?;
            t += dt;
        }
        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn weights_sum_to_one_everywhere() {
        // Random-ish samples (including steep data) all give Σ ω = 1.
        let samples = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [-3.0, 7.0, -2.0, 9.0, 0.5],
            [1e-8, 2e-8, 1.0, 5.0, -4.0],
        ];
        for s in samples {
            let w = weno5_weights(s[0], s[1], s[2], s[3], s[4]);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "Σω = {sum} for {s:?}");
            assert!(w.iter().all(|&x| x >= 0.0), "weights must be nonnegative");
        }
    }

    #[test]
    fn constant_field_reconstructs_exactly() {
        // A constant field reconstructs to the same constant at every interface.
        let recon = weno5_reconstruct_left(2.5, 2.5, 2.5, 2.5, 2.5);
        assert!((recon - 2.5).abs() < 1e-13);
    }

    #[test]
    fn smooth_field_weights_approach_ideal() {
        // On a smooth (slowly varying) field, ω_k → d_k and the reconstruction
        // is ~5th order. Sample a smooth quadratic so curvature is small per cell.
        let dx = 1e-3;
        let f = |x: f64| (x).sin();
        let xc = 0.7;
        let avg = |k: i32| {
            // Cell average ≈ point value for tiny dx (midpoint).
            f(xc + k as f64 * dx)
        };
        let w = weno5_weights(avg(-2), avg(-1), avg(0), avg(1), avg(2));
        for k in 0..3 {
            assert!(
                (w[k] - WENO5_IDEAL_WEIGHTS[k]).abs() < 1e-3,
                "ω{k} = {} vs ideal {}",
                w[k],
                WENO5_IDEAL_WEIGHTS[k]
            );
        }
    }

    #[test]
    fn reconstruction_is_fifth_order_on_smooth_data() {
        // Reconstruct u(x_{i+1/2}) for u = sin(2πx) via cell averages and check
        // the error decays at ~5th order under grid refinement.
        //
        // Exact cell average over [x_L, x_R]:
        //   ā = (1/dx) ∫ sin(2πx) dx = (cos(2πx_L) − cos(2πx_R)) / (2π dx).
        let k_wave = 2.0 * PI;
        let exact_face = |x: f64| (k_wave * x).sin();
        let cell_avg = |xl: f64, dx: f64| -> f64 {
            ((k_wave * xl).cos() - (k_wave * (xl + dx)).cos()) / (k_wave * dx)
        };

        let mut prev_err = f64::INFINITY;
        let mut rates = Vec::new();
        for &nn in &[40usize, 80, 160, 320] {
            let dx = 1.0 / nn as f64;
            // Interface at x* = 0.5 sits at i+1/2 with i = nn/2 − 1.
            let i = nn / 2 - 1;
            let xc_l = i as f64 * dx; // left edge of cell i
            let face_x = (i as f64 + 1.0) * dx; // interface i+1/2
            let avg = |k: i32| cell_avg(xc_l + k as f64 * dx, dx);
            let recon = weno5_reconstruct_left(avg(-2), avg(-1), avg(0), avg(1), avg(2));
            let err = (recon - exact_face(face_x)).abs();
            if prev_err.is_finite() && err > 0.0 {
                rates.push((prev_err / err).log2());
            }
            prev_err = err;
        }
        // Observed order should be close to 5 (allow generous slack for the
        // midpoint→average and finite-precision effects).
        let order = *rates.last().expect("rate");
        assert!(order > 4.5, "WENO5 order too low: {rates:?}");
    }

    #[test]
    fn advect_sine_one_period_preserves_shape() {
        // Advect a sine one full period; with WENO5 + SSP-RK3 the profile returns
        // to itself with small L∞ error and no spurious growth.
        let n = 100;
        let solver = Weno5Advection::new(n, 1.0, 1.0).expect("solver");
        let u0: Vec<f64> = solver
            .cell_centers()
            .iter()
            .map(|&x| (2.0 * PI * x).sin())
            .collect();
        // One period: t = L / a = 1.
        let u = solver.integrate(&u0, 1.0, 0.4).expect("integrate");
        let linf = u
            .iter()
            .zip(&u0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(linf < 5e-2, "L∞ after one period too large: {linf}");
        // No new global maximum beyond the initial amplitude (low dissipation /
        // no spurious growth).
        let max_now = u.iter().cloned().fold(f64::MIN, f64::max);
        assert!(max_now <= 1.0 + 1e-2, "spurious growth: max {max_now}");
    }

    #[test]
    fn weights_collapse_near_discontinuity() {
        // For a step (rough) stencil, the weight on the stencil crossing the jump
        // must collapse toward zero — the essentially-non-oscillatory property.
        // Step located so the FIRST stencil (uses um2,um1,u0) straddles the jump.
        let w = weno5_weights(0.0, 0.0, 1.0, 1.0, 1.0);
        // Stencil 0 spans um2=0,um1=0,u0=1 (crosses the jump) — its β is large,
        // so ω0 should be far below its ideal 0.1.
        assert!(w[0] < 0.05, "ω0 should collapse near jump, got {}", w[0]);
        // Stencils 1,2 lie in the smooth (constant) region and dominate.
        assert!(w[1] + w[2] > 0.9, "smooth stencils should dominate");
    }

    #[test]
    fn advect_step_is_essentially_non_oscillatory() {
        // Advect a step profile; WENO5 must not introduce large new extrema.
        let n = 80;
        let solver = Weno5Advection::new(n, 1.0, 1.0).expect("solver");
        let u0: Vec<f64> = solver
            .cell_centers()
            .iter()
            .map(|&x| if x > 0.5 { 1.0 } else { 0.0 })
            .collect();
        let u = solver.integrate(&u0, 0.5, 0.4).expect("integrate");
        let max_v = u.iter().cloned().fold(f64::MIN, f64::max);
        let min_v = u.iter().cloned().fold(f64::MAX, f64::min);
        // Bounded overshoot/undershoot well within a small tolerance.
        assert!(max_v < 1.0 + 0.05, "overshoot too large: {max_v}");
        assert!(min_v > -0.05, "undershoot too large: {min_v}");
    }

    #[test]
    fn cfl_violation_is_rejected() {
        let solver = Weno5Advection::new(10, 1.0, 1.0).expect("solver");
        let u = vec![0.0; 10];
        // dt = 2 * dx -> CFL = 2 > 1.
        assert!(solver.step(&u, 2.0 * solver.dx).is_err());
    }

    #[test]
    fn negative_speed_advects_rightward_state_correctly() {
        // With a < 0 the wave moves left; after one period it returns. Check
        // the mirrored (right-biased) reconstruction keeps it accurate.
        let n = 100;
        let solver = Weno5Advection::new(n, 1.0, -1.0).expect("solver");
        let u0: Vec<f64> = solver
            .cell_centers()
            .iter()
            .map(|&x| (2.0 * PI * x).sin())
            .collect();
        let u = solver.integrate(&u0, 1.0, 0.4).expect("integrate");
        let linf = u
            .iter()
            .zip(&u0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(linf < 5e-2, "negative-speed L∞ too large: {linf}");
    }

    #[test]
    fn total_mass_is_conserved() {
        // Conservative FV scheme: Σ ū_i dx is invariant in time.
        let n = 64;
        let solver = Weno5Advection::new(n, 1.0, 0.8).expect("solver");
        let u0: Vec<f64> = solver
            .cell_centers()
            .iter()
            .map(|&x| (2.0 * PI * x).sin() + 0.3)
            .collect();
        let mass0: f64 = u0.iter().sum::<f64>() * solver.dx;
        let u = solver.integrate(&u0, 0.37, 0.5).expect("integrate");
        let mass1: f64 = u.iter().sum::<f64>() * solver.dx;
        assert!(
            (mass1 - mass0).abs() < 1e-12,
            "mass drift {}",
            mass1 - mass0
        );
    }
}
