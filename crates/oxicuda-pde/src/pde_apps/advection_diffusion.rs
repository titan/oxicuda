//! Linear advection–diffusion `∂u/∂t + v·∇u = D ∇²u` by explicit finite differences.
//!
//! The advective term is discretised with first-order **upwind** differences
//! (numerically stable, monotone, no spurious oscillations) and the diffusive
//! term with the standard second-order central Laplacian. Time integration is
//! explicit (forward Euler), so the step is bounded by the combined
//! advection/diffusion stability limit
//!
//! ```text
//!   |v| dt / h  +  2 D dt / h²  ≤  1            (1-D)
//! ```
//!
//! which collapses to the pure-advection CFL `|v| dt/h ≤ 1` when `D = 0` and to
//! the pure-diffusion limit `2 D dt/h² ≤ 1` when `v = 0`.
//!
//! # Conserved / structural properties
//!
//! On a periodic torus both the upwind advection and the central diffusion sum
//! to zero by telescoping, so the discrete mass `Σ u_i · h` is conserved to
//! round-off. The upwind advection translates the discrete first moment at
//! exactly the physical speed `v`, and the central diffusion grows the second
//! central moment at exactly the rate `2 D` (the analytic variance law
//! `σ²(t) = σ₀² + 2 D t`).
//!
//! # Cell Péclet number
//!
//! The cell (grid) Péclet number `Pe_h = |v| h / D` characterises the local
//! balance of advection to diffusion; `Pe_h ≲ 2` is the classical regime in
//! which central advection would stay oscillation-free — upwind removes that
//! restriction at the cost of `O(h)` numerical diffusion. It is exposed via
//! [`AdvectionDiffusion1d::cell_peclet`] / [`AdvectionDiffusion2d::cell_peclet`].
//!
//! Reference: LeVeque, *Finite Volume Methods for Hyperbolic Problems*, CUP 2002,
//! and Morton, *Numerical Solution of Convection–Diffusion Problems*, 1996.

use crate::bc::periodic::wrap_index;
use crate::error::{PdeError, PdeResult};

/// Relative tolerance applied to the combined CFL/diffusion stability bound.
const STABILITY_TOL: f64 = 1.0e-12;

/// Boundary condition for the 1-D advection–diffusion solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdvDiffBoundary {
    /// Periodic torus: node `−1` wraps to `n−1` and node `n` wraps to `0`.
    Periodic,
    /// Fixed endpoint values `u[0] = left`, `u[n−1] = right` for all time.
    Dirichlet {
        /// Value clamped at the left endpoint.
        left: f64,
        /// Value clamped at the right endpoint.
        right: f64,
    },
}

/// Explicit upwind/central solver for 1-D advection–diffusion.
#[derive(Debug, Clone)]
pub struct AdvectionDiffusion1d {
    /// Advection velocity `v` (may be negative).
    pub velocity: f64,
    /// Diffusivity `D ≥ 0`.
    pub diffusivity: f64,
    /// Uniform grid spacing `h > 0`.
    pub dx: f64,
    /// Number of grid nodes (`n ≥ 3`).
    pub n: usize,
    /// Boundary condition.
    pub boundary: AdvDiffBoundary,
}

impl AdvectionDiffusion1d {
    /// Build a solver, validating finiteness, `D ≥ 0`, `dx > 0`, `n ≥ 3`.
    pub fn new(
        velocity: f64,
        diffusivity: f64,
        dx: f64,
        n: usize,
        boundary: AdvDiffBoundary,
    ) -> PdeResult<Self> {
        if !velocity.is_finite() {
            return Err(PdeError::InvalidParameter {
                name: "velocity".into(),
                reason: format!("must be finite, got {velocity}"),
            });
        }
        if !(diffusivity.is_finite() && diffusivity >= 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "diffusivity".into(),
                reason: format!("must be finite and >= 0, got {diffusivity}"),
            });
        }
        if !(dx.is_finite() && dx > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dx".into(),
                reason: format!("grid spacing must be finite and > 0, got {dx}"),
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "advection-diffusion requires n >= 3, got {n}"
            )));
        }
        if let AdvDiffBoundary::Dirichlet { left, right } = boundary {
            if !(left.is_finite() && right.is_finite()) {
                return Err(PdeError::InvalidParameter {
                    name: "boundary".into(),
                    reason: "Dirichlet values must be finite".into(),
                });
            }
        }
        Ok(Self {
            velocity,
            diffusivity,
            dx,
            n,
            boundary,
        })
    }

    /// Cell Péclet number `Pe_h = |v| h / D` (`+∞` when `D = 0`).
    #[must_use]
    pub fn cell_peclet(&self) -> f64 {
        if self.diffusivity > 0.0 {
            self.velocity.abs() * self.dx / self.diffusivity
        } else {
            f64::INFINITY
        }
    }

    /// Largest explicit time step from `|v| dt/h + 2 D dt/h² ≤ 1`.
    ///
    /// Returns `+∞` when both `v = 0` and `D = 0` (no evolution).
    #[must_use]
    pub fn stable_dt_max(&self) -> f64 {
        let rate = self.velocity.abs() / self.dx + 2.0 * self.diffusivity / (self.dx * self.dx);
        if rate > 0.0 {
            1.0 / rate
        } else {
            f64::INFINITY
        }
    }

    /// Reject a step that violates the combined stability bound.
    fn check_step(&self, dt: f64) -> PdeResult<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("time step must be finite and > 0, got {dt}"),
            });
        }
        let dt_max = self.stable_dt_max();
        if dt > dt_max * (1.0 + STABILITY_TOL) {
            return Err(PdeError::CflViolation { dt, dt_max });
        }
        Ok(())
    }

    fn check_field(&self, u: &[f64]) -> PdeResult<()> {
        if u.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![u.len()],
            });
        }
        Ok(())
    }

    /// Advance the field `u` in place by one explicit step of size `dt`.
    pub fn step(&self, u: &mut [f64], dt: f64) -> PdeResult<()> {
        self.check_field(u)?;
        self.check_step(dt)?;
        let n = self.n;
        let v = self.velocity;
        let d = self.diffusivity;
        let inv_h = 1.0 / self.dx;
        let inv_h2 = inv_h * inv_h;
        let mut next = vec![0.0; n];
        match self.boundary {
            AdvDiffBoundary::Periodic => {
                for (i, next_i) in next.iter_mut().enumerate() {
                    let im = wrap_index(i as isize - 1, n);
                    let ip = wrap_index(i as isize + 1, n);
                    // Upwind first derivative: backward for v>=0, forward for v<0.
                    let du = if v >= 0.0 { u[i] - u[im] } else { u[ip] - u[i] };
                    let advection = v * du * inv_h;
                    let diffusion = d * (u[ip] - 2.0 * u[i] + u[im]) * inv_h2;
                    *next_i = u[i] + dt * (diffusion - advection);
                }
            }
            AdvDiffBoundary::Dirichlet { left, right } => {
                next[0] = left;
                next[n - 1] = right;
                for i in 1..n - 1 {
                    let du = if v >= 0.0 {
                        u[i] - u[i - 1]
                    } else {
                        u[i + 1] - u[i]
                    };
                    let advection = v * du * inv_h;
                    let diffusion = d * (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_h2;
                    next[i] = u[i] + dt * (diffusion - advection);
                }
            }
        }
        u.copy_from_slice(&next);
        Ok(())
    }

    /// Integrate `n_steps` explicit steps from `u0`, returning the final field.
    pub fn solve(&self, u0: &[f64], dt: f64, n_steps: usize) -> PdeResult<Vec<f64>> {
        self.check_field(u0)?;
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut u = u0.to_vec();
        for _ in 0..n_steps {
            self.step(&mut u, dt)?;
        }
        if u.iter().any(|x| !x.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "advection-diffusion solution diverged to non-finite values".into(),
            ));
        }
        Ok(u)
    }

    /// Discrete mass `Σ u_i · h`.
    #[must_use]
    pub fn total_mass(&self, u: &[f64]) -> f64 {
        u.iter().sum::<f64>() * self.dx
    }
}

/// Boundary condition for the 2-D advection–diffusion solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdvDiffBoundary2d {
    /// Doubly-periodic torus on both axes.
    Periodic,
    /// Homogeneous Dirichlet (`u = 0`) on every boundary node.
    DirichletZero,
}

/// Explicit upwind/central solver for 2-D advection–diffusion on a row-major
/// (`i·ny + j`) structured grid.
#[derive(Debug, Clone)]
pub struct AdvectionDiffusion2d {
    /// Advection velocity `(vx, vy)`.
    pub velocity: (f64, f64),
    /// Diffusivity `D ≥ 0`.
    pub diffusivity: f64,
    /// Grid spacing `(dx, dy)`.
    pub spacing: (f64, f64),
    /// Grid resolution `(nx, ny)`.
    pub grid: (usize, usize),
    /// Boundary condition.
    pub boundary: AdvDiffBoundary2d,
}

impl AdvectionDiffusion2d {
    /// Build a 2-D solver, validating finiteness, `D ≥ 0`, spacings `> 0`,
    /// and `nx, ny ≥ 3`.
    pub fn new(
        velocity: (f64, f64),
        diffusivity: f64,
        spacing: (f64, f64),
        grid: (usize, usize),
        boundary: AdvDiffBoundary2d,
    ) -> PdeResult<Self> {
        let (vx, vy) = velocity;
        let (dx, dy) = spacing;
        let (nx, ny) = grid;
        if !(vx.is_finite() && vy.is_finite()) {
            return Err(PdeError::InvalidParameter {
                name: "velocity".into(),
                reason: "components must be finite".into(),
            });
        }
        if !(diffusivity.is_finite() && diffusivity >= 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "diffusivity".into(),
                reason: format!("must be finite and >= 0, got {diffusivity}"),
            });
        }
        if !(dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "spacing".into(),
                reason: format!("must be finite and > 0, got ({dx}, {dy})"),
            });
        }
        if nx < 3 || ny < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "advection-diffusion 2d requires nx>=3 ny>=3, got ({nx}, {ny})"
            )));
        }
        Ok(Self {
            velocity,
            diffusivity,
            spacing,
            grid,
            boundary,
        })
    }

    /// Cell Péclet number `max(|vx| dx, |vy| dy) / D` (`+∞` when `D = 0`).
    #[must_use]
    pub fn cell_peclet(&self) -> f64 {
        let (vx, vy) = self.velocity;
        let (dx, dy) = self.spacing;
        if self.diffusivity > 0.0 {
            (vx.abs() * dx).max(vy.abs() * dy) / self.diffusivity
        } else {
            f64::INFINITY
        }
    }

    /// Largest explicit step from the 2-D advection/diffusion stability bound.
    #[must_use]
    pub fn stable_dt_max(&self) -> f64 {
        let (vx, vy) = self.velocity;
        let (dx, dy) = self.spacing;
        let rate = vx.abs() / dx
            + vy.abs() / dy
            + 2.0 * self.diffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy));
        if rate > 0.0 {
            1.0 / rate
        } else {
            f64::INFINITY
        }
    }

    fn check_step(&self, dt: f64) -> PdeResult<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("time step must be finite and > 0, got {dt}"),
            });
        }
        let dt_max = self.stable_dt_max();
        if dt > dt_max * (1.0 + STABILITY_TOL) {
            return Err(PdeError::CflViolation { dt, dt_max });
        }
        Ok(())
    }

    fn check_field(&self, u: &[f64]) -> PdeResult<()> {
        let (nx, ny) = self.grid;
        if u.len() != nx * ny {
            return Err(PdeError::ShapeMismatch {
                expected: vec![nx * ny],
                got: vec![u.len()],
            });
        }
        Ok(())
    }

    /// Advance the row-major field `u` in place by one explicit step.
    pub fn step(&self, u: &mut [f64], dt: f64) -> PdeResult<()> {
        self.check_field(u)?;
        self.check_step(dt)?;
        let (vx, vy) = self.velocity;
        let (dx, dy) = self.spacing;
        let (nx, ny) = self.grid;
        let d = self.diffusivity;
        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;
        let inv_dx2 = inv_dx * inv_dx;
        let inv_dy2 = inv_dy * inv_dy;
        let mut next = vec![0.0; nx * ny];
        let periodic = matches!(self.boundary, AdvDiffBoundary2d::Periodic);
        for i in 0..nx {
            let interior_i = i > 0 && i < nx - 1;
            for j in 0..ny {
                let idx = i * ny + j;
                let interior = periodic || (interior_i && j > 0 && j < ny - 1);
                if !interior {
                    next[idx] = 0.0; // homogeneous Dirichlet boundary
                    continue;
                }
                let (im, ip, jm, jp) = if periodic {
                    (
                        wrap_index(i as isize - 1, nx),
                        wrap_index(i as isize + 1, nx),
                        wrap_index(j as isize - 1, ny),
                        wrap_index(j as isize + 1, ny),
                    )
                } else {
                    (i - 1, i + 1, j - 1, j + 1)
                };
                let c = u[idx];
                let du_x = if vx >= 0.0 {
                    c - u[im * ny + j]
                } else {
                    u[ip * ny + j] - c
                };
                let du_y = if vy >= 0.0 {
                    c - u[i * ny + jm]
                } else {
                    u[i * ny + jp] - c
                };
                let advection = vx * du_x * inv_dx + vy * du_y * inv_dy;
                let diffusion = d
                    * ((u[ip * ny + j] - 2.0 * c + u[im * ny + j]) * inv_dx2
                        + (u[i * ny + jp] - 2.0 * c + u[i * ny + jm]) * inv_dy2);
                next[idx] = c + dt * (diffusion - advection);
            }
        }
        u.copy_from_slice(&next);
        Ok(())
    }

    /// Integrate `n_steps` explicit steps from `u0`, returning the final field.
    pub fn solve(&self, u0: &[f64], dt: f64, n_steps: usize) -> PdeResult<Vec<f64>> {
        self.check_field(u0)?;
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut u = u0.to_vec();
        for _ in 0..n_steps {
            self.step(&mut u, dt)?;
        }
        if u.iter().any(|x| !x.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "advection-diffusion 2d solution diverged to non-finite values".into(),
            ));
        }
        Ok(u)
    }

    /// Discrete mass `Σ u · dx · dy`.
    #[must_use]
    pub fn total_mass(&self, u: &[f64]) -> f64 {
        let (dx, dy) = self.spacing;
        u.iter().sum::<f64>() * dx * dy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First moment (centroid) of a non-negative profile over node coordinates.
    fn centroid(u: &[f64], dx: f64) -> f64 {
        let mass: f64 = u.iter().sum();
        let m1: f64 = u.iter().enumerate().map(|(i, &v)| i as f64 * dx * v).sum();
        m1 / mass
    }

    /// Second central moment (variance) of a non-negative profile.
    fn variance(u: &[f64], dx: f64) -> f64 {
        let mass: f64 = u.iter().sum();
        let mean = centroid(u, dx);
        u.iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = i as f64 * dx;
                (x - mean) * (x - mean) * v
            })
            .sum::<f64>()
            / mass
    }

    #[test]
    fn pure_advection_translates_profile_one_cell_per_step() {
        // D = 0, Courant = 1: upwind reduces to an exact one-cell shift, so the
        // peak translates by exactly v·t (well within a cell).
        let n = 64;
        let dx = 1.0 / n as f64;
        let v = 1.0;
        let solver =
            AdvectionDiffusion1d::new(v, 0.0, dx, n, AdvDiffBoundary::Periodic).expect("solver");
        let dt = dx / v; // Courant = 1
        let center = n / 4;
        let mut u0 = vec![0.0; n];
        for (i, val) in u0.iter_mut().enumerate() {
            let d = (i as f64 - center as f64) / 4.0;
            *val = (-d * d).exp();
        }
        let steps = 12usize;
        let u = solver.solve(&u0, dt, steps).expect("solve");
        let argmax = |w: &[f64]| {
            w.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(im, m), (k, &x)| {
                    if x > m { (k, x) } else { (im, m) }
                })
                .0
        };
        let peak = argmax(&u);
        let expected = center + steps; // v·t / dx = steps cells
        assert!(
            (peak as isize - expected as isize).abs() <= 1,
            "peak at {peak}, expected ~{expected}"
        );
        // Exact shift preserves the amplitude.
        let max_amp = u.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!((max_amp - 1.0).abs() < 1.0e-9, "amplitude {max_amp}");
    }

    #[test]
    fn upwind_centroid_moves_at_speed_v() {
        // Genuine upwind (Courant 0.5) smears the pulse but advects the centroid
        // (discrete first moment) at exactly the physical speed v.
        let n = 200;
        let dx = 1.0 / n as f64;
        let v = 0.7;
        let solver =
            AdvectionDiffusion1d::new(v, 0.0, dx, n, AdvDiffBoundary::Periodic).expect("solver");
        let dt = 0.5 * dx / v;
        let center = n / 4;
        let u0: Vec<f64> = (0..n)
            .map(|i| {
                let d = (i as f64 - center as f64) / 6.0;
                (-d * d).exp()
            })
            .collect();
        let c0 = centroid(&u0, dx);
        let steps = 40usize;
        let u = solver.solve(&u0, dt, steps).expect("solve");
        let c1 = centroid(&u, dx);
        let expected = v * dt * steps as f64;
        assert!(
            ((c1 - c0) - expected).abs() < dx,
            "centroid moved {}, expected {expected}",
            c1 - c0
        );
    }

    #[test]
    fn pure_diffusion_variance_grows_as_two_d_t() {
        // v = 0: the central Laplacian grows the second central moment at exactly
        // the analytic rate dσ²/dt = 2D (variance law σ²=σ₀²+2Dt).
        let n = 120;
        let length = 12.0;
        let dx = length / n as f64;
        let diff = 0.1;
        let solver =
            AdvectionDiffusion1d::new(0.0, diff, dx, n, AdvDiffBoundary::Periodic).expect("solver");
        let dt = 0.4 * dx * dx / (2.0 * diff);
        let center_x = 0.5 * length;
        let sigma0 = 0.6;
        let u0: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx - center_x;
                (-x * x / (2.0 * sigma0 * sigma0)).exp()
            })
            .collect();
        let var0 = variance(&u0, dx);
        let steps = 150usize;
        let t = dt * steps as f64;
        let u = solver.solve(&u0, dt, steps).expect("solve");
        let var1 = variance(&u, dx);
        let rate = (var1 - var0) / t;
        assert!(
            (rate - 2.0 * diff).abs() / (2.0 * diff) < 0.02,
            "variance growth rate {rate}, expected {}",
            2.0 * diff
        );
    }

    #[test]
    fn periodic_mass_is_conserved() {
        // Both upwind advection and central diffusion telescope to zero on a
        // torus, so total mass is conserved to round-off.
        let n = 96;
        let dx = 1.0 / n as f64;
        let solver =
            AdvectionDiffusion1d::new(0.8, 0.05, dx, n, AdvDiffBoundary::Periodic).expect("solver");
        let u0: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).sin())
            .collect();
        let m0 = solver.total_mass(&u0);
        let dt = 0.5 * solver.stable_dt_max();
        let u = solver.solve(&u0, dt, 300).expect("solve");
        let m1 = solver.total_mass(&u);
        assert!((m1 - m0).abs() < 1.0e-10, "mass drift {}", (m1 - m0).abs());
    }

    #[test]
    fn stability_limit_rejects_large_step() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let solver =
            AdvectionDiffusion1d::new(1.0, 0.1, dx, n, AdvDiffBoundary::Periodic).expect("solver");
        let dt_bad = 2.0 * solver.stable_dt_max();
        assert!(matches!(
            solver.step(&mut vec![0.0; n], dt_bad),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn cell_peclet_reported() {
        let n = 20;
        let solver =
            AdvectionDiffusion1d::new(2.0, 0.5, 0.1, n, AdvDiffBoundary::Periodic).expect("solver");
        // Pe = |v| h / D = 2 * 0.1 / 0.5 = 0.4
        assert!((solver.cell_peclet() - 0.4).abs() < 1.0e-12);
        let pure =
            AdvectionDiffusion1d::new(2.0, 0.0, 0.1, n, AdvDiffBoundary::Periodic).expect("solver");
        assert!(pure.cell_peclet().is_infinite());
    }

    #[test]
    fn dirichlet_diffusion_relaxes_to_endpoints() {
        // Pure diffusion with fixed ends relaxes towards the linear steady state.
        let n = 41;
        let dx = 1.0 / (n - 1) as f64;
        let solver = AdvectionDiffusion1d::new(
            0.0,
            1.0,
            dx,
            n,
            AdvDiffBoundary::Dirichlet {
                left: 0.0,
                right: 1.0,
            },
        )
        .expect("solver");
        let u0 = vec![0.0; n];
        let dt = 0.4 * solver.stable_dt_max();
        let u = solver.solve(&u0, dt, 4000).expect("solve");
        for (i, &ui) in u.iter().enumerate() {
            let exact = i as f64 * dx; // steady linear profile 0 -> 1
            assert!((ui - exact).abs() < 0.02, "node {i}: {ui} vs {exact}");
        }
    }

    #[test]
    fn solver_construction_validates_inputs() {
        assert!(
            AdvectionDiffusion1d::new(f64::NAN, 0.1, 0.1, 8, AdvDiffBoundary::Periodic).is_err()
        );
        assert!(AdvectionDiffusion1d::new(1.0, -0.1, 0.1, 8, AdvDiffBoundary::Periodic).is_err());
        assert!(AdvectionDiffusion1d::new(1.0, 0.1, 0.0, 8, AdvDiffBoundary::Periodic).is_err());
        assert!(AdvectionDiffusion1d::new(1.0, 0.1, 0.1, 2, AdvDiffBoundary::Periodic).is_err());
    }

    #[test]
    fn two_d_periodic_mass_conserved() {
        let nx = 24;
        let ny = 24;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let solver = AdvectionDiffusion2d::new(
            (0.6, -0.4),
            0.03,
            (dx, dy),
            (nx, ny),
            AdvDiffBoundary2d::Periodic,
        )
        .expect("solver");
        let mut u0 = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                u0[i * ny + j] = 1.0
                    + 0.3 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            }
        }
        let m0 = solver.total_mass(&u0);
        let dt = 0.5 * solver.stable_dt_max();
        let u = solver.solve(&u0, dt, 120).expect("solve");
        let m1 = solver.total_mass(&u);
        assert!(
            (m1 - m0).abs() < 1.0e-10,
            "2d mass drift {}",
            (m1 - m0).abs()
        );
        assert!(u.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn two_d_pure_x_advection_translates() {
        // vy=0, D=0, Courant_x = 1: the field shifts one cell in x per step.
        let nx = 32;
        let ny = 8;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let vx = 1.0;
        let solver = AdvectionDiffusion2d::new(
            (vx, 0.0),
            0.0,
            (dx, dy),
            (nx, ny),
            AdvDiffBoundary2d::Periodic,
        )
        .expect("solver");
        let mut u0 = vec![0.0; nx * ny];
        let ci = nx / 4;
        for i in 0..nx {
            let d = (i as f64 - ci as f64) / 3.0;
            let val = (-d * d).exp();
            for j in 0..ny {
                u0[i * ny + j] = val;
            }
        }
        let dt = dx / vx;
        let steps = 6usize;
        let u = solver.solve(&u0, dt, steps).expect("solve");
        // Row j=0: peak should be at ci + steps.
        let row: Vec<f64> = (0..nx).map(|i| u[i * ny]).collect();
        let peak = row
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(im, m), (k, &x)| {
                if x > m { (k, x) } else { (im, m) }
            })
            .0;
        assert!(
            (peak as isize - (ci + steps) as isize).abs() <= 1,
            "x-peak {peak}, expected ~{}",
            ci + steps
        );
    }

    #[test]
    fn two_d_dirichlet_zero_decays() {
        let nx = 20;
        let ny = 20;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;
        let solver = AdvectionDiffusion2d::new(
            (0.0, 0.0),
            1.0,
            (dx, dy),
            (nx, ny),
            AdvDiffBoundary2d::DirichletZero,
        )
        .expect("solver");
        let mut u0 = vec![0.0; nx * ny];
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                u0[i * ny + j] = 1.0;
            }
        }
        let e0: f64 = u0.iter().map(|v| v * v).sum();
        let dt = 0.4 * solver.stable_dt_max();
        let u = solver.solve(&u0, dt, 200).expect("solve");
        let e1: f64 = u.iter().map(|v| v * v).sum();
        assert!(e1 < e0, "energy should decay: {e0} -> {e1}");
        assert!(u.iter().all(|x| x.is_finite()));
    }
}
