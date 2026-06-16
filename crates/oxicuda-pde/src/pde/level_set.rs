//! Level-set method for implicit interface tracking on a 2-D uniform grid.
//!
//! An interface `Γ(t)` is represented as the zero level set of a scalar field
//! `φ(x, t)` (negative inside, positive outside), with `|∇φ| = 1` for a signed
//! distance function. Three motions are supported:
//!
//! * **External advection** `∂φ/∂t + V·∇φ = 0` — first-order *upwind*
//!   finite differences (information drawn from the inflow side);
//! * **Normal-speed motion** `∂φ/∂t + F|∇φ| = 0` — the Osher–Sethian Godunov
//!   upwind Hamiltonian for `|∇φ|`;
//! * **Reinitialisation** — relax `∂φ/∂τ + sgn(φ₀)(|∇φ| − 1) = 0` to steady
//!   state (Sussman–Smereka–Osher) to restore the signed-distance property while
//!   pinning the zero level set.
//!
//! The grid is row-major `i·ny + j` (`i` along x, `j` along y) with spacings
//! `dx, dy`. Outside the rectangle a homogeneous-Neumann ghost (edge replication)
//! is used, which is exact for interfaces that stay in the interior.
//!
//! References: Osher & Sethian, *Fronts propagating with curvature-dependent
//! speed*, J. Comput. Phys. 79 (1988) 12–49; Sussman, Smereka & Osher,
//! *A level set approach for computing solutions to incompressible two-phase
//! flow*, J. Comput. Phys. 114 (1994) 146–159.

use crate::error::{PdeError, PdeResult};

/// Tolerance on the CFL number when validating a time step.
const CFL_TOL: f64 = 1.0e-12;

/// Validate a positive, finite time step.
fn check_dt(dt: f64) -> PdeResult<()> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(PdeError::InvalidParameter {
            name: "dt".into(),
            reason: format!("time step must be finite and > 0, got {dt}"),
        });
    }
    Ok(())
}

/// Osher–Sethian `|∇φ|` for an *expanding* front (effective speed `> 0`):
/// take the inflow-biased one-sided derivatives. `a,b` are `D⁻ₓ,D⁺ₓ` and `c,d`
/// are `D⁻ᵧ,D⁺ᵧ`.
#[inline]
fn godunov_grad_expand(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let gx = a.max(0.0).powi(2).max(b.min(0.0).powi(2));
    let gy = c.max(0.0).powi(2).max(d.min(0.0).powi(2));
    (gx + gy).sqrt()
}

/// Osher–Sethian `|∇φ|` for a *shrinking* front (effective speed `< 0`).
#[inline]
fn godunov_grad_shrink(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let gx = a.min(0.0).powi(2).max(b.max(0.0).powi(2));
    let gy = c.min(0.0).powi(2).max(d.max(0.0).powi(2));
    (gx + gy).sqrt()
}

/// Level-set solver on a uniform 2-D rectangular grid.
#[derive(Debug, Clone)]
pub struct LevelSet {
    /// Grid points along x (`nx ≥ 3`).
    pub nx: usize,
    /// Grid points along y (`ny ≥ 3`).
    pub ny: usize,
    /// Grid spacing along x (`dx > 0`).
    pub dx: f64,
    /// Grid spacing along y (`dy > 0`).
    pub dy: f64,
}

impl LevelSet {
    /// Build a solver. Validates `nx, ny ≥ 3` and `dx, dy > 0`.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> PdeResult<Self> {
        if nx < 3 || ny < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "level set requires nx,ny >= 3, got nx={nx} ny={ny}"
            )));
        }
        if !(dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "spacing".into(),
                reason: format!("dx,dy must be finite and > 0, got dx={dx} dy={dy}"),
            });
        }
        Ok(Self { nx, ny, dx, dy })
    }

    /// Total number of grid nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Whether the grid is empty (always `false` for a valid solver).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample `φ` with homogeneous-Neumann (edge-replicating) ghost cells.
    #[inline]
    fn sample(&self, phi: &[f64], i: isize, j: isize) -> f64 {
        let ii = i.clamp(0, self.nx as isize - 1) as usize;
        let jj = j.clamp(0, self.ny as isize - 1) as usize;
        phi[ii * self.ny + jj]
    }

    /// One-sided differences `(D⁻ₓ, D⁺ₓ, D⁻ᵧ, D⁺ᵧ)` at node `(i, j)`.
    #[inline]
    fn godunov_diffs(&self, phi: &[f64], i: usize, j: usize) -> (f64, f64, f64, f64) {
        let (ii, jj) = (i as isize, j as isize);
        let center = phi[i * self.ny + j];
        let dmx = (center - self.sample(phi, ii - 1, jj)) / self.dx;
        let dpx = (self.sample(phi, ii + 1, jj) - center) / self.dx;
        let dmy = (center - self.sample(phi, ii, jj - 1)) / self.dy;
        let dpy = (self.sample(phi, ii, jj + 1) - center) / self.dy;
        (dmx, dpx, dmy, dpy)
    }

    fn check_field(&self, f: &[f64], name: &str) -> PdeResult<()> {
        if f.len() != self.len() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.len()],
                got: vec![f.len()],
            });
        }
        if f.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(format!(
                "{name} contains non-finite values"
            )));
        }
        Ok(())
    }

    /// Advect `φ` under an external velocity field `(vel_x, vel_y)` for one upwind
    /// step `∂φ/∂t + V·∇φ = 0`.
    ///
    /// Rejects steps violating the CFL bound `dt (max|u|/dx + max|v|/dy) ≤ 1`.
    pub fn advect(&self, phi: &mut [f64], vel_x: &[f64], vel_y: &[f64], dt: f64) -> PdeResult<()> {
        self.check_field(phi, "phi")?;
        self.check_field(vel_x, "vel_x")?;
        self.check_field(vel_y, "vel_y")?;
        check_dt(dt)?;

        let umax = vel_x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let vmax = vel_y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let rate = umax / self.dx + vmax / self.dy;
        if dt * rate > 1.0 + CFL_TOL {
            let dt_max = if rate > 0.0 {
                1.0 / rate
            } else {
                f64::INFINITY
            };
            return Err(PdeError::CflViolation { dt, dt_max });
        }

        let (nx, ny) = (self.nx, self.ny);
        let mut out = phi.to_vec();
        for i in 0..nx {
            let ii = i as isize;
            for j in 0..ny {
                let jj = j as isize;
                let id = i * ny + j;
                let u = vel_x[id];
                let v = vel_y[id];
                let center = phi[id];
                let dphidx = if u > 0.0 {
                    (center - self.sample(phi, ii - 1, jj)) / self.dx
                } else {
                    (self.sample(phi, ii + 1, jj) - center) / self.dx
                };
                let dphidy = if v > 0.0 {
                    (center - self.sample(phi, ii, jj - 1)) / self.dy
                } else {
                    (self.sample(phi, ii, jj + 1) - center) / self.dy
                };
                out[id] = center - dt * (u * dphidx + v * dphidy);
            }
        }
        phi.copy_from_slice(&out);
        Ok(())
    }

    /// Move the interface in its normal direction at (possibly spatially varying)
    /// `speed` for one step `∂φ/∂t + F|∇φ| = 0` via the Osher–Sethian scheme.
    ///
    /// With the convention `φ < 0` inside, `F > 0` grows the enclosed region.
    /// Rejects steps violating `dt max|F| (1/dx + 1/dy) ≤ 1`.
    pub fn propagate_normal(&self, phi: &mut [f64], speed: &[f64], dt: f64) -> PdeResult<()> {
        self.check_field(phi, "phi")?;
        self.check_field(speed, "speed")?;
        check_dt(dt)?;

        let fmax = speed.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let rate = fmax * (1.0 / self.dx + 1.0 / self.dy);
        if dt * rate > 1.0 + CFL_TOL {
            let dt_max = if rate > 0.0 {
                1.0 / rate
            } else {
                f64::INFINITY
            };
            return Err(PdeError::CflViolation { dt, dt_max });
        }

        let (nx, ny) = (self.nx, self.ny);
        let mut out = phi.to_vec();
        for i in 0..nx {
            for j in 0..ny {
                let id = i * ny + j;
                let f = speed[id];
                if f == 0.0 {
                    continue;
                }
                let (a, b, c, d) = self.godunov_diffs(phi, i, j);
                let grad = if f > 0.0 {
                    godunov_grad_expand(a, b, c, d)
                } else {
                    godunov_grad_shrink(a, b, c, d)
                };
                out[id] = phi[id] - dt * f * grad;
            }
        }
        phi.copy_from_slice(&out);
        Ok(())
    }

    /// Reinitialise `φ` to a signed distance function (`|∇φ| → 1`) by relaxing
    /// `∂φ/∂τ + sgn(φ₀)(|∇φ| − 1) = 0` for `n_iter` pseudo-time iterations.
    ///
    /// A smoothed sign `φ₀/√(φ₀² + h²)` keeps the zero level set pinned. The
    /// pseudo-time step is the CFL-safe `Δτ = ½ min(dx, dy)`.
    pub fn reinitialize(&self, phi: &mut [f64], n_iter: usize) -> PdeResult<()> {
        self.check_field(phi, "phi")?;
        let (nx, ny) = (self.nx, self.ny);
        let eps_s = self.dx.max(self.dy);
        let eps_s2 = eps_s * eps_s;
        let dtau = 0.5 * self.dx.min(self.dy);
        let phi0 = phi.to_vec();
        let mut out = vec![0.0; self.len()];
        for _ in 0..n_iter {
            for i in 0..nx {
                for j in 0..ny {
                    let id = i * ny + j;
                    let p0 = phi0[id];
                    let sign = p0 / (p0 * p0 + eps_s2).sqrt();
                    let (a, b, c, d) = self.godunov_diffs(phi, i, j);
                    let grad = if p0 > 0.0 {
                        godunov_grad_expand(a, b, c, d)
                    } else if p0 < 0.0 {
                        godunov_grad_shrink(a, b, c, d)
                    } else {
                        0.0
                    };
                    out[id] = phi[id] - dtau * sign * (grad - 1.0);
                }
            }
            phi.copy_from_slice(&out);
        }
        if phi.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "reinitialisation diverged to non-finite values".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_circle(ls: &LevelSet, cx: f64, cy: f64, r: f64) -> Vec<f64> {
        let mut phi = vec![0.0; ls.len()];
        for i in 0..ls.nx {
            for j in 0..ls.ny {
                let x = i as f64 * ls.dx;
                let y = j as f64 * ls.dy;
                phi[i * ls.ny + j] = (x - cx).hypot(y - cy) - r;
            }
        }
        phi
    }

    /// Centroid x-coordinate of the `φ < 0` region.
    fn inside_centroid_x(ls: &LevelSet, phi: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for i in 0..ls.nx {
            for j in 0..ls.ny {
                if phi[i * ls.ny + j] < 0.0 {
                    sum += i as f64 * ls.dx;
                    count += 1.0;
                }
            }
        }
        sum / count
    }

    /// Radius along the +x axis from the grid centre via a zero crossing.
    fn radius_along_x(ls: &LevelSet, phi: &[f64], ci: usize, cj: usize) -> f64 {
        let mut prev = phi[ci * ls.ny + cj];
        for i in ci + 1..ls.nx {
            let cur = phi[i * ls.ny + cj];
            if prev <= 0.0 && cur > 0.0 {
                let frac = -prev / (cur - prev);
                let x_zero = ((i - 1) as f64 + frac) * ls.dx;
                return x_zero - ci as f64 * ls.dx;
            }
            prev = cur;
        }
        f64::NAN
    }

    #[test]
    fn circle_translates_with_constant_velocity() {
        let ls = LevelSet::new(41, 41, 0.05, 0.05).expect("solver");
        let mut phi = signed_circle(&ls, 1.0, 1.0, 0.5);
        let n = ls.len();
        let (u, v) = (0.5, 0.0);
        let vel_x = vec![u; n];
        let vel_y = vec![v; n];
        let dt = 0.05; // CFL number 0.5
        let steps = 8;
        for _ in 0..steps {
            ls.advect(&mut phi, &vel_x, &vel_y, dt).expect("advect");
        }
        let t = dt * steps as f64;
        let cx_new = inside_centroid_x(&ls, &phi);
        let expected = 1.0 + u * t;
        assert!(
            (cx_new - expected).abs() < 2.0 * ls.dx,
            "centroid x={cx_new}, expected {expected}"
        );
        assert!(phi.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn reinitialization_recovers_unit_gradient() {
        let ls = LevelSet::new(41, 41, 0.05, 0.05).expect("solver");
        let (cx, cy, r) = (1.0, 1.0, 0.5);
        // Non-distance initial data: φ = (x−cx)² + (y−cy)² − R² ⇒ |∇φ| = 2ρ.
        let mut phi = vec![0.0; ls.len()];
        for i in 0..ls.nx {
            for j in 0..ls.ny {
                let x = i as f64 * ls.dx;
                let y = j as f64 * ls.dy;
                phi[i * ls.ny + j] = (x - cx).powi(2) + (y - cy).powi(2) - r * r;
            }
        }
        ls.reinitialize(&mut phi, 40).expect("reinit");
        // Central-difference |∇φ| at a few interface-adjacent nodes ≈ 1.
        let probes = [(30usize, 20usize), (20, 30), (27, 27)];
        for (i, j) in probes {
            let gx = (phi[(i + 1) * ls.ny + j] - phi[(i - 1) * ls.ny + j]) / (2.0 * ls.dx);
            let gy = (phi[i * ls.ny + j + 1] - phi[i * ls.ny + j - 1]) / (2.0 * ls.dy);
            let mag = gx.hypot(gy);
            assert!((mag - 1.0).abs() < 0.15, "|∇φ| at ({i},{j}) = {mag}");
        }
    }

    #[test]
    fn reinitialization_preserves_sign_away_from_interface() {
        let ls = LevelSet::new(41, 41, 0.05, 0.05).expect("solver");
        let phi0 = signed_circle(&ls, 1.0, 1.0, 0.5);
        let mut phi = phi0.clone();
        ls.reinitialize(&mut phi, 25).expect("reinit");
        let band = 2.0 * ls.dx;
        for (k, &p0) in phi0.iter().enumerate() {
            if p0.abs() > band {
                assert!(
                    p0.signum() == phi[k].signum(),
                    "sign flipped at {k}: {p0} -> {}",
                    phi[k]
                );
            }
        }
    }

    #[test]
    fn normal_motion_grows_circle_at_prescribed_speed() {
        let nx = 61;
        let dx = 2.0 / (nx - 1) as f64;
        let ls = LevelSet::new(nx, nx, dx, dx).expect("solver");
        let (ci, cj) = (nx / 2, nx / 2);
        let cx = ci as f64 * dx;
        let mut phi = signed_circle(&ls, cx, cx, 0.5);
        let speed = vec![0.5; ls.len()];
        let dt = 0.02; // CFL number 0.6
        let steps = 10;
        let r0 = radius_along_x(&ls, &phi, ci, cj);
        for _ in 0..steps {
            ls.propagate_normal(&mut phi, &speed, dt).expect("normal");
        }
        let r1 = radius_along_x(&ls, &phi, ci, cj);
        let expected = r0 + 0.5 * dt * steps as f64;
        assert!(
            (r1 - expected).abs() < 2.0 * dx,
            "radius {r1}, expected {expected} (r0={r0})"
        );
    }

    #[test]
    fn normal_motion_shrinks_circle() {
        let nx = 61;
        let dx = 2.0 / (nx - 1) as f64;
        let ls = LevelSet::new(nx, nx, dx, dx).expect("solver");
        let (ci, cj) = (nx / 2, nx / 2);
        let cx = ci as f64 * dx;
        let mut phi = signed_circle(&ls, cx, cx, 0.5);
        let speed = vec![-0.5; ls.len()];
        let dt = 0.02;
        let steps = 10;
        let r0 = radius_along_x(&ls, &phi, ci, cj);
        for _ in 0..steps {
            ls.propagate_normal(&mut phi, &speed, dt).expect("normal");
        }
        let r1 = radius_along_x(&ls, &phi, ci, cj);
        let expected = r0 - 0.5 * dt * steps as f64;
        assert!(
            (r1 - expected).abs() < 2.0 * dx,
            "radius {r1}, expected {expected}"
        );
    }

    #[test]
    fn cfl_violation_is_rejected() {
        let ls = LevelSet::new(21, 21, 0.05, 0.05).expect("solver");
        let n = ls.len();
        let mut phi = vec![0.0; n];
        let vel_x = vec![1.0; n];
        let vel_y = vec![0.0; n];
        // dt·(1/0.05) = dt·20; dt = 1 ⇒ CFL 20 ≫ 1.
        assert!(matches!(
            ls.advect(&mut phi, &vel_x, &vel_y, 1.0),
            Err(PdeError::CflViolation { .. })
        ));
    }

    #[test]
    fn shape_and_grid_errors_are_rejected() {
        assert!(LevelSet::new(2, 8, 0.1, 0.1).is_err());
        assert!(LevelSet::new(8, 8, 0.0, 0.1).is_err());
        let ls = LevelSet::new(8, 8, 0.1, 0.1).expect("solver");
        let mut phi = vec![0.0; ls.len() - 1];
        let vel = vec![0.0; ls.len()];
        assert!(matches!(
            ls.advect(&mut phi, &vel, &vel, 0.01),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }
}
