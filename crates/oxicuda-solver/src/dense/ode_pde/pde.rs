//! PDE types and solvers: Heat, Wave, Poisson, Advection (1-D).

use crate::error::{SolverError, SolverResult};

use super::utils::{apply_bc_1d, solve_tridiagonal};

// =========================================================================
// PDE types
// =========================================================================

/// One-dimensional uniform grid.
#[derive(Debug, Clone)]
pub struct Grid1D {
    /// Left boundary.
    pub x_min: f64,
    /// Right boundary.
    pub x_max: f64,
    /// Number of grid points.
    pub nx: usize,
    /// Grid spacing (computed).
    pub dx: f64,
}

impl Grid1D {
    /// Create a uniform 1-D grid with `nx` points from `x_min` to `x_max`.
    pub fn new(x_min: f64, x_max: f64, nx: usize) -> Self {
        let dx = if nx > 1 {
            (x_max - x_min) / (nx - 1) as f64
        } else {
            0.0
        };
        Self {
            x_min,
            x_max,
            nx,
            dx,
        }
    }

    /// Return the coordinate of grid point `i`.
    pub fn point(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx
    }
}

/// Two-dimensional uniform grid.
#[derive(Debug, Clone)]
pub struct Grid2D {
    /// X-direction range.
    pub x_min: f64,
    /// X-direction range.
    pub x_max: f64,
    /// Y-direction range.
    pub y_min: f64,
    /// Y-direction range.
    pub y_max: f64,
    /// Number of grid points in x.
    pub nx: usize,
    /// Number of grid points in y.
    pub ny: usize,
    /// Spacing in x.
    pub dx: f64,
    /// Spacing in y.
    pub dy: f64,
}

impl Grid2D {
    /// Create a uniform 2-D grid.
    pub fn new(x_min: f64, x_max: f64, nx: usize, y_min: f64, y_max: f64, ny: usize) -> Self {
        let dx = if nx > 1 {
            (x_max - x_min) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (y_max - y_min) / (ny - 1) as f64
        } else {
            0.0
        };
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            nx,
            ny,
            dx,
            dy,
        }
    }
}

/// Boundary condition type.
#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    /// Fixed value at the boundary.
    Dirichlet(f64),
    /// Fixed derivative at the boundary.
    Neumann(f64),
    /// Periodic boundary (left = right).
    Periodic,
}

/// Configuration for a 1-D PDE solve.
#[derive(Debug, Clone)]
pub struct PdeConfig {
    /// Spatial grid.
    pub grid: Grid1D,
    /// Time step.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
    /// Left boundary condition.
    pub bc_left: BoundaryCondition,
    /// Right boundary condition.
    pub bc_right: BoundaryCondition,
}

// =========================================================================
// PDE solvers
// =========================================================================

/// 1-D heat equation solver: du/dt = alpha * d²u/dx².
pub struct HeatEquation1D {
    /// Thermal diffusivity.
    pub alpha: f64,
}

impl HeatEquation1D {
    /// Maximum stable time step for the explicit (FTCS) scheme.
    ///
    /// For stability we need dt <= dx² / (2 * alpha).
    pub fn stability_limit(&self, dx: f64) -> f64 {
        dx * dx / (2.0 * self.alpha)
    }

    /// Solve using Forward-Time Central-Space (FTCS) explicit scheme.
    pub fn solve_explicit(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "heat_explicit: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let r = self.alpha * dt / (dx * dx);

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            // Interior points
            for i in 1..nx - 1 {
                u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            }

            // Boundary conditions
            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);

            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }

    /// Solve using Crank-Nicolson (implicit) scheme.
    ///
    /// Unconditionally stable, second-order in both time and space.
    /// Reduces to a tridiagonal system at each time step.
    pub fn solve_implicit(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "heat_implicit: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "heat_implicit: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let r = self.alpha * dt / (dx * dx);

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        // Interior system size
        let m = nx - 2;

        for _ in 0..config.num_steps {
            // Build RHS from explicit half: (I + r/2 * A) * u_interior
            let mut rhs = vec![0.0; m];
            for (i, rhs_i) in rhs.iter_mut().enumerate() {
                let idx = i + 1; // grid index
                *rhs_i = u[idx] + 0.5 * r * (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]);
            }

            // Add boundary contributions
            match config.bc_left {
                BoundaryCondition::Dirichlet(val) => {
                    rhs[0] += 0.5 * r * val;
                }
                BoundaryCondition::Neumann(_) | BoundaryCondition::Periodic => {}
            }
            match config.bc_right {
                BoundaryCondition::Dirichlet(val) => {
                    if m > 0 {
                        rhs[m - 1] += 0.5 * r * val;
                    }
                }
                BoundaryCondition::Neumann(_) | BoundaryCondition::Periodic => {}
            }

            // Tridiagonal system: (I - r/2 * A) * u_new = rhs
            // sub-diag: -r/2, main: 1+r, super-diag: -r/2
            let sub = vec![-0.5 * r; m.saturating_sub(1)];
            let main = vec![1.0 + r; m];
            let sup = vec![-0.5 * r; m.saturating_sub(1)];

            let interior = solve_tridiagonal(&sub, &main, &sup, &rhs)?;

            // Assemble full solution
            let mut u_new = vec![0.0; nx];
            u_new[1..(m + 1)].copy_from_slice(&interior[..m]);

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);

            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }
}

/// 1-D wave equation solver: d²u/dt² = c² * d²u/dx².
pub struct WaveEquation1D {
    /// Wave speed.
    pub c: f64,
}

impl WaveEquation1D {
    /// Compute the Courant number: c * dt / dx.
    pub fn courant_number(&self, dx: f64, dt: f64) -> f64 {
        self.c * dt / dx
    }

    /// Solve using the leapfrog / Störmer-Verlet scheme.
    ///
    /// `u0` is the initial displacement, `v0` the initial velocity.
    /// Stability requires Courant number <= 1.
    pub fn solve(&self, u0: &[f64], v0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx || v0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "wave_solve: u0/v0 length mismatch with nx ({nx})"
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "wave_solve: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.c * dt / dx;
        let cfl2 = cfl * cfl;

        // u^{n-1} and u^{n}
        let mut u_prev = u0.to_vec();
        let mut u_cur = vec![0.0; nx];

        // First step uses Taylor expansion: u^1 = u^0 + dt*v0 + 0.5*dt²*c²*u''
        for i in 1..nx - 1 {
            let d2u = (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]) / (dx * dx);
            u_cur[i] = u0[i] + dt * v0[i] + 0.5 * dt * dt * self.c * self.c * d2u;
        }
        apply_bc_1d(&mut u_cur, &config.bc_left, &config.bc_right, nx);

        let mut results = vec![u_prev.clone(), u_cur.clone()];

        // Leapfrog: u^{n+1} = 2*u^n - u^{n-1} + cfl² * (u_{i+1} - 2*u_i + u_{i-1})
        for _ in 1..config.num_steps {
            let mut u_next = vec![0.0; nx];
            for i in 1..nx - 1 {
                u_next[i] = 2.0 * u_cur[i] - u_prev[i]
                    + cfl2 * (u_cur[i + 1] - 2.0 * u_cur[i] + u_cur[i - 1]);
            }
            apply_bc_1d(&mut u_next, &config.bc_left, &config.bc_right, nx);

            u_prev = u_cur;
            u_cur = u_next;
            results.push(u_cur.clone());
        }

        Ok(results)
    }
}

/// 1-D Poisson equation solver: -u'' = f, with boundary conditions.
pub struct Poisson1D;

impl Poisson1D {
    /// Solve -u'' = f on the grid with specified boundary conditions.
    ///
    /// Uses a tridiagonal direct solve (Thomas algorithm).
    pub fn solve(&self, f: &[f64], config: &PdeConfig) -> SolverResult<Vec<f64>> {
        let nx = config.grid.nx;
        if f.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "poisson: f length ({}) != nx ({nx})",
                f.len()
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "poisson: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dx2 = dx * dx;
        let m = nx - 2; // interior points

        // Build tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = dx²*f_i
        let sub = vec![-1.0; m.saturating_sub(1)];
        let main = vec![2.0; m];
        let sup = vec![-1.0; m.saturating_sub(1)];

        let mut rhs = vec![0.0; m];
        for i in 0..m {
            rhs[i] = dx2 * f[i + 1];
        }

        // Add boundary contributions
        match config.bc_left {
            BoundaryCondition::Dirichlet(val) => {
                rhs[0] += val;
            }
            BoundaryCondition::Neumann(val) => {
                // Ghost point approach: u_{-1} = u_1 - 2*dx*val
                rhs[0] += -2.0 * dx * val; // approximate
            }
            BoundaryCondition::Periodic => {}
        }
        match config.bc_right {
            BoundaryCondition::Dirichlet(val) => {
                if m > 0 {
                    rhs[m - 1] += val;
                }
            }
            BoundaryCondition::Neumann(val) => {
                if m > 0 {
                    rhs[m - 1] += 2.0 * dx * val;
                }
            }
            BoundaryCondition::Periodic => {}
        }

        let interior = solve_tridiagonal(&sub, &main, &sup, &rhs)?;

        // Assemble full solution
        let mut u = vec![0.0; nx];
        u[1..(m + 1)].copy_from_slice(&interior[..m]);
        apply_bc_1d(&mut u, &config.bc_left, &config.bc_right, nx);

        Ok(u)
    }
}

/// 1-D advection equation solver: du/dt + a * du/dx = 0.
pub struct AdvectionEquation1D {
    /// Advection velocity.
    pub a: f64,
}

impl AdvectionEquation1D {
    /// Solve using the first-order upwind scheme.
    pub fn solve_upwind(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "advection_upwind: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.a * dt / dx;

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            for i in 1..nx - 1 {
                if self.a >= 0.0 {
                    // Upwind from left
                    u_new[i] = u[i] - cfl * (u[i] - u[i - 1]);
                } else {
                    // Upwind from right
                    u_new[i] = u[i] - cfl * (u[i + 1] - u[i]);
                }
            }

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);
            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }

    /// Solve using the Lax-Wendroff scheme (second-order).
    pub fn solve_lax_wendroff(
        &self,
        u0: &[f64],
        config: &PdeConfig,
    ) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "advection_lw: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.a * dt / dx;
        let cfl2 = cfl * cfl;

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            for i in 1..nx - 1 {
                u_new[i] = u[i] - 0.5 * cfl * (u[i + 1] - u[i - 1])
                    + 0.5 * cfl2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            }

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);
            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Maximum absolute deviation between two equal-length slices.
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Sample `sin(pi * x)` on every grid node.
    fn sine_ic(grid: &Grid1D) -> Vec<f64> {
        (0..grid.nx).map(|i| (PI * grid.point(i)).sin()).collect()
    }

    // ---------------------------------------------------------------------
    // Grid geometry
    // ---------------------------------------------------------------------

    #[test]
    fn grid1d_spacing_and_points() {
        // Five nodes on [0, 1] => spacing 0.25 and exact node coordinates.
        let grid = Grid1D::new(0.0, 1.0, 5);
        assert!((grid.dx - 0.25).abs() < 1e-15);
        assert!((grid.point(0) - 0.0).abs() < 1e-15);
        assert!((grid.point(2) - 0.5).abs() < 1e-15);
        assert!((grid.point(4) - 1.0).abs() < 1e-15);

        // Degenerate single-node grid has zero spacing (no division by zero).
        let degenerate = Grid1D::new(0.0, 1.0, 1);
        assert_eq!(degenerate.dx, 0.0);
    }

    // ---------------------------------------------------------------------
    // Heat equation: u_t = alpha u_xx
    // ---------------------------------------------------------------------

    #[test]
    fn heat_ftcs_matches_analytic_decay_and_stability_limit() {
        // u(x,0) = sin(pi x), Dirichlet 0 BCs, alpha = 1.
        // Exact solution: u(x,t) = exp(-pi^2 t) sin(pi x).
        let alpha = 1.0;
        let heat = HeatEquation1D { alpha };
        let grid = Grid1D::new(0.0, 1.0, 41);
        let dx = grid.dx;

        // Stability limit is exactly dx^2 / (2 alpha).
        assert!((heat.stability_limit(dx) - dx * dx / (2.0 * alpha)).abs() < 1e-15);

        // Choose r = 0.4 < 0.5 so the explicit FTCS scheme is stable.
        let r = 0.4;
        let dt = r * dx * dx / alpha;
        let num_steps = 80;
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        let u0 = sine_ic(&grid);
        let history = heat.solve_explicit(&u0, &config).expect("heat explicit");
        let final_u = history.last().expect("non-empty history");

        let t = num_steps as f64 * dt;
        let factor = (-PI * PI * t).exp();
        let exact: Vec<f64> = (0..grid.nx)
            .map(|i| factor * (PI * grid.point(i)).sin())
            .collect();

        // Decay has actually happened (mid-point amplitude shrank well below 1).
        assert!(final_u[grid.nx / 2] < 0.85 && final_u[grid.nx / 2] > 0.78);
        // First-order-in-time, second-order-in-space error stays tiny.
        assert!(max_abs_diff(final_u, &exact) < 1e-3);
    }

    #[test]
    fn heat_crank_nicolson_spatial_convergence_order() {
        // Crank-Nicolson is O(dt^2 + dx^2). With dt fixed and small the temporal
        // error is negligible, so halving dx must cut the error by ~4x (2nd order).
        let heat = HeatEquation1D { alpha: 1.0 };
        let dt = 1e-4;
        let num_steps = 200; // t_final = 0.02, identical for all grids
        let t = num_steps as f64 * dt;
        let factor = (-PI * PI * t).exp();

        let mut errors = Vec::new();
        for &nx in &[11usize, 21, 41] {
            let grid = Grid1D::new(0.0, 1.0, nx);
            let config = PdeConfig {
                grid: grid.clone(),
                dt,
                num_steps,
                bc_left: BoundaryCondition::Dirichlet(0.0),
                bc_right: BoundaryCondition::Dirichlet(0.0),
            };
            let u0 = sine_ic(&grid);
            let history = heat.solve_implicit(&u0, &config).expect("heat CN");
            let final_u = history.last().expect("non-empty history");
            let exact: Vec<f64> = (0..nx)
                .map(|i| factor * (PI * grid.point(i)).sin())
                .collect();
            errors.push(max_abs_diff(final_u, &exact));
        }

        // Each refinement quarters the error (observed ratios ~3.99).
        for window in errors.windows(2) {
            let ratio = window[0] / window[1];
            assert!(
                ratio > 3.7 && ratio < 4.3,
                "expected ~4x error reduction, got ratio {ratio}"
            );
        }
        assert!(errors.last().expect("errors") < &2e-4);
    }

    #[test]
    fn heat_constant_field_and_steady_state_dirichlet() {
        let heat = HeatEquation1D { alpha: 1.0 };

        // (a) A constant field consistent with the BCs is preserved exactly.
        let grid = Grid1D::new(0.0, 1.0, 11);
        let dt = 0.4 * grid.dx * grid.dx;
        let const_cfg = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 50,
            bc_left: BoundaryCondition::Dirichlet(3.0),
            bc_right: BoundaryCondition::Dirichlet(3.0),
        };
        let u0 = vec![3.0; grid.nx];
        let history = heat.solve_explicit(&u0, &const_cfg).expect("heat const");
        let final_u = history.last().expect("history");
        assert!(final_u.iter().all(|&v| (v - 3.0).abs() < 1e-12));

        // (b) From a zero initial field with mismatched Dirichlet ends (2 and 5)
        // the diffusion relaxes to the exact steady linear profile u = 2 + 3x,
        // and the boundary values are pinned at every step.
        let steady_cfg = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 400, // t = 1.6, lowest mode decays by exp(-pi^2*1.6) ~ 1e-7
            bc_left: BoundaryCondition::Dirichlet(2.0),
            bc_right: BoundaryCondition::Dirichlet(5.0),
        };
        let zero = vec![0.0; grid.nx];
        let hist = heat
            .solve_explicit(&zero, &steady_cfg)
            .expect("heat steady");
        // history[0] is the raw initial condition; the boundary values are
        // enforced from the first stepped frame onward.
        for frame in hist.iter().skip(1) {
            assert!((frame[0] - 2.0).abs() < 1e-12);
            assert!((frame[grid.nx - 1] - 5.0).abs() < 1e-12);
        }
        let final_u = hist.last().expect("history");
        let steady: Vec<f64> = (0..grid.nx).map(|i| 2.0 + 3.0 * grid.point(i)).collect();
        assert!(max_abs_diff(final_u, &steady) < 1e-3);
    }

    // ---------------------------------------------------------------------
    // Poisson equation: -u'' = f
    // ---------------------------------------------------------------------

    #[test]
    fn poisson_polynomial_solutions_exact() {
        let solver = Poisson1D;

        // (a) -u'' = 2, u(0)=u(1)=0 => u = x(1-x). The 3-point stencil is exact
        // for quadratics, so the numerical solution matches to machine precision.
        let grid = Grid1D::new(0.0, 1.0, 21);
        let cfg = PdeConfig {
            grid: grid.clone(),
            dt: 0.0,
            num_steps: 0,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };
        let f = vec![2.0; grid.nx];
        let u = solver.solve(&f, &cfg).expect("poisson quadratic");
        let exact: Vec<f64> = (0..grid.nx)
            .map(|i| grid.point(i) * (1.0 - grid.point(i)))
            .collect();
        assert!(max_abs_diff(&u, &exact) < 1e-12);

        // (b) -u'' = 0 with u(0)=0, u(1)=1 => u = x. Linear is reproduced exactly
        // and verifies the nonzero-Dirichlet right-hand-side contribution.
        let cfg2 = PdeConfig {
            grid: grid.clone(),
            dt: 0.0,
            num_steps: 0,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(1.0),
        };
        let zero_f = vec![0.0; grid.nx];
        let u_lin = solver.solve(&zero_f, &cfg2).expect("poisson linear");
        let exact_lin: Vec<f64> = (0..grid.nx).map(|i| grid.point(i)).collect();
        assert!(max_abs_diff(&u_lin, &exact_lin) < 1e-12);
    }

    #[test]
    fn poisson_sine_second_order_convergence() {
        // -u'' = pi^2 sin(pi x), u(0)=u(1)=0 => exact u = sin(pi x).
        // The discretization error decreases as O(dx^2) under refinement.
        let solver = Poisson1D;
        let mut errors = Vec::new();
        for &nx in &[11usize, 21, 41] {
            let grid = Grid1D::new(0.0, 1.0, nx);
            let cfg = PdeConfig {
                grid: grid.clone(),
                dt: 0.0,
                num_steps: 0,
                bc_left: BoundaryCondition::Dirichlet(0.0),
                bc_right: BoundaryCondition::Dirichlet(0.0),
            };
            let f: Vec<f64> = (0..nx)
                .map(|i| PI * PI * (PI * grid.point(i)).sin())
                .collect();
            let u = solver.solve(&f, &cfg).expect("poisson sine");
            let exact: Vec<f64> = (0..nx).map(|i| (PI * grid.point(i)).sin()).collect();
            errors.push(max_abs_diff(&u, &exact));
        }
        for window in errors.windows(2) {
            let ratio = window[0] / window[1];
            assert!(
                ratio > 3.7 && ratio < 4.3,
                "expected ~4x error reduction, got ratio {ratio}"
            );
        }
        assert!(errors.last().expect("errors") < &1e-3);
    }

    // ---------------------------------------------------------------------
    // Wave equation: u_tt = c^2 u_xx
    // ---------------------------------------------------------------------

    #[test]
    fn wave_standing_wave_analytic_symmetry_pinned() {
        // u(x,0) = sin(pi x), u_t(x,0) = 0, c = 1, Dirichlet 0 BCs.
        // Exact standing wave: u(x,t) = cos(pi t) sin(pi x).
        let wave = WaveEquation1D { c: 1.0 };
        let grid = Grid1D::new(0.0, 1.0, 41);
        let dx = grid.dx;
        let cfl = 0.5;
        let dt = cfl * dx / wave.c;
        assert!((wave.courant_number(dx, dt) - cfl).abs() < 1e-15);
        let num_steps = 8; // t = 0.1

        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };
        let u0 = sine_ic(&grid);
        let v0 = vec![0.0; grid.nx];
        let history = wave.solve(&u0, &v0, &config).expect("wave solve");

        // Dirichlet boundaries pinned at zero throughout the evolution.
        for frame in &history {
            assert!(frame[0].abs() < 1e-14);
            assert!(frame[grid.nx - 1].abs() < 1e-14);
        }

        let final_u = history.last().expect("history");
        let t = num_steps as f64 * dt;
        let factor = (PI * wave.c * t).cos();
        let exact: Vec<f64> = (0..grid.nx)
            .map(|i| factor * (PI * grid.point(i)).sin())
            .collect();
        assert!(max_abs_diff(final_u, &exact) < 1e-3);

        // Symmetric IC about x = 0.5 => the solution stays mirror-symmetric.
        for i in 0..grid.nx {
            assert!((final_u[i] - final_u[grid.nx - 1 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn wave_leapfrog_second_order_convergence() {
        // Fixed Courant number 0.5; refining the grid (dt ~ dx) cuts the error
        // by ~4x per refinement, confirming the scheme is O(dt^2 + dx^2).
        let wave = WaveEquation1D { c: 1.0 };
        let mut errors = Vec::new();
        for &(nx, num_steps) in &[(21usize, 4usize), (41, 8), (81, 16)] {
            let grid = Grid1D::new(0.0, 1.0, nx);
            let dt = 0.5 * grid.dx / wave.c;
            let config = PdeConfig {
                grid: grid.clone(),
                dt,
                num_steps,
                bc_left: BoundaryCondition::Dirichlet(0.0),
                bc_right: BoundaryCondition::Dirichlet(0.0),
            };
            let u0 = sine_ic(&grid);
            let v0 = vec![0.0; nx];
            let history = wave.solve(&u0, &v0, &config).expect("wave solve");
            let final_u = history.last().expect("history");
            let t = num_steps as f64 * dt; // = 0.1 for every grid
            let factor = (PI * wave.c * t).cos();
            let exact: Vec<f64> = (0..nx)
                .map(|i| factor * (PI * grid.point(i)).sin())
                .collect();
            errors.push(max_abs_diff(final_u, &exact));
        }
        for window in errors.windows(2) {
            let ratio = window[0] / window[1];
            assert!(
                ratio > 3.7 && ratio < 4.3,
                "expected ~4x error reduction, got ratio {ratio}"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Advection equation: u_t + a u_x = 0
    // ---------------------------------------------------------------------

    #[test]
    fn advection_upwind_constant_and_cfl_one_transport() {
        let adv = AdvectionEquation1D { a: 1.0 };
        let grid = Grid1D::new(0.0, 10.0, 21);
        let dx = grid.dx;

        // (a) A constant field consistent with the inflow BC is preserved.
        let const_cfg = PdeConfig {
            grid: grid.clone(),
            dt: 0.8 * dx / adv.a,
            num_steps: 25,
            bc_left: BoundaryCondition::Dirichlet(4.0),
            bc_right: BoundaryCondition::Dirichlet(4.0),
        };
        let u0 = vec![4.0; grid.nx];
        let hist = adv.solve_upwind(&u0, &const_cfg).expect("advection const");
        assert!(
            hist.last()
                .expect("history")
                .iter()
                .all(|&v| (v - 4.0).abs() < 1e-12)
        );

        // (b) At Courant number exactly 1 first-order upwind transports the
        // profile by one cell per step with no diffusion: u_i^{n+1} = u_{i-1}^n.
        let dt = dx / adv.a; // CFL = 1
        let shift_cfg = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 3,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };
        // Narrow bump centred at node 10 (x = 5), far from both boundaries.
        let bump: Vec<f64> = (0..grid.nx)
            .map(|i| {
                let z = (grid.point(i) - 5.0) / 0.7;
                (-z * z).exp()
            })
            .collect();
        let result = adv
            .solve_upwind(&bump, &shift_cfg)
            .expect("advection shift");
        let final_u = result.last().expect("history");
        // After 3 steps node i holds the original value at node i-3.
        for i in 1..grid.nx - 1 {
            let expected = if i >= 3 { bump[i - 3] } else { 0.0 };
            assert!((final_u[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn advection_lax_wendroff_cfl_one_transport() {
        // The second-order Lax-Wendroff scheme is also an exact one-cell shift
        // at Courant number 1: u_i^{n+1} = u_{i-1}^n.
        let adv = AdvectionEquation1D { a: 1.0 };
        let grid = Grid1D::new(0.0, 10.0, 21);
        let dx = grid.dx;
        let dt = dx / adv.a; // CFL = 1
        let cfg = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 3,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };
        let bump: Vec<f64> = (0..grid.nx)
            .map(|i| {
                let z = (grid.point(i) - 5.0) / 0.7;
                (-z * z).exp()
            })
            .collect();
        let result = adv
            .solve_lax_wendroff(&bump, &cfg)
            .expect("lax-wendroff shift");
        let final_u = result.last().expect("history");
        for i in 1..grid.nx - 1 {
            let expected = if i >= 3 { bump[i - 3] } else { 0.0 };
            assert!((final_u[i] - expected).abs() < 1e-10);
        }
    }
}
