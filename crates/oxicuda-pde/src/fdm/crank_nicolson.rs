//! Crank-Nicolson (θ-method) implicit solver for the 1D heat equation.
//!
//! Solves `u_t = α·u_xx + s(x, t)` on `[x0, x1]` with time-dependent Dirichlet
//! boundary conditions, using the generalised θ-scheme:
//!
//! ```text
//! (u^{n+1} − u^n)/Δt = α·[ θ·L u^{n+1} + (1−θ)·L u^n ] + θ·s^{n+1} + (1−θ)·s^n
//! ```
//!
//! where `L` is the second-order central-difference Laplacian. `θ = 1/2` gives
//! the classical Crank-Nicolson scheme (second-order in time, A-stable);
//! `θ = 1` gives backward Euler and `θ = 0` forward Euler. For `θ ≥ 1/2` the
//! scheme is unconditionally stable, so no CFL restriction applies.
//!
//! Rearranging with `r = α·Δt/h²` gives the tridiagonal system
//! ```text
//! (I − θ·r·A) u^{n+1} = (I + (1−θ)·r·A) u^n + Δt·[θ·s^{n+1} + (1−θ)·s^n] + bc
//! ```
//! which is solved with the [`crate::fdm::poisson_1d::thomas_solve`]
//! tridiagonal algorithm in `O(n)` per step. This struct-based driver differs
//! from the single-step [`crate::fdm::heat_1d::crank_nicolson_step`] helper by
//! supporting an arbitrary θ, a space- and time-dependent source term, and
//! time-varying boundary values across a multi-step integration.

use crate::error::{PdeError, PdeResult};
use crate::fdm::poisson_1d::thomas_solve;
use crate::mesh::Mesh1d;

/// Crank-Nicolson / θ-method heat-equation solver on a fixed 1D mesh.
#[derive(Debug, Clone)]
pub struct CrankNicolson {
    /// Diffusion coefficient `α > 0`.
    pub alpha: f64,
    /// Time step `Δt > 0`.
    pub dt: f64,
    /// Implicitness parameter `θ ∈ [0, 1]` (0.5 = Crank-Nicolson).
    pub theta: f64,
}

impl CrankNicolson {
    /// Build a θ-method solver. For genuine Crank-Nicolson (`θ = 0.5`) use the
    /// [`CrankNicolson::with_half_theta`] shortcut.
    ///
    /// Returns [`PdeError::InvalidParameter`] for non-positive `alpha`/`dt` or
    /// `theta` outside `[0, 1]`.
    pub fn new(alpha: f64, dt: f64, theta: f64) -> PdeResult<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "alpha".into(),
                reason: format!("must be a finite value > 0, got {alpha}"),
            });
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be a finite value > 0, got {dt}"),
            });
        }
        if !(0.0..=1.0).contains(&theta) {
            return Err(PdeError::InvalidParameter {
                name: "theta".into(),
                reason: format!("must be in [0, 1], got {theta}"),
            });
        }
        Ok(Self { alpha, dt, theta })
    }

    /// Convenience constructor with `θ = 0.5` (classical Crank-Nicolson).
    pub fn with_half_theta(alpha: f64, dt: f64) -> PdeResult<Self> {
        Self::new(alpha, dt, 0.5)
    }

    /// `true` when the scheme is unconditionally stable (`θ ≥ 1/2`).
    #[must_use]
    pub fn is_unconditionally_stable(&self) -> bool {
        self.theta >= 0.5
    }

    /// Advance the solution by a single time step.
    ///
    /// * `mesh` — uniform 1D mesh.
    /// * `u` — current solution (length `mesh.n`); overwritten in place.
    /// * `source_old`, `source_new` — source term `s(x, tⁿ)` and `s(x, tⁿ⁺¹)`
    ///   sampled at the grid nodes (length `mesh.n` each).
    /// * `ua_old`/`ua_new`, `ub_old`/`ub_new` — left/right Dirichlet values at
    ///   the old and new time levels (linear-in-time within the step).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        mesh: &Mesh1d,
        u: &mut [f64],
        source_old: &[f64],
        source_new: &[f64],
        ua_old: f64,
        ua_new: f64,
        ub_old: f64,
        ub_new: f64,
    ) -> PdeResult<()> {
        let n = mesh.n;
        if u.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![u.len()],
            });
        }
        if source_old.len() != n || source_new.len() != n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![n],
                got: vec![source_old.len().min(source_new.len())],
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "crank-nicolson requires n >= 3, got {n}"
            )));
        }
        let h = mesh.h();
        if h <= 0.0 {
            return Err(PdeError::InvalidGrid(format!("non-positive h={h}")));
        }
        if self.theta < 0.5 {
            // Explicit-leaning θ: enforce the forward-Euler-like stability bound.
            let r = self.alpha * self.dt / (h * h);
            let limit = 0.5 / (1.0 - self.theta);
            if r > limit + 1.0e-12 {
                return Err(PdeError::CflViolation {
                    dt: self.dt,
                    dt_max: limit * h * h / self.alpha,
                });
            }
        }

        let r = self.alpha * self.dt / (h * h);
        let theta = self.theta;
        let m = n - 2;

        // LHS tridiagonal: (I − θ r A), A = tridiag(1, −2, 1).
        let mut sub = vec![-theta * r; m];
        let mut diag = vec![1.0 + 2.0 * theta * r; m];
        let mut sup = vec![-theta * r; m];

        // RHS = (I + (1−θ) r A) uⁿ + Δt[θ sⁿ⁺¹ + (1−θ) sⁿ] + implicit-boundary term.
        //
        // For each interior node the explicit discrete Laplacian uses the
        // *old-time* neighbours; at the first/last interior node one neighbour
        // is the boundary, taken at the old level (`ua_old` / `ub_old`). The
        // implicit coupling `−θ r u_boundary^{n+1}` from the LHS is moved to the
        // RHS using the *new* boundary value (`ua_new` / `ub_new`).
        let one_minus = 1.0 - theta;
        let mut rhs = vec![0.0; m];
        for (i, rhs_i) in rhs.iter_mut().enumerate().take(m) {
            let gi = i + 1; // global interior index
            let left = if gi == 1 { ua_old } else { u[gi - 1] };
            let right = if gi == n - 2 { ub_old } else { u[gi + 1] };
            let lap = left - 2.0 * u[gi] + right;
            let src = self.dt * (theta * source_new[gi] + one_minus * source_old[gi]);
            *rhs_i = u[gi] + one_minus * r * lap + src;
        }
        rhs[0] += theta * r * ua_new;
        rhs[m - 1] += theta * r * ub_new;

        sub[0] = 0.0;
        sup[m - 1] = 0.0;
        let interior = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;

        u[0] = ua_new;
        u[n - 1] = ub_new;
        u[1..n - 1].copy_from_slice(&interior);
        Ok(())
    }

    /// Advance the solution by a single step with a *constant* (time- and
    /// space-independent) source and constant Dirichlet boundary values.
    pub fn step_constant(
        &self,
        mesh: &Mesh1d,
        u: &mut [f64],
        source: f64,
        ua: f64,
        ub: f64,
    ) -> PdeResult<()> {
        let s = vec![source; mesh.n];
        self.step(mesh, u, &s, &s, ua, ua, ub, ub)
    }

    /// Integrate `n_steps` steps of the homogeneous heat equation
    /// (`s ≡ 0`) with fixed Dirichlet boundary values.
    ///
    /// Returns the solution at time `n_steps · Δt`.
    pub fn solve_homogeneous(
        &self,
        mesh: &Mesh1d,
        u0: &[f64],
        ua: f64,
        ub: f64,
        n_steps: usize,
    ) -> PdeResult<Vec<f64>> {
        let mut u = u0.to_vec();
        if u.len() != mesh.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![mesh.n],
                got: vec![u.len()],
            });
        }
        let zero = vec![0.0; mesh.n];
        for _ in 0..n_steps {
            self.step(mesh, &mut u, &zero, &zero, ua, ua, ub, ub)?;
        }
        Ok(u)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_initial(mesh: &Mesh1d) -> Vec<f64> {
        mesh.nodes.iter().map(|x| (PI * x).sin()).collect()
    }

    #[test]
    fn constructor_rejects_bad_parameters() {
        assert!(CrankNicolson::new(-1.0, 0.1, 0.5).is_err());
        assert!(CrankNicolson::new(1.0, 0.0, 0.5).is_err());
        assert!(CrankNicolson::new(1.0, 0.1, 1.5).is_err());
        assert!(CrankNicolson::new(1.0, 0.1, -0.1).is_err());
        assert!(CrankNicolson::new(1.0, 0.1, 0.5).is_ok());
    }

    #[test]
    fn crank_nicolson_helper_sets_half_theta() {
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        assert!((cn.theta - 0.5).abs() < 1e-15);
        assert!(cn.is_unconditionally_stable());
    }

    #[test]
    fn matches_analytic_decay() {
        // u(x,0)=sin(πx) decays like exp(−π²αt) on [0,1] with u(0)=u(1)=0.
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let alpha = 1.0;
        let dt = 0.001;
        let cn = CrankNicolson::with_half_theta(alpha, dt).expect("ok");
        let u0 = sine_initial(&mesh);
        let t_final = 0.05;
        let n_steps = (t_final / dt).round() as usize;
        let u = cn
            .solve_homogeneous(&mesh, &u0, 0.0, 0.0, n_steps)
            .expect("ok");
        let amp = (-PI * PI * alpha * t_final).exp();
        let mid = mesh.n / 2;
        let analytic = (PI * mesh.nodes[mid]).sin() * amp;
        assert!(
            (u[mid] - analytic).abs() < 1e-3,
            "u={} analytic={analytic}",
            u[mid]
        );
    }

    #[test]
    fn unconditionally_stable_with_large_dt() {
        // θ=0.5 with a very large dt must not blow up; solution decays to 0.
        let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
        let dt = 10.0 * mesh.h() * mesh.h();
        let cn = CrankNicolson::new(1.0, dt, 0.5).expect("ok");
        let u0 = sine_initial(&mesh);
        let u = cn.solve_homogeneous(&mesh, &u0, 0.0, 0.0, 80).expect("ok");
        for v in &u {
            assert!(v.abs() < 1e-2, "value did not decay: {v}");
        }
    }

    #[test]
    fn backward_euler_theta_one_stable() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
        let dt = 5.0 * mesh.h() * mesh.h();
        let cn = CrankNicolson::new(1.0, dt, 1.0).expect("ok");
        let u0 = sine_initial(&mesh);
        let u = cn.solve_homogeneous(&mesh, &u0, 0.0, 0.0, 50).expect("ok");
        for v in &u {
            assert!(v.abs() < 1e-2);
        }
    }

    #[test]
    fn steady_state_with_constant_source() {
        // u_t = u_xx + 2, u(0)=u(1)=0. Steady state solves −u'' = 2 → u=x(1−x).
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let dt = 0.002;
        let cn = CrankNicolson::with_half_theta(1.0, dt).expect("ok");
        let mut u = vec![0.0; mesh.n];
        for _ in 0..3000 {
            cn.step_constant(&mesh, &mut u, 2.0, 0.0, 0.0).expect("ok");
        }
        for (i, &ui) in u.iter().enumerate() {
            let x = mesh.nodes[i];
            let expected = x * (1.0 - x);
            assert!(
                (ui - expected).abs() < 2e-3,
                "i={i} u={ui} expected={expected}"
            );
        }
    }

    #[test]
    fn boundary_values_are_applied() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n];
        cn.step_constant(&mesh, &mut u, 0.0, 1.0, 2.0).expect("ok");
        assert!((u[0] - 1.0).abs() < 1e-15);
        assert!((u[mesh.n - 1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn nonzero_boundary_steady_state_is_linear() {
        // u_t = u_xx, u(0)=1, u(1)=3, no source. Steady state is linear: 1+2x.
        let mesh = Mesh1d::uniform(0.0, 1.0, 21).expect("ok");
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n];
        for _ in 0..2000 {
            cn.step_constant(&mesh, &mut u, 0.0, 1.0, 3.0).expect("ok");
        }
        for (i, &ui) in u.iter().enumerate() {
            let x = mesh.nodes[i];
            let expected = 1.0 + 2.0 * x;
            assert!(
                (ui - expected).abs() < 2e-3,
                "i={i} u={ui} expected={expected}"
            );
        }
    }

    #[test]
    fn second_order_time_accuracy() {
        // Isolate the *temporal* discretisation error on a fixed spatial grid by
        // using a tiny-dt run as the reference (Richardson self-convergence).
        // Halving dt should then shrink the temporal error by ~4× for CN.
        let mesh = Mesh1d::uniform(0.0, 1.0, 41).expect("ok");
        let alpha = 1.0;
        let t_final = 0.04;
        let mid = mesh.n / 2;

        let solve = |dt: f64| -> f64 {
            let cn = CrankNicolson::with_half_theta(alpha, dt).expect("ok");
            let u0 = sine_initial(&mesh);
            let n_steps = (t_final / dt).round() as usize;
            let u = cn
                .solve_homogeneous(&mesh, &u0, 0.0, 0.0, n_steps)
                .expect("ok");
            u[mid]
        };
        let reference = solve(t_final / 4096.0); // essentially exact in time
        let e_coarse = (solve(t_final / 16.0) - reference).abs();
        let e_fine = (solve(t_final / 32.0) - reference).abs();
        let ratio = e_coarse / e_fine.max(1e-15);
        assert!(ratio > 3.0, "second-order ratio too low: {ratio}");
    }

    #[test]
    fn shape_mismatch_errors() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n - 1];
        let s = vec![0.0; mesh.n];
        assert!(cn.step(&mesh, &mut u, &s, &s, 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn source_shape_mismatch_errors() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; mesh.n];
        let bad = vec![0.0; mesh.n - 2];
        let good = vec![0.0; mesh.n];
        assert!(
            cn.step(&mesh, &mut u, &bad, &good, 0.0, 0.0, 0.0, 0.0)
                .is_err()
        );
    }

    #[test]
    fn explicit_theta_cfl_violation_detected() {
        // θ=0 (forward Euler) with too-large dt must trip the CFL guard.
        let mesh = Mesh1d::uniform(0.0, 1.0, 11).expect("ok");
        let dt = 5.0 * mesh.h() * mesh.h(); // r = 5 ≫ 0.5
        let cn = CrankNicolson::new(1.0, dt, 0.0).expect("ok");
        let mut u = vec![0.5; mesh.n];
        let s = vec![0.0; mesh.n];
        assert!(cn.step(&mesh, &mut u, &s, &s, 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn solution_stays_finite() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 31).expect("ok");
        let cn = CrankNicolson::with_half_theta(0.5, 0.005).expect("ok");
        let u0 = sine_initial(&mesh);
        let u = cn.solve_homogeneous(&mesh, &u0, 0.0, 0.0, 100).expect("ok");
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn too_small_mesh_errors() {
        let mesh = Mesh1d::uniform(0.0, 1.0, 2).expect("ok");
        let cn = CrankNicolson::with_half_theta(1.0, 0.01).expect("ok");
        let mut u = vec![0.0; 2];
        let s = vec![0.0; 2];
        assert!(cn.step(&mesh, &mut u, &s, &s, 0.0, 0.0, 0.0, 0.0).is_err());
    }
}
