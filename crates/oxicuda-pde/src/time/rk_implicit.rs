//! Fully-implicit Runge-Kutta methods (collocation type) for stiff ODE systems.
//!
//! For `y' = f(t, y)` an `s`-stage implicit Runge-Kutta method advances the
//! solution through stage derivatives `K₁,…,K_s` that satisfy the coupled
//! nonlinear system
//!
//! ```text
//! Kᵢ = f( t + cᵢ·dt ,  y + dt·Σⱼ aᵢⱼ Kⱼ ) ,   i = 1,…,s
//! y_{n+1} = y + dt·Σᵢ bᵢ Kᵢ .
//! ```
//!
//! The whole `(s·d)`-dimensional system is solved by Newton's method: the block
//! Jacobian `∂Rᵢ/∂Kₗ = δᵢₗ I − dt·aᵢₗ Jᵢ` (with `Jᵢ = ∂f/∂y` at stage `i`) is
//! assembled into a dense matrix and factorised with the crate's Gaussian
//! elimination. Unlike the diagonally-implicit SDIRK methods in
//! [`crate::time::sdirk`], the tableaux here are *full* (dense `A`), giving the
//! high stage order of collocation schemes.
//!
//! Two classical methods are provided:
//! - **Gauss-Legendre (2-stage)** — order 4, A-stable, symplectic, the unique
//!   2-stage collocation at the Gauss points.
//! - **Radau IIA (3-stage)** — order 5, L-stable, stiffly accurate (`b` equals
//!   the last row of `A`), the workhorse for very stiff problems.
//!
//! # Reference
//! E. Hairer & G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
//! Differential-Algebraic Problems*, 2nd ed., Springer (1996), §IV.5, §IV.8.

use crate::error::{PdeError, PdeResult};
use crate::spectral::chebyshev::gauss_solve_dense;
use crate::time::sdirk::SdirkConfig;

/// Selectable fully-implicit Runge-Kutta method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicitRkMethod {
    /// 2-stage Gauss-Legendre collocation — order 4, A-stable, symplectic.
    GaussLegendre4,
    /// 3-stage Radau IIA collocation — order 5, L-stable, stiffly accurate.
    RadauIia5,
}

impl ImplicitRkMethod {
    /// Number of stages `s`.
    #[must_use]
    pub fn stages(self) -> usize {
        match self {
            ImplicitRkMethod::GaussLegendre4 => 2,
            ImplicitRkMethod::RadauIia5 => 3,
        }
    }

    /// Formal order of accuracy.
    #[must_use]
    pub fn order(self) -> usize {
        match self {
            ImplicitRkMethod::GaussLegendre4 => 4,
            ImplicitRkMethod::RadauIia5 => 5,
        }
    }

    /// Butcher tableau `(c, a, b, s)` with `a` stored row-major (`s×s`).
    #[must_use]
    pub fn tableau(self) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        match self {
            ImplicitRkMethod::GaussLegendre4 => {
                let r = 3.0_f64.sqrt() / 6.0; // √3 / 6
                let c = vec![0.5 - r, 0.5 + r];
                let a = vec![0.25, 0.25 - r, 0.25 + r, 0.25];
                let b = vec![0.5, 0.5];
                (c, a, b, 2)
            }
            ImplicitRkMethod::RadauIia5 => {
                let s6 = 6.0_f64.sqrt(); // √6
                let c = vec![(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0];
                let a = vec![
                    (88.0 - 7.0 * s6) / 360.0,
                    (296.0 - 169.0 * s6) / 1800.0,
                    (-2.0 + 3.0 * s6) / 225.0,
                    (296.0 + 169.0 * s6) / 1800.0,
                    (88.0 + 7.0 * s6) / 360.0,
                    (-2.0 - 3.0 * s6) / 225.0,
                    (16.0 - s6) / 36.0,
                    (16.0 + s6) / 36.0,
                    1.0 / 9.0,
                ];
                let b = vec![(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0];
                (c, a, b, 3)
            }
        }
    }
}

/// Driver for a fully-implicit Runge-Kutta integrator.
#[derive(Debug, Clone, Copy)]
pub struct ImplicitRk {
    /// Chosen tableau.
    pub method: ImplicitRkMethod,
    /// Newton iteration controls ([`SdirkConfig`] reused for `{tol, max_iter}`).
    pub cfg: SdirkConfig,
}

impl ImplicitRk {
    /// Build a driver with the default Newton configuration.
    #[must_use]
    pub fn new(method: ImplicitRkMethod) -> Self {
        Self {
            method,
            cfg: SdirkConfig::default(),
        }
    }

    /// Build a driver with a custom Newton configuration.
    #[must_use]
    pub fn with_config(method: ImplicitRkMethod, cfg: SdirkConfig) -> Self {
        Self { method, cfg }
    }

    fn validate(&self, u: &[f64], dt: f64) -> PdeResult<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be a finite value > 0, got {dt}"),
            });
        }
        if u.is_empty() {
            return Err(PdeError::InvalidParameter {
                name: "u".into(),
                reason: "state vector must be non-empty".into(),
            });
        }
        if self.cfg.tol <= 0.0 {
            return Err(PdeError::InvalidParameter {
                name: "tol".into(),
                reason: "must be positive".into(),
            });
        }
        if self.cfg.max_iter == 0 {
            return Err(PdeError::InvalidParameter {
                name: "max_iter".into(),
                reason: "must be at least 1".into(),
            });
        }
        Ok(())
    }

    /// Advance `u` in place by a single step of size `dt`.
    ///
    /// `f(t, y)` returns `dy/dt` (length `d`); `jac(t, y)` returns the `d×d`
    /// Jacobian `∂f/∂y` in row-major order (length `d²`).
    ///
    /// # Errors
    /// [`PdeError::InvalidParameter`] for bad `dt`/`u`/`cfg`,
    /// [`PdeError::DimensionMismatch`] if a closure returns the wrong length,
    /// [`PdeError::NotConverged`] if Newton fails within `cfg.max_iter`, or
    /// [`PdeError::SingularMatrix`] if a Newton system is singular.
    pub fn step<F, J>(&self, u: &mut [f64], t: f64, dt: f64, f: F, jac: J) -> PdeResult<()>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
        J: Fn(f64, &[f64]) -> Vec<f64>,
    {
        self.validate(u, dt)?;
        let d = u.len();
        let (c, a, b, s) = self.method.tableau();
        let sd = s * d;

        // Initial guess: every stage derivative equals f(t, y).
        let f0 = f(t, u);
        if f0.len() != d {
            return Err(PdeError::DimensionMismatch { a: f0.len(), b: d });
        }
        let mut k = vec![0.0_f64; sd];
        for i in 0..s {
            k[i * d..i * d + d].copy_from_slice(&f0);
        }

        let mut residual = f64::INFINITY;
        let mut converged = false;
        for _ in 0..self.cfg.max_iter {
            // Assemble the (s·d)×(s·d) Newton matrix M and RHS = −R.
            let mut m = vec![0.0_f64; sd * sd];
            let mut rhs = vec![0.0_f64; sd];

            for i in 0..s {
                // Stage argument Yᵢ = u + dt·Σⱼ aᵢⱼ Kⱼ.
                let mut yi = u.to_vec();
                for j in 0..s {
                    let aij = a[i * s + j];
                    for r in 0..d {
                        yi[r] += dt * aij * k[j * d + r];
                    }
                }
                let ti = t + c[i] * dt;
                let fi = f(ti, &yi);
                if fi.len() != d {
                    return Err(PdeError::DimensionMismatch { a: fi.len(), b: d });
                }
                let ji = jac(ti, &yi);
                if ji.len() != d * d {
                    return Err(PdeError::DimensionMismatch {
                        a: ji.len(),
                        b: d * d,
                    });
                }
                // Residual block: Rᵢ = Kᵢ − fᵢ;  RHS holds −Rᵢ.
                for r in 0..d {
                    rhs[i * d + r] = -(k[i * d + r] - fi[r]);
                }
                // Jacobian blocks: ∂Rᵢ/∂Kₗ = δᵢₗ I − dt·aᵢₗ Jᵢ.
                for l in 0..s {
                    let ail = a[i * s + l];
                    for r in 0..d {
                        for cc in 0..d {
                            let mut v = -dt * ail * ji[r * d + cc];
                            if l == i && r == cc {
                                v += 1.0;
                            }
                            m[(i * d + r) * sd + (l * d + cc)] = v;
                        }
                    }
                }
            }

            let delta = gauss_solve_dense(&mut m, &mut rhs, sd)?;
            residual = 0.0;
            for q in 0..sd {
                k[q] += delta[q];
                residual = residual.max(delta[q].abs());
            }
            if residual < self.cfg.tol {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(PdeError::NotConverged {
                iter: self.cfg.max_iter,
                residual,
            });
        }

        // Update: u_{n+1} = u + dt·Σᵢ bᵢ Kᵢ.
        for i in 0..s {
            let bi = b[i];
            for r in 0..d {
                u[r] += dt * bi * k[i * d + r];
            }
        }
        Ok(())
    }

    /// Integrate `n_steps` steps from `t0` to `t0 + n_steps·dt`.
    ///
    /// # Errors
    /// As [`Self::step`], plus [`PdeError::InvalidParameter`] when `n_steps == 0`.
    pub fn integrate<F, J>(
        &self,
        u: &mut [f64],
        t0: f64,
        dt: f64,
        n_steps: usize,
        f: F,
        jac: J,
    ) -> PdeResult<()>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
        J: Fn(f64, &[f64]) -> Vec<f64>,
    {
        self.validate(u, dt)?;
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be positive".into(),
            });
        }
        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            self.step(u, t, dt, &f, &jac)?;
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Function-pointer alias so the `decay` helper has a simple (non-`impl Fn`)
    /// return type; capture-free closures coerce to `fn` pointers, which are
    /// `Copy` and so can be passed by value to the generic integrator and reused.
    type RhsFn = fn(f64, &[f64]) -> Vec<f64>;

    fn decay() -> (RhsFn, RhsFn) {
        (
            |_t: f64, y: &[f64]| vec![-y[0]],
            |_t: f64, _y: &[f64]| vec![-1.0],
        )
    }

    // ── Tableau sanity ────────────────────────────────────────────────────────

    #[test]
    fn tableau_consistency() {
        for method in [
            ImplicitRkMethod::GaussLegendre4,
            ImplicitRkMethod::RadauIia5,
        ] {
            let (c, a, b, s) = method.tableau();
            // b must sum to 1.
            let bsum: f64 = b.iter().sum();
            assert!((bsum - 1.0).abs() < 1e-13, "Σb = {bsum}");
            // Row sums of A equal c (collocation consistency).
            for i in 0..s {
                let rowsum: f64 = (0..s).map(|j| a[i * s + j]).sum();
                assert!(
                    (rowsum - c[i]).abs() < 1e-13,
                    "row {i}: {rowsum} vs {}",
                    c[i]
                );
            }
        }
        // Radau IIA is stiffly accurate: b equals the last row of A.
        let (_c, a, b, s) = ImplicitRkMethod::RadauIia5.tableau();
        for j in 0..s {
            assert!((a[(s - 1) * s + j] - b[j]).abs() < 1e-13);
        }
        assert!((ImplicitRkMethod::RadauIia5.tableau().0[2] - 1.0).abs() < 1e-13);
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn validation_errors() {
        let solver = ImplicitRk::new(ImplicitRkMethod::GaussLegendre4);
        let (f, j) = decay();
        let mut u = vec![1.0];
        assert!(solver.step(&mut u, 0.0, 0.0, f, j).is_err()); // dt = 0
        let mut empty: Vec<f64> = vec![];
        assert!(solver.step(&mut empty, 0.0, 0.1, f, j).is_err());
        assert!(solver.integrate(&mut u, 0.0, 0.1, 0, f, j).is_err()); // n_steps = 0
    }

    // ── Accuracy ──────────────────────────────────────────────────────────────

    #[test]
    fn gauss_exponential_decay() {
        // y' = −y, y(1) = exp(−1); order-4 method, error well below 1e-6.
        let solver = ImplicitRk::new(ImplicitRkMethod::GaussLegendre4);
        let (f, j) = decay();
        let mut u = vec![1.0];
        solver
            .integrate(&mut u, 0.0, 0.1, 10, f, j)
            .expect("integrate");
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1e-6,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn radau_exponential_decay() {
        // y' = −y, y(1) = exp(−1); order-5 method, error far below 1e-7.
        let solver = ImplicitRk::new(ImplicitRkMethod::RadauIia5);
        let (f, j) = decay();
        let mut u = vec![1.0];
        solver
            .integrate(&mut u, 0.0, 0.1, 10, f, j)
            .expect("integrate");
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1e-8,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn radau_stiff_stable() {
        // y' = −λy with λ = 100 and dt = 0.1 (λ·dt = 10 ≫ 2). Explicit Euler
        // would blow up; the L-stable Radau scheme stays bounded and tracks the
        // exact decay exp(−λ t).
        let solver = ImplicitRk::new(ImplicitRkMethod::RadauIia5);
        let lambda = 100.0;
        let f = |_t: f64, y: &[f64]| vec![-lambda * y[0]];
        let jac = |_t: f64, _y: &[f64]| vec![-lambda];
        let mut u = vec![1.0];
        let dt = 0.1;
        let n = 5usize; // t_final = 0.5
        solver
            .integrate(&mut u, 0.0, dt, n, f, jac)
            .expect("integrate");
        let t_final = dt * n as f64;
        let expected = (-lambda * t_final).exp();
        assert!(u[0].is_finite() && u[0].abs() < 1.0, "u = {}", u[0]);
        assert!(
            (u[0] - expected).abs() < 1e-2,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn gauss_fourth_order_in_dt() {
        // Halving dt cuts the global error by ≈2⁴ = 16×.
        let solver = ImplicitRk::new(ImplicitRkMethod::GaussLegendre4);
        let (f, j) = decay();
        let exact = (-1.0_f64).exp();

        let mut u1 = vec![1.0];
        solver.integrate(&mut u1, 0.0, 0.25, 4, f, j).expect("ok");
        let e1 = (u1[0] - exact).abs();
        let mut u2 = vec![1.0];
        solver.integrate(&mut u2, 0.0, 0.125, 8, f, j).expect("ok");
        let e2 = (u2[0] - exact).abs();

        let ratio = e1 / e2.max(1e-15);
        assert!(ratio > 12.0, "expected ≈16× reduction, got {ratio:.2}");
    }

    #[test]
    fn radau_fifth_order_in_dt() {
        // Halving dt cuts the global error by ≈2⁵ = 32×.
        let solver = ImplicitRk::new(ImplicitRkMethod::RadauIia5);
        let (f, j) = decay();
        let exact = (-1.0_f64).exp();

        let mut u1 = vec![1.0];
        solver.integrate(&mut u1, 0.0, 0.25, 4, f, j).expect("ok");
        let e1 = (u1[0] - exact).abs();
        let mut u2 = vec![1.0];
        solver.integrate(&mut u2, 0.0, 0.125, 8, f, j).expect("ok");
        let e2 = (u2[0] - exact).abs();

        let ratio = e1 / e2.max(1e-15);
        assert!(ratio > 20.0, "expected ≈32× reduction, got {ratio:.2}");
    }

    // ── Systems ───────────────────────────────────────────────────────────────

    #[test]
    fn gauss_harmonic_oscillator_energy() {
        // [q, p]' = [p, −q]; the symplectic Gauss method conserves the energy
        // ½(p² + q²) almost exactly over a full period.
        let solver = ImplicitRk::new(ImplicitRkMethod::GaussLegendre4);
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let jac = |_t: f64, _y: &[f64]| vec![0.0, 1.0, -1.0, 0.0];
        let mut u = vec![1.0, 0.0];
        let n = 200usize;
        let dt = 2.0 * PI / n as f64;
        solver
            .integrate(&mut u, 0.0, dt, n, f, jac)
            .expect("integrate");
        let energy = 0.5 * (u[0] * u[0] + u[1] * u[1]);
        assert!((energy - 0.5).abs() < 1e-4, "energy = {energy}");
        assert!((u[0] - 1.0).abs() < 1e-2, "q(2π) = {}", u[0]);
        assert!(u[1].abs() < 1e-2, "p(2π) = {}", u[1]);
    }

    #[test]
    fn radau_two_dimensional_stiff_system() {
        // Decoupled stiff system y₀' = −50 y₀, y₁' = −y₁.
        let solver = ImplicitRk::new(ImplicitRkMethod::RadauIia5);
        let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0], -y[1]];
        let jac = |_t: f64, _y: &[f64]| vec![-50.0, 0.0, 0.0, -1.0];
        let mut u = vec![1.0, 2.0];
        solver
            .integrate(&mut u, 0.0, 0.05, 20, f, jac)
            .expect("integrate");
        let t = 1.0;
        assert!((u[0] - (-50.0_f64 * t).exp()).abs() < 1e-3, "u0 = {}", u[0]);
        assert!((u[1] - 2.0 * (-t).exp()).abs() < 1e-3, "u1 = {}", u[1]);
    }

    #[test]
    fn nonlinear_logistic_growth() {
        // y' = y(1 − y), y(0) = 0.5, exact y(t) = 1/(1 + e^{−t}); exercises the
        // nonlinear Newton coupling (Jacobian 1 − 2y).
        let solver = ImplicitRk::new(ImplicitRkMethod::RadauIia5);
        let f = |_t: f64, y: &[f64]| vec![y[0] * (1.0 - y[0])];
        let jac = |_t: f64, y: &[f64]| vec![1.0 - 2.0 * y[0]];
        let mut u = vec![0.5];
        solver
            .integrate(&mut u, 0.0, 0.1, 20, f, jac)
            .expect("integrate");
        let expected = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!(
            (u[0] - expected).abs() < 1e-5,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    // ── Consistency / failure paths ───────────────────────────────────────────

    #[test]
    fn step_matches_integrate_loop() {
        let solver = ImplicitRk::new(ImplicitRkMethod::GaussLegendre4);
        let (f, j) = decay();
        let dt = 0.05;
        let mut u_loop = vec![1.0];
        for kstep in 0..10usize {
            solver
                .step(&mut u_loop, kstep as f64 * dt, dt, f, j)
                .expect("ok");
        }
        let mut u_int = vec![1.0];
        solver.integrate(&mut u_int, 0.0, dt, 10, f, j).expect("ok");
        assert!((u_loop[0] - u_int[0]).abs() < 1e-14);
    }

    #[test]
    fn newton_not_converged_with_one_iteration() {
        // One Newton iteration cannot satisfy the |ΔK|-based stopping test, so
        // the solve reports NotConverged.
        let cfg = SdirkConfig {
            tol: 1e-12,
            max_iter: 1,
        };
        let solver = ImplicitRk::with_config(ImplicitRkMethod::RadauIia5, cfg);
        let (f, j) = decay();
        let mut u = vec![1.0];
        let res = solver.step(&mut u, 0.0, 0.5, f, j);
        assert!(matches!(res, Err(PdeError::NotConverged { .. })));
    }
}
