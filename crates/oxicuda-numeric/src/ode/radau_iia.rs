//! Radau IIA implicit Runge–Kutta integrator (3-stage, order 5) for stiff ODEs.
//!
//! Radau IIA is a fully-implicit, L-stable, stiffly-accurate collocation method.
//! The 3-stage member has classical order 5 and stage order 3. A step couples the
//! `s = 3` stage values through the Butcher tableau
//!
//! ```text
//! c | A           c = ((4-√6)/10, (4+√6)/10, 1)
//! --+----
//!   | bᵀ          bᵀ = last row of A   (stiffly accurate)
//! ```
//!
//! The coupled stage system is solved by *simplified Newton iteration*: the
//! Jacobian `J = ∂f/∂y` is frozen at the start of the step and the iteration
//! matrix `M = I_{s·d} - h (A ⊗ J)` is factorised once via LU and reused for every
//! Newton sweep of that step. A forward-difference Jacobian is used unless an
//! analytic one is supplied. Being stiffly accurate, `y_{n+1} = Y_s`.
//!
//! Reference: E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II:
//! Stiff and Differential-Algebraic Problems*, 2nd ed., Springer (1996), §IV.5–IV.8.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

use super::finite_diff_jacobian;

/// Number of Radau IIA stages.
const STAGES: usize = 3;

/// Configuration for the [`RadauIia`] integrator.
#[derive(Debug, Clone, Copy)]
pub struct RadauConfig {
    /// L2 tolerance on the simplified-Newton correction for stage convergence.
    pub newton_tol: f64,
    /// Maximum simplified-Newton iterations per step.
    pub max_newton_iter: usize,
    /// Relative perturbation used for the forward-difference Jacobian.
    pub fd_eps: f64,
}

impl Default for RadauConfig {
    fn default() -> Self {
        Self {
            newton_tol: 1.0e-10,
            max_newton_iter: 50,
            fd_eps: 1.0e-7,
        }
    }
}

/// 3-stage, order-5 Radau IIA integrator for stiff initial-value problems.
#[derive(Debug, Clone, Copy, Default)]
pub struct RadauIia {
    /// Solver configuration.
    pub config: RadauConfig,
}

impl RadauIia {
    /// Create an integrator with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an integrator with the supplied configuration.
    pub fn with_config(config: RadauConfig) -> Self {
        Self { config }
    }

    /// Integrate `y' = f(t, y)` from `t0` to `t_end`, returning the final state.
    ///
    /// A forward-difference Jacobian is used. The interval is covered by
    /// `ceil((t_end - t0)/h)` uniform sub-steps so integration lands exactly on
    /// `t_end`.
    pub fn integrate<F>(
        &self,
        f: F,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> NumericResult<Vec<f64>>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
    {
        let eps = self.config.fd_eps;
        let jac = |t: f64, y: &[f64]| finite_diff_jacobian(&f, t, y, eps);
        let (_, mut traj) = self.integrate_core(&f, &jac, t0, y0, t_end, h)?;
        traj.pop().ok_or(NumericError::EmptyInput)
    }

    /// Integrate with a user-supplied analytic Jacobian `jac(t, y)` (row-major
    /// `d × d`), returning the final state.
    pub fn integrate_with_jacobian<F, J>(
        &self,
        f: F,
        jac: J,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> NumericResult<Vec<f64>>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
        J: Fn(f64, &[f64]) -> Vec<f64>,
    {
        let jwrap = |t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(jac(t, y)) };
        let (_, mut traj) = self.integrate_core(&f, &jwrap, t0, y0, t_end, h)?;
        traj.pop().ok_or(NumericError::EmptyInput)
    }

    /// Integrate and return the full `(times, states)` trajectory (FD Jacobian).
    pub fn integrate_dense<F>(
        &self,
        f: F,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
    {
        let eps = self.config.fd_eps;
        let jac = |t: f64, y: &[f64]| finite_diff_jacobian(&f, t, y, eps);
        self.integrate_core(&f, &jac, t0, y0, t_end, h)
    }

    fn integrate_core<F, J>(
        &self,
        f: &F,
        jac: &J,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
        J: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    {
        if !h.is_finite() || h <= 0.0 {
            return Err(NumericError::InvalidStepSize { step: h });
        }
        if !t0.is_finite() || !t_end.is_finite() {
            return Err(NumericError::InvalidParameter(
                "Radau IIA: t0 and t_end must be finite".into(),
            ));
        }
        if y0.is_empty() {
            return Err(NumericError::EmptyInput);
        }
        let dim = y0.len();
        let total = t_end - t0;
        let mut times = vec![t0];
        let mut traj = vec![y0.to_vec()];
        if total == 0.0 {
            return Ok((times, traj));
        }
        if total < 0.0 {
            return Err(NumericError::InvalidParameter(
                "Radau IIA: t_end must be ≥ t0 (forward integration with h > 0)".into(),
            ));
        }
        let n_steps = (total / h).ceil().max(1.0) as usize;
        let h_step = total / n_steps as f64;

        let (a, c) = butcher_tableau();
        let mut t = t0;
        let mut y = y0.to_vec();
        for _ in 0..n_steps {
            y = self.radau_step(f, jac, &a, &c, t, &y, h_step, dim)?;
            t += h_step;
            times.push(t);
            traj.push(y.clone());
        }
        Ok((times, traj))
    }

    /// Perform a single Radau IIA step from `(t, y)` with step size `h`.
    fn radau_step<F, J>(
        &self,
        f: &F,
        jac: &J,
        a: &[[f64; STAGES]; STAGES],
        c: &[f64; STAGES],
        t: f64,
        y: &[f64],
        h: f64,
        dim: usize,
    ) -> NumericResult<Vec<f64>>
    where
        F: Fn(f64, &[f64]) -> Vec<f64>,
        J: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
    {
        let big = STAGES * dim;
        let jn = jac(t, y)?;
        if jn.len() != dim * dim {
            return Err(NumericError::ShapeMismatch {
                expected: vec![dim, dim],
                got: vec![jn.len()],
            });
        }

        // Iteration matrix M = I_{s·d} - h (A ⊗ Jₙ).
        let mut m = vec![0.0_f64; big * big];
        for bi in 0..STAGES {
            for bk in 0..STAGES {
                let aik = a[bi][bk];
                for p in 0..dim {
                    for q in 0..dim {
                        let mut val = -h * aik * jn[p * dim + q];
                        if bi == bk && p == q {
                            val += 1.0;
                        }
                        m[(bi * dim + p) * big + (bk * dim + q)] = val;
                    }
                }
            }
        }
        let (lu, piv, _) = lu_decompose(&m, big)?;

        // Stage values, all initialised at yₙ.
        let mut stage = vec![y.to_vec(); STAGES];
        let mut converged = false;
        let mut residual = f64::INFINITY;
        for _ in 0..self.config.max_newton_iter {
            // Stage derivatives Fᵢ = f(t + cᵢ h, Yᵢ).
            let mut fstage: Vec<Vec<f64>> = Vec::with_capacity(STAGES);
            for i in 0..STAGES {
                let fi = f(t + c[i] * h, &stage[i]);
                if fi.len() != dim {
                    return Err(NumericError::ShapeMismatch {
                        expected: vec![dim],
                        got: vec![fi.len()],
                    });
                }
                fstage.push(fi);
            }
            // Right-hand side -R where Rᵢ = Yᵢ - yₙ - h Σⱼ aᵢⱼ Fⱼ.
            let mut rhs = vec![0.0_f64; big];
            for i in 0..STAGES {
                for p in 0..dim {
                    let mut acc = stage[i][p] - y[p];
                    for j in 0..STAGES {
                        acc -= h * a[i][j] * fstage[j][p];
                    }
                    rhs[i * dim + p] = -acc;
                }
            }
            let dy = lu_solve(&lu, &piv, big, &rhs)?;
            let mut dnorm = 0.0_f64;
            for i in 0..STAGES {
                for p in 0..dim {
                    let d = dy[i * dim + p];
                    stage[i][p] += d;
                    dnorm += d * d;
                }
            }
            dnorm = dnorm.sqrt();
            if !dnorm.is_finite() {
                return Err(NumericError::NumericalInstability(format!(
                    "Radau IIA: Newton iteration diverged at t={t}"
                )));
            }
            residual = dnorm;
            if dnorm <= self.config.newton_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(NumericError::NotConverged {
                iter: self.config.max_newton_iter,
                residual,
            });
        }
        // Stiffly accurate: y_{n+1} = Y_s (the last stage).
        Ok(stage[STAGES - 1].clone())
    }
}

/// Butcher tableau `(A, c)` of the 3-stage Radau IIA method, computed from √6.
fn butcher_tableau() -> ([[f64; STAGES]; STAGES], [f64; STAGES]) {
    let s = 6.0_f64.sqrt();
    let a = [
        [
            (88.0 - 7.0 * s) / 360.0,
            (296.0 - 169.0 * s) / 1800.0,
            (-2.0 + 3.0 * s) / 225.0,
        ],
        [
            (296.0 + 169.0 * s) / 1800.0,
            (88.0 + 7.0 * s) / 360.0,
            (-2.0 - 3.0 * s) / 225.0,
        ],
        [(16.0 - s) / 36.0, (16.0 + s) / 36.0, 1.0 / 9.0],
    ];
    let c = [(4.0 - s) / 10.0, (4.0 + s) / 10.0, 1.0];
    (a, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linf(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn tableau_consistency() {
        // Row sums of A equal c, and b (last row) sums to 1.
        let (a, c) = butcher_tableau();
        for i in 0..STAGES {
            let row: f64 = a[i].iter().sum();
            assert!((row - c[i]).abs() < 1.0e-13, "row {i}");
        }
        let bsum: f64 = a[STAGES - 1].iter().sum();
        assert!((bsum - 1.0).abs() < 1.0e-13);
    }

    #[test]
    fn scalar_exact_decay() {
        // y' = -y, y(0)=1  ⇒  y(1) = e⁻¹.
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let yf = solver.integrate(f, 0.0, &[1.0], 1.0, 0.1).expect("ok");
        assert!((yf[0] - (-1.0_f64).exp()).abs() < 1.0e-9, "got {}", yf[0]);
    }

    #[test]
    fn stiff_stable_where_rk4_blows_up() {
        // y' = -15 y. With h=0.2, hλ=-3 ⇒ explicit RK4 amplifies (|R|≈1.375>1),
        // whereas L-stable Radau IIA matches e^{-15} ≈ 3.06e-7 closely.
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-15.0 * y[0]];
        let yf = solver.integrate(f, 0.0, &[1.0], 1.0, 0.2).expect("ok");
        let exact = (-15.0_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1.0e-4,
            "radau={}, exact={}",
            yf[0],
            exact
        );
        assert!(yf[0].abs() < 1.0e-3, "must stay bounded, got {}", yf[0]);
    }

    #[test]
    fn linear_system_known_solution() {
        // y' = A y, A=[[-2,1],[1,-2]], y0=[1,0].
        // Eigenvalues -1,-3 ⇒ y(t)=½(e⁻ᵗ+e⁻³ᵗ, e⁻ᵗ-e⁻³ᵗ).
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-2.0 * y[0] + y[1], y[0] - 2.0 * y[1]];
        let yf = solver.integrate(f, 0.0, &[1.0, 0.0], 1.0, 0.1).expect("ok");
        let em1 = (-1.0_f64).exp();
        let em3 = (-3.0_f64).exp();
        let exact = [0.5 * (em1 + em3), 0.5 * (em1 - em3)];
        assert!(linf(&yf, &exact) < 1.0e-7, "yf={yf:?}, exact={exact:?}");
    }

    #[test]
    fn two_d_stiff_pair() {
        // A=[[-101,100],[100,-101]], eigenvalues -1 and -201 (stiffness 201).
        // y0=[1,0] ⇒ y(t)=½e⁻ᵗ(1,1)+½e⁻²⁰¹ᵗ(1,-1); the fast mode is killed.
        let solver = RadauIia::new();
        let f =
            |_t: f64, y: &[f64]| vec![-101.0 * y[0] + 100.0 * y[1], 100.0 * y[0] - 101.0 * y[1]];
        let yf = solver
            .integrate(f, 0.0, &[1.0, 0.0], 0.5, 0.05)
            .expect("ok");
        let slow = 0.5 * (-0.5_f64).exp();
        assert!((yf[0] - slow).abs() < 1.0e-4, "y0={}", yf[0]);
        assert!((yf[1] - slow).abs() < 1.0e-4, "y1={}", yf[1]);
        assert!(yf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn order_five_convergence() {
        // y' = -2 t y, y(0)=1 ⇒ y=e^{-t²} (non-stiff): observe order 5.
        let solver = RadauIia::new();
        let f = |t: f64, y: &[f64]| vec![-2.0 * t * y[0]];
        let exact = (-1.0_f64).exp(); // y(1) = e⁻¹
        let e_coarse = (solver.integrate(f, 0.0, &[1.0], 1.0, 0.2).expect("ok")[0] - exact).abs();
        let e_fine = (solver.integrate(f, 0.0, &[1.0], 1.0, 0.1).expect("ok")[0] - exact).abs();
        // order 5 ⇒ ≈32× reduction when halving h; require ≥16×.
        assert!(
            e_fine < e_coarse / 16.0,
            "coarse={e_coarse:e}, fine={e_fine:e}"
        );
    }

    #[test]
    fn analytic_jacobian_matches_fd() {
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-15.0 * y[0]];
        let jac = |_t: f64, _y: &[f64]| vec![-15.0];
        let y_fd = solver.integrate(f, 0.0, &[1.0], 1.0, 0.2).expect("ok");
        let y_an = solver
            .integrate_with_jacobian(f, jac, 0.0, &[1.0], 1.0, 0.2)
            .expect("ok");
        assert!((y_fd[0] - y_an[0]).abs() < 1.0e-12);
    }

    #[test]
    fn dense_output_endpoints() {
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let (ts, ys) = solver
            .integrate_dense(f, 0.0, &[1.0], 1.0, 0.25)
            .expect("ok");
        assert_eq!(ts.len(), ys.len());
        assert!(ts[0].abs() < 1.0e-15);
        assert!((ts[ts.len() - 1] - 1.0).abs() < 1.0e-12);
        assert!((ys[0][0] - 1.0).abs() < 1.0e-15);
        assert!((ys[ys.len() - 1][0] - (-1.0_f64).exp()).abs() < 1.0e-5);
    }

    #[test]
    fn rejects_bad_step() {
        let solver = RadauIia::new();
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        assert!(solver.integrate(f, 0.0, &[1.0], 1.0, 0.0).is_err());
        assert!(solver.integrate(f, 0.0, &[1.0], 1.0, -0.1).is_err());
    }
}
