//! Singly Diagonally Implicit Runge–Kutta (SDIRK) integrator — 3-stage, order 3,
//! A- and L-stable, stiffly accurate (Crouzeix's method).
//!
//! Every diagonal entry of the Butcher matrix equals a single value `γ`, the
//! smallest real root of
//!
//! ```text
//! t³ - 3 t² + (3/2) t - 1/6 = 0,    γ ≈ 0.4358665215,
//! ```
//!
//! which makes the method A- and L-stable with classical order 3. The tableau is
//!
//! ```text
//! γ           | γ          0    0
//! (1+γ)/2     | (1-γ)/2    γ    0
//! 1           | b₁         b₂   γ
//! ------------+------------------------
//!             | b₁         b₂   γ
//! ```
//!
//! with `b₁ = -(6γ²-16γ+1)/4`, `b₂ = (6γ²-20γ+5)/4`. Because the diagonal is
//! constant the `s = 3` stages are solved *sequentially* — each is a single
//! `d × d` implicit equation `Yᵢ = sᵢ + h γ f(tᵢ, Yᵢ)` — and they all share the
//! iteration matrix `M = I_d - h γ J`, factorised once per step (simplified
//! Newton). The method is stiffly accurate (`bᵀ` = last row of `A`, `c_s = 1`),
//! so `y_{n+1} = Y_s`.
//!
//! Reference: M. Crouzeix (1975/1980); E. Hairer and G. Wanner, *Solving Ordinary
//! Differential Equations II*, 2nd ed., Springer (1996), §IV.6.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

use super::finite_diff_jacobian;

/// Number of SDIRK stages.
const STAGES: usize = 3;

/// Diagonal coefficient γ: smallest real root of `t³ - 3t² + (3/2)t - 1/6`.
const GAMMA: f64 = 0.435_866_521_508_458_999_4;

/// Configuration for the [`Sdirk`] integrator.
#[derive(Debug, Clone, Copy)]
pub struct SdirkConfig {
    /// L2 tolerance on the per-stage Newton residual.
    pub newton_tol: f64,
    /// Maximum Newton iterations per stage.
    pub max_newton_iter: usize,
    /// Relative perturbation used for the forward-difference Jacobian.
    pub fd_eps: f64,
}

impl Default for SdirkConfig {
    fn default() -> Self {
        Self {
            newton_tol: 1.0e-10,
            max_newton_iter: 50,
            fd_eps: 1.0e-7,
        }
    }
}

/// 3-stage, order-3, L-stable SDIRK integrator for stiff initial-value problems.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sdirk {
    /// Solver configuration.
    pub config: SdirkConfig,
}

impl Sdirk {
    /// Create an integrator with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an integrator with the supplied configuration.
    pub fn with_config(config: SdirkConfig) -> Self {
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
                "SDIRK: t0 and t_end must be finite".into(),
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
                "SDIRK: t_end must be ≥ t0 (forward integration with h > 0)".into(),
            ));
        }
        let n_steps = (total / h).ceil().max(1.0) as usize;
        let h_step = total / n_steps as f64;

        let (a, c) = butcher_tableau();
        let mut t = t0;
        let mut y = y0.to_vec();
        for _ in 0..n_steps {
            y = self.sdirk_step(f, jac, &a, &c, t, &y, h_step, dim)?;
            t += h_step;
            times.push(t);
            traj.push(y.clone());
        }
        Ok((times, traj))
    }

    /// Perform a single SDIRK step from `(t, y)` with step size `h`.
    fn sdirk_step<F, J>(
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
        let jn = jac(t, y)?;
        if jn.len() != dim * dim {
            return Err(NumericError::ShapeMismatch {
                expected: vec![dim, dim],
                got: vec![jn.len()],
            });
        }
        // Shared iteration matrix M = I_d - h γ Jₙ, factorised once for all stages.
        let mut m = vec![0.0_f64; dim * dim];
        for p in 0..dim {
            for q in 0..dim {
                let mut val = -h * GAMMA * jn[p * dim + q];
                if p == q {
                    val += 1.0;
                }
                m[p * dim + q] = val;
            }
        }
        let (lu, piv, _) = lu_decompose(&m, dim)?;

        let mut fstage: Vec<Vec<f64>> = Vec::with_capacity(STAGES);
        let mut stage_vals: Vec<Vec<f64>> = Vec::with_capacity(STAGES);
        for i in 0..STAGES {
            // Explicit accumulation sᵢ = yₙ + h Σ_{j<i} aᵢⱼ Fⱼ.
            let mut si = y.to_vec();
            for j in 0..i {
                for p in 0..dim {
                    si[p] += h * a[i][j] * fstage[j][p];
                }
            }
            let ti = t + c[i] * h;
            // Newton solve of g(Yᵢ) = Yᵢ - sᵢ - h γ f(tᵢ, Yᵢ) = 0.
            let mut yi = si.clone();
            let mut last_res = f64::INFINITY;
            let mut converged_fi: Option<Vec<f64>> = None;
            for _ in 0..self.config.max_newton_iter {
                let fi = f(ti, &yi);
                if fi.len() != dim {
                    return Err(NumericError::ShapeMismatch {
                        expected: vec![dim],
                        got: vec![fi.len()],
                    });
                }
                let mut gv = vec![0.0_f64; dim];
                for p in 0..dim {
                    gv[p] = yi[p] - si[p] - h * GAMMA * fi[p];
                }
                let gnorm = gv.iter().map(|x| x * x).sum::<f64>().sqrt();
                last_res = gnorm;
                if gnorm <= self.config.newton_tol {
                    converged_fi = Some(fi);
                    break;
                }
                let delta = lu_solve(&lu, &piv, dim, &gv)?;
                for p in 0..dim {
                    yi[p] -= delta[p];
                }
                if !yi.iter().all(|v| v.is_finite()) {
                    return Err(NumericError::NumericalInstability(format!(
                        "SDIRK: Newton iteration diverged at t={t}, stage {i}"
                    )));
                }
            }
            let fi = match converged_fi {
                Some(fi) => fi,
                None => {
                    return Err(NumericError::NotConverged {
                        iter: self.config.max_newton_iter,
                        residual: last_res,
                    });
                }
            };
            fstage.push(fi);
            stage_vals.push(yi);
        }
        // Stiffly accurate: y_{n+1} = Y_s (the last stage value).
        stage_vals.pop().ok_or(NumericError::EmptyInput)
    }
}

/// Butcher tableau `(A, c)` of the 3-stage Crouzeix SDIRK method.
fn butcher_tableau() -> ([[f64; STAGES]; STAGES], [f64; STAGES]) {
    let g = GAMMA;
    let g2 = g * g;
    let b1 = -(6.0 * g2 - 16.0 * g + 1.0) / 4.0;
    let b2 = (6.0 * g2 - 20.0 * g + 5.0) / 4.0;
    let a = [[g, 0.0, 0.0], [(1.0 - g) / 2.0, g, 0.0], [b1, b2, g]];
    let c = [g, (1.0 + g) / 2.0, 1.0];
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
    fn gamma_satisfies_order_cubic() {
        // γ must be a root of t³ - 3t² + (3/2)t - 1/6.
        let g = GAMMA;
        let v = g * g * g - 3.0 * g * g + 1.5 * g - 1.0 / 6.0;
        assert!(v.abs() < 1.0e-13, "cubic residual = {v:e}");
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
        let solver = Sdirk::new();
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let yf = solver.integrate(f, 0.0, &[1.0], 1.0, 0.05).expect("ok");
        let err = (yf[0] - (-1.0_f64).exp()).abs();
        assert!(err < 1.0e-4, "got {}, err={err:e}", yf[0]);
    }

    #[test]
    fn stiff_stable_where_rk4_blows_up() {
        // y' = -15 y. With h=0.2, hλ=-3 ⇒ explicit RK4 amplifies; the L-stable
        // SDIRK stays bounded and matches e^{-15} ≈ 3.06e-7.
        let solver = Sdirk::new();
        let f = |_t: f64, y: &[f64]| vec![-15.0 * y[0]];
        let yf = solver.integrate(f, 0.0, &[1.0], 1.0, 0.2).expect("ok");
        let exact = (-15.0_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1.0e-4,
            "sdirk={}, exact={}",
            yf[0],
            exact
        );
        assert!(yf[0].abs() < 1.0e-3, "must stay bounded, got {}", yf[0]);
    }

    #[test]
    fn linear_system_known_solution() {
        // y' = A y, A=[[-2,1],[1,-2]], y0=[1,0] ⇒ y(t)=½(e⁻ᵗ+e⁻³ᵗ, e⁻ᵗ-e⁻³ᵗ).
        let solver = Sdirk::new();
        let f = |_t: f64, y: &[f64]| vec![-2.0 * y[0] + y[1], y[0] - 2.0 * y[1]];
        let yf = solver
            .integrate(f, 0.0, &[1.0, 0.0], 1.0, 0.05)
            .expect("ok");
        let em1 = (-1.0_f64).exp();
        let em3 = (-3.0_f64).exp();
        let exact = [0.5 * (em1 + em3), 0.5 * (em1 - em3)];
        assert!(linf(&yf, &exact) < 1.0e-4, "yf={yf:?}, exact={exact:?}");
    }

    #[test]
    fn two_d_stiff_pair() {
        // A=[[-101,100],[100,-101]], eigenvalues -1 and -201 (stiffness 201).
        let solver = Sdirk::new();
        let f =
            |_t: f64, y: &[f64]| vec![-101.0 * y[0] + 100.0 * y[1], 100.0 * y[0] - 101.0 * y[1]];
        let yf = solver
            .integrate(f, 0.0, &[1.0, 0.0], 0.5, 0.05)
            .expect("ok");
        let slow = 0.5 * (-0.5_f64).exp();
        assert!((yf[0] - slow).abs() < 1.0e-3, "y0={}", yf[0]);
        assert!((yf[1] - slow).abs() < 1.0e-3, "y1={}", yf[1]);
        assert!(yf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn order_three_convergence() {
        // y' = -2 t y, y(0)=1 ⇒ y=e^{-t²} (non-stiff): observe order 3.
        let solver = Sdirk::new();
        let f = |t: f64, y: &[f64]| vec![-2.0 * t * y[0]];
        let exact = (-1.0_f64).exp(); // y(1) = e⁻¹
        let e_coarse = (solver.integrate(f, 0.0, &[1.0], 1.0, 0.1).expect("ok")[0] - exact).abs();
        let e_fine = (solver.integrate(f, 0.0, &[1.0], 1.0, 0.05).expect("ok")[0] - exact).abs();
        // order 3 ⇒ ≈8× reduction when halving h; require ≥4×.
        assert!(
            e_fine < e_coarse / 4.0,
            "coarse={e_coarse:e}, fine={e_fine:e}"
        );
    }

    #[test]
    fn analytic_jacobian_matches_fd() {
        let solver = Sdirk::new();
        let f = |_t: f64, y: &[f64]| vec![-15.0 * y[0]];
        let jac = |_t: f64, _y: &[f64]| vec![-15.0];
        let y_fd = solver.integrate(f, 0.0, &[1.0], 1.0, 0.2).expect("ok");
        let y_an = solver
            .integrate_with_jacobian(f, jac, 0.0, &[1.0], 1.0, 0.2)
            .expect("ok");
        assert!((y_fd[0] - y_an[0]).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_bad_step() {
        let solver = Sdirk::new();
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        assert!(solver.integrate(f, 0.0, &[1.0], 1.0, 0.0).is_err());
        assert!(solver.integrate(f, 0.0, &[1.0], 1.0, f64::NAN).is_err());
    }

    #[test]
    fn empty_state_rejected() {
        let solver = Sdirk::new();
        let f = |_t: f64, _y: &[f64]| Vec::<f64>::new();
        assert!(solver.integrate(f, 0.0, &[], 1.0, 0.1).is_err());
    }
}
