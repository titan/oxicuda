//! Fluent builder over the linear-programming backends.
//!
//! The LP backends each take a different positional argument list — the
//! interior-point methods want `(a, m, n, b, c, max_iter, tol)` while the
//! revised simplex additionally needs a starting basis.  [`LpSolverBuilder`]
//! hides those differences behind a small fluent API:
//!
//! ```
//! use oxicuda_cvx::{LpSolverBuilder, LpMethod, LpProblem};
//!
//! // min −x − y  s.t.  x + y + s = 1,  x, y, s ≥ 0   ⇒   optimum −1.
//! let lp = LpProblem::new(
//!     vec![1.0, 1.0, 1.0], 1, 3, vec![1.0], vec![-1.0, -1.0, 0.0],
//! ).unwrap();
//!
//! let sol = LpSolverBuilder::new()
//!     .tolerance(1e-9)
//!     .max_iter(200)
//!     .method(LpMethod::Mehrotra)
//!     .solve(&lp)
//!     .unwrap();
//! assert!((sol.objective + 1.0).abs() < 1e-3);
//! ```
//!
//! Every setter returns `self` so calls chain, and [`LpSolverBuilder::solve`]
//! dispatches to the backend named by [`LpMethod`].

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::dot;
use crate::lp::primal_dual_lp::primal_dual_lp;
use crate::lp::revised_simplex::SimplexStatus;
use crate::lp::{mehrotra_predictor_corrector, revised_simplex};
use crate::problem::LpProblem;

/// Which LP backend [`LpSolverBuilder::solve`] dispatches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LpMethod {
    /// Revised simplex with Bland's anti-cycling rule
    /// ([`fn@crate::lp::revised_simplex`]).  Requires a feasible starting basis;
    /// when [`LpProblem::initial_basis`] is `None` the trailing `m` columns are
    /// used as a slack basis.
    Simplex,
    /// Mehrotra predictor-corrector interior-point method
    /// ([`crate::lp::mehrotra_predictor_corrector`]).  The default.
    #[default]
    Mehrotra,
    /// Long-step primal-dual interior-point method
    /// ([`fn@crate::lp::primal_dual_lp`]).
    PrimalDual,
}

/// Uniform optimum returned by [`LpSolverBuilder::solve`] regardless of backend.
#[derive(Debug, Clone)]
pub struct LpSolution {
    /// Primal optimum `x`, length `n`.
    pub x: Vec<f64>,
    /// Objective value `cᵀx`.
    pub objective: f64,
    /// Iterations performed by the backend.
    pub iter: usize,
    /// Simplex termination status — `Some(..)` only for [`LpMethod::Simplex`].
    pub status: Option<SimplexStatus>,
    /// Equality-constraint multipliers `y` — empty for the simplex path.
    pub y: Vec<f64>,
    /// Dual slacks `z` — empty for the simplex path.
    pub z: Vec<f64>,
}

/// Fluent configurator that dispatches a [`LpProblem`] to one of the LP
/// backends.
///
/// Construct with [`LpSolverBuilder::new`] (or [`Default`]), tune with the
/// chainable setters, then call [`LpSolverBuilder::solve`].
#[derive(Debug, Clone, Copy)]
pub struct LpSolverBuilder {
    tolerance: f64,
    max_iter: usize,
    method: LpMethod,
}

impl Default for LpSolverBuilder {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-9,
            max_iter: 200,
            method: LpMethod::Mehrotra,
        }
    }
}

impl LpSolverBuilder {
    /// A builder with sane defaults: tolerance `1e-9`, `200` iterations, the
    /// Mehrotra predictor-corrector method.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the convergence tolerance applied to the interior-point residuals
    /// (ignored by the simplex path, which is exact).
    #[must_use]
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the maximum iteration count.
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Select the backend ([`LpMethod::Simplex`], [`LpMethod::Mehrotra`], or
    /// [`LpMethod::PrimalDual`]).
    #[must_use]
    pub fn method(mut self, method: LpMethod) -> Self {
        self.method = method;
        self
    }

    /// The currently configured tolerance.
    #[must_use]
    pub fn current_tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The currently configured iteration cap.
    #[must_use]
    pub fn current_max_iter(&self) -> usize {
        self.max_iter
    }

    /// The currently configured backend.
    #[must_use]
    pub fn current_method(&self) -> LpMethod {
        self.method
    }

    /// Solve `problem` with the configured backend.
    ///
    /// # Errors
    ///
    /// Propagates the backend's error.  For [`LpMethod::Simplex`] returns
    /// [`CvxError::InvalidParameter`] when no basis is supplied and `n < m`
    /// (no trailing slack basis exists).
    pub fn solve(&self, problem: &LpProblem) -> CvxResult<LpSolution> {
        match self.method {
            LpMethod::Mehrotra => {
                let res = mehrotra_predictor_corrector(
                    &problem.a,
                    problem.m,
                    problem.n,
                    &problem.b,
                    &problem.c,
                    self.max_iter,
                    self.tolerance,
                )?;
                let objective = dot(&problem.c, &res.x)?;
                Ok(LpSolution {
                    x: res.x,
                    objective,
                    iter: res.iter,
                    status: None,
                    y: res.y,
                    z: res.z,
                })
            }
            LpMethod::PrimalDual => {
                let res = primal_dual_lp(
                    &problem.a,
                    problem.m,
                    problem.n,
                    &problem.b,
                    &problem.c,
                    self.max_iter,
                    self.tolerance,
                )?;
                let objective = dot(&problem.c, &res.x)?;
                Ok(LpSolution {
                    x: res.x,
                    objective,
                    iter: res.iter,
                    status: None,
                    y: res.y,
                    z: res.z,
                })
            }
            LpMethod::Simplex => {
                let basis = match &problem.initial_basis {
                    Some(b) => b.clone(),
                    None => {
                        if problem.n < problem.m {
                            return Err(CvxError::InvalidParameter(
                                "simplex requires a starting basis when n < m".into(),
                            ));
                        }
                        // Trailing m columns form the natural slack basis for a
                        // problem written as `A x + s = b`.
                        (problem.n - problem.m..problem.n).collect()
                    }
                };
                let res = revised_simplex(
                    &problem.a,
                    problem.m,
                    problem.n,
                    &problem.b,
                    &problem.c,
                    &basis,
                    self.max_iter,
                )?;
                Ok(LpSolution {
                    x: res.x,
                    objective: res.objective,
                    iter: res.iter,
                    status: Some(res.status),
                    y: Vec::new(),
                    z: Vec::new(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_lp() -> LpProblem {
        // min −x − y  s.t.  x + y + s = 1,  x, y, s ≥ 0   ⇒   optimum −1.
        LpProblem::new(
            vec![1.0_f64, 1.0, 1.0],
            1,
            3,
            vec![1.0_f64],
            vec![-1.0_f64, -1.0, 0.0],
        )
        .expect("valid")
    }

    #[test]
    fn defaults_are_sane() {
        let builder = LpSolverBuilder::new();
        assert_eq!(builder.current_method(), LpMethod::Mehrotra);
        assert!((builder.current_tolerance() - 1.0e-9).abs() < 1e-18);
        assert_eq!(builder.current_max_iter(), 200);
        // The default builder solves the LP to the known optimum.
        let sol = builder.solve(&small_lp()).expect("solves");
        assert!((sol.objective + 1.0).abs() < 1e-3, "obj={}", sol.objective);
    }

    #[test]
    fn setters_chain_and_configure() {
        let builder = LpSolverBuilder::new()
            .tolerance(1e-7)
            .max_iter(123)
            .method(LpMethod::Simplex);
        assert_eq!(builder.current_method(), LpMethod::Simplex);
        assert!((builder.current_tolerance() - 1e-7).abs() < 1e-18);
        assert_eq!(builder.current_max_iter(), 123);
    }

    #[test]
    fn mehrotra_matches_direct() {
        let lp = small_lp();
        let sol = LpSolverBuilder::new()
            .method(LpMethod::Mehrotra)
            .tolerance(1e-7)
            .max_iter(200)
            .solve(&lp)
            .expect("solves");
        let direct = mehrotra_predictor_corrector(&lp.a, lp.m, lp.n, &lp.b, &lp.c, 200, 1e-7)
            .expect("direct");
        assert_eq!(sol.x.len(), direct.x.len());
        for (b, d) in sol.x.iter().zip(direct.x.iter()) {
            assert!((b - d).abs() < 1e-12, "x mismatch: {b} vs {d}");
        }
        assert_eq!(sol.iter, direct.iter);
        let direct_obj: f64 = direct
            .x
            .iter()
            .zip(lp.c.iter())
            .map(|(xi, ci)| xi * ci)
            .sum();
        assert!((sol.objective - direct_obj).abs() < 1e-12);
    }

    #[test]
    fn primal_dual_matches_direct() {
        let lp = small_lp();
        let sol = LpSolverBuilder::new()
            .method(LpMethod::PrimalDual)
            .tolerance(1e-7)
            .max_iter(200)
            .solve(&lp)
            .expect("solves");
        let direct = primal_dual_lp(&lp.a, lp.m, lp.n, &lp.b, &lp.c, 200, 1e-7).expect("direct");
        for (b, d) in sol.x.iter().zip(direct.x.iter()) {
            assert!((b - d).abs() < 1e-12, "x mismatch: {b} vs {d}");
        }
        assert_eq!(sol.iter, direct.iter);
    }

    #[test]
    fn simplex_matches_direct() {
        let lp = small_lp().with_basis(vec![2usize]);
        let sol = LpSolverBuilder::new()
            .method(LpMethod::Simplex)
            .max_iter(100)
            .solve(&lp)
            .expect("solves");
        let direct =
            revised_simplex(&lp.a, lp.m, lp.n, &lp.b, &lp.c, &[2usize], 100).expect("direct");
        assert_eq!(sol.status, Some(SimplexStatus::Optimal));
        assert_eq!(sol.status, Some(direct.status));
        for (b, d) in sol.x.iter().zip(direct.x.iter()) {
            assert!((b - d).abs() < 1e-12, "x mismatch: {b} vs {d}");
        }
        assert!((sol.objective - direct.objective).abs() < 1e-12);
        assert_eq!(sol.iter, direct.iter);
    }

    #[test]
    fn simplex_default_basis_uses_trailing_slack() {
        // No explicit basis: the builder should pick the trailing m = 1 column.
        let sol = LpSolverBuilder::new()
            .method(LpMethod::Simplex)
            .solve(&small_lp())
            .expect("solves");
        assert_eq!(sol.status, Some(SimplexStatus::Optimal));
        assert!((sol.objective + 1.0).abs() < 1e-9);
    }

    #[test]
    fn both_methods_reach_same_optimum() {
        let lp = small_lp().with_basis(vec![2usize]);
        let simplex = LpSolverBuilder::new()
            .method(LpMethod::Simplex)
            .solve(&lp)
            .expect("simplex");
        let mehrotra = LpSolverBuilder::new()
            .method(LpMethod::Mehrotra)
            .solve(&lp)
            .expect("mehrotra");
        assert!((simplex.objective + 1.0).abs() < 1e-9);
        assert!((mehrotra.objective + 1.0).abs() < 1e-3);
        assert!((simplex.objective - mehrotra.objective).abs() < 1e-3);
    }
}
