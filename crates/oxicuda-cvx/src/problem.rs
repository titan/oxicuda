//! Unified problem specification and generic solver dispatch.
//!
//! The convex solvers in this crate each expose a bespoke, slice-based entry
//! point (`mehrotra_predictor_corrector`, `mehrotra_qp`, `mehrotra_socp`,
//! `sdp_interior_point`, …).  That is efficient but means a caller must know,
//! up front, *which* backend a given problem maps to.  This module adds a thin
//! ergonomic layer on top: a small family of self-contained problem-data
//! structs ([`LpProblem`], [`QpProblem`], [`SocpProblem`], [`SdpProblem`]) that
//! all implement the [`ProblemSpec`] trait, plus a single generic entry point
//!
//! ```
//! use oxicuda_cvx::{solve, LpProblem};
//!
//! // min −x − y  s.t.  x + y + s = 1,  x, y, s ≥ 0   ⇒   optimum −1.
//! let lp = LpProblem::new(
//!     vec![1.0, 1.0, 1.0], 1, 3, vec![1.0], vec![-1.0, -1.0, 0.0],
//! ).unwrap();
//! let sol = solve(&lp).unwrap();
//! assert!((sol.objective + 1.0).abs() < 1e-3);
//! ```
//!
//! [`solve`] reads [`ProblemSpec::form`] to classify the program and routes it
//! to the matching interior-point / cone backend through
//! [`ProblemSpec::dispatch`], returning a uniform [`ProblemSolution`].
//!
//! Note: a second, OptNet-specific `QpProblem` lives in
//! [`crate::differentiable`]; that one carries explicit equality **and**
//! inequality blocks (`Q, q, A, b, G, h`) for differentiating through the KKT
//! system.  The [`QpProblem`] here is the standard interior-point form
//! `min ½xᵀPx + qᵀx s.t. Ax = b, x ≥ 0` and is reached as
//! `oxicuda_cvx::problem::QpProblem` to keep the two names unambiguous.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{dot, mat_vec};
use crate::lp::mehrotra_predictor_corrector;
use crate::qp::mehrotra_qp;
use crate::sdp::sdp_interior_point;
use crate::socp::{MehrotraSocpConfig, mehrotra_socp};

/// Default interior-point iteration cap for the generic LP / QP dispatch.
const DEFAULT_MAX_ITER: usize = 200;
/// Default convergence tolerance for the generic LP / QP dispatch.
const DEFAULT_TOL: f64 = 1.0e-9;
/// Default iteration cap for the generic SDP dispatch.
const DEFAULT_SDP_MAX_ITER: usize = 200;
/// Default convergence tolerance for the generic SDP dispatch.
const DEFAULT_SDP_TOL: f64 = 1.0e-7;

/// Canonical convex-program family a [`ProblemSpec`] represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemForm {
    /// Linear program `min cᵀx s.t. Ax = b, x ≥ 0`.
    Lp,
    /// Quadratic program `min ½xᵀPx + qᵀx s.t. Ax = b, x ≥ 0`.
    Qp,
    /// Second-order cone program `min cᵀx s.t. Ax = b, x ∈ K`.
    Socp,
    /// Semidefinite program `min tr(CX) s.t. tr(AₖX) = bₖ, X ⪰ 0`.
    Sdp,
}

/// Uniform optimum returned by the generic [`solve`] dispatch.
#[derive(Debug, Clone)]
pub struct ProblemSolution {
    /// Primal optimum.  For [`ProblemForm::Sdp`] this is the row-major `n × n`
    /// matrix `X`; for the other forms it is the primal vector `x`.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub objective: f64,
    /// Iterations performed by the underlying backend.
    pub iter: usize,
    /// The form that produced this solution (mirrors [`ProblemSpec::form`]).
    pub form: ProblemForm,
}

/// A convex program that can classify itself ([`form`](ProblemSpec::form)) and
/// hand itself to the matching backend ([`dispatch`](ProblemSpec::dispatch)).
///
/// Prefer the free function [`solve`] over calling [`dispatch`](ProblemSpec::dispatch)
/// directly; `solve` is the generic entry point and keeps the [`ProblemForm`]
/// consistent with the returned [`ProblemSolution`].
pub trait ProblemSpec {
    /// The canonical family this problem belongs to.
    fn form(&self) -> ProblemForm;

    /// Solve the problem with the backend selected for its [`form`](ProblemSpec::form).
    ///
    /// # Errors
    ///
    /// Propagates the underlying solver's error (bad dimensions, singular KKT
    /// system, non-convergence, …).
    fn dispatch(&self) -> CvxResult<ProblemSolution>;
}

/// Generic convex-program entry point.
///
/// Routes `problem` to the backend matching its [`ProblemSpec::form`] and
/// returns a uniform [`ProblemSolution`].
///
/// # Errors
///
/// Propagates any error returned by the selected backend.
pub fn solve(problem: &impl ProblemSpec) -> CvxResult<ProblemSolution> {
    // `form()` is the routing key; `dispatch()` invokes the concrete backend
    // and stamps the same form onto the result, which we assert stays in sync.
    let solution = problem.dispatch()?;
    debug_assert_eq!(
        solution.form,
        problem.form(),
        "dispatch returned a solution whose form disagrees with ProblemSpec::form",
    );
    Ok(solution)
}

// ───────────────────────────── Linear program ──────────────────────────────

/// Standard-form linear program `min cᵀx s.t. Ax = b, x ≥ 0`.
///
/// `a` is the `m × n` constraint matrix in row-major order.  The optional
/// `initial_basis` is consumed only by the simplex path of
/// [`crate::LpSolverBuilder`]; the generic [`solve`] dispatch uses the
/// Mehrotra predictor-corrector interior-point method and ignores it.
#[derive(Debug, Clone)]
pub struct LpProblem {
    /// Constraint matrix `A`, row-major `m × n`.
    pub a: Vec<f64>,
    /// Number of equality constraints `m`.
    pub m: usize,
    /// Number of primal variables `n`.
    pub n: usize,
    /// Right-hand side `b`, length `m`.
    pub b: Vec<f64>,
    /// Linear cost `c`, length `n`.
    pub c: Vec<f64>,
    /// Optional starting basis (size `m`) for the simplex path.
    pub initial_basis: Option<Vec<usize>>,
}

impl LpProblem {
    /// Build and dimension-check a standard-form LP.
    ///
    /// # Errors
    ///
    /// [`CvxError::InvalidParameter`] when `n == 0`, and
    /// [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] when `a`,
    /// `b`, or `c` are inconsistent with `(m, n)`.
    pub fn new(a: Vec<f64>, m: usize, n: usize, b: Vec<f64>, c: Vec<f64>) -> CvxResult<Self> {
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "LpProblem requires at least one variable (n >= 1)".into(),
            ));
        }
        if a.len() != m * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![a.len()],
            });
        }
        if b.len() != m {
            return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
        }
        if c.len() != n {
            return Err(CvxError::DimensionMismatch { a: c.len(), b: n });
        }
        Ok(Self {
            a,
            m,
            n,
            b,
            c,
            initial_basis: None,
        })
    }

    /// Attach an explicit starting basis (size `m`) for the simplex path.
    #[must_use]
    pub fn with_basis(mut self, basis: Vec<usize>) -> Self {
        self.initial_basis = Some(basis);
        self
    }
}

impl ProblemSpec for LpProblem {
    fn form(&self) -> ProblemForm {
        ProblemForm::Lp
    }

    fn dispatch(&self) -> CvxResult<ProblemSolution> {
        let res = mehrotra_predictor_corrector(
            &self.a,
            self.m,
            self.n,
            &self.b,
            &self.c,
            DEFAULT_MAX_ITER,
            DEFAULT_TOL,
        )?;
        let objective = dot(&self.c, &res.x)?;
        Ok(ProblemSolution {
            x: res.x,
            objective,
            iter: res.iter,
            form: ProblemForm::Lp,
        })
    }
}

// ─────────────────────────── Quadratic program ──────────────────────────────

/// Standard-form quadratic program `min ½xᵀPx + qᵀx s.t. Ax = b, x ≥ 0`.
///
/// This is the interior-point form solved by [`fn@crate::qp::mehrotra_qp`].  For
/// the richer equality-and-inequality form used to differentiate through the
/// KKT system, see [`crate::differentiable::QpProblem`].
#[derive(Debug, Clone)]
pub struct QpProblem {
    /// Symmetric PSD Hessian `P`, row-major `n × n`.
    pub p_mat: Vec<f64>,
    /// Number of primal variables `n`.
    pub n: usize,
    /// Linear cost `q`, length `n`.
    pub q: Vec<f64>,
    /// Constraint matrix `A`, row-major `m × n`.
    pub a: Vec<f64>,
    /// Number of equality constraints `m`.
    pub m: usize,
    /// Right-hand side `b`, length `m`.
    pub b: Vec<f64>,
}

impl QpProblem {
    /// Build and dimension-check a standard-form QP.
    ///
    /// # Errors
    ///
    /// [`CvxError::InvalidParameter`] when `n == 0`, and
    /// [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] when any
    /// datum is inconsistent with `(m, n)`.
    pub fn new(
        p_mat: Vec<f64>,
        n: usize,
        q: Vec<f64>,
        a: Vec<f64>,
        m: usize,
        b: Vec<f64>,
    ) -> CvxResult<Self> {
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "QpProblem requires at least one variable (n >= 1)".into(),
            ));
        }
        if p_mat.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![p_mat.len()],
            });
        }
        if q.len() != n {
            return Err(CvxError::DimensionMismatch { a: q.len(), b: n });
        }
        if a.len() != m * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![a.len()],
            });
        }
        if b.len() != m {
            return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
        }
        Ok(Self {
            p_mat,
            n,
            q,
            a,
            m,
            b,
        })
    }
}

impl ProblemSpec for QpProblem {
    fn form(&self) -> ProblemForm {
        ProblemForm::Qp
    }

    fn dispatch(&self) -> CvxResult<ProblemSolution> {
        let res = mehrotra_qp(
            &self.p_mat,
            self.n,
            &self.q,
            &self.a,
            self.m,
            &self.b,
            DEFAULT_MAX_ITER,
            DEFAULT_TOL,
        )?;
        // objective = ½ xᵀ P x + qᵀ x.
        let px = mat_vec(&self.p_mat, self.n, self.n, &res.x)?;
        let objective = 0.5 * dot(&res.x, &px)? + dot(&self.q, &res.x)?;
        Ok(ProblemSolution {
            x: res.x,
            objective,
            iter: res.iter,
            form: ProblemForm::Qp,
        })
    }
}

// ────────────────────────── Second-order cone program ───────────────────────

/// Second-order cone program `min cᵀx s.t. Ax = b, x ∈ K`, where
/// `K = K₁ × … × K_p` is a product of Lorentz cones described by `cone_dims`.
///
/// Dispatched to [`fn@crate::socp::mehrotra_socp`] with a default
/// [`MehrotraSocpConfig`].
#[derive(Debug, Clone)]
pub struct SocpProblem {
    /// Constraint matrix `A`, row-major `m × n`.
    pub a: Vec<f64>,
    /// Number of equality constraints `m`.
    pub m: usize,
    /// Number of primal variables `n` (must equal `Σ cone_dims`).
    pub n: usize,
    /// Right-hand side `b`, length `m`.
    pub b: Vec<f64>,
    /// Linear cost `c`, length `n`.
    pub c: Vec<f64>,
    /// Dimensions of the product second-order cones (each `≥ 1`).
    pub cone_dims: Vec<usize>,
}

impl SocpProblem {
    /// Build and dimension-check an SOCP.
    ///
    /// # Errors
    ///
    /// [`CvxError::InvalidParameter`] when `n == 0` or `cone_dims` is empty, and
    /// [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] when any
    /// datum (including `Σ cone_dims = n`) is inconsistent.
    pub fn new(
        a: Vec<f64>,
        m: usize,
        n: usize,
        b: Vec<f64>,
        c: Vec<f64>,
        cone_dims: Vec<usize>,
    ) -> CvxResult<Self> {
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "SocpProblem requires at least one variable (n >= 1)".into(),
            ));
        }
        if a.len() != m * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![a.len()],
            });
        }
        if b.len() != m {
            return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
        }
        if c.len() != n {
            return Err(CvxError::DimensionMismatch { a: c.len(), b: n });
        }
        if cone_dims.is_empty() {
            return Err(CvxError::InvalidParameter("cone_dims is empty".into()));
        }
        let cone_sum: usize = cone_dims.iter().sum();
        if cone_sum != n {
            return Err(CvxError::DimensionMismatch { a: cone_sum, b: n });
        }
        Ok(Self {
            a,
            m,
            n,
            b,
            c,
            cone_dims,
        })
    }
}

impl ProblemSpec for SocpProblem {
    fn form(&self) -> ProblemForm {
        ProblemForm::Socp
    }

    fn dispatch(&self) -> CvxResult<ProblemSolution> {
        let cfg = MehrotraSocpConfig::default();
        let res = mehrotra_socp(
            &self.a,
            self.m,
            self.n,
            &self.b,
            &self.c,
            &self.cone_dims,
            &cfg,
        )?;
        let objective = dot(&self.c, &res.x)?;
        Ok(ProblemSolution {
            x: res.x,
            objective,
            iter: res.iter,
            form: ProblemForm::Socp,
        })
    }
}

// ───────────────────────────── Semidefinite program ─────────────────────────

/// Semidefinite program `min tr(CX) s.t. tr(AₖX) = bₖ, X ⪰ 0` with symmetric
/// `n × n` matrices.
///
/// Dispatched to [`fn@crate::sdp::sdp_interior_point`].
#[derive(Debug, Clone)]
pub struct SdpProblem {
    /// Cost matrix `C`, row-major `n × n`.
    pub c: Vec<f64>,
    /// Matrix dimension `n`.
    pub n: usize,
    /// Constraint matrices `Aₖ`, each row-major `n × n`.
    pub a_list: Vec<Vec<f64>>,
    /// Right-hand side `b`, length `a_list.len()`.
    pub b: Vec<f64>,
}

impl SdpProblem {
    /// Build and dimension-check an SDP.
    ///
    /// # Errors
    ///
    /// [`CvxError::InvalidParameter`] when `n == 0`, and
    /// [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] when `C`,
    /// any `Aₖ`, or `b` are inconsistent with `n`.
    pub fn new(c: Vec<f64>, n: usize, a_list: Vec<Vec<f64>>, b: Vec<f64>) -> CvxResult<Self> {
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "SdpProblem requires n >= 1".into(),
            ));
        }
        if c.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![c.len()],
            });
        }
        if b.len() != a_list.len() {
            return Err(CvxError::DimensionMismatch {
                a: b.len(),
                b: a_list.len(),
            });
        }
        for ak in &a_list {
            if ak.len() != n * n {
                return Err(CvxError::ShapeMismatch {
                    expected: vec![n, n],
                    got: vec![ak.len()],
                });
            }
        }
        Ok(Self { c, n, a_list, b })
    }
}

impl ProblemSpec for SdpProblem {
    fn form(&self) -> ProblemForm {
        ProblemForm::Sdp
    }

    fn dispatch(&self) -> CvxResult<ProblemSolution> {
        let res = sdp_interior_point(
            &self.c,
            self.n,
            &self.a_list,
            &self.b,
            DEFAULT_SDP_MAX_ITER,
            DEFAULT_SDP_TOL,
        )?;
        Ok(ProblemSolution {
            x: res.x,
            objective: res.objective,
            iter: res.iter,
            form: ProblemForm::Sdp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LP dispatch matches a direct interior-point call ────────────────────
    #[test]
    fn lp_dispatch_matches_direct() {
        // min −x − y  s.t.  x + y + s = 1,  x, y, s ≥ 0   ⇒   optimum −1.
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let c = vec![-1.0_f64, -1.0, 0.0];
        let lp = LpProblem::new(a.clone(), 1, 3, b.clone(), c.clone()).expect("valid");
        assert_eq!(lp.form(), ProblemForm::Lp);

        let via_trait = solve(&lp).expect("dispatch solves");
        let direct = mehrotra_predictor_corrector(&a, 1, 3, &b, &c, DEFAULT_MAX_ITER, DEFAULT_TOL)
            .expect("direct solves");

        assert_eq!(via_trait.x.len(), direct.x.len());
        for (g, d) in via_trait.x.iter().zip(direct.x.iter()) {
            assert!((g - d).abs() < 1e-12, "x mismatch: {g} vs {d}");
        }
        assert_eq!(via_trait.iter, direct.iter);
        assert!((via_trait.objective + 1.0).abs() < 1e-3);
    }

    // ── QP dispatch matches a direct Mehrotra-QP call ───────────────────────
    #[test]
    fn qp_dispatch_matches_direct() {
        // min ½‖x‖²  s.t.  x1 + x2 = 1, x ≥ 0   ⇒   x* = (0.5, 0.5), obj 0.25.
        let p = vec![1.0_f64, 0.0, 0.0, 1.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let qp = QpProblem::new(p.clone(), 2, q.clone(), a.clone(), 1, b.clone()).expect("valid");
        assert_eq!(qp.form(), ProblemForm::Qp);

        let via_trait = solve(&qp).expect("dispatch solves");
        let direct = mehrotra_qp(&p, 2, &q, &a, 1, &b, DEFAULT_MAX_ITER, DEFAULT_TOL)
            .expect("direct solves");

        for (g, d) in via_trait.x.iter().zip(direct.x.iter()) {
            assert!((g - d).abs() < 1e-12, "x mismatch: {g} vs {d}");
        }
        assert_eq!(via_trait.iter, direct.iter);
        assert!((via_trait.x[0] - 0.5).abs() < 1e-4);
        assert!((via_trait.objective - 0.25).abs() < 1e-3);
    }

    // ── A single generic function consumes both LP and QP via the trait ─────
    #[test]
    fn generic_dispatch_over_trait_object() {
        fn optimum<P: ProblemSpec>(p: &P) -> (ProblemForm, f64) {
            let s = solve(p).expect("solves");
            (s.form, s.objective)
        }

        let lp = LpProblem::new(vec![1.0, 1.0, 1.0], 1, 3, vec![1.0], vec![-1.0, -1.0, 0.0])
            .expect("valid");
        let qp = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            1,
            vec![1.0],
        )
        .expect("valid");

        let (lp_form, lp_obj) = optimum(&lp);
        let (qp_form, qp_obj) = optimum(&qp);
        assert_eq!(lp_form, ProblemForm::Lp);
        assert_eq!(qp_form, ProblemForm::Qp);
        assert!((lp_obj + 1.0).abs() < 1e-3);
        assert!((qp_obj - 0.25).abs() < 1e-3);
    }

    // ── SOCP dispatch matches a direct Mehrotra-SOCP call ───────────────────
    #[test]
    fn socp_dispatch_matches_direct() {
        // min x₀  s.t.  x₁ = 1, x₂ = 0, x ∈ SOC(3)   ⇒   x* ≈ (1, 1, 0).
        let a = vec![0.0_f64, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0_f64, 0.0];
        let c = vec![1.0_f64, 0.0, 0.0];
        let cone = vec![3usize];
        let socp =
            SocpProblem::new(a.clone(), 2, 3, b.clone(), c.clone(), cone.clone()).expect("valid");
        assert_eq!(socp.form(), ProblemForm::Socp);

        let via_trait = solve(&socp).expect("dispatch solves");
        let direct = mehrotra_socp(&a, 2, 3, &b, &c, &cone, &MehrotraSocpConfig::default())
            .expect("direct solves");

        for (g, d) in via_trait.x.iter().zip(direct.x.iter()) {
            assert!((g - d).abs() < 1e-12, "x mismatch: {g} vs {d}");
        }
        assert_eq!(via_trait.iter, direct.iter);
        assert!((via_trait.x[0] - 1.0).abs() < 1e-5);
        assert!((via_trait.objective - 1.0).abs() < 1e-5);
    }

    // ── SDP dispatch matches a direct interior-point call ───────────────────
    #[test]
    fn sdp_dispatch_matches_direct() {
        // min tr(C X)  s.t.  tr(X) = 1, X ⪰ 0,  with C = I   ⇒   tr(C X) = 1.
        let c = vec![1.0_f64, 0.0, 0.0, 1.0];
        let a1 = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = vec![1.0_f64];
        let sdp = SdpProblem::new(c.clone(), 2, vec![a1.clone()], b.clone()).expect("valid");
        assert_eq!(sdp.form(), ProblemForm::Sdp);

        let via_trait = solve(&sdp).expect("dispatch solves");
        let direct = sdp_interior_point(&c, 2, &[a1], &b, DEFAULT_SDP_MAX_ITER, DEFAULT_SDP_TOL)
            .expect("direct solves");

        assert_eq!(via_trait.x.len(), direct.x.len());
        for (g, d) in via_trait.x.iter().zip(direct.x.iter()) {
            assert!((g - d).abs() < 1e-12, "X mismatch: {g} vs {d}");
        }
        assert_eq!(via_trait.iter, direct.iter);
        assert!((via_trait.objective - 1.0).abs() < 1e-3);
    }

    // ── Constructors reject inconsistent dimensions ─────────────────────────
    #[test]
    fn constructors_validate_dimensions() {
        assert!(matches!(
            LpProblem::new(vec![1.0, 1.0], 1, 3, vec![1.0], vec![-1.0, -1.0, 0.0]),
            Err(CvxError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            QpProblem::new(
                vec![1.0, 0.0, 0.0],
                2,
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                1,
                vec![1.0]
            ),
            Err(CvxError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            SocpProblem::new(vec![0.0, 1.0], 1, 2, vec![1.0], vec![0.0, -1.0], vec![]),
            Err(CvxError::InvalidParameter(_))
        ));
        assert!(matches!(
            SdpProblem::new(vec![1.0, 0.0, 0.0], 2, vec![], vec![]),
            Err(CvxError::ShapeMismatch { .. })
        ));
    }
}
