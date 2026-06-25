//! Differentiating through the KKT conditions of a quadratic program (OptNet).
//!
//! This module implements the differentiable-optimisation layer of Amos & Kolter
//! (2017), *"OptNet: Differentiable Optimization as a Layer in Neural Networks"*.
//! It treats the solution of a (convex) quadratic program as an implicit function
//! of the problem data and differentiates it via the **implicit function theorem
//! applied to the Karush–Kuhn–Tucker (KKT) system**.
//!
//! # Problem
//!
//! ```text
//!   minimise_z   ½ zᵀ Q z + qᵀ z
//!   subject to   A z = b           (equality,   ν multipliers)
//!                G z ≤ h           (inequality, λ ≥ 0 multipliers)
//! ```
//!
//! with `Q ⪰ 0`.  At a primal–dual optimum `(z*, ν*, λ*)` the KKT conditions are
//!
//! ```text
//!   Q z* + q + Aᵀ ν* + Gᵀ λ* = 0          (stationarity)
//!   A z* − b = 0                           (primal feasibility, equalities)
//!   diag(λ*) (G z* − h) = 0                (complementary slackness)
//! ```
//!
//! # Backward pass
//!
//! Differentiating the KKT residual map and applying the implicit function
//! theorem, OptNet's backward pass solves the **transposed** KKT system
//!
//! ```text
//!   ⎡ Q          Gᵀ diag(λ*)   Aᵀ ⎤ ⎡ d_z ⎤     ⎡ (∂ℓ/∂z*)ᵀ ⎤
//!   ⎢ G          diag(Gz*−h)    0 ⎥ ⎢ d_λ ⎥ = − ⎢     0      ⎥
//!   ⎣ A               0         0 ⎦ ⎣ d_ν ⎦     ⎣     0      ⎦
//! ```
//!
//! for `(d_z, d_λ, d_ν)` given an upstream gradient `∂ℓ/∂z*`, then assembles the
//! parameter gradients via outer products:
//!
//! ```text
//!   ∇_Q ℓ = ½ (d_z z*ᵀ + z* d_zᵀ)
//!   ∇_q ℓ = d_z
//!   ∇_A ℓ = d_ν z*ᵀ + ν* d_zᵀ
//!   ∇_b ℓ = − d_ν
//!   ∇_G ℓ = diag(λ*) (d_λ z*ᵀ + λ* d_zᵀ)
//!   ∇_h ℓ = − diag(λ*) d_λ
//! ```
//!
//! The (3,3) saddle block above is generally indefinite, so the assembled matrix
//! is regularised with a tiny diagonal shift before being factorised with the
//! crate's dense LU solver; this keeps the backward solve robust when an
//! inequality is exactly active (`Gz* − h = 0`) or the active set is degenerate.
//!
//! All computation is in pure Rust with no external linear-algebra dependency;
//! matrices are stored row-major as flat `Vec<f64>`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// Quadratic-program data in OptNet form.
///
/// ```text
///   minimise_z   ½ zᵀ Q z + qᵀ z   s.t.   A z = b,   G z ≤ h
/// ```
///
/// All matrices are row-major. Equality (`A`, `b`) and inequality (`G`, `h`)
/// blocks are optional in the sense that either may have zero rows.
#[derive(Debug, Clone)]
pub struct QpProblem {
    /// Symmetric positive-semidefinite Hessian `Q`, row-major `n × n`.
    pub q_mat: Vec<f64>,
    /// Linear cost `q`, length `n`.
    pub q_vec: Vec<f64>,
    /// Equality constraint matrix `A`, row-major `p × n` (`p` may be 0).
    pub a_mat: Vec<f64>,
    /// Equality right-hand side `b`, length `p`.
    pub b_vec: Vec<f64>,
    /// Inequality constraint matrix `G`, row-major `m × n` (`m` may be 0).
    pub g_mat: Vec<f64>,
    /// Inequality right-hand side `h`, length `m`.
    pub h_vec: Vec<f64>,
    /// Number of primal variables `n`.
    pub n: usize,
    /// Number of equality constraints `p`.
    pub p: usize,
    /// Number of inequality constraints `m`.
    pub m: usize,
}

impl QpProblem {
    /// Build a QP problem, validating all dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::InvalidParameter`] if `n == 0`, and
    /// [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] when any
    /// matrix or vector length is inconsistent with `(n, p, m)`.
    pub fn new(
        q_mat: Vec<f64>,
        q_vec: Vec<f64>,
        a_mat: Vec<f64>,
        b_vec: Vec<f64>,
        g_mat: Vec<f64>,
        h_vec: Vec<f64>,
    ) -> CvxResult<Self> {
        let n = q_vec.len();
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "QpProblem requires at least one variable (n >= 1)".into(),
            ));
        }
        if q_mat.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![q_mat.len()],
            });
        }
        let p = b_vec.len();
        if a_mat.len() != p * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![p, n],
                got: vec![a_mat.len()],
            });
        }
        let m = h_vec.len();
        if g_mat.len() != m * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![g_mat.len()],
            });
        }
        Ok(Self {
            q_mat,
            q_vec,
            a_mat,
            b_vec,
            g_mat,
            h_vec,
            n,
            p,
            m,
        })
    }
}

/// Primal–dual optimum of a [`QpProblem`].
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// Primal optimum `z*`, length `n`.
    pub z: Vec<f64>,
    /// Inequality multipliers `λ* ≥ 0`, length `m`.
    pub lam: Vec<f64>,
    /// Equality multipliers `ν*`, length `p`.
    pub nu: Vec<f64>,
    /// Number of interior-point iterations performed.
    pub iter: usize,
    /// Final surrogate duality gap `μ = (λᵀ s) / m` (`0` if `m == 0`).
    pub mu: f64,
    /// Whether the KKT residuals fell below the requested tolerance.
    pub converged: bool,
}

/// Parameter gradients returned by the OptNet backward pass.
///
/// Each field is the gradient of the scalar loss `ℓ` with respect to the
/// corresponding problem datum, in the same row-major layout as the input.
#[derive(Debug, Clone)]
pub struct QpParamGrads {
    /// `∇_Q ℓ`, row-major `n × n` (symmetric).
    pub d_q_mat: Vec<f64>,
    /// `∇_q ℓ`, length `n`.
    pub d_q_vec: Vec<f64>,
    /// `∇_A ℓ`, row-major `p × n`.
    pub d_a_mat: Vec<f64>,
    /// `∇_b ℓ`, length `p`.
    pub d_b_vec: Vec<f64>,
    /// `∇_G ℓ`, row-major `m × n`.
    pub d_g_mat: Vec<f64>,
    /// `∇_h ℓ`, length `m`.
    pub d_h_vec: Vec<f64>,
}

/// Tunable parameters for the forward interior-point solve.
#[derive(Debug, Clone)]
pub struct OptNetConfig {
    /// Maximum interior-point iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the KKT residual norms and the duality gap.
    pub tol: f64,
    /// Fraction-to-boundary parameter `η ∈ (0, 1)` for inequality step lengths.
    pub frac_to_boundary: f64,
    /// Diagonal regularisation added to the backward KKT system for stability.
    pub backward_reg: f64,
}

impl Default for OptNetConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1.0e-10,
            frac_to_boundary: 0.99,
            backward_reg: 1.0e-9,
        }
    }
}

/// A differentiable QP layer in the OptNet sense.
///
/// Construct it with [`OptNetLayer::solve`], which runs the forward
/// interior-point solve and caches the optimum, then call
/// [`OptNetLayer::backward`] with an upstream gradient `∂ℓ/∂z*` to obtain the
/// gradients of the loss with respect to every problem datum.
#[derive(Debug, Clone)]
pub struct OptNetLayer {
    problem: QpProblem,
    solution: QpSolution,
    config: OptNetConfig,
}

impl OptNetLayer {
    /// Forward pass: solve the QP and cache `(z*, λ*, ν*)`.
    ///
    /// Uses a self-contained primal–dual interior-point method (a Mehrotra-style
    /// predictor–corrector) specialised to the OptNet QP form
    /// `min ½zᵀQz+qᵀz s.t. Az=b, Gz≤h`. The Hessian `Q` is symmetrised and given
    /// a small diagonal floor so that strictly positive-semidefinite (including
    /// `Q = 0`) inputs yield a non-singular reduced system.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`QpProblem::new`] and returns
    /// [`CvxError::InvalidParameter`] for a non-positive tolerance,
    /// [`CvxError::SingularMatrix`] / [`CvxError::NumericalInstability`] if the
    /// interior-point linear systems break down, and [`CvxError::Infeasible`] if
    /// no strictly feasible interior iterate can be maintained.
    pub fn solve(problem: QpProblem, config: OptNetConfig) -> CvxResult<Self> {
        if config.tol <= 0.0 {
            return Err(CvxError::InvalidParameter("tol must be positive".into()));
        }
        if !(0.0..1.0).contains(&config.frac_to_boundary) || config.frac_to_boundary <= 0.0 {
            return Err(CvxError::InvalidParameter(
                "frac_to_boundary must lie in (0, 1)".into(),
            ));
        }
        let solution = solve_qp_interior_point(&problem, &config)?;
        Ok(Self {
            problem,
            solution,
            config,
        })
    }

    /// Forward pass with default configuration.
    ///
    /// # Errors
    ///
    /// See [`OptNetLayer::solve`].
    pub fn solve_default(problem: QpProblem) -> CvxResult<Self> {
        Self::solve(problem, OptNetConfig::default())
    }

    /// The cached primal–dual optimum.
    #[must_use]
    pub fn solution(&self) -> &QpSolution {
        &self.solution
    }

    /// The problem data.
    #[must_use]
    pub fn problem(&self) -> &QpProblem {
        &self.problem
    }

    /// The primal optimum `z*`.
    #[must_use]
    pub fn z(&self) -> &[f64] {
        &self.solution.z
    }

    /// Backward pass: differentiate the loss through the KKT system.
    ///
    /// Given the upstream gradient `grad_z = ∂ℓ/∂z*` (length `n`), assemble and
    /// solve the transposed KKT system for `(d_z, d_λ, d_ν)` and form the
    /// parameter gradients. The KKT matrix is regularised with a tiny diagonal
    /// shift (`config.backward_reg`) so that active inequalities or degenerate
    /// active sets do not make it singular.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::DimensionMismatch`] if `grad_z.len() != n` and
    /// [`CvxError::SingularMatrix`] if the (regularised) KKT system still cannot
    /// be factorised.
    pub fn backward(&self, grad_z: &[f64]) -> CvxResult<QpParamGrads> {
        let n = self.problem.n;
        let p = self.problem.p;
        let m = self.problem.m;
        if grad_z.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: grad_z.len(),
                b: n,
            });
        }

        let z = &self.solution.z;
        let lam = &self.solution.lam;
        let nu = &self.solution.nu;

        // ── Assemble the transposed KKT matrix ──────────────────────────────
        //
        // Order the unknowns as [d_z (n) ; d_λ (m) ; d_ν (p)] so the system is
        //
        //   ⎡ Q            Gᵀ diag(λ)    Aᵀ ⎤ ⎡ d_z ⎤     ⎡ grad_z ⎤
        //   ⎢ diag(λ) G    diag(Gz−h)    0  ⎥ ⎢ d_λ ⎥ = − ⎢   0    ⎥ .
        //   ⎣ A             0            0  ⎦ ⎣ d_ν ⎦     ⎣   0    ⎦
        //
        // The middle block-row is the complementarity differential
        // diag(λ) G d_z + diag(Gz−h) d_λ = 0, i.e. the rows are scaled by λ
        // relative to the form printed in the module docs; this scaling makes
        // the factor diag(λ) appear directly in ∇_G and ∇_h below and keeps the
        // assembled matrix better balanced.
        let dim = n + m + p;
        let mut kkt = vec![0.0_f64; dim * dim];

        // (0,0): Q  (symmetrised).
        for i in 0..n {
            for j in 0..n {
                let qij = 0.5 * (self.problem.q_mat[i * n + j] + self.problem.q_mat[j * n + i]);
                kkt[i * dim + j] = qij;
            }
        }

        // s = G z − h  (inequality slacks at the optimum, ≤ 0 and ~0 if active).
        let gz = if m > 0 {
            mat_vec(&self.problem.g_mat, m, n, z)?
        } else {
            Vec::new()
        };

        // (0,1): Gᵀ diag(λ)   →  kkt[i, n+k] = G[k, i] · λ[k]
        // (1,0): diag(λ) G    →  kkt[n+k, i] = λ[k] · G[k, i]
        // (1,1): diag(Gz − h) →  kkt[n+k, n+k] = (Gz[k] − h[k])
        for k in 0..m {
            let lk = lam[k];
            let sk = gz[k] - self.problem.h_vec[k];
            for i in 0..n {
                let gki = self.problem.g_mat[k * n + i];
                kkt[i * dim + (n + k)] = gki * lk;
                kkt[(n + k) * dim + i] = lk * gki;
            }
            kkt[(n + k) * dim + (n + k)] = sk;
        }

        // (0,2): Aᵀ  →  kkt[i, n+m+l] = A[l, i]
        // (2,0): A   →  kkt[n+m+l, i] = A[l, i]
        for l in 0..p {
            for i in 0..n {
                let ali = self.problem.a_mat[l * n + i];
                kkt[i * dim + (n + m + l)] = ali;
                kkt[(n + m + l) * dim + i] = ali;
            }
        }

        // Diagonal regularisation for the (indefinite) saddle block. A symmetric
        // signed shift keeps the (1,1)/(2,2) negative/zero blocks invertible
        // without biasing the primal (1,1) block in the wrong direction.
        let reg = self.config.backward_reg.max(0.0);
        if reg > 0.0 {
            for i in 0..n {
                kkt[i * dim + i] += reg;
            }
            for k in 0..(m + p) {
                let idx = n + k;
                kkt[idx * dim + idx] -= reg;
            }
        }

        // ── Right-hand side: −[grad_z ; 0 ; 0] ──────────────────────────────
        let mut rhs = vec![0.0_f64; dim];
        for i in 0..n {
            rhs[i] = -grad_z[i];
        }

        // ── Solve ───────────────────────────────────────────────────────────
        let sol = solve_dense(&kkt, dim, &rhs)?;
        let d_z = &sol[..n];
        let d_lam = &sol[n..n + m];
        let d_nu = &sol[n + m..];

        // ── Assemble parameter gradients ────────────────────────────────────
        //
        //   ∇_Q ℓ = ½ (d_z z*ᵀ + z* d_zᵀ)
        let mut d_q_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                d_q_mat[i * n + j] = 0.5 * (d_z[i] * z[j] + z[i] * d_z[j]);
            }
        }

        //   ∇_q ℓ = d_z
        let d_q_vec = d_z.to_vec();

        //   ∇_A ℓ = d_ν z*ᵀ + ν* d_zᵀ
        let mut d_a_mat = vec![0.0_f64; p * n];
        for l in 0..p {
            for j in 0..n {
                d_a_mat[l * n + j] = d_nu[l] * z[j] + nu[l] * d_z[j];
            }
        }

        //   ∇_b ℓ = − d_ν
        let d_b_vec: Vec<f64> = d_nu.iter().map(|v| -v).collect();

        //   ∇_G ℓ = diag(λ*) (d_λ z*ᵀ + λ* d_zᵀ)
        let mut d_g_mat = vec![0.0_f64; m * n];
        for k in 0..m {
            let lk = lam[k];
            for j in 0..n {
                d_g_mat[k * n + j] = lk * (d_lam[k] * z[j] + lam[k] * d_z[j]);
            }
        }

        //   ∇_h ℓ = − diag(λ*) d_λ
        let d_h_vec: Vec<f64> = (0..m).map(|k| -lam[k] * d_lam[k]).collect();

        Ok(QpParamGrads {
            d_q_mat,
            d_q_vec,
            d_a_mat,
            d_b_vec,
            d_g_mat,
            d_h_vec,
        })
    }
}

/// One interior-point Newton step `(Δz, Δs, Δλ, Δν)`.
type NewtonStep = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Maximum step `α ∈ (0, 1]` keeping `v + α dv ≥ 0` component-wise.
fn max_step_nonneg(v: &[f64], dv: &[f64]) -> f64 {
    let mut alpha = 1.0_f64;
    for (vi, dvi) in v.iter().zip(dv.iter()) {
        if *dvi < 0.0 {
            let r = -vi / dvi;
            if r < alpha {
                alpha = r;
            }
        }
    }
    alpha
}

/// Primal–dual interior-point solve of the OptNet QP form.
///
/// Solves `min ½zᵀQz+qᵀz s.t. Az=b, Gz≤h` by introducing inequality slacks
/// `s = h − Gz ≥ 0` with multipliers `λ ≥ 0` and applying a Mehrotra
/// predictor–corrector to the perturbed KKT system
///
/// ```text
///   Q z + q + Aᵀ ν + Gᵀ λ = 0
///   A z − b = 0
///   G z + s − h = 0,   s ≥ 0
///   diag(λ) s = μ 1,   λ ≥ 0   (μ → 0)
/// ```
///
/// The reduced 3×3 block system in `(Δz, Δs, Δν)` is condensed to the symmetric
/// indefinite `(n + p)` system in `(Δz, Δν)` by eliminating `Δs` and `Δλ`.
fn solve_qp_interior_point(prob: &QpProblem, cfg: &OptNetConfig) -> CvxResult<QpSolution> {
    let n = prob.n;
    let p = prob.p;
    let m = prob.m;

    // Symmetrise Q and add a tiny floor so Q ⪰ 0 (incl. Q = 0) gives an SPD-ish
    // (1,1) block once combined with the slack regulariser Gᵀ diag(λ/s) G.
    let q_floor = 1.0e-10;
    let mut q_sym = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            q_sym[i * n + j] = 0.5 * (prob.q_mat[i * n + j] + prob.q_mat[j * n + i]);
        }
        q_sym[i * n + i] += q_floor;
    }

    // ── Pure equality / unconstrained case: single linear KKT solve ─────────
    if m == 0 {
        return solve_equality_qp(prob, &q_sym, cfg);
    }

    // ── Initialise a strictly interior point for the inequalities ───────────
    let mut z = vec![0.0_f64; n];
    let mut nu = vec![0.0_f64; p];
    let mut lam = vec![1.0_f64; m];
    let mut s = vec![1.0_f64; m];

    let dim = n + p; // reduced (Δz, Δν) system size

    let mut last_mu = f64::INFINITY;
    for it in 0..cfg.max_iter {
        // Residuals.
        //   r_dual = Q z + q + Aᵀ ν + Gᵀ λ
        //   r_eq   = A z − b
        //   r_ineq = G z + s − h
        //   r_cent = λ ⊙ s
        let qz = mat_vec(&q_sym, n, n, &z)?;
        let at_nu = if p > 0 {
            mat_t_vec(&prob.a_mat, p, n, &nu)?
        } else {
            vec![0.0_f64; n]
        };
        let gt_lam = mat_t_vec(&prob.g_mat, m, n, &lam)?;
        let mut r_dual = vec![0.0_f64; n];
        for i in 0..n {
            r_dual[i] = qz[i] + prob.q_vec[i] + at_nu[i] + gt_lam[i];
        }

        let mut r_eq = vec![0.0_f64; p];
        if p > 0 {
            let az = mat_vec(&prob.a_mat, p, n, &z)?;
            for l in 0..p {
                r_eq[l] = az[l] - prob.b_vec[l];
            }
        }

        let gz = mat_vec(&prob.g_mat, m, n, &z)?;
        let mut r_ineq = vec![0.0_f64; m];
        for k in 0..m {
            r_ineq[k] = gz[k] + s[k] - prob.h_vec[k];
        }

        let mu: f64 = (0..m).map(|k| lam[k] * s[k]).sum::<f64>() / m as f64;

        // Convergence test.
        if norm2(&r_dual) < cfg.tol
            && norm2(&r_eq) < cfg.tol
            && norm2(&r_ineq) < cfg.tol
            && mu < cfg.tol
        {
            return Ok(QpSolution {
                z,
                lam,
                nu,
                iter: it,
                mu,
                converged: true,
            });
        }

        // Guard against stalls: if μ stops decreasing while residuals are tiny.
        if mu > last_mu * (1.0 + 1.0e-6)
            && norm2(&r_dual) < cfg.tol.sqrt()
            && norm2(&r_eq) < cfg.tol.sqrt()
            && norm2(&r_ineq) < cfg.tol.sqrt()
        {
            return Ok(QpSolution {
                z,
                lam,
                nu,
                iter: it,
                mu,
                converged: true,
            });
        }
        last_mu = mu;

        // ── Reduced KKT matrix M = [ Q + Gᵀ Σ G   Aᵀ ; A   0 ] ─────────────
        // with Σ = diag(λ / s).  Factorise once, reuse for predictor+corrector.
        let mut sigma = vec![0.0_f64; m];
        for k in 0..m {
            let sk = s[k].max(1.0e-14);
            sigma[k] = lam[k] / sk;
        }

        let mut m_red = vec![0.0_f64; dim * dim];
        // (0,0): Q + Gᵀ Σ G.
        for i in 0..n {
            for j in 0..n {
                m_red[i * dim + j] = q_sym[i * n + j];
            }
        }
        for (k, &sig) in sigma.iter().enumerate().take(m) {
            let base = k * n;
            for i in 0..n {
                let gki = prob.g_mat[base + i];
                if gki == 0.0 {
                    continue;
                }
                let w = sig * gki;
                for j in 0..n {
                    m_red[i * dim + j] += w * prob.g_mat[base + j];
                }
            }
        }
        // (0,1) Aᵀ and (1,0) A.
        for l in 0..p {
            for i in 0..n {
                let ali = prob.a_mat[l * n + i];
                m_red[i * dim + (n + l)] = ali;
                m_red[(n + l) * dim + i] = ali;
            }
        }
        // Tiny negative shift on the (2,2) zero block for invertibility.
        for l in 0..p {
            let idx = n + l;
            m_red[idx * dim + idx] -= 1.0e-12;
        }

        let (lu, piv) = match crate::linalg::solve::lu_decompose(&m_red, dim) {
            Ok(v) => v,
            Err(_) => {
                return Err(CvxError::NumericalInstability(
                    "reduced KKT factorisation failed in interior-point QP".into(),
                ));
            }
        };

        // Affine (predictor) step: r_cent = λ⊙s (σ = 0).
        let r_cent_aff: Vec<f64> = (0..m).map(|k| lam[k] * s[k]).collect();
        let (dz_a, ds_a, dlam_a, dnu_a) = solve_reduced_step(
            prob,
            &sigma,
            &lu,
            &piv,
            dim,
            &r_dual,
            &r_eq,
            &r_ineq,
            &r_cent_aff,
            &s,
            &lam,
        )?;

        // Affine step lengths.
        let alpha_p_aff = max_step_nonneg(&s, &ds_a);
        let alpha_d_aff = max_step_nonneg(&lam, &dlam_a);
        let alpha_aff = alpha_p_aff.min(alpha_d_aff);

        // Centering parameter via Mehrotra heuristic σ = (μ_aff/μ)³.
        let mu_aff: f64 = (0..m)
            .map(|k| (lam[k] + alpha_aff * dlam_a[k]) * (s[k] + alpha_aff * ds_a[k]))
            .sum::<f64>()
            / m as f64;
        let sigma_c = if mu < 1.0e-300 {
            0.0
        } else {
            (mu_aff / mu).powi(3).clamp(0.0, 1.0)
        };

        // Corrector step: r_cent = λ⊙s + Δλ_aff⊙Δs_aff − σμ.
        let r_cent_cor: Vec<f64> = (0..m)
            .map(|k| lam[k] * s[k] + dlam_a[k] * ds_a[k] - sigma_c * mu)
            .collect();
        let (dz, ds, dlam, dnu) = solve_reduced_step(
            prob,
            &sigma,
            &lu,
            &piv,
            dim,
            &r_dual,
            &r_eq,
            &r_ineq,
            &r_cent_cor,
            &s,
            &lam,
        )?;
        // dz_a / dnu_a are consumed only via the affine step lengths above.
        let _ = (&dz_a, &dnu_a);

        // Final step lengths (fraction-to-boundary).
        let eta = cfg.frac_to_boundary;
        let alpha_p = (eta * max_step_nonneg(&s, &ds)).min(1.0);
        let alpha_d = (eta * max_step_nonneg(&lam, &dlam)).min(1.0);

        if !alpha_p.is_finite() || !alpha_d.is_finite() {
            return Err(CvxError::NumericalInstability(
                "non-finite step length in interior-point QP".into(),
            ));
        }

        // Update iterates.
        for i in 0..n {
            z[i] += alpha_p * dz[i];
        }
        for l in 0..p {
            nu[l] += alpha_d * dnu[l];
        }
        for k in 0..m {
            s[k] += alpha_p * ds[k];
            lam[k] += alpha_d * dlam[k];
            // Keep strictly interior.
            if s[k] <= 0.0 {
                s[k] = 1.0e-12;
            }
            if lam[k] <= 0.0 {
                lam[k] = 1.0e-12;
            }
        }
    }

    // Max iterations reached: report best iterate (non-converged).
    let mu: f64 = if m > 0 {
        (0..m).map(|k| lam[k] * s[k]).sum::<f64>() / m as f64
    } else {
        0.0
    };
    Ok(QpSolution {
        z,
        lam,
        nu,
        iter: cfg.max_iter,
        mu,
        converged: false,
    })
}

/// Solve one Newton (predictor or corrector) step of the reduced interior-point
/// system, returning `(Δz, Δs, Δλ, Δν)`.
///
/// Eliminates `Δs = −r_ineq − G Δz` and `Δλ = −Σ Δs − (r_cent ⊘ s)` to reach the
/// `(n + p)` system `M [Δz; Δν] = rhs`, then back-substitutes.
#[allow(clippy::too_many_arguments)]
fn solve_reduced_step(
    prob: &QpProblem,
    sigma: &[f64],
    lu: &[f64],
    piv: &[usize],
    dim: usize,
    r_dual: &[f64],
    r_eq: &[f64],
    r_ineq: &[f64],
    r_cent: &[f64],
    s: &[f64],
    lam: &[f64],
) -> CvxResult<NewtonStep> {
    let n = prob.n;
    let p = prob.p;
    let m = prob.m;

    // Right-hand side of the reduced system:
    //   rhs_z = −r_dual + Gᵀ ( Σ r_ineq − (r_cent ⊘ s) )
    //   rhs_ν = −r_eq
    let mut w = vec![0.0_f64; m];
    for k in 0..m {
        let sk = s[k].max(1.0e-14);
        w[k] = sigma[k] * r_ineq[k] - r_cent[k] / sk;
    }
    let gt_w = if m > 0 {
        mat_t_vec(&prob.g_mat, m, n, &w)?
    } else {
        vec![0.0_f64; n]
    };

    // RHS_z = −r_dual − Gᵀ(Σ r_ineq − r_cent ⊘ s);  RHS_ν = −r_eq.
    let mut rhs = vec![0.0_f64; dim];
    for i in 0..n {
        rhs[i] = -r_dual[i] - gt_w[i];
    }
    for (l, re) in r_eq.iter().enumerate().take(p) {
        rhs[n + l] = -re;
    }

    let sol = crate::linalg::solve::lu_solve(lu, piv, dim, &rhs)?;
    let dz = sol[..n].to_vec();
    let dnu = sol[n..].to_vec();

    // Recover Δs = −r_ineq − G Δz.
    let g_dz = if m > 0 {
        mat_vec(&prob.g_mat, m, n, &dz)?
    } else {
        vec![0.0_f64; m]
    };
    let mut ds = vec![0.0_f64; m];
    for k in 0..m {
        ds[k] = -r_ineq[k] - g_dz[k];
    }

    // Recover Δλ = −(r_cent + λ Δs) ⊘ s.
    let mut dlam = vec![0.0_f64; m];
    for k in 0..m {
        let sk = s[k].max(1.0e-14);
        dlam[k] = -(r_cent[k] + lam[k] * ds[k]) / sk;
    }

    Ok((dz, ds, dlam, dnu))
}

/// Solve a pure equality-constrained (or unconstrained) QP in one linear solve.
///
/// For `min ½zᵀQz+qᵀz s.t. Az=b` the optimum satisfies the KKT system
///
/// ```text
///   [ Q  Aᵀ ] [ z ]   [ −q ]
///   [ A  0  ] [ ν ] = [  b ] .
/// ```
fn solve_equality_qp(prob: &QpProblem, q_sym: &[f64], cfg: &OptNetConfig) -> CvxResult<QpSolution> {
    let n = prob.n;
    let p = prob.p;
    let dim = n + p;
    let mut kkt = vec![0.0_f64; dim * dim];
    for i in 0..n {
        for j in 0..n {
            kkt[i * dim + j] = q_sym[i * n + j];
        }
    }
    for l in 0..p {
        for i in 0..n {
            let ali = prob.a_mat[l * n + i];
            kkt[i * dim + (n + l)] = ali;
            kkt[(n + l) * dim + i] = ali;
        }
    }
    // Tiny shift on the (2,2) zero block in case A is rank-deficient.
    for l in 0..p {
        let idx = n + l;
        kkt[idx * dim + idx] -= 1.0e-12;
    }

    let mut rhs = vec![0.0_f64; dim];
    for (ri, qi) in rhs.iter_mut().zip(prob.q_vec.iter()).take(n) {
        *ri = -qi;
    }
    rhs[n..(n + p)].copy_from_slice(&prob.b_vec[..p]);

    let sol = solve_dense(&kkt, dim, &rhs)?;
    let z = sol[..n].to_vec();
    let nu = sol[n..].to_vec();

    // Verify residual for an honest `converged` flag.
    let qz = mat_vec(q_sym, n, n, &z)?;
    let at_nu = if p > 0 {
        mat_t_vec(&prob.a_mat, p, n, &nu)?
    } else {
        vec![0.0_f64; n]
    };
    let mut r_dual = vec![0.0_f64; n];
    for i in 0..n {
        r_dual[i] = qz[i] + prob.q_vec[i] + at_nu[i];
    }
    let mut r_eq = vec![0.0_f64; p];
    if p > 0 {
        let az = mat_vec(&prob.a_mat, p, n, &z)?;
        for l in 0..p {
            r_eq[l] = az[l] - prob.b_vec[l];
        }
    }
    let converged = norm2(&r_dual) < cfg.tol.max(1.0e-7) && norm2(&r_eq) < cfg.tol.max(1.0e-7);

    Ok(QpSolution {
        z,
        lam: Vec::new(),
        nu,
        iter: 1,
        mu: 0.0,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central finite-difference of `loss(re-solved z*(θ))` w.r.t. one scalar
    /// entry of a problem datum. `mutate` applies `θ_k ± δ` to a fresh problem.
    fn fd_grad<F, M>(
        base: &QpProblem,
        cfg: &OptNetConfig,
        len: usize,
        loss: F,
        mutate: M,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
        M: Fn(&mut QpProblem, usize, f64),
    {
        let delta = 1.0e-6;
        let mut out = vec![0.0_f64; len];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut prob_p = base.clone();
            mutate(&mut prob_p, k, delta);
            let layer_p = OptNetLayer::solve(prob_p, cfg.clone()).expect("forward +");
            let lp = loss(&layer_p.solution.z);

            let mut prob_m = base.clone();
            mutate(&mut prob_m, k, -delta);
            let layer_m = OptNetLayer::solve(prob_m, cfg.clone()).expect("forward -");
            let lm = loss(&layer_m.solution.z);

            *slot = (lp - lm) / (2.0 * delta);
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() < tol,
                "{what}[{i}]: {ai} vs {bi} (|Δ|={})",
                (ai - bi).abs()
            );
        }
    }

    // ── Test 1: equality-constrained QP, analytic dz*/dθ ────────────────────
    #[test]
    fn equality_qp_matches_analytic_gradient() {
        // min ½‖z‖²  s.t.  z1 + z2 = c.   Closed form: z* = (c/2, c/2),
        // so ∂z*/∂b = (1/2, 1/2).  Loss ℓ = ½‖z*‖² ⇒ ∂ℓ/∂z* = z*.
        let c = 1.0;
        let prob = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![c],
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let layer = OptNetLayer::solve_default(prob).expect("forward");
        assert!((layer.z()[0] - 0.5).abs() < 1e-8);
        assert!((layer.z()[1] - 0.5).abs() < 1e-8);

        let grad_z = layer.z().to_vec(); // ∂ℓ/∂z* = z*
        let grads = layer.backward(&grad_z).expect("backward");

        // dℓ/db: chain rule (z*)ᵀ ∂z*/∂b = (0.5,0.5)·(0.5,0.5) = 0.5.
        assert!(
            (grads.d_b_vec[0] - 0.5).abs() < 1e-6,
            "d_b={} expected 0.5",
            grads.d_b_vec[0]
        );
    }

    // ── Test 2: equality QP, finite-difference on b and q ───────────────────
    #[test]
    fn equality_qp_fd_b_and_q() {
        let prob = QpProblem::new(
            vec![2.0, 0.5, 0.5, 3.0],
            vec![0.3, -0.7],
            vec![1.0, 2.0],
            vec![1.5],
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let cfg = OptNetConfig::default();
        let layer = OptNetLayer::solve(prob.clone(), cfg.clone()).expect("forward");
        let grad_z = layer.z().to_vec(); // ℓ = ½‖z‖²
        let grads = layer.backward(&grad_z).expect("backward");

        let loss = |z: &[f64]| 0.5 * z.iter().map(|v| v * v).sum::<f64>();
        let fd_b = fd_grad(&prob, &cfg, 1, loss, |pr, k, d| pr.b_vec[k] += d);
        assert_close(&grads.d_b_vec, &fd_b, 1e-4, "d_b vs FD");

        let fd_q = fd_grad(&prob, &cfg, 2, loss, |pr, k, d| pr.q_vec[k] += d);
        assert_close(&grads.d_q_vec, &fd_q, 1e-4, "d_q vs FD");
    }

    // ── Test 3: box projection (inequality-only), FD on q ───────────────────
    #[test]
    fn box_projection_fd_q() {
        // Projection of q0 onto box [-1,1]²: min ½‖z‖² + qᵀz  with G z ≤ h,
        // G = [I; −I], h = [1,1,1,1].  z* = clamp(−q, −1, 1).
        let q0 = vec![0.4, -0.3];
        let g = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            -1.0, 0.0, //
            0.0, -1.0,
        ];
        let h = vec![1.0, 1.0, 1.0, 1.0];
        let prob = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            q0.clone(),
            Vec::new(),
            Vec::new(),
            g,
            h,
        )
        .expect("problem");
        let cfg = OptNetConfig::default();
        let layer = OptNetLayer::solve(prob.clone(), cfg.clone()).expect("forward");
        // Interior optimum z* = −q0 (both inequalities inactive).
        assert!(
            (layer.z()[0] - (-q0[0])).abs() < 1e-6,
            "z0={}",
            layer.z()[0]
        );
        assert!(
            (layer.z()[1] - (-q0[1])).abs() < 1e-6,
            "z1={}",
            layer.z()[1]
        );

        let grad_z = layer.z().to_vec();
        let grads = layer.backward(&grad_z).expect("backward");
        let loss = |z: &[f64]| 0.5 * z.iter().map(|v| v * v).sum::<f64>();
        let fd_q = fd_grad(&prob, &cfg, 2, loss, |pr, k, d| pr.q_vec[k] += d);
        assert_close(&grads.d_q_vec, &fd_q, 1e-4, "d_q vs FD (box)");
    }

    // ── Test 4: active inequality contributes, inactive λ≈0 ─────────────────
    #[test]
    fn active_inequality_gradient() {
        // min ½‖z − a‖²  with a = (2, 0), box [-1,1]².  Equivalent
        // Q=I, q=−a.  z* = (1, 0): constraint z1 ≤ 1 ACTIVE, others inactive.
        let a = [2.0_f64, 0.0];
        let q0 = vec![-a[0], -a[1]];
        let g = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let h = vec![1.0, 1.0, 1.0, 1.0];
        let prob = QpProblem::new(vec![1.0, 0.0, 0.0, 1.0], q0, Vec::new(), Vec::new(), g, h)
            .expect("problem");
        let cfg = OptNetConfig::default();
        let layer = OptNetLayer::solve(prob.clone(), cfg.clone()).expect("forward");
        assert!((layer.z()[0] - 1.0).abs() < 1e-5, "z0={}", layer.z()[0]);
        assert!(layer.z()[1].abs() < 1e-5, "z1={}", layer.z()[1]);

        // λ for the first (active) constraint > 0; others ≈ 0.
        let lam = &layer.solution().lam;
        assert!(lam[0] > 0.5, "active λ0={} should be ~1", lam[0]);
        assert!(lam[1].abs() < 1e-4 && lam[2].abs() < 1e-4 && lam[3].abs() < 1e-4);

        // Gradient of ℓ = ½‖z‖² w.r.t. h via FD (active row should matter).
        let grad_z = layer.z().to_vec();
        let grads = layer.backward(&grad_z).expect("backward");
        let loss = |z: &[f64]| 0.5 * z.iter().map(|v| v * v).sum::<f64>();
        let fd_h = fd_grad(&prob, &cfg, 4, loss, |pr, k, d| pr.h_vec[k] += d);
        assert_close(&grads.d_h_vec, &fd_h, 2e-4, "d_h vs FD (active)");
        // Inactive constraints (2,3,4) contribute ~0 gradient.
        assert!(grads.d_h_vec[1].abs() < 1e-5);
        assert!(grads.d_h_vec[2].abs() < 1e-5);
        assert!(grads.d_h_vec[3].abs() < 1e-5);
    }

    // ── Test 5: mixed equality + inequality, FD on all six params ───────────
    #[test]
    fn mixed_qp_fd_all_params() {
        // min ½zᵀQz+qᵀz  s.t.  z0+z1+z2 = 1 (eq),  z ≥ 0 i.e. −z ≤ 0 (ineq).
        let n = 3;
        let q_mat = vec![
            2.0, 0.2, 0.0, //
            0.2, 2.0, 0.1, //
            0.0, 0.1, 2.0,
        ];
        let q_vec = vec![-0.5, 0.1, 0.3];
        let a_mat = vec![1.0, 1.0, 1.0];
        let b_vec = vec![1.0];
        // −z ≤ 0  ⇒ G = −I, h = 0.
        let g_mat = vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        let h_vec = vec![0.0, 0.0, 0.0];
        let prob = QpProblem::new(q_mat, q_vec, a_mat, b_vec, g_mat, h_vec).expect("problem");
        let cfg = OptNetConfig {
            tol: 1.0e-11,
            ..Default::default()
        };
        let layer = OptNetLayer::solve(prob.clone(), cfg.clone()).expect("forward");
        // Feasibility checks.
        let zsum: f64 = layer.z().iter().sum();
        assert!((zsum - 1.0).abs() < 1e-6, "sum z = {zsum}");
        for &zi in layer.z() {
            assert!(zi > -1e-6, "z negative: {zi}");
        }

        // Random-ish linear loss ℓ = wᵀ z* to exercise full outer products.
        let w = [0.7_f64, -0.4, 1.1];
        let loss = move |z: &[f64]| z.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();
        let grad_z = w.to_vec(); // ∂ℓ/∂z* = w
        let grads = layer.backward(&grad_z).expect("backward");

        // q (length 3).
        let fd_q = fd_grad(&prob, &cfg, n, loss, |pr, k, d| pr.q_vec[k] += d);
        assert_close(&grads.d_q_vec, &fd_q, 2e-4, "d_q vs FD (mixed)");
        // b (length 1).
        let fd_b = fd_grad(&prob, &cfg, 1, loss, |pr, k, d| pr.b_vec[k] += d);
        assert_close(&grads.d_b_vec, &fd_b, 2e-4, "d_b vs FD (mixed)");
        // h (length 3).
        let fd_h = fd_grad(&prob, &cfg, 3, loss, |pr, k, d| pr.h_vec[k] += d);
        assert_close(&grads.d_h_vec, &fd_h, 2e-4, "d_h vs FD (mixed)");
    }

    // ── Test 6: FD on matrix params Q, A, G (mixed problem) ─────────────────
    #[test]
    fn mixed_qp_fd_matrix_params() {
        let n = 2;
        let q_mat = vec![2.0, 0.3, 0.3, 2.0];
        let q_vec = vec![-1.0, 0.5];
        let a_mat = vec![1.0, 1.0];
        let b_vec = vec![0.8];
        let g_mat = vec![-1.0, 0.0, 0.0, -1.0]; // z ≥ 0
        let h_vec = vec![0.0, 0.0];
        let prob = QpProblem::new(q_mat, q_vec, a_mat, b_vec, g_mat, h_vec).expect("problem");
        let cfg = OptNetConfig {
            tol: 1.0e-11,
            ..Default::default()
        };
        let layer = OptNetLayer::solve(prob.clone(), cfg.clone()).expect("forward");
        let w = [0.6_f64, 1.3];
        let loss = move |z: &[f64]| z.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();
        let grad_z = w.to_vec();
        let grads = layer.backward(&grad_z).expect("backward");

        // ∇_Q : finite-difference must be done on the SYMMETRIC perturbation to
        // match the analytic symmetric gradient. Perturb (i,j) and (j,i) together.
        let mut fd_q_mat = vec![0.0_f64; n * n];
        let delta = 1.0e-6;
        for i in 0..n {
            for j in 0..n {
                let mut pp = prob.clone();
                pp.q_mat[i * n + j] += delta;
                if i != j {
                    pp.q_mat[j * n + i] += delta;
                }
                let lp = loss(OptNetLayer::solve(pp, cfg.clone()).expect("fp").z());
                let mut pm = prob.clone();
                pm.q_mat[i * n + j] -= delta;
                if i != j {
                    pm.q_mat[j * n + i] -= delta;
                }
                let lm = loss(OptNetLayer::solve(pm, cfg.clone()).expect("fm").z());
                fd_q_mat[i * n + j] = (lp - lm) / (2.0 * delta);
            }
        }
        // Our analytic ∇_Q is symmetric; symmetrise the FD too for comparison of
        // the off-diagonal (the FD above already perturbs symmetrically, giving
        // the total derivative w.r.t. the symmetric entry, i.e. dQ_ij + dQ_ji).
        let mut analytic_sym = grads.d_q_mat.clone();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    analytic_sym[i * n + j] = grads.d_q_mat[i * n + j] + grads.d_q_mat[j * n + i];
                }
            }
        }
        assert_close(&analytic_sym, &fd_q_mat, 2e-4, "d_Q vs FD");

        // ∇_A (1×2).
        let fd_a = fd_grad(&prob, &cfg, 2, loss, |pr, k, d| pr.a_mat[k] += d);
        assert_close(&grads.d_a_mat, &fd_a, 2e-4, "d_A vs FD");

        // ∇_G (2×2).
        let fd_g = fd_grad(&prob, &cfg, 4, loss, |pr, k, d| pr.g_mat[k] += d);
        assert_close(&grads.d_g_mat, &fd_g, 3e-4, "d_G vs FD");
    }

    // ── Test 7: ∇_Q symmetry property ───────────────────────────────────────
    #[test]
    fn grad_q_is_symmetric() {
        let prob = QpProblem::new(
            vec![3.0, 1.0, 1.0, 2.0],
            vec![0.2, -0.4],
            vec![1.0, 1.0],
            vec![1.0],
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let layer = OptNetLayer::solve_default(prob).expect("forward");
        let grad_z = vec![0.9_f64, -0.2];
        let grads = layer.backward(&grad_z).expect("backward");
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (grads.d_q_mat[i * n + j] - grads.d_q_mat[j * n + i]).abs() < 1e-12,
                    "∇_Q not symmetric at ({i},{j})"
                );
            }
        }
    }

    // ── Test 8: inactive constraints yield ~zero gradient ───────────────────
    #[test]
    fn inactive_constraints_zero_gradient() {
        // Unconstrained-interior optimum well inside a wide box ⇒ all λ ≈ 0,
        // so ∇_G and ∇_h are ≈ 0.
        let prob = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.1, -0.2],
            Vec::new(),
            Vec::new(),
            vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            vec![10.0, 10.0, 10.0, 10.0], // box [-10,10]², optimum at (−0.1,0.2)
        )
        .expect("problem");
        let layer = OptNetLayer::solve_default(prob).expect("forward");
        for &l in &layer.solution().lam {
            assert!(l.abs() < 1e-5, "expected inactive λ≈0, got {l}");
        }
        let grad_z = vec![1.0_f64, 1.0];
        let grads = layer.backward(&grad_z).expect("backward");
        for &g in &grads.d_g_mat {
            assert!(g.abs() < 1e-6, "expected ∇_G≈0, got {g}");
        }
        for &g in &grads.d_h_vec {
            assert!(g.abs() < 1e-6, "expected ∇_h≈0, got {g}");
        }
    }

    // ── Test 9: dimension-mismatch errors in QpProblem::new ─────────────────
    #[test]
    fn problem_dimension_errors() {
        // Q wrong size.
        assert!(matches!(
            QpProblem::new(
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0],
                vec![],
                vec![],
                vec![],
                vec![]
            ),
            Err(CvxError::ShapeMismatch { .. })
        ));
        // A wrong size (b says p=1, but A has 1 element for n=2).
        assert!(matches!(
            QpProblem::new(
                vec![1.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0],
                vec![1.0],
                vec![1.0],
                vec![],
                vec![]
            ),
            Err(CvxError::ShapeMismatch { .. })
        ));
        // G wrong size.
        assert!(matches!(
            QpProblem::new(
                vec![1.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0],
                vec![],
                vec![],
                vec![1.0, 0.0, 0.0],
                vec![1.0]
            ),
            Err(CvxError::ShapeMismatch { .. })
        ));
        // n == 0.
        assert!(matches!(
            QpProblem::new(vec![], vec![], vec![], vec![], vec![], vec![]),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    // ── Test 10: backward gradient length mismatch ──────────────────────────
    #[test]
    fn backward_grad_length_error() {
        let prob = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0],
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let layer = OptNetLayer::solve_default(prob).expect("forward");
        let err = layer.backward(&[1.0]); // wrong length (n=2)
        assert!(matches!(err, Err(CvxError::DimensionMismatch { .. })));
    }

    // ── Test 11: invalid config rejected ────────────────────────────────────
    #[test]
    fn invalid_config_rejected() {
        let prob = QpProblem::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let bad_tol = OptNetConfig {
            tol: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            OptNetLayer::solve(prob.clone(), bad_tol),
            Err(CvxError::InvalidParameter(_))
        ));
        let bad_eta = OptNetConfig {
            frac_to_boundary: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            OptNetLayer::solve(prob, bad_eta),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    // ── Test 12: non-PSD Q is regularised, KKT still solvable ───────────────
    #[test]
    fn nonconvex_q_regularised_backward_ok() {
        // Indefinite Q (eigenvalues 3, −1). The forward solve floors Q and the
        // equality constraint pins z; the backward KKT is regularised so the
        // factorisation still succeeds and returns finite gradients.
        let prob = QpProblem::new(
            vec![1.0, 2.0, 2.0, 1.0], // indefinite
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0],
            Vec::new(),
            Vec::new(),
        )
        .expect("problem");
        let layer = OptNetLayer::solve_default(prob).expect("forward");
        // Equality forces z0+z1=1; symmetry ⇒ z=(0.5,0.5).
        assert!((layer.z()[0] - 0.5).abs() < 1e-6 && (layer.z()[1] - 0.5).abs() < 1e-6);
        let grad_z = vec![0.3_f64, -0.1];
        let grads = layer.backward(&grad_z).expect("backward");
        for &g in &grads.d_q_vec {
            assert!(g.is_finite(), "non-finite gradient");
        }
        for &g in &grads.d_q_mat {
            assert!(g.is_finite(), "non-finite Q gradient");
        }
    }
}
