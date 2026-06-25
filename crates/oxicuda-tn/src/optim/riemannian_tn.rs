//! Riemannian optimisation on the fixed-rank matrix manifold `M_r`.
//!
//! `M_r = { X ∈ ℝ^{m×n} : rank(X) = r }` is a smooth embedded submanifold of
//! `ℝ^{m×n}`. It is the local building block of every fixed-rank tensor-network format:
//! a thin SVD `X = U Σ Vᵀ` is exactly a (left, right) gauge pair, and any unfolding of a
//! TT-core / MPS tensor is a fixed-rank matrix. The geometry implemented here therefore
//! transfers directly to the tensor-network setting (Hauru 2021; Vandereycken 2013).
//!
//! ## Geometry
//!
//! At a point `X = U Σ Vᵀ ∈ M_r` (thin SVD, `U ∈ ℝ^{m×r}`, `V ∈ ℝ^{n×r}` column
//! orthonormal, `Σ` diagonal positive), the tangent space is
//!
//! ```text
//! T_X M_r = { U Mᵀ_v + U_p Vᵀ + U Vᵀ_p },     (equivalently the image of P_X below)
//! ```
//!
//! and the **orthogonal projector** of an ambient direction `Z ∈ ℝ^{m×n}` onto `T_X M_r`
//! under the Frobenius inner product is
//!
//! ```text
//! P_X(Z) = U Uᵀ Z + Z V Vᵀ − U Uᵀ Z V Vᵀ.
//! ```
//!
//! The **metric-projection retraction** maps a tangent vector `ξ` back to the manifold by
//! truncating `X + ξ` to its best rank-`r` approximation (Eckart–Young), keeping the
//! iterate on `M_r`:
//!
//! ```text
//! R_X(ξ) = SVD_r(X + ξ).
//! ```
//!
//! The **Riemannian gradient** of a smooth cost `f` is the tangent projection of the
//! Euclidean gradient: `grad f(X) = P_X(∇f(X))`.
//!
//! ## Solvers
//!
//! - Riemannian **gradient descent** with Armijo backtracking line search.
//! - Riemannian **conjugate gradient** (Fletcher–Reeves) with **vector transport** of the
//!   previous search direction, realised by re-projecting it onto the new tangent space.
//!
//! ## Canonical problem
//!
//! Low-rank matrix approximation `min_{X∈M_r} ½‖X − A‖²` (whose minimiser is the truncated
//! rank-`r` SVD of `A`, by Eckart–Young) and masked low-rank completion
//! `min_{X∈M_r} ½‖P_Ω(X − A)‖²`.
//!
//! ## References
//!
//! - Vandereycken (2013). *Low-rank matrix completion by Riemannian optimization.*
//!   SIAM J. Optim. 23(2), 1214–1236.
//! - Absil, Mahony & Sepulchre (2008). *Optimization Algorithms on Matrix Manifolds.*
//! - Hauru, Van Damme & Haegeman (2021). *Riemannian optimization of isometric tensor
//!   networks.* SciPost Phys. 10, 040.

use crate::handle::LcgRng;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ─── configuration ──────────────────────────────────────────────────────────────

/// Which Riemannian first-order method to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiemannianTnMethod {
    /// Steepest descent: search direction is the negated Riemannian gradient.
    GradientDescent,
    /// Conjugate gradient (Fletcher–Reeves) with vector transport of the prior direction.
    ConjugateGradient,
}

/// Configuration for [`RiemannianTn::optimize`].
#[derive(Debug, Clone)]
pub struct RiemannianTnConfig {
    /// Target rank `r` of the manifold `M_r`.
    pub rank: usize,
    /// Maximum number of outer Riemannian iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the Riemannian gradient norm `‖grad f(X)‖_F`.
    pub tol: f64,
    /// First-order method to use.
    pub method: RiemannianTnMethod,
    /// Initial trial step length for the Armijo backtracking line search.
    pub initial_step: f64,
    /// Armijo sufficient-decrease parameter `c ∈ (0, 1)`.
    pub armijo_c: f64,
    /// Backtracking contraction factor `β ∈ (0, 1)` applied to the step on rejection.
    pub backtrack_beta: f64,
    /// Maximum number of backtracking trials per outer iteration.
    pub max_line_search: usize,
    /// RNG seed used when an explicit warm-start point is not supplied.
    pub seed: u64,
}

impl Default for RiemannianTnConfig {
    fn default() -> Self {
        Self {
            rank: 1,
            max_iter: 200,
            tol: 1e-9,
            method: RiemannianTnMethod::ConjugateGradient,
            initial_step: 1.0,
            armijo_c: 1e-4,
            backtrack_beta: 0.5,
            max_line_search: 40,
            seed: 0x5151_2026,
        }
    }
}

impl RiemannianTnConfig {
    /// Validate the configuration against the problem dimensions `m × n`.
    fn validate(&self, m: usize, n: usize) -> TnResult<()> {
        if m == 0 || n == 0 {
            return Err(TnError::EmptyInput);
        }
        let min_dim = m.min(n);
        if self.rank == 0 {
            return Err(TnError::InvalidRank(0));
        }
        if self.rank > min_dim {
            return Err(TnError::RankExceedsLimit {
                rank: self.rank,
                max: min_dim,
            });
        }
        if self.max_iter == 0 {
            return Err(TnError::InvalidParameter {
                name: "max_iter".into(),
                reason: "must be ≥ 1".into(),
            });
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(TnError::InvalidParameter {
                name: "tol".into(),
                reason: "must be a finite non-negative number".into(),
            });
        }
        if !(self.initial_step.is_finite() && self.initial_step > 0.0) {
            return Err(TnError::InvalidParameter {
                name: "initial_step".into(),
                reason: "must be a finite positive number".into(),
            });
        }
        if !(self.armijo_c > 0.0 && self.armijo_c < 1.0) {
            return Err(TnError::InvalidParameter {
                name: "armijo_c".into(),
                reason: "must lie in (0, 1)".into(),
            });
        }
        if !(self.backtrack_beta > 0.0 && self.backtrack_beta < 1.0) {
            return Err(TnError::InvalidParameter {
                name: "backtrack_beta".into(),
                reason: "must lie in (0, 1)".into(),
            });
        }
        if self.max_line_search == 0 {
            return Err(TnError::InvalidParameter {
                name: "max_line_search".into(),
                reason: "must be ≥ 1".into(),
            });
        }
        Ok(())
    }
}

// ─── manifold point ─────────────────────────────────────────────────────────────

/// A point `X = U Σ Vᵀ` on the fixed-rank manifold `M_r`, stored in thin-SVD form.
///
/// - `u`: `m × r` row-major, column-orthonormal (`Uᵀ U = I_r`).
/// - `s`: length `r`, the (positive) singular values, descending.
/// - `vt`: `r × n` row-major, i.e. `Vᵀ` (row-orthonormal, `V Vᵀ` rows orthonormal).
#[derive(Debug, Clone)]
pub struct TnPoint {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub r: usize,
}

impl TnPoint {
    /// Materialise the dense `m × n` row-major matrix `X = U Σ Vᵀ`.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let mut acc = 0.0f64;
                for c in 0..self.r {
                    acc += self.u[i * self.r + c] * self.s[c] * self.vt[c * self.n + j];
                }
                out[i * self.n + j] = acc;
            }
        }
        out
    }

    /// The numerical rank: count of singular values exceeding `tol`.
    #[must_use]
    pub fn numerical_rank(&self, tol: f64) -> usize {
        self.s.iter().filter(|&&x| x > tol).count()
    }
}

// ─── optimisation result ────────────────────────────────────────────────────────

/// Outcome of a Riemannian optimisation run.
#[derive(Debug, Clone)]
pub struct TnResultData {
    /// The final manifold point `X* = U Σ Vᵀ`.
    pub point: TnPoint,
    /// Final cost value `f(X*)`.
    pub objective: f64,
    /// Final Riemannian-gradient Frobenius norm `‖grad f(X*)‖_F`.
    pub grad_norm: f64,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Whether the gradient-norm tolerance was met.
    pub converged: bool,
    /// Objective value at the end of each outer iteration (monotone non-increasing).
    pub objective_history: Vec<f64>,
}

// ─── the manifold + optimiser ───────────────────────────────────────────────────

/// The fixed-rank matrix manifold `M_r` of `m × n` matrices, with its Riemannian
/// geometry and first-order solvers.
#[derive(Debug, Clone)]
pub struct FixedRankManifold {
    m: usize,
    n: usize,
    r: usize,
}

/// Convenience alias: a Riemannian optimiser *is* a manifold endowed with solvers.
pub type RiemannianTn = FixedRankManifold;

impl FixedRankManifold {
    /// Create the manifold `M_r` of `m × n` rank-`r` matrices.
    ///
    /// # Errors
    /// - [`TnError::EmptyInput`] if `m == 0` or `n == 0`.
    /// - [`TnError::InvalidRank`] if `r == 0`.
    /// - [`TnError::RankExceedsLimit`] if `r > min(m, n)`.
    pub fn new(m: usize, n: usize, r: usize) -> TnResult<Self> {
        if m == 0 || n == 0 {
            return Err(TnError::EmptyInput);
        }
        let min_dim = m.min(n);
        if r == 0 {
            return Err(TnError::InvalidRank(0));
        }
        if r > min_dim {
            return Err(TnError::RankExceedsLimit {
                rank: r,
                max: min_dim,
            });
        }
        Ok(Self { m, n, r })
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.m
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.n
    }

    /// Manifold rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.r
    }

    // ── point construction ──────────────────────────────────────────────────────

    /// Build a manifold point from a dense `m × n` row-major matrix by truncating to
    /// rank `r` (metric projection onto `M_r`).
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `matrix.len() != m * n`.
    /// - any SVD failure is propagated.
    pub fn point_from_dense(&self, matrix: &[f64]) -> TnResult<TnPoint> {
        if matrix.len() != self.m * self.n {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.m, self.n],
                got: vec![matrix.len()],
            });
        }
        self.truncate_rank_r(matrix)
    }

    /// A random starting point on `M_r`, drawn as `truncate_r(G)` for a standard-normal
    /// matrix `G`. Deterministic for a fixed `seed`.
    ///
    /// # Errors
    /// Propagates any SVD failure on the random matrix.
    pub fn random_point(&self, seed: u64) -> TnResult<TnPoint> {
        let mut rng = LcgRng::new(seed);
        let g: Vec<f64> = (0..self.m * self.n).map(|_| rng.next_normal()).collect();
        self.truncate_rank_r(&g)
    }

    // ── tangent-space projection ────────────────────────────────────────────────

    /// Orthogonally project an ambient direction `z` (`m × n` row-major) onto the
    /// tangent space `T_X M_r`:
    ///
    /// `P_X(Z) = U Uᵀ Z + Z V Vᵀ − U Uᵀ Z V Vᵀ`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `z.len() != m * n`.
    pub fn project_tangent(&self, x: &TnPoint, z: &[f64]) -> TnResult<Vec<f64>> {
        self.check_point(x)?;
        if z.len() != self.m * self.n {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.m, self.n],
                got: vec![z.len()],
            });
        }
        let (m, n, r) = (self.m, self.n, self.r);

        // a = Uᵀ Z   (r × n)
        let a = matmul_tn(&x.u, m, r, z, n); // (r×m)·(m×n)
        // b = Z V    (m × r) = Z · Vᵀᵀ ; vt is (r×n) = Vᵀ, so V = vtᵀ, Z·V via vt rows.
        let b = matmul_nt(z, m, n, &x.vt, r); // (m×n)·(n×r)
        // c = Uᵀ Z V (r × r) = a · V
        let c = matmul_nt(&a, r, n, &x.vt, r); // (r×n)·(n×r)

        // term1 = U a         (m × n)
        // term2 = b Vᵀ        (m × n)
        // term3 = U c Vᵀ      (m × n)
        let mut out = vec![0.0f64; m * n];
        // U·a
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for c_idx in 0..r {
                    acc += x.u[i * r + c_idx] * a[c_idx * n + j];
                }
                out[i * n + j] += acc;
            }
        }
        // b·Vᵀ  (b is m×r, Vᵀ is r×n = vt)
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for c_idx in 0..r {
                    acc += b[i * r + c_idx] * x.vt[c_idx * n + j];
                }
                out[i * n + j] += acc;
            }
        }
        // − U·c·Vᵀ : first uc = U·c (m×r), then uc·Vᵀ.
        let uc = matmul_nn(&x.u, m, r, &c, r); // (m×r)·(r×r)
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for c_idx in 0..r {
                    acc += uc[i * r + c_idx] * x.vt[c_idx * n + j];
                }
                out[i * n + j] -= acc;
            }
        }
        Ok(out)
    }

    // ── retraction ──────────────────────────────────────────────────────────────

    /// Metric-projection retraction `R_X(ξ) = SVD_r(X + ξ)`: add the tangent vector `xi`
    /// (`m × n` row-major) to the dense point and truncate back to rank `r`.
    ///
    /// `R_X(0) == X` (up to SVD reconstruction tolerance) and `rank(R_X(ξ)) == r`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `xi.len() != m * n`.
    /// - any SVD failure is propagated.
    pub fn retract(&self, x: &TnPoint, xi: &[f64]) -> TnResult<TnPoint> {
        self.check_point(x)?;
        if xi.len() != self.m * self.n {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.m, self.n],
                got: vec![xi.len()],
            });
        }
        let dense = x.to_dense();
        let mut sum = vec![0.0f64; self.m * self.n];
        for idx in 0..sum.len() {
            sum[idx] = dense[idx] + xi[idx];
        }
        self.truncate_rank_r(&sum)
    }

    // ── Riemannian gradient ─────────────────────────────────────────────────────

    /// Riemannian gradient `grad f(X) = P_X(∇f(X))`: project the supplied Euclidean
    /// gradient `egrad` (`m × n` row-major) onto the tangent space at `x`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `egrad.len() != m * n`.
    pub fn rgrad(&self, x: &TnPoint, egrad: &[f64]) -> TnResult<Vec<f64>> {
        self.project_tangent(x, egrad)
    }

    // ── driver ──────────────────────────────────────────────────────────────────

    /// Run Riemannian optimisation of `cost` with Euclidean gradient `egrad`.
    ///
    /// `cost(&dense_x) -> f(X)` and `egrad(&dense_x) -> ∇f(X)` are closures over the
    /// dense `m × n` row-major materialisation of the current iterate. `x0` is the warm
    /// start; pass `None` to draw a deterministic random start on `M_r`.
    ///
    /// The objective is non-increasing across outer iterations thanks to the Armijo
    /// backtracking line search composed with the retraction.
    ///
    /// # Errors
    /// - configuration / dimension errors are validated up front;
    /// - SVD failures inside the retraction are propagated;
    /// - the gradient closure returning a wrong-length vector yields
    ///   [`TnError::ShapeMismatch`].
    pub fn optimize<C, G>(
        &self,
        cost: C,
        egrad: G,
        x0: Option<TnPoint>,
        config: &RiemannianTnConfig,
    ) -> TnResult<TnResultData>
    where
        C: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        config.validate(self.m, self.n)?;
        if config.rank != self.r {
            return Err(TnError::InvalidParameter {
                name: "rank".into(),
                reason: format!(
                    "config rank {} does not match manifold rank {}",
                    config.rank, self.r
                ),
            });
        }

        let mut x = match x0 {
            Some(p) => {
                self.check_point(&p)?;
                p
            }
            None => self.random_point(config.seed)?,
        };

        let mut dense = x.to_dense();
        let mut f_x = cost(&dense);
        if !f_x.is_finite() {
            return Err(TnError::NumericalInstability(
                "initial cost is not finite".into(),
            ));
        }

        let mut objective_history = Vec::with_capacity(config.max_iter + 1);
        objective_history.push(f_x);

        // Previous Riemannian gradient and previous (transported) search direction, used
        // by the conjugate-gradient branch.
        let mut prev_rgrad: Option<Vec<f64>> = None;
        let mut prev_dir: Option<Vec<f64>> = None;

        let mut grad_norm = f64::INFINITY;
        let mut converged = false;
        let mut iterations = 0usize;

        for _iter in 0..config.max_iter {
            let eg = egrad(&dense);
            if eg.len() != self.m * self.n {
                return Err(TnError::ShapeMismatch {
                    expected: vec![self.m, self.n],
                    got: vec![eg.len()],
                });
            }
            let rg = self.project_tangent(&x, &eg)?;
            grad_norm = frob_norm(&rg);

            if grad_norm <= config.tol {
                converged = true;
                break;
            }

            // Search direction.
            let direction = match config.method {
                RiemannianTnMethod::GradientDescent => negate(&rg),
                RiemannianTnMethod::ConjugateGradient => {
                    self.cg_direction(&x, &rg, prev_rgrad.as_deref(), prev_dir.as_deref())?
                }
            };

            // Directional derivative ⟨grad f, η⟩ ; must be a descent direction.
            let dir_deriv = frob_inner(&rg, &direction);
            let descent_dir = if dir_deriv >= -1e-30 {
                // Fallback to steepest descent if conjugacy lost descent property.
                negate(&rg)
            } else {
                direction
            };
            let slope = frob_inner(&rg, &descent_dir); // < 0

            // Armijo backtracking line search along the retraction.
            let mut step = config.initial_step;
            let mut accepted: Option<(TnPoint, Vec<f64>, f64)> = None;
            for _ls in 0..config.max_line_search {
                let xi = scale(&descent_dir, step);
                let candidate = self.retract(&x, &xi)?;
                let cand_dense = candidate.to_dense();
                let f_cand = cost(&cand_dense);
                if f_cand.is_finite() && f_cand <= f_x + config.armijo_c * step * slope {
                    accepted = Some((candidate, cand_dense, f_cand));
                    break;
                }
                step *= config.backtrack_beta;
            }

            match accepted {
                Some((candidate, cand_dense, f_cand)) => {
                    x = candidate;
                    dense = cand_dense;
                    f_x = f_cand;
                    objective_history.push(f_x);
                    prev_rgrad = Some(rg);
                    prev_dir = Some(descent_dir);
                    iterations += 1;
                }
                None => {
                    // No admissible step: a first-order stationary point for this
                    // step regime. Record the current gradient norm and stop.
                    iterations += 1;
                    break;
                }
            }
        }

        Ok(TnResultData {
            point: x,
            objective: f_x,
            grad_norm,
            iterations,
            converged,
            objective_history,
        })
    }

    // ── internal helpers ─────────────────────────────────────────────────────────

    /// Conjugate-gradient (Fletcher–Reeves) search direction with vector transport.
    ///
    /// The previous search direction lives in `T_{X_old} M_r`; we transport it into
    /// `T_{X_new} M_r` by orthogonal re-projection `P_{X_new}(·)` before combining.
    fn cg_direction(
        &self,
        x: &TnPoint,
        rg: &[f64],
        prev_rgrad: Option<&[f64]>,
        prev_dir: Option<&[f64]>,
    ) -> TnResult<Vec<f64>> {
        match (prev_rgrad, prev_dir) {
            (Some(prev_g), Some(prev_d)) => {
                let denom = frob_inner(prev_g, prev_g);
                let beta = if denom > 1e-30 {
                    (frob_inner(rg, rg) / denom).max(0.0)
                } else {
                    0.0
                };
                // Transport the old direction into the current tangent space.
                let transported = self.project_tangent(x, prev_d)?;
                // η = −grad + β · 𝒯(η_prev)
                let mut dir = negate(rg);
                for idx in 0..dir.len() {
                    dir[idx] += beta * transported[idx];
                }
                Ok(dir)
            }
            _ => Ok(negate(rg)),
        }
    }

    /// Truncate a dense `m × n` matrix to its best rank-`r` approximation, returned in
    /// thin-SVD form as a [`TnPoint`].
    fn truncate_rank_r(&self, matrix: &[f64]) -> TnResult<TnPoint> {
        let svd = svd_jacobi(matrix, self.m, self.n)?;
        let r = self.r;
        // svd.u: m×k, svd.s: k, svd.vt: k×n with k = min(m,n) ≥ r.
        let k = svd.k;
        if k < r {
            return Err(TnError::RankExceedsLimit { rank: r, max: k });
        }
        let mut u = vec![0.0f64; self.m * r];
        for i in 0..self.m {
            for c in 0..r {
                u[i * r + c] = svd.u[i * k + c];
            }
        }
        let s = svd.s[..r].to_vec();
        let mut vt = vec![0.0f64; r * self.n];
        for c in 0..r {
            for j in 0..self.n {
                vt[c * self.n + j] = svd.vt[c * self.n + j];
            }
        }
        Ok(TnPoint {
            u,
            s,
            vt,
            m: self.m,
            n: self.n,
            r,
        })
    }

    /// Validate that a point matches this manifold's dimensions and rank.
    fn check_point(&self, x: &TnPoint) -> TnResult<()> {
        if x.m != self.m || x.n != self.n {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.m, self.n],
                got: vec![x.m, x.n],
            });
        }
        if x.r != self.r {
            return Err(TnError::InvalidRank(x.r));
        }
        if x.u.len() != self.m * self.r || x.s.len() != self.r || x.vt.len() != self.r * self.n {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.m * self.r, self.r, self.r * self.n],
                got: vec![x.u.len(), x.s.len(), x.vt.len()],
            });
        }
        Ok(())
    }
}

// ─── dense linear-algebra helpers (row-major) ────────────────────────────────────

/// `Aᵀ · B` where `a` is `p × q` row-major and `b` is `p × s` row-major; result `q × s`.
fn matmul_tn(a: &[f64], p: usize, q: usize, b: &[f64], s: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; q * s];
    for i in 0..q {
        for j in 0..s {
            let mut acc = 0.0f64;
            for kk in 0..p {
                acc += a[kk * q + i] * b[kk * s + j];
            }
            out[i * s + j] = acc;
        }
    }
    out
}

/// `A · B` where `a` is `p × q` row-major and `b` is `q × s` row-major; result `p × s`.
fn matmul_nn(a: &[f64], p: usize, q: usize, b: &[f64], s: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; p * s];
    for i in 0..p {
        for j in 0..s {
            let mut acc = 0.0f64;
            for kk in 0..q {
                acc += a[i * q + kk] * b[kk * s + j];
            }
            out[i * s + j] = acc;
        }
    }
    out
}

/// `A · Bᵀ` where `a` is `p × q` row-major and `b` is `s × q` row-major; result `p × s`.
fn matmul_nt(a: &[f64], p: usize, q: usize, b: &[f64], s: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; p * s];
    for i in 0..p {
        for j in 0..s {
            let mut acc = 0.0f64;
            for kk in 0..q {
                acc += a[i * q + kk] * b[j * q + kk];
            }
            out[i * s + j] = acc;
        }
    }
    out
}

/// Frobenius inner product `⟨A, B⟩ = Σ a_ij b_ij`.
fn frob_inner(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Frobenius norm `‖A‖_F`.
fn frob_norm(a: &[f64]) -> f64 {
    frob_inner(a, a).sqrt()
}

/// Element-wise negation `−A`.
fn negate(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| -x).collect()
}

/// Element-wise scaling `α · A`.
fn scale(a: &[f64], alpha: f64) -> Vec<f64> {
    a.iter().map(|x| alpha * x).collect()
}

// ─── canonical cost functions ────────────────────────────────────────────────────

/// Objective `½‖X − A‖²` for dense `m × n` row-major matrices `x` and `a`.
///
/// # Errors
/// - [`TnError::DimensionMismatch`] if the two matrices differ in length.
pub fn low_rank_objective(x: &[f64], a: &[f64]) -> TnResult<f64> {
    if x.len() != a.len() {
        return Err(TnError::DimensionMismatch {
            a: x.len(),
            b: a.len(),
        });
    }
    Ok(0.5 * x.iter().zip(a).map(|(p, q)| (p - q) * (p - q)).sum::<f64>())
}

/// Euclidean gradient `∇(½‖X − A‖²) = X − A`.
///
/// # Errors
/// - [`TnError::DimensionMismatch`] if the two matrices differ in length.
pub fn low_rank_egrad(x: &[f64], a: &[f64]) -> TnResult<Vec<f64>> {
    if x.len() != a.len() {
        return Err(TnError::DimensionMismatch {
            a: x.len(),
            b: a.len(),
        });
    }
    Ok(x.iter().zip(a).map(|(p, q)| p - q).collect())
}

/// Masked completion objective `½‖P_Ω(X − A)‖²`: only entries where `mask` is `true`
/// contribute. `mask`, `x`, `a` share length `m·n`.
///
/// # Errors
/// - [`TnError::DimensionMismatch`] if the three slices differ in length.
pub fn low_rank_completion_objective(x: &[f64], a: &[f64], mask: &[bool]) -> TnResult<f64> {
    if x.len() != a.len() || x.len() != mask.len() {
        return Err(TnError::DimensionMismatch {
            a: x.len(),
            b: a.len().min(mask.len()),
        });
    }
    let mut acc = 0.0f64;
    for idx in 0..x.len() {
        if mask[idx] {
            let d = x[idx] - a[idx];
            acc += d * d;
        }
    }
    Ok(0.5 * acc)
}

/// Euclidean gradient of the masked completion objective: `P_Ω(X − A)` (zero off `Ω`).
///
/// # Errors
/// - [`TnError::DimensionMismatch`] if the three slices differ in length.
pub fn low_rank_completion_egrad(x: &[f64], a: &[f64], mask: &[bool]) -> TnResult<Vec<f64>> {
    if x.len() != a.len() || x.len() != mask.len() {
        return Err(TnError::DimensionMismatch {
            a: x.len(),
            b: a.len().min(mask.len()),
        });
    }
    let mut g = vec![0.0f64; x.len()];
    for idx in 0..x.len() {
        if mask[idx] {
            g[idx] = x[idx] - a[idx];
        }
    }
    Ok(g)
}

/// Eckart–Young oracle: the optimal value of `min_{rank≤r} ½‖X − A‖²` is
/// `½ Σ_{i>r} σ_i²`, the discarded tail of the singular spectrum of `A`.
///
/// # Errors
/// - [`TnError::ShapeMismatch`] if `a.len() != m * n`.
/// - any SVD failure is propagated.
pub fn eckart_young_objective(a: &[f64], m: usize, n: usize, r: usize) -> TnResult<f64> {
    let svd = svd_jacobi(a, m, n)?;
    let tail: f64 = svd.s.iter().skip(r).map(|&s| s * s).sum();
    Ok(0.5 * tail)
}

// ─── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Reconstruct the dense rank-r matrix from a thin SVD triple, used for oracle checks.
    fn dense_from_svd(u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize, r: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for c in 0..r {
                    acc += u[i * r + c] * s[c] * vt[c * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn random_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..m * n).map(|_| rng.next_normal()).collect()
    }

    /// Build a planted rank-r matrix `L Rᵀ` with `L: m×r`, `R: n×r` Gaussian.
    fn planted_rank_r(m: usize, n: usize, r: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let l: Vec<f64> = (0..m * r).map(|_| rng.next_normal()).collect();
        let rr: Vec<f64> = (0..n * r).map(|_| rng.next_normal()).collect();
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for c in 0..r {
                    acc += l[i * r + c] * rr[j * r + c];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    // ── geometry ─────────────────────────────────────────────────────────────────

    #[test]
    fn retraction_at_zero_is_identity() {
        let m = 5;
        let n = 4;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let a = random_matrix(m, n, 1);
        let x = mani.point_from_dense(&a).expect("point");
        let zero = vec![0.0f64; m * n];
        let rx = mani.retract(&x, &zero).expect("retract");
        // R_X(0) == X up to SVD reconstruction tolerance.
        assert!(fro_diff(&x.to_dense(), &rx.to_dense()) < 1e-9);
    }

    #[test]
    fn retraction_preserves_rank() {
        let m = 6;
        let n = 5;
        let r = 3;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let x = mani.random_point(7).expect("rand point");
        // A generic tangent step.
        let eg = random_matrix(m, n, 9);
        let xi = mani.rgrad(&x, &eg).expect("rgrad");
        let rx = mani.retract(&x, &xi).expect("retract");
        assert_eq!(rx.r, r);
        // All r singular values strictly positive ⇒ exact rank r.
        assert_eq!(rx.numerical_rank(1e-8), r);
    }

    #[test]
    fn rgrad_lies_in_tangent_space() {
        // P_X is an orthogonal projector ⇒ idempotent: P_X(rgrad) == rgrad.
        let m = 5;
        let n = 6;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let x = mani.random_point(3).expect("point");
        let eg = random_matrix(m, n, 4);
        let rg = mani.rgrad(&x, &eg).expect("rgrad");
        let rg2 = mani.project_tangent(&x, &rg).expect("project");
        assert!(fro_diff(&rg, &rg2) < 1e-10);
    }

    #[test]
    fn tangent_is_orthogonal_complement() {
        // The residual Z − P_X(Z) is orthogonal to the tangent vector P_X(Z).
        let m = 4;
        let n = 4;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let x = mani.random_point(21).expect("point");
        let z = random_matrix(m, n, 22);
        let pz = mani.project_tangent(&x, &z).expect("project");
        let residual: Vec<f64> = z.iter().zip(&pz).map(|(a, b)| a - b).collect();
        let ip = frob_inner(&residual, &pz);
        assert!(ip.abs() < 1e-9, "⟨Z−P,P⟩ = {ip}");
    }

    // ── Eckart–Young convergence (the headline oracle) ───────────────────────────

    #[test]
    fn eckart_young_low_rank_approximation() {
        let m = 8;
        let n = 6;
        let r = 3;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let a = random_matrix(m, n, 100);

        let cfg = RiemannianTnConfig {
            rank: r,
            max_iter: 300,
            tol: 1e-11,
            method: RiemannianTnMethod::ConjugateGradient,
            ..Default::default()
        };
        let cost = |x: &[f64]| low_rank_objective(x, &a).unwrap_or(f64::INFINITY);
        let egrad = |x: &[f64]| low_rank_egrad(x, &a).unwrap_or_default();
        let res = mani.optimize(cost, egrad, None, &cfg).expect("optimize");

        // Optimal objective = ½ Σ_{i>r} σ_i² (the discarded SVD tail).
        let oracle = eckart_young_objective(&a, m, n, r).expect("oracle");
        assert!(
            (res.objective - oracle).abs() < 1e-6,
            "objective {} vs oracle {}",
            res.objective,
            oracle
        );

        // X* matches the truncated rank-r SVD of A to tolerance.
        let svd = svd_jacobi(&a, m, n).expect("svd");
        let mut u_r = vec![0.0f64; m * r];
        for i in 0..m {
            for c in 0..r {
                u_r[i * r + c] = svd.u[i * svd.k + c];
            }
        }
        let s_r = svd.s[..r].to_vec();
        let vt_r = svd.vt[..r * n].to_vec();
        let x_star_ref = dense_from_svd(&u_r, &s_r, &vt_r, m, n, r);
        assert!(
            fro_diff(&res.point.to_dense(), &x_star_ref) < 1e-4,
            "X* differs from truncated SVD"
        );

        // At the optimum the Riemannian gradient norm is ≈ 0.
        assert!(res.grad_norm < 1e-4, "grad norm {}", res.grad_norm);
        assert!(res.converged || res.grad_norm < 1e-4);
    }

    #[test]
    fn gradient_descent_also_reaches_oracle() {
        let m = 6;
        let n = 6;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let a = random_matrix(m, n, 202);
        let cfg = RiemannianTnConfig {
            rank: r,
            max_iter: 500,
            tol: 1e-10,
            method: RiemannianTnMethod::GradientDescent,
            ..Default::default()
        };
        let cost = |x: &[f64]| low_rank_objective(x, &a).unwrap_or(f64::INFINITY);
        let egrad = |x: &[f64]| low_rank_egrad(x, &a).unwrap_or_default();
        let res = mani.optimize(cost, egrad, None, &cfg).expect("optimize");
        let oracle = eckart_young_objective(&a, m, n, r).expect("oracle");
        assert!(
            (res.objective - oracle).abs() < 1e-5,
            "GD objective {} vs oracle {}",
            res.objective,
            oracle
        );
    }

    #[test]
    fn objective_is_monotone_nonincreasing() {
        let m = 7;
        let n = 5;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let a = random_matrix(m, n, 303);
        let cfg = RiemannianTnConfig {
            rank: r,
            max_iter: 120,
            method: RiemannianTnMethod::ConjugateGradient,
            ..Default::default()
        };
        let cost = |x: &[f64]| low_rank_objective(x, &a).unwrap_or(f64::INFINITY);
        let egrad = |x: &[f64]| low_rank_egrad(x, &a).unwrap_or_default();
        let res = mani.optimize(cost, egrad, None, &cfg).expect("optimize");
        for w in res.objective_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "objective increased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn deterministic_for_fixed_start() {
        let m = 5;
        let n = 5;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let a = random_matrix(m, n, 404);
        let cfg = RiemannianTnConfig {
            rank: r,
            seed: 12345,
            ..Default::default()
        };
        let cost = |x: &[f64]| low_rank_objective(x, &a).unwrap_or(f64::INFINITY);
        let egrad = |x: &[f64]| low_rank_egrad(x, &a).unwrap_or_default();
        let r1 = mani.optimize(cost, egrad, None, &cfg).expect("opt1");
        let r2 = mani.optimize(cost, egrad, None, &cfg).expect("opt2");
        assert_eq!(r1.iterations, r2.iterations);
        assert!((r1.objective - r2.objective).abs() < 1e-14);
        assert!(fro_diff(&r1.point.to_dense(), &r2.point.to_dense()) < 1e-12);
    }

    // ── masked completion ────────────────────────────────────────────────────────

    #[test]
    fn masked_completion_recovers_planted_rank_r() {
        let m = 10;
        let n = 9;
        let r = 2;
        let mani = FixedRankManifold::new(m, n, r).expect("manifold");
        let truth = planted_rank_r(m, n, r, 500);

        // Observe ~75% of entries.
        let mut rng = LcgRng::new(900);
        let mask: Vec<bool> = (0..m * n).map(|_| rng.next_f64() < 0.75).collect();

        let cfg = RiemannianTnConfig {
            rank: r,
            max_iter: 600,
            tol: 1e-10,
            method: RiemannianTnMethod::ConjugateGradient,
            seed: 4242,
            ..Default::default()
        };
        let truth_c = truth.clone();
        let mask_c = mask.clone();
        let cost = move |x: &[f64]| {
            low_rank_completion_objective(x, &truth_c, &mask_c).unwrap_or(f64::INFINITY)
        };
        let truth_g = truth.clone();
        let mask_g = mask.clone();
        let egrad =
            move |x: &[f64]| low_rank_completion_egrad(x, &truth_g, &mask_g).unwrap_or_default();
        let res = mani.optimize(cost, egrad, None, &cfg).expect("optimize");

        // Reconstruction recovers the *full* matrix (including unobserved entries).
        let recon = res.point.to_dense();
        let rel = fro_diff(&recon, &truth) / frob_norm(&truth).max(1e-30);
        assert!(rel < 1e-3, "completion relative error {rel}");
        // Observed-entry residual is essentially zero.
        assert!(
            res.objective < 1e-6,
            "completion objective {}",
            res.objective
        );
    }

    // ── error paths ──────────────────────────────────────────────────────────────

    #[test]
    fn rank_exceeds_min_dim_errors() {
        let err = FixedRankManifold::new(4, 3, 4);
        assert!(matches!(
            err,
            Err(TnError::RankExceedsLimit { rank: 4, max: 3 })
        ));
    }

    #[test]
    fn rank_zero_errors() {
        let err = FixedRankManifold::new(4, 4, 0);
        assert!(matches!(err, Err(TnError::InvalidRank(0))));
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            FixedRankManifold::new(0, 5, 1),
            Err(TnError::EmptyInput)
        ));
        assert!(matches!(
            FixedRankManifold::new(5, 0, 1),
            Err(TnError::EmptyInput)
        ));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mani = FixedRankManifold::new(4, 4, 2).expect("manifold");
        // Wrong-length ambient matrix for projection.
        let x = mani.random_point(1).expect("point");
        let bad = vec![0.0f64; 7];
        assert!(matches!(
            mani.project_tangent(&x, &bad),
            Err(TnError::ShapeMismatch { .. })
        ));
        // Wrong-length dense for point construction.
        assert!(matches!(
            mani.point_from_dense(&bad),
            Err(TnError::ShapeMismatch { .. })
        ));
        // Cost helper length mismatch.
        let a = vec![0.0f64; 16];
        let x_short = vec![0.0f64; 9];
        assert!(matches!(
            low_rank_objective(&x_short, &a),
            Err(TnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn config_rank_mismatch_errors() {
        let mani = FixedRankManifold::new(5, 5, 2).expect("manifold");
        let a = random_matrix(5, 5, 1);
        let cfg = RiemannianTnConfig {
            rank: 3, // does not match manifold rank 2
            ..Default::default()
        };
        let cost = |x: &[f64]| low_rank_objective(x, &a).unwrap_or(f64::INFINITY);
        let egrad = |x: &[f64]| low_rank_egrad(x, &a).unwrap_or_default();
        assert!(matches!(
            mani.optimize(cost, egrad, None, &cfg),
            Err(TnError::InvalidParameter { .. })
        ));
    }
}
