//! OSQP — Operator Splitting solver for Quadratic Programs.
//!
//! Solves `min ½ xᵀP x + qᵀx  s.t.  l ≤ A x ≤ u` via the ADMM operator-splitting
//! scheme of Stellato, Banjac, Goulart, Bemporad & Boyd (2020),
//! "OSQP: An Operator Splitting Solver for Quadratic Programs".
//!
//! # Algorithm
//!
//! The problem is reformulated by introducing a splitting variable `z = A x`
//! constrained to the box `[l, u]`. ADMM then alternates between:
//!
//! 1. **Equality-constrained QP step** — solve the linear (KKT) system
//!    ```text
//!    [ P + σI    Aᵀ    ] [ x_{k+1} ]   [ σ x_k − q          ]
//!    [ A      −(1/ρ)I  ] [ ν       ] = [ z_k − (1/ρ) y_k    ]
//!    ```
//!    and set `z̃ = A x_{k+1}`.
//! 2. **Over-relaxation** — `x̂ = α x_{k+1} + (1−α) x_k`,
//!    `z̃ᵣ = α z̃ + (1−α) z_k` with relaxation parameter `α ∈ (0, 2)`.
//! 3. **Projection step** — `z_{k+1} = clamp(z̃ᵣ + (1/ρ) y_k, l, u)`.
//! 4. **Dual update** — `y_{k+1} = y_k + ρ (z̃ᵣ − z_{k+1})`.
//!
//! Convergence is declared when the primal residual `‖A x − z‖_∞` and the dual
//! residual `‖P x + q + Aᵀ y‖_∞` both fall below `eps_abs + eps_rel · scale`,
//! where the scale terms track the magnitudes of the relevant quantities.
//!
//! The `(n+m) × (n+m)` KKT matrix is symmetric quasi-definite (positive-definite
//! `(1,1)` block, negative-definite `(2,2)` block); it is factorised once with a
//! self-contained `f32` LU decomposition with partial pivoting and re-solved each
//! iteration against the freshly assembled right-hand side.

use crate::error::{CvxError, CvxResult};

/// Configuration for the [`Osqp`] solver.
#[derive(Debug, Clone)]
pub struct OsqpConfig {
    /// ADMM penalty parameter `ρ > 0` (step size on the splitting constraint).
    pub rho: f32,
    /// Regularisation `σ > 0` added to the `(1,1)` KKT block (ensures it is SPD).
    pub sigma: f32,
    /// Over-relaxation parameter `α ∈ (0, 2)` (`α = 1` is plain ADMM).
    pub alpha: f32,
    /// Maximum number of ADMM iterations (`≥ 1`).
    pub max_iter: usize,
    /// Absolute tolerance for the termination criterion.
    pub eps_abs: f32,
    /// Relative tolerance for the termination criterion.
    pub eps_rel: f32,
}

impl Default for OsqpConfig {
    fn default() -> Self {
        Self {
            rho: 0.1,
            sigma: 1.0e-6,
            alpha: 1.6,
            max_iter: 4000,
            eps_abs: 1.0e-4,
            eps_rel: 1.0e-4,
        }
    }
}

/// Result of an [`Osqp`] solve.
#[derive(Debug, Clone)]
pub struct OsqpResult {
    /// Primal solution `x ∈ ℝⁿ`.
    pub x: Vec<f32>,
    /// Dual solution `y ∈ ℝᵐ` (multipliers of the box constraint `l ≤ A x ≤ u`).
    pub y: Vec<f32>,
    /// Number of ADMM iterations performed.
    pub iterations: usize,
    /// Whether the residual termination criterion was met.
    pub converged: bool,
    /// Objective `½ xᵀP x + qᵀx` evaluated at the returned `x`.
    pub objective: f32,
}

/// Operator-splitting QP solver (OSQP).
pub struct Osqp;

impl Osqp {
    /// Solve `min ½ xᵀP x + qᵀx  s.t.  l ≤ A x ≤ u`.
    ///
    /// * `p` — `n × n` symmetric positive-semidefinite matrix, row-major.
    /// * `q` — length-`n` linear term.
    /// * `a` — `m × n` constraint matrix, row-major.
    /// * `l`, `u` — length-`m` lower / upper bounds (use large magnitudes for `±∞`).
    /// * `n` — number of variables (`≥ 1`).
    /// * `m` — number of constraints (`≥ 0`).
    /// * `warm_start` — optional `(x0, y0)` to initialise the iterates.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::InvalidParameter`] / [`CvxError::ShapeMismatch`] /
    /// [`CvxError::DimensionMismatch`] on malformed input, and
    /// [`CvxError::SingularMatrix`] if the KKT system cannot be factorised.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        p: &[f32],
        q: &[f32],
        a: &[f32],
        l: &[f32],
        u: &[f32],
        n: usize,
        m: usize,
        cfg: &OsqpConfig,
        warm_start: Option<(&[f32], &[f32])>,
    ) -> CvxResult<OsqpResult> {
        Self::validate(p, q, a, l, u, n, m, cfg)?;

        let rho = cfg.rho;
        let sigma = cfg.sigma;
        let alpha = cfg.alpha;
        let inv_rho = 1.0_f32 / rho;
        let kkt_dim = n + m;

        // Assemble and factorise the KKT matrix once.
        //   [ P + σI    Aᵀ    ]
        //   [ A      −(1/ρ)I  ]
        let mut kkt = vec![0.0_f32; kkt_dim * kkt_dim];
        for i in 0..n {
            for j in 0..n {
                kkt[i * kkt_dim + j] = p[i * n + j];
            }
            kkt[i * kkt_dim + i] += sigma;
            for k in 0..m {
                kkt[i * kkt_dim + (n + k)] = a[k * n + i];
            }
        }
        for k in 0..m {
            for j in 0..n {
                kkt[(n + k) * kkt_dim + j] = a[k * n + j];
            }
            kkt[(n + k) * kkt_dim + (n + k)] = -inv_rho;
        }
        let factor = LuF32::factor(&kkt, kkt_dim)?;

        // Initialise iterates (warm start if supplied).
        let mut x = vec![0.0_f32; n];
        let mut y = vec![0.0_f32; m];
        if let Some((x0, y0)) = warm_start {
            if x0.len() != n {
                return Err(CvxError::DimensionMismatch { a: x0.len(), b: n });
            }
            if y0.len() != m {
                return Err(CvxError::DimensionMismatch { a: y0.len(), b: m });
            }
            x.copy_from_slice(x0);
            y.copy_from_slice(y0);
        }
        // z initialised to the (clamped) image A x of the starting point.
        let mut z = mat_vec(a, m, n, &x);
        for k in 0..m {
            z[k] = clamp(z[k], l[k], u[k]);
        }

        let mut rhs = vec![0.0_f32; kkt_dim];
        let mut iterations = 0usize;
        let mut converged = false;

        for it in 0..cfg.max_iter {
            iterations = it + 1;

            // Build RHS: top = σ x − q ; bottom = z − (1/ρ) y.
            for j in 0..n {
                rhs[j] = sigma * x[j] - q[j];
            }
            for k in 0..m {
                rhs[n + k] = z[k] - inv_rho * y[k];
            }
            let sol = factor.solve(&rhs)?;
            let x_next = &sol[..n];

            // z̃ = A x_{k+1}.
            let z_tilde = mat_vec(a, m, n, x_next);

            // Over-relaxation and z update.
            let mut z_new = vec![0.0_f32; m];
            for k in 0..m {
                let z_relaxed = alpha * z_tilde[k] + (1.0 - alpha) * z[k];
                z_new[k] = clamp(z_relaxed + inv_rho * y[k], l[k], u[k]);
                // Dual update: y_{k+1} = y_k + ρ (z̃ᵣ − z_{k+1}).
                y[k] += rho * (z_relaxed - z_new[k]);
            }

            // Over-relaxed x update (carries the relaxation into the primal too).
            for j in 0..n {
                x[j] = alpha * x_next[j] + (1.0 - alpha) * x[j];
            }
            z = z_new;

            // Termination: scaled primal / dual residuals.
            let ax = mat_vec(a, m, n, &x);
            let mut prim_res = 0.0_f32;
            let mut ax_norm = 0.0_f32;
            let mut z_norm = 0.0_f32;
            for k in 0..m {
                prim_res = prim_res.max((ax[k] - z[k]).abs());
                ax_norm = ax_norm.max(ax[k].abs());
                z_norm = z_norm.max(z[k].abs());
            }

            let px = mat_vec(p, n, n, &x);
            let aty = mat_t_vec(a, m, n, &y);
            let mut dual_res = 0.0_f32;
            let mut px_norm = 0.0_f32;
            let mut q_norm = 0.0_f32;
            let mut aty_norm = 0.0_f32;
            for j in 0..n {
                let r = px[j] + q[j] + aty[j];
                dual_res = dual_res.max(r.abs());
                px_norm = px_norm.max(px[j].abs());
                q_norm = q_norm.max(q[j].abs());
                aty_norm = aty_norm.max(aty[j].abs());
            }

            let eps_prim = cfg.eps_abs + cfg.eps_rel * ax_norm.max(z_norm);
            let eps_dual = cfg.eps_abs + cfg.eps_rel * px_norm.max(q_norm).max(aty_norm);
            if prim_res <= eps_prim && dual_res <= eps_dual {
                converged = true;
                break;
            }
        }

        let objective = quad_obj(p, q, n, &x);
        Ok(OsqpResult {
            x,
            y,
            iterations,
            converged,
            objective,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        p: &[f32],
        q: &[f32],
        a: &[f32],
        l: &[f32],
        u: &[f32],
        n: usize,
        m: usize,
        cfg: &OsqpConfig,
    ) -> CvxResult<()> {
        if n == 0 {
            return Err(CvxError::InvalidParameter(
                "OSQP requires n ≥ 1".to_string(),
            ));
        }
        if cfg.max_iter == 0 {
            return Err(CvxError::InvalidParameter(
                "OSQP requires max_iter ≥ 1".to_string(),
            ));
        }
        if cfg.alpha <= 0.0 || cfg.alpha >= 2.0 || !cfg.alpha.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "OSQP relaxation alpha must lie in (0, 2), got {}",
                cfg.alpha
            )));
        }
        if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "OSQP rho must be > 0, got {}",
                cfg.rho
            )));
        }
        if cfg.sigma <= 0.0 || !cfg.sigma.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "OSQP sigma must be > 0, got {}",
                cfg.sigma
            )));
        }
        if p.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![p.len()],
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
        if l.len() != m {
            return Err(CvxError::DimensionMismatch { a: l.len(), b: m });
        }
        if u.len() != m {
            return Err(CvxError::DimensionMismatch { a: u.len(), b: m });
        }
        for k in 0..m {
            if l[k] > u[k] {
                return Err(CvxError::InvalidParameter(format!(
                    "OSQP bound l[{k}]={} exceeds u[{k}]={}",
                    l[k], u[k]
                )));
            }
        }
        // Symmetry check on P (within tolerance).
        let sym_tol = 1.0e-4_f32;
        for i in 0..n {
            for j in (i + 1)..n {
                if (p[i * n + j] - p[j * n + i]).abs() > sym_tol {
                    return Err(CvxError::InvalidParameter(format!(
                        "OSQP matrix P is not symmetric at ({i},{j}): {} vs {}",
                        p[i * n + j],
                        p[j * n + i]
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Clamp `v` into `[lo, hi]`.
#[inline]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `y = M x` for a row-major `rows × cols` matrix.
fn mat_vec(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0_f32; rows];
    for i in 0..rows {
        let mut s = 0.0_f32;
        for j in 0..cols {
            s += mat[i * cols + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// `y = Mᵀ x` for a row-major `rows × cols` matrix (result length `cols`).
fn mat_t_vec(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0_f32; cols];
    for i in 0..rows {
        let xi = x[i];
        for j in 0..cols {
            y[j] += mat[i * cols + j] * xi;
        }
    }
    y
}

/// `½ xᵀP x + qᵀx` for a row-major `n × n` matrix `P`.
fn quad_obj(p: &[f32], q: &[f32], n: usize, x: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..n {
        let mut row = 0.0_f32;
        for j in 0..n {
            row += p[i * n + j] * x[j];
        }
        sum += 0.5 * x[i] * row + q[i] * x[i];
    }
    sum
}

/// Self-contained `f32` LU factorisation with partial pivoting (Doolittle form).
///
/// The KKT matrix is symmetric quasi-definite (indefinite), so Cholesky does not
/// apply; partial-pivoted LU is numerically robust for this class of systems.
struct LuF32 {
    lu: Vec<f32>,
    piv: Vec<usize>,
    n: usize,
}

impl LuF32 {
    /// Factorise the row-major `n × n` matrix `a`.
    fn factor(a: &[f32], n: usize) -> CvxResult<Self> {
        if a.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![a.len()],
            });
        }
        let mut lu = a.to_vec();
        let mut piv = vec![0usize; n];
        for i in 0..n {
            let mut max_v = lu[i * n + i].abs();
            let mut max_r = i;
            for r in (i + 1)..n {
                let v = lu[r * n + i].abs();
                if v > max_v {
                    max_v = v;
                    max_r = r;
                }
            }
            if max_v < 1.0e-30 {
                return Err(CvxError::SingularMatrix(format!(
                    "OSQP KKT zero pivot at column {i}"
                )));
            }
            piv[i] = max_r;
            if max_r != i {
                for c in 0..n {
                    lu.swap(i * n + c, max_r * n + c);
                }
            }
            let inv_pivot = 1.0_f32 / lu[i * n + i];
            for r in (i + 1)..n {
                let f = lu[r * n + i] * inv_pivot;
                lu[r * n + i] = f;
                for c in (i + 1)..n {
                    let v = lu[r * n + c] - f * lu[i * n + c];
                    lu[r * n + c] = v;
                }
            }
        }
        Ok(Self { lu, piv, n })
    }

    /// Solve `A x = b` using the stored factorisation.
    fn solve(&self, b: &[f32]) -> CvxResult<Vec<f32>> {
        let n = self.n;
        if b.len() != n {
            return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
        }
        let mut x = b.to_vec();
        for (i, &p) in self.piv.iter().enumerate().take(n) {
            if p != i {
                x.swap(i, p);
            }
        }
        // Forward substitution (unit lower triangular L).
        for i in 0..n {
            let row = &self.lu[i * n..i * n + i];
            let mut s = x[i];
            for (xj, &lij) in x.iter().zip(row.iter()) {
                s -= lij * xj;
            }
            x[i] = s;
        }
        // Back substitution (upper triangular U).
        for i in (0..n).rev() {
            let row = &self.lu[i * n + i + 1..i * n + n];
            let mut s = x[i];
            for (xj, &uij) in x[i + 1..n].iter().zip(row.iter()) {
                s -= uij * xj;
            }
            let d = self.lu[i * n + i];
            if d.abs() < 1.0e-30 {
                return Err(CvxError::SingularMatrix(format!(
                    "OSQP KKT zero U[{i},{i}]"
                )));
            }
            x[i] = s / d;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a relaxed config: more iterations, tighter tolerance for tests.
    fn tight_cfg() -> OsqpConfig {
        OsqpConfig {
            rho: 0.5,
            sigma: 1.0e-6,
            alpha: 1.6,
            max_iter: 8000,
            eps_abs: 1.0e-6,
            eps_rel: 1.0e-6,
        }
    }

    const BIG: f32 = 1.0e8;

    #[test]
    fn unconstrained_quadratic_no_rows() {
        // min ½ xᵀP x + qᵀx, P = diag(2, 3), q = (-2, -3) → x = P⁻¹(-q) = (1, 1).
        let p = vec![2.0_f32, 0.0, 0.0, 3.0];
        let q = vec![-2.0_f32, -3.0];
        let a: Vec<f32> = Vec::new();
        let l: Vec<f32> = Vec::new();
        let u: Vec<f32> = Vec::new();
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 0, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1.0e-3, "x1 = {}", res.x[1]);
    }

    #[test]
    fn unconstrained_quadratic_infinite_bounds() {
        // Same problem but with two trivially-loose constraints (l = −big, u = +big).
        let p = vec![2.0_f32, 0.0, 0.0, 2.0];
        let q = vec![-4.0_f32, 6.0];
        // A = I, bounds wide open → unconstrained optimum x = (2, -3).
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = vec![-BIG, -BIG];
        let u = vec![BIG, BIG];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 2, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 2.0).abs() < 1.0e-2, "x0 = {}", res.x[0]);
        assert!((res.x[1] + 3.0).abs() < 1.0e-2, "x1 = {}", res.x[1]);
    }

    #[test]
    fn equality_constrained_known_solution() {
        // min ½ ‖x‖² s.t. x0 + x1 = 1 (l == u == 1). Optimum: (0.5, 0.5).
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![1.0_f32];
        let u = vec![1.0_f32];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 0.5).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 0.5).abs() < 1.0e-3, "x1 = {}", res.x[1]);
    }

    #[test]
    fn box_constrained_clamps_at_bound() {
        // min ½ ‖x − t‖² with t = (5, 5), s.t. 0 ≤ x ≤ 1 (A = I). Optimum: (1, 1).
        // ½‖x−t‖² = ½ xᵀx − tᵀx + const → P = I, q = −t.
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![-5.0_f32, -5.0];
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = vec![0.0_f32, 0.0];
        let u = vec![1.0_f32, 1.0];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 2, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-2, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1.0e-2, "x1 = {}", res.x[1]);
    }

    #[test]
    fn box_constrained_interior_solution() {
        // Same shape but t = (0.3, 0.7) inside [0, 1] → optimum equals t.
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![-0.3_f32, -0.7];
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = vec![0.0_f32, 0.0];
        let u = vec![1.0_f32, 1.0];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 2, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 0.3).abs() < 1.0e-2, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 0.7).abs() < 1.0e-2, "x1 = {}", res.x[1]);
    }

    #[test]
    fn converged_flag_well_conditioned() {
        let p = vec![3.0_f32, 0.5, 0.5, 2.0];
        let q = vec![1.0_f32, -2.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![-BIG];
        let u = vec![BIG];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
    }

    #[test]
    fn objective_below_initial_point() {
        // The returned objective should not exceed the objective at x = 0.
        let p = vec![2.0_f32, 0.0, 0.0, 2.0];
        let q = vec![-2.0_f32, -3.0];
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = vec![-BIG, -BIG];
        let u = vec![BIG, BIG];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 2, &tight_cfg(), None).expect("solve");
        let obj_zero = quad_obj(&p, &q, 2, &[0.0, 0.0]);
        assert!(
            res.objective < obj_zero,
            "obj {} >= {}",
            res.objective,
            obj_zero
        );
    }

    #[test]
    fn warm_start_from_optimum_fast() {
        // Equality-constrained optimum (0.5, 0.5), y from KKT stationarity.
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![1.0_f32];
        let u = vec![1.0_f32];
        // Stationarity: P x + q + Aᵀ y = 0 → (0.5 + y, 0.5 + y) = 0 → y = -0.5.
        let x0 = vec![0.5_f32, 0.5];
        let y0 = vec![-0.5_f32];
        let res =
            Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), Some((&x0, &y0))).expect("solve");
        assert!(res.converged);
        assert!(res.iterations <= 5, "iterations = {}", res.iterations);
    }

    #[test]
    fn kkt_residual_small_at_solution() {
        let p = vec![2.0_f32, 0.0, 0.0, 4.0];
        let q = vec![1.0_f32, -1.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![0.5_f32];
        let u = vec![0.5_f32];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), None).expect("solve");
        // Dual residual: ‖P x + q + Aᵀ y‖_∞.
        let px = mat_vec(&p, 2, 2, &res.x);
        let aty = mat_t_vec(&a, 1, 2, &res.y);
        let mut dual_res = 0.0_f32;
        for ((&pxj, &qj), &atyj) in px.iter().zip(q.iter()).zip(aty.iter()) {
            dual_res = dual_res.max((pxj + qj + atyj).abs());
        }
        // Primal residual: ‖A x − z‖ where z is feasible; here Ax should ≈ 0.5.
        let ax = mat_vec(&a, 1, 2, &res.x);
        assert!((ax[0] - 0.5).abs() < 1.0e-3, "Ax = {}", ax[0]);
        assert!(dual_res < 1.0e-2, "dual residual = {dual_res}");
    }

    #[test]
    fn deterministic_repeated_runs() {
        let p = vec![2.0_f32, 0.3, 0.3, 3.0];
        let q = vec![-1.0_f32, 2.0];
        let a = vec![1.0_f32, -1.0];
        let l = vec![-2.0_f32];
        let u = vec![2.0_f32];
        let r1 = Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), None).expect("solve");
        let r2 = Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &tight_cfg(), None).expect("solve");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.y, r2.y);
    }

    #[test]
    fn no_relaxation_alpha_one_solves() {
        let cfg = OsqpConfig {
            alpha: 1.0,
            ..tight_cfg()
        };
        // min ½ xᵀP x + qᵀx, P = diag(2, 2), q = (-2, -2) → x = (1, 1).
        let p = vec![2.0_f32, 0.0, 0.0, 2.0];
        let q = vec![-2.0_f32, -2.0];
        let a: Vec<f32> = Vec::new();
        let l: Vec<f32> = Vec::new();
        let u: Vec<f32> = Vec::new();
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 0, &cfg, None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-3);
        assert!((res.x[1] - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn one_dimensional_problem() {
        // min ½ * 2 * x² − 6 x → x = 3.
        let p = vec![2.0_f32];
        let q = vec![-6.0_f32];
        let a: Vec<f32> = Vec::new();
        let l: Vec<f32> = Vec::new();
        let u: Vec<f32> = Vec::new();
        let res = Osqp::solve(&p, &q, &a, &l, &u, 1, 0, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!((res.x[0] - 3.0).abs() < 1.0e-3, "x = {}", res.x[0]);
    }

    #[test]
    fn identity_p_clamped() {
        // P = I, q = (2, -3), constraint 0 ≤ x ≤ big via A = I.
        // Unconstrained x = -q = (-2, 3); lower bound clamps x0 to 0.
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![2.0_f32, -3.0];
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = vec![0.0_f32, 0.0];
        let u = vec![BIG, BIG];
        let res = Osqp::solve(&p, &q, &a, &l, &u, 2, 2, &tight_cfg(), None).expect("solve");
        assert!(res.converged);
        assert!(res.x[0].abs() < 1.0e-2, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 3.0).abs() < 1.0e-2, "x1 = {}", res.x[1]);
    }

    #[test]
    fn err_p_not_square() {
        let p = vec![1.0_f32, 0.0, 0.0]; // length 3, not 4.
        let q = vec![0.0_f32, 0.0];
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &OsqpConfig::default(), None),
            Err(CvxError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn err_q_wrong_len() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32]; // should be 2.
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &OsqpConfig::default(), None),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_a_wrong_size() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let a = vec![1.0_f32, 1.0, 1.0]; // should be m*n = 1*2 = 2.
        let l = vec![0.0_f32];
        let u = vec![1.0_f32];
        assert!(matches!(
            Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &OsqpConfig::default(), None),
            Err(CvxError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn err_lu_wrong_len() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![0.0_f32];
        let u = vec![1.0_f32, 2.0]; // wrong length.
        assert!(matches!(
            Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &OsqpConfig::default(), None),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_l_greater_than_u() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let a = vec![1.0_f32, 1.0];
        let l = vec![2.0_f32];
        let u = vec![1.0_f32]; // l > u.
        assert!(matches!(
            Osqp::solve(&p, &q, &a, &l, &u, 2, 1, &OsqpConfig::default(), None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_alpha_out_of_range() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let cfg = OsqpConfig {
            alpha: 2.0,
            ..OsqpConfig::default()
        };
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &cfg, None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_rho_non_positive() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let cfg = OsqpConfig {
            rho: 0.0,
            ..OsqpConfig::default()
        };
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &cfg, None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_max_iter_zero() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let cfg = OsqpConfig {
            max_iter: 0,
            ..OsqpConfig::default()
        };
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &cfg, None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_sigma_non_positive() {
        let p = vec![1.0_f32, 0.0, 0.0, 1.0];
        let q = vec![0.0_f32, 0.0];
        let cfg = OsqpConfig {
            sigma: 0.0,
            ..OsqpConfig::default()
        };
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &cfg, None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_p_not_symmetric() {
        let p = vec![1.0_f32, 5.0, 0.0, 1.0]; // off-diagonals 5 vs 0.
        let q = vec![0.0_f32, 0.0];
        assert!(matches!(
            Osqp::solve(&p, &q, &[], &[], &[], 2, 0, &OsqpConfig::default(), None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_n_zero() {
        let cfg = OsqpConfig::default();
        assert!(matches!(
            Osqp::solve(&[], &[], &[], &[], &[], 0, 0, &cfg, None),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn lu_solves_indefinite_system() {
        // KKT-like quasi-definite matrix [[1, 1], [1, -1]] (indefinite).
        let m = vec![1.0_f32, 1.0, 1.0, -1.0];
        let f = LuF32::factor(&m, 2).expect("factor");
        let sol = f.solve(&[3.0, 1.0]).expect("solve");
        // x0 + x1 = 3 ; x0 − x1 = 1 → x0 = 2, x1 = 1.
        assert!((sol[0] - 2.0).abs() < 1.0e-5);
        assert!((sol[1] - 1.0).abs() < 1.0e-5);
    }
}
