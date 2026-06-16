//! Splitting Conic Solver (SCS) — operator-splitting solver for conic programs.
//!
//! Solves the standard-form conic program
//!
//! ```text
//!   minimize    cᵀ x
//!   subject to  A x + s = b,
//!               s ∈ K,
//! ```
//!
//! where `K` is a Cartesian product of supported cones (zero / non-negative /
//! second-order). The formulation and the operator-splitting (ADMM) iteration
//! follow O'Donoghue, Chu, Parikh & Boyd (2016), "Conic Optimization via
//! Operator Splitting and Homogeneous Self-Dual Embedding".
//!
//! # Algorithm
//!
//! Introduce the slack `s = b − A x ∈ K` and the scaled dual `λ`. Eliminating
//! `s` from the augmented Lagrangian and minimising over `x` gives a fixed
//! linear system whose normal-equation form is
//!
//! ```text
//!   (Aᵀ A) x⁺ = Aᵀ (b − s − λ) − c / ρ.
//! ```
//!
//! The Gram matrix `Aᵀ A` is symmetric positive semidefinite; a small Tikhonov
//! term `reg·I` is added so the factorisation is well defined even when `A` is
//! rank deficient. The full ADMM sweep is
//!
//! ```text
//!   x⁺ = (Aᵀ A + reg·I)⁻¹ ( Aᵀ (b − s − λ) − c / ρ ),
//!   s⁺ = Π_K( b − A x⁺ − λ ),
//!   λ⁺ = λ + ( A x⁺ + s⁺ − b ).
//! ```
//!
//! The dual variable of the equality `A x + s = b` is recovered as `y = ρ λ`,
//! the scaled multiplier of the augmented Lagrangian. Convergence is monitored
//! through the primal residual `‖A x + s − b‖₂` and the dual residual
//! `‖Aᵀ y + c‖₂`; both falling below tolerance reports [`ScsStatus::Solved`].

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::{lu_decompose, lu_solve};

/// A single cone in the Cartesian product cone `K`.
///
/// The cones are stacked in the order they appear in the slice passed to
/// [`scs_solve`], partitioning the constraint rows from top to bottom. The sum
/// of all cone dimensions must equal the number of constraint rows `m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cone {
    /// The zero cone `{0}ⁿ` — enforces equality constraints (`s = 0`). The
    /// Euclidean projection of any point onto `{0}` is the origin.
    Zero(usize),
    /// The non-negative orthant `ℝⁿ₊`. Projection is the component-wise
    /// positive part `max(·, 0)`.
    NonNegative(usize),
    /// The second-order (Lorentz) cone of total dimension `n`,
    /// `{ (t, z) ∈ ℝ × ℝⁿ⁻¹ : ‖z‖₂ ≤ t }`. The first coordinate of the block is
    /// the scalar `t`; the remaining `n − 1` coordinates form the vector `z`.
    SecondOrder(usize),
}

impl Cone {
    /// Dimension (number of constraint rows) occupied by this cone.
    #[must_use]
    pub fn dim(&self) -> usize {
        match *self {
            Cone::Zero(n) | Cone::NonNegative(n) | Cone::SecondOrder(n) => n,
        }
    }
}

/// Termination status of [`scs_solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScsStatus {
    /// Both primal and dual residuals fell below tolerance.
    Solved,
    /// A certificate of primal infeasibility was detected.
    Infeasible,
    /// A certificate of unboundedness (dual infeasibility) was detected.
    Unbounded,
    /// The iteration limit was reached before convergence.
    MaxIter,
}

/// Configuration for the SCS solver.
#[derive(Debug, Clone)]
pub struct ScsConfig {
    /// ADMM penalty parameter `ρ > 0` (scales the linear term `c / ρ`).
    pub rho: f64,
    /// Maximum number of ADMM iterations (`≥ 1`).
    pub max_iter: usize,
    /// Absolute tolerance on the primal residual `‖A x + s − b‖₂`.
    pub eps_primal: f64,
    /// Absolute tolerance on the dual residual `‖Aᵀ λ + c‖₂`.
    pub eps_dual: f64,
    /// Extra Tikhonov regularisation added to `Aᵀ A` on top of the implicit
    /// `1/ρ` term, guarding against rank deficiency.
    pub reg: f64,
}

impl Default for ScsConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iter: 5000,
            eps_primal: 1.0e-6,
            eps_dual: 1.0e-6,
            reg: 1.0e-8,
        }
    }
}

/// Result of an SCS solve.
#[derive(Debug, Clone)]
pub struct ScsResult {
    /// Primal solution `x ∈ ℝⁿ`.
    pub x: Vec<f64>,
    /// Slack `s ∈ K ⊂ ℝᵐ` with `A x + s ≈ b`.
    pub s: Vec<f64>,
    /// Dual solution `y ∈ ℝᵐ` (the multiplier of `A x + s = b`).
    pub y: Vec<f64>,
    /// Objective `cᵀ x` evaluated at the returned `x`.
    pub objective: f64,
    /// Final primal residual `‖A x + s − b‖₂`.
    pub primal_residual: f64,
    /// Final dual residual `‖Aᵀ y + c‖₂`.
    pub dual_residual: f64,
    /// Number of ADMM iterations performed.
    pub iterations: usize,
    /// Termination status.
    pub status: ScsStatus,
}

/// Project a point `v ∈ ℝᵐ` onto the product cone `K` described by `cones`.
///
/// The cones partition `v` top to bottom; each block is projected
/// independently. Returns the projected point `Π_K(v)`.
fn project_cone(v: &[f64], cones: &[Cone]) -> CvxResult<Vec<f64>> {
    let mut out = vec![0.0_f64; v.len()];
    let mut offset = 0usize;
    for cone in cones {
        let n = cone.dim();
        if offset + n > v.len() {
            return Err(CvxError::DimensionMismatch {
                a: offset + n,
                b: v.len(),
            });
        }
        let block = &v[offset..offset + n];
        match *cone {
            Cone::Zero(_) => {
                // Π_{0}(v) = 0.
                for slot in out[offset..offset + n].iter_mut() {
                    *slot = 0.0;
                }
            }
            Cone::NonNegative(_) => {
                // Π_{ℝ₊}(v) = max(v, 0) component-wise.
                for (slot, &val) in out[offset..offset + n].iter_mut().zip(block.iter()) {
                    *slot = val.max(0.0);
                }
            }
            Cone::SecondOrder(_) => {
                project_soc_block(block, &mut out[offset..offset + n]);
            }
        }
        offset += n;
    }
    if offset != v.len() {
        return Err(CvxError::DimensionMismatch {
            a: offset,
            b: v.len(),
        });
    }
    Ok(out)
}

/// Project a single second-order-cone block `(t, z)` into `dst`.
///
/// With `t = block[0]` and `z = block[1..]`:
/// - if `‖z‖ ≤ t` the point is already inside → identity;
/// - if `‖z‖ ≤ −t` the projection is the origin;
/// - otherwise scale by `α = (‖z‖ + t) / (2 ‖z‖)`, giving `(α‖z‖, α z)`.
fn project_soc_block(block: &[f64], dst: &mut [f64]) {
    if block.is_empty() {
        return;
    }
    let t = block[0];
    let z = &block[1..];
    let nz = norm2(z);
    if nz <= t {
        // Interior (or boundary from inside): identity.
        dst.copy_from_slice(block);
    } else if nz <= -t {
        // Polar interior: projects to the origin.
        for slot in dst.iter_mut() {
            *slot = 0.0;
        }
    } else {
        let scale = (nz + t) / (2.0 * nz);
        dst[0] = scale * nz;
        for (slot, &zi) in dst[1..].iter_mut().zip(z.iter()) {
            *slot = scale * zi;
        }
    }
}

/// Validate the cone partition against the constraint row count.
fn validate_cones(cones: &[Cone], m: usize) -> CvxResult<()> {
    let total: usize = cones.iter().map(Cone::dim).sum();
    if total != m {
        return Err(CvxError::InvalidConfiguration(format!(
            "cone dimensions sum to {total} but there are {m} constraint rows"
        )));
    }
    Ok(())
}

/// Solve the standard-form conic program `min cᵀx s.t. A x + s = b, s ∈ K`.
///
/// * `a` — `m × n` constraint matrix in row-major order.
/// * `b` — length-`m` right-hand side.
/// * `c` — length-`n` objective vector.
/// * `cones` — partition of the `m` rows into product cones (top to bottom).
/// * `n` — number of variables (`≥ 1`).
/// * `m` — number of constraint rows.
/// * `cfg` — solver configuration.
///
/// # Errors
///
/// Returns [`CvxError::InvalidParameter`] / [`CvxError::ShapeMismatch`] /
/// [`CvxError::DimensionMismatch`] on malformed input,
/// [`CvxError::InvalidConfiguration`] if the cones do not tile the rows, and
/// [`CvxError::SingularMatrix`] if the regularised Gram system cannot be
/// factorised.
pub fn scs_solve(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    cones: &[Cone],
    n: usize,
    m: usize,
    cfg: &ScsConfig,
) -> CvxResult<ScsResult> {
    if n == 0 {
        return Err(CvxError::InvalidParameter("SCS requires n ≥ 1".to_string()));
    }
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "SCS requires max_iter ≥ 1".to_string(),
        ));
    }
    if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "SCS rho must be > 0, got {}",
            cfg.rho
        )));
    }
    if cfg.reg <= 0.0 || !cfg.reg.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "SCS reg must be > 0 (Tikhonov term for the Gram factorisation), got {}",
            cfg.reg
        )));
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
    validate_cones(cones, m)?;

    // Gram matrix G = Aᵀ A + (reg) I. Only a small Tikhonov term is added so the
    // factorisation is well defined under rank deficiency; adding a larger term
    // would bias the recovered dual away from stationarity (see below).
    let mut gram = vec![0.0_f64; n * n];
    for k in 0..m {
        let row = k * n;
        for i in 0..n {
            let aki = a[row + i];
            for j in 0..n {
                gram[i * n + j] += aki * a[row + j];
            }
        }
    }
    for i in 0..n {
        gram[i * n + i] += cfg.reg;
    }
    let (lu, piv) = lu_decompose(&gram, n)?;

    let mut x = vec![0.0_f64; n];
    let mut s = vec![0.0_f64; m];
    let mut lambda = vec![0.0_f64; m];

    let mut primal_residual = f64::INFINITY;
    let mut dual_residual = f64::INFINITY;
    let mut iterations = 0usize;
    let mut status = ScsStatus::MaxIter;

    for it in 0..cfg.max_iter {
        iterations = it + 1;

        // RHS = Aᵀ (b − s − λ) − c / ρ.
        let mut tmp = vec![0.0_f64; m];
        for k in 0..m {
            tmp[k] = b[k] - s[k] - lambda[k];
        }
        let mut rhs = mat_t_vec(a, m, n, &tmp)?;
        for j in 0..n {
            rhs[j] -= c[j] / cfg.rho;
        }
        // x⁺ = G⁻¹ rhs.
        x = lu_solve(&lu, &piv, n, &rhs)?;

        // s⁺ = Π_K(b − A x⁺ − λ).
        let ax = mat_vec(a, m, n, &x)?;
        let mut pre = vec![0.0_f64; m];
        for k in 0..m {
            pre[k] = b[k] - ax[k] - lambda[k];
        }
        s = project_cone(&pre, cones)?;

        // λ⁺ = λ + (A x⁺ + s⁺ − b); also primal residual r = A x⁺ + s⁺ − b.
        let mut r = vec![0.0_f64; m];
        for k in 0..m {
            r[k] = ax[k] + s[k] - b[k];
            lambda[k] += r[k];
        }
        primal_residual = norm2(&r);

        // The recovered dual variable of `A x + s = b` is `y = ρ λ`: the scaled
        // multiplier of the augmented Lagrangian. Dual feasibility is
        // ‖Aᵀ y + c‖₂, which vanishes at the fixed point (up to the tiny
        // `reg`-term, since at stationarity Aᵀ(ρλ) + c = −ρ·reg·x).
        let y_iter: Vec<f64> = lambda.iter().map(|li| cfg.rho * li).collect();
        let mut aty = mat_t_vec(a, m, n, &y_iter)?;
        for j in 0..n {
            aty[j] += c[j];
        }
        dual_residual = norm2(&aty);

        if primal_residual < cfg.eps_primal && dual_residual < cfg.eps_dual {
            status = ScsStatus::Solved;
            break;
        }
    }

    let objective = c.iter().zip(x.iter()).map(|(ci, xi)| ci * xi).sum();
    let y: Vec<f64> = lambda.iter().map(|li| cfg.rho * li).collect();
    Ok(ScsResult {
        x,
        s,
        y,
        objective,
        primal_residual,
        dual_residual,
        iterations,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tight_cfg() -> ScsConfig {
        ScsConfig {
            rho: 1.0,
            max_iter: 50_000,
            eps_primal: 1.0e-7,
            eps_dual: 1.0e-7,
            reg: 1.0e-9,
        }
    }

    #[test]
    fn cone_dim_reports_size() {
        assert_eq!(Cone::Zero(3).dim(), 3);
        assert_eq!(Cone::NonNegative(2).dim(), 2);
        assert_eq!(Cone::SecondOrder(4).dim(), 4);
    }

    #[test]
    fn zero_cone_projects_to_origin() {
        let v = vec![1.0, -2.0, 3.0];
        let p = project_cone(&v, &[Cone::Zero(3)]).expect("project");
        for pi in p {
            assert!(pi.abs() < 1.0e-15);
        }
    }

    #[test]
    fn nonneg_cone_clamps_negative() {
        let v = vec![1.0, -2.0, 0.5, -0.1];
        let p = project_cone(&v, &[Cone::NonNegative(4)]).expect("project");
        assert_eq!(p, vec![1.0, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn soc_interior_point_unchanged() {
        // (t, z) = (5, (3, 4)): ‖z‖ = 5 ≤ t = 5 → boundary-from-inside identity.
        let v = vec![5.0, 3.0, 4.0];
        let p = project_cone(&v, &[Cone::SecondOrder(3)]).expect("project");
        assert!((p[0] - 5.0).abs() < 1.0e-12);
        assert!((p[1] - 3.0).abs() < 1.0e-12);
        assert!((p[2] - 4.0).abs() < 1.0e-12);

        // Strict interior: (t, z) = (10, (3, 4)).
        let v2 = vec![10.0, 3.0, 4.0];
        let p2 = project_cone(&v2, &[Cone::SecondOrder(3)]).expect("project");
        assert_eq!(p2, v2);
    }

    #[test]
    fn soc_polar_interior_maps_to_zero() {
        // (t, z) = (-5, (3, 4)): ‖z‖ = 5 ≤ −t = 5 → origin.
        let v = vec![-5.0, 3.0, 4.0];
        let p = project_cone(&v, &[Cone::SecondOrder(3)]).expect("project");
        for pi in p {
            assert!(pi.abs() < 1.0e-12);
        }
        // Deep in the polar cone.
        let v2 = vec![-10.0, 3.0, 4.0];
        let p2 = project_cone(&v2, &[Cone::SecondOrder(3)]).expect("project");
        for pi in p2 {
            assert!(pi.abs() < 1.0e-12);
        }
    }

    #[test]
    fn soc_boundary_case_scaled_and_in_cone() {
        // (t, z) = (0, (3, 4)): ‖z‖ = 5, t = 0 → α = (5 + 0)/(2·5) = 0.5.
        // result = (0.5·5, 0.5·(3,4)) = (2.5, (1.5, 2)).
        let v = vec![0.0, 3.0, 4.0];
        let p = project_cone(&v, &[Cone::SecondOrder(3)]).expect("project");
        assert!((p[0] - 2.5).abs() < 1.0e-12);
        assert!((p[1] - 1.5).abs() < 1.0e-12);
        assert!((p[2] - 2.0).abs() < 1.0e-12);
        // The result lies in the cone: ‖z'‖ ≤ t'.
        let t_proj = p[0];
        let z_proj_norm = norm2(&p[1..]);
        assert!(
            z_proj_norm <= t_proj + 1.0e-12,
            "‖z'‖={z_proj_norm} > t'={t_proj}"
        );
    }

    #[test]
    fn validate_cones_rejects_mismatch() {
        assert!(validate_cones(&[Cone::Zero(2)], 3).is_err());
        assert!(validate_cones(&[Cone::Zero(1), Cone::NonNegative(2)], 3).is_ok());
    }

    #[test]
    fn lp_equality_plus_nonneg() {
        // min −x₁ s.t. x₁ + x₂ = 1 (Zero cone), x ≥ 0 (NonNeg).
        // Standard SCS form: A x + s = b, s ∈ K.
        // Row 0: equality x₁ + x₂ = 1   → s₀ = 0  (Zero cone).
        // Rows 1,2: −x ≤ 0 i.e. −xᵢ + sᵢ = 0, sᵢ ≥ 0 → xᵢ ≥ 0 (NonNeg).
        // c = (−1, 0). Optimum: x = (1, 0), objective = −1.
        let n = 2;
        let m = 3;
        #[rustfmt::skip]
        let a = vec![
            1.0, 1.0,   // equality
            -1.0, 0.0,  // x₁ ≥ 0
            0.0, -1.0,  // x₂ ≥ 0
        ];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0];
        let cones = [Cone::Zero(1), Cone::NonNegative(2)];
        let res = scs_solve(&a, &b, &c, &cones, n, m, &tight_cfg()).expect("solve");
        assert_eq!(res.status, ScsStatus::Solved);
        assert!((res.x[0] - 1.0).abs() < 1.0e-4, "x0 = {}", res.x[0]);
        assert!(res.x[1].abs() < 1.0e-4, "x1 = {}", res.x[1]);
        assert!(
            (res.objective + 1.0).abs() < 1.0e-4,
            "obj = {}",
            res.objective
        );
    }

    #[test]
    fn box_type_lp_to_origin() {
        // min x₁ + x₂ s.t. 0 ≤ xᵢ ≤ 1.
        // Constraints as A x + s = b, s ∈ ℝ₊⁴:
        //   −x₁ + s = 0 (x₁ ≥ 0),  x₁ + s = 1 (x₁ ≤ 1),
        //   −x₂ + s = 0 (x₂ ≥ 0),  x₂ + s = 1 (x₂ ≤ 1).
        // c = (1, 1). Optimum: x = (0, 0).
        let n = 2;
        let m = 4;
        #[rustfmt::skip]
        let a = vec![
            -1.0, 0.0,
             1.0, 0.0,
             0.0, -1.0,
             0.0, 1.0,
        ];
        let b = vec![0.0, 1.0, 0.0, 1.0];
        let c = vec![1.0, 1.0];
        let cones = [Cone::NonNegative(4)];
        let res = scs_solve(&a, &b, &c, &cones, n, m, &tight_cfg()).expect("solve");
        assert_eq!(res.status, ScsStatus::Solved);
        assert!(res.x[0].abs() < 1.0e-4, "x0 = {}", res.x[0]);
        assert!(res.x[1].abs() < 1.0e-4, "x1 = {}", res.x[1]);
    }

    #[test]
    fn convergence_satisfies_feasibility_and_stationarity() {
        // Reuse the equality + nonneg LP and verify both residual bounds.
        let n = 2;
        let m = 3;
        #[rustfmt::skip]
        let a = vec![
            1.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0];
        let cones = [Cone::Zero(1), Cone::NonNegative(2)];
        let res = scs_solve(&a, &b, &c, &cones, n, m, &tight_cfg()).expect("solve");
        // Primal feasibility ‖A x* + s* − b‖ < 1e-4.
        let ax = mat_vec(&a, m, n, &res.x).expect("ax");
        let mut r = vec![0.0_f64; m];
        for k in 0..m {
            r[k] = ax[k] + res.s[k] - b[k];
        }
        assert!(norm2(&r) < 1.0e-4, "primal residual = {}", norm2(&r));
        // Dual feasibility ‖Aᵀ y* + c‖ < 1e-4.
        let mut aty = mat_t_vec(&a, m, n, &res.y).expect("aty");
        for j in 0..n {
            aty[j] += c[j];
        }
        assert!(norm2(&aty) < 1.0e-4, "dual residual = {}", norm2(&aty));
    }

    #[test]
    fn equality_plus_soc_known_optimum() {
        // Variables x = (x0, x1, x2). Equality x0 = 1.
        // SOC over (x0, x1, x2): ‖(x1, x2)‖ ≤ x0.
        // minimize x1  → with x0 = 1, the SOC is the disk x1² + x2² ≤ 1.
        // Optimum: x1 = −1, x2 = 0; objective = −1.
        //
        // Rows: equality (Zero 1): x0 + s0 = 1, s0 = 0.
        //       SOC (3): introduce s = (s_t, s_a, s_b) with
        //         x0 + s_t = 0 → s_t = −x0 ... we instead place the cone directly
        //         on the variables by A = −I on the SOC block, b = 0, so
        //         s = (x0, x1, x2) ∈ SOC ⇒ ‖(x1, x2)‖ ≤ x0.
        let n = 3;
        let m = 4;
        #[rustfmt::skip]
        let a = vec![
            // equality x0 = 1
            1.0, 0.0, 0.0,
            // SOC block: s = -A x + b = (x0, x1, x2) when A = -I, b = 0
            -1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
        ];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let cones = [Cone::Zero(1), Cone::SecondOrder(3)];
        let cfg = ScsConfig {
            rho: 1.0,
            max_iter: 200_000,
            eps_primal: 1.0e-6,
            eps_dual: 1.0e-6,
            reg: 1.0e-9,
        };
        let res = scs_solve(&a, &b, &c, &cones, n, m, &cfg).expect("solve");
        assert_eq!(res.status, ScsStatus::Solved);
        assert!((res.x[0] - 1.0).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!((res.x[1] + 1.0).abs() < 5.0e-3, "x1 = {}", res.x[1]);
        assert!(res.x[2].abs() < 5.0e-3, "x2 = {}", res.x[2]);
        assert!(
            (res.objective + 1.0).abs() < 5.0e-3,
            "obj = {}",
            res.objective
        );
    }

    #[test]
    fn err_n_zero() {
        let cfg = ScsConfig::default();
        assert!(matches!(
            scs_solve(&[], &[], &[], &[], 0, 0, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_rho_non_positive() {
        let cfg = ScsConfig {
            rho: 0.0,
            ..ScsConfig::default()
        };
        assert!(matches!(
            scs_solve(&[1.0], &[0.0], &[1.0], &[Cone::Zero(1)], 1, 1, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_a_shape_mismatch() {
        let cfg = ScsConfig::default();
        let a = vec![1.0, 1.0, 1.0]; // should be m*n = 1*2 = 2.
        assert!(matches!(
            scs_solve(&a, &[0.0], &[1.0, 0.0], &[Cone::Zero(1)], 2, 1, &cfg),
            Err(CvxError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn err_cone_partition_mismatch() {
        let cfg = ScsConfig::default();
        let a = vec![1.0, 1.0]; // m=1, n=2.
        assert!(matches!(
            scs_solve(&a, &[1.0], &[1.0, 0.0], &[Cone::Zero(2)], 2, 1, &cfg),
            Err(CvxError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn deterministic_repeated_runs() {
        let n = 2;
        let m = 3;
        #[rustfmt::skip]
        let a = vec![
            1.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0];
        let cones = [Cone::Zero(1), Cone::NonNegative(2)];
        let r1 = scs_solve(&a, &b, &c, &cones, n, m, &tight_cfg()).expect("solve");
        let r2 = scs_solve(&a, &b, &c, &cones, n, m, &tight_cfg()).expect("solve");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.x, r2.x);
    }
}
