//! Dykstra's Alternating Projections (POCS) algorithm.
//!
//! Finds the projection of an initial point `x₀` onto the intersection of
//! `m` closed convex sets `C₁ ∩ C₂ ∩ … ∩ Cₘ` using the incremental correction
//! scheme introduced by Dykstra (1983).
//!
//! # Algorithm (Dykstra 1983)
//!
//! ```text
//! Init: z ← x₀,  pᵢ ← 0  (i = 1..m)
//! repeat until convergence:
//!   z_prev ← z
//!   for i in 0..m:
//!     v      ← z + pᵢ          (shift by Dykstra correction)
//!     y      ← Proj_{Cᵢ}(v)   (project shifted point)
//!     pᵢ    ← v − y            (update correction)
//!     z      ← y
//!   if ‖z − z_prev‖₂ < tol: break
//! return z
//! ```
//!
//! Unlike the plain alternating-projections method (von Neumann), Dykstra's
//! corrections guarantee convergence to the *nearest point* in the intersection
//! (not merely any point) whenever each `Proj_{Cᵢ}` is the metric projection.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Borrowed metric-projection operator: maps a slice of length `n` to a
/// projected `Vec<f64>` of the same length (or a [`CvxError`]).
pub type ProjFn<'a> = &'a dyn Fn(&[f64]) -> CvxResult<Vec<f64>>;

// ── Public types ─────────────────────────────────────────────────────────────

/// Outcome of [`dykstra_pocs`].
#[derive(Debug, Clone)]
pub struct DykstraResult {
    /// The computed projection (or best iterate if `!converged`).
    pub x: Vec<f64>,
    /// Number of outer iterations executed.
    pub iter: usize,
    /// Final iterate-change residual `‖z_k − z_{k-1}‖₂`.
    pub residual: f64,
    /// `true` if the residual dropped below `tol` before `max_iter`.
    pub converged: bool,
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Project `x0` onto `C₁ ∩ … ∩ Cₘ` via Dykstra's POCS algorithm.
///
/// # Arguments
///
/// * `projections` – Slice of callable projection operators.  Each closure
///   receives a slice of length `n` and must return a `Vec<f64>` of the same
///   length.
/// * `x0` – Starting point of length `n`.
/// * `max_iter` – Maximum number of outer iterations (one full cycle over all
///   sets counts as one iteration).
/// * `tol` – Convergence tolerance on `‖z_k − z_{k-1}‖₂`.  Must be positive.
///
/// # Errors
///
/// * [`CvxError::EmptyInput`] if `projections` or `x0` is empty.
/// * [`CvxError::InvalidParameter`] if `tol ≤ 0`.
/// * [`CvxError::DimensionMismatch`] if any projection returns a vector whose
///   length differs from `n`.
/// * Any error propagated from within a projection closure.
pub fn dykstra_pocs(
    projections: &[ProjFn<'_>],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<DykstraResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if projections.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if tol <= 0.0 {
        return Err(CvxError::InvalidParameter("tol must be positive".into()));
    }

    let n = x0.len();
    let m = projections.len();

    // ── Initialisation ────────────────────────────────────────────────────────
    let mut z: Vec<f64> = x0.to_vec();
    // Dykstra corrections: one vector of length n per set.
    let mut corrections: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; m];

    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut iter = 0_usize;

    // ── Main loop ─────────────────────────────────────────────────────────────
    for k in 0..max_iter {
        iter = k + 1;
        let z_prev = z.clone();

        for i in 0..m {
            // v = z + p_i  (shift by accumulated Dykstra correction)
            let v: Vec<f64> = z
                .iter()
                .zip(corrections[i].iter())
                .map(|(zj, pj)| zj + pj)
                .collect();

            // y = Proj_{C_i}(v)
            let y = projections[i](&v)?;

            // Dimension guard on very first projection call.
            if k == 0 && i == 0 && y.len() != n {
                return Err(CvxError::DimensionMismatch { a: y.len(), b: n });
            }

            // p_i ← v − y  (Dykstra incremental correction)
            for j in 0..n {
                corrections[i][j] = v[j] - y[j];
            }

            z = y;
        }

        // Convergence check: ‖z_k − z_{k-1}‖₂
        let diff: Vec<f64> = z
            .iter()
            .zip(z_prev.iter())
            .map(|(zj, pj)| zj - pj)
            .collect();
        residual = norm2(&diff);

        if residual < tol {
            converged = true;
            break;
        }
    }

    Ok(DykstraResult {
        x: z,
        iter,
        residual,
        converged,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper projection factories ───────────────────────────────────────────

    /// Build a halfspace projection onto `{x : <a, x> ≤ b}`.
    ///
    /// Uses the formula  Proj(x) = x − max(0, (<a,x>−b) / ‖a‖²) · a.
    fn make_halfspace_proj(a: Vec<f64>, b: f64) -> impl Fn(&[f64]) -> CvxResult<Vec<f64>> {
        move |x: &[f64]| {
            let dot: f64 = x.iter().zip(a.iter()).map(|(xi, ai)| xi * ai).sum();
            let excess = dot - b;
            if excess <= 0.0 {
                return Ok(x.to_vec());
            }
            let a_sq: f64 = a.iter().map(|ai| ai * ai).sum();
            let scale = excess / a_sq;
            Ok(x.iter()
                .zip(a.iter())
                .map(|(xi, ai)| xi - scale * ai)
                .collect())
        }
    }

    /// Build a projection onto the Euclidean ball `{x : ‖x‖₂ ≤ r}`.
    fn make_l2_ball_proj(r: f64) -> impl Fn(&[f64]) -> CvxResult<Vec<f64>> {
        move |x: &[f64]| {
            let norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
            if norm <= r {
                Ok(x.to_vec())
            } else {
                Ok(x.iter().map(|xi| xi * r / norm).collect())
            }
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// A single projection set: Dykstra reduces to the ordinary projector.
    #[test]
    fn test_single_set_is_ordinary_projection() {
        let proj = make_l2_ball_proj(1.0);
        let projs: Vec<ProjFn<'_>> = vec![&proj];
        let x0 = [3.0_f64, 0.0];
        let res = dykstra_pocs(&projs, &x0, 200, 1e-8).expect("dykstra ok");
        assert!(res.converged, "should converge");
        assert!(
            (res.x[0] - 1.0).abs() < 1e-6,
            "x[0] ≈ 1.0, got {}",
            res.x[0]
        );
        assert!(res.x[1].abs() < 1e-6, "x[1] ≈ 0.0, got {}", res.x[1]);
    }

    /// Intersection of two axis-aligned halfspaces: {x₁≤1} ∩ {x₂≤1}.
    #[test]
    fn test_two_halfspaces_simple() {
        let h1 = make_halfspace_proj(vec![1.0, 0.0], 1.0);
        let h2 = make_halfspace_proj(vec![0.0, 1.0], 1.0);
        let projs: Vec<ProjFn<'_>> = vec![&h1, &h2];
        let x0 = [2.0_f64, 2.0];
        let res = dykstra_pocs(&projs, &x0, 200, 1e-8).expect("dykstra ok");
        assert!(
            (res.x[0] - 1.0).abs() < 1e-4,
            "x[0] ≈ 1.0, got {}",
            res.x[0]
        );
        assert!(
            (res.x[1] - 1.0).abs() < 1e-4,
            "x[1] ≈ 1.0, got {}",
            res.x[1]
        );
    }

    /// L2-ball ∩ non-negative orthant; start at (-2,-2) → (0,0).
    #[test]
    fn test_l2_ball_and_box() {
        // Non-negative orthant via two halfspaces: {x₁≥0} = {-x₁≤0}, {x₂≥0} = {-x₂≤0}
        let ball = make_l2_ball_proj(1.0);
        let nn1 = make_halfspace_proj(vec![-1.0, 0.0], 0.0);
        let nn2 = make_halfspace_proj(vec![0.0, -1.0], 0.0);
        let projs: Vec<ProjFn<'_>> = vec![&ball, &nn1, &nn2];
        let x0 = [-2.0_f64, -2.0];
        let res = dykstra_pocs(&projs, &x0, 1000, 1e-8).expect("dykstra ok");
        // Result must lie in L2 ball
        assert!(
            norm2(&res.x) <= 1.0 + 1e-6,
            "must be in L2-ball, ‖x‖ = {}",
            norm2(&res.x)
        );
        // Result must be non-negative
        assert!(res.x[0] >= -1e-6, "x[0] must be ≥ 0, got {}", res.x[0]);
        assert!(res.x[1] >= -1e-6, "x[1] must be ≥ 0, got {}", res.x[1]);
    }

    /// Convergence flag is set for the two-halfspace problem.
    #[test]
    fn test_convergence_flag_set() {
        let h1 = make_halfspace_proj(vec![1.0, 0.0], 1.0);
        let h2 = make_halfspace_proj(vec![0.0, 1.0], 1.0);
        let projs: Vec<ProjFn<'_>> = vec![&h1, &h2];
        let res = dykstra_pocs(&projs, &[2.0, 2.0], 1000, 1e-8).expect("dykstra ok");
        assert!(res.converged, "should have converged");
    }

    /// Empty projections slice → `EmptyInput` error.
    #[test]
    fn test_empty_sets_error() {
        let projs: &[ProjFn<'_>] = &[];
        let result = dykstra_pocs(projs, &[1.0, 2.0], 100, 1e-6);
        assert!(
            matches!(result, Err(CvxError::EmptyInput)),
            "expected EmptyInput, got {result:?}"
        );
    }

    /// Empty `x0` → `EmptyInput` error.
    #[test]
    fn test_empty_x0_error() {
        let proj = make_l2_ball_proj(1.0);
        let projs: Vec<ProjFn<'_>> = vec![&proj];
        let result = dykstra_pocs(&projs, &[], 100, 1e-6);
        assert!(
            matches!(result, Err(CvxError::EmptyInput)),
            "expected EmptyInput, got {result:?}"
        );
    }

    /// Negative tolerance → `InvalidParameter` error.
    #[test]
    fn test_negative_tol_error() {
        let proj = make_l2_ball_proj(1.0);
        let projs: Vec<ProjFn<'_>> = vec![&proj];
        let result = dykstra_pocs(&projs, &[1.0, 2.0], 100, -1.0);
        assert!(
            matches!(result, Err(CvxError::InvalidParameter(_))),
            "expected InvalidParameter, got {result:?}"
        );
    }

    /// Three halfplanes: {x₁≤2} ∩ {x₂≤2} ∩ {x₁+x₂≤3}; start at (5,5).
    #[test]
    fn test_three_sets() {
        let h1 = make_halfspace_proj(vec![1.0, 0.0], 2.0);
        let h2 = make_halfspace_proj(vec![0.0, 1.0], 2.0);
        let h3 = make_halfspace_proj(vec![1.0, 1.0], 3.0);
        let projs: Vec<ProjFn<'_>> = vec![&h1, &h2, &h3];
        let res = dykstra_pocs(&projs, &[5.0, 5.0], 1000, 1e-8).expect("dykstra ok");
        assert!(res.x[0] <= 2.0 + 1e-4, "x[0] ≤ 2, got {}", res.x[0]);
        assert!(res.x[1] <= 2.0 + 1e-4, "x[1] ≤ 2, got {}", res.x[1]);
        assert!(
            res.x[0] + res.x[1] <= 3.0 + 1e-4,
            "x[0]+x[1] ≤ 3, got {}",
            res.x[0] + res.x[1]
        );
    }

    /// Single L2-ball; compare against analytically known projection of (3,4).
    ///
    /// The nearest point on the unit sphere to (3,4) is (3/5, 4/5).
    #[test]
    fn test_vs_direct_projection() {
        let proj = make_l2_ball_proj(1.0);
        let projs: Vec<ProjFn<'_>> = vec![&proj];
        let res = dykstra_pocs(&projs, &[3.0, 4.0], 1000, 1e-10).expect("dykstra ok");
        let expected = [3.0_f64 / 5.0, 4.0_f64 / 5.0];
        assert!(
            (res.x[0] - expected[0]).abs() < 1e-8,
            "x[0] = {} ≠ {}",
            res.x[0],
            expected[0]
        );
        assert!(
            (res.x[1] - expected[1]).abs() < 1e-8,
            "x[1] = {} ≠ {}",
            res.x[1],
            expected[1]
        );
    }
}
