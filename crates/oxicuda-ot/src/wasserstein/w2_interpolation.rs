//! McCann displacement interpolation in Wasserstein-2 space (McCann 1997).
//!
//! Given two probability measures `μ₀, μ₁` and the optimal transport map `T`
//! pushing `μ₀` onto `μ₁`, the **displacement interpolant** is the curve of
//! measures
//!
//! ```text
//! μ_t = ((1 − t)·Id + t·T)_# μ₀ ,   t ∈ [0, 1] ,
//! ```
//!
//! i.e. the push-forward of `μ₀` under the convex combination of the identity
//! map and the transport map. This is the constant-speed geodesic joining `μ₀`
//! to `μ₁` in the Wasserstein-2 metric: `W₂(μ_s, μ_t) = |t − s|·W₂(μ₀, μ₁)`,
//! and mass travels along straight lines at uniform speed (McCann's
//! interpolation), unlike the linear interpolation `(1−t)μ₀ + t μ₁` which
//! merely fades one measure into the other.
//!
//! Two regimes are provided:
//!
//! * **1D exact** — the monotone (quantile) coupling is the unique optimal map
//!   in one dimension, so the interpolant follows from sorted support points
//!   matched in increasing order. Implemented for equal-weight discrete
//!   measures of equal cardinality.
//! * **Discrete via plan** — for arbitrary support sets in any dimension, the
//!   barycentric displacement of a transport plan `P` (e.g. from Sinkhorn or
//!   the network simplex) gives the interpolated support
//!   `z_ij(t) = (1−t)·x_i + t·y_j` weighted by `P_ij`.

use crate::error::{OtError, OtResult};

/// A discrete measure produced by displacement interpolation: weighted support
/// points in `ℝ^{dim}` stored row-major.
#[derive(Debug, Clone)]
pub struct InterpolatedMeasure {
    /// Interpolation parameter `t ∈ [0, 1]` that produced this measure.
    pub t: f32,
    /// Support coordinates, length `n_support · dim`, row-major.
    pub support: Vec<f32>,
    /// Mass at each support point, length `n_support`.
    pub weights: Vec<f32>,
    /// Number of support points.
    pub n_support: usize,
    /// Ambient dimension.
    pub dim: usize,
}

/// Linearly interpolate two equal-length, equal-weight 1D point clouds along
/// the McCann geodesic at parameter `t`.
///
/// The supports `x` and `y` are sorted ascending and matched by rank (the
/// monotone optimal 1D coupling). Returns the interpolated support
/// `z_k = (1−t)·x_(k) + t·y_(k)` carrying uniform mass `1/n`.
pub fn displacement_interpolate_1d(x: &[f32], y: &[f32], t: f32) -> OtResult<Vec<f32>> {
    if x.is_empty() || y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if x.len() != y.len() {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: y.len(),
        });
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(OtError::Internal {
            msg: format!("interpolation parameter t={t} must lie in [0, 1]"),
        });
    }
    for &v in x.iter().chain(y.iter()) {
        if !v.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite support coordinate".to_string(),
            });
        }
    }
    let mut sx = x.to_vec();
    let mut sy = y.to_vec();
    sx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut z = vec![0.0_f32; x.len()];
    for k in 0..x.len() {
        z[k] = (1.0 - t) * sx[k] + t * sy[k];
    }
    Ok(z)
}

/// Validate a plan-based displacement interpolation call and return `(m, n)`.
fn validate_plan(
    plan: &[f32],
    source_x: &[f32],
    target_y: &[f32],
    m: usize,
    n: usize,
    dim: usize,
    t: f32,
) -> OtResult<()> {
    if m == 0 || n == 0 || dim == 0 {
        return Err(OtError::EmptyInput);
    }
    if plan.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: plan.len(),
            b_len: m * n,
        });
    }
    if source_x.len() != m * dim {
        return Err(OtError::MarginalMismatch {
            m,
            n: dim,
            a_len: source_x.len(),
            b_len: m * dim,
        });
    }
    if target_y.len() != n * dim {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: dim,
            a_len: target_y.len(),
            b_len: n * dim,
        });
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(OtError::Internal {
            msg: format!("interpolation parameter t={t} must lie in [0, 1]"),
        });
    }
    for &p in plan {
        if !p.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite plan entry".to_string(),
            });
        }
        if p < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

/// Displacement interpolation of a discrete OT plan in arbitrary dimension.
///
/// Each plan cell `(i, j)` carrying mass `P_ij > 0` contributes one support
/// point `z_ij(t) = (1−t)·x_i + t·y_j` to the interpolated measure. Cells with
/// (near) zero mass are dropped. The output weights are renormalised to sum to
/// the total transported mass so the result is a valid (sub-)probability
/// measure consistent with `P`.
///
/// `source_x` is `m · dim` row-major, `target_y` is `n · dim` row-major, and
/// `plan` is `m · n` row-major.
pub fn displacement_interpolate_plan(
    plan: &[f32],
    source_x: &[f32],
    target_y: &[f32],
    m: usize,
    n: usize,
    dim: usize,
    t: f32,
) -> OtResult<InterpolatedMeasure> {
    validate_plan(plan, source_x, target_y, m, n, dim, t)?;

    // Threshold below which a plan cell is considered empty.
    const TINY: f32 = 1e-12;
    let mut support: Vec<f32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut total = 0.0_f32;

    for i in 0..m {
        let xo = i * dim;
        let ro = i * n;
        for j in 0..n {
            let p = plan[ro + j];
            if p <= TINY {
                continue;
            }
            let yo = j * dim;
            for d in 0..dim {
                let z = (1.0 - t) * source_x[xo + d] + t * target_y[yo + d];
                support.push(z);
            }
            weights.push(p);
            total += p;
        }
    }

    if weights.is_empty() {
        return Err(OtError::Internal {
            msg: "transport plan has no mass".to_string(),
        });
    }

    // Renormalise so weights sum to one (the plan's total transported mass).
    let inv = 1.0 / total;
    for w in weights.iter_mut() {
        *w *= inv;
    }

    let n_support = weights.len();
    Ok(InterpolatedMeasure {
        t,
        support,
        weights,
        n_support,
        dim,
    })
}

/// Sample a full geodesic path of `n_steps + 1` interpolated measures at evenly
/// spaced parameters `t_k = k / n_steps`, `k = 0 … n_steps`.
///
/// Endpoints recover (a copy of) the source measure at `t = 0` and the target
/// measure at `t = 1` (up to the plan's transported mass).
pub fn displacement_path_plan(
    plan: &[f32],
    source_x: &[f32],
    target_y: &[f32],
    m: usize,
    n: usize,
    dim: usize,
    n_steps: usize,
) -> OtResult<Vec<InterpolatedMeasure>> {
    if n_steps == 0 {
        return Err(OtError::BadCount { got: n_steps });
    }
    let mut path = Vec::with_capacity(n_steps + 1);
    for k in 0..=n_steps {
        let t = k as f32 / n_steps as f32;
        path.push(displacement_interpolate_plan(
            plan, source_x, target_y, m, n, dim, t,
        )?);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn interp_1d_midpoint_is_average() {
        let x = vec![0.0_f32, 2.0, 4.0];
        let y = vec![1.0_f32, 3.0, 5.0];
        let z = displacement_interpolate_1d(&x, &y, 0.5).expect("ok");
        assert_eq!(z.len(), 3);
        assert!(approx(z[0], 0.5, 1e-6));
        assert!(approx(z[1], 2.5, 1e-6));
        assert!(approx(z[2], 4.5, 1e-6));
    }

    #[test]
    fn interp_1d_endpoints_recover_inputs() {
        let x = vec![3.0_f32, 1.0, 2.0];
        let y = vec![9.0_f32, 7.0, 8.0];
        let z0 = displacement_interpolate_1d(&x, &y, 0.0).expect("ok");
        let z1 = displacement_interpolate_1d(&x, &y, 1.0).expect("ok");
        // Sorted endpoints.
        assert!(approx(z0[0], 1.0, 1e-6) && approx(z0[2], 3.0, 1e-6));
        assert!(approx(z1[0], 7.0, 1e-6) && approx(z1[2], 9.0, 1e-6));
    }

    #[test]
    fn interp_1d_sorts_before_matching() {
        // Reverse-ordered y must be matched in increasing order.
        let x = vec![0.0_f32, 1.0];
        let y = vec![10.0_f32, 0.0];
        let z = displacement_interpolate_1d(&x, &y, 0.5).expect("ok");
        // sorted: x=(0,1), y=(0,10) → z=(0, 5.5)
        assert!(approx(z[0], 0.0, 1e-6));
        assert!(approx(z[1], 5.5, 1e-6));
    }

    #[test]
    fn interp_1d_length_mismatch_rejected() {
        let res = displacement_interpolate_1d(&[1.0_f32], &[1.0_f32, 2.0], 0.5);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn interp_1d_out_of_range_t_rejected() {
        let res = displacement_interpolate_1d(&[1.0_f32], &[2.0_f32], 1.5);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn interp_1d_empty_rejected() {
        let res = displacement_interpolate_1d(&[], &[], 0.5);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn plan_midpoint_diagonal_two_diracs() {
        // Two source points mapped identically to two targets via a diagonal
        // plan: midpoints lie halfway in 2D.
        let m = 2;
        let n = 2;
        let dim = 2;
        let x = vec![0.0_f32, 0.0, 1.0, 1.0];
        let y = vec![2.0_f32, 0.0, 3.0, 1.0];
        let plan = vec![0.5_f32, 0.0, 0.0, 0.5];
        let im = displacement_interpolate_plan(&plan, &x, &y, m, n, dim, 0.5).expect("ok");
        assert_eq!(im.n_support, 2);
        // (0,0)->(2,0): midpoint (1,0); (1,1)->(3,1): midpoint (2,1)
        assert!(approx(im.support[0], 1.0, 1e-6) && approx(im.support[1], 0.0, 1e-6));
        assert!(approx(im.support[2], 2.0, 1e-6) && approx(im.support[3], 1.0, 1e-6));
        let wsum: f32 = im.weights.iter().sum();
        assert!(approx(wsum, 1.0, 1e-6));
    }

    #[test]
    fn plan_endpoints_recover_supports() {
        let m = 2;
        let n = 2;
        let dim = 1;
        let x = vec![0.0_f32, 5.0];
        let y = vec![10.0_f32, 20.0];
        let plan = vec![0.5_f32, 0.0, 0.0, 0.5];
        let t0 = displacement_interpolate_plan(&plan, &x, &y, m, n, dim, 0.0).expect("ok");
        let t1 = displacement_interpolate_plan(&plan, &x, &y, m, n, dim, 1.0).expect("ok");
        // t=0 → source supports {0,5}; t=1 → target supports {10,20}.
        assert!(approx(t0.support[0], 0.0, 1e-6) && approx(t0.support[1], 5.0, 1e-6));
        assert!(approx(t1.support[0], 10.0, 1e-6) && approx(t1.support[1], 20.0, 1e-6));
    }

    #[test]
    fn plan_drops_zero_cells() {
        let m = 2;
        let n = 2;
        let dim = 1;
        let x = vec![0.0_f32, 1.0];
        let y = vec![2.0_f32, 3.0];
        // Only one non-zero cell.
        let plan = vec![1.0_f32, 0.0, 0.0, 0.0];
        let im = displacement_interpolate_plan(&plan, &x, &y, m, n, dim, 0.5).expect("ok");
        assert_eq!(im.n_support, 1);
        assert!(approx(im.weights[0], 1.0, 1e-6));
    }

    #[test]
    fn plan_negative_weight_rejected() {
        let plan = vec![0.5_f32, -0.5, 0.0, 1.0];
        let res =
            displacement_interpolate_plan(&plan, &[0.0_f32, 1.0], &[2.0_f32, 3.0], 2, 2, 1, 0.5);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn plan_shape_mismatch_rejected() {
        let res = displacement_interpolate_plan(
            &[1.0_f32; 3],
            &[0.0_f32; 2],
            &[1.0_f32; 2],
            2,
            2,
            1,
            0.5,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn path_has_correct_length_and_endpoints() {
        let m = 1;
        let n = 1;
        let dim = 1;
        let x = vec![0.0_f32];
        let y = vec![4.0_f32];
        let plan = vec![1.0_f32];
        let path = displacement_path_plan(&plan, &x, &y, m, n, dim, 4).expect("ok");
        assert_eq!(path.len(), 5);
        // Uniform-speed: support at step k is k.
        for (k, im) in path.iter().enumerate() {
            assert!(approx(im.support[0], k as f32, 1e-6), "step {k}");
        }
    }

    #[test]
    fn path_zero_steps_rejected() {
        let res = displacement_path_plan(&[1.0_f32], &[0.0_f32], &[1.0_f32], 1, 1, 1, 0);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }
}
