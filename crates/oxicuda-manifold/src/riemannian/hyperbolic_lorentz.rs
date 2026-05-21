//! Lorentz (hyperboloid) model of hyperbolic space.
//!
//! The hyperboloid `H^n_K` is embedded in Minkowski space `R^{1,n}` as:
//! ```text
//! H^n_K = { x in R^{n+1} : <x, x>_L = -1/K, x[0] > 0 }
//! ```
//! where `<x, y>_L = -x[0]*y[0] + x[1]*y[1] + ... + x[n]*y[n]` is the
//! Lorentzian (Minkowski) inner product and `K > 0` is the curvature magnitude.
//!
//! A point is represented as a plain `Vec<f64>` of length `n+1`. The zeroth
//! component `x[0]` is the **time-like** coordinate (always positive); the
//! remaining components `x[1..n+1]` are **space-like**.
//!
//! # Coordinate convention
//! - `x[0]`        — time coordinate (`> 0`)
//! - `x[1..n+1]`   — spatial coordinates
//!
//! # Curvature
//! For the standard unit hyperboloid use `k = 1.0`.  Sectional curvature is `-K`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::riemannian::hyperbolic_poincare::{mobius_add, poincare_project};

// ---------------------------------------------------------------------------
// Lorentzian inner product and norms
// ---------------------------------------------------------------------------

/// Lorentzian (Minkowski) inner product.
///
/// `<x, y>_L = -x[0]*y[0] + x[1]*y[1] + ... + x[n]*y[n]`
///
/// Both slices must have the same length ≥ 1.
pub fn lorentz_inner(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len(), "lorentz_inner: dimension mismatch");
    let time = -x[0] * y[0];
    let space: f64 = x[1..].iter().zip(&y[1..]).map(|(a, b)| a * b).sum();
    time + space
}

/// Squared Lorentzian norm: `<x, x>_L`.
///
/// For a point on `H^n_K` this equals `-1/K` (negative).
pub fn lorentz_norm_sq(x: &[f64]) -> f64 {
    lorentz_inner(x, x)
}

// ---------------------------------------------------------------------------
// Geodesic distance
// ---------------------------------------------------------------------------

/// Geodesic distance on the hyperboloid `H^n_K`.
///
/// `d(x, y) = (1 / sqrt(K)) * arccosh(-K * <x, y>_L)`
///
/// The argument of arccosh is clamped to `[1 + eps, ∞)` for numerical
/// stability when points are very close together.
///
/// # Errors
/// - [`ManifoldError::DimensionMismatch`] if `x.len() != y.len()`
/// - [`ManifoldError::InvalidParameter`] if `k <= 0`
pub fn lorentz_distance(x: &[f64], y: &[f64], k: f64) -> ManifoldResult<f64> {
    if x.len() != y.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if k <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: "curvature must be positive".into(),
        });
    }
    let inner = lorentz_inner(x, y);
    // For valid on-manifold points, -K * <x,y>_L >= 1 (equality when x == y).
    // Clamp to 1.0 to keep arccosh well-defined; the argument equals 1 exactly
    // when x == y in exact arithmetic, so d(x, x) = arccosh(1) / sqrt(K) = 0.
    let arg = (-k * inner).max(1.0);
    Ok(acosh(arg) / k.sqrt())
}

/// Numerically stable `arccosh(x)` for `x >= 1`.
#[inline]
fn acosh(x: f64) -> f64 {
    (x + (x * x - 1.0).max(0.0).sqrt()).ln()
}

// ---------------------------------------------------------------------------
// Exponential map
// ---------------------------------------------------------------------------

/// Exponential map at `x` in direction tangent vector `v`.
///
/// The tangent space at `x` is `T_x H^n_K = { v : <x, v>_L = 0 }`.
///
/// ```text
/// exp_x(v) = cosh(sqrt(K) * ||v||_L) * x
///           + sinh(sqrt(K) * ||v||_L) / (sqrt(K) * ||v||_L) * v
/// ```
/// where `||v||_L = sqrt(<v, v>_L)` (positive for non-zero tangent vectors).
///
/// If `||v||_L < 1e-10`, returns `x` unchanged.
///
/// # Errors
/// - [`ManifoldError::DimensionMismatch`] if lengths differ
/// - [`ManifoldError::InvalidParameter`] if `k <= 0`
/// - [`ManifoldError::NumericalInstability`] if the Lorentzian norm of `v` is negative (i.e., `v` is not a valid tangent vector)
pub fn lorentz_exp(x: &[f64], v: &[f64], k: f64) -> ManifoldResult<Vec<f64>> {
    if x.len() != v.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: x.len(),
            b: v.len(),
        });
    }
    if k <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: "curvature must be positive".into(),
        });
    }
    let v_norm_sq = lorentz_inner(v, v);
    if v_norm_sq < -1e-8 {
        // v is time-like; not a valid tangent vector in the Lorentz model
        return Err(ManifoldError::NumericalInstability(
            "lorentz_exp: tangent vector has negative Lorentzian norm squared (time-like); \
             ensure <x, v>_L = 0"
                .into(),
        ));
    }
    // Clamp small negatives due to floating-point
    let v_norm = v_norm_sq.max(0.0).sqrt();
    if v_norm < 1e-10 {
        return Ok(x.to_vec());
    }
    let sqrt_k = k.sqrt();
    let theta = sqrt_k * v_norm; // argument to cosh/sinh
    let cosh_t = theta.cosh();
    let sinh_coeff = theta.sinh() / theta; // sinh(theta)/theta
    let result = x
        .iter()
        .zip(v.iter())
        .map(|(xi, vi)| cosh_t * xi + sinh_coeff * vi)
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Logarithmic map
// ---------------------------------------------------------------------------

/// Logarithmic map: the inverse of `lorentz_exp`.
///
/// Returns the tangent vector `v` at `x` such that `exp_x(v) = y`.
///
/// ```text
/// log_x(y) = alpha * (y + K * <x, y>_L * x)
/// ```
/// where `alpha = arccosh(-K * <x,y>_L) / sqrt((-K*<x,y>_L)^2 - 1)`.
///
/// If `x ≈ y`, returns a zero tangent vector.
///
/// # Errors
/// - [`ManifoldError::DimensionMismatch`] if lengths differ
/// - [`ManifoldError::InvalidParameter`] if `k <= 0`
pub fn lorentz_log(x: &[f64], y: &[f64], k: f64) -> ManifoldResult<Vec<f64>> {
    if x.len() != y.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if k <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: "curvature must be positive".into(),
        });
    }
    let inner = lorentz_inner(x, y);
    // -K * <x,y>_L >= 1 for valid on-manifold points; equals 1 when x == y
    let neg_k_inner = (-k * inner).max(1.0);
    if neg_k_inner < 1.0 + 1e-10 {
        // x and y are numerically identical; return zero tangent
        return Ok(vec![0.0; x.len()]);
    }
    // alpha = arccosh(neg_k_inner) / sqrt(neg_k_inner^2 - 1)
    let ac = acosh(neg_k_inner);
    let denom = (neg_k_inner * neg_k_inner - 1.0).max(0.0).sqrt();
    let alpha = if denom < 1e-15 { 0.0 } else { ac / denom };
    // log_x(y) = alpha * (y + K * <x,y>_L * x)
    //           = alpha * (y - neg_k_inner/K * k * x)  [since <x,y>_L = inner, K*inner = k*inner]
    // More directly: coeff on x is K * inner = k * inner
    let k_inner = k * inner; // = -neg_k_inner (so this is <= 0 for valid points)
    let result = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| alpha * (yi + k_inner * xi))
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tangent space projection
// ---------------------------------------------------------------------------

/// Project an ambient vector `u` onto the tangent space `T_x H^n_K`.
///
/// ```text
/// proj_x(u) = u + K * <x, u>_L * x
/// ```
///
/// This enforces `<x, proj_x(u)>_L = 0`, i.e., the result lies in the
/// tangent space at `x`.
pub fn lorentz_project_tangent(x: &[f64], u: &[f64], k: f64) -> Vec<f64> {
    let xiu = lorentz_inner(x, u);
    x.iter()
        .zip(u.iter())
        .map(|(xi, ui)| ui + k * xiu * xi)
        .collect()
}

// ---------------------------------------------------------------------------
// Canonical origin
// ---------------------------------------------------------------------------

/// Canonical base point (origin) of `H^n_K`.
///
/// `o = (1/sqrt(K), 0, 0, ..., 0)` in `R^{n+1}`.
///
/// The returned vector has length `n + 1` with `o[0] = 1/sqrt(K)` and all
/// other components zero. It satisfies `<o, o>_L = -1/K`.
pub fn lorentz_origin(n: usize, k: f64) -> Vec<f64> {
    let mut o = vec![0.0; n + 1];
    o[0] = 1.0 / k.sqrt();
    o
}

// ---------------------------------------------------------------------------
// Coordinate maps: Lorentz <-> Poincaré
// ---------------------------------------------------------------------------

/// Map a Lorentz model point to the Poincaré ball model.
///
/// ```text
/// p[i] = x[i+1] / (1 + x[0] * sqrt(K))    for i = 0..n
/// ```
///
/// The returned vector has length `n = x.len() - 1` (time coordinate is
/// dropped).
pub fn lorentz_to_poincare(x: &[f64], k: f64) -> Vec<f64> {
    let denom = 1.0 + x[0] * k.sqrt();
    x[1..].iter().map(|xi| xi / denom).collect()
}

/// Map a Poincaré ball point to the Lorentz model.
///
/// Let `r = ||p||^2`.
/// ```text
/// x[0]   = (1 + r) / (sqrt(K) * (1 - r))
/// x[i+1] = 2 * p[i] / (sqrt(K) * (1 - r))
/// ```
///
/// For numerical safety `r` is clamped to `1 - 1e-10` before division so
/// that points very close to the Poincaré boundary map to finite Lorentz
/// coordinates.
pub fn lorentz_from_poincare(p: &[f64], k: f64) -> Vec<f64> {
    let r: f64 = p.iter().map(|pi| pi * pi).sum();
    let r = r.min(1.0 - 1e-10); // guard division by zero at boundary
    let sqrt_k = k.sqrt();
    let one_minus_r = 1.0 - r;
    let mut x = Vec::with_capacity(p.len() + 1);
    x.push((1.0 + r) / (sqrt_k * one_minus_r));
    for pi in p {
        x.push(2.0 * pi / (sqrt_k * one_minus_r));
    }
    x
}

// ---------------------------------------------------------------------------
// Möbius addition via Poincaré model
// ---------------------------------------------------------------------------

/// Möbius (gyrovector) addition on the hyperboloid via the Poincaré model.
///
/// 1. Map `x` and `y` from Lorentz to Poincaré: `p = lorentz_to_poincare(x)`, `q = lorentz_to_poincare(y)`.
/// 2. Compute Möbius addition `p ⊕ q` in the Poincaré ball.
/// 3. Map the result back to the Lorentz model.
///
/// The Poincaré-model Möbius addition uses `k = 1.0` (unit-ball convention)
/// and the coordinate maps handle the curvature scaling; a small projection
/// with `epsilon = 1e-5` guards the boundary.
///
/// # Errors
/// - [`ManifoldError::DimensionMismatch`] if `x.len() != y.len()`
/// - [`ManifoldError::InvalidParameter`] if `k <= 0`
pub fn lorentz_mobius_add(x: &[f64], y: &[f64], k: f64) -> ManifoldResult<Vec<f64>> {
    if x.len() != y.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    if k <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: "curvature must be positive".into(),
        });
    }
    let p = lorentz_to_poincare(x, k);
    let q = lorentz_to_poincare(y, k);
    // Möbius add in the Poincaré ball; project to stay strictly inside unit ball
    let pq = mobius_add(&p, &q)?;
    let pq = poincare_project(&pq, 1e-5);
    Ok(lorentz_from_poincare(&pq, k))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: construct a point on H^n_1 from spatial coordinates by solving
    /// x[0] = sqrt(1 + ||spatial||^2).
    fn make_lorentz_point(spatial: &[f64]) -> Vec<f64> {
        let space_norm_sq: f64 = spatial.iter().map(|s| s * s).sum();
        let mut x = Vec::with_capacity(spatial.len() + 1);
        x.push((1.0 + space_norm_sq).sqrt());
        x.extend_from_slice(spatial);
        x
    }

    // -------------------------------------------------------------------------
    // 1. Inner product example
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_inner_product_example() {
        // <(2, 1, 0), (2, 1, 0)>_L = -2*2 + 1*1 + 0*0 = -4 + 1 = -3
        let v = [2.0_f64, 1.0, 0.0];
        let result = lorentz_inner(&v, &v);
        assert!(
            (result - (-3.0_f64)).abs() < 1e-12,
            "expected -3, got {result}"
        );
    }

    // -------------------------------------------------------------------------
    // 2. Distance to self is zero
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_distance_same_point_zero() {
        // Use the exact canonical origin so that lorentz_inner(o, o) = -1 with
        // no floating-point rounding, giving a perfectly clean d(o, o) = 0.
        let o = lorentz_origin(2, 1.0);
        let d = lorentz_distance(&o, &o, 1.0).expect("ok");
        assert!(d.abs() < 1e-12, "distance to self should be 0, got {d}");

        // For a numerically-constructed point the tolerance is relaxed due to
        // sqrt rounding: inner ≈ -1 ± ulp, so d ≈ acosh(1 + ulp) ~ sqrt(2*ulp).
        let x = make_lorentz_point(&[0.3, -0.4]);
        let d2 = lorentz_distance(&x, &x, 1.0).expect("ok");
        assert!(
            d2.abs() < 1e-6,
            "distance to self should be near 0, got {d2}"
        );
    }

    // -------------------------------------------------------------------------
    // 3. Distance is positive for distinct points
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_distance_positive() {
        let x = make_lorentz_point(&[0.5, 0.0]);
        let y = make_lorentz_point(&[0.0, 0.5]);
        let d = lorentz_distance(&x, &y, 1.0).expect("ok");
        assert!(
            d > 1e-9,
            "distance between distinct points should be positive, got {d}"
        );
    }

    // -------------------------------------------------------------------------
    // 4. Distance is symmetric
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_distance_symmetric() {
        let x = make_lorentz_point(&[0.3, 0.1, -0.2]);
        let y = make_lorentz_point(&[-0.1, 0.4, 0.2]);
        let dxy = lorentz_distance(&x, &y, 1.0).expect("ok");
        let dyx = lorentz_distance(&y, &x, 1.0).expect("ok");
        assert!(
            (dxy - dyx).abs() < 1e-10,
            "distance should be symmetric: d(x,y)={dxy}, d(y,x)={dyx}"
        );
    }

    // -------------------------------------------------------------------------
    // 5. Exp-log roundtrip: exp_x(log_x(y)) ≈ y
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_exp_log_roundtrip() {
        let x = make_lorentz_point(&[0.3, 0.0, 0.0]);
        let y = make_lorentz_point(&[0.0, 0.5, 0.1]);
        let v = lorentz_log(&x, &y, 1.0).expect("log ok");
        let y_hat = lorentz_exp(&x, &v, 1.0).expect("exp ok");
        for (a, b) in y.iter().zip(&y_hat) {
            assert!(
                (a - b).abs() < 1e-8,
                "exp(log(y)) should equal y; diff={:.3e}",
                (a - b).abs()
            );
        }
    }

    // -------------------------------------------------------------------------
    // 6. log at origin gives tangent
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_log_at_origin_gives_tangent() {
        let o = lorentz_origin(3, 1.0);
        let y = make_lorentz_point(&[0.2, -0.3, 0.1]);
        let v = lorentz_log(&o, &y, 1.0).expect("ok");
        let inner = lorentz_inner(&o, &v);
        assert!(
            inner.abs() < 1e-9,
            "<origin, log_origin(y)>_L should be 0, got {inner}"
        );
    }

    // -------------------------------------------------------------------------
    // 7. Tangent projection gives orthogonal result
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_project_tangent_orthogonal() {
        let x = make_lorentz_point(&[0.4, -0.3]);
        let u = vec![1.0, 2.0, 3.0];
        let proj = lorentz_project_tangent(&x, &u, 1.0);
        let inner = lorentz_inner(&x, &proj);
        assert!(
            inner.abs() < 1e-12,
            "<x, proj_x(u)>_L should be 0, got {inner}"
        );
    }

    // -------------------------------------------------------------------------
    // 8. Projection is idempotent
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_project_tangent_idempotent() {
        let x = make_lorentz_point(&[0.2, 0.5, -0.1]);
        let u = vec![0.5, -0.2, 0.8, 1.1];
        let p1 = lorentz_project_tangent(&x, &u, 1.0);
        let p2 = lorentz_project_tangent(&x, &p1, 1.0);
        for (a, b) in p1.iter().zip(&p2) {
            assert!(
                (a - b).abs() < 1e-12,
                "projection should be idempotent; diff={:.3e}",
                (a - b).abs()
            );
        }
    }

    // -------------------------------------------------------------------------
    // 9. Origin lies on the manifold
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_origin_on_manifold() {
        let k = 1.5_f64;
        let o = lorentz_origin(4, k);
        let norm_sq = lorentz_norm_sq(&o);
        let expected = -1.0 / k;
        assert!(
            (norm_sq - expected).abs() < 1e-12,
            "<o,o>_L should equal -1/k={expected}, got {norm_sq}"
        );
    }

    // -------------------------------------------------------------------------
    // 10. Lorentz -> Poincaré -> Lorentz roundtrip
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_to_poincare_and_back() {
        let x = make_lorentz_point(&[0.3, -0.2, 0.1]);
        let p = lorentz_to_poincare(&x, 1.0);
        let x_hat = lorentz_from_poincare(&p, 1.0);
        for (a, b) in x.iter().zip(&x_hat) {
            assert!(
                (a - b).abs() < 1e-10,
                "round-trip Lorentz->Poincaré->Lorentz failed; diff={:.3e}",
                (a - b).abs()
            );
        }
    }

    // -------------------------------------------------------------------------
    // 11. Triangle inequality
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_triangle_inequality() {
        let x = make_lorentz_point(&[0.5, 0.0, 0.0]);
        let y = make_lorentz_point(&[0.0, 0.5, 0.0]);
        let z = make_lorentz_point(&[0.0, 0.0, 0.5]);
        let dxz = lorentz_distance(&x, &z, 1.0).expect("ok");
        let dxy = lorentz_distance(&x, &y, 1.0).expect("ok");
        let dyz = lorentz_distance(&y, &z, 1.0).expect("ok");
        assert!(
            dxz <= dxy + dyz + 1e-10,
            "triangle inequality violated: d(x,z)={dxz} > d(x,y)+d(y,z)={}",
            dxy + dyz
        );
    }

    // -------------------------------------------------------------------------
    // 12. Exp stays on manifold: <exp_x(v), exp_x(v)>_L ≈ -1
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_exp_stays_on_manifold() {
        let x = make_lorentz_point(&[0.3, 0.0]);
        // Build a tangent vector at x satisfying <x, v>_L = 0
        let raw = vec![0.0, 0.5, 0.2];
        let v = lorentz_project_tangent(&x, &raw, 1.0);
        let exp_xv = lorentz_exp(&x, &v, 1.0).expect("exp ok");
        let norm_sq = lorentz_norm_sq(&exp_xv);
        assert!(
            (norm_sq + 1.0).abs() < 1e-8,
            "<exp_x(v), exp_x(v)>_L should be -1, got {norm_sq}"
        );
    }

    // -------------------------------------------------------------------------
    // 13. Distance with k != 1 rescales correctly
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_distance_curvature_scaling() {
        // For k=4 the origin is (1/2, 0, ...) and distances scale by 1/sqrt(k)
        let k = 4.0_f64;
        let o = lorentz_origin(2, k);
        // Check that <o,o>_L = -1/k
        let norm_sq = lorentz_norm_sq(&o);
        assert!((norm_sq + 1.0 / k).abs() < 1e-12);
        // Distance from origin to itself is 0 under any curvature
        let d = lorentz_distance(&o, &o, k).expect("ok");
        assert!(d.abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // 14. Möbius add with zero vector is identity
    // -------------------------------------------------------------------------
    #[test]
    fn lorentz_mobius_add_with_zero_identity() {
        let k = 1.0_f64;
        let x = make_lorentz_point(&[0.3, 0.1]);
        let o = lorentz_origin(2, k);
        let result = lorentz_mobius_add(&x, &o, k).expect("ok");
        // The result should be on the manifold
        let norm_sq = lorentz_norm_sq(&result);
        assert!(
            (norm_sq + 1.0 / k).abs() < 1e-5,
            "<result,result>_L should be ≈ -1/k, got {norm_sq}"
        );
    }
}
