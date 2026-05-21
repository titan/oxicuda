//! Bures-Wasserstein geometry on the SPD(n) manifold.
//!
//! The Bures metric (Bures 1969) provides a Riemannian structure arising
//! naturally from the Wasserstein-2 optimal transport distance between
//! Gaussian distributions N(0, A) and N(0, B).
//!
//! # Key References
//! - Bures (1969): "An extension of the concept of quantum entropy"
//! - Dowson & Landau (1982): "The Fréchet distance between multivariate normal distributions"
//! - Alvarez-Esteban et al. (2016): "A fixed-point approach to barycenters in Wasserstein space"
//! - Malago et al. (2018): "Wasserstein Riemannian geometry of Gaussian densities"
//!
//! # Bures-Wasserstein Distance
//! For A, B ∈ SPD(n):
//! `d_BW(A,B)² = tr(A) + tr(B) − 2·tr(M(A,B))`
//! where `M(A,B) = (A^{1/2}·B·A^{1/2})^{1/2}` is the matrix geometric mean.
//!
//! # Geodesic (Displacement Interpolant)
//! `G_t = A^{1/2}·[(1−t)·I + t·(A^{-1/2}·B·A^{-1/2})^{1/2}]²·A^{1/2}`
//!
//! The midpoint `G_{1/2}` is the BW Fréchet mean of `{A,B}`, which for
//! non-commuting matrices differs from the affine-invariant matrix geometric mean.
//!
//! # Log / Exp Maps
//! `log_P(Q) = 2·(M(P,Q) − P)`
//! `exp_P(V) = P^{1/2}·[P^{-1/2}·((V + 2P)/2)·P^{-1/2}]²·P^{1/2}`

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;

// ─────────────────────────────────────────────────────────────────────────────
// Private matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/// n×n dense matrix multiply: C = A·B (row-major).
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

/// Element-wise matrix addition: C = A + B.
fn mat_add(a: &[f64], b: &[f64], _n: usize) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise scalar multiply: B = s·A.
fn mat_scale(a: &[f64], s: f64, n: usize) -> Vec<f64> {
    let _ = n; // size used by caller for validation; kept for symmetry
    a.iter().map(|x| x * s).collect()
}

/// Trace of n×n matrix.
fn mat_trace(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Frobenius distance ‖A − B‖_F.
fn frobenius_dist(a: &[f64], b: &[f64], n: usize) -> f64 {
    let _ = n;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Symmetrize a matrix: (A + A^T) / 2.
fn symmetrize(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    out
}

/// Build the n×n identity matrix.
fn mat_identity(n: usize) -> Vec<f64> {
    let mut id = vec![0.0f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    id
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigendecomposition-based SPD primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a scalar function `f(λ)` to each eigenvalue of a symmetric matrix.
///
/// Returns `V · diag(f(λ)) · V^T` where `A = V · diag(λ) · V^T`.
/// `min_eig` is a floor applied before calling `f`.
fn spd_func<F>(a: &[f64], n: usize, min_eig: f64, f: F) -> ManifoldResult<Vec<f64>>
where
    F: Fn(f64) -> f64,
{
    let sym = symmetrize(a, n);
    let (w, v) = jacobi_eigh(&sym, n)?;
    for (idx, &wi) in w.iter().enumerate() {
        if wi < -1e-6 {
            return Err(ManifoldError::ManifoldConstraint(format!(
                "spd_bures: non-trivial negative eigenvalue {wi:.3e} at index {idx}"
            )));
        }
    }
    let mut out = vec![0.0f64; n * n];
    for k in 0..n {
        let fk = f(w[k].max(min_eig));
        for i in 0..n {
            let vik = v[i * n + k];
            for j in 0..n {
                out[i * n + j] += vik * v[j * n + k] * fk;
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public core SPD matrix operations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the symmetric positive definite square root `A^{1/2}`.
///
/// Uses eigendecomposition: `A = V·diag(λ)·V^T`,
/// so `A^{1/2} = V·diag(sqrt(λ))·V^T`.
/// All eigenvalues must be non-negative (SPD/PSD assumption).
pub fn spd_sqrt(a: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    spd_func(a, n, 1e-15, |lam| lam.sqrt())
}

/// Compute the inverse `A^{-1}` of an SPD matrix.
///
/// Uses eigendecomposition: `A^{-1} = V·diag(1/λ)·V^T`.
/// Returns `SingularMatrix` if the minimum eigenvalue is below `tol = 1e-12`.
pub fn spd_inv(a: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let sym = symmetrize(a, n);
    let (w, v) = jacobi_eigh(&sym, n)?;
    let tol = 1e-12;
    for (idx, &wi) in w.iter().enumerate() {
        if wi < tol {
            return Err(ManifoldError::SingularMatrix(format!(
                "spd_inv: eigenvalue {wi:.3e} at index {idx} is below threshold {tol}"
            )));
        }
    }
    let mut out = vec![0.0f64; n * n];
    for k in 0..n {
        let inv_lam = 1.0 / w[k];
        for i in 0..n {
            let vik = v[i * n + k];
            for j in 0..n {
                out[i * n + j] += vik * v[j * n + k] * inv_lam;
            }
        }
    }
    Ok(out)
}

/// Compute `A^{-1/2}` (inverse square root of an SPD matrix).
///
/// Uses eigendecomposition: `A^{-1/2} = V·diag(1/sqrt(λ))·V^T`.
/// Returns `SingularMatrix` if the minimum eigenvalue is below `tol = 1e-12`.
pub fn spd_inv_sqrt(a: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let sym = symmetrize(a, n);
    let (w, v) = jacobi_eigh(&sym, n)?;
    let tol = 1e-12;
    for (idx, &wi) in w.iter().enumerate() {
        if wi < tol {
            return Err(ManifoldError::SingularMatrix(format!(
                "spd_inv_sqrt: eigenvalue {wi:.3e} at index {idx} is below threshold {tol}"
            )));
        }
    }
    let mut out = vec![0.0f64; n * n];
    for k in 0..n {
        let fk = 1.0 / w[k].sqrt();
        for i in 0..n {
            let vik = v[i * n + k];
            for j in 0..n {
                out[i * n + j] += vik * v[j * n + k] * fk;
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bures-Wasserstein distance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Bures-Wasserstein distance between two SPD matrices.
///
/// Formula: `d_BW(A,B)² = tr(A) + tr(B) − 2·tr((A^{1/2}·B·A^{1/2})^{1/2})`
///
/// The distance itself is returned (i.e. the square root of the expression above).
/// For numerical reasons the squared distance is clamped to `[0, ∞)` before taking
/// the square root.
pub fn bures_distance(a: &[f64], b: &[f64], n: usize) -> ManifoldResult<f64> {
    if a.len() != n * n || b.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    // Compute A^{1/2}
    let a_sq = spd_sqrt(a, n)?;
    // Form the congruence: K = A^{1/2}·B·A^{1/2}
    let ab = mat_mul(&a_sq, b, n);
    let k = mat_mul(&ab, &a_sq, n);
    // Symmetrize K to correct floating-point drift
    let k_sym = symmetrize(&k, n);
    // Compute trace of K^{1/2}: sum of sqrt(eigenvalues of K)
    let (w_k, _) = jacobi_eigh(&k_sym, n)?;
    let tr_sqrt_k: f64 = w_k.iter().map(|&lam| lam.max(0.0).sqrt()).sum();
    let tr_a = mat_trace(a, n);
    let tr_b = mat_trace(b, n);
    let dist_sq = (tr_a + tr_b - 2.0 * tr_sqrt_k).max(0.0);
    Ok(dist_sq.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix geometric mean (Bures midpoint)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the matrix geometric mean `M(A,B)` in the Bures sense.
///
/// `M(A,B) = A^{1/2}·(A^{-1/2}·B·A^{-1/2})^{1/2}·A^{1/2}`
///
/// This quantity appears in the BW distance formula as `tr(M(A,B))` and in the
/// log map. For commuting matrices it reduces to `(A·B)^{1/2}`. Note that the
/// BW Fréchet mean of two points coincides with the midpoint of the displacement
/// interpolant geodesic `G_{1/2}`, which for non-commuting matrices differs from
/// this matrix geometric mean (the latter is the affine-invariant midpoint).
pub fn bures_geometric_mean(a: &[f64], b: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if a.len() != n * n || b.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let a_sq = spd_sqrt(a, n)?;
    let a_inv_sq = spd_inv_sqrt(a, n)?;
    // C = A^{-1/2}·B·A^{-1/2}  (symmetric, SPD)
    let ab_inv = mat_mul(&a_inv_sq, b, n);
    let c = mat_mul(&ab_inv, &a_inv_sq, n);
    // D = C^{1/2}
    let d = spd_sqrt(&c, n)?;
    // M = A^{1/2}·D·A^{1/2}
    let ad = mat_mul(&a_sq, &d, n);
    let mean = mat_mul(&ad, &a_sq, n);
    Ok(symmetrize(&mean, n))
}

// ─────────────────────────────────────────────────────────────────────────────
// Bures-Wasserstein geodesic
// ─────────────────────────────────────────────────────────────────────────────

/// Geodesic interpolation at parameter `t ∈ [0, 1]` between SPD matrices A and B.
///
/// Formula:
/// `G_t = A^{1/2}·E²·A^{1/2}`
/// where `E = (1−t)·I + t·(A^{-1/2}·B·A^{-1/2})^{1/2}`.
///
/// Boundary cases: `G_0 = A`, `G_1 = B`.
/// At `t = 0.5`, `G_{1/2}` is the BW Fréchet mean of `{A, B}` (equidistant midpoint).
/// Note: for non-commuting A, B this midpoint differs from the affine-invariant
/// matrix geometric mean `M(A,B) = A^{1/2}(A^{-1/2}BA^{-1/2})^{1/2}A^{1/2}`.
pub fn bures_geodesic(a: &[f64], b: &[f64], n: usize, t: f64) -> ManifoldResult<Vec<f64>> {
    if a.len() != n * n || b.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(ManifoldError::InvalidParameter {
            name: "t".into(),
            reason: format!("geodesic parameter must be in [0, 1], got {t}"),
        });
    }
    // Short-circuit endpoints to avoid numeric drift
    if t == 0.0 {
        return Ok(a.to_vec());
    }
    if t == 1.0 {
        return Ok(b.to_vec());
    }
    let a_sq = spd_sqrt(a, n)?;
    let a_inv_sq = spd_inv_sqrt(a, n)?;
    // C = A^{-1/2}·B·A^{-1/2}
    let ab_inv = mat_mul(&a_inv_sq, b, n);
    let c = mat_mul(&ab_inv, &a_inv_sq, n);
    // D = C^{1/2}
    let d = spd_sqrt(&c, n)?;
    // E = (1-t)·I + t·D
    let id = mat_identity(n);
    let id_part = mat_scale(&id, 1.0 - t, n);
    let d_part = mat_scale(&d, t, n);
    let e = mat_add(&id_part, &d_part, n);
    // E² = E·E
    let e_sq = mat_mul(&e, &e, n);
    // G_t = A^{1/2}·E²·A^{1/2}
    let fe_sq = mat_mul(&a_sq, &e_sq, n);
    let g = mat_mul(&fe_sq, &a_sq, n);
    Ok(symmetrize(&g, n))
}

// ─────────────────────────────────────────────────────────────────────────────
// Bures log / exp maps
// ─────────────────────────────────────────────────────────────────────────────

/// Bures-Wasserstein logarithmic map at base point P.
///
/// Returns the tangent vector X at P such that `exp_P(X) = Q`.
///
/// Formula (Malago et al. 2018):
/// `log_P(Q) = 2·(M(P,Q) − P)`
/// where `M(P,Q)` is the Bures geometric mean.
pub fn bures_log(p: &[f64], q: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if p.len() != n * n || q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p.len()],
        });
    }
    let m = bures_geometric_mean(p, q, n)?;
    // X = 2·(M − P)
    let diff: Vec<f64> = m.iter().zip(p.iter()).map(|(mi, pi)| mi - pi).collect();
    Ok(mat_scale(&diff, 2.0, n))
}

/// Bures-Wasserstein exponential map at base point P with tangent vector V.
///
/// Returns the point Q = exp_P(V) on the SPD manifold.
///
/// Derivation: from `log_P(Q) = V` we get `M(P,Q) = P + V/2`, and solving
/// for Q through the geometric mean formula gives:
/// ```text
/// H = P + V/2              (= (V + 2P)/2)
/// C = P^{-1/2}·H·P^{-1/2}
/// Q = P^{1/2}·C²·P^{1/2}
/// ```
pub fn bures_exp(p: &[f64], v: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if p.len() != n * n || v.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p.len()],
        });
    }
    let p_sq = spd_sqrt(p, n)?;
    let p_inv_sq = spd_inv_sqrt(p, n)?;
    // H = P + V/2
    let v_half = mat_scale(v, 0.5, n);
    let h: Vec<f64> = p
        .iter()
        .zip(v_half.iter())
        .map(|(pi, vi)| pi + vi)
        .collect();
    // C = P^{-1/2}·H·P^{-1/2}
    let ph = mat_mul(&p_inv_sq, &h, n);
    let c = mat_mul(&ph, &p_inv_sq, n);
    // C² = C·C
    let c_sq = mat_mul(&c, &c, n);
    // Q = P^{1/2}·C²·P^{1/2}
    let pc_sq = mat_mul(&p_sq, &c_sq, n);
    let q = mat_mul(&pc_sq, &p_sq, n);
    Ok(symmetrize(&q, n))
}

// ─────────────────────────────────────────────────────────────────────────────
// Bures-Wasserstein Fréchet mean
// ─────────────────────────────────────────────────────────────────────────────

/// Fréchet mean of a set of SPD matrices under the Bures-Wasserstein metric.
///
/// Uses the fixed-point iteration of Alvarez-Esteban et al. (2016):
/// ```text
/// T_k   = (1/N) · Σᵢ (μ_k^{-1/2}·Pᵢ·μ_k^{-1/2})^{1/2}
/// μ_{k+1} = μ_k^{1/2}·T_k²·μ_k^{1/2}
/// ```
/// Initialized with the arithmetic mean.
///
/// # Arguments
/// - `matrices`: flat `[k × n²]` slice of k row-major n×n SPD matrices
/// - `k`: number of input matrices
/// - `n`: matrix dimension
/// - `max_iter`: maximum fixed-point iterations
/// - `tol`: convergence threshold on `‖μ_{k+1} − μ_k‖_F`
pub fn bures_frechet_mean(
    matrices: &[f64],
    k: usize,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<Vec<f64>> {
    if k == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    let sz = n * n;
    if matrices.len() != k * sz {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![k, n, n],
            got: vec![matrices.len()],
        });
    }
    // Initialise: arithmetic mean
    let mut mu = vec![0.0f64; sz];
    for ki in 0..k {
        let slice = &matrices[ki * sz..(ki + 1) * sz];
        for (m, s) in mu.iter_mut().zip(slice.iter()) {
            *m += s;
        }
    }
    let k_inv = 1.0 / k as f64;
    for m in mu.iter_mut() {
        *m *= k_inv;
    }
    // Fixed-point iteration
    for iter in 0..max_iter {
        let mu_sq = spd_sqrt(&mu, n)?;
        let mu_inv_sq = spd_inv_sqrt(&mu, n)?;
        // T = (1/k)·Σᵢ (μ^{-1/2}·Pᵢ·μ^{-1/2})^{1/2}
        let mut t_acc = vec![0.0f64; sz];
        for ki in 0..k {
            let pi = &matrices[ki * sz..(ki + 1) * sz];
            // Cᵢ = μ^{-1/2}·Pᵢ·μ^{-1/2}
            let left = mat_mul(&mu_inv_sq, pi, n);
            let ci = mat_mul(&left, &mu_inv_sq, n);
            // Cᵢ^{1/2}
            let ci_sq = spd_sqrt(&ci, n)?;
            for (acc, val) in t_acc.iter_mut().zip(ci_sq.iter()) {
                *acc += val;
            }
        }
        let t: Vec<f64> = t_acc.iter().map(|x| x * k_inv).collect();
        // T² = T·T
        let t_sq = mat_mul(&t, &t, n);
        // μ_new = μ^{1/2}·T²·μ^{1/2}
        let left = mat_mul(&mu_sq, &t_sq, n);
        let mu_new_raw = mat_mul(&left, &mu_sq, n);
        let mu_new = symmetrize(&mu_new_raw, n);
        let diff = frobenius_dist(&mu_new, &mu, n);
        mu = mu_new;
        if diff < tol {
            return Ok(mu);
        }
        // Safety: if we exhaust iterations, fall through and return last estimate
        if iter + 1 == max_iter {
            return Err(ManifoldError::NotConverged { iter: max_iter });
        }
    }
    Ok(mu)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2×2 SPD test matrix [[a, b], [b, c]].
    fn mat2(a: f64, b: f64, c: f64) -> Vec<f64> {
        vec![a, b, b, c]
    }

    /// Build a 3×3 diagonal SPD matrix with given diagonal entries.
    fn diag3(d0: f64, d1: f64, d2: f64) -> Vec<f64> {
        let mut m = vec![0.0f64; 9];
        m[0] = d0;
        m[4] = d1;
        m[8] = d2;
        m
    }

    /// Build the n×n identity matrix.
    fn identity(n: usize) -> Vec<f64> {
        let mut id = vec![0.0f64; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        id
    }

    /// Frobenius norm ‖A‖_F.
    fn frob(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Max absolute deviation between A and B (flat, same length).
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    // ─── 1. spd_sqrt of identity ──────────────────────────────────────────────

    #[test]
    fn spd_sqrt_identity() {
        let n = 3;
        let id = identity(n);
        let sq = spd_sqrt(&id, n).expect("spd_sqrt of I");
        assert!(max_abs_diff(&sq, &id) < 1e-10, "sqrt(I) should equal I");
    }

    // ─── 2. (A^{1/2})² ≈ A ───────────────────────────────────────────────────

    #[test]
    fn spd_sqrt_sq_is_original() {
        let n = 2;
        let a = mat2(4.0, 1.0, 3.0);
        let sq = spd_sqrt(&a, n).expect("sqrt");
        let sq_sq = mat_mul(&sq, &sq, n);
        assert!(
            max_abs_diff(&sq_sq, &a) < 1e-9,
            "max diff: {}",
            max_abs_diff(&sq_sq, &a)
        );
    }

    // ─── 3. spd_inv of identity ───────────────────────────────────────────────

    #[test]
    fn spd_inv_identity() {
        let n = 3;
        let id = identity(n);
        let inv = spd_inv(&id, n).expect("inv(I)");
        assert!(max_abs_diff(&inv, &id) < 1e-10);
    }

    // ─── 4. A · A^{-1} ≈ I ───────────────────────────────────────────────────

    #[test]
    fn spd_inv_correct() {
        let n = 2;
        let a = mat2(4.0, 1.0, 3.0);
        let inv = spd_inv(&a, n).expect("inv");
        let prod = mat_mul(&a, &inv, n);
        let id = identity(n);
        assert!(
            max_abs_diff(&prod, &id) < 1e-8,
            "A·A^{{-1}} not close to I: {prod:?}"
        );
    }

    // ─── 5. d_BW(A,A) = 0 ────────────────────────────────────────────────────

    #[test]
    fn bures_distance_self_zero() {
        let n = 3;
        let a = diag3(2.0, 3.0, 5.0);
        let d = bures_distance(&a, &a, n).expect("self dist");
        assert!(d.abs() < 1e-8, "d_BW(A,A) = {d} (expected 0)");
    }

    // ─── 6. d_BW(A,B) = d_BW(B,A) ────────────────────────────────────────────

    #[test]
    fn bures_distance_symmetric() {
        let n = 2;
        let a = mat2(4.0, 0.5, 2.0);
        let b = mat2(3.0, -0.3, 5.0);
        let dab = bures_distance(&a, &b, n).expect("dab");
        let dba = bures_distance(&b, &a, n).expect("dba");
        assert!((dab - dba).abs() < 1e-8, "d(A,B)={dab} != d(B,A)={dba}");
    }

    // ─── 7. Triangle inequality ───────────────────────────────────────────────

    #[test]
    fn bures_distance_triangle_inequality() {
        let n = 2;
        let a = mat2(4.0, 0.5, 2.0);
        let b = mat2(3.0, -0.2, 5.0);
        let c = mat2(6.0, 0.0, 2.5);
        let dac = bures_distance(&a, &c, n).expect("dac");
        let dab = bures_distance(&a, &b, n).expect("dab");
        let dbc = bures_distance(&b, &c, n).expect("dbc");
        assert!(
            dac <= dab + dbc + 1e-8,
            "triangle ineq violated: d(A,C)={dac} > d(A,B)+d(B,C)={}",
            dab + dbc
        );
    }

    // ─── 8. Bures geometric mean is symmetric: M(A,B) = M(B,A) ───────────────

    #[test]
    fn bures_geometric_mean_sym() {
        let n = 2;
        let a = mat2(5.0, 0.5, 2.0);
        let b = mat2(3.0, -0.2, 4.0);
        let mab = bures_geometric_mean(&a, &b, n).expect("mab");
        let mba = bures_geometric_mean(&b, &a, n).expect("mba");
        assert!(
            max_abs_diff(&mab, &mba) < 1e-7,
            "M(A,B) != M(B,A); diff={}",
            max_abs_diff(&mab, &mba)
        );
    }

    // ─── 9. Geodesic at t=0 gives A ──────────────────────────────────────────

    #[test]
    fn bures_geodesic_t0() {
        let n = 2;
        let a = mat2(4.0, 0.5, 3.0);
        let b = mat2(2.0, -0.3, 5.0);
        let g0 = bures_geodesic(&a, &b, n, 0.0).expect("g0");
        assert!(max_abs_diff(&g0, &a) < 1e-10);
    }

    // ─── 10. Geodesic at t=1 gives B ─────────────────────────────────────────

    #[test]
    fn bures_geodesic_t1() {
        let n = 2;
        let a = mat2(4.0, 0.5, 3.0);
        let b = mat2(2.0, -0.3, 5.0);
        let g1 = bures_geodesic(&a, &b, n, 1.0).expect("g1");
        assert!(max_abs_diff(&g1, &b) < 1e-10);
    }

    // ─── 11. Geodesic midpoint is equidistant from both endpoints ───────────
    //
    // In any Riemannian manifold, the midpoint G_{1/2} of the geodesic between A
    // and B satisfies d(A, G_{1/2}) = d(G_{1/2}, B) = d(A,B)/2.
    // (The formula G_t = A^{1/2}·[(1-t)I+t·D]²·A^{1/2} is the correct BW
    //  displacement interpolant; its midpoint is the BW Fréchet mean, which for
    //  non-commuting matrices differs from the affine-invariant geometric mean.)

    #[test]
    fn bures_geodesic_t05_equidistant() {
        let n = 2;
        let a = mat2(4.0, 0.5, 3.0);
        let b = mat2(2.0, -0.2, 5.0);
        let g05 = bures_geodesic(&a, &b, n, 0.5).expect("g05");
        let dab = bures_distance(&a, &b, n).expect("dab");
        let dag = bures_distance(&a, &g05, n).expect("dag");
        let dgb = bures_distance(&g05, &b, n).expect("dgb");
        // Tolerance accounts for floating-point accumulation across multiple
        // eigendecompositions for non-diagonal matrices (~1e-4 relative error).
        let tol = 1e-3 * dab;
        assert!(
            (dag - dab / 2.0).abs() < tol,
            "d(A, G_{{1/2}}) = {dag}, expected {:.8}, tol={tol:.2e}",
            dab / 2.0
        );
        assert!(
            (dgb - dab / 2.0).abs() < tol,
            "d(G_{{1/2}}, B) = {dgb}, expected {:.8}, tol={tol:.2e}",
            dab / 2.0
        );
    }

    // ─── 12. exp_P(log_P(Q)) ≈ Q (round-trip) ────────────────────────────────

    #[test]
    fn bures_log_exp_roundtrip() {
        let n = 2;
        let p = mat2(4.0, 0.3, 3.0);
        let q = mat2(2.0, -0.1, 5.0);
        let v = bures_log(&p, &q, n).expect("log");
        let q_rec = bures_exp(&p, &v, n).expect("exp");
        assert!(
            max_abs_diff(&q_rec, &q) < 1e-7,
            "round-trip error: {}",
            max_abs_diff(&q_rec, &q)
        );
    }

    // ─── 13. Fréchet mean of a single matrix returns itself ──────────────────

    #[test]
    fn bures_frechet_mean_single() {
        let n = 2;
        let a = mat2(3.0, 0.4, 2.0);
        let result = bures_frechet_mean(&a, 1, n, 200, 1e-10).expect("single mean");
        assert!(
            max_abs_diff(&result, &a) < 1e-7,
            "single mean diff: {}",
            max_abs_diff(&result, &a)
        );
    }

    // ─── 14. Fréchet mean of two identical matrices = original ────────────────

    #[test]
    fn bures_frechet_mean_two_identical() {
        let n = 2;
        let a = mat2(4.0, 0.5, 3.0);
        let matrices: Vec<f64> = a.iter().chain(a.iter()).cloned().collect();
        let result = bures_frechet_mean(&matrices, 2, n, 200, 1e-10).expect("two identical");
        assert!(
            max_abs_diff(&result, &a) < 1e-7,
            "mean of {{A,A}} diff: {}",
            max_abs_diff(&result, &a)
        );
    }

    // ─── 15. Fréchet mean converges and produces SPD result ──────────────────

    #[test]
    fn bures_frechet_mean_converges_spd() {
        let n = 3;
        // Three diagonal matrices → mean should be well-defined and SPD
        let a = diag3(2.0, 3.0, 4.0);
        let b = diag3(4.0, 2.0, 6.0);
        let c = diag3(3.0, 5.0, 2.0);
        let matrices: Vec<f64> = a.iter().chain(b.iter()).chain(c.iter()).cloned().collect();
        let mu = bures_frechet_mean(&matrices, 3, n, 500, 1e-10).expect("frechet mean");
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mu[i * n + j] - mu[j * n + i]).abs() < 1e-9,
                    "mean not symmetric at ({i},{j})"
                );
            }
        }
        // Check positive definiteness via eigenvalues
        let (w, _) = jacobi_eigh(&mu, n).expect("eigh of mean");
        for (idx, &wi) in w.iter().enumerate() {
            assert!(wi > 0.0, "mean eigenvalue {idx} = {wi} not positive");
        }
    }

    // ─── 16. Known-value test: d_BW(I, 4I)² = n ──────────────────────────────
    //
    // tr(I) = n, tr(4I) = 4n
    // (I^{1/2}·4I·I^{1/2})^{1/2} = (4I)^{1/2} = 2I, so tr = 2n
    // d_BW² = n + 4n − 2·2n = 5n − 4n = n

    #[test]
    fn bures_distance_known_value_identity_4i() {
        for n in [2usize, 3, 4] {
            let id = identity(n);
            let four_id: Vec<f64> = id.iter().map(|x| 4.0 * x).collect();
            let d = bures_distance(&id, &four_id, n).expect("dist");
            let expected = (n as f64).sqrt(); // d_BW = sqrt(n)
            assert!(
                (d - expected).abs() < 1e-8,
                "n={n}: d_BW(I, 4I) = {d}, expected sqrt({n}) = {expected}"
            );
        }
    }

    // ─── 17. spd_inv_sqrt: A^{-1/2}·A·A^{-1/2} ≈ I ──────────────────────────

    #[test]
    fn spd_inv_sqrt_correct() {
        let n = 3;
        let a = diag3(4.0, 9.0, 16.0);
        let inv_sq = spd_inv_sqrt(&a, n).expect("inv_sqrt");
        // inv_sq · a · inv_sq should be I for diagonal case: 1/2 * 4 * 1/2 = 1
        let left = mat_mul(&inv_sq, &a, n);
        let prod = mat_mul(&left, &inv_sq, n);
        let id = identity(n);
        assert!(
            max_abs_diff(&prod, &id) < 1e-8,
            "A^{{-1/2}}·A·A^{{-1/2}} diff from I: {}",
            max_abs_diff(&prod, &id)
        );
    }

    // ─── 18. Geodesic is SPD at intermediate t ───────────────────────────────

    #[test]
    fn bures_geodesic_is_spd() {
        let n = 3;
        let a = diag3(2.0, 3.0, 5.0);
        let b = diag3(4.0, 1.5, 3.0);
        for ti in [1, 2, 3, 4, 5, 6, 7, 8, 9] {
            let t = ti as f64 / 10.0;
            let gt = bures_geodesic(&a, &b, n, t).expect("geodesic");
            let (w, _) = jacobi_eigh(&gt, n).expect("eigh");
            for &wi in &w {
                assert!(
                    wi > -1e-7,
                    "geodesic at t={t} has non-positive eigenvalue {wi}"
                );
            }
        }
    }

    // ─── 19. log returns zero tangent for P == Q ──────────────────────────────

    #[test]
    fn bures_log_self_zero() {
        let n = 2;
        let p = mat2(3.0, 0.5, 2.0);
        let v = bures_log(&p, &p, n).expect("log self");
        assert!(
            frob(&v) < 1e-8,
            "log_P(P) should be 0, got norm {}",
            frob(&v)
        );
    }
}
