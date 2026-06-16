//! Matrix-based binding schemes: MBAT and VTB.
//!
//! Two related Vector Symbolic Architecture (VSA) binding operators that realise the
//! role↔filler product as a *matrix–vector* multiplication rather than a circular
//! convolution.
//!
//! # MBAT — Matrix Binding by Additive Terms
//!
//! Reference: S. I. Gallant & T. W. Okaywe, "Representing Objects, Relations, and Sequences,"
//! *Neural Computation* 25:8 (2013).
//!
//! A *role* is a square `D×D` matrix `M` (stored flattened row-major, length `D*D`). Binding a
//! filler vector `f` is the matrix–vector product, and unbinding is the inverse map. When `M`
//! is orthonormal the inverse is simply the transpose `Mᵀ`, so binding is cheaply reversible:
//!
//! ```text
//! bind(M, f)[i]      = Σ_j M[i·D + j] · f[j]          (y = M · f)
//! unbind(M, y)[j]    = Σ_i M[i·D + j] · y[i]          (x = Mᵀ · y)
//! Mᵀ·M ≈ I  ⇒  unbind(M, bind(M, f)) ≈ f
//! ```
//!
//! # VTB — Vector-derived Transformation Binding
//!
//! Reference: J. Gosmann & C. Eliasmith, "Vector-Derived Transformation Binding: An Improved
//! Binding Operation for Deep Symbol-Like Processing in Neural Networks,"
//! *Neural Computation* 31:5 (2019).
//!
//! VTB binds two vectors `a`, `b` of dimension `D` where `D = d²` must be a perfect square. The
//! first operand `a` is reshaped into a `d×d` matrix `Vₐ` (scaled by `d^{1/4}`); the second
//! operand `b` is reshaped into a `d×d` matrix `B` (row-major), and binding is the
//! matrix–matrix product `Vₐ · B`, flattened back to length `D`:
//!
//! ```text
//! d   = √D
//! s   = d^{1/4}
//! Vₐ[r][c] = s · a[r·d + c]              (reshape a, scaled)
//! B[c][k]  = b[c·d + k]                  (reshape b)
//! bind(a, b):   out[r·d + k] = Σ_c Vₐ[r][c] · b[c·d + k]      (Vₐ · B)
//! unbind(a, y): out[c·d + k] = Σ_r Vₐ[r][c] · y[r·d + k]      (Vₐᵀ · Y)
//! ```
//!
//! Unbinding applies the transpose `Vₐᵀ`. When `a` is a *unitary* VTB vector — one whose `Vₐ`
//! is orthogonal — the transpose is the exact inverse and recovery is perfect:
//! `unbind(a, bind(a, b)) = b`. For an arbitrary unit `a`, `Vₐ` is only approximately
//! orthogonal, so recovery is approximate.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

// ── MBAT ─────────────────────────────────────────────────────────────────────

/// Bind a `filler` with a `D×D` role `matrix` (row-major): `y = M · f`.
///
/// Computes `y[i] = Σ_j matrix[i·dim + j] · filler[j]`.
///
/// # Errors
///
/// - [`HdcError::ZeroDimension`] if `dim == 0`.
/// - [`HdcError::DimensionMismatch`] if `matrix.len() != dim*dim` or `filler.len() != dim`.
pub fn mbat_bind(matrix: &[f32], filler: &[f32], dim: usize) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    if matrix.len() != dim * dim {
        return Err(HdcError::DimensionMismatch {
            expected: dim * dim,
            got: matrix.len(),
        });
    }
    if filler.len() != dim {
        return Err(HdcError::DimensionMismatch {
            expected: dim,
            got: filler.len(),
        });
    }
    let mut out = vec![0f32; dim];
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &matrix[i * dim..(i + 1) * dim];
        let acc: f64 = row
            .iter()
            .zip(filler.iter())
            .map(|(&m, &f)| (m as f64) * (f as f64))
            .sum();
        *slot = acc as f32;
    }
    Ok(out)
}

/// Unbind via the transpose of a `D×D` role `matrix` (row-major): `x = Mᵀ · y`.
///
/// Computes `y[j] = Σ_i matrix[i·dim + j] · bound[i]`. For an orthonormal `M` this inverts
/// [`mbat_bind`], since `Mᵀ·M ≈ I`.
///
/// # Errors
///
/// - [`HdcError::ZeroDimension`] if `dim == 0`.
/// - [`HdcError::DimensionMismatch`] if `matrix.len() != dim*dim` or `bound.len() != dim`.
pub fn mbat_unbind_transpose(matrix: &[f32], bound: &[f32], dim: usize) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    if matrix.len() != dim * dim {
        return Err(HdcError::DimensionMismatch {
            expected: dim * dim,
            got: matrix.len(),
        });
    }
    if bound.len() != dim {
        return Err(HdcError::DimensionMismatch {
            expected: dim,
            got: bound.len(),
        });
    }
    let mut out = vec![0f64; dim];
    for (i, &y) in bound.iter().enumerate() {
        let yi = y as f64;
        let row = &matrix[i * dim..(i + 1) * dim];
        for (slot, &m) in out.iter_mut().zip(row.iter()) {
            *slot += (m as f64) * yi;
        }
    }
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// Build a random orthonormal `D×D` matrix (flattened row-major) by Gram-Schmidt.
///
/// Each row is drawn uniformly in `[-1, 1]`, then modified Gram-Schmidt orthonormalises the
/// rows: every row has its projections onto the previously fixed rows subtracted before being
/// normalised. The result satisfies `Mᵀ·M ≈ I` (rows orthonormal), so
/// [`mbat_unbind_transpose`] inverts [`mbat_bind`].
///
/// # Errors
///
/// - [`HdcError::ZeroDimension`] if `dim == 0`.
/// - [`HdcError::DivisionByZero`] if a row collapses to near-zero norm after orthogonalisation
///   (a degenerate draw); retry with a different seed.
pub fn random_orthogonal_matrix(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    // Build the rows one at a time as f64 for numerical stability.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut candidate: Vec<f64> = (0..dim)
            .map(|_| (rng.next_f32() as f64) * 2.0 - 1.0)
            .collect();
        // Subtract projections onto the already-orthonormal rows.
        for basis in &rows {
            let dot: f64 = candidate
                .iter()
                .zip(basis.iter())
                .map(|(&c, &b)| c * b)
                .sum();
            for (c, &b) in candidate.iter_mut().zip(basis.iter()) {
                *c -= dot * b;
            }
        }
        let norm = candidate.iter().map(|&c| c * c).sum::<f64>().sqrt();
        if norm < 1e-9 {
            return Err(HdcError::DivisionByZero);
        }
        let inv = 1.0 / norm;
        for c in candidate.iter_mut() {
            *c *= inv;
        }
        rows.push(candidate);
    }
    let mut out = vec![0f32; dim * dim];
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i * dim + j] = v as f32;
        }
    }
    Ok(out)
}

// ── VTB ──────────────────────────────────────────────────────────────────────

/// Return the integer square root of `d` if `d` is a perfect square, else `None`.
///
/// Uses a floating-point estimate refined over `{r-1, r, r+1}` to avoid rounding errors near
/// exact squares. `0` maps to `Some(0)` and `1` to `Some(1)`.
#[must_use]
pub fn is_perfect_square(d: usize) -> Option<usize> {
    let r = (d as f64).sqrt() as usize;
    [r.saturating_sub(1), r, r + 1]
        .into_iter()
        .find(|&cand| cand * cand == d)
}

/// The largest perfect square not exceeding `len` (used as the `expected` mismatch hint).
fn floor_square(len: usize) -> usize {
    let r = (len as f64).sqrt() as usize;
    // Walk down until r*r <= len (guards float over-estimate).
    let mut r = r + 1;
    while r > 0 && r * r > len {
        r -= 1;
    }
    r * r
}

/// VTB bind of two equal-length vectors whose length is a perfect square: `Vₐ · B`.
///
/// With `d = √D` and `s = d^{1/4}`, both operands are reshaped to `d×d` matrices and multiplied:
/// `out[r·d + k] = Σ_c (s · a[r·d + c]) · b[c·d + k]`, the flattened product of the scaled
/// reshape of `a` with the reshape of `b`.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `a` (and hence `b`) is empty.
/// - [`HdcError::DimensionMismatch`] if `a.len() != b.len()`, or the length is not a perfect
///   square (the `expected` field reports the largest square `≤ len`).
pub fn vtb_bind(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let len = a.len();
    let Some(d) = is_perfect_square(len) else {
        return Err(HdcError::DimensionMismatch {
            expected: floor_square(len),
            got: len,
        });
    };
    let s = (d as f64).powf(0.25);
    let mut out = vec![0f32; len];
    // out[r][k] = s · Σ_c a[r][c] · b[c][k]  (Vₐ · B), all row-major d×d.
    for r in 0..d {
        let a_row = &a[r * d..(r + 1) * d];
        for k in 0..d {
            let mut acc = 0f64;
            for (c, &av) in a_row.iter().enumerate() {
                acc += (av as f64) * (b[c * d + k] as f64);
            }
            out[r * d + k] = (s * acc) as f32;
        }
    }
    Ok(out)
}

/// VTB unbind: apply the transpose `Vₐᵀ` to recover `b`.
///
/// `out[c·d + k] = Σ_r (d^{1/4} · a[r·d + c]) · bound[r·d + k]`, the flattened product `Vₐᵀ · Y`.
/// When `a` is a unitary VTB vector (`Vₐ` orthogonal) this is the exact inverse of
/// [`vtb_bind`]: `vtb_unbind(a, vtb_bind(a, b)) = b`; otherwise it is approximate.
///
/// # Errors
///
/// Same validation as [`vtb_bind`].
pub fn vtb_unbind(a: &[f32], bound: &[f32]) -> HdcResult<Vec<f32>> {
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != bound.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: bound.len(),
        });
    }
    let len = a.len();
    let Some(d) = is_perfect_square(len) else {
        return Err(HdcError::DimensionMismatch {
            expected: floor_square(len),
            got: len,
        });
    };
    let s = (d as f64).powf(0.25);
    let mut out = vec![0f32; len];
    // out[c][k] = s · Σ_r a[r][c] · bound[r][k]  (Vₐᵀ · Y), all row-major d×d.
    for c in 0..d {
        for k in 0..d {
            let mut acc = 0f64;
            for r in 0..d {
                acc += (a[r * d + c] as f64) * (bound[r * d + k] as f64);
            }
            out[c * d + k] = (s * acc) as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a random unit-norm f32 vector of length `dim`.
    fn random_unit(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x = ((*x as f64) / norm) as f32;
            }
        }
        v
    }

    /// Build a *unitary* VTB vector of length `dim = d²`: a vector whose scaled reshape
    /// `Vₐ = d^{1/4}·reshape(a)` is orthogonal. We pick a random orthonormal `d×d` matrix `Q`
    /// and set `reshape(a) = d^{−1/4}·Q`, so `Vₐ = Q`. The transpose is then the exact inverse,
    /// which is the precondition under which VTB self-inverts.
    fn unitary_vtb(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let d = is_perfect_square(dim).expect("dim must be a perfect square");
        let q = random_orthogonal_matrix(d, rng).expect("orthogonal");
        let scale = (d as f64).powf(-0.25);
        q.iter().map(|&v| ((v as f64) * scale) as f32).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum();
        let na = a
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let nb = b
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        (dot / (na * nb)) as f32
    }

    // ── MBAT ──────────────────────────────────────────────────────────────────

    #[test]
    fn mbat_bind_manual_2x2() {
        // M = [[1, 2], [3, 4]], f = [5, 6] → y = [1*5+2*6, 3*5+4*6] = [17, 39].
        let m = vec![1.0f32, 2.0, 3.0, 4.0];
        let f = vec![5.0f32, 6.0];
        let y = mbat_bind(&m, &f, 2).expect("bind");
        assert!((y[0] - 17.0).abs() < 1e-5, "y0={}", y[0]);
        assert!((y[1] - 39.0).abs() < 1e-5, "y1={}", y[1]);
    }

    #[test]
    fn mbat_unbind_transpose_manual_2x2() {
        // Mᵀ·y with M = [[1,2],[3,4]], y = [17, 39] → [1*17+3*39, 2*17+4*39] = [134, 190].
        let m = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![17.0f32, 39.0];
        let x = mbat_unbind_transpose(&m, &y, 2).expect("unbind");
        assert!((x[0] - 134.0).abs() < 1e-4, "x0={}", x[0]);
        assert!((x[1] - 190.0).abs() < 1e-4, "x1={}", x[1]);
    }

    #[test]
    fn mbat_orthogonal_round_trip() {
        let mut rng = LcgRng::new(0xABCD_1234);
        let dim = 8;
        let m = random_orthogonal_matrix(dim, &mut rng).expect("ortho");
        let f = random_unit(dim, &mut rng);
        let bound = mbat_bind(&m, &f, dim).expect("bind");
        let recovered = mbat_unbind_transpose(&m, &bound, dim).expect("unbind");
        for (r, orig) in recovered.iter().zip(f.iter()) {
            assert!((r - orig).abs() < 1e-3, "recovered {r} != filler {orig}");
        }
    }

    #[test]
    fn orthogonal_matrix_rows_are_orthonormal() {
        let mut rng = LcgRng::new(0x55AA_55AA);
        let dim = 8;
        let m = random_orthogonal_matrix(dim, &mut rng).expect("ortho");
        for i in 0..dim {
            let row_i = &m[i * dim..(i + 1) * dim];
            for j in 0..dim {
                let row_j = &m[j * dim..(j + 1) * dim];
                let dot: f64 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-4, "row {i} self-dot {dot}");
                } else {
                    assert!(dot.abs() < 1e-4, "rows {i},{j} dot {dot}");
                }
            }
        }
    }

    #[test]
    fn mbat_rejects_bad_matrix_len() {
        // dim=2 expects matrix len 4; give 3.
        let res = mbat_bind(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn mbat_rejects_bad_filler_len() {
        // dim=2 expects filler len 2; give 3.
        let res = mbat_bind(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0], 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn mbat_rejects_zero_dim() {
        assert!(matches!(
            mbat_bind(&[], &[], 0),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            mbat_unbind_transpose(&[], &[], 0),
            Err(HdcError::ZeroDimension)
        ));
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            random_orthogonal_matrix(0, &mut rng),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn mbat_orthogonal_round_trip_large() {
        // A larger D to exercise the Gram-Schmidt stability.
        let mut rng = LcgRng::new(0x0F0F_0F0F);
        let dim = 32;
        let m = random_orthogonal_matrix(dim, &mut rng).expect("ortho");
        let f = random_unit(dim, &mut rng);
        let bound = mbat_bind(&m, &f, dim).expect("bind");
        let recovered = mbat_unbind_transpose(&m, &bound, dim).expect("unbind");
        let sim = cosine(&recovered, &f);
        assert!(sim > 0.999, "round-trip cosine {sim}");
    }

    // ── VTB ───────────────────────────────────────────────────────────────────

    #[test]
    fn is_perfect_square_correctness() {
        assert_eq!(is_perfect_square(4), Some(2));
        assert_eq!(is_perfect_square(9), Some(3));
        assert_eq!(is_perfect_square(8), None);
        assert_eq!(is_perfect_square(16), Some(4));
        assert_eq!(is_perfect_square(0), Some(0));
        assert_eq!(is_perfect_square(1), Some(1));
        assert_eq!(is_perfect_square(15), None);
        assert_eq!(is_perfect_square(10000), Some(100));
    }

    #[test]
    fn vtb_round_trip_d16() {
        // With a unitary `a` (orthogonal Vₐ), the transpose unbind recovers `b` essentially
        // exactly. `b` is an arbitrary random unit vector.
        let mut rng = LcgRng::new(0xDEAD_BEEF);
        let a = unitary_vtb(16, &mut rng);
        let b = random_unit(16, &mut rng);
        let bound = vtb_bind(&a, &b).expect("bind");
        let recovered = vtb_unbind(&a, &bound).expect("unbind");
        let sim = cosine(&recovered, &b);
        assert!(sim > 0.95, "vtb round-trip cosine {sim}");
    }

    #[test]
    fn vtb_round_trip_d64() {
        let mut rng = LcgRng::new(0x1234_5678);
        let a = unitary_vtb(64, &mut rng);
        let b = random_unit(64, &mut rng);
        let bound = vtb_bind(&a, &b).expect("bind");
        let recovered = vtb_unbind(&a, &bound).expect("unbind");
        let sim = cosine(&recovered, &b);
        assert!(sim > 0.95, "vtb round-trip cosine {sim}");
    }

    #[test]
    fn vtb_rejects_non_square_length() {
        // len 8 is not a perfect square.
        let a = vec![0.1f32; 8];
        let b = vec![0.2f32; 8];
        let res = vtb_bind(&a, &b);
        match res {
            Err(HdcError::DimensionMismatch { expected, got }) => {
                assert_eq!(got, 8);
                assert_eq!(expected, 4, "floor square of 8 should be 4");
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn vtb_rejects_mismatched_lengths() {
        let a = vec![0.1f32; 16];
        let b = vec![0.2f32; 9];
        let res = vtb_bind(&a, &b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn vtb_rejects_empty() {
        let res = vtb_bind(&[], &[]);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn vtb_d1_trivial() {
        // d=1: Va = [[1^{1/4} * a0]] = [[a0]], bind = a0*b0, unbind = a0*(a0*b0).
        let a = vec![1.0f32];
        let b = vec![7.0f32];
        let bound = vtb_bind(&a, &b).expect("bind");
        assert!((bound[0] - 7.0).abs() < 1e-5, "bound={}", bound[0]);
        let recovered = vtb_unbind(&a, &bound).expect("unbind");
        assert!(
            (recovered[0] - 7.0).abs() < 1e-5,
            "recovered={}",
            recovered[0]
        );
    }

    #[test]
    fn vtb_unbind_rejects_mismatch() {
        let a = vec![0.1f32; 16];
        let bound = vec![0.2f32; 8];
        let res = vtb_unbind(&a, &bound);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn vtb_bind_preserves_length() {
        let mut rng = LcgRng::new(0xFEED);
        let a = random_unit(25, &mut rng);
        let b = random_unit(25, &mut rng);
        let bound = vtb_bind(&a, &b).expect("bind");
        assert_eq!(bound.len(), 25);
    }
}
