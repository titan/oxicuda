//! Strassen fast matrix multiplication.
//!
//! Strassen (1969) reduces the naive O(n³) matrix multiplication to
//! O(n^{log₂7}) ≈ O(n^{2.807}) by replacing 8 recursive sub-multiplications
//! in the standard divide-and-conquer approach with 7 multiplications at the
//! cost of more additions.
//!
//! ## Algorithm
//!
//! For an `n×n` matrix multiply, partition each matrix into four `n/2 × n/2`
//! blocks:
//! ```text
//!   A = [A11 A12]    B = [B11 B12]    C = [C11 C12]
//!       [A21 A22]        [B21 B22]        [C21 C22]
//! ```
//! Compute 7 products (instead of 8):
//! ```text
//!   M1 = (A11 + A22) × (B11 + B22)
//!   M2 = (A21 + A22) × B11
//!   M3 = A11 × (B12 - B22)
//!   M4 = A22 × (B21 - B11)
//!   M5 = (A11 + A12) × B22
//!   M6 = (A21 - A11) × (B11 + B12)
//!   M7 = (A12 - A22) × (B21 + B22)
//!
//!   C11 = M1 + M4 - M5 + M7
//!   C12 = M3 + M5
//!   C21 = M2 + M4
//!   C22 = M1 - M2 + M3 + M6
//! ```
//!
//! For `n` not a power of two, the matrices are zero-padded to the next power
//! of two internally; the extra rows/columns are stripped from the result.
//!
//! ## Threshold
//!
//! Below a configurable `base_n`, the algorithm falls back to the naive
//! `O(n³)` GEMM to avoid excessive recursion overhead.
//!
//! # Reference
//! - Strassen, V. (1969). "Gaussian elimination is not optimal".
//!   Numerische Mathematik, 13(4), 354–356.

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Strassen matrix multiplication: `C = A @ B`.
///
/// Both `A` and `B` are square `n × n` row-major matrices provided as `f64`
/// slices.  The result is returned as a new `Vec<f64>` of length `n × n`.
///
/// Uses a default recursion threshold of `base_n = 64`: matrices of size ≤ 64
/// are multiplied using the naive O(n³) algorithm.
///
/// # Errors
///
/// - [`BlasError::InvalidDimension`] if `n == 0`.
/// - [`BlasError::BufferTooSmall`] if `a` or `b` is shorter than `n * n`.
pub fn strassen(a: &[f64], b: &[f64], n: usize) -> BlasResult<Vec<f64>> {
    strassen_with_threshold(a, b, n, 64)
}

/// Strassen matrix multiplication with a configurable recursion threshold.
///
/// When the current sub-problem size ≤ `base_n`, falls back to
/// [`naive_dgemm`].
///
/// # Errors
///
/// Same as [`strassen`].
pub fn strassen_with_threshold(
    a: &[f64],
    b: &[f64],
    n: usize,
    base_n: usize,
) -> BlasResult<Vec<f64>> {
    if n == 0 {
        return Err(BlasError::InvalidDimension(
            "matrix dimension n must be > 0".to_string(),
        ));
    }
    if a.len() < n * n {
        return Err(BlasError::BufferTooSmall {
            expected: n * n,
            actual: a.len(),
        });
    }
    if b.len() < n * n {
        return Err(BlasError::BufferTooSmall {
            expected: n * n,
            actual: b.len(),
        });
    }

    // Pad to power of 2 if needed
    let padded = next_pow2(n);
    if padded == n {
        Ok(strassen_pow2(a, b, n, base_n))
    } else {
        // Zero-pad both matrices to padded × padded
        let a_pad = pad_matrix(a, n, padded);
        let b_pad = pad_matrix(b, n, padded);
        let c_pad = strassen_pow2(&a_pad, &b_pad, padded, base_n);
        // Extract top-left n × n block
        Ok(extract_block(&c_pad, padded, n))
    }
}

/// Standard naive O(n³) square GEMM for comparison and as base case.
///
/// Computes `C = A @ B` where all matrices are `n × n` row-major.
/// Returns the result as a `Vec<f64>` of length `n * n`.
pub fn naive_dgemm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for p in 0..n {
            let aip = a[i * n + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

/// Strassen on a power-of-2 sized matrix.
fn strassen_pow2(a: &[f64], b: &[f64], n: usize, base_n: usize) -> Vec<f64> {
    debug_assert!(n.is_power_of_two() || n == 1);

    // Base case
    if n <= base_n || n == 1 {
        return naive_dgemm(a, b, n);
    }

    let half = n / 2;

    // Extract sub-blocks (row-major, size half × half)
    let a11 = sub_block(a, n, 0, 0, half);
    let a12 = sub_block(a, n, 0, half, half);
    let a21 = sub_block(a, n, half, 0, half);
    let a22 = sub_block(a, n, half, half, half);

    let b11 = sub_block(b, n, 0, 0, half);
    let b12 = sub_block(b, n, 0, half, half);
    let b21 = sub_block(b, n, half, 0, half);
    let b22 = sub_block(b, n, half, half, half);

    // --- 7 Strassen products ------------------------------------------------
    // M1 = (A11 + A22) × (B11 + B22)
    let m1 = strassen_pow2(
        &mat_add(&a11, &a22, half),
        &mat_add(&b11, &b22, half),
        half,
        base_n,
    );
    // M2 = (A21 + A22) × B11
    let m2 = strassen_pow2(&mat_add(&a21, &a22, half), &b11, half, base_n);
    // M3 = A11 × (B12 - B22)
    let m3 = strassen_pow2(&a11, &mat_sub(&b12, &b22, half), half, base_n);
    // M4 = A22 × (B21 - B11)
    let m4 = strassen_pow2(&a22, &mat_sub(&b21, &b11, half), half, base_n);
    // M5 = (A11 + A12) × B22
    let m5 = strassen_pow2(&mat_add(&a11, &a12, half), &b22, half, base_n);
    // M6 = (A21 - A11) × (B11 + B12)
    let m6 = strassen_pow2(
        &mat_sub(&a21, &a11, half),
        &mat_add(&b11, &b12, half),
        half,
        base_n,
    );
    // M7 = (A12 - A22) × (B21 + B22)
    let m7 = strassen_pow2(
        &mat_sub(&a12, &a22, half),
        &mat_add(&b21, &b22, half),
        half,
        base_n,
    );

    // --- Assemble result quadrants ------------------------------------------
    // C11 = M1 + M4 - M5 + M7
    let c11 = mat_add(&mat_sub(&mat_add(&m1, &m4, half), &m5, half), &m7, half);
    // C12 = M3 + M5
    let c12 = mat_add(&m3, &m5, half);
    // C21 = M2 + M4
    let c21 = mat_add(&m2, &m4, half);
    // C22 = M1 - M2 + M3 + M6
    let c22 = mat_add(&mat_add(&mat_sub(&m1, &m2, half), &m3, half), &m6, half);

    // Merge quadrants into n × n result
    assemble_blocks(&c11, &c12, &c21, &c22, n, half)
}

// ---------------------------------------------------------------------------
// Matrix helpers (operate on half × half sub-matrices)
// ---------------------------------------------------------------------------

/// Extract the `size × size` sub-block starting at `(row_off, col_off)`
/// from an `n × n` row-major matrix.
fn sub_block(mat: &[f64], n: usize, row_off: usize, col_off: usize, size: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(size * size);
    for i in 0..size {
        let src = &mat[(row_off + i) * n + col_off..(row_off + i) * n + col_off + size];
        out.extend_from_slice(src);
    }
    out
}

/// Pointwise addition of two `n × n` matrices.
pub(crate) fn mat_add(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    a[..n * n]
        .iter()
        .zip(b[..n * n].iter())
        .map(|(&x, &y)| x + y)
        .collect()
}

/// Pointwise subtraction `a - b` of two `n × n` matrices.
pub(crate) fn mat_sub(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    a[..n * n]
        .iter()
        .zip(b[..n * n].iter())
        .map(|(&x, &y)| x - y)
        .collect()
}

/// Assemble four `half × half` quadrant blocks into an `n × n` matrix.
fn assemble_blocks(
    c11: &[f64],
    c12: &[f64],
    c21: &[f64],
    c22: &[f64],
    n: usize,
    half: usize,
) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..half {
        // Upper half: C11 | C12
        c[i * n..i * n + half].copy_from_slice(&c11[i * half..(i + 1) * half]);
        c[i * n + half..i * n + half + half].copy_from_slice(&c12[i * half..(i + 1) * half]);
        // Lower half: C21 | C22
        c[(i + half) * n..(i + half) * n + half].copy_from_slice(&c21[i * half..(i + 1) * half]);
        c[(i + half) * n + half..(i + half) * n + half + half]
            .copy_from_slice(&c22[i * half..(i + 1) * half]);
    }
    c
}

/// Zero-pad an `n × n` matrix to a `padded × padded` matrix.
fn pad_matrix(mat: &[f64], n: usize, padded: usize) -> Vec<f64> {
    debug_assert!(padded >= n, "pad_matrix: padded={padded} must be >= n={n}");
    debug_assert!(
        mat.len() >= n * n,
        "pad_matrix: mat.len()={} must be >= n*n={}",
        mat.len(),
        n * n
    );
    let mut out = vec![0.0_f64; padded * padded];
    for i in 0..n {
        let src_start = i * n;
        let src_end = src_start + n;
        let dst_start = i * padded;
        let dst_end = dst_start + n;
        out[dst_start..dst_end].copy_from_slice(&mat[src_start..src_end]);
    }
    out
}

/// Extract the top-left `out_n × out_n` block from a `full_n × full_n` matrix.
fn extract_block(mat: &[f64], full_n: usize, out_n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(out_n * out_n);
    for i in 0..out_n {
        out.extend_from_slice(&mat[i * full_n..i * full_n + out_n]);
    }
    out
}

/// Smallest power of two >= `n`.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Max absolute difference between two `f64` slices.
    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn strassen_2x2_correct() {
        // [1 2] × [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let c = strassen(&a, &b, 2).expect("strassen");
        let expected = vec![19.0_f64, 22.0, 43.0, 50.0];
        assert!(
            max_diff(&c, &expected) < 1e-10,
            "2×2 product incorrect: {c:?}"
        );
    }

    #[test]
    fn strassen_4x4_matches_naive() {
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let b: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.7 + 1.0).collect();
        let naive = naive_dgemm(&a, &b, n);
        let stras = strassen(&a, &b, n).expect("strassen");
        assert!(
            max_diff(&stras, &naive) < 1e-10,
            "4×4 Strassen != naive: diff={:.2e}",
            max_diff(&stras, &naive)
        );
    }

    #[test]
    fn strassen_8x8_matches_naive() {
        let n = 8;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.2 + 0.5).cos()).collect();
        let naive = naive_dgemm(&a, &b, n);
        let stras = strassen(&a, &b, n).expect("strassen");
        assert!(
            max_diff(&stras, &naive) < 1e-9,
            "8×8 Strassen != naive: diff={:.2e}",
            max_diff(&stras, &naive)
        );
    }

    #[test]
    fn strassen_identity_matrix() {
        // A @ I = A
        let n = 4;
        let a: Vec<f64> = (1..=n * n).map(|x| x as f64).collect();
        let mut identity = vec![0.0_f64; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let c = strassen(&a, &identity, n).expect("strassen");
        assert!(max_diff(&c, &a) < 1e-12, "A @ I should equal A");
    }

    #[test]
    fn strassen_zero_matrix() {
        let n = 4;
        let a: Vec<f64> = (1..=n * n).map(|x| x as f64).collect();
        let zero = vec![0.0_f64; n * n];
        let c = strassen(&a, &zero, n).expect("strassen");
        assert!(max_diff(&c, &zero) < 1e-12, "A @ 0 should equal 0");
    }

    #[test]
    fn strassen_n_not_pow2_padded_internally() {
        // n=3: pad to 4, result should be correct 3×3 product
        let n = 3;
        let a: Vec<f64> = (1..=n * n).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=n * n).map(|x| (x * 2) as f64).collect();
        let naive = naive_dgemm(&a, &b, n);
        let stras = strassen(&a, &b, n).expect("strassen");
        assert!(
            max_diff(&stras, &naive) < 1e-10,
            "3×3 Strassen (padded to 4) != naive"
        );
    }

    #[test]
    fn strassen_threshold_equals_n_no_recursion() {
        // base_n = n means: always use naive, result still correct
        let n = 8;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.5 + 0.1).collect();
        let b: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.3 + 0.2).collect();
        let naive = naive_dgemm(&a, &b, n);
        let stras = strassen_with_threshold(&a, &b, n, n).expect("strassen");
        assert!(
            max_diff(&stras, &naive) < 1e-10,
            "threshold=n should give naive result"
        );
    }

    #[test]
    fn strassen_associative() {
        // (AB)C ≈ A(BC)
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.1 + 0.05).collect();
        let b: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.07 + 0.03).collect();
        let c: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.13 + 0.02).collect();

        let ab = strassen(&a, &b, n).expect("AB");
        let ab_c = strassen(&ab, &c, n).expect("AB_C");
        let bc = strassen(&b, &c, n).expect("BC");
        let a_bc = strassen(&a, &bc, n).expect("A_BC");

        let err = max_diff(&ab_c, &a_bc);
        assert!(err < 1e-9, "(AB)C != A(BC), diff={err:.2e}");
    }

    #[test]
    fn strassen_error_bounded() {
        let n = 16;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.01).sin()).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.02).cos()).collect();
        let naive = naive_dgemm(&a, &b, n);
        let stras = strassen(&a, &b, n).expect("strassen");
        let err = max_diff(&stras, &naive);
        assert!(err < 1e-10, "Strassen error {err:.2e} should be < 1e-10");
    }

    #[test]
    fn strassen_1x1() {
        let a = vec![7.0_f64];
        let b = vec![6.0_f64];
        let c = strassen(&a, &b, 1).expect("strassen 1×1");
        assert!(
            (c[0] - 42.0).abs() < 1e-12,
            "1×1: expected 42, got {}",
            c[0]
        );
    }
}
