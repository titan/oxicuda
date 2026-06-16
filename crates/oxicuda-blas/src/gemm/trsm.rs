//! Triangular solve with multiple right-hand sides (STRSM, CPU reference).
//!
//! Solves the BLAS Level-3 triangular matrix equation
//!
//! ```text
//! A · X = alpha · B
//! ```
//!
//! for `X`, where `A ∈ ℝ^{m×m}` is triangular (lower or upper, with unit or
//! non-unit diagonal) and `B, X ∈ ℝ^{m×n}` are row-major. The solution
//! overwrites `B` in place (cuBLAS / LAPACK convention). This is the
//! left-side, no-transpose case (`side = Left`, `trans = N`), which is the
//! variant LU- and Cholesky-based linear solvers rely on for forward/back
//! substitution against many right-hand sides at once.
//!
//! * **Lower** triangular → forward substitution (rows `0 → m-1`).
//! * **Upper** triangular → back substitution (rows `m-1 → 0`).

use crate::error::{BlasError, BlasResult};

/// Solve `A · X = alpha · B`, overwriting `B` with `X` (left, no-transpose).
///
/// * `m` — order of the triangular matrix `A` and row count of `B`.
/// * `n` — number of right-hand-side columns in `B`.
/// * `a` — `m · m` elements, row-major `[m × m]`; only the referenced triangle
///   is read.
/// * `b` — `m · n` elements (in/out), row-major `[m × n]`.
/// * `lower` — `true` if `A` is lower triangular, `false` if upper.
/// * `unit_diag` — `true` treats the diagonal of `A` as all ones (and does not
///   read it), matching `CUBLAS_DIAG_UNIT`.
///
/// # Errors
///
/// * [`BlasError::InvalidDimension`] if `m == 0` or `n == 0`.
/// * [`BlasError::DimensionMismatch`] if `a.len() != m*m` or `b.len() != m*n`.
/// * [`BlasError::InvalidArgument`] if a non-unit diagonal entry is zero
///   (singular triangular factor).
pub fn strsm(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &mut [f32],
    lower: bool,
    unit_diag: bool,
) -> BlasResult<()> {
    if m == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "strsm: m and n must be ≥ 1 (got m={m}, n={n})"
        )));
    }
    if a.len() != m * m {
        return Err(BlasError::DimensionMismatch(format!(
            "strsm: A has {} elements, expected m*m = {}",
            a.len(),
            m * m
        )));
    }
    if b.len() != m * n {
        return Err(BlasError::DimensionMismatch(format!(
            "strsm: B has {} elements, expected m*n = {}",
            b.len(),
            m * n
        )));
    }

    // Pre-scale the right-hand side by alpha.
    if alpha != 1.0 {
        for bv in b.iter_mut() {
            *bv *= alpha;
        }
    }

    // Solve column-by-column. For each RHS column `col`, solve A·x = b[:,col].
    if lower {
        // Forward substitution: x_i = (b_i − Σ_{j<i} A_ij x_j) / A_ii.
        for i in 0..m {
            for col in 0..n {
                let mut sum = b[i * n + col];
                for j in 0..i {
                    sum -= a[i * m + j] * b[j * n + col];
                }
                if unit_diag {
                    b[i * n + col] = sum;
                } else {
                    let diag = a[i * m + i];
                    if diag == 0.0 {
                        return Err(BlasError::InvalidArgument(format!(
                            "strsm: zero diagonal at A[{i},{i}] (singular)"
                        )));
                    }
                    b[i * n + col] = sum / diag;
                }
            }
        }
    } else {
        // Back substitution: x_i = (b_i − Σ_{j>i} A_ij x_j) / A_ii.
        for i in (0..m).rev() {
            for col in 0..n {
                let mut sum = b[i * n + col];
                for j in (i + 1)..m {
                    sum -= a[i * m + j] * b[j * n + col];
                }
                if unit_diag {
                    b[i * n + col] = sum;
                } else {
                    let diag = a[i * m + i];
                    if diag == 0.0 {
                        return Err(BlasError::InvalidArgument(format!(
                            "strsm: zero diagonal at A[{i},{i}] (singular)"
                        )));
                    }
                    b[i * n + col] = sum / diag;
                }
            }
        }
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense `A · X` for verification (row-major).
    fn matmul(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..m {
                    acc += a[i * m + p] * x[p * n + col];
                }
                out[i * n + col] = acc;
            }
        }
        out
    }

    #[test]
    fn identity_lower_solves() {
        // A = I (lower) → X = B.
        let (m, n) = (3, 2);
        let mut a = vec![0.0_f32; m * m];
        for i in 0..m {
            a[i * m + i] = 1.0;
        }
        let b0: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let mut b = b0.clone();
        strsm(m, n, 1.0, &a, &mut b, true, false).expect("trsm");
        for (got, exp) in b.iter().zip(b0.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn lower_forward_substitution() {
        // A = [[2,0],[3,4]] lower. Solve A X = B with B = [[2],[11]].
        // Row0: 2·x0 = 2 → x0 = 1. Row1: 3·1 + 4·x1 = 11 → x1 = 2.
        let (m, n) = (2, 1);
        let a = vec![2.0, 0.0, 3.0, 4.0];
        let mut b = vec![2.0, 11.0];
        strsm(m, n, 1.0, &a, &mut b, true, false).expect("trsm");
        assert!((b[0] - 1.0).abs() < 1e-5);
        assert!((b[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn upper_back_substitution() {
        // A = [[4,1],[0,2]] upper. Solve A X = B with B = [[6],[4]].
        // Row1: 2·x1 = 4 → x1 = 2. Row0: 4·x0 + 1·2 = 6 → x0 = 1.
        let (m, n) = (2, 1);
        let a = vec![4.0, 1.0, 0.0, 2.0];
        let mut b = vec![6.0, 4.0];
        strsm(m, n, 1.0, &a, &mut b, false, false).expect("trsm");
        assert!((b[0] - 1.0).abs() < 1e-5);
        assert!((b[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn solve_then_multiply_roundtrip() {
        // Solve A X = B, then verify A·X ≈ B.
        let (m, n) = (4, 3);
        // Lower triangular, well-conditioned.
        let mut a = vec![0.0_f32; m * m];
        for i in 0..m {
            a[i * m + i] = (i as f32) + 3.0;
            for j in 0..i {
                a[i * m + j] = 0.5 * ((i + j) as f32).sin();
            }
        }
        let b0: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let mut b = b0.clone();
        strsm(m, n, 1.0, &a, &mut b, true, false).expect("trsm");
        let recon = matmul(&a, &b, m, n);
        for (got, exp) in recon.iter().zip(b0.iter()) {
            assert!((got - exp).abs() < 1e-4, "recon {got} vs B {exp}");
        }
    }

    #[test]
    fn unit_diagonal_ignored() {
        // Unit-diag treats diagonal as 1 regardless of stored value.
        // A_stored = [[9,0],[3,9]] but unit_diag → effectively [[1,0],[3,1]].
        // Solve A X = B, B = [[5],[5]]: x0 = 5; 3·5 + x1 = 5 → x1 = -10.
        let (m, n) = (2, 1);
        let a = vec![9.0, 0.0, 3.0, 9.0];
        let mut b = vec![5.0, 5.0];
        strsm(m, n, 1.0, &a, &mut b, true, true).expect("trsm");
        assert!((b[0] - 5.0).abs() < 1e-5);
        assert!((b[1] + 10.0).abs() < 1e-5);
    }

    #[test]
    fn alpha_scaling() {
        // alpha=2: solve A X = 2B. A=I → X = 2B.
        let (m, n) = (2, 2);
        let mut a = vec![0.0_f32; m * m];
        for i in 0..m {
            a[i * m + i] = 1.0;
        }
        let mut b = vec![1.0, 2.0, 3.0, 4.0];
        strsm(m, n, 2.0, &a, &mut b, true, false).expect("trsm");
        assert_eq!(b, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn singular_diagonal_error() {
        let (m, n) = (2, 1);
        let a = vec![0.0, 0.0, 3.0, 4.0]; // A[0,0] = 0
        let mut b = vec![1.0, 2.0];
        let err = strsm(m, n, 1.0, &a, &mut b, true, false);
        assert!(matches!(err, Err(BlasError::InvalidArgument(_))));
    }

    #[test]
    fn dim_mismatch_error() {
        let (m, n) = (3, 2);
        let a = vec![1.0_f32; m * m - 1]; // wrong
        let mut b = vec![0.0_f32; m * n];
        let err = strsm(m, n, 1.0, &a, &mut b, true, false);
        assert!(matches!(err, Err(BlasError::DimensionMismatch(_))));
    }

    #[test]
    fn m_0_error() {
        let mut b = vec![0.0_f32; 0];
        let err = strsm(0, 2, 1.0, &[], &mut b, true, false);
        assert!(matches!(err, Err(BlasError::InvalidDimension(_))));
    }

    #[test]
    fn multiple_rhs_independent() {
        // Two RHS columns solved independently.
        let (m, n) = (2, 2);
        let a = vec![2.0, 0.0, 0.0, 5.0]; // diagonal
        let mut b = vec![4.0, 10.0, 6.0, 15.0]; // cols: [4;6], [10;15]
        strsm(m, n, 1.0, &a, &mut b, true, false).expect("trsm");
        // col0: [4/2; 6/5]=[2;1.2]; col1: [10/2;15/5]=[5;3].
        assert!((b[0] - 2.0).abs() < 1e-5);
        assert!((b[2] - 1.2).abs() < 1e-5);
        assert!((b[1] - 5.0).abs() < 1e-5);
        assert!((b[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn output_finite() {
        let (m, n) = (5, 3);
        let mut a = vec![0.0_f32; m * m];
        for i in 0..m {
            a[i * m + i] = (i as f32) + 2.0;
            for j in (i + 1)..m {
                a[i * m + j] = 0.2 * ((i * j) as f32).cos();
            }
        }
        let mut b: Vec<f32> = (0..m * n).map(|i| (i as f32).sin()).collect();
        strsm(m, n, 1.0, &a, &mut b, false, false).expect("trsm");
        for &v in &b {
            assert!(v.is_finite());
        }
    }
}
