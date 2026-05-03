//! HiPPO-LegS (Legendre Stability) matrix initialization.
//!
//! The HiPPO framework provides principled initialization of SSM matrices
//! so that the hidden state optimally compresses the history of the input
//! signal via projection onto Legendre polynomials.
//!
//! ## HiPPO-LegS matrices
//!
//! The continuous-time recurrence matrices are:
//!
//! ```text
//! A[n, k] = −√(2n+1) · √(2k+1)   for n > k
//! A[n, n] = −(n + 1)
//! A[n, k] = 0                      for n < k
//!
//! B[n] = √(2n+1)
//! ```
//!
//! ## NPLR decomposition
//!
//! HiPPO-LegS admits the Normal Plus Low Rank (NPLR) decomposition
//! `A = Λ − P·Q^T` where:
//!
//! ```text
//! λ[n]  = −(n + 0.5)
//! p[n]  = √(n + 0.5)
//! q[n]  = p[n]
//! ```
//!
//! This structure enables the efficient Cauchy-kernel computation of the
//! SSM convolution filter used by S4.

use crate::error::{MambaError, MambaResult};

// ─── HiPPO-LegS full matrices ────────────────────────────────────────────────

/// Compute the HiPPO-LegS (Legendre Stability) A matrix (`N×N`, row-major)
/// and B vector (`N`).
///
/// The matrices are defined as:
/// - `A[n, k] = −√(2n+1)·√(2k+1)` for `n > k`
/// - `A[n, n] = −(n + 1)`
/// - `A[n, k] = 0` for `n < k`
/// - `B[n] = √(2n+1)`
///
/// # Errors
///
/// [`MambaError::InvalidSsmOrder`] if `n == 0`.
pub fn hippo_legs(n: usize) -> MambaResult<(Vec<f32>, Vec<f32>)> {
    if n == 0 {
        return Err(MambaError::InvalidSsmOrder(0));
    }

    let mut a_flat = vec![0.0_f32; n * n];
    let mut b = Vec::with_capacity(n);

    for row in 0..n {
        let sqrt_2row_p1 = ((2 * row + 1) as f32).sqrt();
        b.push(sqrt_2row_p1);

        for col in 0..n {
            let idx = row * n + col;
            if col < row {
                // Strictly lower-triangular: A[row, col] = -sqrt(2*row+1)*sqrt(2*col+1)
                let sqrt_2col_p1 = ((2 * col + 1) as f32).sqrt();
                a_flat[idx] = -(sqrt_2row_p1 * sqrt_2col_p1);
            } else if col == row {
                // Diagonal: A[n, n] = -(n+1)
                a_flat[idx] = -((row + 1) as f32);
            }
            // col > row: zero (already initialised)
        }
    }

    Ok((a_flat, b))
}

// ─── HiPPO-LegS diagonal ─────────────────────────────────────────────────────

/// Extract the diagonal of the HiPPO-LegS A matrix.
///
/// Returns `A[i, i] = −(i + 1)` for `i ∈ 0..n`.
///
/// # Errors
///
/// [`MambaError::InvalidSsmOrder`] if `n == 0`.
pub fn hippo_legs_diag(n: usize) -> MambaResult<Vec<f32>> {
    if n == 0 {
        return Err(MambaError::InvalidSsmOrder(0));
    }
    Ok((0..n).map(|i| -((i + 1) as f32)).collect())
}

// ─── HiPPO-LegS NPLR decomposition ──────────────────────────────────────────

/// Compute the NPLR (Normal Plus Low Rank) decomposition of HiPPO-LegS.
///
/// Returns `(lambda, p, q)` where:
/// - `lambda[n] = −(n + 0.5)` — the diagonal eigenvalue correction
/// - `p[n] = √(n + 0.5)` — the left rank-1 vector
/// - `q[n] = p[n]` — the right rank-1 vector (symmetric for HiPPO-LegS)
///
/// The full A matrix can be reconstructed as `A[i,j] = lambda[i]·δ_{ij} − p[i]·q[j]`.
///
/// # Errors
///
/// [`MambaError::InvalidSsmOrder`] if `n == 0`.
pub fn hippo_nplr(n: usize) -> MambaResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if n == 0 {
        return Err(MambaError::InvalidSsmOrder(0));
    }

    let mut lambda = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);

    for i in 0..n {
        let half_i = i as f32 + 0.5_f32;
        lambda.push(-half_i);
        let pq_val = half_i.sqrt();
        p.push(pq_val);
        q.push(pq_val);
    }

    Ok((lambda, p, q))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── hippo_legs basic cases ────────────────────────────────────────────────

    /// n=1: A=[-1.0], B=[1.0]
    #[test]
    fn hippo_legs_n1() {
        let (a, b) = hippo_legs(1).expect("n=1 is valid");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert!(
            (a[0] - (-1.0_f32)).abs() < EPS,
            "A[0,0] should be -1, got {}",
            a[0]
        );
        assert!(
            (b[0] - 1.0_f32).abs() < EPS,
            "B[0] should be 1, got {}",
            b[0]
        );
    }

    /// n=2: check all four A entries and both B entries.
    ///
    /// A[0,0] = -(0+1) = -1
    /// A[1,0] = -sqrt(3)*sqrt(1) = -sqrt(3)
    /// A[0,1] = 0  (upper triangular)
    /// A[1,1] = -(1+1) = -2
    /// B[0] = sqrt(1) = 1
    /// B[1] = sqrt(3)
    #[test]
    fn hippo_legs_n2() {
        let (a, b) = hippo_legs(2).expect("n=2 is valid");
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 2);

        // Diagonal
        assert!(
            (a[0] - (-1.0_f32)).abs() < EPS,
            "A[0,0]={} expected -1",
            a[0]
        );
        assert!(
            (a[3] - (-2.0_f32)).abs() < EPS,
            "A[1,1]={} expected -2",
            a[3]
        );
        // Lower triangular
        let expected_a10 = -(3.0_f32).sqrt();
        assert!(
            (a[2] - expected_a10).abs() < EPS,
            "A[1,0]={} expected -sqrt(3)={}",
            a[2],
            expected_a10
        );
        // Upper triangular (should be 0)
        assert!((a[1]).abs() < EPS, "A[0,1]={} should be 0", a[1]);
        // B
        assert!(
            (b[0] - 1.0_f32).abs() < EPS,
            "B[0] should be 1, got {}",
            b[0]
        );
        assert!(
            (b[1] - (3.0_f32).sqrt()).abs() < EPS,
            "B[1] should be sqrt(3), got {}",
            b[1]
        );
    }

    /// n=0 returns an error.
    #[test]
    fn hippo_legs_zero_error() {
        let err = hippo_legs(0).expect_err("n=0 should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    /// Diagonal of A[n,n] must equal -(n+1) for all n.
    #[test]
    fn hippo_legs_diagonal() {
        let n = 8;
        let (a, _) = hippo_legs(n).expect("valid n");
        for i in 0..n {
            let diag = a[i * n + i];
            let expected = -((i + 1) as f32);
            assert!(
                (diag - expected).abs() < EPS,
                "A[{i},{i}]={diag} expected {expected}"
            );
        }
    }

    /// Upper-triangular entries of A (row < col) must be zero.
    #[test]
    fn hippo_legs_lower_triangular() {
        let n = 6;
        let (a, _) = hippo_legs(n).expect("valid n");
        for row in 0..n {
            for col in (row + 1)..n {
                let val = a[row * n + col];
                assert!(
                    val.abs() < EPS,
                    "A[{row},{col}]={val} should be 0 (upper-triangular)"
                );
            }
        }
    }

    /// All B entries must be strictly positive.
    #[test]
    fn hippo_legs_b_positive() {
        let n = 12;
        let (_, b) = hippo_legs(n).expect("valid n");
        for (i, &v) in b.iter().enumerate() {
            assert!(v > 0.0, "B[{i}]={v} should be positive");
        }
    }

    /// n=4: all A and B entries finite.
    #[test]
    fn hippo_legs_n4_finite() {
        let (a, b) = hippo_legs(4).expect("valid n");
        for (i, &v) in a.iter().enumerate() {
            assert!(v.is_finite(), "A_flat[{i}]={v} not finite");
        }
        for (i, &v) in b.iter().enumerate() {
            assert!(v.is_finite(), "B[{i}]={v} not finite");
        }
    }

    // ── hippo_legs_diag ───────────────────────────────────────────────────────

    /// hippo_legs_diag returns -(i+1) for each i.
    #[test]
    fn hippo_diag_values() {
        let n = 10;
        let diag = hippo_legs_diag(n).expect("valid n");
        assert_eq!(diag.len(), n);
        for (i, &v) in diag.iter().enumerate() {
            let expected = -((i + 1) as f32);
            assert!(
                (v - expected).abs() < EPS,
                "diag[{i}]={v} expected {expected}"
            );
        }
    }

    /// hippo_legs_diag consistency with hippo_legs full matrix.
    #[test]
    fn hippo_diag_matches_full_matrix() {
        let n = 7;
        let (a_flat, _) = hippo_legs(n).expect("valid n");
        let diag = hippo_legs_diag(n).expect("valid n");
        for i in 0..n {
            assert!(
                (a_flat[i * n + i] - diag[i]).abs() < EPS,
                "diag mismatch at i={i}: full={}, diag={}",
                a_flat[i * n + i],
                diag[i]
            );
        }
    }

    // ── hippo_nplr ────────────────────────────────────────────────────────────

    /// n=4: all NPLR components finite.
    #[test]
    fn hippo_nplr_n4_finite() {
        let (lambda, p, q) = hippo_nplr(4).expect("valid n");
        for (i, &v) in lambda.iter().enumerate() {
            assert!(v.is_finite(), "lambda[{i}]={v} not finite");
        }
        for (i, &v) in p.iter().enumerate() {
            assert!(v.is_finite(), "p[{i}]={v} not finite");
        }
        for (i, &v) in q.iter().enumerate() {
            assert!(v.is_finite(), "q[{i}]={v} not finite");
        }
    }

    /// p[n] = sqrt(n + 0.5) > 0 for all n.
    #[test]
    fn hippo_nplr_p_positive() {
        let n = 16;
        let (_, p, _) = hippo_nplr(n).expect("valid n");
        for (i, &v) in p.iter().enumerate() {
            assert!(v > 0.0, "p[{i}]={v} should be positive");
            let expected = (i as f32 + 0.5_f32).sqrt();
            assert!(
                (v - expected).abs() < EPS,
                "p[{i}]={v} expected sqrt({:.1})={}",
                i as f32 + 0.5,
                expected
            );
        }
    }

    /// lambda[n] = -(n + 0.5) < 0 for all n.
    #[test]
    fn hippo_nplr_lambda_negative() {
        let n = 16;
        let (lambda, _, _) = hippo_nplr(n).expect("valid n");
        for (i, &v) in lambda.iter().enumerate() {
            assert!(v < 0.0, "lambda[{i}]={v} should be negative");
            let expected = -(i as f32 + 0.5_f32);
            assert!(
                (v - expected).abs() < EPS,
                "lambda[{i}]={v} expected {}",
                expected
            );
        }
    }

    /// n=64 produces finite output for all functions.
    #[test]
    fn hippo_legs_large_n() {
        let n = 64;
        let (a, b) = hippo_legs(n).expect("n=64 is valid");
        assert_eq!(a.len(), n * n);
        assert_eq!(b.len(), n);
        assert!(a.iter().all(|v| v.is_finite()), "A not all finite for n=64");
        assert!(b.iter().all(|v| v.is_finite()), "B not all finite for n=64");
        let (lambda, p, q) = hippo_nplr(n).expect("n=64 is valid");
        assert!(lambda.iter().all(|v| v.is_finite()));
        assert!(p.iter().all(|v| v.is_finite()));
        assert!(q.iter().all(|v| v.is_finite()));
    }
}
