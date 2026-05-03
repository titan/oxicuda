//! PowerSGD: Low-rank gradient compression via power iteration.
//!
//! Vogels et al., "PowerSGD: Practical Low-Rank Gradient Compression for
//! Distributed Optimization", NeurIPS 2019.
//!
//! Approximates a gradient matrix M [m×n] by a rank-r product P@Q^T,
//! using randomized power iteration and modified Gram-Schmidt orthogonalization.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// PowerSGD compression configuration.
#[derive(Debug, Clone, Copy)]
pub struct PowerSgdCompressor {
    /// Rank r of the approximation (must be < min(m, n)).
    pub rank: usize,
    /// Number of power iteration steps (2-4 typical).
    pub n_power_iter: usize,
}

impl PowerSgdCompressor {
    /// Create a validated compressor.
    ///
    /// # Errors
    /// Returns `InvalidRank` if rank == 0.
    pub fn new(rank: usize, n_power_iter: usize) -> FedResult<Self> {
        if rank == 0 {
            return Err(FedError::InvalidRank { rank: 0, dim: 1 });
        }
        Ok(Self {
            rank,
            n_power_iter: n_power_iter.max(1),
        })
    }
}

/// Matrix-vector multiply: C = A @ B, where A is [m×k], B is [k×n], C is [m×n].
/// All matrices in row-major order.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Transpose a matrix [m×n] → [n×m] in row-major order.
fn transpose(mat: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * m];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = mat[i * n + j];
        }
    }
    out
}

/// Modified Gram-Schmidt orthogonalization of columns of a [m×r] matrix.
/// Modifies in-place.
fn gram_schmidt(mat: &mut [f32], m: usize, r: usize) {
    for j in 0..r {
        // Compute column j norm
        let mut norm_sq = 0.0_f32;
        for i in 0..m {
            let v = mat[i * r + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(1e-10);
        for i in 0..m {
            mat[i * r + j] /= norm;
        }
        // Subtract projections from subsequent columns
        for k in (j + 1)..r {
            let mut dot = 0.0_f32;
            for i in 0..m {
                dot += mat[i * r + j] * mat[i * r + k];
            }
            for i in 0..m {
                mat[i * r + k] -= dot * mat[i * r + j];
            }
        }
    }
}

/// Compress a matrix M [m×n] to a low-rank approximation (P [m×r], Q [n×r]).
///
/// Algorithm:
/// 1. Initialize Q randomly [n×r]
/// 2. For `n_power_iter` iterations:
///    a. P = M @ Q, orthogonalise P (Gram-Schmidt)
///    b. Q = M^T @ P, orthogonalise Q
/// 3. Final P = M @ Q (not orthogonalised)
///
/// Reconstruction: M ≈ P @ Q^T.
///
/// # Errors
/// Returns `InvalidRank` if rank ≥ min(m, n), `DimensionMismatch` if
/// `matrix.len() != m * n`, or `Internal` if m or n is 0.
pub fn compress(
    matrix: &[f32],
    m: usize,
    n: usize,
    compressor: &PowerSgdCompressor,
    rng: &mut LcgRng,
) -> FedResult<(Vec<f32>, Vec<f32>)> {
    if m == 0 || n == 0 {
        return Err(FedError::Internal(
            "matrix dimensions must be positive".into(),
        ));
    }
    if matrix.len() != m * n {
        return Err(FedError::DimensionMismatch {
            expected: m * n,
            got: matrix.len(),
        });
    }
    let r = compressor.rank;
    let min_dim = m.min(n);
    if r >= min_dim {
        return Err(FedError::InvalidRank {
            rank: r,
            dim: min_dim,
        });
    }

    // Initialize Q [n×r] with random values
    let mut q = (0..n * r)
        .map(|_| rng.next_f32() - 0.5)
        .collect::<Vec<f32>>();

    // Power iteration
    for _ in 0..compressor.n_power_iter {
        // P = M @ Q  [m×r]
        let mut p = matmul(matrix, &q, m, n, r);
        gram_schmidt(&mut p, m, r);

        // Q = M^T @ P  [n×r]
        let mt = transpose(matrix, m, n);
        q = matmul(&mt, &p, n, m, r);
        gram_schmidt(&mut q, n, r);
    }

    // Final P = M @ Q  [m×r] (not orthogonalised — captures the actual projection)
    let p = matmul(matrix, &q, m, n, r);

    Ok((p, q))
}

/// Decompress: M_approx = P @ Q^T, where P is [m×r] and Q is [n×r].
///
/// # Errors
/// Returns `DimensionMismatch` if slice lengths don't match m×r and n×r.
pub fn decompress(p: &[f32], q: &[f32], m: usize, n: usize, rank: usize) -> FedResult<Vec<f32>> {
    if p.len() != m * rank {
        return Err(FedError::DimensionMismatch {
            expected: m * rank,
            got: p.len(),
        });
    }
    if q.len() != n * rank {
        return Err(FedError::DimensionMismatch {
            expected: n * rank,
            got: q.len(),
        });
    }
    // M_approx = P @ Q^T  [m×n] = matmul(P [m×r], Q^T [r×n])
    let qt = transpose(q, n, rank); // [r×n]
    Ok(matmul(p, &qt, m, rank, n))
}

/// Compute the residual error: `residual = original - reconstructed`.
///
/// Used for error feedback: on the next round, the client sends
/// `gradient + residual` to correct for compression loss.
#[must_use]
pub fn residual(original: &[f32], reconstructed: &[f32]) -> Vec<f32> {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| o - r)
        .collect()
}

/// Compute the Frobenius norm of a matrix.
#[must_use]
pub fn frobenius_norm(matrix: &[f32]) -> f32 {
    matrix.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powersgd_compressor_invalid_rank() {
        assert!(matches!(
            PowerSgdCompressor::new(0, 2),
            Err(FedError::InvalidRank { .. })
        ));
    }

    #[test]
    fn powersgd_compress_decompress_rank1_exact() {
        // A rank-1 matrix: M = u * v^T, rank-1 approx should be exact
        let m = 4;
        let n = 3;
        let u = [1.0f32, 2.0, 3.0, 4.0]; // [4×1]
        let v = [1.0f32, 0.5, 0.25]; // [3×1]
        // M = outer product u * v^T  [4×3]
        let matrix: Vec<f32> = u
            .iter()
            .flat_map(|&ui| v.iter().map(move |&vj| ui * vj))
            .collect();

        let compressor = PowerSgdCompressor::new(1, 3).expect("test invariant: valid compressor");
        let mut rng = LcgRng::new(42);
        let (p, q) =
            compress(&matrix, m, n, &compressor, &mut rng).expect("test invariant: valid compress");
        let reconstructed = decompress(&p, &q, m, n, 1).expect("test invariant: valid decompress");

        // Frobenius error should be small for rank-1 input
        let err = frobenius_norm(&residual(&matrix, &reconstructed));
        let orig_norm = frobenius_norm(&matrix);
        assert!(
            err / orig_norm < 0.05,
            "rank-1 approx of rank-1 matrix should be near-exact, err/norm={:.4}",
            err / orig_norm
        );
    }

    #[test]
    fn powersgd_rank_exceeds_min_dim_error() {
        let m = 3;
        let n = 2;
        let matrix = vec![1.0f32; m * n];
        let compressor = PowerSgdCompressor::new(2, 2).expect("test invariant: valid compressor");
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            compress(&matrix, m, n, &compressor, &mut rng),
            Err(FedError::InvalidRank { .. })
        ));
    }

    #[test]
    fn powersgd_decompress_dimension_mismatch() {
        let p = vec![1.0f32, 2.0, 3.0]; // [3×1]
        let q = vec![1.0f32, 2.0]; // [2×1]
        // Correct dims: p [3×1], q [2×1], but passing wrong sizes
        let wrong_p = vec![1.0f32, 2.0]; // wrong length
        assert!(matches!(
            decompress(&wrong_p, &q, 3, 2, 1),
            Err(FedError::DimensionMismatch { .. })
        ));
        let _ = (p, q); // suppress unused warning
    }

    #[test]
    fn residual_zero_for_perfect_decompress() {
        let orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let res = residual(&orig, &orig);
        assert!(res.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn matmul_correctness() {
        // [2×2] @ [2×2]
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![3.0f32, 4.0, 5.0, 6.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert!((c[0] - 3.0).abs() < 1e-6 && (c[1] - 4.0).abs() < 1e-6);
    }
}
