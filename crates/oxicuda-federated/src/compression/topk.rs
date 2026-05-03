//! Top-k gradient sparsification with error feedback.
//!
//! Lin et al., "Deep Gradient Compression", ICLR 2018.
//!
//! Sends only the k largest-magnitude gradient elements each round,
//! and accumulates the error (residual) for subsequent rounds.

use crate::error::{FedError, FedResult};

/// Compress a gradient by keeping only the top-k elements by magnitude.
///
/// Returns `(sparse, threshold)` where:
/// - `sparse[i] = gradient[i]` if `|gradient[i]|` is in the top-k, else 0.0
/// - `threshold` is the k-th largest absolute value (used by receiver to reconstruct)
///
/// # Errors
/// Returns `InsufficientClients` if k == 0, or `DimensionMismatch` if k > n.
pub fn topk_sparsify(gradient: &[f32], k: usize) -> FedResult<(Vec<f32>, f32)> {
    if k == 0 {
        return Err(FedError::InsufficientClients { min: 1, got: 0 });
    }
    let n = gradient.len();
    if k > n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: k,
        });
    }

    // Compute absolute values and find k-th largest threshold
    let mut abs_vals: Vec<f32> = gradient.iter().map(|&g| g.abs()).collect();
    // Partial sort: put the k largest at the front
    abs_vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = abs_vals[k - 1];

    // Zero out elements below threshold
    let sparse: Vec<f32> = gradient
        .iter()
        .map(|&g| if g.abs() >= threshold { g } else { 0.0 })
        .collect();

    Ok((sparse, threshold))
}

/// Accumulate the sparsification error into an error buffer.
///
/// `error += gradient - sparse`
/// On the next round, the client sends `gradient_actual + error_buffer`.
///
/// # Errors
/// Returns `DimensionMismatch` if slice lengths differ.
pub fn error_feedback(error: &mut [f32], gradient: &[f32], sparse: &[f32]) -> FedResult<()> {
    let n = error.len();
    if gradient.len() != n || sparse.len() != n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: gradient.len().min(sparse.len()),
        });
    }
    for ((e, &g), &s) in error.iter_mut().zip(gradient.iter()).zip(sparse.iter()) {
        *e += g - s;
    }
    Ok(())
}

/// Decompress a sparse gradient (identity for top-k — just returns the sparse vector).
///
/// The sparse representation is already in dense format (zeros where not selected).
/// This function validates dimensions and is provided for API symmetry.
///
/// # Errors
/// Returns `DimensionMismatch` if lengths differ.
pub fn decompress(sparse: &[f32], expected_len: usize) -> FedResult<Vec<f32>> {
    if sparse.len() != expected_len {
        return Err(FedError::DimensionMismatch {
            expected: expected_len,
            got: sparse.len(),
        });
    }
    Ok(sparse.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_sparsify_basic() {
        let grad = vec![0.1f32, 0.5, 0.2, 0.9];
        let (sparse, threshold) = topk_sparsify(&grad, 2).expect("test invariant: valid topk");
        // Top 2 should be 0.5 and 0.9
        assert_eq!(sparse[0], 0.0, "0.1 should be zeroed");
        assert_ne!(sparse[1], 0.0, "0.5 should be kept");
        assert_eq!(sparse[2], 0.0, "0.2 should be zeroed");
        assert_ne!(sparse[3], 0.0, "0.9 should be kept");
        assert!(threshold >= 0.0);
    }

    #[test]
    fn topk_sparsify_k_zero_error() {
        let grad = vec![1.0f32, 2.0];
        assert!(matches!(
            topk_sparsify(&grad, 0),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn topk_sparsify_k_exceeds_n() {
        let grad = vec![1.0f32, 2.0];
        assert!(matches!(
            topk_sparsify(&grad, 5),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn topk_sparsify_preserves_signs() {
        let grad = vec![-0.9f32, 0.1, -0.5, 0.2];
        let (sparse, _) = topk_sparsify(&grad, 2).expect("test invariant: valid topk");
        assert!(sparse[0] < 0.0, "-0.9 should be kept with negative sign");
        assert!(sparse[2] < 0.0, "-0.5 should be kept with negative sign");
    }

    #[test]
    fn error_feedback_accumulates() {
        let gradient = vec![1.0f32, 0.5, 0.2, 0.9];
        let (sparse, _) = topk_sparsify(&gradient, 2).expect("test invariant: valid topk");
        let mut error = vec![0.0f32; 4];
        error_feedback(&mut error, &gradient, &sparse)
            .expect("test invariant: valid error feedback");
        // Zeroed elements should have their values in error
        assert!(error[0].abs() > 0.0 || error[2].abs() > 0.0);
    }

    #[test]
    fn decompress_identity() {
        let sparse = vec![0.0f32, 0.5, 0.0, 0.9];
        let out = decompress(&sparse, 4).expect("test invariant: valid decompress");
        assert_eq!(out, sparse);
    }

    #[test]
    fn decompress_dimension_mismatch() {
        let sparse = vec![0.0f32, 0.5];
        assert!(matches!(
            decompress(&sparse, 4),
            Err(FedError::DimensionMismatch { .. })
        ));
    }
}
