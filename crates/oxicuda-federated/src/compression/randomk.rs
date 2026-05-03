//! Random-k gradient sparsification (unbiased estimator).
//!
//! Stich et al., "Sparsified SGD with Memory", NeurIPS 2018.
//!
//! Selects k elements uniformly at random and scales them by n/k to
//! form an unbiased estimator of the full gradient.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Compress a gradient by selecting k elements uniformly at random.
///
/// Returns a sparse vector where the selected k elements are scaled by
/// `n / k` (unbiased estimator), and all others are 0.
///
/// # Arguments
/// - `gradient` — input gradient of length n
/// - `k` — number of elements to keep (must be in [1, n])
/// - `rng` — random number generator for index selection
///
/// # Errors
/// Returns `InsufficientClients` if k == 0, or `DimensionMismatch` if k > n.
pub fn random_sparsify(gradient: &[f32], k: usize, rng: &mut LcgRng) -> FedResult<Vec<f32>> {
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

    // Build index list and shuffle to get k random indices
    let mut indices: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut indices);
    let selected: std::collections::HashSet<usize> = indices[..k].iter().cloned().collect();

    // Scale factor for unbiased estimation
    let scale = n as f32 / k as f32;

    let sparse: Vec<f32> = gradient
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            if selected.contains(&i) {
                g * scale
            } else {
                0.0
            }
        })
        .collect();

    Ok(sparse)
}

/// Decompress a random-k sparse gradient (unscale and return dense vector).
///
/// Since the encoding already stores the scaled values in dense format,
/// this function is the identity for API symmetry.
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

/// Estimate the compression ratio (sparsity).
#[must_use]
pub fn compression_ratio(n: usize, k: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    k as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_sparsify_count() {
        let grad = vec![1.0f32; 20];
        let mut rng = LcgRng::new(42);
        let sparse =
            random_sparsify(&grad, 5, &mut rng).expect("test invariant: valid random sparsify");
        let nonzero = sparse.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 5, "should have exactly k=5 non-zero elements");
    }

    #[test]
    fn random_sparsify_unbiased() {
        // Mean of many sparse estimates should equal original gradient
        let n = 100;
        let grad: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let k = 10;
        let n_trials = 5_000;
        let mut rng = LcgRng::new(7);
        let mut sum = vec![0.0f32; n];
        for _ in 0..n_trials {
            let sparse = random_sparsify(&grad, k, &mut rng)
                .expect("test invariant: valid random sparsify trial");
            for (s, &v) in sum.iter_mut().zip(sparse.iter()) {
                *s += v;
            }
        }
        // Average should be close to original gradient
        let mean: Vec<f32> = sum.iter().map(|&s| s / n_trials as f32).collect();
        for (m, &g) in mean.iter().zip(grad.iter()) {
            assert!(
                (m - g).abs() < 0.15,
                "unbiased estimate mean={m} should be close to grad={g}"
            );
        }
    }

    #[test]
    fn random_sparsify_k_zero_error() {
        let grad = vec![1.0f32, 2.0];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            random_sparsify(&grad, 0, &mut rng),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn random_sparsify_k_exceeds_n() {
        let grad = vec![1.0f32, 2.0];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            random_sparsify(&grad, 5, &mut rng),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn compression_ratio_value() {
        assert!((compression_ratio(100, 10) - 0.1).abs() < 1e-6);
        assert!((compression_ratio(100, 100) - 1.0).abs() < 1e-6);
        assert!((compression_ratio(0, 0) - 0.0).abs() < 1e-6);
    }
}
