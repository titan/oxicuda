//! Generic accuracy metrics for sketch validation.

use crate::error::{SketchError, SketchResult};

/// Relative error: |estimate - truth| / max(|truth|, eps).
#[must_use]
pub fn relative_error(estimate: f64, truth: f64) -> f64 {
    let denom = truth.abs().max(1.0e-300);
    (estimate - truth).abs() / denom
}

/// Absolute error.
#[must_use]
pub fn absolute_error(estimate: f64, truth: f64) -> f64 {
    (estimate - truth).abs()
}

/// Accuracy = 1 - relative_error (clamped to [0, 1]).
#[must_use]
pub fn accuracy(estimate: f64, truth: f64) -> f64 {
    (1.0 - relative_error(estimate, truth)).clamp(0.0, 1.0)
}

/// Mean absolute error across paired estimate/truth slices.
pub fn mean_absolute_error(estimates: &[f64], truths: &[f64]) -> SketchResult<f64> {
    if estimates.len() != truths.len() {
        return Err(SketchError::DimensionMismatch {
            a: estimates.len(),
            b: truths.len(),
        });
    }
    if estimates.is_empty() {
        return Ok(0.0);
    }
    let s: f64 = estimates
        .iter()
        .zip(truths)
        .map(|(a, b)| (a - b).abs())
        .sum();
    Ok(s / estimates.len() as f64)
}

/// Recall-at-k: fraction of items in `truth_top_k` (size at most k) recovered in `est_top_k`.
pub fn recall_at_k(est_top_k: &[u64], truth_top_k: &[u64], k: usize) -> f64 {
    if k == 0 || truth_top_k.is_empty() {
        return 0.0;
    }
    use std::collections::BTreeSet;
    let est_set: BTreeSet<u64> = est_top_k.iter().take(k).copied().collect();
    let truth_set: BTreeSet<u64> = truth_top_k.iter().take(k).copied().collect();
    let inter = est_set.intersection(&truth_set).count();
    inter as f64 / truth_set.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relerr_zero_when_match() {
        assert!(relative_error(5.0, 5.0).abs() < 1e-12);
    }

    #[test]
    fn abserr_basic() {
        assert!((absolute_error(3.0, 5.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn accuracy_full_match() {
        assert!((accuracy(5.0, 5.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mae_dim_mismatch_err() {
        assert!(mean_absolute_error(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn mae_basic() {
        let mae = mean_absolute_error(&[1.0, 2.0, 3.0], &[1.5, 2.0, 5.0]).expect("ok");
        // |0.5| + 0 + |2| = 2.5; mean = 2.5 / 3.
        assert!((mae - (2.5 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn recall_basic() {
        let est = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2, 9, 8, 7];
        let r = recall_at_k(&est, &truth, 5);
        // Common items: {1, 2}; truth has 5; recall = 2/5.
        assert!((r - 0.4).abs() < 1e-12);
    }
}
