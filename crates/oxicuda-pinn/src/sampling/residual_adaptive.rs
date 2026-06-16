//! Importance-weighted adaptive residual sampling.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Importance-weighted sampling with probability ∝ |residual|^power.
///
/// # Arguments
/// - `candidates`: flat `[n_candidates × d]` array of candidate points.
/// - `residuals`: `[n_candidates]` residual values at each candidate.
/// - `n_candidates`: number of candidate points.
/// - `d`: dimensionality of each point.
/// - `n_sample`: number of points to draw.
/// - `power`: concentration exponent (`power=2` → squared residuals).
/// - `rng`: random number generator.
///
/// Returns flat `[n_sample × d]` selected points.
pub fn residual_adaptive_sample(
    candidates: &[f32],
    residuals: &[f32],
    n_candidates: usize,
    d: usize,
    n_sample: usize,
    power: f32,
    rng: &mut LcgRng,
) -> PinnResult<Vec<f32>> {
    if n_candidates == 0 || candidates.is_empty() {
        return Err(PinnError::EmptyCollocationSet);
    }
    if candidates.len() != n_candidates * d {
        return Err(PinnError::DimensionMismatch {
            expected: n_candidates * d,
            got: candidates.len(),
        });
    }
    if residuals.len() != n_candidates {
        return Err(PinnError::DimensionMismatch {
            expected: n_candidates,
            got: residuals.len(),
        });
    }
    if n_sample == 0 {
        return Ok(Vec::new());
    }

    // Compute weights ∝ |residual|^power
    let weights: Vec<f32> = residuals.iter().map(|&r| r.abs().powf(power)).collect();
    let total: f32 = weights.iter().sum();

    // If all weights are zero, fall back to uniform
    let probs: Vec<f32> = if total < 1e-30 {
        vec![1.0 / n_candidates as f32; n_candidates]
    } else {
        weights.iter().map(|&w| w / total).collect()
    };

    // Build CDF
    let mut cdf = vec![0.0_f32; n_candidates + 1];
    for i in 0..n_candidates {
        cdf[i + 1] = cdf[i] + probs[i];
    }
    // Ensure CDF ends at exactly 1
    cdf[n_candidates] = 1.0;

    // Sample via inverse CDF (binary search)
    let mut output = vec![0.0_f32; n_sample * d];
    for s in 0..n_sample {
        let u = rng.next_f32();
        // Binary search for u in CDF
        let idx = cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n_candidates - 1);
        let src_start = idx * d;
        let dst_start = s * d;
        output[dst_start..dst_start + d].copy_from_slice(&candidates[src_start..src_start + d]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_correct_output_shape() {
        let candidates: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let residuals = vec![1.0_f32; 10];
        let mut rng = LcgRng::new(1);
        let out = residual_adaptive_sample(&candidates, &residuals, 10, 1, 5, 2.0, &mut rng)
            .expect("residual adaptive sample with valid inputs should succeed");
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn sample_2d_output_shape() {
        let candidates: Vec<f32> = (0..20).map(|i| i as f32).collect(); // 10 × 2
        let residuals = vec![0.5_f32; 10];
        let mut rng = LcgRng::new(2);
        let out = residual_adaptive_sample(&candidates, &residuals, 10, 2, 4, 1.0, &mut rng)
            .expect(
                "2D residual adaptive sample with uniform residuals and valid args should succeed",
            );
        assert_eq!(out.len(), 8); // 4 × 2
    }

    #[test]
    fn high_residual_more_likely_selected() {
        // One point has residual=10, others have residual=0.01
        let n = 10;
        let mut candidates = vec![0.0_f32; n];
        let mut residuals = vec![0.01_f32; n];
        candidates[5] = 99.0; // special point
        residuals[5] = 10.0; // high residual
        let mut rng = LcgRng::new(42);
        let out =
            residual_adaptive_sample(&candidates, &residuals, n, 1, 100, 2.0, &mut rng).expect("residual adaptive sample with skewed residuals (one high, rest low) should succeed");
        let count_special = out.iter().filter(|&&v| (v - 99.0).abs() < 1e-5).count();
        assert!(
            count_special > 50,
            "High-residual point should dominate: count={count_special}"
        );
    }

    #[test]
    fn zero_residuals_uniform_fallback() {
        let candidates: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let residuals = vec![0.0_f32; 5];
        let mut rng = LcgRng::new(3);
        let out =
            residual_adaptive_sample(&candidates, &residuals, 5, 1, 10, 2.0, &mut rng).expect("residual adaptive sample with all-zero residuals should fall back to uniform sampling");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_candidates_error() {
        let mut rng = LcgRng::new(4);
        let result = residual_adaptive_sample(&[], &[], 0, 1, 5, 1.0, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn zero_n_sample_returns_empty() {
        let candidates = vec![1.0_f32; 5];
        let residuals = vec![1.0_f32; 5];
        let mut rng = LcgRng::new(5);
        let out = residual_adaptive_sample(&candidates, &residuals, 5, 1, 0, 1.0, &mut rng)
            .expect("residual adaptive sample with n_sample=0 should return an empty vec");
        assert!(out.is_empty());
    }

    #[test]
    fn output_points_from_candidates() {
        // All output points must be values that appear in candidates
        let candidates: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residuals = vec![1.0_f32; 5];
        let mut rng = LcgRng::new(6);
        let out = residual_adaptive_sample(&candidates, &residuals, 5, 1, 20, 1.0, &mut rng)
            .expect("residual adaptive sample from small known candidate set should succeed");
        for &v in &out {
            assert!(
                candidates.iter().any(|&c| (c - v).abs() < 1e-5),
                "Sampled value {v} not in candidates"
            );
        }
    }
}
