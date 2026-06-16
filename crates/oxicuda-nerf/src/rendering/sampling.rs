//! Ray sampling strategies: stratified sampling and importance resampling.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Stratified sampling ──────────────────────────────────────────────────────

/// Sample `n_samples` positions along a ray between `t_near` and `t_far`
/// using stratified (jittered) sampling.
///
/// `t_i = t_near + (i + U(0,1)) / n_samples * (t_far - t_near)` for i = 0..n_samples
///
/// # Errors
///
/// Returns `InvalidBounds` if `t_far <= t_near`,
/// `InvalidSampleCount` if `n_samples == 0`.
pub fn stratified_sample(
    t_near: f32,
    t_far: f32,
    n_samples: usize,
    rng: &mut LcgRng,
) -> NerfResult<Vec<f32>> {
    if t_far <= t_near {
        return Err(NerfError::InvalidBounds {
            near: t_near,
            far: t_far,
        });
    }
    if n_samples == 0 {
        return Err(NerfError::InvalidSampleCount { n: 0 });
    }

    let span = t_far - t_near;
    let inv_n = 1.0 / n_samples as f32;

    let t_vals: Vec<f32> = (0..n_samples)
        .map(|i| {
            let jitter = rng.next_f32();
            t_near + (i as f32 + jitter) * inv_n * span
        })
        .collect();

    Ok(t_vals)
}

// ─── Importance resampling ────────────────────────────────────────────────────

/// Draw `n_fine` sample positions using inverse-CDF sampling from coarse weights.
///
/// Implements hierarchical NeRF sampling:
/// 1. Build CDF from weights (with small ε for numerical stability, renormalize).
/// 2. Draw n_fine uniform samples u_j in [0, 1].
/// 3. Binary search for u_j in CDF to get t_low, t_high.
/// 4. Linearly interpolate for exact t position.
///
/// # Errors
///
/// Returns `DimensionMismatch` if `coarse_t.len() != weights.len()`,
/// `InvalidSampleCount` if `n_fine == 0` or coarse arrays are empty.
pub fn importance_sample(
    coarse_t: &[f32],
    weights: &[f32],
    n_fine: usize,
    rng: &mut LcgRng,
) -> NerfResult<Vec<f32>> {
    if coarse_t.len() != weights.len() {
        return Err(NerfError::DimensionMismatch {
            expected: coarse_t.len(),
            got: weights.len(),
        });
    }
    if coarse_t.is_empty() {
        return Err(NerfError::EmptyInput);
    }
    if n_fine == 0 {
        return Err(NerfError::InvalidSampleCount { n: 0 });
    }

    let n = weights.len();
    let eps = 1e-5_f32;

    // Build normalized CDF
    let w_sum: f32 = weights.iter().map(|&w| w.max(0.0) + eps).sum();
    let mut cdf = Vec::with_capacity(n + 1);
    cdf.push(0.0_f32);
    let mut running = 0.0_f32;
    for &w in weights {
        running += w.max(0.0) + eps;
        cdf.push(running / w_sum);
    }
    // Clamp last entry to exactly 1.0
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    // Draw n_fine samples via inverse CDF
    let t_vals: Vec<f32> = (0..n_fine)
        .map(|_| {
            let u = rng.next_f32();
            // Binary search for u in cdf[1..n+1]
            let idx = cdf
                .partition_point(|&c| c <= u)
                .saturating_sub(1)
                .min(n - 1);
            let c_lo = cdf[idx];
            let c_hi = cdf[idx + 1];
            let t_lo = coarse_t[idx];
            let t_hi = if idx + 1 < n {
                coarse_t[idx + 1]
            } else {
                coarse_t[idx]
            };

            // Linear interpolation
            let denom = c_hi - c_lo;
            if denom < 1e-10 {
                t_lo
            } else {
                t_lo + (u - c_lo) / denom * (t_hi - t_lo)
            }
        })
        .collect();

    Ok(t_vals)
}

// ─── Merge samples ────────────────────────────────────────────────────────────

/// Merge and sort coarse and fine sample positions, removing near-duplicates.
///
/// Returns a sorted, deduplicated `Vec<f32>`.
#[must_use]
pub fn merge_samples(coarse_t: &[f32], fine_t: &[f32]) -> Vec<f32> {
    let mut merged: Vec<f32> = coarse_t.iter().chain(fine_t.iter()).copied().collect();
    merged.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Deduplicate values within epsilon
    let mut dedup = Vec::with_capacity(merged.len());
    for t in merged {
        if dedup
            .last()
            .is_none_or(|&prev: &f32| (t - prev).abs() > 1e-7)
        {
            dedup.push(t);
        }
    }
    dedup
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_count() {
        let mut rng = LcgRng::new(42);
        let t =
            stratified_sample(0.0, 1.0, 64, &mut rng).expect("stratified_sample should succeed");
        assert_eq!(t.len(), 64);
    }

    #[test]
    fn stratified_in_bounds() {
        let mut rng = LcgRng::new(7);
        let t =
            stratified_sample(0.1, 5.0, 32, &mut rng).expect("stratified_sample should succeed");
        for &v in &t {
            assert!((0.1..=5.0).contains(&v), "t={v} out of bounds");
        }
    }

    #[test]
    fn importance_count() {
        let mut rng = LcgRng::new(99);
        let coarse_t = vec![0.1, 0.2, 0.3, 0.4];
        let weights = vec![0.1, 0.5, 0.3, 0.1];
        let fine = importance_sample(&coarse_t, &weights, 8, &mut rng)
            .expect("importance_sample should succeed");
        assert_eq!(fine.len(), 8);
    }

    #[test]
    fn merge_sorted() {
        let coarse = [0.1, 0.3, 0.5];
        let fine = [0.2, 0.4];
        let merged = merge_samples(&coarse, &fine);
        assert!(merged.windows(2).all(|w| w[0] <= w[1]));
    }
}
