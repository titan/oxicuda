use crate::handle::LcgRng;

/// DARE-style random pruning of a weight delta.
///
/// Each element of `delta` is kept with probability `density` and scaled by `1/density`
/// to preserve the expected magnitude; elements not kept are set to zero.
///
/// `density` must be in (0, 1]. Returns a pruned delta of the same length.
#[must_use]
pub fn dare_prune(delta: &[f32], density: f32, rng: &mut LcgRng) -> Vec<f32> {
    let density = density.clamp(1e-6, 1.0);
    let scale = 1.0 / density;
    delta
        .iter()
        .map(|&v| {
            let u = rng.next_f32();
            if u < density { v * scale } else { 0.0 }
        })
        .collect()
}

/// Compute the per-element sign consensus across multiple delta vectors.
///
/// Returns `+1`, `-1`, or `0` (as `i8`) for each position based on the sign of the sum.
/// All slices must have the same length.
#[must_use]
pub fn sign_consensus(deltas: &[&[f32]]) -> Vec<i8> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let n = deltas[0].len();
    let mut sums = vec![0.0_f32; n];
    for &delta in deltas {
        for (s, &v) in sums.iter_mut().zip(delta.iter()) {
            *s += v;
        }
    }
    sums.iter()
        .map(|&s| {
            if s > 0.0 {
                1_i8
            } else if s < 0.0 {
                -1_i8
            } else {
                0_i8
            }
        })
        .collect()
}

/// Compute the weighted sum `Σ_i w_i · delta_i` of multiple delta vectors.
///
/// All slices in `deltas` must have equal length. `weights` must have the same length as `deltas`.
/// Returns a vector of the same length as each delta.
#[must_use]
pub fn weighted_sum(deltas: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let n = deltas[0].len();
    let mut result = vec![0.0_f32; n];
    for (&w, &delta) in weights.iter().zip(deltas.iter()) {
        for (r, &v) in result.iter_mut().zip(delta.iter()) {
            *r += w * v;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dare_drop_and_rescale_exact() {
        // DARE keeps each entry with probability `density`, rescales survivors by
        // 1/density, and zeros the rest. Reconstruct the exact keep/scale decision
        // from an independent RNG seeded identically and require bit-exact agreement,
        // pinning both the 1/density rescale and the zeroing.
        let delta = vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let density = 0.5_f32;
        let scale = 1.0_f32 / density;
        let seed = 123_u64;

        let pruned = dare_prune(&delta, density, &mut LcgRng::new(seed));
        assert_eq!(pruned.len(), delta.len());

        let mut ref_rng = LcgRng::new(seed);
        for (&v, &p) in delta.iter().zip(pruned.iter()) {
            let u = ref_rng.next_f32();
            let expected = if u < density { v * scale } else { 0.0 };
            assert_eq!(p, expected, "DARE entry mismatch");
            if p != 0.0 {
                assert_eq!(p, v * scale, "survivor not rescaled by 1/density");
            }
        }
    }

    #[test]
    fn dare_rescale_unbiased_in_expectation() {
        // The 1/density rescale makes DARE an unbiased estimator of the original:
        // the mean of the pruned vector approximates the constant input value.
        let value = 2.0_f32;
        let density = 0.5_f32;
        let delta = vec![value; 16_384];
        let pruned = dare_prune(&delta, density, &mut LcgRng::new(7));
        let mean = pruned.iter().sum::<f32>() / pruned.len() as f32;
        assert!(
            (mean - value).abs() < 0.2,
            "DARE mean {mean} not within 0.2 of expected {value}"
        );
    }
}
