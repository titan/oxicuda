//! Temporal statistics pooling.
//!
//! Aggregates a `[T, C]` feature tensor over the time axis, producing a
//! `[2 * C]` vector whose first `C` elements are the per-channel mean and
//! whose second `C` elements are the per-channel standard deviation.
//!
//! This two-pass approach (mean first, variance second) avoids catastrophic
//! cancellation in the one-pass Welford formula for small `T`.  The variance
//! is Bessel-corrected when `T > 1`, and the standard deviation is clamped
//! to a minimum of `1e-10` to prevent division-by-zero in downstream layers.

use crate::error::{AudioError, AudioResult};

// ─── Public API ──────────────────────────────────────────────────────────────

/// Temporal statistics pooling: concatenate per-channel mean and std over time.
///
/// `features` — `[T, C]` row-major tensor.
/// Returns `[2 * C]` = `[mean_0, …, mean_{C-1}, std_0, …, std_{C-1}]`.
///
/// Variance is Bessel-corrected when `T > 1`; otherwise the denominator is 1.
/// Output standard deviations are clamped to a minimum of `1e-10`.
///
/// # Errors
///
/// - [`AudioError::InvalidSequenceLength`] when `t == 0`.
/// - [`AudioError::InvalidEmbedDim`] when `c == 0`.
/// - [`AudioError::DimensionMismatch`] when `features.len() != t * c`.
pub fn stats_pool(features: &[f32], t: usize, c: usize) -> AudioResult<Vec<f32>> {
    if t == 0 {
        return Err(AudioError::InvalidSequenceLength(0));
    }
    if c == 0 {
        return Err(AudioError::InvalidEmbedDim(0));
    }
    let expected = t * c;
    if features.len() != expected {
        return Err(AudioError::DimensionMismatch {
            expected,
            got: features.len(),
        });
    }

    // ── Pass 1: compute per-channel mean ────────────────────────────────────
    let mut mean = vec![0.0_f32; c];
    for frame in 0..t {
        let row = &features[frame * c..(frame + 1) * c];
        for (ch, &val) in row.iter().enumerate() {
            mean[ch] += val;
        }
    }
    let inv_t = 1.0 / t as f32;
    for m in &mut mean {
        *m *= inv_t;
    }

    // ── Pass 2: compute per-channel variance (Bessel-corrected) ─────────────
    let mut var = vec![0.0_f32; c];
    for frame in 0..t {
        let row = &features[frame * c..(frame + 1) * c];
        for (ch, &val) in row.iter().enumerate() {
            let diff = val - mean[ch];
            var[ch] += diff * diff;
        }
    }
    // Bessel correction: divide by (T - 1) when T > 1, else T = 1 → denominator = 1.
    let denom = if t > 1 { (t - 1) as f32 } else { 1.0_f32 };
    let min_std: f32 = 1e-10;

    let mut output = Vec::with_capacity(2 * c);
    output.extend_from_slice(&mean);
    for v in &var {
        let std = (v / denom).sqrt().max(min_std);
        output.push(std);
    }

    Ok(output)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_pool_output_shape() {
        let t = 10_usize;
        let c = 8_usize;
        let features = vec![1.0_f32; t * c];
        let out = stats_pool(&features, t, c).expect("ok");
        assert_eq!(out.len(), 2 * c);
    }

    #[test]
    fn stats_pool_constant_zero_std() {
        // All frames constant → std should be clamped to min_std.
        let t = 5_usize;
        let c = 3_usize;
        let val = 3.7_f32;
        let features = vec![val; t * c];
        let out = stats_pool(&features, t, c).expect("ok");
        for ch in 0..c {
            assert!((out[ch] - val).abs() < 1e-6, "mean mismatch ch={ch}");
            assert!(
                (out[c + ch] - 1e-10_f32).abs() < 1e-12,
                "std should be clamped ch={ch}"
            );
        }
    }

    #[test]
    fn stats_pool_known_values() {
        // 3 frames, 2 channels
        // frame 0: [1, 2], frame 1: [3, 4], frame 2: [5, 6]
        // mean_0 = (1+3+5)/3 = 3.0  mean_1 = (2+4+6)/3 = 4.0
        // var_0 (Bessel): ((1-3)^2+(3-3)^2+(5-3)^2) / 2 = (4+0+4)/2 = 4 → std=2
        // var_1 (Bessel): ((2-4)^2+(4-4)^2+(6-4)^2) / 2 = 4 → std=2
        let features = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = stats_pool(&features, 3, 2).expect("ok");
        assert!((out[0] - 3.0).abs() < 1e-5, "mean_0={}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-5, "mean_1={}", out[1]);
        assert!((out[2] - 2.0).abs() < 1e-5, "std_0={}", out[2]);
        assert!((out[3] - 2.0).abs() < 1e-5, "std_1={}", out[3]);
    }

    #[test]
    fn stats_pool_zero_t_error() {
        let features = vec![1.0_f32; 4];
        let err = stats_pool(&features, 0, 4).unwrap_err();
        assert!(matches!(err, AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn stats_pool_zero_c_error() {
        let features = vec![1.0_f32; 4];
        let err = stats_pool(&features, 4, 0).unwrap_err();
        assert!(matches!(err, AudioError::InvalidEmbedDim(0)));
    }

    #[test]
    fn stats_pool_single_frame() {
        // T=1: Bessel denom=1 → var = (x - mean)^2 / 1 = 0 → std clamped to 1e-10.
        let features = vec![2.0_f32, 5.0];
        let out = stats_pool(&features, 1, 2).expect("ok");
        assert_eq!(out.len(), 4);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
        // std should be clamped since single-frame variance = 0
        assert!((out[2] - 1e-10_f32).abs() < 1e-12);
        assert!((out[3] - 1e-10_f32).abs() < 1e-12);
    }

    #[test]
    fn stats_pool_output_finite() {
        use crate::handle::LcgRng;
        let t = 20_usize;
        let c = 16_usize;
        let mut rng = LcgRng::new(42);
        let mut features = vec![0.0_f32; t * c];
        rng.fill_normal(&mut features);
        let out = stats_pool(&features, t, c).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite value in output"
        );
    }

    #[test]
    fn stats_pool_mean_correct() {
        // Linearly increasing values: verify mean computation is accurate.
        let t = 4_usize;
        let c = 1_usize;
        let features = vec![1.0_f32, 3.0, 5.0, 7.0]; // mean = 4.0
        let out = stats_pool(&features, t, c).expect("ok");
        assert!((out[0] - 4.0).abs() < 1e-5, "mean={}", out[0]);
    }

    #[test]
    fn stats_pool_dim_mismatch_error() {
        let features = vec![1.0_f32; 5];
        let err = stats_pool(&features, 3, 2).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }
}
