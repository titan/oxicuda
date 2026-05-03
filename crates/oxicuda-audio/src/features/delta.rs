//! Delta and delta-delta (acceleration) features for audio sequences.
//!
//! Appends first- and second-order finite-difference features to a log-mel
//! (or any frame-level) feature matrix. The central-difference formula with
//! a `window` of `N` is:
//!
//! ```text
//! δ[t] = Σ_{n=1}^{N} n * (x[t+n] - x[t-n]) / (2 * Σ_{n=1}^{N} n²)
//! ```
//!
//! Frames at the boundary are replicated (edge-padding).

use crate::error::{AudioError, AudioResult};

/// Compute delta features for a `[T, F]` feature matrix.
///
/// The window is the number of adjacent frames on each side. Boundary frames
/// are replicated (edge-pad). Returns a new `Vec<f32>` of shape `[T, F]`.
///
/// # Errors
///
/// Returns `AudioError::InvalidSequenceLength` if `t == 0`,
/// `AudioError::InvalidNumMels` if `f == 0`.
pub fn compute_delta(features: &[f32], t: usize, f: usize, window: usize) -> AudioResult<Vec<f32>> {
    if t == 0 {
        return Err(AudioError::InvalidSequenceLength(t));
    }
    if f == 0 {
        return Err(AudioError::InvalidNumMels(f));
    }
    let win = window.max(1);

    // Denominator: 2 * Σ_{n=1}^{win} n²
    let denom: f32 = 2.0 * (1..=win).map(|n| (n * n) as f32).sum::<f32>();

    let mut out = vec![0.0f32; t * f];

    for frame in 0..t {
        for dim in 0..f {
            let mut acc = 0.0f32;
            for n in 1..=win {
                let ahead = if frame + n < t { frame + n } else { t - 1 };
                let behind = frame.saturating_sub(n);
                acc += n as f32 * (features[ahead * f + dim] - features[behind * f + dim]);
            }
            out[frame * f + dim] = acc / denom;
        }
    }
    Ok(out)
}

/// Compute delta-delta (second-order) features by applying `compute_delta`
/// twice.
///
/// Returns `[T, F]` shaped output — the double-derivative of the input.
///
/// # Errors
///
/// Propagates errors from `compute_delta`.
pub fn compute_delta_delta(
    features: &[f32],
    t: usize,
    f: usize,
    window: usize,
) -> AudioResult<Vec<f32>> {
    let delta = compute_delta(features, t, f, window)?;
    compute_delta(&delta, t, f, window)
}

/// Stack original, delta, and delta-delta features horizontally.
///
/// Returns a new `Vec<f32>` of shape `[T, 3*F]`. Useful for feeding to
/// ASR encoders that consume the full feature stack.
///
/// # Errors
///
/// Propagates errors from `compute_delta`.
pub fn stack_delta_features(
    features: &[f32],
    t: usize,
    f: usize,
    window: usize,
) -> AudioResult<Vec<f32>> {
    let delta = compute_delta(features, t, f, window)?;
    let delta2 = compute_delta(&delta, t, f, window)?;

    let mut out = vec![0.0f32; t * 3 * f];
    for frame in 0..t {
        for dim in 0..f {
            out[frame * 3 * f + dim] = features[frame * f + dim];
            out[frame * 3 * f + f + dim] = delta[frame * f + dim];
            out[frame * 3 * f + 2 * f + dim] = delta2[frame * f + dim];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_ramp_is_constant() {
        // x[t, d] = t · c  → δ[t, d] ≈ c (central diff over any window)
        let t = 20;
        let f = 4;
        let c = 2.0f32;
        let features: Vec<f32> = (0..t * f).map(|i| c * (i / f) as f32).collect();
        let delta = compute_delta(&features, t, f, 2).expect("ok");
        // Interior frames should be very close to c
        for frame in 3..t - 3 {
            for dim in 0..f {
                let d = delta[frame * f + dim];
                assert!((d - c).abs() < 1e-3, "frame={frame} dim={dim} d={d}");
            }
        }
    }

    #[test]
    fn delta_constant_is_zero() {
        let t = 10;
        let f = 3;
        let features = vec![5.0f32; t * f];
        let delta = compute_delta(&features, t, f, 2).expect("ok");
        assert!(delta.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn delta_delta_constant_is_zero() {
        let t = 15;
        let f = 5;
        let features = vec![1.0f32; t * f];
        let dd = compute_delta_delta(&features, t, f, 2).expect("ok");
        assert!(dd.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn delta_output_shape() {
        let t = 8;
        let f = 6;
        let features = vec![0.0f32; t * f];
        let delta = compute_delta(&features, t, f, 1).expect("ok");
        assert_eq!(delta.len(), t * f);
    }

    #[test]
    fn stack_delta_features_shape() {
        let t = 10;
        let f = 4;
        let features = vec![1.0f32; t * f];
        let stacked = stack_delta_features(&features, t, f, 1).expect("ok");
        assert_eq!(stacked.len(), t * 3 * f);
    }

    #[test]
    fn stack_delta_first_band_equals_original() {
        let t = 5;
        let f = 3;
        let features: Vec<f32> = (0..t * f).map(|i| i as f32).collect();
        let stacked = stack_delta_features(&features, t, f, 1).expect("ok");
        for frame in 0..t {
            for dim in 0..f {
                assert_eq!(stacked[frame * 3 * f + dim], features[frame * f + dim]);
            }
        }
    }

    #[test]
    fn delta_zero_t_error() {
        assert!(compute_delta(&[], 0, 4, 2).is_err());
    }

    #[test]
    fn delta_zero_f_error() {
        assert!(compute_delta(&[], 4, 0, 2).is_err());
    }

    #[test]
    fn delta_window_zero_treated_as_one() {
        let t = 5;
        let f = 2;
        let features: Vec<f32> = (0..t * f).map(|i| i as f32 * 0.5).collect();
        let d = compute_delta(&features, t, f, 0).expect("ok");
        assert_eq!(d.len(), t * f);
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn delta_boundary_is_finite() {
        let t = 3;
        let f = 2;
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = compute_delta(&features, t, f, 2).expect("ok");
        assert!(d.iter().all(|v| v.is_finite()));
    }
}
