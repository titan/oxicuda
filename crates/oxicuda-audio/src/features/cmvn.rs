//! Cepstral Mean and Variance Normalisation (CMVN).
//!
//! Standardises log-mel (or any frame-level) features to zero mean and unit
//! variance on a per-channel (per-mel-bin) basis. The normalised value is
//! `y = (x - mean) * inv_std` where `inv_std = 1 / std`.

use crate::error::{AudioError, AudioResult};

/// Pre-computed CMVN statistics for a single corpus or utterance.
///
/// Lengths of `mean` and `inv_std` must equal the number of feature
/// dimensions (mel bins) of the input.
#[derive(Debug, Clone)]
pub struct CmvnConfig {
    /// Per-channel mean (length `F`).
    pub mean: Vec<f32>,
    /// Per-channel inverse standard deviation (length `F`).
    pub inv_std: Vec<f32>,
}

impl CmvnConfig {
    /// Construct CMVN statistics from raw mean and standard deviation vectors.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::ShapeMismatch` if lengths differ, or
    /// `AudioError::EmptyInput` if vectors are empty.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> AudioResult<Self> {
        if mean.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "mean is empty".into(),
            });
        }
        if mean.len() != std.len() {
            return Err(AudioError::ShapeMismatch {
                msg: format!("mean length {} != std length {}", mean.len(), std.len()),
            });
        }
        let inv_std: Vec<f32> = std.iter().map(|&s| 1.0 / s.max(1e-10)).collect();
        Ok(Self { mean, inv_std })
    }

    /// Number of feature dimensions.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.mean.len()
    }
}

/// Compute per-channel mean and standard deviation from a `[T, F]` feature matrix.
///
/// Returns `(mean, std)` each of length `F`. Uses Bessel-corrected variance
/// when `T > 1`, otherwise returns std = 1.
///
/// # Errors
///
/// Returns `AudioError::InvalidNumMels` if `f == 0`, or
/// `AudioError::InvalidSequenceLength` if `t == 0`.
pub fn compute_cmvn(features: &[f32], t: usize, f: usize) -> AudioResult<CmvnConfig> {
    if f == 0 {
        return Err(AudioError::InvalidNumMels(f));
    }
    if t == 0 {
        return Err(AudioError::InvalidSequenceLength(t));
    }

    let mut mean = vec![0.0f32; f];
    let mut var = vec![0.0f32; f];

    for frame in 0..t {
        for dim in 0..f {
            mean[dim] += features[frame * f + dim];
        }
    }
    let scale = 1.0 / t as f32;
    for m in mean.iter_mut() {
        *m *= scale;
    }

    for frame in 0..t {
        for dim in 0..f {
            let diff = features[frame * f + dim] - mean[dim];
            var[dim] += diff * diff;
        }
    }
    let denom = if t > 1 { (t - 1) as f32 } else { 1.0 };
    let std: Vec<f32> = var.iter().map(|&v| (v / denom).sqrt().max(1e-10)).collect();

    CmvnConfig::new(mean, std)
}

/// Apply CMVN statistics to a `[T, F]` feature matrix in-place.
///
/// # Errors
///
/// Returns `AudioError::DimensionMismatch` if the feature width does not
/// match `config.dim()`.
pub fn apply_cmvn(
    features: &mut [f32],
    t: usize,
    f: usize,
    config: &CmvnConfig,
) -> AudioResult<()> {
    if f != config.dim() {
        return Err(AudioError::DimensionMismatch {
            expected: config.dim(),
            got: f,
        });
    }
    for frame in 0..t {
        for dim in 0..f {
            let idx = frame * f + dim;
            features[idx] = (features[idx] - config.mean[dim]) * config.inv_std[dim];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        if n == 1 {
            return vec![start];
        }
        (0..n)
            .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
            .collect()
    }

    #[test]
    fn cmvn_config_new_ok() {
        let cfg = CmvnConfig::new(vec![1.0, 2.0], vec![0.5, 0.5]).expect("ok");
        assert_eq!(cfg.dim(), 2);
        assert!((cfg.inv_std[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn cmvn_config_empty_error() {
        assert!(CmvnConfig::new(vec![], vec![]).is_err());
    }

    #[test]
    fn cmvn_config_shape_mismatch() {
        let r = CmvnConfig::new(vec![1.0, 2.0], vec![1.0]);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn compute_cmvn_basic() {
        let t = 100;
        let f = 4;
        // Constant features → std = ~0 (but clamped), mean = constant
        let features = vec![3.0f32; t * f];
        let cfg = compute_cmvn(&features, t, f).expect("ok");
        assert_eq!(cfg.dim(), f);
        for m in &cfg.mean {
            assert!((*m - 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn compute_cmvn_zero_f_error() {
        assert!(compute_cmvn(&[], 10, 0).is_err());
    }

    #[test]
    fn compute_cmvn_zero_t_error() {
        assert!(compute_cmvn(&[], 0, 4).is_err());
    }

    #[test]
    fn apply_cmvn_normalises() {
        let t = 50;
        let f = 8;
        let mut features = linspace(-10.0, 10.0, t * f);
        let cfg = compute_cmvn(&features, t, f).expect("cmvn ok");
        apply_cmvn(&mut features, t, f, &cfg).expect("apply ok");

        let mut sum = 0.0f32;
        for frame in 0..t {
            for dim in 0..f {
                sum += features[frame * f + dim];
            }
        }
        let mean_after = sum / (t * f) as f32;
        assert!(mean_after.abs() < 1e-3, "mean after CMVN = {mean_after}");
    }

    #[test]
    fn apply_cmvn_dim_mismatch() {
        let cfg = CmvnConfig::new(vec![0.0f32; 4], vec![1.0f32; 4]).expect("ok");
        let mut features = vec![1.0f32; 10 * 8];
        let r = apply_cmvn(&mut features, 10, 8, &cfg);
        assert!(matches!(
            r.unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn cmvn_round_trip_unit_variance() {
        let t = 40;
        let f = 6;
        let mut features: Vec<f32> = (0..t * f).map(|i| (i as f32) * 0.1 - 2.0).collect();
        let cfg = compute_cmvn(&features, t, f).expect("ok");
        apply_cmvn(&mut features, t, f, &cfg).expect("ok");

        let mut sum_sq = 0.0f32;
        for frame in 0..t {
            for dim in 0..f {
                sum_sq += features[frame * f + dim].powi(2);
            }
        }
        let var = sum_sq / ((t - 1) * f) as f32;
        assert!((var - 1.0).abs() < 0.05, "variance ≈ 1, got {var}");
    }
}
