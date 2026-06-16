//! Statistical anomaly detection: MAD detector, Z-score detector,
//! and percentile threshold estimation.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the median of a slice (does not require pre-sorted input).
fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    } else {
        sorted[n / 2]
    }
}

// ─── MadDetector ─────────────────────────────────────────────────────────────

/// Median Absolute Deviation anomaly detector.
///
/// Score = `max_j |x_j − med_j| / (MAD_j + ε)`.
pub struct MadDetector {
    medians: Vec<f32>,
    mads: Vec<f32>,
    n_features: usize,
}

impl MadDetector {
    /// Create an unfitted detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            medians: Vec::new(),
            mads: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit: compute per-feature medians and MADs.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        self.n_features = n_features;
        let mut medians = Vec::with_capacity(n_features);
        let mut mads = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col: Vec<f32> = (0..n_samples).map(|i| data[i * n_features + j]).collect();
            let med = median(&col);
            let abs_devs: Vec<f32> = col.iter().map(|v| (v - med).abs()).collect();
            let mad = median(&abs_devs);
            medians.push(med);
            mads.push(mad);
        }

        self.medians = medians;
        self.mads = mads;
        Ok(())
    }

    /// Score: max normalised MAD score across features.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let max_score = x
            .iter()
            .zip(self.medians.iter())
            .zip(self.mads.iter())
            .map(|((xi, med), mad)| (xi - med).abs() / (mad + 1e-8))
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(max_score)
    }

    /// Batch scoring; `x` is `[n * n_features]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }
}

impl Default for MadDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ZScoreDetector ───────────────────────────────────────────────────────────

/// Z-score anomaly detector.
///
/// Score = `max_j |x_j − μ_j| / (σ_j + ε)`.
pub struct ZScoreDetector {
    mean: Vec<f32>,
    std: Vec<f32>,
    n_features: usize,
}

impl ZScoreDetector {
    /// Create an unfitted detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            std: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit: compute per-feature means and standard deviations.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples < 2 {
            return Err(AnomalyError::InsufficientSamples {
                need: 2,
                got: n_samples,
            });
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        self.n_features = n_features;
        let mut mean = vec![0.0_f32; n_features];
        let mut var = vec![0.0_f32; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += data[i * n_features + j];
            }
        }
        let inv_n = 1.0 / n_samples as f32;
        for v in &mut mean {
            *v *= inv_n;
        }

        for i in 0..n_samples {
            for j in 0..n_features {
                let diff = data[i * n_features + j] - mean[j];
                var[j] += diff * diff;
            }
        }
        let denom = (n_samples - 1) as f32;
        let std: Vec<f32> = var.iter().map(|v| (v / denom).sqrt()).collect();

        self.mean = mean;
        self.std = std;
        Ok(())
    }

    /// Score: max z-score across features.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if self.n_features == 0 {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let max_z = x
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((xi, mu), sigma)| (xi - mu).abs() / (sigma + 1e-8))
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(max_z)
    }

    /// Batch scoring; `x` is `[n * n_features]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }
}

impl Default for ZScoreDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── percentile_threshold ─────────────────────────────────────────────────────

/// Compute the `percentile`-th percentile threshold of a score distribution.
///
/// `percentile` must be in `[0.0, 100.0]`.  Uses linear interpolation
/// between the two adjacent sorted values.
pub fn percentile_threshold(scores: &[f32], percentile: f32) -> AnomalyResult<f32> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(AnomalyError::InvalidThresholdPercentile { p: percentile });
    }
    if scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }
    let mut sorted = scores.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    let idx_f = percentile / 100.0 * (n - 1) as f32;
    let lo = idx_f.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx_f - lo as f32;
    Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AnomalyResult;

    #[test]
    fn mad_detector_basic() -> AnomalyResult<()> {
        let data: Vec<f32> = (0..20_usize)
            .flat_map(|i| vec![i as f32, (i * 2) as f32])
            .collect();
        let mut det = MadDetector::new();
        det.fit(&data, 20, 2)?;
        let s = det.score(&[10.0_f32, 20.0])?;
        assert!(s.is_finite(), "s={s}");
        Ok(())
    }

    #[test]
    fn zscore_detector_outlier() -> AnomalyResult<()> {
        let data: Vec<f32> = (0..20_usize).map(|i| i as f32).collect();
        let mut det = ZScoreDetector::new();
        det.fit(&data, 20, 1)?;
        let s_normal = det.score(&[10.0_f32])?;
        let s_outlier = det.score(&[1000.0_f32])?;
        assert!(s_outlier > s_normal, "{s_outlier} > {s_normal}");
        Ok(())
    }

    #[test]
    fn percentile_threshold_50() -> AnomalyResult<()> {
        let scores = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let med = percentile_threshold(&scores, 50.0)?;
        assert!((med - 3.0).abs() < 1e-5, "med={med}");
        Ok(())
    }

    #[test]
    fn percentile_threshold_100() -> AnomalyResult<()> {
        let scores = vec![1.0_f32, 2.0, 5.0, 3.0];
        let max = percentile_threshold(&scores, 100.0)?;
        assert!((max - 5.0).abs() < 1e-5, "max={max}");
        Ok(())
    }
}
