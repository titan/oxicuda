//! Anomaly ensemble: combine multiple detector scores.
//!
//! Scores from each detector are normalised to `[0, 1]` via min-max scaling
//! from the training distribution, then aggregated by the chosen method.

#![allow(clippy::module_inception)]

use crate::error::{AnomalyError, AnomalyResult};

// ─── EnsembleMethod ───────────────────────────────────────────────────────────

/// Strategy for combining normalised detector scores.
pub enum EnsembleMethod {
    /// Unweighted average of all detector scores.
    Average,
    /// Maximum over all detector scores.
    Maximum,
    /// Weighted average; weights should sum to 1 (normalised internally).
    Weighted(Vec<f32>),
}

// ─── AnomalyEnsemble ─────────────────────────────────────────────────────────

/// Ensemble anomaly detector combining `n_detectors` individual scores.
pub struct AnomalyEnsemble {
    score_min: Vec<f32>,
    score_max: Vec<f32>,
    method: EnsembleMethod,
    n_detectors: usize,
}

impl AnomalyEnsemble {
    /// Construct an unfitted ensemble.
    #[must_use]
    pub fn new(method: EnsembleMethod, n_detectors: usize) -> Self {
        Self {
            score_min: vec![0.0_f32; n_detectors],
            score_max: vec![1.0_f32; n_detectors],
            method,
            n_detectors,
        }
    }

    /// Fit min-max normalisation from training scores.
    ///
    /// `train_scores` is `[n_samples * n_detectors]` row-major
    /// (each row = scores from all detectors for one sample).
    pub fn fit(&mut self, train_scores: &[f32], n_samples: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if train_scores.len() != n_samples * self.n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * self.n_detectors,
                got: train_scores.len(),
            });
        }

        let mut min_v = vec![f32::INFINITY; self.n_detectors];
        let mut max_v = vec![f32::NEG_INFINITY; self.n_detectors];

        for i in 0..n_samples {
            for d in 0..self.n_detectors {
                let v = train_scores[i * self.n_detectors + d];
                if v < min_v[d] {
                    min_v[d] = v;
                }
                if v > max_v[d] {
                    max_v[d] = v;
                }
            }
        }

        // Avoid degenerate range
        for d in 0..self.n_detectors {
            if (max_v[d] - min_v[d]).abs() < 1e-8 {
                max_v[d] = min_v[d] + 1.0;
            }
        }

        self.score_min = min_v;
        self.score_max = max_v;
        Ok(())
    }

    /// Combine scores from a single sample into one ensemble score ∈ `[0, 1]`.
    ///
    /// `scores` is `[n_detectors]`.
    pub fn combine(&self, scores: &[f32]) -> AnomalyResult<f32> {
        if scores.len() != self.n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.n_detectors,
                got: scores.len(),
            });
        }

        // Normalise each detector score to [0, 1]
        let normed: Vec<f32> = scores
            .iter()
            .zip(self.score_min.iter())
            .zip(self.score_max.iter())
            .map(|((s, mn), mx)| ((s - mn) / (mx - mn)).clamp(0.0, 1.0))
            .collect();

        let result = match &self.method {
            EnsembleMethod::Average => normed.iter().sum::<f32>() / self.n_detectors as f32,
            EnsembleMethod::Maximum => normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            EnsembleMethod::Weighted(w) => {
                if w.len() != self.n_detectors {
                    return Err(AnomalyError::DimensionMismatch {
                        expected: self.n_detectors,
                        got: w.len(),
                    });
                }
                let weight_sum: f32 = w.iter().sum();
                let effective_sum = if weight_sum.abs() < 1e-8 {
                    1.0
                } else {
                    weight_sum
                };
                normed
                    .iter()
                    .zip(w.iter())
                    .map(|(s, wi)| s * wi)
                    .sum::<f32>()
                    / effective_sum
            }
        };

        Ok(result.clamp(0.0, 1.0))
    }

    /// Batch combine; `scores` is `[n * n_detectors]`; returns `[n]`.
    pub fn combine_batch(&self, scores: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if scores.len() != n * self.n_detectors {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_detectors,
                got: scores.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &scores[i * self.n_detectors..(i + 1) * self.n_detectors];
            out.push(self.combine(row)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_average_basic() {
        let mut ens = AnomalyEnsemble::new(EnsembleMethod::Average, 2);
        let train = vec![0.0_f32, 0.0, 1.0, 1.0, 0.5, 0.5];
        ens.fit(&train, 3).unwrap();
        let s = ens.combine(&[0.5_f32, 0.5]).unwrap();
        assert!((0.0..=1.0).contains(&s), "s={s}");
    }

    #[test]
    fn ensemble_maximum_basic() {
        let mut ens = AnomalyEnsemble::new(EnsembleMethod::Maximum, 2);
        let train = vec![0.0_f32, 0.0, 1.0, 1.0];
        ens.fit(&train, 2).unwrap();
        let s = ens.combine(&[0.3_f32, 0.9]).unwrap();
        assert!((s - 0.9).abs() < 0.01, "s={s}");
    }
}
