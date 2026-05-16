//! Dataset of survival observations with optional covariates and strata.

use crate::data::observation::Observation;
use crate::error::{SurvivalError, SurvivalResult};

/// A complete survival dataset: observations + optional covariates + optional strata.
///
/// `covariates[i]` is the feature vector for observation `i`, shape `(n, p)`.
/// `strata[i]` is the stratum index for observation `i`. When `None`, all observations share one stratum.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub observations: Vec<Observation>,
    pub covariates: Option<Vec<Vec<f64>>>,
    pub strata: Option<Vec<usize>>,
}

impl Dataset {
    /// Construct a dataset from observations, validating shapes.
    pub fn new(
        observations: Vec<Observation>,
        covariates: Option<Vec<Vec<f64>>>,
        strata: Option<Vec<usize>>,
    ) -> SurvivalResult<Self> {
        if observations.is_empty() {
            return Err(SurvivalError::EmptyDataset);
        }
        let n = observations.len();
        if let Some(c) = &covariates {
            if c.len() != n {
                return Err(SurvivalError::ShapeMismatch {
                    expected: vec![n],
                    got: vec![c.len()],
                });
            }
            if !c.is_empty() {
                let p = c[0].len();
                for (i, row) in c.iter().enumerate() {
                    if row.len() != p {
                        return Err(SurvivalError::ShapeMismatch {
                            expected: vec![p],
                            got: vec![row.len()],
                        });
                    }
                    if row.iter().any(|v| !v.is_finite()) {
                        return Err(SurvivalError::InvalidParameter(format!(
                            "non-finite covariate at row {i}"
                        )));
                    }
                }
            }
        }
        if let Some(s) = &strata {
            if s.len() != n {
                return Err(SurvivalError::ShapeMismatch {
                    expected: vec![n],
                    got: vec![s.len()],
                });
            }
        }
        Ok(Self {
            observations,
            covariates,
            strata,
        })
    }

    /// Convenience constructor from raw `(time, event)` vectors.
    pub fn from_arrays(times: &[f64], events: &[bool]) -> SurvivalResult<Self> {
        if times.len() != events.len() {
            return Err(SurvivalError::ShapeMismatch {
                expected: vec![times.len()],
                got: vec![events.len()],
            });
        }
        let mut obs = Vec::with_capacity(times.len());
        for (t, e) in times.iter().zip(events.iter()) {
            obs.push(Observation::new(*t, *e)?);
        }
        Self::new(obs, None, None)
    }

    /// Number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Number of covariates per observation (0 if no covariates).
    #[must_use]
    pub fn n_features(&self) -> usize {
        match &self.covariates {
            Some(c) if !c.is_empty() => c[0].len(),
            _ => 0,
        }
    }

    /// Total number of events (uncensored observations).
    #[must_use]
    pub fn n_events(&self) -> usize {
        self.observations.iter().filter(|o| o.event).count()
    }

    /// Vector of follow-up times.
    #[must_use]
    pub fn times(&self) -> Vec<f64> {
        self.observations.iter().map(|o| o.time).collect()
    }

    /// Vector of event indicators (as f64).
    #[must_use]
    pub fn events_f64(&self) -> Vec<f64> {
        self.observations
            .iter()
            .map(|o| if o.event { 1.0 } else { 0.0 })
            .collect()
    }

    /// Sorted indices in ascending time order (stable for ties).
    #[must_use]
    pub fn order_by_time(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len()).collect();
        idx.sort_by(|&a, &b| {
            self.observations[a]
                .time
                .partial_cmp(&self.observations[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(t: f64, e: bool) -> Observation {
        Observation::new(t, e).expect("ok")
    }

    #[test]
    fn dataset_from_arrays_ok() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true, false, true]).expect("ok");
        assert_eq!(d.len(), 3);
        assert_eq!(d.n_events(), 2);
        assert_eq!(d.n_features(), 0);
    }

    #[test]
    fn dataset_rejects_empty() {
        assert!(Dataset::new(vec![], None, None).is_err());
    }

    #[test]
    fn dataset_rejects_mismatched_strata() {
        let d = Dataset::new(vec![obs(1.0, true)], None, Some(vec![0, 1]));
        assert!(d.is_err());
    }

    #[test]
    fn dataset_covariates_validated() {
        let d = Dataset::new(
            vec![obs(1.0, true), obs(2.0, false)],
            Some(vec![vec![1.0, 2.0], vec![3.0]]),
            None,
        );
        assert!(d.is_err());
    }

    #[test]
    fn dataset_order_by_time_correct() {
        let d = Dataset::from_arrays(&[3.0, 1.0, 2.0], &[true, true, true]).expect("ok");
        assert_eq!(d.order_by_time(), vec![1, 2, 0]);
    }
}
