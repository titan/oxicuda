//! Counting-process intervals `(start, stop, event, x(t))` for time-varying Cox.

use crate::error::{SurvivalError, SurvivalResult};

/// One row of the counting-process representation.
#[derive(Debug, Clone)]
pub struct CountingInterval {
    pub subject_id: usize,
    pub start: f64,
    pub stop: f64,
    pub event: bool,
    pub covariates: Vec<f64>,
}

impl CountingInterval {
    /// Construct, validating `0 <= start < stop`.
    pub fn new(
        subject_id: usize,
        start: f64,
        stop: f64,
        event: bool,
        covariates: Vec<f64>,
    ) -> SurvivalResult<Self> {
        if !start.is_finite() || !stop.is_finite() {
            return Err(SurvivalError::InvalidParameter(
                "non-finite interval bounds".to_string(),
            ));
        }
        if start < 0.0 {
            return Err(SurvivalError::NegativeTime(start));
        }
        if stop <= start {
            return Err(SurvivalError::InvalidParameter(
                "stop must exceed start".to_string(),
            ));
        }
        for v in &covariates {
            if !v.is_finite() {
                return Err(SurvivalError::InvalidParameter(
                    "non-finite covariate".to_string(),
                ));
            }
        }
        Ok(Self {
            subject_id,
            start,
            stop,
            event,
            covariates,
        })
    }
}

/// Collection of counting-process intervals.
#[derive(Debug, Clone)]
pub struct CountingProcessDataset {
    pub intervals: Vec<CountingInterval>,
}

impl CountingProcessDataset {
    /// Construct, requiring at least one interval and consistent covariate dimensions.
    pub fn new(intervals: Vec<CountingInterval>) -> SurvivalResult<Self> {
        if intervals.is_empty() {
            return Err(SurvivalError::EmptyDataset);
        }
        let p = intervals[0].covariates.len();
        for iv in &intervals {
            if iv.covariates.len() != p {
                return Err(SurvivalError::ShapeMismatch {
                    expected: vec![p],
                    got: vec![iv.covariates.len()],
                });
            }
        }
        Ok(Self { intervals })
    }

    /// Number of intervals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Number of covariates per interval.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.intervals
            .first()
            .map(|iv| iv.covariates.len())
            .unwrap_or(0)
    }

    /// Number of intervals where an event occurred.
    #[must_use]
    pub fn n_events(&self) -> usize {
        self.intervals.iter().filter(|iv| iv.event).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_new_valid() {
        let ci = CountingInterval::new(0, 0.0, 1.0, true, vec![1.0, 2.0]).expect("ok");
        assert_eq!(ci.subject_id, 0);
    }

    #[test]
    fn ci_rejects_zero_length() {
        assert!(CountingInterval::new(0, 1.0, 1.0, true, vec![]).is_err());
    }

    #[test]
    fn ci_rejects_neg_start() {
        assert!(CountingInterval::new(0, -1.0, 1.0, true, vec![]).is_err());
    }

    #[test]
    fn dataset_consistent_dims() {
        let a = CountingInterval::new(0, 0.0, 1.0, true, vec![1.0]).expect("ok");
        let b = CountingInterval::new(0, 1.0, 2.0, false, vec![2.0]).expect("ok");
        let d = CountingProcessDataset::new(vec![a, b]).expect("ok");
        assert_eq!(d.len(), 2);
        assert_eq!(d.n_features(), 1);
        assert_eq!(d.n_events(), 1);
    }
}
