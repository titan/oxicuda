//! Single survival observation: a `(time, event)` pair.

use crate::error::{SurvivalError, SurvivalResult};

/// A single survival observation.
///
/// - `time`: follow-up time (must be >= 0).
/// - `event`: `true` if the event of interest was observed (uncensored),
///   `false` for right-censored observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub time: f64,
    pub event: bool,
}

impl Observation {
    /// Construct an `Observation`, validating that `time >= 0`.
    pub fn new(time: f64, event: bool) -> SurvivalResult<Self> {
        if !time.is_finite() || time < 0.0 {
            return Err(SurvivalError::NegativeTime(time));
        }
        Ok(Self { time, event })
    }

    /// Whether this is a censored (non-event) observation.
    #[must_use]
    pub fn is_censored(&self) -> bool {
        !self.event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_construct_valid() {
        let o = Observation::new(5.0, true).expect("ok");
        assert_eq!(o.time, 5.0);
        assert!(o.event);
        assert!(!o.is_censored());
    }

    #[test]
    fn observation_rejects_negative_time() {
        assert!(Observation::new(-1.0, true).is_err());
    }

    #[test]
    fn observation_rejects_nan() {
        assert!(Observation::new(f64::NAN, true).is_err());
    }

    #[test]
    fn observation_zero_time_ok() {
        let o = Observation::new(0.0, false).expect("ok");
        assert!(o.is_censored());
    }
}
