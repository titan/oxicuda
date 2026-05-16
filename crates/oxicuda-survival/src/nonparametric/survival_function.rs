//! Step-function representation of a survival curve `S(t)`.

use crate::error::{SurvivalError, SurvivalResult};

/// Right-continuous step function `S(t)` with breakpoints at `times[i]`.
#[derive(Debug, Clone)]
pub struct SurvivalFunction {
    pub times: Vec<f64>,
    pub survival: Vec<f64>,
}

impl SurvivalFunction {
    /// Build a new `SurvivalFunction` from sorted times and survival values.
    pub fn new(times: Vec<f64>, survival: Vec<f64>) -> SurvivalResult<Self> {
        if times.len() != survival.len() {
            return Err(SurvivalError::ShapeMismatch {
                expected: vec![times.len()],
                got: vec![survival.len()],
            });
        }
        for w in times.windows(2) {
            if w[1] < w[0] {
                return Err(SurvivalError::InvalidParameter(
                    "times must be non-decreasing".to_string(),
                ));
            }
        }
        for s in &survival {
            if !s.is_finite() || *s < 0.0 || *s > 1.0 + 1.0e-9 {
                return Err(SurvivalError::InvalidParameter(format!(
                    "survival probability out of [0,1]: {s}"
                )));
            }
        }
        Ok(Self { times, survival })
    }

    /// Evaluate `S(t)` for arbitrary t (right-continuous step interpolation).
    /// Returns 1.0 for `t < times[0]`.
    #[must_use]
    pub fn eval(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 1.0;
        }
        if t < self.times[0] {
            return 1.0;
        }
        // last i with times[i] <= t
        let mut s = self.survival[0];
        for i in 0..self.times.len() {
            if self.times[i] <= t {
                s = self.survival[i];
            } else {
                break;
            }
        }
        s
    }

    /// Number of breakpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_before_first_is_one() {
        let s = SurvivalFunction::new(vec![1.0, 2.0], vec![0.5, 0.25]).expect("ok");
        assert_eq!(s.eval(0.0), 1.0);
        assert_eq!(s.eval(1.0), 0.5);
        assert_eq!(s.eval(1.5), 0.5);
        assert_eq!(s.eval(2.0), 0.25);
        assert_eq!(s.eval(10.0), 0.25);
    }

    #[test]
    fn rejects_decreasing_times() {
        let s = SurvivalFunction::new(vec![2.0, 1.0], vec![0.5, 0.25]);
        assert!(s.is_err());
    }

    #[test]
    fn rejects_invalid_probability() {
        let s = SurvivalFunction::new(vec![1.0], vec![1.5]);
        assert!(s.is_err());
    }
}
