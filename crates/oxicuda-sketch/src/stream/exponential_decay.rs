//! Exponential-decay weighted streaming statistics.
//!
//! Maintain a running weighted mean / sum that exponentially de-emphasises old values:
//! `S_n = (1 - alpha) * S_{n-1} + alpha * x_n` for some decay factor `alpha ∈ (0, 1]`.

use crate::error::{SketchError, SketchResult};

/// Exponential-decay statistics.
#[derive(Debug, Clone)]
pub struct ExponentialDecay {
    pub alpha: f64,
    pub mean: f64,
    pub weight_sum: f64,
    pub initialised: bool,
}

impl ExponentialDecay {
    /// New decay accumulator with smoothing parameter `alpha ∈ (0, 1]`.
    pub fn new(alpha: f64) -> SketchResult<Self> {
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be in (0, 1]".to_string(),
            });
        }
        Ok(Self {
            alpha,
            mean: 0.0,
            weight_sum: 0.0,
            initialised: false,
        })
    }

    /// Add an observation.
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if !self.initialised {
            self.mean = x;
            self.weight_sum = 1.0;
            self.initialised = true;
            return;
        }
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * x;
        self.weight_sum = (1.0 - self.alpha) * self.weight_sum + self.alpha;
    }

    /// Current exponentially-smoothed mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Effective accumulated weight.
    #[must_use]
    pub fn effective_weight(&self) -> f64 {
        self.weight_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_invalid_alpha() {
        assert!(ExponentialDecay::new(0.0).is_err());
        assert!(ExponentialDecay::new(2.0).is_err());
    }

    #[test]
    fn decay_constant_input_stays_constant() {
        let mut d = ExponentialDecay::new(0.1).expect("ok");
        for _ in 0..1000 {
            d.add(7.0);
        }
        assert!((d.mean() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn decay_responds_to_change() {
        let mut d = ExponentialDecay::new(0.5).expect("ok");
        d.add(0.0);
        d.add(10.0);
        // After one step at alpha=0.5: mean = 5.
        assert!((d.mean() - 5.0).abs() < 1e-9);
    }
}
