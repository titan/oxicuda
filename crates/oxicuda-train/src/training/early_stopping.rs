//! Early stopping — halt training when a monitored metric stops improving.
//!
//! Early stopping is the most widely used regularisation heuristic for
//! iterative training: a validation metric is monitored each epoch, and once it
//! fails to improve for `patience` consecutive epochs (by at least `min_delta`)
//! training is terminated and the best-so-far weights are restored.  This
//! prevents the over-fitting that occurs if SGD is run past the point of
//! minimum validation error.
//!
//! The monitor supports both directions via [`EarlyStopMode`]:
//!
//! * [`EarlyStopMode::Min`] — lower is better (e.g. validation loss).
//! * [`EarlyStopMode::Max`] — higher is better (e.g. accuracy).
//!
//! A metric is treated as an *improvement* only if it beats the current best by
//! strictly more than `min_delta`, which avoids stopping (or, conversely,
//! resetting the patience counter) on numerical noise.

use crate::error::{TrainError, TrainResult};

// ─── Mode ─────────────────────────────────────────────────────────────────────

/// Optimisation direction of the monitored metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStopMode {
    /// Lower values are better (loss-like metrics).
    Min,
    /// Higher values are better (accuracy-like metrics).
    Max,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`EarlyStopping`].
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of consecutive non-improving epochs tolerated before stopping.
    pub patience: u64,
    /// Minimum change to qualify as an improvement (must be ≥ 0).
    pub min_delta: f64,
    /// Direction of improvement.
    pub mode: EarlyStopMode,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            mode: EarlyStopMode::Min,
        }
    }
}

impl EarlyStoppingConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] if `min_delta < 0` or NaN.
    fn validate(&self) -> TrainResult<()> {
        if self.min_delta < 0.0 || self.min_delta.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("min_delta must be non-negative, got {}", self.min_delta),
            });
        }
        Ok(())
    }
}

// ─── Monitor ──────────────────────────────────────────────────────────────────

/// Early-stopping metric monitor.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best: f64,
    /// Epoch index (0-based) at which the best metric was observed.
    best_epoch: u64,
    /// Consecutive epochs without improvement.
    bad_epochs: u64,
    /// Total `update` calls made.
    epoch: u64,
    stopped: bool,
}

impl EarlyStopping {
    /// Create an early-stopping monitor.
    ///
    /// # Errors
    ///
    /// Any error from validation of `config`.
    pub fn new(config: EarlyStoppingConfig) -> TrainResult<Self> {
        config.validate()?;
        let best = match config.mode {
            EarlyStopMode::Min => f64::INFINITY,
            EarlyStopMode::Max => f64::NEG_INFINITY,
        };
        Ok(Self {
            config,
            best,
            best_epoch: 0,
            bad_epochs: 0,
            epoch: 0,
            stopped: false,
        })
    }

    /// Whether `candidate` is a strict improvement over the current best by at
    /// least `min_delta`.
    #[inline]
    fn is_improvement(&self, candidate: f64) -> bool {
        match self.config.mode {
            EarlyStopMode::Min => candidate < self.best - self.config.min_delta,
            EarlyStopMode::Max => candidate > self.best + self.config.min_delta,
        }
    }

    /// Record a new metric value and return `true` if training should stop.
    ///
    /// The first observation always counts as an improvement (the initial best
    /// is ±∞).
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] if `metric` is NaN.
    pub fn update(&mut self, metric: f64) -> TrainResult<bool> {
        if metric.is_nan() {
            return Err(TrainError::Internal {
                msg: "early-stopping metric is NaN".into(),
            });
        }
        if self.is_improvement(metric) {
            self.best = metric;
            self.best_epoch = self.epoch;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs >= self.config.patience {
                self.stopped = true;
            }
        }
        self.epoch += 1;
        Ok(self.stopped)
    }

    /// Whether the stop criterion has been triggered.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Best metric value observed so far.
    #[must_use]
    pub fn best(&self) -> f64 {
        self.best
    }

    /// Epoch (0-based) at which the best metric was observed.
    #[must_use]
    pub fn best_epoch(&self) -> u64 {
        self.best_epoch
    }

    /// Consecutive non-improving epochs accumulated.
    #[must_use]
    pub fn bad_epochs(&self) -> u64 {
        self.bad_epochs
    }

    /// Number of `update` calls made.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Reset the monitor to its initial state.
    pub fn reset(&mut self) {
        self.best = match self.config.mode {
            EarlyStopMode::Min => f64::INFINITY,
            EarlyStopMode::Max => f64::NEG_INFINITY,
        };
        self.best_epoch = 0;
        self.bad_epochs = 0;
        self.epoch = 0;
        self.stopped = false;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(patience: u64, min_delta: f64, mode: EarlyStopMode) -> EarlyStoppingConfig {
        EarlyStoppingConfig {
            patience,
            min_delta,
            mode,
        }
    }

    #[test]
    fn rejects_negative_min_delta() {
        assert!(matches!(
            EarlyStopping::new(cfg(3, -1.0, EarlyStopMode::Min)),
            Err(TrainError::Internal { .. })
        ));
    }

    /// Min mode stops after `patience` consecutive non-improving epochs.
    #[test]
    fn min_mode_stops_after_patience() {
        let mut es = EarlyStopping::new(cfg(3, 0.0, EarlyStopMode::Min)).expect("valid");
        assert!(!es.update(1.0).expect("ok")); // best = 1.0
        assert!(!es.update(1.0).expect("ok")); // bad #1
        assert!(!es.update(1.0).expect("ok")); // bad #2
        assert!(es.update(1.0).expect("ok")); // bad #3 → stop
        assert!(es.should_stop());
    }

    /// An improvement resets the patience counter.
    #[test]
    fn improvement_resets_counter() {
        let mut es = EarlyStopping::new(cfg(2, 0.0, EarlyStopMode::Min)).expect("valid");
        es.update(1.0).expect("ok"); // best 1.0
        es.update(1.0).expect("ok"); // bad #1
        es.update(0.5).expect("ok"); // improvement → reset
        assert_eq!(es.bad_epochs(), 0);
        assert!(!es.should_stop());
        assert!((es.best() - 0.5).abs() < 1e-12);
    }

    /// Max mode treats higher metrics as better.
    #[test]
    fn max_mode_tracks_maximum() {
        let mut es = EarlyStopping::new(cfg(2, 0.0, EarlyStopMode::Max)).expect("valid");
        es.update(0.80).expect("ok"); // best 0.80
        es.update(0.85).expect("ok"); // improvement
        assert!((es.best() - 0.85).abs() < 1e-12);
        es.update(0.85).expect("ok"); // bad #1
        let stop = es.update(0.85).expect("ok"); // bad #2 → stop
        assert!(stop);
    }

    /// `min_delta` suppresses tiny noisy improvements: a 1e-5 gain when
    /// min_delta=1e-3 does NOT reset the counter.
    #[test]
    fn min_delta_suppresses_noise() {
        let mut es = EarlyStopping::new(cfg(2, 1e-3, EarlyStopMode::Min)).expect("valid");
        es.update(1.0).expect("ok"); // best 1.0
        es.update(1.0 - 1e-5).expect("ok"); // below min_delta → bad #1
        assert_eq!(es.bad_epochs(), 1);
        let stop = es.update(1.0 - 2e-5).expect("ok"); // still below → bad #2 → stop
        assert!(stop);
        // A change exceeding min_delta would have counted as improvement.
    }

    /// `best_epoch` records when the optimum occurred.
    #[test]
    fn tracks_best_epoch() {
        let mut es = EarlyStopping::new(cfg(10, 0.0, EarlyStopMode::Min)).expect("valid");
        let metrics = [1.0, 0.8, 0.6, 0.7, 0.65];
        for &m in &metrics {
            es.update(m).expect("ok");
        }
        // Best 0.6 at index 2.
        assert_eq!(es.best_epoch(), 2);
        assert!((es.best() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn nan_metric_errors() {
        let mut es = EarlyStopping::new(cfg(3, 0.0, EarlyStopMode::Min)).expect("valid");
        assert!(matches!(
            es.update(f64::NAN),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn reset_restores_initial() {
        let mut es = EarlyStopping::new(cfg(2, 0.0, EarlyStopMode::Min)).expect("valid");
        es.update(1.0).expect("ok");
        es.update(1.0).expect("ok");
        es.reset();
        assert_eq!(es.epoch(), 0);
        assert_eq!(es.bad_epochs(), 0);
        assert!(!es.should_stop());
        assert!(es.best().is_infinite());
    }

    /// A monotonically improving sequence never stops.
    #[test]
    fn monotone_improvement_never_stops() {
        let mut es = EarlyStopping::new(cfg(3, 0.0, EarlyStopMode::Min)).expect("valid");
        for i in 0..20 {
            let m = 1.0 - i as f64 * 0.01;
            assert!(
                !es.update(m).expect("ok"),
                "should not stop while improving"
            );
        }
    }
}
