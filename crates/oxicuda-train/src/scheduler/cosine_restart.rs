//! SGDR — Cosine annealing with warm restarts (Loshchilov & Hutter, 2017).
//!
//! "SGDR: Stochastic Gradient Descent with Warm Restarts" (arXiv:1608.03983).
//!
//! Inside each *restart cycle* `i` the learning rate follows a cosine decay
//! from `eta_max` down to `eta_min`, then is instantly **restarted** back up to
//! `eta_max` to begin the next cycle.  Letting `T_cur` be the number of steps
//! since the last restart and `T_i` the length of the current cycle:
//!
//! ```text
//! lr(T_cur) = eta_min + ½·(eta_max − eta_min)·(1 + cos(π·T_cur / T_i))
//! ```
//!
//! Cycle lengths grow geometrically: `T_{i+1} = ⌈t_mult · T_i⌉` (with
//! `t_mult = 1` giving fixed-length cycles).  The warm restarts repeatedly
//! kick the optimiser out of sharp minima, often improving generalisation and
//! enabling cheap "snapshot ensembles" taken at the end of each cycle.
//!
//! Implements the crate's [`crate::lr_scheduler::LrScheduler`] trait so it is a
//! drop-in for any optimiser via `set_lr`.

use crate::error::{TrainError, TrainResult};
use crate::lr_scheduler::LrScheduler;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`CosineAnnealingWarmRestarts`] scheduler.
#[derive(Debug, Clone)]
pub struct CosineRestartConfig {
    /// Peak learning rate `eta_max` at the start of every cycle (must be > 0).
    pub eta_max: f64,
    /// Trough learning rate `eta_min` at the end of every cycle (≥ 0; must be
    /// `< eta_max`).
    pub eta_min: f64,
    /// Length of the first cycle `T_0` in steps (must be ≥ 1).
    pub t_0: u64,
    /// Geometric cycle-length multiplier `t_mult` (must be ≥ 1.0).  `1.0` keeps
    /// every cycle `t_0` long; `2.0` doubles each successive cycle.
    pub t_mult: f64,
}

impl Default for CosineRestartConfig {
    fn default() -> Self {
        Self {
            eta_max: 1e-2,
            eta_min: 0.0,
            t_0: 10,
            t_mult: 1.0,
        }
    }
}

impl CosineRestartConfig {
    /// Validate every field.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if `eta_max ≤ 0`.
    /// * [`TrainError::Internal`] for any other out-of-range field.
    fn validate(&self) -> TrainResult<()> {
        if self.eta_max <= 0.0 || self.eta_max.is_nan() {
            return Err(TrainError::InvalidLearningRate { lr: self.eta_max });
        }
        if self.eta_min < 0.0 || self.eta_min.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("eta_min must be non-negative, got {}", self.eta_min),
            });
        }
        if self.eta_min >= self.eta_max {
            return Err(TrainError::Internal {
                msg: format!(
                    "eta_min ({}) must be < eta_max ({})",
                    self.eta_min, self.eta_max
                ),
            });
        }
        if self.t_0 == 0 {
            return Err(TrainError::Internal {
                msg: "t_0 must be >= 1".into(),
            });
        }
        if self.t_mult < 1.0 || self.t_mult.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("t_mult must be >= 1.0, got {}", self.t_mult),
            });
        }
        Ok(())
    }
}

// ─── Scheduler ───────────────────────────────────────────────────────────────

/// Cosine annealing scheduler with periodic warm restarts (SGDR).
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    config: CosineRestartConfig,
    current_lr: f64,
    /// Total steps taken.
    steps: u64,
    /// Steps elapsed within the current cycle (`T_cur`).
    t_cur: u64,
    /// Length of the current cycle (`T_i`).
    t_i: u64,
    /// Zero-based index of the current restart cycle.
    cycle: u64,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new SGDR scheduler.
    ///
    /// # Errors
    ///
    /// Any error from `CosineRestartConfig::validate`.
    pub fn new(config: CosineRestartConfig) -> TrainResult<Self> {
        config.validate()?;
        let t_i = config.t_0;
        Ok(Self {
            current_lr: config.eta_max,
            config,
            steps: 0,
            t_cur: 0,
            t_i,
            cycle: 0,
        })
    }

    /// Closed-form LR within the current cycle for a given `t_cur`.
    #[inline]
    fn cosine(&self, t_cur: u64) -> f64 {
        let frac = t_cur as f64 / self.t_i as f64;
        let cos_val = (std::f64::consts::PI * frac).cos();
        self.config.eta_min + 0.5 * (self.config.eta_max - self.config.eta_min) * (1.0 + cos_val)
    }

    /// Zero-based index of the current restart cycle.
    #[must_use]
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// Length (in steps) of the current cycle `T_i`.
    #[must_use]
    pub fn cycle_length(&self) -> u64 {
        self.t_i
    }

    /// Peak learning rate `eta_max`.
    #[must_use]
    pub fn eta_max(&self) -> f64 {
        self.config.eta_max
    }
}

impl LrScheduler for CosineAnnealingWarmRestarts {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        // Advance position within the cycle; restart when the cycle completes.
        self.t_cur += 1;
        if self.t_cur >= self.t_i {
            // Restart: begin a new (possibly longer) cycle.
            self.t_cur = 0;
            self.cycle += 1;
            if self.config.t_mult > 1.0 {
                self.t_i = (self.t_i as f64 * self.config.t_mult).ceil() as u64;
            }
        }
        self.current_lr = self.cosine(self.t_cur);
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "CosineAnnealingWarmRestarts"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CosineRestartConfig {
        CosineRestartConfig {
            eta_max: 1.0,
            eta_min: 0.0,
            t_0: 4,
            t_mult: 1.0,
        }
    }

    #[test]
    fn rejects_bad_eta_max() {
        let mut c = cfg();
        c.eta_max = 0.0;
        assert!(matches!(
            CosineAnnealingWarmRestarts::new(c),
            Err(TrainError::InvalidLearningRate { .. })
        ));
    }

    #[test]
    fn rejects_eta_min_ge_max() {
        let mut c = cfg();
        c.eta_min = 1.0;
        assert!(matches!(
            CosineAnnealingWarmRestarts::new(c),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn rejects_bad_t_mult() {
        let mut c = cfg();
        c.t_mult = 0.5;
        assert!(matches!(
            CosineAnnealingWarmRestarts::new(c),
            Err(TrainError::Internal { .. })
        ));
    }

    /// LR at sampled steps matches the closed-form cosine of the fixed cycle.
    #[test]
    fn matches_closed_form_fixed_cycle() {
        let t0 = 4;
        let mut s = CosineAnnealingWarmRestarts::new(cfg()).expect("valid");
        // Steps within first cycle: t_cur = 1,2,3, then restart at step 4.
        for step in 1..=3u64 {
            let lr = s.step();
            let expect = 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / t0 as f64).cos());
            assert!(
                (lr - expect).abs() < 1e-12,
                "step {step}: lr {lr} vs {expect}"
            );
        }
    }

    /// At a restart boundary the LR jumps back up to eta_max.
    #[test]
    fn restarts_to_peak() {
        let mut s = CosineAnnealingWarmRestarts::new(cfg()).expect("valid");
        // 4 steps complete cycle 0; step 4 restarts → t_cur=0 → lr=eta_max.
        let mut lr = 0.0;
        for _ in 0..4 {
            lr = s.step();
        }
        assert!(
            (lr - 1.0).abs() < 1e-12,
            "expected restart to peak, got {lr}"
        );
        assert_eq!(s.cycle(), 1);
    }

    /// Geometric cycle growth: with t_mult=2, cycle lengths go 2,4,8,...
    #[test]
    fn geometric_cycle_growth() {
        let c = CosineRestartConfig {
            eta_max: 1.0,
            eta_min: 0.0,
            t_0: 2,
            t_mult: 2.0,
        };
        let mut s = CosineAnnealingWarmRestarts::new(c).expect("valid");
        assert_eq!(s.cycle_length(), 2);
        // Complete first cycle (2 steps).
        s.step();
        s.step();
        assert_eq!(s.cycle(), 1);
        assert_eq!(s.cycle_length(), 4, "second cycle should double to 4");
        // Complete second cycle (4 steps).
        for _ in 0..4 {
            s.step();
        }
        assert_eq!(s.cycle(), 2);
        assert_eq!(s.cycle_length(), 8, "third cycle should double to 8");
    }

    /// LR stays within [eta_min, eta_max] across many steps and restarts.
    #[test]
    fn lr_bounded() {
        let c = CosineRestartConfig {
            eta_max: 0.5,
            eta_min: 0.05,
            t_0: 3,
            t_mult: 1.5,
        };
        let mut s = CosineAnnealingWarmRestarts::new(c).expect("valid");
        for _ in 0..200 {
            let lr = s.step();
            assert!(
                (0.05 - 1e-9..=0.5 + 1e-9).contains(&lr),
                "lr {lr} out of bounds"
            );
        }
    }

    #[test]
    fn steps_done_counts() {
        let mut s = CosineAnnealingWarmRestarts::new(cfg()).expect("valid");
        for _ in 0..7 {
            s.step();
        }
        assert_eq!(s.steps_done(), 7);
    }

    /// The trough of each cycle approaches eta_min just before the restart.
    #[test]
    fn trough_near_eta_min() {
        let c = CosineRestartConfig {
            eta_max: 1.0,
            eta_min: 0.0,
            t_0: 8,
            t_mult: 1.0,
        };
        let mut s = CosineAnnealingWarmRestarts::new(c).expect("valid");
        let mut min_lr = f64::INFINITY;
        for _ in 0..8 {
            min_lr = min_lr.min(s.step());
        }
        // The smallest within-cycle LR (at t_cur = t_0-1 = 7) is close to 0.
        assert!(
            min_lr < 0.05,
            "trough should approach eta_min, got {min_lr}"
        );
    }
}
