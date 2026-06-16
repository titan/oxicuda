//! Warmup-Stable-Decay (WSD) learning rate scheduler.
//!
//! WSD (Hu et al., 2024 — "MiniCPM") divides training into three phases:
//!
//! 1. **Warmup** — linear ramp from `initial_lr` to `stable_lr` over `warmup_steps`.
//! 2. **Stable** — constant `stable_lr` for `stable_steps` steps.
//! 3. **Decay** — cosine annealing from `stable_lr` to `final_lr` over `decay_steps`.
//!
//! After `total_steps = warmup + stable + decay`, the scheduler returns `final_lr`
//! indefinitely.
//!
//! ## Cosine decay formula
//!
//! ```text
//! lr(t) = final_lr + (stable_lr − final_lr) × 0.5 × (1 + cos(π·t / decay_steps))
//! ```
//! where `t ∈ [0, decay_steps)` is the step within the decay phase.

use crate::error::TrainResult;
use std::f32::consts::PI;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`WsdScheduler`].
#[derive(Debug, Clone)]
pub struct WsdConfig {
    /// Number of linear warmup steps.
    pub warmup_steps: usize,
    /// Number of steps at stable learning rate.
    pub stable_steps: usize,
    /// Number of cosine decay steps.
    pub decay_steps: usize,
    /// Starting learning rate for warmup (step 0).
    pub initial_lr: f32,
    /// Peak / stable learning rate.
    pub stable_lr: f32,
    /// Final (minimum) learning rate after decay.
    pub final_lr: f32,
}

// ─── Scheduler ───────────────────────────────────────────────────────────────

/// Warmup-Stable-Decay learning rate scheduler.
///
/// Provides a pure-function `get_lr(step)` interface so it can be used either
/// step-by-step or queried at arbitrary steps.
pub struct WsdScheduler {
    config: WsdConfig,
}

impl WsdScheduler {
    /// Create a new [`WsdScheduler`].
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Ok` unconditionally.  Validation of
    /// learning-rate ordering is left to the caller.
    pub fn new(config: WsdConfig) -> TrainResult<Self> {
        Ok(Self { config })
    }

    /// Return the learning rate for a given absolute `step`.
    ///
    /// * `step < warmup_steps`  → linear warmup from `initial_lr` to `stable_lr`
    /// * `warmup_steps <= step < warmup + stable` → `stable_lr`
    /// * `warmup + stable <= step < total` → cosine decay to `final_lr`
    /// * `step >= total` → `final_lr`
    #[must_use]
    pub fn get_lr(&self, step: usize) -> f32 {
        let w = self.config.warmup_steps;
        let s = self.config.stable_steps;
        let d = self.config.decay_steps;

        if step < w {
            // Linear warmup: handle zero-length warmup gracefully
            if w == 0 {
                return self.config.stable_lr;
            }
            let t = step as f32 / w as f32;
            self.config.initial_lr + t * (self.config.stable_lr - self.config.initial_lr)
        } else if step < w + s {
            self.config.stable_lr
        } else if step < w + s + d {
            let t = (step - w - s) as f32;
            let cos_val = (PI * t / d as f32).cos();
            self.config.final_lr
                + (self.config.stable_lr - self.config.final_lr) * 0.5 * (1.0 + cos_val)
        } else {
            self.config.final_lr
        }
    }

    /// Total number of steps covered by warmup + stable + decay.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.config.warmup_steps + self.config.stable_steps + self.config.decay_steps
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &WsdConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler() -> WsdScheduler {
        WsdScheduler::new(WsdConfig {
            warmup_steps: 10,
            stable_steps: 20,
            decay_steps: 30,
            initial_lr: 0.0,
            stable_lr: 1e-3,
            final_lr: 1e-5,
        })
        .expect("valid config")
    }

    /// At step 0 the LR must equal initial_lr.
    #[test]
    fn warmup_start_at_initial() {
        let sched = make_scheduler();
        let lr = sched.get_lr(0);
        assert!(
            (lr - 0.0_f32).abs() < 1e-9,
            "step 0 should equal initial_lr=0, got {lr}"
        );
    }

    /// At the first stable step (step == warmup_steps) LR equals stable_lr.
    #[test]
    fn warmup_end_at_stable() {
        let sched = make_scheduler();
        // step = warmup_steps falls into the stable region (w <= step < w+s)
        let lr = sched.get_lr(10);
        assert!(
            (lr - 1e-3_f32).abs() < 1e-9,
            "step=warmup_steps should equal stable_lr=1e-3, got {lr}"
        );
    }

    /// Several steps inside the stable region must all return stable_lr.
    #[test]
    fn stable_constant() {
        let sched = make_scheduler();
        for step in 10..30 {
            let lr = sched.get_lr(step);
            assert!(
                (lr - 1e-3_f32).abs() < 1e-9,
                "step {step} in stable region should return stable_lr=1e-3, got {lr}"
            );
        }
    }

    /// The first decay step (step == warmup + stable) must start at stable_lr.
    #[test]
    fn decay_starts_at_stable() {
        let sched = make_scheduler();
        // cos(0) = 1 → final_lr + (stable - final) * 0.5 * 2 = stable_lr
        let lr = sched.get_lr(30); // warmup(10) + stable(20) = 30
        assert!(
            (lr - 1e-3_f32).abs() < 1e-6,
            "decay start should equal stable_lr≈1e-3, got {lr}"
        );
    }

    /// At the last step of decay LR should equal final_lr.
    #[test]
    fn decay_ends_at_final() {
        let sched = make_scheduler();
        let total = sched.total_steps();
        // step = total falls into the "else final_lr" branch
        let lr = sched.get_lr(total);
        assert!(
            (lr - 1e-5_f32).abs() < 1e-9,
            "step >= total should return final_lr=1e-5, got {lr}"
        );
    }

    /// All LRs across the full schedule must be >= 0 when all configured LRs >= 0.
    #[test]
    fn lr_nonneg() {
        let sched = make_scheduler();
        let total = sched.total_steps();
        for step in 0..=total + 5 {
            let lr = sched.get_lr(step);
            assert!(lr >= 0.0, "lr should be >= 0 at step {step}, got {lr}");
        }
    }

    /// During warmup the LRs must be non-decreasing when stable_lr > initial_lr.
    #[test]
    fn monotone_warmup() {
        let sched = make_scheduler();
        let mut prev = sched.get_lr(0);
        for step in 1..=10 {
            let cur = sched.get_lr(step);
            assert!(
                cur >= prev - 1e-9,
                "warmup LR should be non-decreasing at step {step}: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    /// At the midpoint of decay the LR should match the analytic cosine formula.
    #[test]
    fn decay_cosine_shape() {
        let sched = make_scheduler();
        // Midpoint of decay: t = decay_steps / 2 = 15
        // step = warmup(10) + stable(20) + 15 = 45
        let step = 45_usize;
        let t = 15_f32;
        let d = 30_f32;
        let stable_lr = 1e-3_f32;
        let final_lr = 1e-5_f32;
        let expected = final_lr + (stable_lr - final_lr) * 0.5 * (1.0 + (PI * t / d).cos());
        let actual = sched.get_lr(step);
        assert!(
            (actual - expected).abs() < 1e-7,
            "decay midpoint: expected {expected}, got {actual}"
        );
    }

    /// A step beyond total_steps must return final_lr.
    #[test]
    fn get_lr_after_total_returns_final() {
        let sched = make_scheduler();
        let total = sched.total_steps();
        let lr = sched.get_lr(total + 100);
        assert!(
            (lr - 1e-5_f32).abs() < 1e-9,
            "step beyond total should return final_lr=1e-5, got {lr}"
        );
    }

    /// Zero-length warmup should jump immediately to stable_lr at step 0.
    #[test]
    fn zero_warmup_starts_stable() {
        let sched = WsdScheduler::new(WsdConfig {
            warmup_steps: 0,
            stable_steps: 10,
            decay_steps: 10,
            initial_lr: 0.0,
            stable_lr: 5e-4,
            final_lr: 0.0,
        })
        .expect("valid config");
        let lr = sched.get_lr(0);
        assert!(
            (lr - 5e-4_f32).abs() < 1e-9,
            "zero warmup: step 0 should equal stable_lr, got {lr}"
        );
    }

    /// total_steps must equal warmup + stable + decay.
    #[test]
    fn total_steps_correct() {
        let sched = make_scheduler();
        assert_eq!(sched.total_steps(), 60, "total_steps should be 10+20+30=60");
    }
}
