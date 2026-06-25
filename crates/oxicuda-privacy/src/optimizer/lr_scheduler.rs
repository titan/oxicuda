//! DP-aware learning-rate schedulers.
//!
//! References:
//! - Loshchilov & Hutter (2017), "SGDR: Stochastic Gradient Descent with Warm
//!   Restarts", ICLR — cosine annealing.
//! - Goyal et al. (2017), "Accurate, Large Minibatch SGD" — linear warmup.
//! - Bu, Wang, Zha & Karypis (2021), "Automatic Clipping" and the DP-SGD
//!   literature motivate *noise-aware* and *budget-aware* schedules: because the
//!   gradient SNR in DP training degrades as the privacy budget is consumed and
//!   as the noise multiplier `σ` rises, the learning rate should respond to the
//!   *privacy* state, not just the step index.
//!
//! # What "DP-aware" adds over standard schedulers
//! Two of the schedules here are specific to differentially private training:
//!
//! - [`LrSchedule::BudgetAware`]: scales the LR by the *fraction of privacy
//!   budget remaining*, `η_t = η₀ · (1 − ρ_spent/ρ_total)^p`.  Early steps
//!   (budget-rich, low cumulative noise) take large steps; as the zCDP budget is
//!   exhausted and estimates get noisier, the step shrinks — annealing keyed to
//!   privacy expenditure rather than wall-clock step.
//! - [`LrSchedule::NoiseAware`]: scales the LR inversely with the per-step noise
//!   standard deviation, `η_t = η₀ / (1 + κ·σ_t·C_t)`, damping updates exactly
//!   when the injected DP noise (`σ·C`) is large.
//!
//! The classic step-index schedules (constant, step-decay, exponential, cosine
//! with warmup) are also provided so a DP optimiser has a single uniform LR
//! interface.

use crate::error::{PrivacyError, PrivacyResult};

/// A learning-rate schedule.
#[derive(Debug, Clone)]
pub enum LrSchedule {
    /// Fixed learning rate `η₀` for all steps.
    Constant {
        /// Base learning rate.
        base_lr: f64,
    },
    /// Step decay: `η = η₀ · γ^⌊t / period⌋`.
    StepDecay {
        /// Base learning rate `η₀`.
        base_lr: f64,
        /// Multiplicative decay `γ ∈ (0, 1]` applied every `period` steps.
        gamma: f64,
        /// Steps between successive decays (`≥ 1`).
        period: usize,
    },
    /// Exponential decay: `η = η₀ · exp(−rate · t)`.
    Exponential {
        /// Base learning rate `η₀`.
        base_lr: f64,
        /// Decay rate `≥ 0`.
        rate: f64,
    },
    /// Cosine annealing with optional linear warmup (SGDR).
    ///
    /// During `[0, warmup)` the LR ramps linearly `0 → η₀`; thereafter it
    /// anneals `η₀ → η_min` over `[warmup, total_steps)` following a half-cosine.
    CosineWarmup {
        /// Peak learning rate `η₀` (reached at the end of warmup).
        base_lr: f64,
        /// Minimum (floor) learning rate `η_min ≥ 0`.
        min_lr: f64,
        /// Number of linear-warmup steps.
        warmup: usize,
        /// Total scheduled steps (`> warmup`).
        total_steps: usize,
    },
    /// **DP budget-aware**: `η = η₀ · (1 − ρ_spent/ρ_total)^power`, annealing as
    /// the zCDP budget is consumed.
    BudgetAware {
        /// Base learning rate `η₀`.
        base_lr: f64,
        /// Total zCDP budget `ρ_total > 0`.
        rho_total: f64,
        /// Annealing exponent `power ≥ 0` (1.0 = linear in remaining budget).
        power: f64,
    },
    /// **DP noise-aware**: `η = η₀ / (1 + κ·σ·C)`, damping the step when the
    /// per-step injected noise std `σ·C` is large.
    NoiseAware {
        /// Base learning rate `η₀`.
        base_lr: f64,
        /// Sensitivity to the noise magnitude `κ ≥ 0`.
        kappa: f64,
    },
}

impl LrSchedule {
    /// Validate the schedule parameters.
    ///
    /// # Errors
    /// - `InvalidParameter` if any rate / bound is out of its valid range.
    pub fn validate(&self) -> PrivacyResult<()> {
        let pos = |name: &str, v: f64| -> PrivacyResult<()> {
            if v <= 0.0 {
                return Err(PrivacyError::InvalidParameter(format!(
                    "{name} must be positive, got {v}"
                )));
            }
            Ok(())
        };
        let nonneg = |name: &str, v: f64| -> PrivacyResult<()> {
            if v < 0.0 {
                return Err(PrivacyError::InvalidParameter(format!(
                    "{name} must be ≥ 0, got {v}"
                )));
            }
            Ok(())
        };
        match *self {
            LrSchedule::Constant { base_lr } => pos("base_lr", base_lr),
            LrSchedule::StepDecay {
                base_lr,
                gamma,
                period,
            } => {
                pos("base_lr", base_lr)?;
                if !(gamma > 0.0 && gamma <= 1.0) {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "gamma must be in (0,1], got {gamma}"
                    )));
                }
                if period == 0 {
                    return Err(PrivacyError::InvalidParameter("period must be ≥ 1".into()));
                }
                Ok(())
            }
            LrSchedule::Exponential { base_lr, rate } => {
                pos("base_lr", base_lr)?;
                nonneg("rate", rate)
            }
            LrSchedule::CosineWarmup {
                base_lr,
                min_lr,
                warmup,
                total_steps,
            } => {
                pos("base_lr", base_lr)?;
                nonneg("min_lr", min_lr)?;
                if min_lr > base_lr {
                    return Err(PrivacyError::InvalidParameter(
                        "min_lr must be ≤ base_lr".into(),
                    ));
                }
                if total_steps <= warmup {
                    return Err(PrivacyError::InvalidParameter(
                        "total_steps must be > warmup".into(),
                    ));
                }
                Ok(())
            }
            LrSchedule::BudgetAware {
                base_lr,
                rho_total,
                power,
            } => {
                pos("base_lr", base_lr)?;
                pos("rho_total", rho_total)?;
                nonneg("power", power)
            }
            LrSchedule::NoiseAware { base_lr, kappa } => {
                pos("base_lr", base_lr)?;
                nonneg("kappa", kappa)
            }
        }
    }

    /// Learning rate at integer step `t` (0-indexed) for the step-index
    /// schedules (`Constant`, `StepDecay`, `Exponential`, `CosineWarmup`).
    ///
    /// For the DP-state schedules use [`LrSchedule::lr_at_budget`] /
    /// [`LrSchedule::lr_at_noise`]; calling this on them returns the base LR.
    #[must_use]
    pub fn lr_at_step(&self, t: usize) -> f64 {
        match *self {
            LrSchedule::Constant { base_lr } => base_lr,
            LrSchedule::StepDecay {
                base_lr,
                gamma,
                period,
            } => {
                let k = (t / period.max(1)) as i32;
                base_lr * gamma.powi(k)
            }
            LrSchedule::Exponential { base_lr, rate } => base_lr * (-rate * t as f64).exp(),
            LrSchedule::CosineWarmup {
                base_lr,
                min_lr,
                warmup,
                total_steps,
            } => {
                if t < warmup {
                    // Linear warmup 0 → base_lr (at step `warmup`).
                    let frac = (t as f64 + 1.0) / (warmup as f64).max(1.0);
                    base_lr * frac.min(1.0)
                } else {
                    let denom = (total_steps - warmup).max(1) as f64;
                    let progress = ((t - warmup) as f64 / denom).min(1.0);
                    let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                    min_lr + (base_lr - min_lr) * cos
                }
            }
            LrSchedule::BudgetAware { base_lr, .. } | LrSchedule::NoiseAware { base_lr, .. } => {
                base_lr
            }
        }
    }

    /// Learning rate for the [`LrSchedule::BudgetAware`] schedule given the zCDP
    /// budget already spent.
    ///
    /// Returns `η₀ · (1 − ρ_spent/ρ_total)^power`, clamped to `[0, η₀]`.  For
    /// non-budget schedules, falls back to [`Self::lr_at_step`] semantics by returning
    /// the base LR scaled by `1` (i.e. ignores the budget).
    ///
    /// # Errors
    /// - `InvalidParameter` if `rho_spent < 0`.
    pub fn lr_at_budget(&self, rho_spent: f64) -> PrivacyResult<f64> {
        if rho_spent < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "rho_spent must be ≥ 0, got {rho_spent}"
            )));
        }
        match *self {
            LrSchedule::BudgetAware {
                base_lr,
                rho_total,
                power,
            } => {
                let remaining = (1.0 - rho_spent / rho_total).clamp(0.0, 1.0);
                Ok(base_lr * remaining.powf(power))
            }
            _ => Ok(self.base_lr()),
        }
    }

    /// Learning rate for the [`LrSchedule::NoiseAware`] schedule given the
    /// per-step injected noise std `noise_std = σ·C`.
    ///
    /// Returns `η₀ / (1 + κ·noise_std)`.  For non-noise schedules returns the
    /// base LR.
    ///
    /// # Errors
    /// - `InvalidParameter` if `noise_std < 0`.
    pub fn lr_at_noise(&self, noise_std: f64) -> PrivacyResult<f64> {
        if noise_std < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_std must be ≥ 0, got {noise_std}"
            )));
        }
        match *self {
            LrSchedule::NoiseAware { base_lr, kappa } => Ok(base_lr / (1.0 + kappa * noise_std)),
            _ => Ok(self.base_lr()),
        }
    }

    /// The base (peak) learning rate `η₀` of this schedule.
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        match *self {
            LrSchedule::Constant { base_lr }
            | LrSchedule::StepDecay { base_lr, .. }
            | LrSchedule::Exponential { base_lr, .. }
            | LrSchedule::CosineWarmup { base_lr, .. }
            | LrSchedule::BudgetAware { base_lr, .. }
            | LrSchedule::NoiseAware { base_lr, .. } => base_lr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_constant_is_flat() {
        let s = LrSchedule::Constant { base_lr: 0.1 };
        s.validate().expect("valid");
        for t in [0, 5, 100, 10_000] {
            assert!(approx(s.lr_at_step(t), 0.1, 1e-15));
        }
    }

    #[test]
    fn test_step_decay_halves_each_period() {
        let s = LrSchedule::StepDecay {
            base_lr: 1.0,
            gamma: 0.5,
            period: 10,
        };
        s.validate().expect("valid");
        assert!(approx(s.lr_at_step(0), 1.0, 1e-12));
        assert!(approx(s.lr_at_step(9), 1.0, 1e-12));
        assert!(approx(s.lr_at_step(10), 0.5, 1e-12));
        assert!(approx(s.lr_at_step(20), 0.25, 1e-12));
        assert!(approx(s.lr_at_step(30), 0.125, 1e-12));
    }

    #[test]
    fn test_exponential_decreasing() {
        let s = LrSchedule::Exponential {
            base_lr: 0.5,
            rate: 0.01,
        };
        s.validate().expect("valid");
        assert!(approx(s.lr_at_step(0), 0.5, 1e-12));
        let a = s.lr_at_step(10);
        let b = s.lr_at_step(100);
        assert!(a > b && b > 0.0, "should monotonically decay: {a} > {b}");
        assert!(approx(a, 0.5 * (-0.1f64).exp(), 1e-12));
    }

    #[test]
    fn test_cosine_warmup_shape() {
        let s = LrSchedule::CosineWarmup {
            base_lr: 1.0,
            min_lr: 0.0,
            warmup: 10,
            total_steps: 110,
        };
        s.validate().expect("valid");
        // Warmup ramps up.
        assert!(s.lr_at_step(0) < s.lr_at_step(5));
        // Peak ≈ base_lr at end of warmup.
        assert!(approx(s.lr_at_step(9), 1.0, 1e-9));
        // Mid-anneal (progress 0.5) ≈ 0.5·base_lr.
        let mid = s.lr_at_step(10 + 50);
        assert!(approx(mid, 0.5, 1e-2), "cosine midpoint {mid}");
        // End ≈ min_lr.
        let end = s.lr_at_step(109);
        assert!(end < 0.05, "cosine end {end} should approach min_lr");
    }

    #[test]
    fn test_budget_aware_anneals_with_spend() {
        let s = LrSchedule::BudgetAware {
            base_lr: 0.2,
            rho_total: 4.0,
            power: 1.0,
        };
        s.validate().expect("valid");
        assert!(approx(s.lr_at_budget(0.0).expect("a"), 0.2, 1e-12));
        // Half budget spent → half LR (linear power).
        assert!(approx(s.lr_at_budget(2.0).expect("b"), 0.1, 1e-12));
        // Fully spent → 0.
        assert!(approx(s.lr_at_budget(4.0).expect("c"), 0.0, 1e-12));
        // Over-spent clamps to 0, not negative.
        assert!(approx(s.lr_at_budget(8.0).expect("d"), 0.0, 1e-12));
    }

    #[test]
    fn test_budget_aware_power_curvature() {
        let s = LrSchedule::BudgetAware {
            base_lr: 1.0,
            rho_total: 1.0,
            power: 2.0,
        };
        // At 50% spent, remaining=0.5, lr = 0.5² = 0.25.
        assert!(approx(s.lr_at_budget(0.5).expect("x"), 0.25, 1e-12));
    }

    #[test]
    fn test_noise_aware_damps_with_noise() {
        let s = LrSchedule::NoiseAware {
            base_lr: 0.1,
            kappa: 2.0,
        };
        s.validate().expect("valid");
        assert!(approx(s.lr_at_noise(0.0).expect("a"), 0.1, 1e-12));
        // noise_std=1 → 0.1/(1+2) = 0.0333…
        assert!(approx(s.lr_at_noise(1.0).expect("b"), 0.1 / 3.0, 1e-12));
        // Larger noise → smaller LR.
        assert!(s.lr_at_noise(5.0).expect("c") < s.lr_at_noise(1.0).expect("d"));
    }

    #[test]
    fn test_cross_schedule_fallbacks() {
        // budget query on a noise schedule returns base LR (graceful fallback).
        let n = LrSchedule::NoiseAware {
            base_lr: 0.3,
            kappa: 1.0,
        };
        assert!(approx(n.lr_at_budget(1.0).expect("x"), 0.3, 1e-12));
        let b = LrSchedule::BudgetAware {
            base_lr: 0.4,
            rho_total: 1.0,
            power: 1.0,
        };
        assert!(approx(b.lr_at_noise(2.0).expect("y"), 0.4, 1e-12));
    }

    #[test]
    fn test_validation_rejects_bad_params() {
        assert!(LrSchedule::Constant { base_lr: 0.0 }.validate().is_err());
        assert!(
            LrSchedule::StepDecay {
                base_lr: 1.0,
                gamma: 1.5,
                period: 10
            }
            .validate()
            .is_err()
        );
        assert!(
            LrSchedule::StepDecay {
                base_lr: 1.0,
                gamma: 0.5,
                period: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            LrSchedule::CosineWarmup {
                base_lr: 1.0,
                min_lr: 2.0,
                warmup: 10,
                total_steps: 100
            }
            .validate()
            .is_err()
        );
        assert!(
            LrSchedule::CosineWarmup {
                base_lr: 1.0,
                min_lr: 0.0,
                warmup: 100,
                total_steps: 100
            }
            .validate()
            .is_err()
        );
        assert!(
            LrSchedule::BudgetAware {
                base_lr: 1.0,
                rho_total: 0.0,
                power: 1.0
            }
            .validate()
            .is_err()
        );
        let s = LrSchedule::BudgetAware {
            base_lr: 1.0,
            rho_total: 1.0,
            power: 1.0,
        };
        assert!(s.lr_at_budget(-1.0).is_err());
        let n = LrSchedule::NoiseAware {
            base_lr: 1.0,
            kappa: 1.0,
        };
        assert!(n.lr_at_noise(-1.0).is_err());
    }
}
