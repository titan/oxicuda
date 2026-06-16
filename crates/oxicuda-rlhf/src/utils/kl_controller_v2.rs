//! Multiplicative KL-divergence controller (Ziegler 2019).
//!
//! Implements the adaptive KL coefficient update from:
//! Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019).
//!
//! This is a *different* controller from the proportional one in
//! `ppo_rlhf::kl_control`. Both coexist; use this one when you need the
//! multiplicative ratio-based update rule.

use crate::error::{RlhfError, RlhfResult};

/// Configuration for the multiplicative KL controller.
#[derive(Debug, Clone)]
pub struct KlControllerConfig {
    /// Target KL divergence per update step. Must be > 0.
    pub target_kl: f32,
    /// Initial KL penalty coefficient. Must be > 0.
    pub init_kl_coeff: f32,
    /// Planning horizon (reserved for future use). Must be > 0.
    pub horizon: usize,
}

/// Multiplicative KL-divergence controller (Ziegler 2019).
///
/// Adapts the KL penalty coefficient `β` such that the measured KL divergence
/// tracks a target value:
///
/// ```text
/// ratio  = current_kl / target_kl
/// β_new  = β * (1 + ratio) / (1 + 1/ratio)   (current_kl > 0)
/// β_new  = β * 0.99                            (current_kl == 0)
/// β_new  = clamp(β_new, 1e-6, 1e6)
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct KlController {
    kl_coeff: f32,
    config: KlControllerConfig,
}

impl KlController {
    /// Construct a new `KlController`.
    ///
    /// # Errors
    /// - [`RlhfError::KlDivergence`] if `target_kl <= 0`.
    /// - [`RlhfError::Internal`] if `init_kl_coeff <= 0` or `horizon == 0`.
    pub fn new(config: KlControllerConfig) -> RlhfResult<Self> {
        if config.target_kl <= 0.0 {
            return Err(RlhfError::KlDivergence {
                msg: "target_kl must be positive".to_string(),
            });
        }
        if config.init_kl_coeff <= 0.0 {
            return Err(RlhfError::Internal {
                msg: "init_kl_coeff must be positive".to_string(),
            });
        }
        if config.horizon == 0 {
            return Err(RlhfError::Internal {
                msg: "horizon must be > 0".to_string(),
            });
        }
        let kl_coeff = config.init_kl_coeff;
        Ok(Self { kl_coeff, config })
    }

    /// Apply the multiplicative update based on the observed KL divergence.
    ///
    /// When `current_kl` is positive:
    /// ```text
    /// ratio = current_kl / target_kl
    /// kl_coeff *= (1 + ratio) / (1 + 1/ratio)
    /// ```
    /// When `current_kl` is zero, multiply by 0.99 (gentle decrease).
    ///
    /// The coefficient is clamped to `[1e-6, 1e6]` to prevent runaway.
    pub fn update(&mut self, current_kl: f32) {
        let multiplier = if current_kl <= 0.0 {
            // No KL observed; gently reduce the coefficient.
            0.99
        } else {
            let ratio = current_kl / self.config.target_kl;
            // (1 + ratio) / (1 + 1/ratio) = ratio * (1 + ratio) / (ratio + 1) = ratio
            // Wait — let's be explicit: ratio = r, multiplier = (1+r)/(1+1/r)
            // = (1+r) / ((r+1)/r) = r*(1+r)/(r+1) = r
            // So the Ziegler update is simply: kl_coeff *= ratio? That seems too simple.
            // Let's keep the formula as specified: (1 + ratio) / (1 + 1/ratio)
            // Algebraic simplification: (1+r) / (1 + 1/r) = r*(1+r)/(r+1) = r
            // BUT the spec says to use that exact form, so we implement it faithfully.
            // The net effect: kl_coeff *= current_kl / target_kl (the ratio).
            (1.0 + ratio) / (1.0 + 1.0 / ratio)
        };
        self.kl_coeff = (self.kl_coeff * multiplier).clamp(1e-6, 1e6);
    }

    /// Return the current KL penalty coefficient.
    #[must_use]
    #[inline]
    pub fn kl_coeff(&self) -> f32 {
        self.kl_coeff
    }

    /// Compute the KL penalty for a given log-ratio value.
    ///
    /// `penalty = kl_coeff * log_ratio`
    #[must_use]
    #[inline]
    pub fn penalty(&self, log_ratio: f32) -> f32 {
        self.kl_coeff * log_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> KlControllerConfig {
        KlControllerConfig {
            target_kl: 0.1,
            init_kl_coeff: 0.2,
            horizon: 10_000,
        }
    }

    fn make_controller() -> KlController {
        KlController::new(default_config()).expect("valid config")
    }

    #[test]
    fn init_coeff() {
        let ctrl = make_controller();
        assert!(
            (ctrl.kl_coeff() - 0.2).abs() < 1e-6,
            "initial kl_coeff should equal init_kl_coeff, got {}",
            ctrl.kl_coeff()
        );
    }

    #[test]
    fn high_kl_increases_coeff() {
        let mut ctrl = make_controller();
        let before = ctrl.kl_coeff();
        // current_kl = 10 * target_kl => ratio = 10 => multiplier = 11/(1+0.1) > 1
        ctrl.update(default_config().target_kl * 10.0);
        assert!(
            ctrl.kl_coeff() > before,
            "high KL should increase coefficient: before={before}, after={}",
            ctrl.kl_coeff()
        );
    }

    #[test]
    fn low_kl_decreases_coeff() {
        let mut ctrl = make_controller();
        let before = ctrl.kl_coeff();
        // current_kl = 0.01 * target_kl => ratio = 0.01 => multiplier < 1
        ctrl.update(default_config().target_kl * 0.01);
        assert!(
            ctrl.kl_coeff() < before,
            "low KL should decrease coefficient: before={before}, after={}",
            ctrl.kl_coeff()
        );
    }

    #[test]
    fn penalty_scales() {
        let ctrl = make_controller();
        let coeff = ctrl.kl_coeff();
        let p = ctrl.penalty(2.0);
        assert!(
            (p - 2.0 * coeff).abs() < 1e-6,
            "penalty(2.0) should equal 2.0 * kl_coeff(), got penalty={p}, 2*coeff={}",
            2.0 * coeff
        );
    }

    #[test]
    fn multiple_updates() {
        let mut ctrl = make_controller();
        for _ in 0..50 {
            ctrl.update(default_config().target_kl * 10.0);
        }
        assert!(
            ctrl.kl_coeff() > 0.0,
            "kl_coeff must remain positive after many high-KL updates, got {}",
            ctrl.kl_coeff()
        );
    }

    #[test]
    fn target_kl_0_error() {
        let config = KlControllerConfig {
            target_kl: 0.0,
            init_kl_coeff: 0.2,
            horizon: 1000,
        };
        let err = KlController::new(config).expect_err("target_kl=0 must error");
        assert!(
            matches!(err, RlhfError::KlDivergence { .. }),
            "expected KlDivergence error, got {err:?}"
        );
    }

    #[test]
    fn coeff_positive() {
        let mut ctrl = make_controller();
        // Mix high and low KL updates; coefficient must stay strictly positive.
        for i in 0..100 {
            let kl = if i % 2 == 0 { 1.0 } else { 0.001 };
            ctrl.update(kl);
            assert!(
                ctrl.kl_coeff() > 0.0,
                "kl_coeff must be positive after update {i}, got {}",
                ctrl.kl_coeff()
            );
        }
    }

    #[test]
    fn penalty_finite() {
        let mut ctrl = make_controller();
        ctrl.update(0.5);
        let p = ctrl.penalty(1.234_567);
        assert!(
            p.is_finite(),
            "penalty must be finite for finite log_ratio, got {p}"
        );
    }
}
