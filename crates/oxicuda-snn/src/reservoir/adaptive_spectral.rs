#![allow(clippy::needless_range_loop)]
//! Adaptive spectral-radius scheduling for recurrent reservoirs.
//!
//! The *echo-state property* of a reservoir computer is governed by the spectral
//! radius `ρ(W_rec)` of its recurrent weight matrix. A value near (but below)
//! `1` places the reservoir on the *edge of chaos*, where memory capacity and
//! the richness of the dynamic projection are maximised (Jaeger 2001; Legenstein
//! & Maass 2007, "Edge of chaos and prediction of computational performance for
//! neural microcircuit models"). The optimal `ρ` is task dependent and, during
//! training, the *effective* operating point drifts as the readout and input
//! statistics evolve.
//!
//! This module provides a controller that rescales `W_rec` so that its measured
//! spectral radius tracks a moving target `ρ*(t)` produced by one of three
//! policies:
//!
//! * [`crate::reservoir::adaptive_spectral::ScheduleMode::Linear`] — anneal `ρ_start → ρ_end` linearly over
//!   `total_steps`,
//!
//!   ```text
//!   ρ*(t) = ρ_start + (ρ_end − ρ_start) · t / total_steps
//!   ```
//!
//! * [`crate::reservoir::adaptive_spectral::ScheduleMode::Exponential`] — geometric anneal,
//!
//!   ```text
//!   ρ*(t) = ρ_start · (ρ_end / ρ_start)^(t / total_steps)
//!   ```
//!
//! * [`crate::reservoir::adaptive_spectral::ScheduleMode::FeedbackVarianceControl`] — closed-loop control that drives
//!   the reservoir-state variance toward `target_variance`. If the states
//!   *saturate* (variance above target) the radius is lowered; if they *decay*
//!   (variance below target) it is raised:
//!
//!   ```text
//!   ρ_next = clamp( ρ_prev − gain · (var_measured − var_target),  ρ_end, ρ_start )
//!   ```
//!
//! Rescaling is exact up to the accuracy of [`crate::reservoir::lsm::power_iteration_spectral_radius`]:
//! we measure the current radius `ρ_meas` and multiply every entry of `W_rec`
//! by `ρ_target / ρ_meas`, which scales eigenvalues linearly and therefore sets
//! `ρ` to `ρ_target`.

use crate::error::{SnnError, SnnResult};
use crate::reservoir::lsm::power_iteration_spectral_radius;

/// Number of power-iteration sweeps used when measuring the spectral radius.
const POWER_ITERS: usize = 60;

/// Policy that produces the target spectral radius `ρ*` at each training step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScheduleMode {
    /// Linear anneal from `ρ_start` to `ρ_end` across `total_steps`.
    #[default]
    Linear,
    /// Geometric (exponential) anneal from `ρ_start` to `ρ_end`.
    Exponential,
    /// Closed-loop feedback control driving state variance to `target_variance`.
    FeedbackVarianceControl,
}

/// Configuration of an [`AdaptiveSpectralScheduler`].
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveSpectralConfig {
    /// Spectral radius at step `0`. Must be strictly positive.
    pub rho_start: f32,
    /// Spectral radius at step `total_steps`. Must be strictly positive.
    pub rho_end: f32,
    /// Horizon of the schedule, in training steps. Must be `> 0`.
    pub total_steps: usize,
    /// Which scheduling policy to apply.
    pub mode: ScheduleMode,
    /// Target reservoir-state variance for [`crate::reservoir::adaptive_spectral::ScheduleMode::FeedbackVarianceControl`].
    /// Must be `>= 0`.
    pub target_variance: f32,
    /// Proportional feedback gain for variance control. Must be `>= 0`.
    pub gain: f32,
}

impl Default for AdaptiveSpectralConfig {
    fn default() -> Self {
        Self {
            rho_start: 1.1,
            rho_end: 0.9,
            total_steps: 100,
            mode: ScheduleMode::Linear,
            target_variance: 0.25,
            gain: 0.05,
        }
    }
}

impl AdaptiveSpectralConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`SnnError::OutOfRange`] if `rho_start` or `rho_end` is non-positive or
    ///   non-finite, or if `target_variance` / `gain` is negative or non-finite.
    /// * [`SnnError::BadTimesteps`] if `total_steps == 0`.
    pub fn validate(&self) -> SnnResult<()> {
        if self.rho_start <= 0.0 || !self.rho_start.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "rho_start".to_string(),
                val: self.rho_start,
            });
        }
        if self.rho_end <= 0.0 || !self.rho_end.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "rho_end".to_string(),
                val: self.rho_end,
            });
        }
        if self.total_steps == 0 {
            return Err(SnnError::BadTimesteps { got: 0 });
        }
        if self.target_variance < 0.0 || !self.target_variance.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "target_variance".to_string(),
                val: self.target_variance,
            });
        }
        if self.gain < 0.0 || !self.gain.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "gain".to_string(),
                val: self.gain,
            });
        }
        Ok(())
    }
}

/// Stateful controller that schedules and applies the recurrent spectral radius.
#[derive(Debug, Clone)]
pub struct AdaptiveSpectralScheduler {
    /// Validated configuration.
    cfg: AdaptiveSpectralConfig,
    /// Current target radius used by the feedback policy (carries between calls).
    feedback_rho: f32,
}

impl AdaptiveSpectralScheduler {
    /// Build a scheduler from `cfg`, validating its fields.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`AdaptiveSpectralConfig::validate`].
    pub fn new(cfg: AdaptiveSpectralConfig) -> SnnResult<Self> {
        cfg.validate()?;
        let feedback_rho = cfg.rho_start;
        Ok(Self { cfg, feedback_rho })
    }

    /// Immutable view of the configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveSpectralConfig {
        &self.cfg
    }

    /// The lower / upper bounds of the radius (ordered, regardless of anneal
    /// direction).
    #[must_use]
    fn rho_bounds(&self) -> (f32, f32) {
        if self.cfg.rho_start <= self.cfg.rho_end {
            (self.cfg.rho_start, self.cfg.rho_end)
        } else {
            (self.cfg.rho_end, self.cfg.rho_start)
        }
    }

    /// Target spectral radius `ρ*(step)` for the open-loop ([`crate::reservoir::adaptive_spectral::ScheduleMode::Linear`]
    /// / [`crate::reservoir::adaptive_spectral::ScheduleMode::Exponential`]) policies.
    ///
    /// For [`crate::reservoir::adaptive_spectral::ScheduleMode::FeedbackVarianceControl`] this returns the last value
    /// produced by [`update_from_variance`](Self::update_from_variance) (the
    /// closed-loop target is driven by measurements, not by the step index).
    ///
    /// `step` is clamped to `[0, total_steps]`, so the endpoints are hit exactly.
    #[must_use]
    pub fn current_rho(&self, step: usize) -> f32 {
        let total = self.cfg.total_steps;
        let t = step.min(total);
        let frac = t as f32 / total as f32;
        match self.cfg.mode {
            ScheduleMode::Linear => {
                self.cfg.rho_start + (self.cfg.rho_end - self.cfg.rho_start) * frac
            }
            ScheduleMode::Exponential => {
                // ρ_start · (ρ_end / ρ_start)^frac ; both radii are > 0 (validated).
                let ratio = self.cfg.rho_end / self.cfg.rho_start;
                self.cfg.rho_start * ratio.powf(frac)
            }
            ScheduleMode::FeedbackVarianceControl => self.feedback_rho,
        }
    }

    /// Closed-loop update: given the measured reservoir-state variance, produce
    /// (and store) the next target radius.
    ///
    /// Proportional control: an *excess* of variance (saturation) lowers the
    /// radius, a *deficit* (decay) raises it. The result is clamped into the
    /// `[min(ρ_start, ρ_end), max(ρ_start, ρ_end)]` band.
    ///
    /// # Errors
    ///
    /// [`SnnError::OutOfRange`] if `measured_var` is negative or non-finite.
    pub fn update_from_variance(&mut self, measured_var: f32) -> SnnResult<f32> {
        if measured_var < 0.0 || !measured_var.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "measured_var".to_string(),
                val: measured_var,
            });
        }
        let error = measured_var - self.cfg.target_variance;
        let proposed = self.feedback_rho - self.cfg.gain * error;
        let (lo, hi) = self.rho_bounds();
        self.feedback_rho = proposed.clamp(lo, hi);
        Ok(self.feedback_rho)
    }

    /// Reset the internal feedback target back to `ρ_start`.
    pub fn reset(&mut self) {
        self.feedback_rho = self.cfg.rho_start;
    }

    /// Rescale `w` in place so that its spectral radius becomes `current_rho`.
    ///
    /// Measures the present radius `ρ_meas` via power iteration and multiplies
    /// every entry by `current_rho / ρ_meas`. If the matrix is (numerically)
    /// nilpotent — `ρ_meas ≈ 0` — there is nothing to scale and the matrix is
    /// left untouched.
    ///
    /// # Errors
    ///
    /// * [`SnnError::BadDim`] if `n == 0`.
    /// * [`SnnError::BadShape`] if `w.len() != n * n`.
    /// * [`SnnError::OutOfRange`] if `current_rho` is negative or non-finite.
    pub fn rescale_in_place(&self, w: &mut [f32], n: usize, current_rho: f32) -> SnnResult<()> {
        if n == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if w.len() != n * n {
            return Err(SnnError::BadShape {
                expected: n * n,
                got: w.len(),
            });
        }
        if current_rho < 0.0 || !current_rho.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "current_rho".to_string(),
                val: current_rho,
            });
        }
        let measured = power_iteration_spectral_radius(w, n, POWER_ITERS);
        if measured > 1e-12 {
            let scale = current_rho / measured;
            for v in w.iter_mut() {
                *v *= scale;
            }
        }
        Ok(())
    }
}

/// Population variance of a slice (`0` for an empty slice). Helper for driving
/// the feedback controller from a window of reservoir states.
#[must_use]
pub fn state_variance(states: &[f32]) -> f32 {
    let len = states.len();
    if len == 0 {
        return 0.0;
    }
    let mean = states.iter().sum::<f32>() / len as f32;
    states.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / len as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a random `n × n` matrix with iid normal entries.
    fn random_matrix(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut w = vec![0.0_f32; n * n];
        rng.fill_normal(&mut w);
        w
    }

    #[test]
    fn rescale_hits_target_radius() {
        let n = 40;
        let mut w = random_matrix(n, 7);
        let sched =
            AdaptiveSpectralScheduler::new(AdaptiveSpectralConfig::default()).expect("ctor");
        let target = 0.85_f32;
        sched.rescale_in_place(&mut w, n, target).expect("rescale");
        let measured = power_iteration_spectral_radius(&w, n, 200);
        assert!(
            (measured - target).abs() < 0.05,
            "measured ρ={measured} not close to target {target}"
        );
    }

    #[test]
    fn rescale_idempotent_when_already_on_target() {
        // Rescaling twice to the same target must leave the radius unchanged.
        let n = 30;
        let mut w = random_matrix(n, 13);
        let sched =
            AdaptiveSpectralScheduler::new(AdaptiveSpectralConfig::default()).expect("ctor");
        sched.rescale_in_place(&mut w, n, 0.7).expect("rescale 1");
        let after_first = power_iteration_spectral_radius(&w, n, 200);
        sched.rescale_in_place(&mut w, n, 0.7).expect("rescale 2");
        let after_second = power_iteration_spectral_radius(&w, n, 200);
        assert!(
            (after_first - after_second).abs() < 1e-3,
            "radius drifted: {after_first} → {after_second}"
        );
    }

    #[test]
    fn linear_schedule_hits_endpoints() {
        let cfg = AdaptiveSpectralConfig {
            rho_start: 1.2,
            rho_end: 0.8,
            total_steps: 50,
            mode: ScheduleMode::Linear,
            ..AdaptiveSpectralConfig::default()
        };
        let sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        assert!((sched.current_rho(0) - 1.2).abs() < 1e-6);
        assert!((sched.current_rho(50) - 0.8).abs() < 1e-6);
        // Midpoint is the arithmetic mean.
        assert!((sched.current_rho(25) - 1.0).abs() < 1e-6);
        // Past the horizon clamps to the endpoint.
        assert!((sched.current_rho(123) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn exponential_schedule_hits_endpoints_and_is_geometric() {
        let cfg = AdaptiveSpectralConfig {
            rho_start: 2.0,
            rho_end: 0.5,
            total_steps: 100,
            mode: ScheduleMode::Exponential,
            ..AdaptiveSpectralConfig::default()
        };
        let sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        assert!((sched.current_rho(0) - 2.0).abs() < 1e-5);
        assert!((sched.current_rho(100) - 0.5).abs() < 1e-5);
        // Geometric midpoint = sqrt(2.0 * 0.5) = 1.0.
        assert!((sched.current_rho(50) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn feedback_lowers_rho_when_variance_exceeds_target() {
        let cfg = AdaptiveSpectralConfig {
            rho_start: 1.0,
            rho_end: 0.1,
            total_steps: 10,
            mode: ScheduleMode::FeedbackVarianceControl,
            target_variance: 0.2,
            gain: 0.1,
        };
        let mut sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        let before = sched.current_rho(0);
        // Variance well above target → saturation → radius decreases.
        let next = sched.update_from_variance(0.9).expect("update");
        assert!(next < before, "ρ should drop: {before} → {next}");
        assert!(next >= 0.1, "ρ must stay within band, got {next}");
    }

    #[test]
    fn feedback_raises_rho_when_variance_below_target() {
        let cfg = AdaptiveSpectralConfig {
            rho_start: 0.5,
            rho_end: 1.5,
            total_steps: 10,
            mode: ScheduleMode::FeedbackVarianceControl,
            target_variance: 0.4,
            gain: 0.2,
        };
        let mut sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        let before = sched.current_rho(0);
        // Variance below target → decay → radius increases.
        let next = sched.update_from_variance(0.0).expect("update");
        assert!(next > before, "ρ should rise: {before} → {next}");
        assert!(next <= 1.5, "ρ must stay within band, got {next}");
    }

    #[test]
    fn feedback_clamps_into_band() {
        let cfg = AdaptiveSpectralConfig {
            rho_start: 0.9,
            rho_end: 0.3,
            total_steps: 5,
            mode: ScheduleMode::FeedbackVarianceControl,
            target_variance: 0.1,
            gain: 100.0, // huge gain would overshoot without clamping
        };
        let mut sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        let next = sched.update_from_variance(10.0).expect("update");
        assert!((0.3..=0.9).contains(&next), "ρ={next} escaped band");
    }

    #[test]
    fn reset_restores_start() {
        let cfg = AdaptiveSpectralConfig {
            mode: ScheduleMode::FeedbackVarianceControl,
            ..AdaptiveSpectralConfig::default()
        };
        let mut sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        sched.update_from_variance(5.0).expect("update");
        sched.reset();
        assert!((sched.current_rho(0) - cfg.rho_start).abs() < 1e-6);
    }

    #[test]
    fn state_variance_matches_definition() {
        let s = [1.0_f32, 2.0, 3.0, 4.0];
        // mean = 2.5; var = mean((x-2.5)^2) = (2.25+0.25+0.25+2.25)/4 = 1.25.
        assert!((state_variance(&s) - 1.25).abs() < 1e-6);
        assert_eq!(state_variance(&[]), 0.0);
    }

    #[test]
    fn invalid_config_is_error() {
        let bad_rho = AdaptiveSpectralConfig {
            rho_start: 0.0,
            ..AdaptiveSpectralConfig::default()
        };
        assert!(matches!(
            AdaptiveSpectralScheduler::new(bad_rho),
            Err(SnnError::OutOfRange { .. })
        ));
        let bad_steps = AdaptiveSpectralConfig {
            total_steps: 0,
            ..AdaptiveSpectralConfig::default()
        };
        assert!(matches!(
            AdaptiveSpectralScheduler::new(bad_steps),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn rescale_rejects_bad_shape() {
        let sched =
            AdaptiveSpectralScheduler::new(AdaptiveSpectralConfig::default()).expect("ctor");
        let mut w = vec![0.0_f32; 9];
        assert!(matches!(
            sched.rescale_in_place(&mut w, 0, 0.9),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            sched.rescale_in_place(&mut w, 4, 0.9),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            sched.rescale_in_place(&mut w, 3, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn update_rejects_negative_variance() {
        let cfg = AdaptiveSpectralConfig {
            mode: ScheduleMode::FeedbackVarianceControl,
            ..AdaptiveSpectralConfig::default()
        };
        let mut sched = AdaptiveSpectralScheduler::new(cfg).expect("ctor");
        assert!(matches!(
            sched.update_from_variance(-0.1),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn deterministic_given_seed() {
        let n = 20;
        let mut w1 = random_matrix(n, 99);
        let mut w2 = random_matrix(n, 99);
        let sched =
            AdaptiveSpectralScheduler::new(AdaptiveSpectralConfig::default()).expect("ctor");
        sched.rescale_in_place(&mut w1, n, 0.6).expect("rescale 1");
        sched.rescale_in_place(&mut w2, n, 0.6).expect("rescale 2");
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-9, "non-deterministic rescale");
        }
    }
}
