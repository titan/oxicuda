//! Layer-wise Adaptive Temperature Scheduling (LATS).
//!
//! In multi-layer knowledge distillation (TinyBERT, PKD, etc.) a single global temperature
//! forces all layers to operate under the same supervision sharpness. LATS instead maintains a
//! per-layer temperature that adapts during training:
//!
//!   - Layers with high teacher-student KL divergence receive a higher temperature (softer
//!     targets → easier-to-match distribution), accelerating initial alignment.
//!   - Well-aligned layers receive a progressively lower temperature (sharper targets),
//!     squeezing out remaining capacity.
//!
//! Temperature updates use a sign-based gradient on an EMA-smoothed KL divergence,
//! clipped to `[t_min, t_max]`.

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for layer-wise adaptive temperature scheduling.
#[derive(Debug, Clone)]
pub struct AdaptiveTempConfig {
    /// Number of layers to track.
    pub n_layers: usize,
    /// Minimum allowed temperature (> 0).
    pub t_min: f32,
    /// Maximum allowed temperature (> t_min).
    pub t_max: f32,
    /// Initial temperature for every layer (must lie in [t_min, t_max]).
    pub t_init: f32,
    /// EMA decay factor for divergence smoothing, ∈ [0, 1).
    pub ema_decay: f32,
    /// Update temperatures every `update_freq` steps (currently informational;
    /// `update_layer` is always applied when called).
    pub update_freq: usize,
    /// Target KL divergence per layer (EMA converges toward this).
    pub target_divergence: f32,
    /// Step size for temperature adjustment (> 0).
    pub adaptation_rate: f32,
}

impl Default for AdaptiveTempConfig {
    fn default() -> Self {
        Self {
            n_layers: 1,
            t_min: 0.5,
            t_max: 20.0,
            t_init: 4.0,
            ema_decay: 0.9,
            update_freq: 1,
            target_divergence: 0.1,
            adaptation_rate: 0.05,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutable state
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer adaptive temperature state, updated in-place during training.
pub struct AdaptiveTempState {
    /// Current temperature for each layer `[n_layers]`.
    pub temperatures: Vec<f32>,
    /// EMA of the observed KL divergence for each layer `[n_layers]`.
    pub ema_divergences: Vec<f32>,
    /// Total number of `multi_layer_loss` calls performed so far.
    pub step: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless scheduler; all mutable state lives in [`AdaptiveTempState`].
pub struct AdaptiveTempScheduler;

impl AdaptiveTempScheduler {
    /// Initialise state with uniform temperatures and config-derived EMA baselines.
    ///
    /// # Errors
    /// Returns [`DistillError::InvalidConfig`] if any configuration constraint is violated.
    pub fn init(cfg: &AdaptiveTempConfig) -> DistillResult<AdaptiveTempState> {
        Self::validate_config(cfg)?;
        Ok(AdaptiveTempState {
            temperatures: vec![cfg.t_init; cfg.n_layers],
            ema_divergences: vec![cfg.target_divergence; cfg.n_layers],
            step: 0,
        })
    }

    /// Compute the adaptive-temperature KD loss for a single layer.
    ///
    /// Uses `state.temperatures[layer_idx]` as the softmax temperature.
    /// Both logit vectors must be non-empty and have equal length.
    ///
    /// Returns `(loss, raw_kl)` where:
    ///   - `loss = T² · KL(softmax(t/T) ‖ softmax(s/T))` (scaled KL distillation signal).
    ///   - `raw_kl = KL(p_t ‖ p_s)` at the current temperature (unscaled, used for EMA).
    pub fn layer_kd_loss(
        state: &AdaptiveTempState,
        layer_idx: usize,
        s_logits: &[f32],
        t_logits: &[f32],
        label: usize,
    ) -> DistillResult<(f32, f32)> {
        if layer_idx >= state.temperatures.len() {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "layer_idx {} >= n_layers {}",
                    layer_idx,
                    state.temperatures.len()
                ),
            });
        }
        if s_logits.is_empty() || t_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if s_logits.len() != t_logits.len() {
            return Err(DistillError::DimensionMismatch {
                expected: s_logits.len(),
                got: t_logits.len(),
            });
        }
        if label >= s_logits.len() {
            return Err(DistillError::InvalidConfig {
                msg: format!("label {} >= num_classes {}", label, s_logits.len()),
            });
        }

        let temp = state.temperatures[layer_idx];
        let p_s = softmax_with_temp(s_logits, temp);
        let p_t = softmax_with_temp(t_logits, temp);

        // Scaled KL (T²·KL) — the distillation loss returned to the caller.
        let raw_kl = kl_divergence(&p_t, &p_s);
        let scaled_loss = temp * temp * raw_kl;

        let loss = Self::check_finite(scaled_loss, "layer_kd_loss")?;
        let kl_out = Self::check_finite(raw_kl, "layer_kd_loss kl")?;
        Ok((loss, kl_out))
    }

    /// Update the EMA divergence and temperature for one layer.
    ///
    /// EMA update: `ema[i] ← decay · ema[i] + (1 − decay) · observed_kl`
    /// Temperature update (sign-based gradient):
    ///   - `ema > target` → increase T by `adaptation_rate` (soften target)
    ///   - `ema < target` → decrease T by `adaptation_rate` (sharpen target)
    ///   - T is clamped to `[t_min, t_max]` after each step.
    ///
    /// Note: the `step` counter is **not** incremented here; it is incremented once per
    /// call to [`AdaptiveTempScheduler::multi_layer_loss`].
    pub fn update_layer(
        state: &mut AdaptiveTempState,
        layer_idx: usize,
        observed_kl: f32,
        cfg: &AdaptiveTempConfig,
    ) -> DistillResult<()> {
        if layer_idx >= state.temperatures.len() {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "layer_idx {} >= n_layers {}",
                    layer_idx,
                    state.temperatures.len()
                ),
            });
        }

        // EMA smoothing.
        let ema =
            cfg.ema_decay * state.ema_divergences[layer_idx] + (1.0 - cfg.ema_decay) * observed_kl;
        state.ema_divergences[layer_idx] = ema;

        // Sign-based temperature step.
        let delta = if ema > cfg.target_divergence {
            cfg.adaptation_rate
        } else {
            -cfg.adaptation_rate
        };
        state.temperatures[layer_idx] =
            (state.temperatures[layer_idx] + delta).clamp(cfg.t_min, cfg.t_max);

        Ok(())
    }

    /// Compute distillation losses for all layers and update their temperatures.
    ///
    /// `layer_logits` must have exactly `n_layers` entries, each a
    /// `(student_logits, teacher_logits, label)` triple.
    ///
    /// After processing all layers the global `step` counter is incremented once.
    ///
    /// Returns a `Vec<f32>` of per-layer scaled KL losses.
    pub fn multi_layer_loss(
        state: &mut AdaptiveTempState,
        layer_logits: &[(Vec<f32>, Vec<f32>, usize)],
        cfg: &AdaptiveTempConfig,
    ) -> DistillResult<Vec<f32>> {
        if layer_logits.len() != state.temperatures.len() {
            return Err(DistillError::DimensionMismatch {
                expected: state.temperatures.len(),
                got: layer_logits.len(),
            });
        }

        let mut losses = Vec::with_capacity(layer_logits.len());
        for (i, (s_logits, t_logits, label)) in layer_logits.iter().enumerate() {
            let (loss, kl) = Self::layer_kd_loss(state, i, s_logits, t_logits, *label)?;
            Self::update_layer(state, i, kl, cfg)?;
            losses.push(loss);
        }

        state.step += 1;
        Ok(losses)
    }

    /// Total distillation loss = arithmetic mean of per-layer losses.
    pub fn total_loss(per_layer_losses: &[f32]) -> DistillResult<f32> {
        if per_layer_losses.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let mean = per_layer_losses.iter().sum::<f32>() / per_layer_losses.len() as f32;
        Self::check_finite(mean, "total_loss")
    }

    /// Reset state to the initial temperatures and divergence baselines.
    ///
    /// Useful for continual-learning scenarios where a new task begins.
    pub fn reset(state: &mut AdaptiveTempState, cfg: &AdaptiveTempConfig) {
        for t in &mut state.temperatures {
            *t = cfg.t_init;
        }
        for d in &mut state.ema_divergences {
            *d = cfg.target_divergence;
        }
        state.step = 0;
    }

    /// Return the current temperature for `layer_idx`.
    pub fn temperature(state: &AdaptiveTempState, layer_idx: usize) -> DistillResult<f32> {
        if layer_idx >= state.temperatures.len() {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "layer_idx {} >= n_layers {}",
                    layer_idx,
                    state.temperatures.len()
                ),
            });
        }
        Ok(state.temperatures[layer_idx])
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn validate_config(cfg: &AdaptiveTempConfig) -> DistillResult<()> {
        if cfg.n_layers == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_layers must be >= 1".to_owned(),
            });
        }
        if cfg.t_min <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("t_min must be > 0, got {}", cfg.t_min),
            });
        }
        if cfg.t_max <= cfg.t_min {
            return Err(DistillError::InvalidConfig {
                msg: format!("t_max ({}) must be > t_min ({})", cfg.t_max, cfg.t_min),
            });
        }
        if cfg.t_init < cfg.t_min || cfg.t_init > cfg.t_max {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "t_init ({}) must be in [t_min={}, t_max={}]",
                    cfg.t_init, cfg.t_min, cfg.t_max
                ),
            });
        }
        if cfg.ema_decay < 0.0 || cfg.ema_decay >= 1.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("ema_decay must be in [0, 1), got {}", cfg.ema_decay),
            });
        }
        if cfg.adaptation_rate <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("adaptation_rate must be > 0, got {}", cfg.adaptation_rate),
            });
        }
        Ok(())
    }

    fn check_finite(v: f32, ctx: &str) -> DistillResult<f32> {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(DistillError::NumericalError {
                msg: format!("{ctx}: non-finite value {v}"),
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(n: usize) -> AdaptiveTempConfig {
        AdaptiveTempConfig {
            n_layers: n,
            t_min: 0.5,
            t_max: 20.0,
            t_init: 4.0,
            ema_decay: 0.9,
            update_freq: 1,
            target_divergence: 0.1,
            adaptation_rate: 0.05,
        }
    }

    // ── new ───────────────────────────────────────────────────────────────────

    #[test]
    fn new_valid_config_temperatures_equal_t_init() {
        let cfg = default_cfg(3);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        assert_eq!(state.temperatures.len(), 3);
        for &t in &state.temperatures {
            assert!((t - 4.0_f32).abs() < 1e-7);
        }
    }

    #[test]
    fn new_zero_layers_returns_err() {
        let cfg = default_cfg(0);
        let result = AdaptiveTempScheduler::init(&cfg);
        assert!(matches!(result, Err(DistillError::InvalidConfig { .. })));
    }

    #[test]
    fn new_t_min_le_zero_returns_err() {
        let mut cfg = default_cfg(2);
        cfg.t_min = 0.0;
        assert!(AdaptiveTempScheduler::init(&cfg).is_err());
    }

    #[test]
    fn new_t_max_le_t_min_returns_err() {
        let mut cfg = default_cfg(2);
        cfg.t_max = cfg.t_min;
        assert!(AdaptiveTempScheduler::init(&cfg).is_err());
    }

    #[test]
    fn new_t_init_out_of_range_returns_err() {
        let mut cfg = default_cfg(2);
        cfg.t_init = 100.0; // > t_max
        assert!(AdaptiveTempScheduler::init(&cfg).is_err());
    }

    // ── temperature ───────────────────────────────────────────────────────────

    #[test]
    fn temperature_valid_idx() {
        let cfg = default_cfg(2);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let t = AdaptiveTempScheduler::temperature(&state, 0).unwrap();
        assert!((t - 4.0_f32).abs() < 1e-7);
    }

    #[test]
    fn temperature_out_of_range_returns_err() {
        let cfg = default_cfg(2);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let result = AdaptiveTempScheduler::temperature(&state, 5);
        assert!(matches!(result, Err(DistillError::InvalidConfig { .. })));
    }

    // ── layer_kd_loss ─────────────────────────────────────────────────────────

    #[test]
    fn layer_kd_loss_identical_logits_near_zero_kl() {
        let cfg = default_cfg(1);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let logits = vec![1.0_f32, 2.0, 3.0];
        let (loss, kl) =
            AdaptiveTempScheduler::layer_kd_loss(&state, 0, &logits, &logits, 2).unwrap();
        assert!(loss.abs() < 1e-5, "loss={loss}");
        assert!(kl.abs() < 1e-5, "kl={kl}");
    }

    #[test]
    fn layer_kd_loss_mismatched_lengths_returns_err() {
        let cfg = default_cfg(1);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let s = vec![1.0_f32, 2.0];
        let t = vec![1.0_f32, 2.0, 3.0];
        let result = AdaptiveTempScheduler::layer_kd_loss(&state, 0, &s, &t, 0);
        assert!(matches!(
            result,
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn layer_kd_loss_empty_logits_returns_err() {
        let cfg = default_cfg(1);
        let state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let result = AdaptiveTempScheduler::layer_kd_loss(&state, 0, &[], &[], 0);
        assert!(matches!(result, Err(DistillError::EmptyInput)));
    }

    // ── update_layer ──────────────────────────────────────────────────────────

    #[test]
    fn update_layer_high_divergence_increases_temperature() {
        let cfg = default_cfg(1);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let t_before = state.temperatures[0];
        // observed_kl >> target_divergence → temperature should increase
        AdaptiveTempScheduler::update_layer(&mut state, 0, 10.0, &cfg).unwrap();
        assert!(
            state.temperatures[0] > t_before,
            "T should increase: was {t_before}, now {}",
            state.temperatures[0]
        );
    }

    #[test]
    fn update_layer_low_divergence_decreases_temperature() {
        let cfg = default_cfg(1);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let t_before = state.temperatures[0];
        // observed_kl << target_divergence → temperature should decrease
        AdaptiveTempScheduler::update_layer(&mut state, 0, 0.0, &cfg).unwrap();
        assert!(
            state.temperatures[0] < t_before,
            "T should decrease: was {t_before}, now {}",
            state.temperatures[0]
        );
    }

    #[test]
    fn update_layer_temperature_clamped_to_t_max() {
        let mut cfg = default_cfg(1);
        cfg.t_init = cfg.t_max; // start at ceiling
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        // Push divergence very high → would want to increase T beyond t_max
        for _ in 0..20 {
            AdaptiveTempScheduler::update_layer(&mut state, 0, 1000.0, &cfg).unwrap();
        }
        assert!(
            state.temperatures[0] <= cfg.t_max,
            "T exceeded t_max: {}",
            state.temperatures[0]
        );
    }

    #[test]
    fn update_layer_temperature_clamped_to_t_min() {
        let mut cfg = default_cfg(1);
        cfg.t_init = cfg.t_min; // start at floor
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        // Push divergence to zero → would want to decrease T below t_min
        for _ in 0..20 {
            AdaptiveTempScheduler::update_layer(&mut state, 0, 0.0, &cfg).unwrap();
        }
        assert!(
            state.temperatures[0] >= cfg.t_min,
            "T went below t_min: {}",
            state.temperatures[0]
        );
    }

    // ── multi_layer_loss ──────────────────────────────────────────────────────

    #[test]
    fn multi_layer_loss_returns_n_layer_losses() {
        let cfg = default_cfg(3);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let logits = vec![1.0_f32, 2.0, 3.0];
        let layer_data: Vec<(Vec<f32>, Vec<f32>, usize)> = (0..3)
            .map(|_| (logits.clone(), logits.clone(), 0))
            .collect();
        let losses =
            AdaptiveTempScheduler::multi_layer_loss(&mut state, &layer_data, &cfg).unwrap();
        assert_eq!(losses.len(), 3);
    }

    #[test]
    fn multi_layer_loss_wrong_n_layers_returns_err() {
        let cfg = default_cfg(3);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let logits = vec![1.0_f32, 2.0];
        let layer_data: Vec<(Vec<f32>, Vec<f32>, usize)> = (0..2)
            .map(|_| (logits.clone(), logits.clone(), 0))
            .collect();
        let result = AdaptiveTempScheduler::multi_layer_loss(&mut state, &layer_data, &cfg);
        assert!(matches!(
            result,
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn multi_layer_loss_increments_step() {
        let cfg = default_cfg(2);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        assert_eq!(state.step, 0);
        let logits = vec![1.0_f32, 2.0];
        let layer_data: Vec<(Vec<f32>, Vec<f32>, usize)> = (0..2)
            .map(|_| (logits.clone(), logits.clone(), 0))
            .collect();
        AdaptiveTempScheduler::multi_layer_loss(&mut state, &layer_data, &cfg).unwrap();
        assert_eq!(state.step, 1);
        AdaptiveTempScheduler::multi_layer_loss(&mut state, &layer_data, &cfg).unwrap();
        assert_eq!(state.step, 2);
    }

    #[test]
    fn ema_divergence_updates_correctly() {
        let cfg = default_cfg(1);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        // Initial EMA = target_divergence = 0.1.
        // observed_kl = 0.5 → new EMA = 0.9 * 0.1 + 0.1 * 0.5 = 0.09 + 0.05 = 0.14
        AdaptiveTempScheduler::update_layer(&mut state, 0, 0.5, &cfg).unwrap();
        let expected_ema = 0.9_f32 * 0.1_f32 + 0.1_f32 * 0.5_f32;
        assert!(
            (state.ema_divergences[0] - expected_ema).abs() < 1e-6,
            "EMA expected {expected_ema}, got {}",
            state.ema_divergences[0]
        );
    }

    // ── total_loss ────────────────────────────────────────────────────────────

    #[test]
    fn total_loss_mean_of_vec() {
        let losses = vec![1.0_f32, 2.0, 3.0];
        let total = AdaptiveTempScheduler::total_loss(&losses).unwrap();
        assert!((total - 2.0_f32).abs() < 1e-6);
    }

    #[test]
    fn total_loss_empty_returns_err() {
        let result = AdaptiveTempScheduler::total_loss(&[]);
        assert!(matches!(result, Err(DistillError::EmptyInput)));
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn reset_restores_initial_temperatures_and_step() {
        let cfg = default_cfg(2);
        let mut state = AdaptiveTempScheduler::init(&cfg).unwrap();
        let logits = vec![1.0_f32, 2.0];
        let layer_data: Vec<(Vec<f32>, Vec<f32>, usize)> = (0..2)
            .map(|_| (logits.clone(), logits.clone(), 0))
            .collect();
        // Run a few steps to alter state.
        for _ in 0..5 {
            AdaptiveTempScheduler::multi_layer_loss(&mut state, &layer_data, &cfg).unwrap();
        }
        AdaptiveTempScheduler::reset(&mut state, &cfg);
        assert_eq!(state.step, 0);
        for &t in &state.temperatures {
            assert!((t - cfg.t_init).abs() < 1e-7);
        }
        for &d in &state.ema_divergences {
            assert!((d - cfg.target_divergence).abs() < 1e-7);
        }
    }
}
