//! Leaky Integrate-and-Fire (LIF) neuron.
//!
//! Discrete-time LIF dynamics:
//!
//! ```text
//! v_{t+1} = β · v_t + I_t,    β = exp(-dt / τ_m)
//! s_{t+1} = 1 if v_{t+1} ≥ v_th else 0
//! v_{t+1} ← (1 − s) · v_{t+1} + s · v_rest      (Hard reset)
//! v_{t+1} ← v_{t+1} − s · v_th                  (Soft / subtractive reset)
//! ```

use crate::error::{SnnError, SnnResult};

/// Reset behaviour after a spike.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ResetMode {
    /// Hard reset: membrane potential is set to `v_rest` after a spike.
    #[default]
    Hard,
    /// Soft reset: `v_th` is subtracted from the membrane after a spike.
    Soft,
}

/// LIF configuration; `tau_m` and `dt` must be strictly positive.
#[derive(Debug, Clone, Copy)]
pub struct LifConfig {
    /// Membrane time constant `τ_m` in the same time units as `dt`.
    pub tau_m: f32,
    /// Spike threshold.
    pub v_th: f32,
    /// Reset / equilibrium potential.
    pub v_rest: f32,
    /// Integration time step.
    pub dt: f32,
    /// Reset mode applied after a spike.
    pub reset: ResetMode,
}

impl Default for LifConfig {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        }
    }
}

/// Mutable LIF state (membrane potential per neuron).
#[derive(Debug, Clone)]
pub struct LifState {
    /// Membrane potential `v_i` for each neuron, length `n`.
    pub v: Vec<f32>,
}

impl LifState {
    /// Allocate state for `n` neurons with `v` initialised to zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            v: vec![0.0_f32; n],
        }
    }
}

/// Membrane decay factor `β = exp(−dt / τ_m)`.
#[must_use]
pub fn beta(cfg: &LifConfig) -> f32 {
    (-cfg.dt / cfg.tau_m).exp()
}

/// Validate `cfg` and slice lengths used by [`lif_step`].
fn validate(
    state: &LifState,
    current: &[f32],
    cfg: &LifConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_m });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.v_th.is_finite() {
        return Err(SnnError::BadThreshold { v_th: cfg.v_th });
    }
    let n = state.v.len();
    if current.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: current.len(),
        });
    }
    if spikes_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: spikes_out.len(),
        });
    }
    Ok(())
}

/// Advance the LIF state by one timestep.
///
/// `current` is the input current `I_t`, length must match `state.v`.
/// `spikes_out` receives `1.0` where a spike occurred, `0.0` elsewhere.
pub fn lif_step(
    state: &mut LifState,
    current: &[f32],
    cfg: &LifConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate(state, current, cfg, spikes_out)?;
    let b = beta(cfg);
    for ((v, &i_in), s_out) in state
        .v
        .iter_mut()
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        let v_new = b * *v + i_in;
        let spike = if v_new >= cfg.v_th { 1.0_f32 } else { 0.0_f32 };
        *v = match cfg.reset {
            ResetMode::Hard => (1.0 - spike) * v_new + spike * cfg.v_rest,
            ResetMode::Soft => v_new - spike * cfg.v_th,
        };
        *s_out = spike;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_correct() {
        let cfg = LifConfig {
            tau_m: 10.0,
            dt: 1.0,
            ..Default::default()
        };
        let b = beta(&cfg);
        assert!((b - (-0.1_f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn zero_input_exponential_decay() {
        let cfg = LifConfig {
            tau_m: 10.0,
            v_th: 100.0, // unreachable
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = LifState::new(1);
        state.v[0] = 1.0;
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let steps = 50_usize;
        for _ in 0..steps {
            lif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        }
        let expected = (-((steps as f32) * cfg.dt) / cfg.tau_m).exp();
        assert!(
            (state.v[0] - expected).abs() < 1e-4,
            "v={} expected={}",
            state.v[0],
            expected
        );
    }

    #[test]
    fn ramp_input_spikes_count_hard() {
        let cfg = LifConfig {
            tau_m: 1e9, // ~no leak
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = LifState::new(1);
        let current = vec![0.1_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..100 {
            lif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        // With v=0.1*t and reset to 0, expect ~10 spikes in 100 steps.
        assert!((9..=11).contains(&count), "spike count={count}");
    }

    #[test]
    fn hard_reset_clears_to_v_rest() {
        let cfg = LifConfig {
            tau_m: 1e9,
            v_th: 0.5,
            v_rest: -0.25,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = LifState::new(1);
        state.v[0] = 0.4;
        let current = vec![0.2_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        lif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert!((state.v[0] - cfg.v_rest).abs() < 1e-5);
    }

    #[test]
    fn soft_reset_subtracts_v_th() {
        let cfg = LifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Soft,
        };
        let mut state = LifState::new(1);
        state.v[0] = 0.8;
        let current = vec![0.5_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        lif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        // v_new = 0.8 + 0.5 = 1.3; spike=1; v ← 1.3 − 1.0 = 0.3
        assert_eq!(spikes[0], 1.0);
        assert!((state.v[0] - 0.3).abs() < 1e-5, "v={}", state.v[0]);
    }

    #[test]
    fn rejects_bad_tau() {
        let cfg = LifConfig {
            tau_m: 0.0,
            ..Default::default()
        };
        let mut state = LifState::new(2);
        let current = vec![0.0_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        let err = lif_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = LifConfig::default();
        let mut state = LifState::new(2);
        let current = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = lif_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
