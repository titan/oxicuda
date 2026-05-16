//! Pure Integrate-and-Fire neuron (no leak).
//!
//! ```text
//! v_{t+1} = v_t + I_t
//! s_{t+1} = 1 if v_{t+1} ≥ v_th else 0
//! v_{t+1} ← (1 − s) · v_{t+1} + s · v_rest    (Hard reset)
//! v_{t+1} ← v_{t+1} − s · v_th                (Soft reset)
//! ```

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::ResetMode;

/// Configuration for a pure Integrate-and-Fire neuron.
#[derive(Debug, Clone, Copy)]
pub struct IfConfig {
    /// Spike threshold.
    pub v_th: f32,
    /// Reset / equilibrium potential (used in [`ResetMode::Hard`]).
    pub v_rest: f32,
    /// Reset mode applied after a spike.
    pub reset: ResetMode,
}

impl Default for IfConfig {
    fn default() -> Self {
        Self {
            v_th: 1.0,
            v_rest: 0.0,
            reset: ResetMode::Hard,
        }
    }
}

/// Mutable state of a pure Integrate-and-Fire neuron.
#[derive(Debug, Clone)]
pub struct IfState {
    /// Membrane potential `v_i` per neuron, length `n`.
    pub v: Vec<f32>,
}

impl IfState {
    /// Allocate state for `n` neurons with `v` initialised to zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            v: vec![0.0_f32; n],
        }
    }
}

fn validate(state: &IfState, current: &[f32], cfg: &IfConfig, spikes_out: &[f32]) -> SnnResult<()> {
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

/// Advance the IF state by one timestep.
pub fn if_step(
    state: &mut IfState,
    current: &[f32],
    cfg: &IfConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate(state, current, cfg, spikes_out)?;
    for ((v, &i_in), s_out) in state
        .v
        .iter_mut()
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        let v_new = *v + i_in;
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
    fn ramp_spikes_after_v_th_over_i() {
        let cfg = IfConfig {
            v_th: 1.0,
            v_rest: 0.0,
            reset: ResetMode::Hard,
        };
        let mut state = IfState::new(1);
        let current = vec![0.1_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let mut first_spike: Option<usize> = None;
        for t in 0..50 {
            if_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            if spikes[0] == 1.0 && first_spike.is_none() {
                first_spike = Some(t);
                break;
            }
        }
        // v reaches 1.0 after 10 steps of 0.1 (1-indexed = 10th step, 0-indexed = 9)
        assert_eq!(first_spike, Some(9));
    }

    #[test]
    fn hard_reset_to_v_rest() {
        let cfg = IfConfig {
            v_th: 1.0,
            v_rest: -0.5,
            reset: ResetMode::Hard,
        };
        let mut state = IfState::new(1);
        state.v[0] = 0.5;
        let current = vec![0.6_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        if_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert!((state.v[0] - cfg.v_rest).abs() < 1e-6);
    }

    #[test]
    fn soft_reset_subtracts_v_th() {
        let cfg = IfConfig {
            v_th: 1.0,
            v_rest: 0.0,
            reset: ResetMode::Soft,
        };
        let mut state = IfState::new(1);
        state.v[0] = 0.7;
        let current = vec![0.5_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        if_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert!((state.v[0] - 0.2).abs() < 1e-5, "v={}", state.v[0]);
    }

    #[test]
    fn no_spike_below_threshold() {
        let cfg = IfConfig::default();
        let mut state = IfState::new(2);
        let current = vec![0.3_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        if_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes, vec![0.0, 0.0]);
        assert!((state.v[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = IfConfig::default();
        let mut state = IfState::new(2);
        let current = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = if_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
