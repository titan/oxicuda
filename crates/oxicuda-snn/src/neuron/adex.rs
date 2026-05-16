//! Adaptive Exponential Integrate-and-Fire (AdEx) neuron — Brette & Gerstner 2005.
//!
//! ```text
//! C  · dv/dt = −g_L · (v − E_L) + g_L · Δ_T · exp((v − V_T)/Δ_T) − w + I
//! τ_w · dw/dt = a · (v − E_L) − w
//! if v > 0  → v ← v_r,  w ← w + b,  spike = 1
//! ```
//!
//! Default parameters use the canonical Brette-Gerstner 2005 values.

use crate::error::{SnnError, SnnResult};

/// AdEx parameter block; all positive constants are validated at runtime.
#[derive(Debug, Clone, Copy)]
pub struct AdexConfig {
    /// Membrane capacitance C in pF.
    pub c: f32,
    /// Leak conductance g_L in nS.
    pub g_l: f32,
    /// Resting / leak reversal potential E_L in mV.
    pub e_l: f32,
    /// Slope factor Δ_T of the exponential term in mV.
    pub delta_t: f32,
    /// Soft threshold V_T in mV.
    pub v_t: f32,
    /// Adaptation time constant τ_w in ms.
    pub tau_w: f32,
    /// Subthreshold adaptation conductance a in nS.
    pub a: f32,
    /// Spike-triggered adaptation increment b in pA.
    pub b: f32,
    /// After-spike reset potential v_r in mV.
    pub v_r: f32,
    /// Integration step in ms.
    pub dt: f32,
}

impl Default for AdexConfig {
    /// Brette & Gerstner 2005 reference parameters.
    fn default() -> Self {
        Self {
            c: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            delta_t: 2.0,
            v_t: -50.4,
            tau_w: 144.0,
            a: 4.0,
            b: 0.0805,
            v_r: -70.6,
            dt: 0.1,
        }
    }
}

/// AdEx state: membrane `v` and adaptation `w` per neuron.
#[derive(Debug, Clone)]
pub struct AdexState {
    /// Membrane potential `v_i` per neuron, mV.
    pub v: Vec<f32>,
    /// Adaptation current `w_i` per neuron, pA.
    pub w: Vec<f32>,
}

impl AdexState {
    /// Allocate state for `n` neurons; `v ← e_l`, `w ← 0`.
    #[must_use]
    pub fn new(n: usize, e_l: f32) -> Self {
        Self {
            v: vec![e_l; n],
            w: vec![0.0_f32; n],
        }
    }
}

const V_LO: f32 = -100.0;
const V_HI: f32 = 50.0;
const EXP_ARG_CAP: f32 = 50.0;

fn validate(
    state: &AdexState,
    current: &[f32],
    cfg: &AdexConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if cfg.c <= 0.0 || !cfg.c.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "C".into(),
            val: cfg.c,
        });
    }
    if cfg.g_l <= 0.0 || !cfg.g_l.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "g_L".into(),
            val: cfg.g_l,
        });
    }
    if cfg.delta_t <= 0.0 || !cfg.delta_t.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "delta_t".into(),
            val: cfg.delta_t,
        });
    }
    if cfg.tau_w <= 0.0 || !cfg.tau_w.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "tau_w".into(),
            val: cfg.tau_w,
        });
    }
    let n = state.v.len();
    if state.w.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.w.len(),
        });
    }
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

/// Advance an AdEx population by one timestep.
pub fn adex_step(
    state: &mut AdexState,
    current: &[f32],
    cfg: &AdexConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate(state, current, cfg, spikes_out)?;
    for (((v, w), &i_in), s_out) in state
        .v
        .iter_mut()
        .zip(state.w.iter_mut())
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        let exp_arg = ((*v - cfg.v_t) / cfg.delta_t).min(EXP_ARG_CAP);
        let dv =
            (-cfg.g_l * (*v - cfg.e_l) + cfg.g_l * cfg.delta_t * exp_arg.exp() - *w + i_in) / cfg.c;
        let mut v_loc = *v + cfg.dt * dv;

        let dw = (cfg.a * (v_loc - cfg.e_l) - *w) / cfg.tau_w;
        let mut w_loc = *w + cfg.dt * dw;

        let spike = if v_loc > 0.0 {
            v_loc = cfg.v_r;
            w_loc += cfg.b;
            1.0_f32
        } else {
            0.0_f32
        };

        v_loc = v_loc.clamp(V_LO, V_HI);

        *v = v_loc;
        *w = w_loc;
        *s_out = spike;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_finite() {
        let cfg = AdexConfig::default();
        let mut state = AdexState::new(8, cfg.e_l);
        assert_eq!(state.v.len(), 8);
        assert_eq!(state.w.len(), 8);
        let current = vec![400.0_f32; 8];
        let mut spikes = vec![0.0_f32; 8];
        for _ in 0..1000 {
            adex_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        }
        for &v in &state.v {
            assert!(v.is_finite(), "v={v}");
        }
        for &w in &state.w {
            assert!(w.is_finite(), "w={w}");
        }
    }

    #[test]
    fn bursting_preset_finite() {
        // Bursting parameter set from Naud et al. 2008 (initial bursting).
        let cfg = AdexConfig {
            c: 200.0,
            g_l: 10.0,
            e_l: -58.0,
            delta_t: 2.0,
            v_t: -50.0,
            tau_w: 120.0,
            a: 2.0,
            b: 100.0,
            v_r: -46.0,
            dt: 0.1,
        };
        let mut state = AdexState::new(4, cfg.e_l);
        let current = vec![500.0_f32; 4];
        let mut spikes = vec![0.0_f32; 4];
        let mut total = 0_usize;
        for _ in 0..2000 {
            adex_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            total += spikes.iter().filter(|&&s| s == 1.0).count();
        }
        assert!(total > 0, "bursting preset should spike");
        for &v in &state.v {
            assert!(v.is_finite() && (V_LO..=V_HI).contains(&v));
        }
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = AdexConfig::default();
        let mut state = AdexState::new(2, cfg.e_l);
        let current = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = adex_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn rejects_zero_capacitance() {
        let cfg = AdexConfig {
            c: 0.0,
            ..AdexConfig::default()
        };
        let mut state = AdexState::new(2, cfg.e_l);
        let current = vec![0.0_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        let err = adex_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
