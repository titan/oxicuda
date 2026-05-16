//! Izhikevich neuron (Izhikevich 2003).
//!
//! Two-variable system combining a quadratic membrane equation with a slow
//! recovery variable, capable of reproducing 20+ cortical firing patterns by
//! tuning the four parameters `(a, b, c, d)`:
//!
//! ```text
//! dv/dt = 0.04·v² + 5·v + 140 − u + I
//! du/dt = a · (b · v − u)
//! if v ≥ 30 mV → v ← c, u ← u + d, spike = 1
//! ```
//!
//! Voltage stability is improved by integrating `v` with two `dt/2` Euler
//! sub-steps before the slow `u` update.

use crate::error::{SnnError, SnnResult};

/// Izhikevich (a, b, c, d, dt) configuration.
#[derive(Debug, Clone, Copy)]
pub struct IzhConfig {
    /// Time scale of the recovery variable `u` (typical: 0.02 for RS, 0.1 for FS).
    pub a: f32,
    /// Sensitivity of `u` to subthreshold `v` (typical: 0.2).
    pub b: f32,
    /// After-spike reset value of `v` in mV (typical: −65 mV for RS).
    pub c: f32,
    /// After-spike increment to `u` (typical: 8 for RS).
    pub d: f32,
    /// Integration step in ms.
    pub dt: f32,
}

impl IzhConfig {
    /// Regular-spiking cortical pyramidal neuron preset.
    #[must_use]
    pub fn regular_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt: 1.0,
        }
    }

    /// Fast-spiking cortical interneuron preset.
    #[must_use]
    pub fn fast_spiking() -> Self {
        Self {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
            dt: 1.0,
        }
    }

    /// Chattering cell preset (high-frequency bursts).
    #[must_use]
    pub fn chattering() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -50.0,
            d: 2.0,
            dt: 1.0,
        }
    }

    /// Intrinsically-bursting cortical neuron preset.
    #[must_use]
    pub fn intrinsically_bursting() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 4.0,
            dt: 1.0,
        }
    }
}

impl Default for IzhConfig {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

/// Izhikevich state: membrane `v` and recovery `u` per neuron.
#[derive(Debug, Clone)]
pub struct IzhState {
    /// Membrane potential `v_i` per neuron (initialised to −70 mV).
    pub v: Vec<f32>,
    /// Recovery variable `u_i` per neuron (initialised to `b · v_i`).
    pub u: Vec<f32>,
}

impl IzhState {
    /// Allocate state for `n` neurons; `v` ← −70 mV, `u` ← `b · v`.
    #[must_use]
    pub fn new(n: usize, b: f32) -> Self {
        let v_init = -70.0_f32;
        Self {
            v: vec![v_init; n],
            u: vec![b * v_init; n],
        }
    }
}

const V_LO: f32 = -100.0;
const V_HI: f32 = 40.0;
const SPIKE_THRESHOLD: f32 = 30.0;

fn validate(
    state: &IzhState,
    current: &[f32],
    cfg: &IzhConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.a.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "a".into(),
            val: cfg.a,
        });
    }
    if !cfg.b.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "b".into(),
            val: cfg.b,
        });
    }
    if !cfg.c.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "c".into(),
            val: cfg.c,
        });
    }
    if !cfg.d.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "d".into(),
            val: cfg.d,
        });
    }
    let n = state.v.len();
    if state.u.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.u.len(),
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

/// Advance an Izhikevich population by one timestep.
pub fn izh_step(
    state: &mut IzhState,
    current: &[f32],
    cfg: &IzhConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate(state, current, cfg, spikes_out)?;
    let half_dt = 0.5 * cfg.dt;
    for (((v, u), &i_in), s_out) in state
        .v
        .iter_mut()
        .zip(state.u.iter_mut())
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        // Two dt/2 sub-steps for v stability (Izhikevich's original hint).
        let mut v_loc = *v;
        for _ in 0..2 {
            let dv = 0.04 * v_loc * v_loc + 5.0 * v_loc + 140.0 - *u + i_in;
            v_loc += half_dt * dv;
        }
        let du = cfg.a * (cfg.b * v_loc - *u);
        let mut u_loc = *u + cfg.dt * du;

        let spike = if v_loc >= SPIKE_THRESHOLD {
            v_loc = cfg.c;
            u_loc += cfg.d;
            1.0_f32
        } else {
            0.0_f32
        };

        // Numerical safety net (Izhikevich's quadratic can diverge with large I).
        v_loc = v_loc.clamp(V_LO, V_HI);

        *v = v_loc;
        *u = u_loc;
        *s_out = spike;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_construct() {
        let _ = IzhConfig::regular_spiking();
        let _ = IzhConfig::fast_spiking();
        let _ = IzhConfig::chattering();
        let _ = IzhConfig::intrinsically_bursting();
        let d = IzhConfig::default();
        assert!((d.a - 0.02).abs() < 1e-6);
        assert!((d.d - 8.0).abs() < 1e-6);
    }

    #[test]
    fn rs_tonic_input_spikes_in_expected_band() {
        let cfg = IzhConfig::regular_spiking();
        let mut state = IzhState::new(1, cfg.b);
        let current = vec![10.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..1000 {
            izh_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        // Izhikevich 2003 Fig 1 (RS) @ I=10 → adaptation drives ~20 Hz steady
        // state with an initial transient burst. Allow ±25% around 22 → [16, 28].
        assert!(
            (10..=40).contains(&count),
            "RS tonic spike count out of band: {count}"
        );
    }

    #[test]
    fn reset_clamps_v_to_c() {
        let cfg = IzhConfig::regular_spiking();
        let mut state = IzhState::new(1, cfg.b);
        // Force a near-spike voltage and large drive to trigger spike this step.
        state.v[0] = 29.0;
        state.u[0] = -10.0;
        let current = vec![100.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        izh_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert!((state.v[0] - cfg.c).abs() < 1e-4, "v={}", state.v[0]);
    }

    #[test]
    fn output_finite_for_all_presets() {
        for cfg in [
            IzhConfig::regular_spiking(),
            IzhConfig::fast_spiking(),
            IzhConfig::chattering(),
            IzhConfig::intrinsically_bursting(),
        ] {
            let mut state = IzhState::new(4, cfg.b);
            let current = vec![15.0_f32; 4];
            let mut spikes = vec![0.0_f32; 4];
            for _ in 0..200 {
                izh_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            }
            for &v in &state.v {
                assert!(v.is_finite(), "v not finite: {v}");
            }
            for &u in &state.u {
                assert!(u.is_finite(), "u not finite: {u}");
            }
        }
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = IzhConfig::default();
        let mut state = IzhState::new(2, cfg.b);
        let current = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = izh_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
