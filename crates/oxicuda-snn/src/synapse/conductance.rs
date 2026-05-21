//! Exponential current-based (CUBA) and conductance-based (COBA) synapses.
//!
//! Reference: Dayan & Abbott, *Theoretical Neuroscience: Computational and
//! Mathematical Modeling of Neural Systems* (MIT Press, 2001), Chapter 5
//! ("Model Neurons II: Conductances and Morphology"), eqs. (5.32)–(5.35) for
//! the exponentially decaying open-channel fraction and eqs. (5.27)–(5.30) for
//! the conductance-based driving force `g_syn · (E_rev − V)`.
//!
//! Both models share the same first-order exponential decay for the gating
//! variable `g` (synaptic conductance for COBA, "synaptic current amplitude"
//! for CUBA). Given a discrete time step `dt`, an arriving presynaptic spike,
//! and a quantal weight `w`, the exact analytic integration of
//! `τ_syn · dg/dt = −g` between two consecutive spike events yields
//!
//! ```text
//! g_{t+1} = exp(-dt / τ_syn) · g_t + (s_t ? w : 0)
//! ```
//!
//! The two variants differ only in how the synaptic current `I_syn` is read
//! out from `g`:
//!
//! ```text
//! CUBA: I_syn = g                          (Dayan & Abbott eq. 5.32)
//! COBA: I_syn = g · (E_rev − V)            (Dayan & Abbott eq. 5.27)
//! ```
//!
//! The CUBA form is convenient when the postsynaptic neuron's input is a pure
//! current (e.g. LIF without reversal potentials), while the COBA form is the
//! biologically faithful description in which the driving force vanishes at
//! the synapse's reversal potential `E_rev`. Excitatory glutamatergic synapses
//! are typically modelled with `E_rev ≈ 0 mV`, and inhibitory GABA_A synapses
//! with `E_rev ≈ −80 mV` (Dayan & Abbott Table 5.3).
//!
//! All time constants and `dt` are expressed in millisecond units to match the
//! defaults used by the [`crate::neuron`] modules; voltages and reversal
//! potentials are in millivolts.

use crate::error::{SnnError, SnnResult};

/// Configuration for an exponential current-based (CUBA) synapse.
///
/// The single time constant `tau_syn` governs how quickly the synaptic current
/// accumulator decays toward zero between presynaptic spikes; `dt` is the
/// integration step in the same units (ms by default).
#[derive(Debug, Clone, Copy)]
pub struct CubaConfig {
    /// Synaptic time constant `τ_syn` in ms; must be strictly positive.
    pub tau_syn: f64,
    /// Integration step `dt` in ms; must be strictly positive.
    pub dt: f64,
}

impl Default for CubaConfig {
    /// Defaults from Dayan & Abbott Table 5.3 (fast glutamatergic regime):
    /// `τ_syn = 5 ms`, `dt = 1 ms`.
    fn default() -> Self {
        Self {
            tau_syn: 5.0,
            dt: 1.0,
        }
    }
}

/// Mutable per-synapse CUBA state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubaState {
    /// Synaptic current accumulator `g` (same units as `weight`).
    pub g: f64,
}

impl CubaState {
    /// Allocate a fresh state with `g = 0`.
    #[must_use]
    pub fn new() -> Self {
        Self { g: 0.0 }
    }

    /// Allocate a state initialised to a specific `g` value, useful for tests
    /// that need to verify pure-decay behaviour from a non-zero starting point.
    #[must_use]
    pub fn with_g(g: f64) -> Self {
        Self { g }
    }
}

impl Default for CubaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a [`CubaConfig`]; reuses the same error variants as the neuron
/// modules so downstream error-handling pipelines remain uniform.
fn validate_cuba(cfg: &CubaConfig) -> SnnResult<()> {
    if cfg.tau_syn <= 0.0 || !cfg.tau_syn.is_finite() {
        return Err(SnnError::BadTau {
            tau: cfg.tau_syn as f32,
        });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt as f32 });
    }
    Ok(())
}

/// Synaptic decay factor `α = exp(-dt / τ_syn)` for CUBA synapses.
#[must_use]
pub fn cuba_decay(cfg: &CubaConfig) -> f64 {
    (-cfg.dt / cfg.tau_syn).exp()
}

/// Advance a single CUBA synapse by one timestep and return the post-update
/// synaptic current `I_syn = g`.
///
/// Dynamics: `g_{t+1} = α · g_t + (spike_in ? weight : 0)` with
/// `α = exp(-dt / τ_syn)`.
pub fn cuba_step(
    state: &mut CubaState,
    spike_in: bool,
    weight: f64,
    cfg: &CubaConfig,
) -> SnnResult<f64> {
    validate_cuba(cfg)?;
    if !weight.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "weight".into(),
            val: weight as f32,
        });
    }
    let alpha = cuba_decay(cfg);
    let increment = if spike_in { weight } else { 0.0 };
    state.g = alpha * state.g + increment;
    Ok(state.g)
}

/// Advance a slice of CUBA synapses element-wise by one timestep.
///
/// All four slices must have identical length. The output buffer `i_out`
/// receives `I_syn` for each synapse after the update.
pub fn cuba_step_batch(
    states: &mut [CubaState],
    spikes_in: &[bool],
    weights: &[f64],
    i_out: &mut [f64],
    cfg: &CubaConfig,
) -> SnnResult<()> {
    validate_cuba(cfg)?;
    if states.len() != spikes_in.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: spikes_in.len(),
        });
    }
    if states.len() != weights.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: weights.len(),
        });
    }
    if states.len() != i_out.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: i_out.len(),
        });
    }
    let alpha = cuba_decay(cfg);
    for i in 0..states.len() {
        let w = weights[i];
        if !w.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "weight".into(),
                val: w as f32,
            });
        }
        let increment = if spikes_in[i] { w } else { 0.0 };
        states[i].g = alpha * states[i].g + increment;
        i_out[i] = states[i].g;
    }
    Ok(())
}

/// Configuration for an exponential conductance-based (COBA) synapse.
///
/// The synaptic current is `I_syn = g · (E_rev − V)` where `g` is updated with
/// the same exponential dynamics as in [`CubaConfig`]. The reversal potential
/// `E_rev` selects excitatory vs inhibitory character.
#[derive(Debug, Clone, Copy)]
pub struct CobaConfig {
    /// Synaptic time constant `τ_syn` in ms; must be strictly positive.
    pub tau_syn: f64,
    /// Synaptic reversal potential `E_rev` in mV.
    pub e_rev: f64,
    /// Integration step `dt` in ms; must be strictly positive.
    pub dt: f64,
}

impl CobaConfig {
    /// Excitatory glutamatergic preset (AMPA-like): `τ_syn = 5 ms`,
    /// `E_rev = 0 mV` (Dayan & Abbott Table 5.3).
    #[must_use]
    pub fn excitatory() -> Self {
        Self {
            tau_syn: 5.0,
            e_rev: 0.0,
            dt: 1.0,
        }
    }

    /// Inhibitory GABA_A preset: `τ_syn = 10 ms`, `E_rev = −80 mV`
    /// (Dayan & Abbott Table 5.3).
    #[must_use]
    pub fn inhibitory() -> Self {
        Self {
            tau_syn: 10.0,
            e_rev: -80.0,
            dt: 1.0,
        }
    }
}

impl Default for CobaConfig {
    /// Defaults to the excitatory preset, mirroring the convention used by
    /// most spiking-network simulators (Brian2, Nest) when no reversal
    /// potential is specified.
    fn default() -> Self {
        Self::excitatory()
    }
}

/// Mutable per-synapse COBA state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CobaState {
    /// Synaptic conductance accumulator `g` (siemens-like units).
    pub g: f64,
}

impl CobaState {
    /// Allocate a fresh state with `g = 0`.
    #[must_use]
    pub fn new() -> Self {
        Self { g: 0.0 }
    }

    /// Allocate a state initialised to a specific `g` value, useful for tests
    /// that need to verify pure-decay behaviour from a non-zero starting point.
    #[must_use]
    pub fn with_g(g: f64) -> Self {
        Self { g }
    }
}

impl Default for CobaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a [`CobaConfig`]; reuses the same error variants as CUBA.
fn validate_coba(cfg: &CobaConfig) -> SnnResult<()> {
    if cfg.tau_syn <= 0.0 || !cfg.tau_syn.is_finite() {
        return Err(SnnError::BadTau {
            tau: cfg.tau_syn as f32,
        });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt as f32 });
    }
    if !cfg.e_rev.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "e_rev".into(),
            val: cfg.e_rev as f32,
        });
    }
    Ok(())
}

/// Synaptic decay factor `α = exp(-dt / τ_syn)` for COBA synapses.
#[must_use]
pub fn coba_decay(cfg: &CobaConfig) -> f64 {
    (-cfg.dt / cfg.tau_syn).exp()
}

/// Advance a single COBA synapse by one timestep and return the post-update
/// synaptic current `I_syn = g · (E_rev − V)`.
///
/// Dynamics: `g_{t+1} = α · g_t + (spike_in ? weight : 0)` with
/// `α = exp(-dt / τ_syn)`, identical to CUBA; only the readout differs.
pub fn coba_step(
    state: &mut CobaState,
    spike_in: bool,
    weight: f64,
    v: f64,
    cfg: &CobaConfig,
) -> SnnResult<f64> {
    validate_coba(cfg)?;
    if !weight.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "weight".into(),
            val: weight as f32,
        });
    }
    if !v.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "v".into(),
            val: v as f32,
        });
    }
    let alpha = coba_decay(cfg);
    let increment = if spike_in { weight } else { 0.0 };
    state.g = alpha * state.g + increment;
    let i_syn = state.g * (cfg.e_rev - v);
    Ok(i_syn)
}

/// Advance a slice of COBA synapses element-wise by one timestep.
///
/// All five slices (`states`, `spikes_in`, `weights`, `v`, `i_out`) must have
/// identical length; `i_out` receives `I_syn = g · (E_rev − V)` per synapse.
pub fn coba_step_batch(
    states: &mut [CobaState],
    spikes_in: &[bool],
    weights: &[f64],
    v: &[f64],
    i_out: &mut [f64],
    cfg: &CobaConfig,
) -> SnnResult<()> {
    validate_coba(cfg)?;
    if states.len() != spikes_in.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: spikes_in.len(),
        });
    }
    if states.len() != weights.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: weights.len(),
        });
    }
    if states.len() != v.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: v.len(),
        });
    }
    if states.len() != i_out.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: i_out.len(),
        });
    }
    let alpha = coba_decay(cfg);
    for i in 0..states.len() {
        let w = weights[i];
        let vi = v[i];
        if !w.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "weight".into(),
                val: w as f32,
            });
        }
        if !vi.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "v".into(),
                val: vi as f32,
            });
        }
        let increment = if spikes_in[i] { w } else { 0.0 };
        states[i].g = alpha * states[i].g + increment;
        i_out[i] = states[i].g * (cfg.e_rev - vi);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_EXACT: f64 = 1e-9;
    const EPS_ACCUM: f64 = 1e-6;

    fn cuba_cfg() -> CubaConfig {
        CubaConfig::default()
    }

    #[test]
    fn cuba_decays_by_exact_factor_no_spike() {
        // With no spike, g_{t+1} = exp(-dt/τ) · g_t exactly.
        let cfg = cuba_cfg();
        let alpha = (-cfg.dt / cfg.tau_syn).exp();
        let mut s = CubaState::with_g(1.0);
        let i_syn = cuba_step(&mut s, false, 0.0, &cfg).expect("step");
        assert!((s.g - alpha).abs() < EPS_EXACT, "g={} alpha={}", s.g, alpha);
        assert!((i_syn - alpha).abs() < EPS_EXACT);
        // Confirm at a second step too.
        let _ = cuba_step(&mut s, false, 0.0, &cfg).expect("step");
        assert!((s.g - alpha * alpha).abs() < EPS_EXACT);
    }

    #[test]
    fn cuba_single_spike_sets_g_to_weight_from_zero() {
        // Starting at g=0, a single spike of weight w yields g = α·0 + w = w.
        let cfg = cuba_cfg();
        let mut s = CubaState::new();
        let i_syn = cuba_step(&mut s, true, 0.75, &cfg).expect("step");
        assert!((s.g - 0.75).abs() < EPS_EXACT);
        assert!((i_syn - 0.75).abs() < EPS_EXACT);
    }

    #[test]
    fn cuba_linear_superposition_of_two_spikes() {
        // After two consecutive spikes with weights w1, w2 separated by one dt,
        // g should equal α·w1 + w2.
        let cfg = cuba_cfg();
        let alpha = cuba_decay(&cfg);
        let mut s = CubaState::new();
        let _ = cuba_step(&mut s, true, 1.0, &cfg).expect("step");
        let _ = cuba_step(&mut s, true, 0.5, &cfg).expect("step");
        let expected = alpha * 1.0 + 0.5;
        assert!(
            (s.g - expected).abs() < EPS_EXACT,
            "g={} expected={}",
            s.g,
            expected
        );
    }

    #[test]
    fn cuba_converges_to_zero_with_no_spikes() {
        // After many idle steps, g should decay essentially to zero.
        let cfg = cuba_cfg();
        let mut s = CubaState::with_g(10.0);
        for _ in 0..1000 {
            let _ = cuba_step(&mut s, false, 0.0, &cfg).expect("step");
        }
        // exp(-1000/5) = exp(-200) ≈ 1.38e-87 — well below any sensible eps.
        assert!(s.g.abs() < EPS_ACCUM, "g={} should be near zero", s.g);
    }

    #[test]
    fn cuba_rejects_zero_tau() {
        let cfg = CubaConfig {
            tau_syn: 0.0,
            dt: 1.0,
        };
        let mut s = CubaState::new();
        let err = cuba_step(&mut s, false, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn cuba_rejects_negative_tau() {
        let cfg = CubaConfig {
            tau_syn: -1.0,
            dt: 1.0,
        };
        let mut s = CubaState::new();
        let err = cuba_step(&mut s, false, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn cuba_rejects_non_finite_tau() {
        let cfg = CubaConfig {
            tau_syn: f64::NAN,
            dt: 1.0,
        };
        let mut s = CubaState::new();
        let err = cuba_step(&mut s, false, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn cuba_rejects_zero_dt() {
        let cfg = CubaConfig {
            tau_syn: 5.0,
            dt: 0.0,
        };
        let mut s = CubaState::new();
        let err = cuba_step(&mut s, false, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn cuba_rejects_non_finite_weight() {
        let cfg = cuba_cfg();
        let mut s = CubaState::new();
        let err = cuba_step(&mut s, true, f64::NAN, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn cuba_batch_length_mismatch_rejected() {
        let cfg = cuba_cfg();
        let mut states = vec![CubaState::new(); 3];
        let spikes = vec![false; 3];
        let weights = vec![1.0; 2]; // mismatched length
        let mut i_out = vec![0.0; 3];
        let err = cuba_step_batch(&mut states, &spikes, &weights, &mut i_out, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn cuba_batch_i_out_length_mismatch_rejected() {
        let cfg = cuba_cfg();
        let mut states = vec![CubaState::new(); 3];
        let spikes = vec![false; 3];
        let weights = vec![1.0; 3];
        let mut i_out = vec![0.0; 4]; // wrong i_out length
        let err = cuba_step_batch(&mut states, &spikes, &weights, &mut i_out, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn cuba_batch_matches_per_element_step() {
        let cfg = cuba_cfg();
        let n = 5usize;
        let weights = [0.3_f64, 1.2, -0.5, 0.0, 2.1];
        let spikes = [true, false, true, true, false];
        // Batch path.
        let mut batch_states: Vec<CubaState> =
            (0..n).map(|i| CubaState::with_g(0.1 * i as f64)).collect();
        let mut batch_out = vec![0.0_f64; n];
        cuba_step_batch(&mut batch_states, &spikes, &weights, &mut batch_out, &cfg)
            .expect("batch step");
        // Per-element path.
        for i in 0..n {
            let mut s = CubaState::with_g(0.1 * i as f64);
            let i_syn = cuba_step(&mut s, spikes[i], weights[i], &cfg).expect("step");
            assert!(
                (s.g - batch_states[i].g).abs() < EPS_EXACT,
                "g[{i}] batch={} scalar={}",
                batch_states[i].g,
                s.g
            );
            assert!(
                (batch_out[i] - i_syn).abs() < EPS_EXACT,
                "i_out[{i}] batch={} scalar={}",
                batch_out[i],
                i_syn
            );
        }
    }

    #[test]
    fn cuba_batch_empty_is_valid() {
        // Zero-length batch must be a no-op success.
        let cfg = cuba_cfg();
        let mut states: Vec<CubaState> = Vec::new();
        let spikes: Vec<bool> = Vec::new();
        let weights: Vec<f64> = Vec::new();
        let mut i_out: Vec<f64> = Vec::new();
        cuba_step_batch(&mut states, &spikes, &weights, &mut i_out, &cfg)
            .expect("empty batch is valid");
    }

    fn coba_cfg_exc() -> CobaConfig {
        CobaConfig::excitatory()
    }

    fn coba_cfg_inh() -> CobaConfig {
        CobaConfig::inhibitory()
    }

    #[test]
    fn coba_excitatory_default_constructor() {
        let cfg = CobaConfig::excitatory();
        assert!((cfg.e_rev - 0.0).abs() < EPS_EXACT);
        assert!(cfg.tau_syn > 0.0);
        assert!(cfg.dt > 0.0);
    }

    #[test]
    fn coba_inhibitory_default_constructor() {
        let cfg = CobaConfig::inhibitory();
        assert!((cfg.e_rev - (-80.0)).abs() < EPS_EXACT);
        assert!(cfg.tau_syn > 0.0);
        assert!(cfg.dt > 0.0);
    }

    #[test]
    fn coba_default_matches_excitatory() {
        let d = CobaConfig::default();
        let e = CobaConfig::excitatory();
        assert!((d.e_rev - e.e_rev).abs() < EPS_EXACT);
        assert!((d.tau_syn - e.tau_syn).abs() < EPS_EXACT);
        assert!((d.dt - e.dt).abs() < EPS_EXACT);
    }

    #[test]
    fn coba_excitatory_current_is_positive_below_e_rev() {
        // E_rev = 0 mV, V = -60 mV ⇒ driving force = +60 ⇒ I_syn > 0.
        let cfg = coba_cfg_exc();
        let mut s = CobaState::new();
        let i_syn = coba_step(&mut s, true, 0.5, -60.0, &cfg).expect("step");
        assert!(
            i_syn > 0.0,
            "expected positive (depolarising) current, got {i_syn}"
        );
        // g should equal weight (started at 0); driving force = 0 - (-60) = 60.
        let expected = 0.5 * 60.0;
        assert!((i_syn - expected).abs() < EPS_EXACT);
    }

    #[test]
    fn coba_inhibitory_current_is_negative_above_e_rev() {
        // E_rev = -80 mV, V = -60 mV ⇒ driving force = -20 ⇒ I_syn < 0.
        let cfg = coba_cfg_inh();
        let mut s = CobaState::new();
        let i_syn = coba_step(&mut s, true, 0.5, -60.0, &cfg).expect("step");
        assert!(
            i_syn < 0.0,
            "expected negative (hyperpolarising) current, got {i_syn}"
        );
        // g = 0.5; driving force = -80 - (-60) = -20.
        let expected = 0.5 * (-20.0);
        assert!((i_syn - expected).abs() < EPS_EXACT);
    }

    #[test]
    fn coba_current_zero_at_reversal_potential() {
        // When V = E_rev exactly, I_syn must be zero regardless of g.
        let cfg = coba_cfg_exc();
        let mut s = CobaState::with_g(1.0);
        let i_syn = coba_step(&mut s, true, 0.3, cfg.e_rev, &cfg).expect("step");
        // After the update, g > 0 but driving force = 0.
        assert!(s.g > 0.0);
        assert!(i_syn.abs() < EPS_EXACT, "I_syn={i_syn} expected ~0");
        // Repeat with inhibitory cfg.
        let cfg_i = coba_cfg_inh();
        let mut s_i = CobaState::with_g(2.0);
        let i_syn_i = coba_step(&mut s_i, false, 0.0, cfg_i.e_rev, &cfg_i).expect("step");
        assert!(i_syn_i.abs() < EPS_EXACT);
    }

    #[test]
    fn coba_decays_g_identically_to_cuba() {
        // Same τ_syn and dt ⇒ g update factor must match between CUBA and COBA.
        let tau_syn = 7.5;
        let dt = 0.5;
        let cfg_c = CubaConfig { tau_syn, dt };
        let cfg_b = CobaConfig {
            tau_syn,
            e_rev: -10.0,
            dt,
        };
        let mut sc = CubaState::with_g(2.3);
        let mut sb = CobaState::with_g(2.3);
        // Run 50 idle steps and ensure g traces match exactly.
        for _ in 0..50 {
            let _ = cuba_step(&mut sc, false, 0.0, &cfg_c).expect("step");
            let _ = coba_step(&mut sb, false, 0.0, 0.0, &cfg_b).expect("step");
            assert!((sc.g - sb.g).abs() < EPS_EXACT);
        }
    }

    #[test]
    fn coba_rejects_zero_tau() {
        let cfg = CobaConfig {
            tau_syn: 0.0,
            e_rev: 0.0,
            dt: 1.0,
        };
        let mut s = CobaState::new();
        let err = coba_step(&mut s, false, 0.0, -65.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn coba_rejects_zero_dt() {
        let cfg = CobaConfig {
            tau_syn: 5.0,
            e_rev: 0.0,
            dt: 0.0,
        };
        let mut s = CobaState::new();
        let err = coba_step(&mut s, false, 0.0, -65.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn coba_rejects_non_finite_e_rev() {
        let cfg = CobaConfig {
            tau_syn: 5.0,
            e_rev: f64::NAN,
            dt: 1.0,
        };
        let mut s = CobaState::new();
        let err = coba_step(&mut s, false, 0.0, -65.0, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn coba_rejects_non_finite_v() {
        let cfg = coba_cfg_exc();
        let mut s = CobaState::new();
        let err = coba_step(&mut s, true, 0.5, f64::INFINITY, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn coba_batch_length_mismatch_rejected() {
        let cfg = coba_cfg_exc();
        let mut states = vec![CobaState::new(); 3];
        let spikes = vec![false; 3];
        let weights = vec![1.0; 3];
        let v = vec![-60.0_f64; 2]; // mismatched
        let mut i_out = vec![0.0; 3];
        let err = coba_step_batch(&mut states, &spikes, &weights, &v, &mut i_out, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn coba_batch_matches_per_element_step() {
        let cfg = coba_cfg_exc();
        let n = 6usize;
        let weights = [0.5_f64, 1.0, 0.25, 0.0, 2.0, 0.1];
        let spikes = [true, true, false, true, false, true];
        let v_vals = [-65.0_f64, -60.0, -55.0, -50.0, -70.0, -45.0];
        // Batch path.
        let mut batch_states: Vec<CobaState> =
            (0..n).map(|i| CobaState::with_g(0.05 * i as f64)).collect();
        let mut batch_out = vec![0.0_f64; n];
        coba_step_batch(
            &mut batch_states,
            &spikes,
            &weights,
            &v_vals,
            &mut batch_out,
            &cfg,
        )
        .expect("batch step");
        // Per-element path.
        for i in 0..n {
            let mut s = CobaState::with_g(0.05 * i as f64);
            let i_syn = coba_step(&mut s, spikes[i], weights[i], v_vals[i], &cfg).expect("step");
            assert!(
                (s.g - batch_states[i].g).abs() < EPS_EXACT,
                "g[{i}] batch={} scalar={}",
                batch_states[i].g,
                s.g
            );
            assert!(
                (batch_out[i] - i_syn).abs() < EPS_EXACT,
                "i_out[{i}] batch={} scalar={}",
                batch_out[i],
                i_syn
            );
        }
    }

    #[test]
    fn cuba_and_coba_decay_helpers_agree_with_formula() {
        let cfg_c = CubaConfig {
            tau_syn: 10.0,
            dt: 1.0,
        };
        assert!((cuba_decay(&cfg_c) - (-0.1_f64).exp()).abs() < EPS_EXACT);
        let cfg_b = CobaConfig {
            tau_syn: 4.0,
            e_rev: 0.0,
            dt: 0.5,
        };
        assert!((coba_decay(&cfg_b) - (-0.125_f64).exp()).abs() < EPS_EXACT);
    }

    #[test]
    fn cuba_long_run_repeated_spikes_converges_to_steady_state() {
        // Under spike-on-every-step input with constant weight w, the steady
        // state for g is w / (1 - α): g_∞ = α·g_∞ + w ⇒ g_∞ = w/(1-α).
        let cfg = cuba_cfg();
        let alpha = cuba_decay(&cfg);
        let w = 1.0_f64;
        let expected = w / (1.0 - alpha);
        let mut s = CubaState::new();
        for _ in 0..2000 {
            let _ = cuba_step(&mut s, true, w, &cfg).expect("step");
        }
        assert!(
            (s.g - expected).abs() < EPS_ACCUM,
            "g={} steady_state={}",
            s.g,
            expected
        );
    }
}
