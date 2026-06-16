//! Leaky Integrate-and-Fire neuron with an absolute refractory period.
//!
//! Reference: Gerstner, Kistler, Naud & Paninski — *Neuronal Dynamics: From
//! Single Neurons to Networks and Models of Cognition* (Cambridge University
//! Press, 2014), §1.3.1 ("The leaky integrate-and-fire model") and §5.1
//! (refractoriness). The classic Lapicque (1907) LIF augmented with the
//! absolute refractory period `t_ref`: after a spike the membrane is clamped to
//! `v_rest` and held there for `t_ref` time units, during which the neuron
//! integrates no input and cannot fire again — the most widely-used cortical
//! point-neuron abstraction.
//!
//! Discrete-time dynamics with step `dt`:
//!
//! ```text
//! β        = exp(-dt / τ_m)                         # membrane decay factor
//! if refractory_remaining > 0:                      # absolute refractory phase
//!     v_{t+1} = v_rest
//!     s_{t+1} = 0
//!     refractory_remaining ← refractory_remaining − dt
//! else:
//!     v_{t+1} = β · v_t + R · I_t                   # leaky integration
//!     s_{t+1} = (v_{t+1} ≥ v_th)
//!     if s_{t+1}:                                   # spike → enter refractory
//!         v_{t+1} ← v_rest
//!         refractory_remaining ← t_ref
//! ```
//!
//! Setting `t_ref = 0` recovers the plain LIF integrator with a hard reset
//! exactly (the refractory branch is never taken). The membrane resistance `R`
//! scales the input current; the dimensionless default `R = 1` reproduces the
//! current-as-voltage convention used by [`crate::neuron::lif`].

use crate::error::{SnnError, SnnResult};

/// Refractory-LIF configuration; `tau_m`, `dt` strictly positive, `t_ref ≥ 0`.
#[derive(Debug, Clone, Copy)]
pub struct RefractoryLifConfig {
    /// Membrane time constant `τ_m` in the same time units as `dt`.
    pub tau_m: f64,
    /// Spike threshold `v_th`.
    pub v_th: f64,
    /// Resting / reset potential `v_rest`.
    pub v_rest: f64,
    /// Membrane resistance `R` scaling the input current (`I → R·I`).
    pub r_m: f64,
    /// Absolute refractory period `t_ref` (`≥ 0`, same units as `dt`).
    pub t_ref: f64,
    /// Integration step `dt`.
    pub dt: f64,
}

impl Default for RefractoryLifConfig {
    /// Cortical pyramidal defaults (Gerstner et al. 2014, §1.3): `τ_m = 20 ms`,
    /// `v_th = 1`, `v_rest = 0`, `R = 1`, `t_ref = 2 ms`, `dt = 1 ms`.
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 2.0,
            dt: 1.0,
        }
    }
}

/// Mutable per-neuron refractory-LIF state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefractoryLifState {
    /// Membrane potential `v`.
    pub v: f64,
    /// Remaining absolute-refractory time; `> 0` ⇒ neuron is clamped.
    pub refractory_remaining: f64,
}

impl RefractoryLifState {
    /// Allocate a fresh state with `v = v_init` and no refractoriness.
    #[must_use]
    pub fn new(v_init: f64) -> Self {
        Self {
            v: v_init,
            refractory_remaining: 0.0,
        }
    }

    /// `true` while the neuron is inside its absolute refractory window.
    #[must_use]
    pub fn is_refractory(&self) -> bool {
        self.refractory_remaining > 0.0
    }
}

impl Default for RefractoryLifState {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Validate a [`RefractoryLifConfig`]; emits the same error variants as the LIF
/// code so downstream error-handling pipelines remain uniform.
fn validate(cfg: &RefractoryLifConfig) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau {
            tau: cfg.tau_m as f32,
        });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt as f32 });
    }
    if !cfg.v_th.is_finite() {
        return Err(SnnError::BadThreshold {
            v_th: cfg.v_th as f32,
        });
    }
    if !cfg.v_rest.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "v_rest".into(),
            val: cfg.v_rest as f32,
        });
    }
    if !cfg.r_m.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "r_m".into(),
            val: cfg.r_m as f32,
        });
    }
    if cfg.t_ref < 0.0 || !cfg.t_ref.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "t_ref".into(),
            val: cfg.t_ref as f32,
        });
    }
    Ok(())
}

/// Membrane decay factor `β = exp(-dt / τ_m)`.
#[must_use]
pub fn beta(cfg: &RefractoryLifConfig) -> f64 {
    (-cfg.dt / cfg.tau_m).exp()
}

/// Advance a refractory-LIF neuron by one timestep.
///
/// Returns the boolean spike indicator. While `state.refractory_remaining > 0`
/// the membrane is clamped to `v_rest`, the input is ignored, no spike is
/// emitted, and the remaining refractory time is decremented by `dt`.
pub fn refractory_lif_step(
    state: &mut RefractoryLifState,
    input: f64,
    cfg: &RefractoryLifConfig,
) -> SnnResult<bool> {
    validate(cfg)?;
    if !input.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "input".into(),
            val: input as f32,
        });
    }

    if state.refractory_remaining > 0.0 {
        // Absolute refractory phase: clamp, ignore input, count down.
        state.v = cfg.v_rest;
        state.refractory_remaining = (state.refractory_remaining - cfg.dt).max(0.0);
        return Ok(false);
    }

    let b = beta(cfg);
    let v_new = b * state.v + cfg.r_m * input;
    let spike = v_new >= cfg.v_th;
    if spike {
        state.v = cfg.v_rest;
        state.refractory_remaining = cfg.t_ref;
    } else {
        state.v = v_new;
    }
    Ok(spike)
}

/// Advance a slice of refractory-LIF neurons element-wise by one timestep.
///
/// `states`, `inputs`, and `spikes_out` must have identical length; each
/// `spikes_out[i]` receives `1.0` on a spike and `0.0` otherwise.
pub fn refractory_lif_step_batch(
    states: &mut [RefractoryLifState],
    inputs: &[f64],
    spikes_out: &mut [f32],
    cfg: &RefractoryLifConfig,
) -> SnnResult<()> {
    validate(cfg)?;
    if states.len() != inputs.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: inputs.len(),
        });
    }
    if states.len() != spikes_out.len() {
        return Err(SnnError::IncompatibleLength {
            a: states.len(),
            b: spikes_out.len(),
        });
    }
    let b = beta(cfg);
    for ((state, &input), s_out) in states
        .iter_mut()
        .zip(inputs.iter())
        .zip(spikes_out.iter_mut())
    {
        if !input.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "input".into(),
                val: input as f32,
            });
        }
        if state.refractory_remaining > 0.0 {
            state.v = cfg.v_rest;
            state.refractory_remaining = (state.refractory_remaining - cfg.dt).max(0.0);
            *s_out = 0.0;
            continue;
        }
        let v_new = b * state.v + cfg.r_m * input;
        if v_new >= cfg.v_th {
            state.v = cfg.v_rest;
            state.refractory_remaining = cfg.t_ref;
            *s_out = 1.0;
        } else {
            state.v = v_new;
            *s_out = 0.0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> RefractoryLifConfig {
        RefractoryLifConfig::default()
    }

    #[test]
    fn rejects_zero_tau() {
        let cfg = RefractoryLifConfig {
            tau_m: 0.0,
            ..cfg_default()
        };
        let mut s = RefractoryLifState::default();
        assert!(matches!(
            refractory_lif_step(&mut s, 0.0, &cfg),
            Err(SnnError::BadTau { .. })
        ));
    }

    #[test]
    fn rejects_zero_dt() {
        let cfg = RefractoryLifConfig {
            dt: 0.0,
            ..cfg_default()
        };
        let mut s = RefractoryLifState::default();
        assert!(matches!(
            refractory_lif_step(&mut s, 0.0, &cfg),
            Err(SnnError::BadDt { .. })
        ));
    }

    #[test]
    fn rejects_negative_t_ref() {
        let cfg = RefractoryLifConfig {
            t_ref: -1.0,
            ..cfg_default()
        };
        let mut s = RefractoryLifState::default();
        assert!(matches!(
            refractory_lif_step(&mut s, 0.0, &cfg),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_nan_input() {
        let cfg = cfg_default();
        let mut s = RefractoryLifState::default();
        assert!(matches!(
            refractory_lif_step(&mut s, f64::NAN, &cfg),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn beta_matches_formula() {
        let cfg = RefractoryLifConfig {
            tau_m: 10.0,
            dt: 1.0,
            ..cfg_default()
        };
        assert!((beta(&cfg) - (-0.1_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn subthreshold_no_spike() {
        let cfg = RefractoryLifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 2.0,
            dt: 1.0,
        };
        let mut s = RefractoryLifState::default();
        let spike = refractory_lif_step(&mut s, 0.1, &cfg).expect("step");
        assert!(!spike);
        assert!(s.v > 0.0 && s.v < cfg.v_th);
        assert!(!s.is_refractory());
    }

    #[test]
    fn spike_enters_refractory_and_clamps() {
        let cfg = RefractoryLifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 3.0,
            dt: 1.0,
        };
        let mut s = RefractoryLifState::default();
        // Strong drive overshoots threshold → spike, reset to v_rest, refractory.
        let spike = refractory_lif_step(&mut s, 1.5, &cfg).expect("step");
        assert!(spike);
        assert!((s.v - cfg.v_rest).abs() < 1e-12);
        assert!((s.refractory_remaining - cfg.t_ref).abs() < 1e-12);
    }

    #[test]
    fn no_fire_during_refractory_even_with_huge_drive() {
        let cfg = RefractoryLifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 3.0,
            dt: 1.0,
        };
        let mut s = RefractoryLifState::default();
        let _ = refractory_lif_step(&mut s, 1.5, &cfg).expect("step"); // spike, t_ref=3
        // Next 3 steps: huge drive must NOT fire, membrane stays clamped.
        for _ in 0..3 {
            let spike = refractory_lif_step(&mut s, 1000.0, &cfg).expect("step");
            assert!(!spike);
            assert!((s.v - cfg.v_rest).abs() < 1e-12);
        }
        // After refractory expires, the same drive fires again.
        let spike = refractory_lif_step(&mut s, 1000.0, &cfg).expect("step");
        assert!(spike);
    }

    #[test]
    fn refractory_counts_down_by_dt() {
        let cfg = RefractoryLifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 2.5,
            dt: 1.0,
        };
        let mut s = RefractoryLifState::default();
        let _ = refractory_lif_step(&mut s, 1.5, &cfg).expect("step");
        assert!((s.refractory_remaining - 2.5).abs() < 1e-12);
        let _ = refractory_lif_step(&mut s, 0.0, &cfg).expect("step");
        assert!((s.refractory_remaining - 1.5).abs() < 1e-12);
        let _ = refractory_lif_step(&mut s, 0.0, &cfg).expect("step");
        assert!((s.refractory_remaining - 0.5).abs() < 1e-12);
        let _ = refractory_lif_step(&mut s, 0.0, &cfg).expect("step");
        // 0.5 - 1.0 clamped to 0.0 ⇒ no longer refractory.
        assert!((s.refractory_remaining - 0.0).abs() < 1e-12);
        assert!(!s.is_refractory());
    }

    #[test]
    fn t_ref_zero_matches_plain_lif_hard_reset() {
        // With t_ref = 0, behaviour must equal the plain LIF hard-reset integrator.
        let cfg = RefractoryLifConfig {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 0.0,
            dt: 1.0,
        };
        let b = (-cfg.dt / cfg.tau_m).exp();
        let mut s = RefractoryLifState::default();
        let mut v_ref = 0.0_f64;
        let input = 0.15_f64;
        for _ in 0..300 {
            let spike = refractory_lif_step(&mut s, input, &cfg).expect("step");
            let v_new = b * v_ref + input;
            let lif_spike = v_new >= cfg.v_th;
            v_ref = if lif_spike { cfg.v_rest } else { v_new };
            assert_eq!(spike, lif_spike);
            assert!((s.v - v_ref).abs() < 1e-10);
        }
    }

    #[test]
    fn refractory_reduces_max_firing_rate() {
        // Under saturating drive, the firing rate is bounded by 1/(t_ref) spikes
        // per unit time, i.e. fewer spikes than with no refractory period.
        let strong = 1000.0_f64;
        let cfg_ref = RefractoryLifConfig {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 5.0,
            dt: 1.0,
        };
        let cfg_noref = RefractoryLifConfig {
            t_ref: 0.0,
            ..cfg_ref
        };
        let mut s_ref = RefractoryLifState::default();
        let mut s_nor = RefractoryLifState::default();
        let mut c_ref = 0usize;
        let mut c_nor = 0usize;
        for _ in 0..100 {
            if refractory_lif_step(&mut s_ref, strong, &cfg_ref).expect("step") {
                c_ref += 1;
            }
            if refractory_lif_step(&mut s_nor, strong, &cfg_noref).expect("step") {
                c_nor += 1;
            }
        }
        assert!(
            c_ref < c_nor,
            "refractory must reduce rate: {c_ref} < {c_nor}"
        );
        // With t_ref = 5 and dt = 1, at most ~floor(100/6) spikes (1 fire + 5
        // refractory steps per cycle) ≈ 16–17.
        assert!(c_ref <= 17, "rate exceeds 1/t_ref bound: {c_ref}");
    }

    #[test]
    fn r_m_scales_input() {
        // Doubling R doubles the effective drive ⇒ reaches threshold sooner.
        let base = RefractoryLifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            r_m: 1.0,
            t_ref: 0.0,
            dt: 1.0,
        };
        let cfg_hi = RefractoryLifConfig { r_m: 2.0, ..base };
        let mut s = RefractoryLifState::default();
        // R = 2, input 0.6 ⇒ v_new = 1.2 ≥ 1.0 ⇒ spike on the first step.
        let spike = refractory_lif_step(&mut s, 0.6, &cfg_hi).expect("step");
        assert!(spike);
        let mut s2 = RefractoryLifState::default();
        // R = 1, input 0.6 ⇒ v_new = 0.6 < 1.0 ⇒ no spike.
        let spike2 = refractory_lif_step(&mut s2, 0.6, &base).expect("step");
        assert!(!spike2);
    }

    #[test]
    fn batch_matches_scalar() {
        let cfg = cfg_default();
        let inputs = [0.3_f64, 1.5, -0.2, 0.9, 1.2];
        let n = inputs.len();
        let mut batch: Vec<RefractoryLifState> = (0..n)
            .map(|i| RefractoryLifState::new(0.05 * i as f64))
            .collect();
        let mut batch_out = vec![0.0_f32; n];
        // Run several steps so refractory windows get exercised across neurons.
        for _ in 0..10 {
            refractory_lif_step_batch(&mut batch, &inputs, &mut batch_out, &cfg).expect("batch");
        }
        let mut scalar_out = vec![0.0_f32; n];
        let mut scalar: Vec<RefractoryLifState> = (0..n)
            .map(|i| RefractoryLifState::new(0.05 * i as f64))
            .collect();
        for _ in 0..10 {
            for i in 0..n {
                let spike = refractory_lif_step(&mut scalar[i], inputs[i], &cfg).expect("step");
                scalar_out[i] = if spike { 1.0 } else { 0.0 };
            }
        }
        for i in 0..n {
            assert!((scalar[i].v - batch[i].v).abs() < 1e-12, "v[{i}]");
            assert!(
                (scalar[i].refractory_remaining - batch[i].refractory_remaining).abs() < 1e-12,
                "ref[{i}]"
            );
            assert!((scalar_out[i] - batch_out[i]).abs() < 1e-12, "s[{i}]");
        }
    }

    #[test]
    fn batch_length_mismatch_rejected() {
        let cfg = cfg_default();
        let mut states = vec![RefractoryLifState::default(); 3];
        let inputs = vec![0.0_f64; 2];
        let mut out = vec![0.0_f32; 3];
        assert!(matches!(
            refractory_lif_step_batch(&mut states, &inputs, &mut out, &cfg),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }
}
