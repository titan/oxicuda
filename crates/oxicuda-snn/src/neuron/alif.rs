//! Adaptive-threshold Leaky Integrate-and-Fire (ALIF) neuron.
//!
//! Reference: Bellec, Salaj, Subramoney, Legenstein & Maass — *"Long short-term
//! memory and learning-to-learn in networks of spiking neurons"* (NeurIPS 2018,
//! [arXiv:1803.09574](https://arxiv.org/abs/1803.09574)).
//!
//! ALIF augments a vanilla LIF neuron with a slowly decaying adaptive variable
//! `b` that raises the effective firing threshold after each spike, endowing the
//! neuron with multi-timescale memory and significantly extending the temporal
//! credit-assignment window available to spiking RNNs.
//!
//! Discrete-time dynamics with step `dt`:
//!
//! ```text
//! β_m       = exp(-dt / τ_m)              # membrane decay factor
//! ρ_b       = exp(-dt / τ_b)              # adaptation decay factor
//! b_{t+1}   = ρ_b · b_t + (s_t ? β : 0)   # adaptation variable update
//! v_{t+1}   = β_m · v_t + I_t             # membrane update (identical to LIF)
//! v_th_eff  = v_th + b_{t+1}              # effective threshold
//! s_{t+1}   = (v_{t+1} ≥ v_th_eff)        # spike emission
//! ```
//!
//! After spiking, `v` is either reset to `v_rest` ([`crate::neuron::lif::ResetMode::Hard`]) or has
//! the *effective* threshold subtracted ([`crate::neuron::lif::ResetMode::Soft`]). Setting
//! `β = 0` recovers vanilla LIF behaviour exactly; taking `τ_b → ∞` freezes the
//! adaptation variable, so on short windows the model also collapses to LIF.

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::ResetMode;

/// ALIF configuration; all time constants and `dt` must be strictly positive.
#[derive(Debug, Clone, Copy)]
pub struct AlifConfig {
    /// Membrane time constant `τ_m` in the same time units as `dt`.
    pub tau_m: f64,
    /// Adaptation time constant `τ_b`; typically `τ_b ≫ τ_m`.
    pub tau_b: f64,
    /// Baseline spike threshold `v_th`.
    pub v_th: f64,
    /// Resting / equilibrium potential `v_rest`.
    pub v_rest: f64,
    /// Adaptation increment `β` added to `b` on every spike (must be `≥ 0`).
    pub beta: f64,
    /// Integration step `dt`.
    pub dt: f64,
    /// Reset mode applied after a spike.
    pub reset: ResetMode,
}

impl Default for AlifConfig {
    /// Bellec et al. 2018 hyperparameters from the sequential-MNIST experiments
    /// (Table S1): `τ_m = 20 ms`, `τ_b = 200 ms`, `β = 0.07`, `dt = 1 ms`.
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            tau_b: 200.0,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.07,
            dt: 1.0,
            reset: ResetMode::Hard,
        }
    }
}

/// Mutable per-neuron ALIF state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlifState {
    /// Membrane potential `v`.
    pub v: f64,
    /// Adaptive threshold variable `b`.
    pub b: f64,
}

impl AlifState {
    /// Allocate a fresh state with `v = v_init` and `b = 0`.
    #[must_use]
    pub fn new(v_init: f64) -> Self {
        Self { v: v_init, b: 0.0 }
    }
}

impl Default for AlifState {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Validate an [`AlifConfig`]; emits the same error variants as the LIF code so
/// downstream error-handling pipelines remain uniform across neuron types.
fn validate(cfg: &AlifConfig) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau {
            tau: cfg.tau_m as f32,
        });
    }
    if cfg.tau_b <= 0.0 || !cfg.tau_b.is_finite() {
        return Err(SnnError::BadTau {
            tau: cfg.tau_b as f32,
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
    if cfg.beta < 0.0 || !cfg.beta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "beta".into(),
            val: cfg.beta as f32,
        });
    }
    if cfg.v_th <= cfg.v_rest {
        return Err(SnnError::OutOfRange {
            name: "v_th_minus_v_rest".into(),
            val: (cfg.v_th - cfg.v_rest) as f32,
        });
    }
    Ok(())
}

/// Membrane decay factor `β_m = exp(-dt / τ_m)`.
#[must_use]
pub fn beta_m(cfg: &AlifConfig) -> f64 {
    (-cfg.dt / cfg.tau_m).exp()
}

/// Adaptation decay factor `ρ_b = exp(-dt / τ_b)`.
#[must_use]
pub fn rho_b(cfg: &AlifConfig) -> f64 {
    (-cfg.dt / cfg.tau_b).exp()
}

/// Advance an ALIF neuron by one timestep.
///
/// Returns the boolean spike indicator `s_{t+1}`. The state's `v` and `b` are
/// updated in place using the Bellec et al. 2018 first-order discretisation.
pub fn alif_step(state: &mut AlifState, input: f64, cfg: &AlifConfig) -> SnnResult<bool> {
    validate(cfg)?;
    if !input.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "input".into(),
            val: input as f32,
        });
    }
    let bm = beta_m(cfg);
    let rb = rho_b(cfg);

    // Adaptation variable: leak then update; the increment is applied on the
    // *previous* timestep's spike, hence we read from the current `state.b`.
    let b_new = rb * state.b;

    // Membrane update (vanilla LIF integrator).
    let v_new = bm * state.v + input;

    // Effective threshold uses the *updated* `b` (Bellec et al. eq. (3)).
    let v_th_eff = cfg.v_th + b_new;
    let spike = v_new >= v_th_eff;

    // Apply reset.
    let v_after = if spike {
        match cfg.reset {
            ResetMode::Hard => cfg.v_rest,
            ResetMode::Soft => v_new - v_th_eff,
        }
    } else {
        v_new
    };

    // The adaptation variable receives an instantaneous `β` increment on the
    // step at which the spike fires (Bellec et al. NeurIPS 2018 eq. (2)).
    let b_after = if spike { b_new + cfg.beta } else { b_new };

    state.v = v_after;
    state.b = b_after;
    Ok(spike)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> AlifConfig {
        AlifConfig::default()
    }

    #[test]
    fn rejects_zero_tau_m() {
        let cfg = AlifConfig {
            tau_m: 0.0,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let err = alif_step(&mut s, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_negative_tau_b() {
        let cfg = AlifConfig {
            tau_b: -1.0,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let err = alif_step(&mut s, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_zero_dt() {
        let cfg = AlifConfig {
            dt: 0.0,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let err = alif_step(&mut s, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn rejects_negative_beta() {
        let cfg = AlifConfig {
            beta: -0.5,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let err = alif_step(&mut s, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_v_th_not_above_v_rest() {
        let cfg = AlifConfig {
            v_th: 0.0,
            v_rest: 0.0,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let err = alif_step(&mut s, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_nan_input() {
        let cfg = cfg_default();
        let mut s = AlifState::default();
        let err = alif_step(&mut s, f64::NAN, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn subthreshold_no_spike() {
        let cfg = AlifConfig {
            tau_m: 1e9, // ~no leak
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.07,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let spike = alif_step(&mut s, 0.1, &cfg).expect("step");
        assert!(!spike);
        assert!(s.v > 0.0 && s.v < cfg.v_th);
        assert!(s.b.abs() < 1e-12);
    }

    #[test]
    fn spike_at_threshold() {
        let cfg = AlifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.07,
            reset: ResetMode::Hard,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        let spike = alif_step(&mut s, 1.0, &cfg).expect("step");
        assert!(spike);
        assert!((s.v - cfg.v_rest).abs() < 1e-12);
        assert!((s.b - cfg.beta).abs() < 1e-12);
    }

    #[test]
    fn adaptation_raises_effective_threshold() {
        // After one spike, the next spike requires a larger drive than the first.
        let cfg = AlifConfig {
            tau_m: 1e9,
            tau_b: 1e9, // adaptation does not leak away
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.5,
            reset: ResetMode::Hard,
            ..cfg_default()
        };
        let mut s = AlifState::default();
        // First spike at exactly v_th.
        let s1 = alif_step(&mut s, 1.0, &cfg).expect("step");
        assert!(s1);
        // b is now 0.5, so effective threshold ≈ 1.5; input of 1.0 alone must
        // not spike now even though it spiked from v_rest above.
        let s2 = alif_step(&mut s, 1.0, &cfg).expect("step");
        assert!(!s2, "adaptation must suppress second spike, got s2=true");
        // Input of 1.5 should reach the new threshold.
        let s3 = alif_step(&mut s, 1.5, &cfg).expect("step");
        assert!(s3, "input matching v_th_eff must spike");
    }

    #[test]
    fn beta_zero_reduces_to_lif_short_window() {
        // β = 0 ⇒ b stays at 0 ⇒ behaviour identical to vanilla LIF.
        let cfg = AlifConfig {
            tau_m: 20.0,
            tau_b: 200.0,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let bm = (-cfg.dt / cfg.tau_m).exp();
        let mut state = AlifState::default();
        let mut v_ref = 0.0_f64;
        let input = 0.15_f64;
        let mut count_alif = 0usize;
        let mut count_lif = 0usize;
        for _ in 0..200 {
            let s = alif_step(&mut state, input, &cfg).expect("step");
            if s {
                count_alif += 1;
            }
            // Reference LIF integrator.
            let v_new = bm * v_ref + input;
            let lif_spike = v_new >= cfg.v_th;
            v_ref = if lif_spike { cfg.v_rest } else { v_new };
            if lif_spike {
                count_lif += 1;
            }
            assert!((state.v - v_ref).abs() < 1e-10);
            assert!(state.b.abs() < 1e-12);
        }
        assert_eq!(count_alif, count_lif);
    }

    #[test]
    fn tau_b_infinite_matches_lif_on_short_window() {
        // With τ_b → ∞, ρ_b ≈ 1, b is frozen; with no prior spikes, b stays at
        // 0 and ALIF matches LIF before the first spike on short windows.
        let cfg = AlifConfig {
            tau_m: 20.0,
            tau_b: 1e18,
            v_th: 5.0,
            v_rest: 0.0,
            beta: 0.07,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut state = AlifState::default();
        // Subthreshold drive ⇒ no spikes ⇒ b never increments.
        // Drive 0.2 saturates at 0.2/(1-exp(-1/20)) ≈ 4.10 < v_th=5.0.
        for _ in 0..30 {
            let s = alif_step(&mut state, 0.2, &cfg).expect("step");
            assert!(!s);
            assert!(state.b.abs() < 1e-12);
        }
        // Match the LIF v exactly.
        let bm = (-cfg.dt / cfg.tau_m).exp();
        let mut v_ref = 0.0_f64;
        for _ in 0..30 {
            v_ref = bm * v_ref + 0.2;
        }
        assert!((state.v - v_ref).abs() < 1e-9);
    }

    #[test]
    fn hard_vs_soft_reset_differ_on_overshoot() {
        let base = AlifConfig {
            tau_m: 1e9,
            tau_b: 1e9,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.0,
            dt: 1.0,
            ..cfg_default()
        };
        let cfg_hard = AlifConfig {
            reset: ResetMode::Hard,
            ..base
        };
        let cfg_soft = AlifConfig {
            reset: ResetMode::Soft,
            ..base
        };
        let mut h = AlifState::default();
        let mut so = AlifState::default();
        // Strong drive ⇒ overshoot. v_new = 1.3.
        let _ = alif_step(&mut h, 1.3, &cfg_hard).expect("step");
        let _ = alif_step(&mut so, 1.3, &cfg_soft).expect("step");
        assert!((h.v - cfg_hard.v_rest).abs() < 1e-12);
        // Soft: v_new(1.3) - v_th_eff(1.0+0=1.0) = 0.3
        assert!((so.v - 0.3).abs() < 1e-12);
    }

    #[test]
    fn long_window_fewer_spikes_than_lif() {
        // Bellec et al. 2018: adaptation reduces the average firing rate under
        // constant drive relative to a vanilla LIF with identical parameters.
        let cfg = AlifConfig {
            tau_m: 20.0,
            tau_b: 200.0,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.3,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let bm = (-cfg.dt / cfg.tau_m).exp();
        let input = 0.4_f64;
        // ALIF spike count.
        let mut alif_state = AlifState::default();
        let mut alif_count = 0usize;
        // LIF reference spike count.
        let mut v_ref = 0.0_f64;
        let mut lif_count = 0usize;
        for _ in 0..100 {
            if alif_step(&mut alif_state, input, &cfg).expect("step") {
                alif_count += 1;
            }
            let v_new = bm * v_ref + input;
            if v_new >= cfg.v_th {
                lif_count += 1;
                v_ref = cfg.v_rest;
            } else {
                v_ref = v_new;
            }
        }
        assert!(
            alif_count < lif_count,
            "ALIF should fire less than LIF: alif={alif_count}, lif={lif_count}"
        );
    }

    #[test]
    fn deterministic_under_same_input() {
        let cfg = cfg_default();
        let inputs = [0.3_f64, 0.5, 0.7, 0.0, -0.1, 1.2, 0.9, 0.4];
        let mut a = AlifState::default();
        let mut b = AlifState::default();
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        for &i in inputs.iter() {
            sa.push(alif_step(&mut a, i, &cfg).expect("step"));
            sb.push(alif_step(&mut b, i, &cfg).expect("step"));
        }
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }

    #[test]
    fn b_decays_after_100_idle_steps() {
        let cfg = AlifConfig {
            tau_m: 1e9,
            tau_b: 50.0,
            v_th: 1.0,
            v_rest: 0.0,
            beta: 0.5,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let mut s = AlifState::default();
        // Trigger one spike to set b.
        let _ = alif_step(&mut s, 1.0, &cfg).expect("step");
        assert!((s.b - cfg.beta).abs() < 1e-12);
        // 100 idle steps with zero input (subthreshold for any positive v_th).
        for _ in 0..100 {
            let _ = alif_step(&mut s, 0.0, &cfg).expect("step");
        }
        // After 100 idle steps, b should have decayed by ρ_b^100.
        let expected = cfg.beta * (-100.0_f64 * cfg.dt / cfg.tau_b).exp();
        assert!(
            (s.b - expected).abs() < 1e-9,
            "b={} expected={}",
            s.b,
            expected
        );
    }

    #[test]
    fn beta_m_and_rho_b_match_expected() {
        let cfg = AlifConfig {
            tau_m: 10.0,
            tau_b: 100.0,
            dt: 1.0,
            ..cfg_default()
        };
        assert!((beta_m(&cfg) - (-0.1_f64).exp()).abs() < 1e-12);
        assert!((rho_b(&cfg) - (-0.01_f64).exp()).abs() < 1e-12);
    }
}
