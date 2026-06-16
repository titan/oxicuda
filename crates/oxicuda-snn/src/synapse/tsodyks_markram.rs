//! Tsodyks–Markram short-term plasticity (STP): dynamic synapses with
//! facilitation and depression (Tsodyks & Markram 1997; Markram, Wang &
//! Tsodyks 1998; Mongillo, Barak & Tsodyks 2008).
//!
//! Each synapse carries two state variables: the utilisation `u` (an effective
//! release probability) and the fraction of available resources `x` (the "R"
//! variable in some papers). Between spikes both variables relax back toward
//! their baselines, and on each presynaptic spike `u` is incremented
//! (facilitation) before resources proportional to `u·x` are released and
//! removed from the pool (depression).
//!
//! ## Per-timestep update (`dt`-stepped ODE form)
//!
//! ```text
//! Continuous relaxation every step:
//!   x ← x + dt·(1 − x)/τ_rec
//!   u ← u + dt·(U − u)/τ_facil
//!
//! On a presynaptic spike this step:
//!   u        ← u + U·(1 − u)        (facilitation, applied first)
//!   released  = u · x               (synaptic efficacy)
//!   x        ← x·(1 − u)            (resource depletion)
//!   psc       = A · released        (post-synaptic current contribution)
//! otherwise:
//!   psc       = 0
//! ```
//!
//! `u` and `x` are clamped to `[0, 1]`. The baseline `U` together with the two
//! time constants selects the regime: `τ_rec ≫ τ_facil` gives a *depressing*
//! synapse (successive spikes shrink), while `τ_facil ≫ τ_rec` with small `U`
//! gives a *facilitating* synapse (early successive spikes grow before
//! saturating). The [`TmConfig::depressing`] and [`TmConfig::facilitating`]
//! presets provide the canonical parameter sets.

use crate::error::{SnnError, SnnResult};

/// Tsodyks–Markram configuration; time constants and `dt` must be positive.
#[derive(Debug, Clone, Copy)]
pub struct TmConfig {
    /// Baseline utilisation `U ∈ [0, 1]` (release probability at rest).
    pub u_baseline: f32,
    /// Resource recovery time constant `τ_rec` (depression); must be `> 0`.
    pub tau_rec: f32,
    /// Facilitation time constant `τ_facil`; must be `> 0`.
    pub tau_facil: f32,
    /// Absolute synaptic efficacy `A` (weight) scaling the released amount.
    pub a_weight: f32,
    /// Integration time step `dt`; must be `> 0`.
    pub dt: f32,
}

impl Default for TmConfig {
    /// Depressing-synapse default (`U = 0.45`, `τ_rec = 750`, `τ_facil = 50`).
    fn default() -> Self {
        Self::depressing()
    }
}

impl TmConfig {
    /// Facilitating-synapse preset (`U = 0.15`, `τ_rec = 100`, `τ_facil = 750`).
    #[must_use]
    pub fn facilitating() -> Self {
        Self {
            u_baseline: 0.15,
            tau_rec: 100.0,
            tau_facil: 750.0,
            a_weight: 1.0,
            dt: 1.0,
        }
    }

    /// Depressing-synapse preset (`U = 0.45`, `τ_rec = 750`, `τ_facil = 50`).
    #[must_use]
    pub fn depressing() -> Self {
        Self {
            u_baseline: 0.45,
            tau_rec: 750.0,
            tau_facil: 50.0,
            a_weight: 1.0,
            dt: 1.0,
        }
    }
}

/// Mutable per-synapse Tsodyks–Markram state.
#[derive(Debug, Clone)]
pub struct TmState {
    /// Utilisation `u_i` per synapse (initialised to `U`).
    pub u: Vec<f32>,
    /// Available-resource fraction `x_i` per synapse (initialised to `1`).
    pub x: Vec<f32>,
}

impl TmState {
    /// Allocate state for `n` synapses with `u = U` and `x = 1`.
    #[must_use]
    pub fn new(n: usize, cfg: &TmConfig) -> Self {
        Self {
            u: vec![cfg.u_baseline; n],
            x: vec![1.0_f32; n],
        }
    }
}

/// Validate `cfg` and slice lengths used by [`tm_step`].
fn validate_tm(
    state: &TmState,
    pre_spikes: &[f32],
    cfg: &TmConfig,
    psc_out: &[f32],
) -> SnnResult<()> {
    if cfg.tau_rec <= 0.0 || !cfg.tau_rec.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_rec });
    }
    if cfg.tau_facil <= 0.0 || !cfg.tau_facil.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_facil });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.u_baseline.is_finite() || cfg.u_baseline < 0.0 || cfg.u_baseline > 1.0 {
        return Err(SnnError::OutOfRange {
            name: "u_baseline".into(),
            val: cfg.u_baseline,
        });
    }
    if !cfg.a_weight.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "a_weight".into(),
            val: cfg.a_weight,
        });
    }
    let n = state.u.len();
    if state.x.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.x.len(),
        });
    }
    if pre_spikes.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: pre_spikes.len(),
        });
    }
    if psc_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: psc_out.len(),
        });
    }
    Ok(())
}

/// Advance the Tsodyks–Markram synapses by one timestep.
///
/// `pre_spikes` carries `1.0` where a presynaptic spike arrived this step (any
/// non-zero value is treated as a spike). `psc_out` receives the post-synaptic
/// current contribution `A·u·x` on spike steps and `0.0` otherwise.
///
/// # Errors
/// Returns `SnnError` if `tau_rec`/`tau_facil`/`dt` are non-positive or
/// non-finite, `u_baseline`/`a_weight` are out of range, or any slice length
/// does not match `state.u`.
pub fn tm_step(
    state: &mut TmState,
    pre_spikes: &[f32],
    cfg: &TmConfig,
    psc_out: &mut [f32],
) -> SnnResult<()> {
    validate_tm(state, pre_spikes, cfg, psc_out)?;
    let rec_rate = cfg.dt / cfg.tau_rec;
    let facil_rate = cfg.dt / cfg.tau_facil;
    for (((u, x), &spike), psc) in state
        .u
        .iter_mut()
        .zip(state.x.iter_mut())
        .zip(pre_spikes.iter())
        .zip(psc_out.iter_mut())
    {
        // Continuous relaxation toward baselines.
        *x += rec_rate * (1.0 - *x);
        *u += facil_rate * (cfg.u_baseline - *u);
        *x = x.clamp(0.0, 1.0);
        *u = u.clamp(0.0, 1.0);

        if spike != 0.0 {
            // Facilitation first, then release and deplete.
            *u += cfg.u_baseline * (1.0 - *u);
            *u = u.clamp(0.0, 1.0);
            let released = *u * *x;
            *x -= released;
            *x = x.clamp(0.0, 1.0);
            *psc = cfg.a_weight * released;
        } else {
            *psc = 0.0;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn state_init_u_baseline_x_one() {
        let cfg = TmConfig::depressing();
        let s = TmState::new(5, &cfg);
        assert_eq!(s.u.len(), 5);
        assert_eq!(s.x.len(), 5);
        assert!(s.u.iter().all(|&u| (u - cfg.u_baseline).abs() < TOL));
        assert!(s.x.iter().all(|&x| (x - 1.0).abs() < TOL));
    }

    #[test]
    fn depressing_successive_spikes_decrease() {
        let cfg = TmConfig::depressing();
        let mut state = TmState::new(1, &cfg);
        let spike = vec![1.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        let mut amplitudes = Vec::new();
        // Fire several spikes back-to-back (no recovery gap).
        for _ in 0..5 {
            tm_step(&mut state, &spike, &cfg, &mut psc).expect("step");
            amplitudes.push(psc[0]);
        }
        for w in amplitudes.windows(2) {
            assert!(
                w[1] < w[0] + TOL,
                "depressing PSC should not grow: {} -> {}",
                w[0],
                w[1]
            );
        }
        assert!(
            amplitudes[4] < amplitudes[0],
            "5th PSC {} must be below 1st {}",
            amplitudes[4],
            amplitudes[0]
        );
    }

    #[test]
    fn facilitating_successive_spikes_increase_then_saturate() {
        let cfg = TmConfig::facilitating();
        let mut state = TmState::new(1, &cfg);
        let spike = vec![1.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        let mut amplitudes = Vec::new();
        for _ in 0..4 {
            tm_step(&mut state, &spike, &cfg, &mut psc).expect("step");
            amplitudes.push(psc[0]);
        }
        // Early spikes facilitate: the second exceeds the first.
        assert!(
            amplitudes[1] > amplitudes[0],
            "facilitating: 2nd {} should exceed 1st {}",
            amplitudes[1],
            amplitudes[0]
        );
    }

    #[test]
    fn single_spike_released_about_u_times_one() {
        // First spike from rest: x≈1 after one relaxation step, u≈U after
        // relaxation then facilitated. Verify PSC ≈ A·u·x with the realised u, x.
        let cfg = TmConfig::depressing();
        let mut state = TmState::new(1, &cfg);
        let spike = vec![1.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        tm_step(&mut state, &spike, &cfg, &mut psc).expect("step");
        // After the step, state.x is post-depletion; reconstruct released.
        // released = psc / A.
        let released = psc[0] / cfg.a_weight;
        // On the first spike released ≈ u_eff (since x≈1), with u_eff ≳ U.
        assert!(
            released > cfg.u_baseline - 0.05 && released < 1.0,
            "released={} expected near U={}",
            released,
            cfg.u_baseline
        );
    }

    #[test]
    fn resources_recover_toward_one_without_spikes() {
        let cfg = TmConfig::depressing();
        let mut state = TmState::new(1, &cfg);
        state.x[0] = 0.2; // depleted
        let no_spike = vec![0.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        let x0 = state.x[0];
        for _ in 0..10 {
            tm_step(&mut state, &no_spike, &cfg, &mut psc).expect("step");
        }
        assert!(
            state.x[0] > x0,
            "x should recover: {} -> {}",
            x0,
            state.x[0]
        );
        assert!(state.x[0] <= 1.0 + TOL);
    }

    #[test]
    fn u_decays_toward_baseline_without_spikes() {
        let cfg = TmConfig::facilitating();
        let mut state = TmState::new(1, &cfg);
        state.u[0] = 0.9; // elevated above baseline
        let no_spike = vec![0.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        for _ in 0..50 {
            tm_step(&mut state, &no_spike, &cfg, &mut psc).expect("step");
        }
        assert!(
            state.u[0] < 0.9,
            "u should decay toward U={}, got {}",
            cfg.u_baseline,
            state.u[0]
        );
        assert!(state.u[0] >= cfg.u_baseline - TOL);
    }

    #[test]
    fn psc_zero_when_no_spike() {
        let cfg = TmConfig::depressing();
        let mut state = TmState::new(3, &cfg);
        let no_spike = vec![0.0_f32; 3];
        let mut psc = vec![9.0_f32; 3];
        tm_step(&mut state, &no_spike, &cfg, &mut psc).expect("step");
        assert!(psc.iter().all(|&p| p == 0.0), "psc must be 0 with no spike");
    }

    #[test]
    fn u_and_x_stay_clamped() {
        let cfg = TmConfig {
            u_baseline: 0.9,
            tau_rec: 5.0,
            tau_facil: 5.0,
            a_weight: 1.0,
            dt: 1.0,
        };
        let mut state = TmState::new(1, &cfg);
        let spike = vec![1.0_f32; 1];
        let mut psc = vec![0.0_f32; 1];
        for _ in 0..100 {
            tm_step(&mut state, &spike, &cfg, &mut psc).expect("step");
            assert!(state.u[0] >= 0.0 && state.u[0] <= 1.0, "u={}", state.u[0]);
            assert!(state.x[0] >= 0.0 && state.x[0] <= 1.0, "x={}", state.x[0]);
        }
    }

    #[test]
    fn rejects_bad_tau_rec() {
        let cfg = TmConfig {
            tau_rec: 0.0,
            ..TmConfig::depressing()
        };
        let mut state = TmState::new(2, &TmConfig::depressing());
        let spike = vec![0.0_f32; 2];
        let mut psc = vec![0.0_f32; 2];
        let err = tm_step(&mut state, &spike, &cfg, &mut psc);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_bad_tau_facil() {
        let cfg = TmConfig {
            tau_facil: -3.0,
            ..TmConfig::depressing()
        };
        let mut state = TmState::new(2, &TmConfig::depressing());
        let spike = vec![0.0_f32; 2];
        let mut psc = vec![0.0_f32; 2];
        let err = tm_step(&mut state, &spike, &cfg, &mut psc);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_bad_dt() {
        let cfg = TmConfig {
            dt: 0.0,
            ..TmConfig::depressing()
        };
        let mut state = TmState::new(2, &TmConfig::depressing());
        let spike = vec![0.0_f32; 2];
        let mut psc = vec![0.0_f32; 2];
        let err = tm_step(&mut state, &spike, &cfg, &mut psc);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn rejects_u_baseline_above_one() {
        let cfg = TmConfig {
            u_baseline: 1.5,
            ..TmConfig::depressing()
        };
        let mut state = TmState::new(2, &TmConfig::depressing());
        let spike = vec![0.0_f32; 2];
        let mut psc = vec![0.0_f32; 2];
        let err = tm_step(&mut state, &spike, &cfg, &mut psc);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = TmConfig::depressing();
        let mut state = TmState::new(2, &cfg);
        let spike = vec![0.0_f32; 3];
        let mut psc = vec![0.0_f32; 2];
        let err = tm_step(&mut state, &spike, &cfg, &mut psc);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
