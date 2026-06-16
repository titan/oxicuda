//! Alpha-function synapse (Rall 1967; Dayan & Abbott 2001 eq. 5.34).
//!
//! The alpha synapse produces a smooth, finite-rise post-synaptic current whose
//! impulse response is the *alpha function*
//!
//! ```text
//! K(t) = (t / τ) · exp(1 − t / τ),   t ≥ 0,
//! ```
//!
//! which peaks at `t = τ` with unit amplitude (`K(τ) = 1`). Unlike the single-
//! exponential CUBA/COBA synapse, the alpha function has a *finite* rising
//! phase, giving a biologically realistic delayed onset. It is the impulse
//! response of a critically-damped second-order linear filter and is therefore
//! realised exactly in discrete time by a cascade of two identical first-order
//! exponential stages driven by the spike train:
//!
//! ```text
//! decay = exp(-dt / τ)
//! x_{t+1} = decay · x_t + (spike ? w · e / τ : 0)      # source / first stage
//! g_{t+1} = decay · g_t + dt · x_t                     # integrated second stage
//! I_syn   = g_{t+1}                                    # CUBA readout
//! ```
//!
//! The `e / τ` injection constant normalises the kernel so that a single spike
//! of weight `w` yields a continuous-time peak current of exactly `w` at the
//! lag `t = τ`. As `dt → 0` the discrete cascade converges to the continuous
//! alpha kernel; for the finite default `dt = 1 ms ≪ τ` the peak is reproduced
//! to within a percent. Time constants and `dt` are in millisecond units,
//! matching the [`crate::synapse::conductance`] conventions.

use crate::error::{SnnError, SnnResult};

/// Configuration for an alpha-function synapse.
///
/// The single time constant `tau_syn` sets both the rise and decay of the
/// alpha kernel (its peak occurs at `t = tau_syn`); `dt` is the integration
/// step in the same (millisecond) units.
#[derive(Debug, Clone, Copy)]
pub struct AlphaConfig {
    /// Synaptic time constant `τ_syn` in ms; must be strictly positive.
    pub tau_syn: f64,
    /// Integration step `dt` in ms; must be strictly positive.
    pub dt: f64,
}

impl Default for AlphaConfig {
    /// Fast cortical synapse defaults: `τ_syn = 5 ms`, `dt = 1 ms`.
    fn default() -> Self {
        Self {
            tau_syn: 5.0,
            dt: 1.0,
        }
    }
}

/// Mutable per-synapse alpha state (the two-stage cascade variables).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaState {
    /// First-stage (source) variable `x`.
    pub x: f64,
    /// Second-stage (output current) variable `g = I_syn`.
    pub g: f64,
}

impl AlphaState {
    /// Allocate a fresh, quiescent state (`x = 0`, `g = 0`).
    #[must_use]
    pub fn new() -> Self {
        Self { x: 0.0, g: 0.0 }
    }
}

impl Default for AlphaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate an [`AlphaConfig`].
fn validate(cfg: &AlphaConfig) -> SnnResult<()> {
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

/// Per-stage decay factor `exp(-dt / τ_syn)` of the alpha cascade.
#[must_use]
pub fn alpha_decay(cfg: &AlphaConfig) -> f64 {
    (-cfg.dt / cfg.tau_syn).exp()
}

/// Advance a single alpha synapse by one timestep and return the post-update
/// synaptic current `I_syn = g`.
///
/// On a presynaptic spike, the first-stage source variable receives an impulse
/// of `w · e / τ_syn`; the second stage integrates the first. The output `g`
/// then traces the normalised alpha kernel `w · (t/τ) · exp(1 − t/τ)`.
pub fn alpha_step(
    state: &mut AlphaState,
    spike_in: bool,
    weight: f64,
    cfg: &AlphaConfig,
) -> SnnResult<f64> {
    validate(cfg)?;
    if !weight.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "weight".into(),
            val: weight as f32,
        });
    }
    let decay = alpha_decay(cfg);
    // Second stage integrates the *previous* first-stage value (so a single
    // spike produces a finite-rise response rather than an instantaneous jump).
    let g_new = decay * state.g + cfg.dt * state.x;
    let injection = if spike_in {
        weight * std::f64::consts::E / cfg.tau_syn
    } else {
        0.0
    };
    let x_new = decay * state.x + injection;
    state.x = x_new;
    state.g = g_new;
    Ok(state.g)
}

/// Advance a slice of alpha synapses element-wise by one timestep.
///
/// `states`, `spikes_in`, `weights`, and `i_out` must have identical length;
/// `i_out` receives `I_syn = g` per synapse after the update.
pub fn alpha_step_batch(
    states: &mut [AlphaState],
    spikes_in: &[bool],
    weights: &[f64],
    i_out: &mut [f64],
    cfg: &AlphaConfig,
) -> SnnResult<()> {
    validate(cfg)?;
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
    let decay = alpha_decay(cfg);
    let norm = std::f64::consts::E / cfg.tau_syn;
    for (((state, &spike), &w), out) in states
        .iter_mut()
        .zip(spikes_in.iter())
        .zip(weights.iter())
        .zip(i_out.iter_mut())
    {
        if !w.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "weight".into(),
                val: w as f32,
            });
        }
        let g_new = decay * state.g + cfg.dt * state.x;
        let injection = if spike { w * norm } else { 0.0 };
        state.x = decay * state.x + injection;
        state.g = g_new;
        *out = state.g;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn cfg() -> AlphaConfig {
        AlphaConfig::default()
    }

    #[test]
    fn rejects_zero_tau() {
        let cfg = AlphaConfig {
            tau_syn: 0.0,
            dt: 1.0,
        };
        let mut s = AlphaState::new();
        assert!(matches!(
            alpha_step(&mut s, false, 0.0, &cfg),
            Err(SnnError::BadTau { .. })
        ));
    }

    #[test]
    fn rejects_zero_dt() {
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.0,
        };
        let mut s = AlphaState::new();
        assert!(matches!(
            alpha_step(&mut s, false, 0.0, &cfg),
            Err(SnnError::BadDt { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_weight() {
        let cfg = cfg();
        let mut s = AlphaState::new();
        assert!(matches!(
            alpha_step(&mut s, true, f64::NAN, &cfg),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn decay_matches_formula() {
        let cfg = AlphaConfig {
            tau_syn: 10.0,
            dt: 1.0,
        };
        assert!((alpha_decay(&cfg) - (-0.1_f64).exp()).abs() < EPS);
    }

    #[test]
    fn quiescent_stays_zero() {
        let cfg = cfg();
        let mut s = AlphaState::new();
        for _ in 0..50 {
            let i = alpha_step(&mut s, false, 0.0, &cfg).expect("step");
            assert!(i.abs() < EPS);
        }
    }

    #[test]
    fn single_spike_has_finite_rise() {
        // Immediately after a single spike the output current is still zero
        // (the second stage has not yet integrated) — a finite rising phase.
        let cfg = cfg();
        let mut s = AlphaState::new();
        let i0 = alpha_step(&mut s, true, 1.0, &cfg).expect("step");
        assert!(
            i0.abs() < EPS,
            "alpha current must rise from zero, got {i0}"
        );
        // The very next step the current becomes strictly positive.
        let i1 = alpha_step(&mut s, false, 1.0, &cfg).expect("step");
        assert!(i1 > 0.0, "current should rise after the spike, got {i1}");
    }

    #[test]
    fn single_spike_response_rises_then_decays() {
        // The alpha response is unimodal: rise to a single peak, then decay.
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.1,
        };
        let mut s = AlphaState::new();
        let _ = alpha_step(&mut s, true, 1.0, &cfg).expect("step");
        let mut prev = s.g;
        let mut peak = s.g;
        let mut rising_done = false;
        for _ in 0..2000 {
            let i = alpha_step(&mut s, false, 1.0, &cfg).expect("step");
            if i > peak {
                peak = i;
            }
            if i < prev {
                rising_done = true;
            }
            // Once it has started decaying it must never rise again.
            if rising_done {
                assert!(i <= prev + EPS, "non-monotone decay: {prev} -> {i}");
            }
            prev = i;
        }
        assert!(rising_done, "response never started to decay");
        assert!(peak > 0.0);
    }

    #[test]
    fn peak_amplitude_approximates_weight_for_small_dt() {
        // With a fine time step, a unit-weight spike peaks at ~1.0 at t ≈ τ.
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.01,
        };
        let mut s = AlphaState::new();
        let _ = alpha_step(&mut s, true, 1.0, &cfg).expect("step");
        let mut peak = s.g;
        for _ in 0..5000 {
            let i = alpha_step(&mut s, false, 1.0, &cfg).expect("step");
            if i > peak {
                peak = i;
            }
        }
        assert!(
            (peak - 1.0).abs() < 0.02,
            "alpha peak should be ~1.0 for weight 1.0, got {peak}"
        );
    }

    #[test]
    fn peak_time_is_near_tau() {
        // The continuous-time alpha kernel peaks at t = τ; check the discrete
        // peak occurs near that lag.
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.01,
        };
        let mut s = AlphaState::new();
        let _ = alpha_step(&mut s, true, 1.0, &cfg).expect("step");
        let mut peak = s.g;
        let mut peak_step = 0usize;
        for step in 1..5000 {
            let i = alpha_step(&mut s, false, 1.0, &cfg).expect("step");
            if i > peak {
                peak = i;
                peak_step = step;
            }
        }
        let peak_time = peak_step as f64 * cfg.dt;
        assert!(
            (peak_time - cfg.tau_syn).abs() < 0.3,
            "peak time {peak_time} should be near τ={}",
            cfg.tau_syn
        );
    }

    #[test]
    fn amplitude_scales_linearly_with_weight() {
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.1,
        };
        let run_peak = |w: f64| -> f64 {
            let mut s = AlphaState::new();
            let _ = alpha_step(&mut s, true, w, &cfg).expect("step");
            let mut peak = s.g;
            for _ in 0..2000 {
                let i = alpha_step(&mut s, false, w, &cfg).expect("step");
                if i > peak {
                    peak = i;
                }
            }
            peak
        };
        let p1 = run_peak(1.0);
        let p3 = run_peak(3.0);
        assert!((p3 - 3.0 * p1).abs() < 1e-6, "p1={p1} p3={p3}");
    }

    #[test]
    fn current_returns_to_zero_after_long_time() {
        let cfg = cfg();
        let mut s = AlphaState::new();
        let _ = alpha_step(&mut s, true, 1.0, &cfg).expect("step");
        for _ in 0..2000 {
            let _ = alpha_step(&mut s, false, 1.0, &cfg).expect("step");
        }
        assert!(s.g.abs() < 1e-6, "g should decay to ~0, got {}", s.g);
        assert!(s.x.abs() < 1e-6, "x should decay to ~0, got {}", s.x);
    }

    #[test]
    fn two_spikes_superpose_linearly() {
        // Linear filter ⇒ response to two spikes = sum of shifted single
        // responses. Compare a two-spike run to the summed single-spike runs.
        let cfg = AlphaConfig {
            tau_syn: 5.0,
            dt: 0.5,
        };
        let steps = 60usize;
        let spike_a = 0usize;
        let spike_b = 10usize;
        // Combined run.
        let mut sc = AlphaState::new();
        let mut combined = Vec::with_capacity(steps);
        for t in 0..steps {
            let fire = t == spike_a || t == spike_b;
            combined.push(alpha_step(&mut sc, fire, 1.0, &cfg).expect("step"));
        }
        // Single-spike runs.
        let single = |t_fire: usize| -> Vec<f64> {
            let mut s = AlphaState::new();
            let mut out = Vec::with_capacity(steps);
            for t in 0..steps {
                out.push(alpha_step(&mut s, t == t_fire, 1.0, &cfg).expect("step"));
            }
            out
        };
        let ra = single(spike_a);
        let rb = single(spike_b);
        for t in 0..steps {
            assert!(
                (combined[t] - (ra[t] + rb[t])).abs() < 1e-9,
                "superposition failed at t={t}"
            );
        }
    }

    #[test]
    fn batch_matches_scalar() {
        let cfg = cfg();
        let n = 5usize;
        let weights = [0.5_f64, 1.2, -0.3, 0.0, 2.0];
        let spikes = [true, false, true, true, false];
        let mut batch: Vec<AlphaState> = vec![AlphaState::new(); n];
        let mut batch_out = vec![0.0_f64; n];
        // Run several steps to exercise the cascade memory.
        for _ in 0..20 {
            alpha_step_batch(&mut batch, &spikes, &weights, &mut batch_out, &cfg).expect("batch");
        }
        let mut scalar: Vec<AlphaState> = vec![AlphaState::new(); n];
        let mut scalar_out = vec![0.0_f64; n];
        for _ in 0..20 {
            for i in 0..n {
                scalar_out[i] = alpha_step(&mut scalar[i], spikes[i], weights[i], &cfg).expect("s");
            }
        }
        for i in 0..n {
            assert!((scalar[i].g - batch[i].g).abs() < EPS, "g[{i}]");
            assert!((scalar[i].x - batch[i].x).abs() < EPS, "x[{i}]");
            assert!((scalar_out[i] - batch_out[i]).abs() < EPS, "out[{i}]");
        }
    }

    #[test]
    fn batch_length_mismatch_rejected() {
        let cfg = cfg();
        let mut states = vec![AlphaState::new(); 3];
        let spikes = vec![false; 3];
        let weights = vec![1.0_f64; 2];
        let mut out = vec![0.0_f64; 3];
        assert!(matches!(
            alpha_step_batch(&mut states, &spikes, &weights, &mut out, &cfg),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }
}
