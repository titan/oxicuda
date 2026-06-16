//! Spike Response Model (SRM₀) neuron.
//!
//! Reference: Gerstner & Kistler, *Spiking Neuron Models: Single Neurons,
//! Populations, Plasticity* (Cambridge University Press, 2002), Chapter 4
//! ("Formal Spiking Neuron Models"), eqs. (4.1)–(4.24). Unlike the differential
//! formulations used by [`crate::neuron::lif`], the SRM expresses the membrane
//! potential directly as a *superposition of response kernels* triggered by the
//! neuron's own last spike and by incoming presynaptic spikes:
//!
//! ```text
//! u_i(t) = η(t − t̂_i) + Σ_j w_ij Σ_f ε(t − t_j^f) + u_rest
//! ```
//!
//! where `t̂_i` is the time of the neuron's most recent output spike, `t_j^f`
//! enumerates the firing times of presynaptic neuron `j`, `η` is the
//! refractory/reset kernel, and `ε` is the post-synaptic potential (PSP)
//! kernel. A spike is emitted whenever `u_i(t)` crosses the threshold `u_th`
//! from below; the SRM₀ simplification used here keeps only dependence on the
//! single most recent output spike for the refractory term.
//!
//! The PSP kernel is the normalized difference of two exponentials with
//! membrane time constant `τ_m` and synaptic time constant `τ_s`
//! (Gerstner & Kistler eq. 4.2):
//!
//! ```text
//! ε(s) = (s ≥ 0) ? [exp(−s/τ_m) − exp(−s/τ_s)] / (τ_m − τ_s) : 0
//! ```
//!
//! In the degenerate case `τ_m = τ_s` the difference-of-exponentials collapses
//! to the *alpha function* limit `ε(s) = (s/τ²)·exp(−s/τ)` (the analytic limit
//! obtained by l'Hôpital), so the kernel stays finite everywhere.
//!
//! The refractory kernel is a single decaying hyperpolarisation triggered by
//! the neuron's own spike (Gerstner & Kistler eq. 4.4):
//!
//! ```text
//! η(s) = (s ≥ 0) ? −η_0 · exp(−s/τ_r) : 0,   η_0 > 0
//! ```
//!
//! Time constants and `dt` share consistent units (ms by the defaults).

use crate::error::{SnnError, SnnResult};

/// Configuration for the Spike Response Model (SRM₀) neuron.
///
/// All time constants and `dt` must be strictly positive. The PSP kernel
/// remains finite when `tau_m == tau_s` via the alpha-function limit.
#[derive(Debug, Clone, Copy)]
pub struct SrmConfig {
    /// Membrane time constant `τ_m` (slow decay of the PSP), same units as `dt`.
    pub tau_m: f32,
    /// Synaptic time constant `τ_s` (fast rise of the PSP), same units as `dt`.
    pub tau_s: f32,
    /// Refractory time constant `τ_r` governing the reset-kernel decay.
    pub tau_r: f32,
    /// Refractory amplitude `η_0 > 0` (depth of the post-spike reset).
    pub eta_0: f32,
    /// Resting potential `u_rest` added to every membrane evaluation.
    pub u_rest: f32,
    /// Firing threshold `u_th`.
    pub u_th: f32,
    /// Integration time step `dt`.
    pub dt: f32,
}

impl Default for SrmConfig {
    /// Canonical SRM₀ parameters: `τ_m = 10`, `τ_s = 2.5`, `τ_r = 20`,
    /// `η_0 = 5` (large enough to strongly suppress immediate re-firing),
    /// `u_rest = 0`, `u_th = 1`, `dt = 1`.
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            tau_s: 2.5,
            tau_r: 20.0,
            eta_0: 5.0,
            u_rest: 0.0,
            u_th: 1.0,
            dt: 1.0,
        }
    }
}

/// Mutable SRM state.
#[derive(Debug, Clone)]
pub struct SrmState {
    /// Last-evaluated membrane potential `u_i` per neuron, length `n`.
    pub u: Vec<f32>,
    /// Time of each neuron's most recent own spike `t̂_i`; `NEG_INFINITY` when
    /// the neuron has not yet spiked.
    pub last_spike: Vec<f32>,
    /// Current simulation time, advanced by `dt` each [`srm_step`].
    pub time: f32,
}

impl SrmState {
    /// Allocate state for `n` neurons with `u = 0`, no prior spikes, `time = 0`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            u: vec![0.0_f32; n],
            last_spike: vec![f32::NEG_INFINITY; n],
            time: 0.0,
        }
    }
}

/// Post-synaptic potential (PSP) kernel `ε(s)`.
///
/// Returns `0` for `s < 0` (causality). For `s ≥ 0` it evaluates the
/// normalized difference-of-exponentials, falling back to the alpha-function
/// limit `(s/τ²)·exp(−s/τ)` when `τ_m` and `τ_s` coincide.
#[must_use]
pub fn psp_kernel(s: f32, cfg: &SrmConfig) -> f32 {
    if s < 0.0 {
        return 0.0;
    }
    let diff = cfg.tau_m - cfg.tau_s;
    if diff.abs() < f32::EPSILON {
        // Alpha-function limit: lim_{τ_s→τ_m} ε(s) = (s/τ²)·exp(−s/τ).
        let tau = cfg.tau_m;
        (s / (tau * tau)) * (-s / tau).exp()
    } else {
        ((-s / cfg.tau_m).exp() - (-s / cfg.tau_s).exp()) / diff
    }
}

/// Refractory / reset kernel `η(s)`.
///
/// Returns `0` for `s < 0`. For `s ≥ 0` it produces the hyperpolarising reset
/// `−η_0 · exp(−s/τ_r)` that suppresses immediate re-firing after a spike.
#[must_use]
pub fn refractory_kernel(s: f32, cfg: &SrmConfig) -> f32 {
    if s < 0.0 {
        return 0.0;
    }
    -cfg.eta_0 * (-s / cfg.tau_r).exp()
}

/// Sum the PSP kernel over a presynaptic spike-time list, evaluated at time `t`.
///
/// Computes `Σ_f ε(t − t_f)` over the supplied firing times `spike_times`.
/// Spike times at or after `t` contribute zero by the kernel's causality, so
/// they are harmless. Returns a plain `f32`; the value is finite whenever the
/// configuration time constants are finite and positive.
#[must_use]
pub fn psp_train(spike_times: &[f32], t: f32, cfg: &SrmConfig) -> f32 {
    spike_times
        .iter()
        .map(|&t_f| psp_kernel(t - t_f, cfg))
        .sum()
}

/// Validate the SRM configuration and slice lengths used by [`srm_step`].
fn validate(
    state: &SrmState,
    input_psp: &[f32],
    cfg: &SrmConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_m });
    }
    if cfg.tau_s <= 0.0 || !cfg.tau_s.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_s });
    }
    if cfg.tau_r <= 0.0 || !cfg.tau_r.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_r });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.u_th.is_finite() {
        return Err(SnnError::BadThreshold { v_th: cfg.u_th });
    }
    let n = state.u.len();
    if state.last_spike.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.last_spike.len(),
        });
    }
    if input_psp.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: input_psp.len(),
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

/// Advance the SRM state by one timestep.
///
/// `input_psp[i]` is the already-summed weighted PSP contribution
/// `Σ_j w_ij Σ_f ε(t − t_j^f)` to neuron `i` at the current time, so the caller
/// precomputes the synaptic convolution (e.g. with [`psp_train`]). The membrane
/// is reconstructed as `u_i = u_rest + input_psp[i] + η(t − t̂_i)`; a spike is
/// emitted when `u_i ≥ u_th`, in which case `t̂_i` is set to the current time.
/// `spikes_out` receives `1.0` for a spike and `0.0` otherwise, and
/// `state.time` is advanced by `dt`.
pub fn srm_step(
    state: &mut SrmState,
    input_psp: &[f32],
    cfg: &SrmConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate(state, input_psp, cfg, spikes_out)?;
    let time = state.time;
    for ((u, last), (&psp, s_out)) in state
        .u
        .iter_mut()
        .zip(state.last_spike.iter_mut())
        .zip(input_psp.iter().zip(spikes_out.iter_mut()))
    {
        let eta = refractory_kernel(time - *last, cfg);
        let u_new = cfg.u_rest + psp + eta;
        *u = u_new;
        if u_new >= cfg.u_th {
            *s_out = 1.0;
            *last = time;
        } else {
            *s_out = 0.0;
        }
    }
    state.time += cfg.dt;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SrmConfig {
        SrmConfig::default()
    }

    #[test]
    fn psp_kernel_zero_for_negative_s() {
        let c = cfg();
        assert_eq!(psp_kernel(-0.1, &c), 0.0);
        assert_eq!(psp_kernel(-100.0, &c), 0.0);
    }

    #[test]
    fn psp_kernel_positive_then_decays() {
        let c = cfg();
        // At s=0 the kernel is exactly 0 (both exponentials are 1).
        assert!(psp_kernel(0.0, &c).abs() < 1e-6);
        // Rises to a positive peak, then decays toward zero.
        let early = psp_kernel(3.0, &c);
        let late = psp_kernel(60.0, &c);
        assert!(early > 0.0, "early PSP should be positive, got {early}");
        assert!(late >= 0.0 && late < early, "late={late} early={early}");
    }

    #[test]
    fn psp_kernel_alpha_limit_finite_when_equal_tau() {
        let c = SrmConfig {
            tau_m: 5.0,
            tau_s: 5.0,
            ..cfg()
        };
        let v = psp_kernel(5.0, &c);
        assert!(v.is_finite(), "alpha-limit must be finite, got {v}");
        assert!(v > 0.0, "alpha-limit value should be positive, got {v}");
        // Alpha peak is at s = τ; expected (τ/τ²)·exp(−1) = exp(−1)/τ.
        let expected = (-1.0_f32).exp() / 5.0;
        assert!((v - expected).abs() < 1e-5, "v={v} expected={expected}");
    }

    #[test]
    fn refractory_kernel_negative_and_decays() {
        let c = cfg();
        assert_eq!(refractory_kernel(-1.0, &c), 0.0);
        let near = refractory_kernel(0.0, &c);
        let far = refractory_kernel(40.0, &c);
        assert!(near < 0.0, "reset should hyperpolarise, got {near}");
        // Magnitude decays toward zero as s grows.
        assert!(
            far > near,
            "far={far} should be closer to 0 than near={near}"
        );
        assert!((near + c.eta_0).abs() < 1e-6, "η(0) should equal −η_0");
    }

    #[test]
    fn srm_step_spikes_when_input_exceeds_threshold() {
        let c = cfg();
        let mut state = SrmState::new(1);
        let input = vec![2.0_f32]; // well above u_th = 1
        let mut spikes = vec![0.0_f32];
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert_eq!(state.last_spike[0], 0.0);
    }

    #[test]
    fn srm_refractory_suppresses_immediate_refire() {
        let c = cfg();
        let mut state = SrmState::new(1);
        let input = vec![1.2_f32]; // just above threshold
        let mut spikes = vec![0.0_f32];
        // First step: neuron fires (no prior spike, no refractory term).
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        // Second step with identical input: refractory kernel (−η_0) pulls the
        // membrane far below threshold, so it must NOT spike again.
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert_eq!(spikes[0], 0.0, "u={}", state.u[0]);
        assert!(state.u[0] < c.u_th);
    }

    #[test]
    fn srm_time_advances_by_dt() {
        let c = SrmConfig { dt: 0.5, ..cfg() };
        let mut state = SrmState::new(2);
        let input = vec![0.0_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert!((state.time - 0.5).abs() < 1e-6);
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert!((state.time - 1.0).abs() < 1e-6);
    }

    #[test]
    fn srm_rejects_length_mismatch() {
        let c = cfg();
        let mut state = SrmState::new(2);
        let input = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = srm_step(&mut state, &input, &c, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn srm_rejects_bad_tau() {
        let c = SrmConfig {
            tau_m: 0.0,
            ..cfg()
        };
        let mut state = SrmState::new(1);
        let input = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let err = srm_step(&mut state, &input, &c, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn srm_rejects_bad_dt() {
        let c = SrmConfig { dt: 0.0, ..cfg() };
        let mut state = SrmState::new(1);
        let input = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let err = srm_step(&mut state, &input, &c, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn srm_multi_neuron_independence() {
        let c = cfg();
        let mut state = SrmState::new(3);
        // Only neuron 1 receives supra-threshold drive.
        let input = vec![0.0_f32, 2.0, 0.0];
        let mut spikes = vec![0.0_f32; 3];
        srm_step(&mut state, &input, &c, &mut spikes).expect("step");
        assert_eq!(spikes[0], 0.0);
        assert_eq!(spikes[1], 1.0);
        assert_eq!(spikes[2], 0.0);
        assert_eq!(state.last_spike[0], f32::NEG_INFINITY);
        assert_eq!(state.last_spike[1], 0.0);
        assert_eq!(state.last_spike[2], f32::NEG_INFINITY);
    }

    #[test]
    fn psp_train_superposes_two_spikes() {
        let c = cfg();
        let t = 10.0_f32;
        let single_a = psp_kernel(t - 2.0, &c);
        let single_b = psp_kernel(t - 5.0, &c);
        let combined = psp_train(&[2.0, 5.0], t, &c);
        assert!((combined - (single_a + single_b)).abs() < 1e-6);
        // A future spike (after t) contributes nothing.
        let with_future = psp_train(&[2.0, 5.0, 20.0], t, &c);
        assert!((with_future - combined).abs() < 1e-6);
    }
}
