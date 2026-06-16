//! ReSuMe — Remote Supervised Method for spike-train learning.
//!
//! Reference: Ponulak & Kasiński, "Supervised learning in spiking neural
//! networks with ReSuMe: sequence learning, classification, and spike
//! shifting", *Neural Computation* 22(2), 467–510 (2010). ReSuMe trains a
//! synapse so that the actual output spike train `S_o(t)` converges to a
//! *desired* (teacher) spike train `S_d(t)` by combining a Hebbian
//! STDP-like window with an anti-Hebbian term gated by the supervisory error:
//!
//! ```text
//! Δw_i(t) = [S_d(t) − S_o(t)] · [ a + Σ_s W(s) · S_in_i(t − s) ]
//! ```
//!
//! where `S_in_i` is the presynaptic spike train of input `i`, `a` is a
//! non-Hebbian constant that drives the mean firing rate, and `W(s)` is an
//! exponential learning window `W(s) = A · exp(−s/τ)` for `s > 0`. The double
//! convolution over the input history is implemented efficiently with a
//! per-input *eligibility trace* that decays each timestep:
//!
//! ```text
//! tr_i ← tr_i · exp(−dt/τ) + S_in_i(t)
//! Δw_i  = λ · (S_d − S_o) · (a + A · tr_i)
//! ```
//!
//! The error `S_d − S_o ∈ {−1, 0, +1}` selects potentiation when the teacher
//! fires but the output does not, depression in the opposite case, and no
//! Hebbian change when the two trains agree at that timestep. Weights are
//! clamped to `[w_min, w_max]` after each update.

use crate::error::{SnnError, SnnResult};

/// ReSuMe hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct ResumeConfig {
    /// Learning-window time constant `τ` (controls eligibility-trace decay).
    pub tau: f32,
    /// Non-Hebbian constant `a` driving the baseline rate adjustment.
    pub a_const: f32,
    /// Learning-window amplitude `A` (Hebbian gain on the eligibility trace).
    pub a_amp: f32,
    /// Learning rate `λ`.
    pub learning_rate: f32,
    /// Integration step `dt`.
    pub dt: f32,
    /// Hard lower clip on the synaptic weight.
    pub w_min: f32,
    /// Hard upper clip on the synaptic weight.
    pub w_max: f32,
}

impl Default for ResumeConfig {
    /// Canonical settings: `τ = 10`, `a = 0`, `A = 1`, `λ = 0.01`, `dt = 1`,
    /// weights clamped to `[−1, 1]`.
    fn default() -> Self {
        Self {
            tau: 10.0,
            a_const: 0.0,
            a_amp: 1.0,
            learning_rate: 0.01,
            dt: 1.0,
            w_min: -1.0,
            w_max: 1.0,
        }
    }
}

/// Per-input eligibility traces for ReSuMe.
#[derive(Debug, Clone)]
pub struct ResumeState {
    /// Exponentially-decaying presynaptic eligibility trace per input,
    /// length `n_inputs`.
    pub traces: Vec<f32>,
}

impl ResumeState {
    /// Allocate zero-initialised traces for `n_inputs` afferents.
    #[must_use]
    pub fn new(n_inputs: usize) -> Self {
        Self {
            traces: vec![0.0_f32; n_inputs],
        }
    }
}

/// Exponential learning window `W(s) = A · exp(−s/τ)` for `s ≥ 0`, else `0`.
#[must_use]
pub fn learning_window(s: f32, cfg: &ResumeConfig) -> f32 {
    if s < 0.0 {
        return 0.0;
    }
    cfg.a_amp * (-s / cfg.tau).exp()
}

/// Validate the ReSuMe configuration shared across step variants.
fn validate_cfg(cfg: &ResumeConfig) -> SnnResult<()> {
    if cfg.tau <= 0.0 || !cfg.tau.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.w_min.is_finite() || !cfg.w_max.is_finite() || cfg.w_min > cfg.w_max {
        return Err(SnnError::OutOfRange {
            name: "w_min/w_max".into(),
            val: cfg.w_min,
        });
    }
    Ok(())
}

/// Trace decay factor `exp(−dt/τ)` for the ReSuMe eligibility traces.
#[must_use]
pub fn resume_decay(cfg: &ResumeConfig) -> f32 {
    (-cfg.dt / cfg.tau).exp()
}

/// Advance single-output ReSuMe learning by one timestep.
///
/// `weights` holds one weight per input (length `n_inputs`). The presynaptic
/// eligibility traces in `state` are decayed and incremented by `in_spikes`,
/// then each weight is updated by
/// `w_i += λ · (desired − actual) · (a + A · tr_i)` and clamped to
/// `[w_min, w_max]`. `desired` and `actual` are the teacher and output spike
/// indicators at this timestep (typically `0.0` or `1.0`).
pub fn resume_step(
    weights: &mut [f32],
    state: &mut ResumeState,
    in_spikes: &[f32],
    desired: f32,
    actual: f32,
    cfg: &ResumeConfig,
) -> SnnResult<()> {
    validate_cfg(cfg)?;
    let n = weights.len();
    if state.traces.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.traces.len(),
        });
    }
    if in_spikes.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: in_spikes.len(),
        });
    }
    let decay = resume_decay(cfg);
    let error = desired - actual;
    for ((w, tr), &s_in) in weights
        .iter_mut()
        .zip(state.traces.iter_mut())
        .zip(in_spikes.iter())
    {
        *tr = *tr * decay + s_in;
        let dw = cfg.learning_rate * error * (cfg.a_const + cfg.a_amp * *tr);
        *w = (*w + dw).clamp(cfg.w_min, cfg.w_max);
    }
    Ok(())
}

/// Advance multi-output ReSuMe learning by one timestep.
///
/// `weights` is a flat `n_out × n_in` row-major matrix
/// (`w[o, i] = weights[o*n_in + i]`); the per-input eligibility traces in
/// `traces` are shared across all outputs. `desired` and `actual` carry one
/// indicator per output. Each weight is updated with the same rule as
/// [`resume_step`] using its row's `(desired − actual)` error and the shared
/// input trace.
pub fn resume_step_multi(
    weights: &mut [f32],
    traces: &mut ResumeState,
    in_spikes: &[f32],
    desired: &[f32],
    actual: &[f32],
    n_out: usize,
    n_in: usize,
    cfg: &ResumeConfig,
) -> SnnResult<()> {
    validate_cfg(cfg)?;
    if n_out == 0 {
        return Err(SnnError::BadDim { got: n_out });
    }
    if n_in == 0 {
        return Err(SnnError::BadDim { got: n_in });
    }
    if weights.len() != n_out * n_in {
        return Err(SnnError::BadShape {
            expected: n_out * n_in,
            got: weights.len(),
        });
    }
    if traces.traces.len() != n_in {
        return Err(SnnError::IncompatibleLength {
            a: n_in,
            b: traces.traces.len(),
        });
    }
    if in_spikes.len() != n_in {
        return Err(SnnError::IncompatibleLength {
            a: n_in,
            b: in_spikes.len(),
        });
    }
    if desired.len() != n_out {
        return Err(SnnError::IncompatibleLength {
            a: n_out,
            b: desired.len(),
        });
    }
    if actual.len() != n_out {
        return Err(SnnError::IncompatibleLength {
            a: n_out,
            b: actual.len(),
        });
    }
    let decay = resume_decay(cfg);
    // Decay/increment shared input traces once.
    for (tr, &s_in) in traces.traces.iter_mut().zip(in_spikes.iter()) {
        *tr = *tr * decay + s_in;
    }
    // Apply per-output update reusing the shared traces.
    for o in 0..n_out {
        let error = desired[o] - actual[o];
        let row_off = o * n_in;
        for i in 0..n_in {
            let tr = traces.traces[i];
            let dw = cfg.learning_rate * error * (cfg.a_const + cfg.a_amp * tr);
            let w = &mut weights[row_off + i];
            *w = (*w + dw).clamp(cfg.w_min, cfg.w_max);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ResumeConfig {
        ResumeConfig::default()
    }

    #[test]
    fn learning_window_zero_for_negative_decays_for_positive() {
        let c = cfg();
        assert_eq!(learning_window(-1.0, &c), 0.0);
        let near = learning_window(0.0, &c);
        let far = learning_window(50.0, &c);
        assert!((near - c.a_amp).abs() < 1e-6, "W(0) should equal A");
        assert!(far >= 0.0 && far < near, "near={near} far={far}");
    }

    #[test]
    fn positive_error_with_input_increases_weight() {
        let c = cfg();
        let mut w = vec![0.0_f32];
        let mut state = ResumeState::new(1);
        // desired=1, actual=0, input present → potentiation.
        resume_step(&mut w, &mut state, &[1.0], 1.0, 0.0, &c).expect("step");
        assert!(w[0] > 0.0, "weight should increase, got {}", w[0]);
    }

    #[test]
    fn negative_error_decreases_weight() {
        let c = cfg();
        let mut w = vec![0.0_f32];
        let mut state = ResumeState::new(1);
        // desired=0, actual=1, input present → depression.
        resume_step(&mut w, &mut state, &[1.0], 0.0, 1.0, &c).expect("step");
        assert!(w[0] < 0.0, "weight should decrease, got {}", w[0]);
    }

    #[test]
    fn zero_error_no_weight_change() {
        let c = cfg();
        let mut w = vec![0.3_f32, -0.2, 0.5];
        let before = w.clone();
        let mut state = ResumeState::new(3);
        // desired == actual → error 0 → no Hebbian change regardless of input.
        resume_step(&mut w, &mut state, &[1.0, 1.0, 0.0], 1.0, 1.0, &c).expect("step");
        for (a, b) in w.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-7, "weight changed despite zero error");
        }
        // Traces still update even when error is zero.
        assert!(state.traces[0] > 0.0);
    }

    #[test]
    fn traces_decay_toward_zero_without_input() {
        let c = cfg();
        let mut w = vec![0.0_f32];
        let mut state = ResumeState::new(1);
        // Inject one spike to seed the trace.
        resume_step(&mut w, &mut state, &[1.0], 0.0, 0.0, &c).expect("step");
        let seeded = state.traces[0];
        assert!(seeded > 0.0);
        for _ in 0..500 {
            resume_step(&mut w, &mut state, &[0.0], 0.0, 0.0, &c).expect("step");
        }
        assert!(state.traces[0].abs() < 1e-6, "trace={}", state.traces[0]);
    }

    #[test]
    fn clamp_to_w_max_respected() {
        let c = cfg();
        let mut w = vec![0.99_f32];
        let mut state = ResumeState::new(1);
        for _ in 0..1000 {
            resume_step(&mut w, &mut state, &[1.0], 1.0, 0.0, &c).expect("step");
        }
        assert!(w[0] <= c.w_max + 1e-6, "w={}", w[0]);
        assert!((w[0] - c.w_max).abs() < 1e-4, "should saturate at w_max");
    }

    #[test]
    fn clamp_to_w_min_respected() {
        let c = cfg();
        let mut w = vec![-0.99_f32];
        let mut state = ResumeState::new(1);
        for _ in 0..1000 {
            resume_step(&mut w, &mut state, &[1.0], 0.0, 1.0, &c).expect("step");
        }
        assert!(w[0] >= c.w_min - 1e-6, "w={}", w[0]);
        assert!((w[0] - c.w_min).abs() < 1e-4, "should saturate at w_min");
    }

    #[test]
    fn length_mismatch_rejected() {
        let c = cfg();
        let mut w = vec![0.0_f32; 3];
        let mut state = ResumeState::new(3);
        let err = resume_step(&mut w, &mut state, &[1.0; 2], 1.0, 0.0, &c);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn bad_tau_rejected() {
        let c = ResumeConfig { tau: 0.0, ..cfg() };
        let mut w = vec![0.0_f32; 1];
        let mut state = ResumeState::new(1);
        let err = resume_step(&mut w, &mut state, &[1.0], 1.0, 0.0, &c);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn multi_output_shape_mismatch_rejected() {
        let c = cfg();
        let mut w = vec![0.0_f32; 5]; // should be n_out*n_in = 2*3 = 6
        let mut traces = ResumeState::new(3);
        let err = resume_step_multi(
            &mut w,
            &mut traces,
            &[1.0; 3],
            &[1.0; 2],
            &[0.0; 2],
            2,
            3,
            &c,
        );
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn multi_output_only_errored_output_changes() {
        let c = cfg();
        let n_out = 2;
        let n_in = 2;
        let mut w = vec![0.0_f32; n_out * n_in];
        let mut traces = ResumeState::new(n_in);
        // Output 0 has error (1−0=1), output 1 has no error (1−1=0).
        resume_step_multi(
            &mut w,
            &mut traces,
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 1.0],
            n_out,
            n_in,
            &c,
        )
        .expect("step");
        // Row 0 changed, row 1 unchanged.
        assert!(w[0] > 0.0 && w[1] > 0.0, "row 0 should change");
        assert!(
            (w[2]).abs() < 1e-7 && (w[3]).abs() < 1e-7,
            "row 1 must stay 0"
        );
    }

    #[test]
    fn multi_output_spike_length_mismatch_rejected() {
        let c = cfg();
        let mut w = vec![0.0_f32; 6];
        let mut traces = ResumeState::new(3);
        // in_spikes length 2 != n_in 3.
        let err = resume_step_multi(
            &mut w,
            &mut traces,
            &[1.0; 2],
            &[1.0; 2],
            &[0.0; 2],
            2,
            3,
            &c,
        );
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn convergence_monotone_until_clamp() {
        // Repeatedly driving desired=1, actual=0 with a steady input should
        // raise the weight monotonically until it hits w_max.
        let c = cfg();
        let mut w = vec![0.0_f32];
        let mut state = ResumeState::new(1);
        let mut prev = w[0];
        for _ in 0..2000 {
            resume_step(&mut w, &mut state, &[1.0], 1.0, 0.0, &c).expect("step");
            assert!(w[0] >= prev - 1e-6, "weight decreased: {prev} → {}", w[0]);
            prev = w[0];
        }
        assert!(
            (w[0] - c.w_max).abs() < 1e-4,
            "should reach w_max, got {}",
            w[0]
        );
    }
}
