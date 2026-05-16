//! Reward-modulated STDP — Florian (2007), Izhikevich (2007).
//!
//! Standard pair-STDP delta is *not* applied directly; instead it is
//! accumulated into a slow per-synapse eligibility trace `e_ij` that decays
//! between events. A scalar reward signal `r(t)` then gates the actual weight
//! update:
//!
//! ```text
//! e_ij  ← e_ij · exp(−dt/τ_e) + Δw_stdp_ij
//! w_ij  ← w_ij + η · r(t) · e_ij      (then clamp)
//! ```
//!
//! With `r(t) ≡ 0` no learning occurs even though traces evolve normally;
//! with `r(t) > 0` LTP-aligned synapses are reinforced.

use crate::error::{SnnError, SnnResult};
use crate::plasticity::stdp::{StdpConfig, StdpTraces, pair_delta, validate_common};

/// R-STDP hyperparameters: pair-STDP plus eligibility decay and learning rate.
#[derive(Debug, Clone, Copy)]
pub struct RStdpConfig {
    /// Underlying pair-STDP rule.
    pub stdp: StdpConfig,
    /// Eligibility-trace time constant `τ_e` (must be > 0).
    pub tau_e: f32,
    /// Learning rate `η` applied to `r · e_ij`.
    pub eta: f32,
}

impl Default for RStdpConfig {
    fn default() -> Self {
        Self {
            stdp: StdpConfig::default(),
            tau_e: 100.0,
            eta: 0.1,
        }
    }
}

/// Mutable R-STDP state: per-synapse eligibility plus underlying STDP traces.
#[derive(Debug, Clone)]
pub struct RStdpState {
    /// Per-synapse eligibility trace, layout `[n_pre × n_post]` row-major.
    pub eligibility: Vec<f32>,
    /// Underlying pair-STDP traces.
    pub stdp_traces: StdpTraces,
}

impl RStdpState {
    /// Allocate zero state for an `n_pre × n_post` synaptic matrix.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            eligibility: vec![0.0_f32; n_pre * n_post],
            stdp_traces: StdpTraces::new(n_pre, n_post),
        }
    }
}

fn validate_state(state: &RStdpState, n_pre: usize, n_post: usize) -> SnnResult<()> {
    if state.eligibility.len() != n_pre * n_post {
        return Err(SnnError::BadShape {
            expected: n_pre * n_post,
            got: state.eligibility.len(),
        });
    }
    if state.stdp_traces.x_pre.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: state.stdp_traces.x_pre.len(),
        });
    }
    if state.stdp_traces.y_post.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: state.stdp_traces.y_post.len(),
        });
    }
    Ok(())
}

/// One step of reward-modulated STDP.
pub fn r_stdp_step(
    weights: &mut [f32],
    state: &mut RStdpState,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    reward: f32,
    cfg: &RStdpConfig,
) -> SnnResult<()> {
    validate_common(weights, pre_spikes, post_spikes, n_pre, n_post, &cfg.stdp)?;
    validate_state(state, n_pre, n_post)?;
    if cfg.tau_e <= 0.0 || !cfg.tau_e.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_e });
    }
    if !cfg.eta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "eta".into(),
            val: cfg.eta,
        });
    }
    if !reward.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "reward".into(),
            val: reward,
        });
    }

    let decay_pre = (-cfg.stdp.dt / cfg.stdp.tau_plus).exp();
    let decay_post = (-cfg.stdp.dt / cfg.stdp.tau_minus).exp();
    let decay_e = (-cfg.stdp.dt / cfg.tau_e).exp();

    // 1. Decay STDP traces.
    for x in state.stdp_traces.x_pre.iter_mut() {
        *x *= decay_pre;
    }
    for y in state.stdp_traces.y_post.iter_mut() {
        *y *= decay_post;
    }

    // 2. Compute Δw_stdp using current decayed traces.
    let dw = pair_delta(
        pre_spikes,
        post_spikes,
        &state.stdp_traces.x_pre,
        &state.stdp_traces.y_post,
        n_pre,
        n_post,
        &cfg.stdp,
    );

    // 3. Update eligibility: e ← e · decay_e + Δw_stdp.
    for (e, &d) in state.eligibility.iter_mut().zip(dw.iter()) {
        *e = *e * decay_e + d;
    }

    // 4. Apply weight update gated by reward; clamp.
    let g = cfg.eta * reward;
    for (w, &e) in weights.iter_mut().zip(state.eligibility.iter()) {
        let updated = *w + g * e;
        *w = updated.clamp(cfg.stdp.w_min, cfg.stdp.w_max);
    }

    // 5. Increment STDP traces with current spikes (after Δw computation).
    for (x, &s) in state.stdp_traces.x_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in state.stdp_traces.y_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RStdpConfig {
        RStdpConfig {
            stdp: StdpConfig {
                a_plus: 0.05,
                a_minus: 0.06,
                tau_plus: 20.0,
                tau_minus: 20.0,
                dt: 1.0,
                w_min: 0.0,
                w_max: 1.0,
            },
            tau_e: 100.0,
            eta: 0.5,
        }
    }

    #[test]
    fn zero_reward_no_weight_change() {
        let cfg = cfg();
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        // Drive a strong LTP-favouring sequence with reward = 0.
        // Pre spike at t=0
        r_stdp_step(&mut w, &mut state, &[1.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        for _ in 0..3 {
            r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        }
        // Post spike at t=4
        r_stdp_step(&mut w, &mut state, &[0.0], &[1.0], 1, 1, 0.0, &cfg).expect("ok");
        // Eligibility built up but no reward → weight unchanged.
        assert!(
            (w[0] - 0.5).abs() < 1e-7,
            "w changed without reward: {}",
            w[0]
        );
        assert!(
            state.eligibility[0] > 0.0,
            "eligibility should be positive after LTP event"
        );
    }

    #[test]
    fn positive_reward_with_ltp_increases_weight() {
        let cfg = cfg();
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        // Pre at t=0
        r_stdp_step(&mut w, &mut state, &[1.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        // No-op steps so eligibility builds with positive Δw on the post spike below.
        for _ in 0..3 {
            r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        }
        // Post at t=4 with no reward — eligibility now positive
        r_stdp_step(&mut w, &mut state, &[0.0], &[1.0], 1, 1, 0.0, &cfg).expect("ok");
        // Reward at t=5 with no spikes — drives weight up.
        let w_before = w[0];
        r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 1.0, &cfg).expect("ok");
        assert!(
            w[0] > w_before,
            "expected LTP increase: {} → {}",
            w_before,
            w[0]
        );
    }

    #[test]
    fn eligibility_decays() {
        let cfg = cfg();
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        // Trigger an LTP event
        r_stdp_step(&mut w, &mut state, &[1.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        for _ in 0..3 {
            r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        }
        r_stdp_step(&mut w, &mut state, &[0.0], &[1.0], 1, 1, 0.0, &cfg).expect("ok");
        let e0 = state.eligibility[0];
        assert!(e0 > 0.0);
        // Many no-spike steps with no reward → eligibility decays geometrically.
        for _ in 0..200 {
            r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        }
        assert!(
            state.eligibility[0] < e0 * 0.5,
            "eligibility did not decay: {} → {}",
            e0,
            state.eligibility[0]
        );
    }

    #[test]
    fn negative_reward_with_ltp_decreases_weight() {
        let cfg = cfg();
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        r_stdp_step(&mut w, &mut state, &[1.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        for _ in 0..3 {
            r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg).expect("ok");
        }
        r_stdp_step(&mut w, &mut state, &[0.0], &[1.0], 1, 1, 0.0, &cfg).expect("ok");
        let w_before = w[0];
        // Punish: reward = -1
        r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, -1.0, &cfg).expect("ok");
        assert!(
            w[0] < w_before,
            "expected decrease with -reward: {} → {}",
            w_before,
            w[0]
        );
    }

    #[test]
    fn rejects_bad_eta() {
        let mut cfg = cfg();
        cfg.eta = f32::NAN;
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        let err = r_stdp_step(&mut w, &mut state, &[0.0], &[0.0], 1, 1, 0.0, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_reward() {
        let cfg = cfg();
        let mut state = RStdpState::new(1, 1);
        let mut w = vec![0.5_f32];
        let err = r_stdp_step(
            &mut w,
            &mut state,
            &[0.0],
            &[0.0],
            1,
            1,
            f32::INFINITY,
            &cfg,
        );
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
