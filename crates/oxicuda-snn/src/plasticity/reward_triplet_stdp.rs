#![allow(clippy::needless_range_loop)]
//! Reward-modulated triplet STDP with eligibility kernels.
//!
//! Combines the triplet rule (Pfister & Gerstner 2006) with reward-modulated
//! eligibility traces (Florian 2007; Izhikevich 2007). The triplet weight
//! delta is *not* applied to the weights directly; instead it feeds a slow
//! per-synapse eligibility trace `e_ij` that decays with `τ_e`. A (possibly
//! delayed) global reward / dopamine signal `r(t)` then gates the actual weight
//! change:
//!
//! ```text
//! Δw_trip_ij = [post_j] · (A_+ · x1_pre[i] + A2_+ · x1_pre[i] · y2_post[j])
//!            − [pre_i ] · (A_− · y1_post[j] + A2_− · y1_post[j] · x2_pre[i])
//! e_ij  ← e_ij · exp(−dt/τ_e) + Δw_trip_ij
//! w_ij  ← w_ij + lr · r(t) · e_ij            (then clamp)
//! ```
//!
//! The eligibility increment reuses exactly the triplet kernel of
//! [`crate::plasticity::triplet_stdp`] (computed from the *pre-increment* fast
//! traces `x1`/`y1` and the long traces `x2`/`y2`). With `r(t) ≡ 0` no learning
//! occurs even though the eligibility evolves normally; with `r(t) > 0` the
//! synapses whose recent pre/post/post (or pre/pre/post) triplets favoured LTP
//! are reinforced. Weights are clamped to `[w_min, w_max]` after every step.

use crate::error::{SnnError, SnnResult};
use crate::plasticity::triplet_stdp::{TripletStdpConfig, TripletTraces};

/// Reward-modulated triplet-STDP hyperparameters.
#[derive(Debug, Clone)]
pub struct RewardTripletConfig {
    /// Underlying triplet-STDP rule (supplies the eligibility increment kernel).
    pub triplet: TripletStdpConfig,
    /// Eligibility-trace time constant `τ_e` (must be > 0).
    pub tau_e: f32,
    /// Learning rate `lr` applied to `r · e_ij`.
    pub lr: f32,
}

impl Default for RewardTripletConfig {
    fn default() -> Self {
        Self {
            triplet: TripletStdpConfig::default(),
            tau_e: 200.0,
            lr: 0.1,
        }
    }
}

impl RewardTripletConfig {
    /// Construct and validate the configuration.
    ///
    /// # Errors
    /// Returns [`SnnError::BadTau`] for non-positive triplet/eligibility time
    /// constants, [`SnnError::BadDt`] for a non-positive `dt`, and
    /// [`SnnError::OutOfRange`] for a non-finite learning rate.
    pub fn new(triplet: TripletStdpConfig, tau_e: f32, lr: f32) -> SnnResult<Self> {
        let cfg = Self { triplet, tau_e, lr };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate all configuration fields.
    ///
    /// # Errors
    /// See [`RewardTripletConfig::new`].
    pub fn validate(&self) -> SnnResult<()> {
        let t = &self.triplet;
        if t.stdp.dt <= 0.0 || !t.stdp.dt.is_finite() {
            return Err(SnnError::BadDt { dt: t.stdp.dt });
        }
        if t.stdp.tau_plus <= 0.0 || !t.stdp.tau_plus.is_finite() {
            return Err(SnnError::BadTau {
                tau: t.stdp.tau_plus,
            });
        }
        if t.stdp.tau_minus <= 0.0 || !t.stdp.tau_minus.is_finite() {
            return Err(SnnError::BadTau {
                tau: t.stdp.tau_minus,
            });
        }
        if t.tau2_plus <= 0.0 || !t.tau2_plus.is_finite() {
            return Err(SnnError::BadTau { tau: t.tau2_plus });
        }
        if t.tau2_minus <= 0.0 || !t.tau2_minus.is_finite() {
            return Err(SnnError::BadTau { tau: t.tau2_minus });
        }
        if self.tau_e <= 0.0 || !self.tau_e.is_finite() {
            return Err(SnnError::BadTau { tau: self.tau_e });
        }
        if !self.lr.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: self.lr,
            });
        }
        if !t.stdp.w_min.is_finite() || !t.stdp.w_max.is_finite() || t.stdp.w_min > t.stdp.w_max {
            return Err(SnnError::OutOfRange {
                name: "w_min/w_max".into(),
                val: t.stdp.w_min,
            });
        }
        Ok(())
    }
}

/// Mutable reward-triplet state: triplet traces plus a per-synapse eligibility
/// trace laid out `[n_pre × n_post]` row-major.
#[derive(Debug, Clone)]
pub struct RewardTripletState {
    /// Underlying triplet (pair + long) eligibility traces.
    pub traces: TripletTraces,
    /// Slow per-synapse eligibility trace, `[n_pre × n_post]` row-major.
    pub eligibility: Vec<f32>,
}

impl RewardTripletState {
    /// Allocate zero state for an `n_pre × n_post` synaptic matrix.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            traces: TripletTraces::new(n_pre, n_post),
            eligibility: vec![0.0_f32; n_pre * n_post],
        }
    }
}

fn validate_state(state: &RewardTripletState, n_pre: usize, n_post: usize) -> SnnResult<()> {
    if state.traces.pair.x_pre.len() != n_pre || state.traces.x2_pre.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: state.traces.x2_pre.len(),
        });
    }
    if state.traces.pair.y_post.len() != n_post || state.traces.y2_post.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: state.traces.y2_post.len(),
        });
    }
    if state.eligibility.len() != n_pre * n_post {
        return Err(SnnError::BadShape {
            expected: n_pre * n_post,
            got: state.eligibility.len(),
        });
    }
    Ok(())
}

fn validate_io(
    weights: &[f32],
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    reward: f32,
) -> SnnResult<()> {
    if n_pre == 0 {
        return Err(SnnError::BadDim { got: n_pre });
    }
    if n_post == 0 {
        return Err(SnnError::BadDim { got: n_post });
    }
    if weights.len() != n_pre * n_post {
        return Err(SnnError::BadShape {
            expected: n_pre * n_post,
            got: weights.len(),
        });
    }
    if pre_spikes.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: pre_spikes.len(),
        });
    }
    if post_spikes.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: post_spikes.len(),
        });
    }
    if !reward.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "reward".into(),
            val: reward,
        });
    }
    Ok(())
}

/// One step of reward-modulated triplet STDP.
///
/// Decays all triplet traces, computes the triplet eligibility increment from
/// the *pre-increment* trace values, routes it into the slow eligibility trace,
/// applies the reward-gated weight update, clamps weights, and finally
/// increments the fast/long traces with the current spikes.
///
/// # Errors
/// Returns an error if shapes mismatch, the state is malformed, the reward is
/// non-finite, or the config is invalid.
pub fn reward_triplet_step(
    weights: &mut [f32],
    state: &mut RewardTripletState,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    reward: f32,
    n_pre: usize,
    n_post: usize,
    cfg: &RewardTripletConfig,
) -> SnnResult<()> {
    validate_io(weights, pre_spikes, post_spikes, n_pre, n_post, reward)?;
    validate_state(state, n_pre, n_post)?;
    cfg.validate()?;

    let t = &cfg.triplet;
    let decay_plus = (-t.stdp.dt / t.stdp.tau_plus).exp();
    let decay_minus = (-t.stdp.dt / t.stdp.tau_minus).exp();
    let decay2_plus = (-t.stdp.dt / t.tau2_plus).exp();
    let decay2_minus = (-t.stdp.dt / t.tau2_minus).exp();
    let decay_e = (-t.stdp.dt / cfg.tau_e).exp();

    // 1. Decay all triplet traces in place.
    for x in &mut state.traces.pair.x_pre {
        *x *= decay_plus;
    }
    for y in &mut state.traces.pair.y_post {
        *y *= decay_minus;
    }
    for x in &mut state.traces.x2_pre {
        *x *= decay2_plus;
    }
    for y in &mut state.traces.y2_post {
        *y *= decay2_minus;
    }

    // 2. Eligibility increment from the triplet kernel + eligibility decay,
    //    reading the *decayed but not yet incremented* traces. `state.traces`
    //    (read-only here) and `state.eligibility` (written here) are disjoint
    //    fields, so the borrows do not conflict.
    let traces = &state.traces;
    let elig = &mut state.eligibility;
    for i in 0..n_pre {
        let row_off = i * n_post;
        let pre = pre_spikes[i];
        let x1_i = traces.pair.x_pre[i];
        let x2_i = traces.x2_pre[i];
        for j in 0..n_post {
            let post = post_spikes[j];
            let mut dw = 0.0_f32;
            if post != 0.0 {
                dw += t.stdp.a_plus * x1_i;
                dw += t.a2_plus * x1_i * traces.y2_post[j];
            }
            if pre != 0.0 {
                dw -= t.stdp.a_minus * traces.pair.y_post[j];
                dw -= t.a2_minus * traces.pair.y_post[j] * x2_i;
            }
            let e = &mut elig[row_off + j];
            *e = *e * decay_e + dw;
        }
    }

    // 3. Reward-gated weight update + clamp.
    let g = cfg.lr * reward;
    for (w, &e) in weights.iter_mut().zip(state.eligibility.iter()) {
        *w = (*w + g * e).clamp(t.stdp.w_min, t.stdp.w_max);
    }

    // 4. Increment fast + long traces with the current spikes (after eligibility).
    for (x, &s) in state.traces.pair.x_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in state.traces.pair.y_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    for (x, &s) in state.traces.x2_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in state.traces.y2_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plasticity::stdp::StdpConfig;

    fn cfg() -> RewardTripletConfig {
        RewardTripletConfig {
            triplet: TripletStdpConfig {
                stdp: StdpConfig {
                    a_plus: 0.05,
                    a_minus: 0.06,
                    tau_plus: 20.0,
                    tau_minus: 20.0,
                    dt: 1.0,
                    w_min: 0.0,
                    w_max: 1.0,
                },
                a2_plus: 0.01,
                a2_minus: 0.01,
                tau2_plus: 100.0,
                tau2_minus: 100.0,
            },
            tau_e: 200.0,
            lr: 0.5,
        }
    }

    /// Build positive eligibility via a pre→post pairing (no reward applied).
    fn build_ltp_eligibility(
        state: &mut RewardTripletState,
        w: &mut [f32],
        cfg: &RewardTripletConfig,
    ) {
        // Pre at t=0
        reward_triplet_step(w, state, &[1.0], &[0.0], 0.0, 1, 1, cfg).expect("ok");
        for _ in 0..3 {
            reward_triplet_step(w, state, &[0.0], &[0.0], 0.0, 1, 1, cfg).expect("ok");
        }
        // Post at t=4 → builds positive triplet eligibility
        reward_triplet_step(w, state, &[0.0], &[1.0], 0.0, 1, 1, cfg).expect("ok");
    }

    #[test]
    fn zero_reward_leaves_weights_but_eligibility_nonzero() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(1, 1);
        let mut w = vec![0.5_f32];
        build_ltp_eligibility(&mut state, &mut w, &cfg);
        assert!(
            (w[0] - 0.5).abs() < 1e-7,
            "weights changed without reward: {}",
            w[0]
        );
        assert!(
            state.eligibility[0] > 0.0,
            "eligibility should be positive after LTP pairing: {}",
            state.eligibility[0]
        );
    }

    #[test]
    fn positive_reward_after_pairing_potentiates() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(1, 1);
        let mut w = vec![0.5_f32];
        build_ltp_eligibility(&mut state, &mut w, &cfg);
        let w_before = w[0];
        // Reward arrives one step later with no spikes → potentiation.
        reward_triplet_step(&mut w, &mut state, &[0.0], &[0.0], 1.0, 1, 1, &cfg).expect("ok");
        assert!(
            w[0] > w_before,
            "positive reward should potentiate: {w_before} → {}",
            w[0]
        );
    }

    #[test]
    fn negative_reward_after_pairing_depresses() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(1, 1);
        let mut w = vec![0.5_f32];
        build_ltp_eligibility(&mut state, &mut w, &cfg);
        let w_before = w[0];
        reward_triplet_step(&mut w, &mut state, &[0.0], &[0.0], -1.0, 1, 1, &cfg).expect("ok");
        assert!(
            w[0] < w_before,
            "negative reward should depress: {w_before} → {}",
            w[0]
        );
    }

    #[test]
    fn eligibility_decays_over_time() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(1, 1);
        let mut w = vec![0.5_f32];
        build_ltp_eligibility(&mut state, &mut w, &cfg);
        let e0 = state.eligibility[0];
        assert!(e0 > 0.0);
        for _ in 0..400 {
            reward_triplet_step(&mut w, &mut state, &[0.0], &[0.0], 0.0, 1, 1, &cfg).expect("ok");
        }
        assert!(
            state.eligibility[0] < e0 * 0.5,
            "eligibility did not decay: {e0} → {}",
            state.eligibility[0]
        );
    }

    #[test]
    fn rejects_bad_reward() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(1, 1);
        let mut w = vec![0.5_f32];
        let err = reward_triplet_step(&mut w, &mut state, &[0.0], &[0.0], f32::NAN, 1, 1, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_shape() {
        let cfg = cfg();
        let mut state = RewardTripletState::new(2, 2);
        let mut w = vec![0.0_f32; 3];
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 2];
        let err = reward_triplet_step(&mut w, &mut state, &pre, &post, 0.0, 2, 2, &cfg);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }
}
