//! Triplet STDP (Pfister & Gerstner 2006) — extends pair STDP with longer post-synaptic
//! traces so learning can capture frequency dependence.

use crate::error::{SnnError, SnnResult};
use crate::plasticity::stdp::{StdpConfig, StdpTraces};

/// Configuration for the triplet STDP rule.
#[derive(Debug, Clone)]
pub struct TripletStdpConfig {
    /// Underlying pair-rule parameters.
    pub stdp: StdpConfig,
    /// Triplet potentiation amplitude.
    pub a2_plus: f32,
    /// Triplet depression amplitude.
    pub a2_minus: f32,
    /// Time constant of the long pre-synaptic triplet trace.
    pub tau2_plus: f32,
    /// Time constant of the long post-synaptic triplet trace.
    pub tau2_minus: f32,
}

impl Default for TripletStdpConfig {
    fn default() -> Self {
        Self {
            stdp: StdpConfig::default(),
            a2_plus: 5e-3,
            a2_minus: 5e-3,
            tau2_plus: 100.0,
            tau2_minus: 100.0,
        }
    }
}

/// Pair + triplet eligibility traces.
#[derive(Debug, Clone)]
pub struct TripletTraces {
    /// Pair (short-window) traces.
    pub pair: StdpTraces,
    /// Long pre-synaptic trace `x2_pre`.
    pub x2_pre: Vec<f32>,
    /// Long post-synaptic trace `y2_post`.
    pub y2_post: Vec<f32>,
}

impl TripletTraces {
    /// Construct triplet traces of the given dimensions, all initialised to zero.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            pair: StdpTraces::new(n_pre, n_post),
            x2_pre: vec![0.0_f32; n_pre],
            y2_post: vec![0.0_f32; n_post],
        }
    }
}

/// One discrete-time triplet STDP update.
pub fn triplet_stdp_step(
    weights: &mut [f32],
    traces: &mut TripletTraces,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &TripletStdpConfig,
) -> SnnResult<()> {
    if cfg.stdp.dt <= 0.0 {
        return Err(SnnError::BadDt { dt: cfg.stdp.dt });
    }
    if cfg.tau2_plus <= 0.0 {
        return Err(SnnError::BadTau { tau: cfg.tau2_plus });
    }
    if cfg.tau2_minus <= 0.0 {
        return Err(SnnError::BadTau {
            tau: cfg.tau2_minus,
        });
    }
    if weights.len() != n_pre * n_post {
        return Err(SnnError::BadShape {
            expected: n_pre * n_post,
            got: weights.len(),
        });
    }
    if pre_spikes.len() != n_pre {
        return Err(SnnError::BadShape {
            expected: n_pre,
            got: pre_spikes.len(),
        });
    }
    if post_spikes.len() != n_post {
        return Err(SnnError::BadShape {
            expected: n_post,
            got: post_spikes.len(),
        });
    }
    if traces.pair.x_pre.len() != n_pre || traces.x2_pre.len() != n_pre {
        return Err(SnnError::BadShape {
            expected: n_pre,
            got: traces.x2_pre.len(),
        });
    }
    if traces.pair.y_post.len() != n_post || traces.y2_post.len() != n_post {
        return Err(SnnError::BadShape {
            expected: n_post,
            got: traces.y2_post.len(),
        });
    }

    // Read traces at the start of step: standard triplet rule uses the trace
    // value *before* the spike-induced increment when computing the weight update.
    let pre_old = traces.pair.x_pre.clone();
    let post_old = traces.pair.y_post.clone();
    let pre_long_old = traces.x2_pre.clone();
    let post_long_old = traces.y2_post.clone();

    // Decay all traces.
    let decay_plus = (-cfg.stdp.dt / cfg.stdp.tau_plus).exp();
    let decay_minus = (-cfg.stdp.dt / cfg.stdp.tau_minus).exp();
    let decay2_plus = (-cfg.stdp.dt / cfg.tau2_plus).exp();
    let decay2_minus = (-cfg.stdp.dt / cfg.tau2_minus).exp();

    for x in &mut traces.pair.x_pre {
        *x *= decay_plus;
    }
    for y in &mut traces.pair.y_post {
        *y *= decay_minus;
    }
    for x in &mut traces.x2_pre {
        *x *= decay2_plus;
    }
    for y in &mut traces.y2_post {
        *y *= decay2_minus;
    }

    // Increment with current spikes (after decay so traces hold spike-event impulse).
    for (x, &s) in traces.pair.x_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in traces.pair.y_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    for (x, &s) in traces.x2_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in traces.y2_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }

    // Weight update — use OLD traces for the spike-driven term to avoid self-coupling.
    for (i, &pre) in pre_spikes.iter().enumerate() {
        for (j, &post) in post_spikes.iter().enumerate() {
            let w_idx = i * n_post + j;
            let mut delta = 0.0_f32;
            if post != 0.0 {
                delta += cfg.stdp.a_plus * pre_old[i];
                delta += cfg.a2_plus * pre_old[i] * post_long_old[j];
            }
            if pre != 0.0 {
                delta -= cfg.stdp.a_minus * post_old[j];
                delta -= cfg.a2_minus * post_old[j] * pre_long_old[i];
            }
            weights[w_idx] = (weights[w_idx] + delta).clamp(cfg.stdp.w_min, cfg.stdp.w_max);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplet_reduces_to_pair_when_a2_zero() {
        let cfg = TripletStdpConfig {
            a2_plus: 0.0,
            a2_minus: 0.0,
            ..Default::default()
        };
        let mut traces = TripletTraces::new(2, 2);
        let mut weights_triplet = vec![0.5_f32; 4];

        // Pre then post (potentiation event).
        let pre = vec![1.0_f32, 0.0];
        let post = vec![0.0_f32, 0.0];
        triplet_stdp_step(&mut weights_triplet, &mut traces, &pre, &post, 2, 2, &cfg).expect("ok");
        let pre = vec![0.0_f32, 0.0];
        let post = vec![1.0_f32, 0.0];
        triplet_stdp_step(&mut weights_triplet, &mut traces, &pre, &post, 2, 2, &cfg).expect("ok");
        // Should have potentiated w[0,0].
        assert!(weights_triplet[0] > 0.5);
    }

    #[test]
    fn shape_validation() {
        let cfg = TripletStdpConfig::default();
        let mut traces = TripletTraces::new(2, 2);
        let mut weights = vec![0.0_f32; 4];
        let pre = vec![0.0_f32; 3];
        let post = vec![0.0_f32; 2];
        assert!(triplet_stdp_step(&mut weights, &mut traces, &pre, &post, 2, 2, &cfg).is_err());
    }
}
