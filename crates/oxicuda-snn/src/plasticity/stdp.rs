//! Pair-based STDP (Bi & Poo 1998) with exponentially decaying eligibility traces.
//!
//! For each (pre, post) pair the synaptic weight is updated according to the
//! standard pair rule:
//!
//! ```text
//! Δw_ij = +A_+ · x_pre[i]   on a post-synaptic spike at time t (post j fires)
//! Δw_ij = −A_− · y_post[j]  on a pre-synaptic spike at time t (pre i fires)
//! ```
//!
//! where the spike-eligibility traces decay exponentially between events:
//!
//! ```text
//! x_pre  ← x_pre  · exp(−dt/τ_+)  + pre_spikes
//! y_post ← y_post · exp(−dt/τ_−)  + post_spikes
//! ```
//!
//! The combined update is applied once per timestep and then weights are
//! clamped to `[w_min, w_max]`.

use crate::error::{SnnError, SnnResult};

/// Pair-STDP hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct StdpConfig {
    /// LTP amplitude `A_+` (positive).
    pub a_plus: f32,
    /// LTD amplitude `A_−` (positive; subtracted on pre-spikes).
    pub a_minus: f32,
    /// LTP decay time constant `τ_+` (controls `x_pre` decay).
    pub tau_plus: f32,
    /// LTD decay time constant `τ_−` (controls `y_post` decay).
    pub tau_minus: f32,
    /// Discretisation time step.
    pub dt: f32,
    /// Hard lower clip on the synaptic weight.
    pub w_min: f32,
    /// Hard upper clip on the synaptic weight.
    pub w_max: f32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            dt: 1.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

/// Eligibility traces for pair-STDP, sized for `n_pre` and `n_post` neurons.
#[derive(Debug, Clone)]
pub struct StdpTraces {
    /// Pre-synaptic LTP trace, length `n_pre`.
    pub x_pre: Vec<f32>,
    /// Post-synaptic LTD trace, length `n_post`.
    pub y_post: Vec<f32>,
}

impl StdpTraces {
    /// Allocate zero-initialised traces.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            x_pre: vec![0.0_f32; n_pre],
            y_post: vec![0.0_f32; n_post],
        }
    }
}

/// Validate STDP hyperparameters and slice shapes shared across rules.
pub(crate) fn validate_common(
    weights: &[f32],
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &StdpConfig,
) -> SnnResult<()> {
    if n_pre == 0 {
        return Err(SnnError::BadDim { got: n_pre });
    }
    if n_post == 0 {
        return Err(SnnError::BadDim { got: n_post });
    }
    if cfg.tau_plus <= 0.0 || !cfg.tau_plus.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_plus });
    }
    if cfg.tau_minus <= 0.0 || !cfg.tau_minus.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_minus });
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
    Ok(())
}

fn validate_traces(traces: &StdpTraces, n_pre: usize, n_post: usize) -> SnnResult<()> {
    if traces.x_pre.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: traces.x_pre.len(),
        });
    }
    if traces.y_post.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: traces.y_post.len(),
        });
    }
    Ok(())
}

/// Compute the pair-STDP weight delta `Δw_ij` for the *current* spikes using
/// the *current* (decayed but not yet incremented) traces, and return as
/// `[n_pre × n_post]` row-major (`Δw[i, j] = out[i*n_post + j]`).
///
/// Used internally by R-STDP / triplet rules to avoid duplicating the kernel.
pub(crate) fn pair_delta(
    pre_spikes: &[f32],
    post_spikes: &[f32],
    x_pre_decayed: &[f32],
    y_post_decayed: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &StdpConfig,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_pre * n_post];
    for i in 0..n_pre {
        let row_off = i * n_post;
        let pre_event = pre_spikes[i];
        let x_pre_i = x_pre_decayed[i];
        for j in 0..n_post {
            let post_event = post_spikes[j];
            let y_post_j = y_post_decayed[j];
            // LTP on post-spike using current pre-trace.
            let dw_pot = cfg.a_plus * post_event * x_pre_i;
            // LTD on pre-spike using current post-trace.
            let dw_dep = cfg.a_minus * pre_event * y_post_j;
            out[row_off + j] = dw_pot - dw_dep;
        }
    }
    out
}

/// Advance pair-STDP by one step: decay traces, apply Δw, clamp weights, then
/// add the new spikes to the traces.
pub fn stdp_step(
    weights: &mut [f32],
    traces: &mut StdpTraces,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &StdpConfig,
) -> SnnResult<()> {
    validate_common(weights, pre_spikes, post_spikes, n_pre, n_post, cfg)?;
    validate_traces(traces, n_pre, n_post)?;

    let decay_pre = (-cfg.dt / cfg.tau_plus).exp();
    let decay_post = (-cfg.dt / cfg.tau_minus).exp();

    // 1. Decay traces (pre-event values used for Δw).
    for x in traces.x_pre.iter_mut() {
        *x *= decay_pre;
    }
    for y in traces.y_post.iter_mut() {
        *y *= decay_post;
    }

    // 2. Compute Δw using *decayed but not yet incremented* traces.
    let dw = pair_delta(
        pre_spikes,
        post_spikes,
        &traces.x_pre,
        &traces.y_post,
        n_pre,
        n_post,
        cfg,
    );
    for (w, &d) in weights.iter_mut().zip(dw.iter()) {
        let updated = *w + d;
        *w = updated.clamp(cfg.w_min, cfg.w_max);
    }

    // 3. Increment traces with current spikes (after Δw is applied).
    for (x, &s) in traces.x_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in traces.y_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> StdpConfig {
        StdpConfig {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            dt: 1.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    #[test]
    fn post_after_pre_potentiates() {
        // Pre fires at t=0, post fires at t=5: Δw should be positive.
        let mut traces = StdpTraces::new(1, 1);
        let mut w = vec![0.5_f32];
        let cfg = cfg();
        // t=0: pre spikes
        stdp_step(&mut w, &mut traces, &[1.0], &[0.0], 1, 1, &cfg).expect("ok");
        let w_after_pre = w[0];
        // t=1..4: no spikes, traces decay
        for _ in 0..4 {
            stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, &cfg).expect("ok");
        }
        // t=5: post spikes — should be LTP because x_pre still > 0
        stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, &cfg).expect("ok");
        assert!(
            w[0] > w_after_pre,
            "expected LTP: {} → {}",
            w_after_pre,
            w[0]
        );
    }

    #[test]
    fn pre_after_post_depresses() {
        // Post fires at t=0, pre fires at t=5: Δw should be negative.
        let mut traces = StdpTraces::new(1, 1);
        let mut w = vec![0.5_f32];
        let cfg = cfg();
        stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, &cfg).expect("ok");
        let w_after_post = w[0];
        for _ in 0..4 {
            stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, &cfg).expect("ok");
        }
        stdp_step(&mut w, &mut traces, &[1.0], &[0.0], 1, 1, &cfg).expect("ok");
        assert!(
            w[0] < w_after_post,
            "expected LTD: {} → {}",
            w_after_post,
            w[0]
        );
    }

    #[test]
    fn weight_clamping_upper() {
        let mut traces = StdpTraces::new(1, 1);
        traces.x_pre[0] = 100.0; // inflated trace
        let mut w = vec![0.99_f32];
        let cfg = cfg();
        // Force several large LTP events; weight must not exceed w_max.
        for _ in 0..5 {
            stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, &cfg).expect("ok");
        }
        assert!(w[0] <= cfg.w_max + 1e-6);
    }

    #[test]
    fn weight_clamping_lower() {
        let mut traces = StdpTraces::new(1, 1);
        traces.y_post[0] = 100.0;
        let mut w = vec![0.01_f32];
        let cfg = cfg();
        for _ in 0..5 {
            stdp_step(&mut w, &mut traces, &[1.0], &[0.0], 1, 1, &cfg).expect("ok");
        }
        assert!(w[0] >= cfg.w_min - 1e-6);
    }

    #[test]
    fn traces_decay_exponentially() {
        let mut traces = StdpTraces::new(1, 1);
        let mut w = vec![0.0_f32];
        let cfg = cfg();
        stdp_step(&mut w, &mut traces, &[1.0], &[1.0], 1, 1, &cfg).expect("ok");
        // After step: x_pre = 1, y_post = 1 (decayed-from-zero then incremented).
        let x0 = traces.x_pre[0];
        let y0 = traces.y_post[0];
        assert!((x0 - 1.0).abs() < 1e-6);
        assert!((y0 - 1.0).abs() < 1e-6);
        // After another step with no spikes, decayed by exp(-1/20) ≈ 0.9512.
        stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, &cfg).expect("ok");
        let expected = (-1.0_f32 / 20.0).exp();
        assert!((traces.x_pre[0] - expected).abs() < 1e-5);
        assert!((traces.y_post[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn rejects_bad_shape() {
        let mut traces = StdpTraces::new(2, 2);
        let mut w = vec![0.0_f32; 3];
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 2];
        let err = stdp_step(&mut w, &mut traces, &pre, &post, 2, 2, &cfg());
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn no_change_when_no_spikes_and_no_traces() {
        let mut traces = StdpTraces::new(2, 3);
        let mut w = vec![0.5_f32; 6];
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 3];
        stdp_step(&mut w, &mut traces, &pre, &post, 2, 3, &cfg()).expect("ok");
        for &wi in &w {
            assert!((wi - 0.5).abs() < 1e-7);
        }
    }
}
