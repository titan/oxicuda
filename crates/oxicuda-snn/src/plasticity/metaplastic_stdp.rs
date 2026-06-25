#![allow(clippy::needless_range_loop)]
//! Metaplastic STDP — pair-STDP modulated by a BCM-like sliding metaplastic
//! variable (Abraham 2008; Bienenstock-Cooper-Munro 1982).
//!
//! On top of the standard pair rule (Bi & Poo 1998) each *post-synaptic*
//! neuron carries a slow metaplastic state `m_j` that low-pass filters its
//! recent activity. `m_j` plays the role of the BCM sliding modification
//! threshold θ_M: the more strongly a post-synaptic neuron has been firing,
//! the harder it becomes to potentiate its synapses (the LTP/LTD crossover
//! moves up).
//!
//! ```text
//! m_j   ← m_j · exp(−dt/τ_meta) + post_spikes[j]           (slow activity trace)
//! g_j   = 1 / (1 + meta_gain · max(0, m_j − θ_target))     (∈ (0, 1], LTP gain)
//! ```
//!
//! The pair-STDP delta is then split into its potentiation (LTP) and
//! depression (LTD) parts, and only the LTP part is scaled by `g_j`:
//!
//! ```text
//! Δw_ij = g_j · A_+ · post_j · x_pre[i]  −  A_− · pre_i · y_post[j]
//! ```
//!
//! High recent post-synaptic activity (`m_j ≫ θ_target`) drives `g_j → 0`, so
//! the net update is dominated by depression — exactly the BCM sliding-threshold
//! behaviour realised locally at the synapse. With `meta_gain = 0` we have
//! `g_j ≡ 1` and the rule reduces *exactly* to plain pair-STDP.
//!
//! Weights are clamped to `[w_min, w_max]` after every step.

use crate::error::{SnnError, SnnResult};
use crate::plasticity::stdp::{StdpConfig, StdpTraces, validate_common};

/// Metaplastic-STDP hyperparameters: pair-STDP plus a slow metaplastic state.
#[derive(Debug, Clone, Copy)]
pub struct MetaplasticStdpConfig {
    /// Underlying pair-STDP rule.
    pub stdp: StdpConfig,
    /// Time constant `τ_meta` of the slow per-post-neuron metaplastic trace.
    pub tau_meta: f32,
    /// Target activity level `θ_target` (the BCM crossover set-point).
    pub theta_target: f32,
    /// Metaplastic gain: how strongly excess activity suppresses LTP.
    pub meta_gain: f32,
}

impl Default for MetaplasticStdpConfig {
    fn default() -> Self {
        Self {
            stdp: StdpConfig::default(),
            tau_meta: 1000.0,
            theta_target: 0.0,
            meta_gain: 1.0,
        }
    }
}

impl MetaplasticStdpConfig {
    /// Construct from an underlying pair-STDP config and validate the fields.
    ///
    /// # Errors
    /// Returns [`SnnError::BadTau`] for a non-positive `tau_meta` and
    /// [`SnnError::OutOfRange`] for non-finite `theta_target` / `meta_gain`
    /// or a negative `meta_gain`.
    pub fn new(
        stdp: StdpConfig,
        tau_meta: f32,
        theta_target: f32,
        meta_gain: f32,
    ) -> SnnResult<Self> {
        let cfg = Self {
            stdp,
            tau_meta,
            theta_target,
            meta_gain,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the metaplastic-specific fields (the pair fields are checked by
    /// `validate_common` inside the step function).
    ///
    /// # Errors
    /// See [`MetaplasticStdpConfig::new`].
    pub fn validate(&self) -> SnnResult<()> {
        if self.tau_meta <= 0.0 || !self.tau_meta.is_finite() {
            return Err(SnnError::BadTau { tau: self.tau_meta });
        }
        if !self.theta_target.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "theta_target".into(),
                val: self.theta_target,
            });
        }
        if !self.meta_gain.is_finite() || self.meta_gain < 0.0 {
            return Err(SnnError::OutOfRange {
                name: "meta_gain".into(),
                val: self.meta_gain,
            });
        }
        Ok(())
    }
}

/// Mutable metaplastic-STDP state: pair traces plus a slow per-post-neuron
/// metaplastic variable `m_j`.
#[derive(Debug, Clone)]
pub struct MetaplasticTraces {
    /// Underlying pair-STDP eligibility traces.
    pub pair: StdpTraces,
    /// Slow metaplastic state `m_j`, length `n_post` (init 0).
    pub meta: Vec<f32>,
}

impl MetaplasticTraces {
    /// Allocate zero-initialised traces for an `n_pre × n_post` synapse matrix.
    #[must_use]
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self {
            pair: StdpTraces::new(n_pre, n_post),
            meta: vec![0.0_f32; n_post],
        }
    }
}

fn validate_traces(traces: &MetaplasticTraces, n_pre: usize, n_post: usize) -> SnnResult<()> {
    if traces.pair.x_pre.len() != n_pre {
        return Err(SnnError::IncompatibleLength {
            a: n_pre,
            b: traces.pair.x_pre.len(),
        });
    }
    if traces.pair.y_post.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: traces.pair.y_post.len(),
        });
    }
    if traces.meta.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: traces.meta.len(),
        });
    }
    Ok(())
}

/// One step of metaplastic STDP.
///
/// The slow metaplastic trace `m_j` is decayed and incremented with the current
/// post-synaptic spikes; the LTP component of the pair-STDP update is then
/// scaled by the per-post gain `g_j` before weights are updated and clamped.
///
/// # Errors
/// Returns an error if shapes mismatch (via `validate_common`), if any trace
/// length is wrong, or if the metaplastic config is invalid.
pub fn metaplastic_stdp_step(
    weights: &mut [f32],
    traces: &mut MetaplasticTraces,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &MetaplasticStdpConfig,
) -> SnnResult<()> {
    validate_common(weights, pre_spikes, post_spikes, n_pre, n_post, &cfg.stdp)?;
    validate_traces(traces, n_pre, n_post)?;
    cfg.validate()?;

    let decay_pre = (-cfg.stdp.dt / cfg.stdp.tau_plus).exp();
    let decay_post = (-cfg.stdp.dt / cfg.stdp.tau_minus).exp();
    let decay_meta = (-cfg.stdp.dt / cfg.tau_meta).exp();

    // 1. Decay the fast pair traces.
    for x in traces.pair.x_pre.iter_mut() {
        *x *= decay_pre;
    }
    for y in traces.pair.y_post.iter_mut() {
        *y *= decay_post;
    }

    // 2. Decay + increment the slow metaplastic trace, then derive the per-post
    //    LTP gain g_j = 1 / (1 + meta_gain · max(0, m_j − θ_target)).
    let mut gain = vec![1.0_f32; n_post];
    for (j, m) in traces.meta.iter_mut().enumerate() {
        *m = *m * decay_meta + post_spikes[j];
        let excess = (*m - cfg.theta_target).max(0.0);
        gain[j] = 1.0 / (1.0 + cfg.meta_gain * excess);
    }

    // 3. Pair-STDP delta with the LTP part gated by g_j, using the *decayed but
    //    not yet incremented* fast traces.
    for i in 0..n_pre {
        let row_off = i * n_post;
        let pre_event = pre_spikes[i];
        let x_pre_i = traces.pair.x_pre[i];
        for j in 0..n_post {
            let post_event = post_spikes[j];
            let y_post_j = traces.pair.y_post[j];
            let dw_pot = gain[j] * cfg.stdp.a_plus * post_event * x_pre_i;
            let dw_dep = cfg.stdp.a_minus * pre_event * y_post_j;
            let w = &mut weights[row_off + j];
            *w = (*w + dw_pot - dw_dep).clamp(cfg.stdp.w_min, cfg.stdp.w_max);
        }
    }

    // 4. Increment the fast traces with the current spikes (after Δw).
    for (x, &s) in traces.pair.x_pre.iter_mut().zip(pre_spikes.iter()) {
        *x += s;
    }
    for (y, &s) in traces.pair.y_post.iter_mut().zip(post_spikes.iter()) {
        *y += s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_stdp() -> StdpConfig {
        StdpConfig {
            a_plus: 0.05,
            a_minus: 0.06,
            tau_plus: 20.0,
            tau_minus: 20.0,
            dt: 1.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// Drive a clean pre→post pairing and return the resulting weight, optionally
    /// pre-soaking the post neuron with spikes to raise its slow metaplastic
    /// state. A long settling gap (no spikes) lets the *fast* pair traces decay
    /// to ≈0 while the *slow* metaplastic trace (τ_meta ≫ τ_±) persists, so the
    /// only difference between soaked and un-soaked runs is the metaplastic gain.
    fn ltp_outcome(cfg: &MetaplasticStdpConfig, soak_post: usize) -> f32 {
        let mut traces = MetaplasticTraces::new(1, 1);
        let mut w = vec![0.5_f32];
        // Raise m_j with prior post activity (no pre ⇒ no potentiation yet).
        for _ in 0..soak_post {
            metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, cfg).expect("ok");
        }
        // Settle: drain the fast traces (≈20·τ_± ⇒ below 1e-8) but keep the
        // slow meta trace (τ_meta ≫ τ_±, so it stays high).
        for _ in 0..400 {
            metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, cfg).expect("ok");
        }
        // Reset the weight so we read off only the final pairing's net change.
        w[0] = 0.5;
        // Pre at t, post one step later → potentiation gated by m_j.
        metaplastic_stdp_step(&mut w, &mut traces, &[1.0], &[0.0], 1, 1, cfg).expect("ok");
        metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, cfg).expect("ok");
        w[0]
    }

    #[test]
    fn high_post_rate_shifts_toward_depression() {
        // θ_target = 2 so a single post spike (m_j = 1) stays below threshold
        // (gain ≈ 1, plain STDP) while heavy prior activity pushes m_j ≫ θ and
        // suppresses LTP.
        let cfg = MetaplasticStdpConfig {
            stdp: base_stdp(),
            tau_meta: 2000.0,
            theta_target: 2.0,
            meta_gain: 5.0,
        };
        let low = ltp_outcome(&cfg, 0);
        let high = ltp_outcome(&cfg, 40);
        assert!(
            high < low,
            "high prior post activity should suppress LTP: low={low} high={high}"
        );
        // The un-soaked case potentiates essentially as plain STDP.
        assert!(low > 0.5);
    }

    #[test]
    fn reduces_to_plain_stdp_when_gain_zero() {
        // meta_gain = 0 ⇒ g_j ≡ 1 ⇒ identical to plain pair-STDP regardless of m_j.
        let cfg = MetaplasticStdpConfig {
            stdp: base_stdp(),
            tau_meta: 2000.0,
            theta_target: 0.0,
            meta_gain: 0.0,
        };
        let no_soak = ltp_outcome(&cfg, 0);
        let soaked = ltp_outcome(&cfg, 50);
        assert!(
            (no_soak - soaked).abs() < 1e-6,
            "with meta_gain=0 metaplastic state must not matter: {no_soak} vs {soaked}"
        );
    }

    #[test]
    fn meta_trace_decays() {
        let cfg = MetaplasticStdpConfig {
            stdp: base_stdp(),
            tau_meta: 50.0,
            ..MetaplasticStdpConfig::default()
        };
        let mut traces = MetaplasticTraces::new(1, 1);
        let mut w = vec![0.5_f32];
        metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[1.0], 1, 1, &cfg).expect("ok");
        let m0 = traces.meta[0];
        assert!((m0 - 1.0).abs() < 1e-6);
        for _ in 0..100 {
            metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, &cfg).expect("ok");
        }
        assert!(
            traces.meta[0] < m0 * 0.5,
            "metaplastic trace did not decay: {m0} → {}",
            traces.meta[0]
        );
    }

    #[test]
    fn rejects_bad_tau_meta() {
        let cfg = MetaplasticStdpConfig {
            stdp: base_stdp(),
            tau_meta: 0.0,
            ..MetaplasticStdpConfig::default()
        };
        let mut traces = MetaplasticTraces::new(1, 1);
        let mut w = vec![0.5_f32];
        let err = metaplastic_stdp_step(&mut w, &mut traces, &[0.0], &[0.0], 1, 1, &cfg);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_negative_meta_gain() {
        let err = MetaplasticStdpConfig::new(base_stdp(), 100.0, 0.0, -1.0);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_shape() {
        let cfg = MetaplasticStdpConfig::default();
        let mut traces = MetaplasticTraces::new(2, 2);
        let mut w = vec![0.0_f32; 3];
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 2];
        let err = metaplastic_stdp_step(&mut w, &mut traces, &pre, &post, 2, 2, &cfg);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }
}
