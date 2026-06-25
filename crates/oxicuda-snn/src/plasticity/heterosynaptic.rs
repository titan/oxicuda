#![allow(clippy::needless_range_loop)]
//! Heterosynaptic plasticity (Chistiakova et al. 2014; Royer & Paré 2003).
//!
//! Classical (homosynaptic) STDP only changes synapses that were themselves
//! active. *Heterosynaptic* plasticity additionally changes the *inactive*
//! synapses onto a strongly-driven post-synaptic neuron, enforcing a total
//! incoming-weight constraint Σ_i w_ij ≈ `w_total_target`. This implements
//! synaptic competition / weight normalisation and stabilises Hebbian growth.
//!
//! Each post-synaptic neuron `j` carries a slow low-pass trace `a_j` of its
//! recent drive (spike activity). The heterosynaptic term engages in proportion
//! to that drive, so only *strongly / repeatedly* driven neurons renormalise
//! their afferents — the experimentally observed gating of heterosynaptic
//! plasticity:
//!
//! ```text
//! a_j ← a_j · exp(−dt/τ_act) + post_spikes[j]            (slow post-drive trace)
//! ```
//!
//! With `s_j = Σ_i w_ij` and the activity gate `a_j`, each step combines:
//!
//! 1. **Homosynaptic** STDP-style Hebbian term on *active* synapses (pre `i`
//!    and post `j` both spiking):
//!    ```text
//!    Δw_homo_ij = a_homo · pre_i · post_j
//!    ```
//! 2. **Heterosynaptic** normalisation term applied to *all* incoming synapses
//!    of a driven neuron (active *and* inactive), scaled by the post-drive `a_j`:
//!    * Subtractive:  `Δw_hetero_ij = −β · a_j · (s_j − w_total) / n_pre`
//!    * Multiplicative: `Δw_hetero_ij = −β · a_j · w_ij · (s_j − w_total) / s_j`
//!
//! The heterosynaptic term pulls the incoming sum toward the target; because it
//! touches inactive synapses too, those weights also change — the defining
//! signature of heterosynaptic plasticity. Weights are clamped to
//! `[w_min, w_max]` after every step.

use crate::error::{SnnError, SnnResult};

/// Normalisation mode for the heterosynaptic term.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum HeteroMode {
    /// Subtract a constant per-synapse offset (Σ-preserving, additive).
    #[default]
    Subtractive,
    /// Scale each weight proportionally (preserves relative weights).
    Multiplicative,
}

/// Heterosynaptic-plasticity hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct HeterosynapticConfig {
    /// Homosynaptic Hebbian rate `a_homo` on co-active synapses.
    pub a_homo: f32,
    /// Heterosynaptic normalisation rate `β` (≥ 0).
    pub beta: f32,
    /// Target total incoming weight `w_total` per driven post-synaptic neuron.
    pub w_total_target: f32,
    /// Time constant `τ_act` of the slow per-post-neuron drive trace (> 0).
    pub tau_act: f32,
    /// Discretisation time step `dt` (> 0).
    pub dt: f32,
    /// Normalisation mode.
    pub mode: HeteroMode,
    /// Hard lower clip on synaptic weight.
    pub w_min: f32,
    /// Hard upper clip on synaptic weight.
    pub w_max: f32,
}

impl Default for HeterosynapticConfig {
    fn default() -> Self {
        Self {
            a_homo: 0.01,
            beta: 0.05,
            w_total_target: 1.0,
            tau_act: 20.0,
            dt: 1.0,
            mode: HeteroMode::Subtractive,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

impl HeterosynapticConfig {
    /// Construct and validate a heterosynaptic config.
    ///
    /// # Errors
    /// Returns [`SnnError::OutOfRange`] for non-finite fields, a negative
    /// `beta`, or `w_min > w_max`; [`SnnError::BadTau`] for a non-positive
    /// `tau_act`; and [`SnnError::BadDt`] for a non-positive `dt`.
    pub fn new(
        a_homo: f32,
        beta: f32,
        w_total_target: f32,
        tau_act: f32,
        dt: f32,
        mode: HeteroMode,
        w_min: f32,
        w_max: f32,
    ) -> SnnResult<Self> {
        let cfg = Self {
            a_homo,
            beta,
            w_total_target,
            tau_act,
            dt,
            mode,
            w_min,
            w_max,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration fields.
    ///
    /// # Errors
    /// See [`HeterosynapticConfig::new`].
    pub fn validate(&self) -> SnnResult<()> {
        if !self.a_homo.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "a_homo".into(),
                val: self.a_homo,
            });
        }
        if !self.beta.is_finite() || self.beta < 0.0 {
            return Err(SnnError::OutOfRange {
                name: "beta".into(),
                val: self.beta,
            });
        }
        if !self.w_total_target.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "w_total_target".into(),
                val: self.w_total_target,
            });
        }
        if self.tau_act <= 0.0 || !self.tau_act.is_finite() {
            return Err(SnnError::BadTau { tau: self.tau_act });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SnnError::BadDt { dt: self.dt });
        }
        if !self.w_min.is_finite() || !self.w_max.is_finite() || self.w_min > self.w_max {
            return Err(SnnError::OutOfRange {
                name: "w_min/w_max".into(),
                val: self.w_min,
            });
        }
        Ok(())
    }
}

/// Mutable heterosynaptic state: a slow per-post-neuron drive trace `a_j`.
#[derive(Debug, Clone)]
pub struct HeterosynapticTraces {
    /// Low-pass drive trace, length `n_post` (init 0).
    pub post_act: Vec<f32>,
}

impl HeterosynapticTraces {
    /// Allocate a zero-initialised drive trace for `n_post` neurons.
    #[must_use]
    pub fn new(_n_pre: usize, n_post: usize) -> Self {
        Self {
            post_act: vec![0.0_f32; n_post],
        }
    }
}

fn validate_shapes(
    weights: &[f32],
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
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
    Ok(())
}

fn validate_traces(traces: &HeterosynapticTraces, n_post: usize) -> SnnResult<()> {
    if traces.post_act.len() != n_post {
        return Err(SnnError::IncompatibleLength {
            a: n_post,
            b: traces.post_act.len(),
        });
    }
    Ok(())
}

/// One step of heterosynaptic plasticity.
///
/// Updates each post-neuron's slow drive trace `a_j`; then for every neuron
/// that spikes this step applies the homosynaptic Hebbian term to co-active
/// synapses and the activity-gated heterosynaptic normalisation term to *all*
/// of its incoming synapses, then clamps weights. The bounded gate
/// `g_j = a_j / (1 + a_j) ∈ (0, 1)` modulates the normalisation strength while
/// leaving its fixed point (`Σ_i w_ij = w_total`) unchanged.
///
/// The column sum `s_j = Σ_i w_ij` is computed once from the *pre-update*
/// weights so the whole incoming population is normalised consistently.
///
/// # Errors
/// Returns an error if shapes mismatch, the trace length is wrong, or the
/// config is invalid.
pub fn heterosynaptic_step(
    weights: &mut [f32],
    traces: &mut HeterosynapticTraces,
    pre_spikes: &[f32],
    post_spikes: &[f32],
    n_pre: usize,
    n_post: usize,
    cfg: &HeterosynapticConfig,
) -> SnnResult<()> {
    validate_shapes(weights, pre_spikes, post_spikes, n_pre, n_post)?;
    validate_traces(traces, n_post)?;
    cfg.validate()?;

    let decay_act = (-cfg.dt / cfg.tau_act).exp();

    // Update the slow post-drive trace and derive the bounded activity gate.
    let mut gate = vec![0.0_f32; n_post];
    for (j, a) in traces.post_act.iter_mut().enumerate() {
        *a = *a * decay_act + post_spikes[j];
        gate[j] = *a / (1.0 + *a);
    }

    // Pre-update column sums s_j = Σ_i w_ij for every post neuron.
    let mut col_sum = vec![0.0_f32; n_post];
    for i in 0..n_pre {
        let row_off = i * n_post;
        for j in 0..n_post {
            col_sum[j] += weights[row_off + j];
        }
    }

    let inv_n_pre = 1.0 / n_pre as f32;
    for i in 0..n_pre {
        let row_off = i * n_post;
        let pre_active = pre_spikes[i];
        for j in 0..n_post {
            // Only driven post neurons trigger any plasticity this step.
            if post_spikes[j] == 0.0 {
                continue;
            }
            let idx = row_off + j;
            let w = weights[idx];
            // Homosynaptic Hebbian term (active synapses only).
            let dw_homo = cfg.a_homo * pre_active * post_spikes[j];
            // Activity-gated heterosynaptic normalisation term (all synapses).
            let surplus = col_sum[j] - cfg.w_total_target;
            let drive = cfg.beta * gate[j];
            let dw_hetero = match cfg.mode {
                HeteroMode::Subtractive => -drive * surplus * inv_n_pre,
                HeteroMode::Multiplicative => {
                    if col_sum[j].abs() > f32::EPSILON {
                        -drive * w * surplus / col_sum[j]
                    } else {
                        0.0
                    }
                }
            };
            weights[idx] = (w + dw_homo + dw_hetero).clamp(cfg.w_min, cfg.w_max);
        }
    }
    Ok(())
}

/// Convenience: column sum `Σ_i w_ij` for post neuron `j` (row-major weights).
///
/// # Errors
/// Returns [`SnnError::BadShape`] if `weights.len() != n_pre * n_post` and
/// [`SnnError::LayerOutOfRange`] if `j >= n_post`.
pub fn incoming_sum(weights: &[f32], n_pre: usize, n_post: usize, j: usize) -> SnnResult<f32> {
    if weights.len() != n_pre * n_post {
        return Err(SnnError::BadShape {
            expected: n_pre * n_post,
            got: weights.len(),
        });
    }
    if j >= n_post {
        return Err(SnnError::LayerOutOfRange {
            idx: j,
            num_layers: n_post,
        });
    }
    let mut s = 0.0_f32;
    for i in 0..n_pre {
        s += weights[i * n_post + j];
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mode: HeteroMode) -> HeterosynapticConfig {
        HeterosynapticConfig {
            a_homo: 0.0, // isolate the normalisation behaviour by default
            beta: 0.2,
            w_total_target: 1.0,
            tau_act: 20.0,
            dt: 1.0,
            mode,
            w_min: 0.0,
            w_max: 2.0,
        }
    }

    #[test]
    fn subtractive_sum_converges_to_target() {
        // 4 pre, 1 post. Start with an inflated incoming sum (4 × 0.5 = 2.0).
        let n_pre = 4;
        let n_post = 1;
        let mut w = vec![0.5_f32; n_pre];
        let cfg = cfg(HeteroMode::Subtractive);
        let pre = vec![1.0_f32; n_pre];
        let post = vec![1.0_f32]; // post driven every step
        let mut t = HeterosynapticTraces::new(n_pre, n_post);
        for _ in 0..200 {
            heterosynaptic_step(&mut w, &mut t, &pre, &post, n_pre, n_post, &cfg).expect("ok");
        }
        let s = incoming_sum(&w, n_pre, n_post, 0).expect("ok");
        assert!(
            (s - cfg.w_total_target).abs() < 1e-2,
            "incoming sum did not converge: {s} vs target {}",
            cfg.w_total_target
        );
    }

    #[test]
    fn multiplicative_sum_converges_to_target() {
        let n_pre = 3;
        let n_post = 1;
        let mut w = vec![0.8_f32; n_pre]; // sum 2.4
        let cfg = cfg(HeteroMode::Multiplicative);
        let pre = vec![1.0_f32; n_pre];
        let post = vec![1.0_f32];
        let mut t = HeterosynapticTraces::new(n_pre, n_post);
        for _ in 0..400 {
            heterosynaptic_step(&mut w, &mut t, &pre, &post, n_pre, n_post, &cfg).expect("ok");
        }
        let s = incoming_sum(&w, n_pre, n_post, 0).expect("ok");
        assert!(
            (s - cfg.w_total_target).abs() < 1e-2,
            "multiplicative sum did not converge: {s}"
        );
    }

    #[test]
    fn inactive_synapse_also_changes() {
        // Two pre synapses onto one driven post; only pre[0] is active.
        // The heterosynaptic signature: pre[1] (inactive) must still change.
        let n_pre = 2;
        let n_post = 1;
        let mut w = vec![0.9_f32, 0.9_f32]; // sum 1.8 > target
        let cfg = HeterosynapticConfig {
            a_homo: 0.02,
            ..cfg(HeteroMode::Subtractive)
        };
        let pre = vec![1.0_f32, 0.0]; // pre[1] inactive
        let post = vec![1.0_f32];
        let w1_before = w[1];
        let mut t = HeterosynapticTraces::new(n_pre, n_post);
        heterosynaptic_step(&mut w, &mut t, &pre, &post, n_pre, n_post, &cfg).expect("ok");
        assert!(
            (w[1] - w1_before).abs() > 1e-6,
            "inactive synapse must change under heterosynaptic normalisation: {w1_before} → {}",
            w[1]
        );
        // It should have been depressed (sum was above target).
        assert!(w[1] < w1_before);
    }

    #[test]
    fn quiet_post_neuron_is_untouched() {
        // If a post neuron does not fire, none of its synapses change.
        let n_pre = 2;
        let n_post = 2;
        let mut w = vec![0.9_f32; 4];
        let before = w.clone();
        let cfg = cfg(HeteroMode::Subtractive);
        let pre = vec![1.0_f32, 1.0];
        let post = vec![1.0_f32, 0.0]; // only post 0 driven
        let mut t = HeterosynapticTraces::new(n_pre, n_post);
        heterosynaptic_step(&mut w, &mut t, &pre, &post, n_pre, n_post, &cfg).expect("ok");
        // Column 1 (post neuron 1) unchanged.
        assert!((w[1] - before[1]).abs() < 1e-9);
        assert!((w[3] - before[3]).abs() < 1e-9);
        // Column 0 changed.
        assert!((w[0] - before[0]).abs() > 1e-9 || (w[2] - before[2]).abs() > 1e-9);
    }

    #[test]
    fn drive_trace_accumulates_and_gates() {
        // Repeated drive raises the post-activity trace toward its steady state,
        // so a more-driven neuron normalises faster than a freshly-driven one.
        let n_pre = 4;
        let n_post = 1;
        let cfg = cfg(HeteroMode::Subtractive);
        let pre = vec![1.0_f32; n_pre];
        let post = vec![1.0_f32];

        // Warm up the drive trace on one copy.
        let mut t_warm = HeterosynapticTraces::new(n_pre, n_post);
        let mut w_dummy = vec![0.5_f32; n_pre];
        for _ in 0..50 {
            heterosynaptic_step(&mut w_dummy, &mut t_warm, &pre, &post, n_pre, n_post, &cfg)
                .expect("ok");
        }
        assert!(t_warm.post_act[0] > 5.0, "trace should accumulate");

        // Fresh trace vs warmed trace: identical surplus, single step each.
        let mut w_fresh = vec![0.75_f32; n_pre]; // sum 3.0
        let mut t_fresh = HeterosynapticTraces::new(n_pre, n_post);
        heterosynaptic_step(&mut w_fresh, &mut t_fresh, &pre, &post, n_pre, n_post, &cfg)
            .expect("ok");
        let mut w_warm = vec![0.75_f32; n_pre];
        heterosynaptic_step(&mut w_warm, &mut t_warm, &pre, &post, n_pre, n_post, &cfg)
            .expect("ok");
        let s_fresh = incoming_sum(&w_fresh, n_pre, n_post, 0).expect("ok");
        let s_warm = incoming_sum(&w_warm, n_pre, n_post, 0).expect("ok");
        assert!(
            s_warm < s_fresh,
            "stronger drive should pull the sum down faster: warm={s_warm} fresh={s_fresh}"
        );
    }

    #[test]
    fn rejects_negative_beta() {
        let err = HeterosynapticConfig::new(
            0.01,
            -0.1,
            1.0,
            20.0,
            1.0,
            HeteroMode::Subtractive,
            0.0,
            1.0,
        );
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_tau_act() {
        let err =
            HeterosynapticConfig::new(0.01, 0.1, 1.0, 0.0, 1.0, HeteroMode::Subtractive, 0.0, 1.0);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn rejects_bad_shape() {
        let cfg = HeterosynapticConfig::default();
        let mut w = vec![0.0_f32; 3];
        let pre = vec![0.0_f32; 2];
        let post = vec![0.0_f32; 2];
        let mut t = HeterosynapticTraces::new(2, 2);
        let err = heterosynaptic_step(&mut w, &mut t, &pre, &post, 2, 2, &cfg);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }
}
