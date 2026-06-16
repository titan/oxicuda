//! Three-factor eligibility-trace consolidation (Zenke 2021).
//!
//! Synaptic plasticity is split into two timescales. A fast, local *eligibility
//! trace* (synaptic tag) `e_ij` accumulates the coincidence of pre- and
//! post-synaptic activity and decays with a time constant `τ_e`. A slow,
//! *global neuromodulatory* (reward) signal then converts the eligibility trace
//! into a durable change of the synaptic weight `w_ij`. This separates the
//! Hebbian "when did pre and post fire together" detector from the credit-
//! assignment "was that good or bad" signal, which is the defining feature of
//! three-factor learning rules.
//!
//! ```text
//! e_ij ← e_ij · exp(−dt / τ_e) + pre_j · post_i      (per accumulate step)
//! w_ij ← w_ij + lr · gain · reward · e_ij            (per consolidate call)
//! ```
//!
//! State is stored row-major with the post-synaptic neuron as the row index:
//! synapse `(post i, pre j)` lives at flat index `i · n_pre + j`, so both the
//! trace and weight buffers have length `n_post · n_pre`.

use crate::error::{SnnError, SnnResult};

/// Hyper-parameters of the eligibility-trace consolidation rule.
#[derive(Debug, Clone, Copy)]
pub struct EligibilityConsolidationConfig {
    /// Eligibility-trace decay time constant `τ_e` (same units as `dt`); must be `> 0`.
    pub tau_e: f32,
    /// Integration time step `dt` used in the trace decay; must be `> 0`.
    pub dt: f32,
    /// Base learning rate applied during consolidation.
    pub learning_rate: f32,
    /// Neuromodulatory consolidation gain multiplying the reward signal.
    pub consolidation_gain: f32,
}

impl Default for EligibilityConsolidationConfig {
    fn default() -> Self {
        Self {
            tau_e: 20.0,
            dt: 1.0,
            learning_rate: 1e-2,
            consolidation_gain: 1.0,
        }
    }
}

/// Three-factor eligibility-trace consolidation synapse bank.
///
/// Holds the per-synapse eligibility traces and weights for a fully-connected
/// projection from `n_pre` pre-synaptic neurons to `n_post` post-synaptic
/// neurons.
#[derive(Debug, Clone)]
pub struct EligibilityConsolidation {
    /// Eligibility traces `e_ij`, row-major `[n_post × n_pre]`.
    eligibility: Vec<f32>,
    /// Synaptic weights `w_ij`, row-major `[n_post × n_pre]`.
    weights: Vec<f32>,
    /// Number of pre-synaptic neurons.
    n_pre: usize,
    /// Number of post-synaptic neurons.
    n_post: usize,
    /// Rule hyper-parameters.
    cfg: EligibilityConsolidationConfig,
    /// Pre-computed trace decay factor `exp(−dt / τ_e)`.
    decay: f32,
}

impl EligibilityConsolidation {
    /// Allocate a synapse bank with all traces and weights initialised to zero.
    ///
    /// Returns [`SnnError::BadDim`] if either dimension is zero, [`SnnError::BadTau`]
    /// if `τ_e ≤ 0` (or non-finite), and [`SnnError::BadDt`] if `dt ≤ 0` (or
    /// non-finite).
    pub fn new(
        n_pre: usize,
        n_post: usize,
        cfg: EligibilityConsolidationConfig,
    ) -> SnnResult<Self> {
        if n_pre == 0 {
            return Err(SnnError::BadDim { got: n_pre });
        }
        if n_post == 0 {
            return Err(SnnError::BadDim { got: n_post });
        }
        if cfg.tau_e <= 0.0 || !cfg.tau_e.is_finite() {
            return Err(SnnError::BadTau { tau: cfg.tau_e });
        }
        if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
            return Err(SnnError::BadDt { dt: cfg.dt });
        }
        let decay = (-cfg.dt / cfg.tau_e).exp();
        Ok(Self {
            eligibility: vec![0.0_f32; n_post * n_pre],
            weights: vec![0.0_f32; n_post * n_pre],
            n_pre,
            n_post,
            cfg,
            decay,
        })
    }

    /// Number of pre-synaptic neurons.
    #[must_use]
    pub fn n_pre(&self) -> usize {
        self.n_pre
    }

    /// Number of post-synaptic neurons.
    #[must_use]
    pub fn n_post(&self) -> usize {
        self.n_post
    }

    /// Immutable view of the weight buffer, row-major `[n_post × n_pre]`.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Mutable view of the weight buffer, row-major `[n_post × n_pre]`.
    #[must_use]
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Immutable view of the eligibility-trace buffer, row-major `[n_post × n_pre]`.
    #[must_use]
    pub fn traces(&self) -> &[f32] {
        &self.eligibility
    }

    /// Accumulate one timestep of pre/post coincidence into the eligibility traces.
    ///
    /// Each trace first decays by `exp(−dt / τ_e)`, then receives the outer
    /// product `pre_j · post_i`. The weights are left untouched until
    /// [`Self::consolidate`] is called.
    ///
    /// Returns [`SnnError::BadShape`] if `pre_spikes.len() != n_pre` or
    /// `post_spikes.len() != n_post`.
    pub fn accumulate(&mut self, pre_spikes: &[f32], post_spikes: &[f32]) -> SnnResult<()> {
        if pre_spikes.len() != self.n_pre {
            return Err(SnnError::BadShape {
                expected: self.n_pre,
                got: pre_spikes.len(),
            });
        }
        if post_spikes.len() != self.n_post {
            return Err(SnnError::BadShape {
                expected: self.n_post,
                got: post_spikes.len(),
            });
        }
        let decay = self.decay;
        for (i, &post) in post_spikes.iter().enumerate() {
            let row = &mut self.eligibility[i * self.n_pre..(i + 1) * self.n_pre];
            for (e_ij, &pre) in row.iter_mut().zip(pre_spikes.iter()) {
                *e_ij = *e_ij * decay + pre * post;
            }
        }
        Ok(())
    }

    /// Consolidate the current eligibility traces into the weights using a global
    /// reward signal: `w_ij ← w_ij + lr · gain · reward · e_ij`.
    ///
    /// A positive reward strengthens synapses with positive eligibility, a
    /// negative reward weakens them, and a zero reward leaves the weights
    /// unchanged. The eligibility traces themselves are not modified.
    pub fn consolidate(&mut self, reward: f32) {
        let scale = self.cfg.learning_rate * self.cfg.consolidation_gain * reward;
        if scale == 0.0 {
            return;
        }
        for (w_ij, &e_ij) in self.weights.iter_mut().zip(self.eligibility.iter()) {
            *w_ij += scale * e_ij;
        }
    }

    /// Reset all eligibility traces to zero, leaving the weights untouched.
    pub fn reset_traces(&mut self) {
        for e_ij in &mut self.eligibility {
            *e_ij = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EligibilityConsolidationConfig {
        EligibilityConsolidationConfig {
            tau_e: 10.0,
            dt: 1.0,
            learning_rate: 0.1,
            consolidation_gain: 1.0,
        }
    }

    #[test]
    fn trace_decays_toward_zero_without_input() {
        let mut ec = EligibilityConsolidation::new(2, 2, cfg()).expect("ctor");
        // Seed a coincidence so there is something to decay.
        ec.accumulate(&[1.0, 0.0], &[1.0, 0.0]).expect("acc");
        let seeded = ec.traces()[0];
        assert!(seeded > 0.0);
        // Feed zero input repeatedly; the trace must shrink monotonically toward 0.
        // With τ_e = 10, dt = 1 the decay is exp(-0.1) per step; 50 steps reach
        // exp(-5) ≈ 0.0067 of the seed.
        let mut prev = seeded;
        for _ in 0..50 {
            ec.accumulate(&[0.0, 0.0], &[0.0, 0.0]).expect("acc");
            let now = ec.traces()[0];
            assert!(now < prev, "trace did not decay: {now} !< {prev}");
            prev = now;
        }
        assert!(
            prev < 0.05 * seeded,
            "trace failed to approach zero: {prev}"
        );
    }

    #[test]
    fn coincidence_raises_the_right_synapse() {
        let n_pre = 3;
        let n_post = 2;
        let mut ec = EligibilityConsolidation::new(n_pre, n_post, cfg()).expect("ctor");
        // pre neuron 2 and post neuron 1 fire together.
        let pre = [0.0, 0.0, 1.0];
        let post = [0.0, 1.0];
        ec.accumulate(&pre, &post).expect("acc");
        let target = n_pre + 2; // (post 1, pre 2) → 1·n_pre + 2
        for (idx, &e) in ec.traces().iter().enumerate() {
            if idx == target {
                assert!(e > 0.0, "target synapse not potentiated");
            } else {
                assert_eq!(e, 0.0, "non-coincident synapse changed at idx {idx}");
            }
        }
    }

    #[test]
    fn positive_reward_increases_negative_reward_decreases() {
        let mut ec = EligibilityConsolidation::new(2, 2, cfg()).expect("ctor");
        ec.accumulate(&[1.0, 0.0], &[1.0, 0.0]).expect("acc");
        let idx = 0; // (post 0, pre 0)
        assert_eq!(ec.weights()[idx], 0.0);

        ec.consolidate(1.0);
        let after_pos = ec.weights()[idx];
        assert!(
            after_pos > 0.0,
            "positive reward did not strengthen synapse"
        );

        ec.consolidate(-1.0);
        let after_neg = ec.weights()[idx];
        assert!(
            after_neg < after_pos,
            "negative reward did not weaken synapse"
        );
    }

    #[test]
    fn zero_reward_leaves_weights_unchanged() {
        let mut ec = EligibilityConsolidation::new(2, 2, cfg()).expect("ctor");
        ec.accumulate(&[1.0, 1.0], &[1.0, 1.0]).expect("acc");
        let before = ec.weights().to_vec();
        ec.consolidate(0.0);
        assert_eq!(ec.weights(), before.as_slice());
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut ec = EligibilityConsolidation::new(2, 2, cfg()).expect("ctor");
        assert!(matches!(
            ec.accumulate(&[1.0, 0.0, 0.0], &[1.0, 0.0]),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            ec.accumulate(&[1.0, 0.0], &[1.0]),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn non_positive_tau_is_bad_tau() {
        let mut bad = cfg();
        bad.tau_e = 0.0;
        assert!(matches!(
            EligibilityConsolidation::new(2, 2, bad),
            Err(SnnError::BadTau { .. })
        ));
        bad.tau_e = -3.0;
        assert!(matches!(
            EligibilityConsolidation::new(2, 2, bad),
            Err(SnnError::BadTau { .. })
        ));
    }

    #[test]
    fn zero_dimension_is_bad_dim() {
        assert!(matches!(
            EligibilityConsolidation::new(0, 2, cfg()),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            EligibilityConsolidation::new(2, 0, cfg()),
            Err(SnnError::BadDim { .. })
        ));
    }
}
