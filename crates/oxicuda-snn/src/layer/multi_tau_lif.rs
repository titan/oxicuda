//! Multi-τ recurrent Leaky Integrate-and-Fire layer.
//!
//! A *single* membrane time constant forces every neuron in a population onto
//! one temporal scale, which limits how broad a window of past input a spiking
//! layer can integrate. Multi-τ (a.k.a. multi-compartment / dendritic-
//! heterogeneity) LIF neurons instead give each neuron **K parallel membrane
//! sub-states**, one per time constant `τ_k`, and read out the soma potential as
//! a *learned weighted combination* of those sub-states. Different `τ_k` decay
//! at different rates, so the neuron simultaneously tracks fast transients and
//! slow context. This mirrors the heterogeneous-time-constant families of
//! Perez-Nieves et al. (*Nature Communications* 2021, "Neural heterogeneity
//! promotes robust learning") and the temporal dendritic-heterogeneity LIF of
//! Zheng et al. (*Nature Communications* 2024).
//!
//! Discrete-time update for neuron `i`, sub-state `k`
//! (`β_k = exp(−dt / τ_k)`):
//!
//! ```text
//! I_i(t)     = Σ_j W_in[i,j] · x_j(t) + Σ_j W_rec[i,j] · s_j(t−1)
//! m_{i,k}(t) = β_k · m_{i,k}(t−1) + I_i(t)            (per-substate integration)
//! u_i(t)     = Σ_k a_{i,k} · m_{i,k}(t)               (learned soma read-out)
//! s_i(t)     = 1 if u_i(t) ≥ v_th else 0              (threshold)
//! ```
//!
//! On a spike the sub-states are reset. A `Hard` reset clamps every sub-state of
//! the firing neuron to `v_rest`; a `Soft` reset subtracts exactly `v_th` from
//! the soma potential by distributing the subtraction across sub-states in
//! proportion to their read-out weight (`δ_k = v_th · a_k / Σ_j a_j²`, which
//! gives `Σ_k a_k δ_k = v_th`). The recurrent matrix `W_rec` feeds the previous
//! timestep's spikes back into every neuron's drive, making the layer a genuine
//! recurrent spiking network.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::ResetMode;

/// Configuration for a [`MultiTauLif`] layer.
///
/// The number of sub-states `K` is taken from `taus.len()`, so the time
/// constants and the sub-state count can never disagree.
#[derive(Debug, Clone)]
pub struct MultiTauLifConfig {
    /// Number of recurrent neurons `N`; must be `> 0`.
    pub n_neurons: usize,
    /// Input dimensionality; must be `> 0`.
    pub in_dim: usize,
    /// Membrane time constants `τ_k`, one per sub-state. Length defines `K ≥ 1`
    /// and every entry must be strictly positive and finite.
    pub taus: Vec<f32>,
    /// Spike threshold; must be finite.
    pub v_th: f32,
    /// Resting / hard-reset potential; must be finite.
    pub v_rest: f32,
    /// Integration step `dt`; must be `> 0`.
    pub dt: f32,
    /// Reset behaviour applied after a spike.
    pub reset: ResetMode,
}

impl MultiTauLifConfig {
    /// Build a configuration with `v_rest = 0`, `dt = 1` and a hard reset.
    ///
    /// `K` is inferred from `taus.len()`. The full struct can still be built
    /// field-by-field when finer control is required.
    #[must_use]
    pub fn new(n_neurons: usize, in_dim: usize, taus: Vec<f32>, v_th: f32) -> Self {
        Self {
            n_neurons,
            in_dim,
            taus,
            v_th,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        }
    }

    /// Number of membrane sub-states `K = taus.len()`.
    #[must_use]
    pub fn k_sub(&self) -> usize {
        self.taus.len()
    }

    /// Validate every invariant required by [`MultiTauLif::new`].
    fn validate(&self) -> SnnResult<()> {
        if self.n_neurons == 0 {
            return Err(SnnError::BadDim {
                got: self.n_neurons,
            });
        }
        if self.in_dim == 0 {
            return Err(SnnError::BadDim { got: self.in_dim });
        }
        if self.taus.is_empty() {
            return Err(SnnError::BadDim { got: 0 });
        }
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SnnError::BadDt { dt: self.dt });
        }
        if !self.v_th.is_finite() {
            return Err(SnnError::BadThreshold { v_th: self.v_th });
        }
        if !self.v_rest.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "v_rest".to_string(),
                val: self.v_rest,
            });
        }
        for &tau in &self.taus {
            if tau <= 0.0 || !tau.is_finite() {
                return Err(SnnError::BadTau { tau });
            }
        }
        Ok(())
    }
}

/// Recurrent LIF layer in which each neuron carries `K` membrane sub-states with
/// distinct decay time constants and a learned soma read-out.
#[derive(Debug, Clone)]
pub struct MultiTauLif {
    /// Input weight matrix, row-major `[N, in_dim]`.
    pub w_in: Vec<f32>,
    /// Recurrent weight matrix, row-major `[N, N]`.
    pub w_rec: Vec<f32>,
    /// Soma read-out weights `a_{i,k}`, row-major `[N, K]`. Initialised to `1/K`.
    pub combine: Vec<f32>,
    /// Per-substate decay factors `β_k = exp(−dt / τ_k)`, length `K`.
    pub betas: Vec<f32>,
    /// Membrane time constants `τ_k`, length `K` (kept for introspection).
    pub taus: Vec<f32>,
    /// Membrane sub-state potentials `m_{i,k}`, row-major `[N, K]`.
    pub m: Vec<f32>,
    /// Spikes emitted at the previous timestep, length `N` (recurrent input).
    pub last_spikes: Vec<f32>,
    /// Number of neurons `N`.
    pub n: usize,
    /// Number of sub-states `K`.
    pub k: usize,
    /// Input dimensionality.
    pub in_dim: usize,
    /// Spike threshold.
    pub v_th: f32,
    /// Resting / hard-reset potential.
    pub v_rest: f32,
    /// Reset behaviour after a spike.
    pub reset: ResetMode,
}

impl MultiTauLif {
    /// Allocate a layer with Kaiming-normal `W_in` / `W_rec` and uniform `1/K`
    /// soma read-out weights.
    ///
    /// Returns [`SnnError::BadDim`] when `n_neurons`, `in_dim` or `K` is zero,
    /// [`SnnError::BadDt`] for a non-positive `dt`, [`SnnError::BadThreshold`] for
    /// a non-finite `v_th`, [`SnnError::BadTau`] for any non-positive `τ_k`, and
    /// [`SnnError::OutOfRange`] for a non-finite `v_rest`.
    pub fn new(cfg: &MultiTauLifConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        cfg.validate()?;
        let n = cfg.n_neurons;
        let k = cfg.taus.len();
        let in_dim = cfg.in_dim;

        let in_scale = (2.0_f32 / in_dim as f32).sqrt();
        let rec_scale = (1.0_f32 / n as f32).sqrt();
        let mut w_in = vec![0.0_f32; n * in_dim];
        let mut w_rec = vec![0.0_f32; n * n];
        rng.fill_normal(&mut w_in);
        for v in &mut w_in {
            *v *= in_scale;
        }
        rng.fill_normal(&mut w_rec);
        for v in &mut w_rec {
            *v *= rec_scale;
        }

        let betas: Vec<f32> = cfg.taus.iter().map(|&tau| (-cfg.dt / tau).exp()).collect();
        let combine = vec![1.0_f32 / k as f32; n * k];

        Ok(Self {
            w_in,
            w_rec,
            combine,
            betas,
            taus: cfg.taus.clone(),
            m: vec![0.0_f32; n * k],
            last_spikes: vec![0.0_f32; n],
            n,
            k,
            in_dim,
            v_th: cfg.v_th,
            v_rest: cfg.v_rest,
            reset: cfg.reset,
        })
    }

    /// Reset every membrane sub-state and the recurrent spike buffer to zero.
    pub fn reset_state(&mut self) {
        for v in &mut self.m {
            *v = 0.0;
        }
        for s in &mut self.last_spikes {
            *s = 0.0;
        }
    }

    /// Effective soma potential `u_i = Σ_k a_{i,k} · m_{i,k}` for every neuron,
    /// evaluated against the *current* sub-state values without stepping.
    #[must_use]
    pub fn effective_potentials(&self) -> Vec<f32> {
        let mut u = vec![0.0_f32; self.n];
        for (i, u_i) in u.iter_mut().enumerate() {
            let base = i * self.k;
            let m_row = &self.m[base..base + self.k];
            let a_row = &self.combine[base..base + self.k];
            *u_i = m_row.iter().zip(a_row.iter()).map(|(&m, &a)| a * m).sum();
        }
        u
    }

    /// Advance the layer by one timestep, writing binary spikes to `out`.
    ///
    /// `x` is the external input current of length `in_dim`; `out` has length
    /// `N`. Returns [`SnnError::BadShape`] on a length mismatch.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) -> SnnResult<()> {
        if x.len() != self.in_dim {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        if out.len() != self.n {
            return Err(SnnError::BadShape {
                expected: self.n,
                got: out.len(),
            });
        }

        // 1. Synaptic drive per neuron: feed-forward input + recurrent spikes.
        let mut currents = vec![0.0_f32; self.n];
        for (i, cur) in currents.iter_mut().enumerate() {
            let in_row = &self.w_in[i * self.in_dim..(i + 1) * self.in_dim];
            let rec_row = &self.w_rec[i * self.n..(i + 1) * self.n];
            let drive_in: f32 = in_row.iter().zip(x.iter()).map(|(&w, &xj)| w * xj).sum();
            let drive_rec: f32 = rec_row
                .iter()
                .zip(self.last_spikes.iter())
                .map(|(&w, &sj)| w * sj)
                .sum();
            *cur = drive_in + drive_rec;
        }

        // 2. Integrate sub-states, read out the soma, spike and reset.
        for (i, (cur, out_i)) in currents.iter().zip(out.iter_mut()).enumerate() {
            let i_in = *cur;
            let base = i * self.k;
            let mut u = 0.0_f32;
            for kk in 0..self.k {
                let idx = base + kk;
                let m_new = self.betas[kk] * self.m[idx] + i_in;
                self.m[idx] = m_new;
                u += self.combine[idx] * m_new;
            }
            let spiked = u >= self.v_th;
            if spiked {
                match self.reset {
                    ResetMode::Hard => {
                        for kk in 0..self.k {
                            self.m[base + kk] = self.v_rest;
                        }
                    }
                    ResetMode::Soft => {
                        let denom: f32 = self.combine[base..base + self.k]
                            .iter()
                            .map(|&a| a * a)
                            .sum();
                        if denom > 0.0 {
                            for kk in 0..self.k {
                                let a = self.combine[base + kk];
                                self.m[base + kk] -= self.v_th * a / denom;
                            }
                        }
                    }
                }
            }
            *out_i = if spiked { 1.0 } else { 0.0 };
        }

        self.last_spikes.copy_from_slice(out);
        Ok(())
    }

    /// Run a whole sequence. `input` is `[T, in_dim]` row-major; the returned
    /// `[T, N]` buffer holds the spike train. State persists across the
    /// sequence (it is *not* reset between steps).
    ///
    /// Returns [`SnnError::EmptyInput`] for an empty slice and
    /// [`SnnError::BadShape`] if the length is not a multiple of `in_dim`.
    pub fn forward_seq(&mut self, input: &[f32]) -> SnnResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if !input.len().is_multiple_of(self.in_dim) {
            return Err(SnnError::BadShape {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let t_steps = input.len() / self.in_dim;
        let mut out = vec![0.0_f32; t_steps * self.n];
        let mut step = vec![0.0_f32; self.n];
        for t in 0..t_steps {
            let x = &input[t * self.in_dim..(t + 1) * self.in_dim];
            self.forward_step(x, &mut step)?;
            out[t * self.n..(t + 1) * self.n].copy_from_slice(&step);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, in_dim: usize, taus: Vec<f32>) -> MultiTauLifConfig {
        MultiTauLifConfig::new(n, in_dim, taus, 1.0)
    }

    #[test]
    fn constructor_shapes_and_betas() {
        let mut rng = LcgRng::new(1);
        let c = cfg(4, 3, vec![2.0, 8.0, 32.0]);
        let layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        assert_eq!(layer.n, 4);
        assert_eq!(layer.k, 3);
        assert_eq!(layer.w_in.len(), 4 * 3);
        assert_eq!(layer.w_rec.len(), 4 * 4);
        assert_eq!(layer.combine.len(), 4 * 3);
        assert_eq!(layer.m.len(), 4 * 3);
        // β is monotone increasing in τ, and every β ∈ (0, 1).
        assert!(layer.betas[0] < layer.betas[1]);
        assert!(layer.betas[1] < layer.betas[2]);
        for &b in &layer.betas {
            assert!(b > 0.0 && b < 1.0);
        }
        // Uniform read-out weights sum to one per neuron.
        for i in 0..layer.n {
            let s: f32 = layer.combine[i * layer.k..(i + 1) * layer.k].iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn larger_tau_substate_decays_slower() {
        // High threshold + zero recurrence ⇒ no spikes, pure sub-state decay.
        let mut rng = LcgRng::new(2);
        let mut c = cfg(1, 1, vec![2.0, 8.0, 64.0]);
        c.v_th = 1e9;
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        layer.w_in = vec![1.0]; // pass the input straight through
        layer.w_rec = vec![0.0];
        let mut out = vec![0.0_f32; 1];
        // One unit pulse → every sub-state holds I = 1.
        layer.forward_step(&[1.0], &mut out).expect("pulse");
        let after_pulse: Vec<f32> = layer.m.clone();
        for &v in &after_pulse {
            assert!((v - 1.0).abs() < 1e-6, "pulse should load m=1, got {v}");
        }
        // Then decay with zero input for several steps.
        for _ in 0..5 {
            layer.forward_step(&[0.0], &mut out).expect("decay");
        }
        // Slower (larger τ) sub-state must retain more charge.
        assert!(layer.m[0] < layer.m[1], "τ=2 should decay below τ=8");
        assert!(layer.m[1] < layer.m[2], "τ=8 should decay below τ=64");
        // And the decayed value matches β^steps to good precision.
        let expected_fast = layer.betas[0].powi(5);
        assert!((layer.m[0] - expected_fast).abs() < 1e-5);
    }

    #[test]
    fn longer_tau_integrates_more_under_constant_drive() {
        let mut rng = LcgRng::new(3);
        let mut c = cfg(1, 1, vec![2.0, 64.0]);
        c.v_th = 1e9; // never spike, observe steady-state integration
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        layer.w_in = vec![1.0];
        layer.w_rec = vec![0.0];
        let mut out = vec![0.0_f32; 1];
        for _ in 0..50 {
            layer.forward_step(&[0.1], &mut out).expect("drive");
        }
        // Steady state m_k → I/(1−β_k); larger τ ⇒ larger β ⇒ larger plateau.
        assert!(
            layer.m[1] > layer.m[0],
            "long-τ sub-state should integrate more: short={}, long={}",
            layer.m[0],
            layer.m[1]
        );
    }

    #[test]
    fn spike_resets_membrane_hard() {
        let mut rng = LcgRng::new(4);
        let mut c = cfg(1, 1, vec![16.0, 16.0]);
        c.v_th = 0.5;
        c.v_rest = 0.0;
        c.reset = ResetMode::Hard;
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        layer.w_in = vec![1.0];
        layer.w_rec = vec![0.0];
        let mut out = vec![0.0_f32; 1];
        // u = mean of sub-states; both load to 2.0 ⇒ u = 2.0 ≥ 0.5 ⇒ spike.
        layer.forward_step(&[2.0], &mut out).expect("step");
        assert_eq!(out[0], 1.0, "strong drive must spike");
        for &v in &layer.m {
            assert!((v - c.v_rest).abs() < 1e-6, "sub-state not reset: {v}");
        }
    }

    #[test]
    fn soft_reset_subtracts_v_th_from_soma() {
        let mut rng = LcgRng::new(5);
        let mut c = cfg(1, 1, vec![1e9, 1e9]); // β ≈ 1 (no leak) for a clean check
        c.v_th = 1.0;
        c.reset = ResetMode::Soft;
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        layer.w_in = vec![1.0];
        layer.w_rec = vec![0.0];
        let mut out = vec![0.0_f32; 1];
        // Each sub-state ← 1.5 ⇒ u = 1.5 ≥ 1.0 ⇒ spike, soft reset ⇒ u ← 0.5.
        layer.forward_step(&[1.5], &mut out).expect("step");
        assert_eq!(out[0], 1.0);
        let u = layer.effective_potentials()[0];
        assert!(
            (u - 0.5).abs() < 1e-4,
            "soft reset should leave u≈0.5, got {u}"
        );
    }

    #[test]
    fn recurrent_weight_changes_spike_train() {
        let mut rng = LcgRng::new(6);
        let mut c = cfg(4, 2, vec![4.0, 16.0, 64.0]);
        c.v_th = 0.5;
        let base = MultiTauLif::new(&c, &mut rng).expect("ctor");
        let mut no_rec = base.clone();
        for w in &mut no_rec.w_rec {
            *w = 0.0;
        }
        let mut strong_rec = base.clone();
        // Strong self-excitation so a spike at t drives spikes at t+1.
        for w in &mut strong_rec.w_rec {
            *w = 0.0;
        }
        for i in 0..strong_rec.n {
            strong_rec.w_rec[i * strong_rec.n + i] = 5.0;
        }
        // Drive both with the same input sequence.
        let input: Vec<f32> = (0..20 * 2).map(|t| 0.3 + 0.02 * (t % 5) as f32).collect();
        let train_a = no_rec.forward_seq(&input).expect("seq a");
        let train_b = strong_rec.forward_seq(&input).expect("seq b");
        assert_ne!(
            train_a, train_b,
            "recurrent feedback should alter the spike train"
        );
    }

    #[test]
    fn spikes_are_binary_and_finite() {
        let mut rng = LcgRng::new(7);
        let c = cfg(4, 3, vec![3.0, 9.0, 27.0]);
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        let input: Vec<f32> = (0..20 * 3).map(|t| 0.5 + 0.1 * (t % 7) as f32).collect();
        let train = layer.forward_seq(&input).expect("seq");
        assert_eq!(train.len(), 20 * 4);
        for &s in &train {
            assert!(s == 0.0 || s == 1.0, "non-binary spike {s}");
        }
        for &v in &layer.m {
            assert!(v.is_finite(), "non-finite sub-state {v}");
        }
    }

    #[test]
    fn determinism_under_fixed_seed() {
        let c = cfg(4, 3, vec![2.0, 8.0, 32.0]);
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let mut a = MultiTauLif::new(&c, &mut rng_a).expect("a");
        let mut b = MultiTauLif::new(&c, &mut rng_b).expect("b");
        let input: Vec<f32> = (0..20 * 3).map(|t| 0.4 + 0.03 * (t % 6) as f32).collect();
        let ta = a.forward_seq(&input).expect("seq a");
        let tb = b.forward_seq(&input).expect("seq b");
        assert_eq!(ta, tb);
        assert_eq!(a.m, b.m);
    }

    #[test]
    fn reset_state_clears_substates() {
        let mut rng = LcgRng::new(8);
        let c = cfg(3, 2, vec![5.0, 10.0]);
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        let input = vec![0.6_f32; 5 * 2];
        layer.forward_seq(&input).expect("seq");
        layer.reset_state();
        for &v in &layer.m {
            assert_eq!(v, 0.0);
        }
        for &s in &layer.last_spikes {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut rng = LcgRng::new(9);
        let mut zero_n = cfg(0, 2, vec![4.0]);
        zero_n.n_neurons = 0;
        assert!(matches!(
            MultiTauLif::new(&zero_n, &mut rng),
            Err(SnnError::BadDim { .. })
        ));

        let empty_tau = cfg(2, 2, vec![]);
        assert!(matches!(
            MultiTauLif::new(&empty_tau, &mut rng),
            Err(SnnError::BadDim { .. })
        ));

        let mut bad_tau = cfg(2, 2, vec![4.0, -1.0]);
        bad_tau.v_th = 1.0;
        assert!(matches!(
            MultiTauLif::new(&bad_tau, &mut rng),
            Err(SnnError::BadTau { .. })
        ));

        let mut bad_dt = cfg(2, 2, vec![4.0]);
        bad_dt.dt = 0.0;
        assert!(matches!(
            MultiTauLif::new(&bad_dt, &mut rng),
            Err(SnnError::BadDt { .. })
        ));

        let mut bad_th = cfg(2, 2, vec![4.0]);
        bad_th.v_th = f32::NAN;
        assert!(matches!(
            MultiTauLif::new(&bad_th, &mut rng),
            Err(SnnError::BadThreshold { .. })
        ));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let mut rng = LcgRng::new(10);
        let c = cfg(3, 2, vec![4.0, 8.0]);
        let mut layer = MultiTauLif::new(&c, &mut rng).expect("ctor");
        let mut out = vec![0.0_f32; 3];
        assert!(matches!(
            layer.forward_step(&[0.0; 1], &mut out),
            Err(SnnError::BadShape { .. })
        ));
        let mut wrong_out = vec![0.0_f32; 2];
        assert!(matches!(
            layer.forward_step(&[0.0; 2], &mut wrong_out),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(layer.forward_seq(&[]), Err(SnnError::EmptyInput)));
        assert!(matches!(
            layer.forward_seq(&[0.0; 3]),
            Err(SnnError::BadShape { .. })
        ));
    }
}
