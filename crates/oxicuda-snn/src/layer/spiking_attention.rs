//! Spikformer Spiking Self-Attention (SSA), Zhou et al. 2023 ICLR "Spikformer".
//!
//! Spike-driven multi-head self-attention with **no softmax**. Because spikes are
//! non-negative binary values, the quadratic softmax of vanilla attention is
//! replaced by a spiking neuron (SN) acting on the raw attention scores. The
//! formulation implemented here is the canonical Spikformer SSA:
//!
//! ```text
//! Q_s = SN_Q(W_Q · X),   K_s = SN_K(W_K · X),   V_s = SN_V(W_V · X)
//! A   = SN_A( (Q_s · K_sᵀ) · scale )                     // binary attention map
//! O   = W_O · concat_heads( A · V_s )
//! ```
//!
//! All projections (`W_Q`, `W_K`, `W_V`, `W_O`) reuse the spiking-linear
//! convention (`current = W · x`, then a LIF neuron emits binary spikes), so
//! every quantity that flows between stages is a 0/1 spike train. The `scale`
//! factor controls the magnitude of the score map *before* the spiking neuron
//! `SN_A`; a larger `scale` pushes more entries past threshold and raises the
//! attention firing rate. The attention is computed per timestep, independently
//! across the `n_timesteps` slices, with each head operating on a contiguous
//! `head_dim = embed_dim / n_heads` slice of the projected channels.
//!
//! Layout. Input and output are spike trains shaped
//! `[n_timesteps][seq_len][embed_dim]`, stored as one flat row-major
//! `Vec<f32>` of length `n_timesteps * seq_len * embed_dim` whose entries are
//! `0.0` or `1.0`.
//!
//! Design choice (documented per spec). The score-then-value order
//! `SN((Q_s · K_sᵀ) · scale) · V_s` is used (rather than the linear-attention
//! order `Q_s · (K_sᵀ · V_s)`), matching the Spikformer paper's SSA module: the
//! score map is passed through its own spiking neuron so the attention weights
//! are themselves binary spikes, preserving the fully spike-driven property.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Configuration for [`SpikingSelfAttention`].
#[derive(Debug, Clone, Copy)]
pub struct SsaConfig {
    /// Model / embedding dimension. Must be divisible by `n_heads`.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of discrete timesteps in each spike train.
    pub n_timesteps: usize,
    /// Spike threshold shared by the projection and attention neurons.
    pub threshold: f32,
    /// Membrane time constant `τ_m` for the LIF neurons.
    pub tau: f32,
    /// Scale applied to the `Q_s · K_sᵀ` score map before the attention neuron.
    pub scale: f32,
}

impl Default for SsaConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            n_heads: 8,
            n_timesteps: 4,
            threshold: 1.0,
            tau: 2.0,
            scale: 0.125,
        }
    }
}

/// Spikformer spike-driven multi-head self-attention layer.
///
/// Holds the four projection weight matrices (row-major `[embed_dim, embed_dim]`)
/// together with the LIF configuration shared by all spiking neurons. The layer
/// is stateless across calls to [`SpikingSelfAttention::forward`]: each call
/// resets its internal membrane states so the result depends only on the input
/// spike train and the (fixed) weights.
#[derive(Debug, Clone)]
pub struct SpikingSelfAttention {
    /// Query projection weights, row-major `[embed_dim, embed_dim]`.
    pub w_q: Vec<f32>,
    /// Key projection weights, row-major `[embed_dim, embed_dim]`.
    pub w_k: Vec<f32>,
    /// Value projection weights, row-major `[embed_dim, embed_dim]`.
    pub w_v: Vec<f32>,
    /// Output projection weights, row-major `[embed_dim, embed_dim]`.
    pub w_o: Vec<f32>,
    /// Layer configuration.
    pub cfg: SsaConfig,
    /// Per-head dimension `embed_dim / n_heads`.
    pub head_dim: usize,
}

/// Build the LIF config used by every spiking neuron in the attention block.
fn lif_cfg_from(cfg: &SsaConfig) -> LifConfig {
    LifConfig {
        tau_m: cfg.tau,
        v_th: cfg.threshold,
        v_rest: 0.0,
        dt: 1.0,
        reset: crate::neuron::lif::ResetMode::Hard,
    }
}

/// Validate the structural and numeric invariants of an [`SsaConfig`].
fn validate_cfg(cfg: &SsaConfig) -> SnnResult<()> {
    if cfg.embed_dim == 0 {
        return Err(SnnError::BadDim { got: cfg.embed_dim });
    }
    if cfg.n_heads == 0 {
        return Err(SnnError::BadDim { got: cfg.n_heads });
    }
    if cfg.n_timesteps == 0 {
        return Err(SnnError::BadTimesteps {
            got: cfg.n_timesteps,
        });
    }
    if !cfg.embed_dim.is_multiple_of(cfg.n_heads) {
        return Err(SnnError::OutOfRange {
            name: "embed_dim % n_heads".into(),
            val: (cfg.embed_dim % cfg.n_heads) as f32,
        });
    }
    if !cfg.threshold.is_finite() || cfg.threshold <= 0.0 {
        return Err(SnnError::BadThreshold {
            v_th: cfg.threshold,
        });
    }
    if !cfg.tau.is_finite() || cfg.tau <= 0.0 {
        return Err(SnnError::BadTau { tau: cfg.tau });
    }
    if !cfg.scale.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "scale".into(),
            val: cfg.scale,
        });
    }
    Ok(())
}

impl SpikingSelfAttention {
    /// Allocate a new attention layer with Kaiming-normal projection weights.
    ///
    /// Returns an error when `embed_dim` is not divisible by `n_heads`, or when
    /// any dimension / hyper-parameter is out of range.
    pub fn new(cfg: SsaConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        let head_dim = cfg.embed_dim / cfg.n_heads;
        let scale = (2.0_f32 / cfg.embed_dim as f32).sqrt();
        let make = |rng: &mut LcgRng| -> Vec<f32> {
            let mut w = vec![0.0_f32; cfg.embed_dim * cfg.embed_dim];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= scale;
            }
            w
        };
        let w_q = make(rng);
        let w_k = make(rng);
        let w_v = make(rng);
        let w_o = make(rng);
        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            cfg,
            head_dim,
        })
    }

    /// Project one timestep slice `[seq_len, embed_dim]` through `w` then a fresh
    /// LIF neuron, returning the binary spike output `[seq_len, embed_dim]`.
    fn project_spikes(&self, w: &[f32], x_t: &[f32], seq_len: usize) -> SnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let lif = lif_cfg_from(&self.cfg);
        let mut state = LifState::new(d);
        let mut out = vec![0.0_f32; seq_len * d];
        let mut current = vec![0.0_f32; d];
        let mut spikes = vec![0.0_f32; d];
        for s in 0..seq_len {
            let in_off = s * d;
            let x_row = x_t.get(in_off..in_off + d).ok_or(SnnError::Internal {
                msg: "projection input slice out of range".into(),
            })?;
            for (i, c_i) in current.iter_mut().enumerate() {
                let row_off = i * d;
                let w_row = w.get(row_off..row_off + d).ok_or(SnnError::Internal {
                    msg: "projection weight row out of range".into(),
                })?;
                let mut acc = 0.0_f32;
                for (wij, &xj) in w_row.iter().zip(x_row.iter()) {
                    acc += wij * xj;
                }
                *c_i = acc;
            }
            lif_step(&mut state, &current, &lif, &mut spikes)?;
            let out_row = out.get_mut(in_off..in_off + d).ok_or(SnnError::Internal {
                msg: "projection output slice out of range".into(),
            })?;
            out_row.copy_from_slice(&spikes);
        }
        Ok(out)
    }

    /// Compute the attention output for one timestep slice.
    ///
    /// `q_s`, `k_s`, `v_s` are `[seq_len, embed_dim]` binary spike maps. Returns
    /// the concatenated per-head context `[seq_len, embed_dim]` (still binary,
    /// since it is the output of the attention spiking neuron multiplied by the
    /// binary value spikes — see below) prior to the output projection.
    fn attend_step(
        &self,
        q_s: &[f32],
        k_s: &[f32],
        v_s: &[f32],
        seq_len: usize,
    ) -> SnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let h = self.cfg.n_heads;
        let hd = self.head_dim;
        let lif = lif_cfg_from(&self.cfg);
        let mut context = vec![0.0_f32; seq_len * d];
        // Per-head attention.
        for head in 0..h {
            let ch_off = head * hd;
            // Score map S[i][j] = scale * (Q_s[i] · K_s[j]) over this head's channels.
            let mut scores = vec![0.0_f32; seq_len * seq_len];
            for i in 0..seq_len {
                let qi_off = i * d + ch_off;
                let q_row = q_s.get(qi_off..qi_off + hd).ok_or(SnnError::Internal {
                    msg: "query head slice out of range".into(),
                })?;
                for j in 0..seq_len {
                    let kj_off = j * d + ch_off;
                    let k_row = k_s.get(kj_off..kj_off + hd).ok_or(SnnError::Internal {
                        msg: "key head slice out of range".into(),
                    })?;
                    let mut dot = 0.0_f32;
                    for (&qa, &ka) in q_row.iter().zip(k_row.iter()) {
                        dot += qa * ka;
                    }
                    let s_idx = i * seq_len + j;
                    if let Some(slot) = scores.get_mut(s_idx) {
                        *slot = dot * self.cfg.scale;
                    }
                }
            }
            // Attention neuron SN_A: binarise the scaled score map row-by-row.
            // A fresh membrane state per row keeps rows independent.
            let mut attn = vec![0.0_f32; seq_len * seq_len];
            for i in 0..seq_len {
                let mut state = LifState::new(seq_len);
                let row_off = i * seq_len;
                let score_row =
                    scores
                        .get(row_off..row_off + seq_len)
                        .ok_or(SnnError::Internal {
                            msg: "score row out of range".into(),
                        })?;
                let attn_row =
                    attn.get_mut(row_off..row_off + seq_len)
                        .ok_or(SnnError::Internal {
                            msg: "attn row out of range".into(),
                        })?;
                lif_step(&mut state, score_row, &lif, attn_row)?;
            }
            // Context = A · V_s over this head's channels.
            for i in 0..seq_len {
                let attn_off = i * seq_len;
                let attn_row =
                    attn.get(attn_off..attn_off + seq_len)
                        .ok_or(SnnError::Internal {
                            msg: "attn row read out of range".into(),
                        })?;
                let ctx_off = i * d + ch_off;
                let ctx_row = context
                    .get_mut(ctx_off..ctx_off + hd)
                    .ok_or(SnnError::Internal {
                        msg: "context head slice out of range".into(),
                    })?;
                for (j, &a_ij) in attn_row.iter().enumerate() {
                    if a_ij == 0.0 {
                        continue;
                    }
                    let vj_off = j * d + ch_off;
                    let v_row = v_s.get(vj_off..vj_off + hd).ok_or(SnnError::Internal {
                        msg: "value head slice out of range".into(),
                    })?;
                    for (c, &vc) in ctx_row.iter_mut().zip(v_row.iter()) {
                        *c += a_ij * vc;
                    }
                }
            }
        }
        Ok(context)
    }

    /// Apply the output projection `W_O` to the per-timestep context, emitting a
    /// final binary spike train via a fresh LIF neuron.
    fn output_project(&self, context_t: &[f32], seq_len: usize) -> SnnResult<Vec<f32>> {
        self.project_spikes(&self.w_o, context_t, seq_len)
    }

    /// Run the spike-driven self-attention over a full spike train.
    ///
    /// `input` is a flat row-major `[n_timesteps][seq_len][embed_dim]` train of
    /// `0/1` spikes; its length must be a positive multiple of
    /// `n_timesteps * embed_dim`. Returns a spike train of identical shape.
    pub fn forward(&self, input: &[f32]) -> SnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let t = self.cfg.n_timesteps;
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let per_t = input.len() / t;
        if per_t * t != input.len() || per_t == 0 {
            return Err(SnnError::BadShape {
                expected: t,
                got: input.len(),
            });
        }
        if !per_t.is_multiple_of(d) {
            return Err(SnnError::BadShape {
                expected: d,
                got: per_t,
            });
        }
        let seq_len = per_t / d;
        let mut out = vec![0.0_f32; input.len()];
        for ti in 0..t {
            let t_off = ti * per_t;
            let x_t = input.get(t_off..t_off + per_t).ok_or(SnnError::Internal {
                msg: "timestep input slice out of range".into(),
            })?;
            let q_s = self.project_spikes(&self.w_q, x_t, seq_len)?;
            let k_s = self.project_spikes(&self.w_k, x_t, seq_len)?;
            let v_s = self.project_spikes(&self.w_v, x_t, seq_len)?;
            let context = self.attend_step(&q_s, &k_s, &v_s, seq_len)?;
            let y_t = self.output_project(&context, seq_len)?;
            let out_t = out
                .get_mut(t_off..t_off + per_t)
                .ok_or(SnnError::Internal {
                    msg: "timestep output slice out of range".into(),
                })?;
            out_t.copy_from_slice(&y_t);
        }
        Ok(out)
    }

    /// Mean firing rate of a spike train in `[0, 1]` (fraction of `1.0` entries).
    #[must_use]
    pub fn firing_rate(spikes: &[f32]) -> f32 {
        if spikes.is_empty() {
            return 0.0;
        }
        let fired = spikes.iter().filter(|&&s| s != 0.0).count();
        fired as f32 / spikes.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> SsaConfig {
        SsaConfig {
            embed_dim: 8,
            n_heads: 2,
            n_timesteps: 3,
            threshold: 0.5,
            tau: 2.0,
            scale: 0.5,
        }
    }

    fn random_spike_train(cfg: &SsaConfig, seq_len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let len = cfg.n_timesteps * seq_len * cfg.embed_dim;
        let mut v = vec![0.0_f32; len];
        for x in &mut v {
            *x = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
        }
        v
    }

    #[test]
    fn output_shape_equals_input_shape() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(1);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 5;
        let input = random_spike_train(&cfg, seq_len, 7);
        let out = attn.forward(&input).expect("forward");
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn outputs_are_binary_spikes() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(2);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 4;
        let input = random_spike_train(&cfg, seq_len, 11);
        let out = attn.forward(&input).expect("forward");
        for &s in &out {
            assert!(s == 0.0 || s == 1.0, "non-binary spike: {s}");
        }
    }

    #[test]
    fn embed_dim_not_divisible_by_heads_errors() {
        let cfg = SsaConfig {
            embed_dim: 10,
            n_heads: 3,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(3);
        let err = SpikingSelfAttention::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn zero_spike_input_gives_zero_output() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(4);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 6;
        let len = cfg.n_timesteps * seq_len * cfg.embed_dim;
        let input = vec![0.0_f32; len];
        let out = attn.forward(&input).expect("forward");
        // No input current anywhere -> no projection spikes -> zero scores ->
        // (threshold > 0) no attention spikes -> zero context -> zero output.
        assert!(out.iter().all(|&s| s == 0.0), "expected all-zero output");
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = small_cfg();
        let seq_len = 5;
        let input = random_spike_train(&cfg, seq_len, 99);
        let mut rng_a = LcgRng::new(123);
        let attn_a = SpikingSelfAttention::new(cfg, &mut rng_a).expect("ctor");
        let mut rng_b = LcgRng::new(123);
        let attn_b = SpikingSelfAttention::new(cfg, &mut rng_b).expect("ctor");
        let out_a = attn_a.forward(&input).expect("a");
        let out_b = attn_b.forward(&input).expect("b");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn repeated_forward_is_idempotent_stateless() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(5);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 4;
        let input = random_spike_train(&cfg, seq_len, 31);
        let first = attn.forward(&input).expect("first");
        let second = attn.forward(&input).expect("second");
        assert_eq!(first, second, "layer must be stateless across calls");
    }

    #[test]
    fn single_vs_multi_head_can_differ() {
        let seq_len = 5;
        let cfg1 = SsaConfig {
            n_heads: 1,
            ..small_cfg()
        };
        let cfg2 = SsaConfig {
            n_heads: 4,
            ..small_cfg()
        };
        // Same weights for both (same seed) so only the head split differs.
        let mut rng1 = LcgRng::new(77);
        let a1 = SpikingSelfAttention::new(cfg1, &mut rng1).expect("a1");
        let mut rng2 = LcgRng::new(77);
        let a2 = SpikingSelfAttention::new(cfg2, &mut rng2).expect("a2");
        let input = random_spike_train(&cfg1, seq_len, 17);
        let o1 = a1.forward(&input).expect("o1");
        let o2 = a2.forward(&input).expect("o2");
        assert_eq!(o1.len(), o2.len());
        // The per-head channel split changes the score maps, so the binarised
        // attention -- and hence the output -- differs for a non-trivial input.
        assert_ne!(o1, o2, "n_heads=1 vs n_heads=4 should differ");
    }

    #[test]
    fn firing_rate_in_unit_interval() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(6);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 7;
        let input = random_spike_train(&cfg, seq_len, 53);
        let out = attn.forward(&input).expect("forward");
        let r = SpikingSelfAttention::firing_rate(&out);
        assert!((0.0..=1.0).contains(&r), "firing rate out of range: {r}");
    }

    #[test]
    fn changing_scale_changes_activation() {
        let seq_len = 5;
        let lo = SsaConfig {
            scale: 0.01,
            ..small_cfg()
        };
        let hi = SsaConfig {
            scale: 4.0,
            ..small_cfg()
        };
        let mut rng_lo = LcgRng::new(202);
        let a_lo = SpikingSelfAttention::new(lo, &mut rng_lo).expect("lo");
        let mut rng_hi = LcgRng::new(202);
        let a_hi = SpikingSelfAttention::new(hi, &mut rng_hi).expect("hi");
        let input = random_spike_train(&lo, seq_len, 41);
        let o_lo = a_lo.forward(&input).expect("o_lo");
        let o_hi = a_hi.forward(&input).expect("o_hi");
        // A larger pre-SN scale pushes more score entries above threshold, which
        // changes the binarised attention map and therefore the output train.
        assert_ne!(o_lo, o_hi, "scale should influence activation");
    }

    #[test]
    fn non_negativity_preserved() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(7);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 6;
        let input = random_spike_train(&cfg, seq_len, 23);
        let out = attn.forward(&input).expect("forward");
        for &s in &out {
            assert!(s >= 0.0, "negative spike value: {s}");
        }
    }

    #[test]
    fn empty_input_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(8);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let err = attn.forward(&[]);
        assert!(matches!(err, Err(SnnError::EmptyInput)));
    }

    #[test]
    fn non_positive_threshold_errors() {
        let cfg = SsaConfig {
            threshold: 0.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(9);
        let err = SpikingSelfAttention::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadThreshold { .. })));
    }

    #[test]
    fn zero_timesteps_errors() {
        let cfg = SsaConfig {
            n_timesteps: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(10);
        let err = SpikingSelfAttention::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadTimesteps { .. })));
    }

    #[test]
    fn seq_embed_length_mismatch_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(11);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        // Length is a multiple of n_timesteps but the per-timestep slice is not
        // a multiple of embed_dim -> BadShape.
        let bad_len = cfg.n_timesteps * (cfg.embed_dim + 1);
        let input = vec![0.0_f32; bad_len];
        let err = attn.forward(&input);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn length_not_multiple_of_timesteps_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(12);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        // n_timesteps = 3; pick a length not divisible by 3.
        let input = vec![1.0_f32; cfg.embed_dim * 3 + 1];
        let err = attn.forward(&input);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn zero_dim_config_errors() {
        let cfg = SsaConfig {
            embed_dim: 0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(13);
        let err = SpikingSelfAttention::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadDim { .. })));
    }

    #[test]
    fn bad_tau_errors() {
        let cfg = SsaConfig {
            tau: -1.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(14);
        let err = SpikingSelfAttention::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn high_drive_produces_some_spikes() {
        // All-ones input with a generous scale should fire at least one spike.
        let cfg = SsaConfig {
            threshold: 0.5,
            scale: 2.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(15);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        let seq_len = 5;
        let len = cfg.n_timesteps * seq_len * cfg.embed_dim;
        let input = vec![1.0_f32; len];
        let out = attn.forward(&input).expect("forward");
        let any = out.contains(&1.0);
        assert!(any, "expected at least one output spike under strong drive");
    }

    #[test]
    fn head_dim_is_embed_over_heads() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(16);
        let attn = SpikingSelfAttention::new(cfg, &mut rng).expect("ctor");
        assert_eq!(attn.head_dim, cfg.embed_dim / cfg.n_heads);
    }
}
