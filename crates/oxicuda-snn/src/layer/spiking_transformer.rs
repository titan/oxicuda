#![allow(clippy::needless_range_loop)]
//! Spikformer encoder block (Zhou et al. 2023, ICLR "Spikformer: When Spiking
//! Neural Network Meets Transformer").
//!
//! A Spikformer encoder block is the spiking analogue of a standard Transformer
//! encoder layer. It chains two residual sublayers, each fully spike-driven:
//!
//! ```text
//! u = SEW( SSA(x), x )                       // spike-driven self-attention sublayer
//! y = SEW( SMLP(u), u )                      // spiking MLP / feed-forward sublayer
//! ```
//!
//! where `SSA` is the Spikformer Spiking Self-Attention (see
//! [`crate::layer::spiking_attention::SpikingSelfAttention`]) and `SMLP` is a
//! two-layer spiking feed-forward network
//!
//! ```text
//! h  = SN_1( W_1 · x )      with  W_1 : embed_dim → mlp_dim
//! z  = SN_2( W_2 · h )      with  W_2 : mlp_dim → embed_dim
//! ```
//!
//! Both `SN_1`, `SN_2` are LIF spiking neurons, so every intermediate quantity is
//! a `0/1` spike train.
//!
//! Residual (SEW shortcut). Spikformer / SEW-ResNet (Fang et al. 2021, "Deep
//! Residual Learning in Spiking Neural Networks") add the identity short-cut at
//! the *spike* level rather than at the membrane / pre-activation level. The
//! element-wise function used here is the membrane-additive `OR` connect:
//!
//! ```text
//! SEW(s, x)[k] = 1 if s[k] + x[k] >= 1 else 0          (== s[k] OR x[k])
//! ```
//!
//! This is the additive (`s + x`) shortcut followed by the spike unit-step, which
//! is the SEW residual that keeps the block output a binary spike train (so the
//! block is composable: its output can feed straight into the next Spikformer
//! block as another `[T][L][D]` spike train). The identity term guarantees that
//! input spikes are carried through even when a sublayer is silent, the property
//! that lets SEW residuals train very deep SNNs.
//!
//! Layout. Input and output are spike trains shaped `[n_timesteps][seq_len][embed_dim]`,
//! stored as one flat row-major `Vec<f32>` of length
//! `n_timesteps * seq_len * embed_dim` whose entries are `0.0` or `1.0`. This is
//! exactly the layout consumed and produced by
//! [`crate::layer::spiking_attention::SpikingSelfAttention::forward`].

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::layer::spiking_attention::{SpikingSelfAttention, SsaConfig};
use crate::neuron::lif::{LifConfig, LifState, ResetMode, lif_step};

/// Configuration for a [`SpikformerBlock`].
#[derive(Debug, Clone, Copy)]
pub struct SpikformerBlockConfig {
    /// Model / embedding dimension `D`. Must be divisible by `n_heads`.
    pub embed_dim: usize,
    /// Number of self-attention heads.
    pub n_heads: usize,
    /// Feed-forward expansion ratio; the MLP hidden width is
    /// `mlp_dim = embed_dim * mlp_ratio` (rounded, at least `1`).
    pub mlp_ratio: usize,
    /// Number of discrete timesteps `T` in each spike train.
    pub n_timesteps: usize,
    /// Sequence length `L` (number of tokens) the block expects.
    pub seq_len: usize,
    /// Spike threshold shared by every LIF neuron in the block.
    pub threshold: f32,
    /// Membrane time constant `τ_m` for every LIF neuron in the block.
    pub tau: f32,
    /// Scale applied to the attention score map before the attention neuron.
    pub scale: f32,
}

impl Default for SpikformerBlockConfig {
    fn default() -> Self {
        Self {
            embed_dim: 64,
            n_heads: 8,
            mlp_ratio: 4,
            n_timesteps: 4,
            seq_len: 16,
            threshold: 1.0,
            tau: 2.0,
            scale: 0.125,
        }
    }
}

impl SpikformerBlockConfig {
    /// Hidden width of the feed-forward sublayer, `embed_dim * mlp_ratio`.
    #[must_use]
    pub fn mlp_dim(&self) -> usize {
        (self.embed_dim * self.mlp_ratio).max(1)
    }
}

/// A Spikformer Transformer encoder block: spike-driven self-attention with a
/// SEW residual, followed by a spiking feed-forward network with its own SEW
/// residual.
///
/// The block is stateless across calls to [`SpikformerBlock::forward`]: each call
/// rebuilds the LIF membrane states of the feed-forward neurons, and the inner
/// [`SpikingSelfAttention`] is itself stateless, so the output depends only on
/// the input spike train and the (fixed) weights.
#[derive(Debug, Clone)]
pub struct SpikformerBlock {
    /// Spike-driven self-attention sublayer.
    pub attn: SpikingSelfAttention,
    /// First feed-forward weight matrix `W_1`, row-major `[mlp_dim, embed_dim]`.
    pub mlp_fc1: Vec<f32>,
    /// Second feed-forward weight matrix `W_2`, row-major `[embed_dim, mlp_dim]`.
    pub mlp_fc2: Vec<f32>,
    /// LIF configuration shared by the two feed-forward spiking neurons.
    pub mlp_lif: LifConfig,
    /// Block configuration.
    pub cfg: SpikformerBlockConfig,
    /// Hidden width of the feed-forward sublayer.
    pub mlp_dim: usize,
}

/// Build the LIF config used by the feed-forward neurons (Hard reset, `dt = 1`).
fn lif_cfg_from(cfg: &SpikformerBlockConfig) -> LifConfig {
    LifConfig {
        tau_m: cfg.tau,
        v_th: cfg.threshold,
        v_rest: 0.0,
        dt: 1.0,
        reset: ResetMode::Hard,
    }
}

/// Validate the structural and numeric invariants of a [`SpikformerBlockConfig`].
fn validate_cfg(cfg: &SpikformerBlockConfig) -> SnnResult<()> {
    if cfg.embed_dim == 0 {
        return Err(SnnError::BadDim { got: cfg.embed_dim });
    }
    if cfg.n_heads == 0 {
        return Err(SnnError::BadDim { got: cfg.n_heads });
    }
    if cfg.mlp_ratio == 0 {
        return Err(SnnError::BadDim { got: cfg.mlp_ratio });
    }
    if cfg.seq_len == 0 {
        return Err(SnnError::BadDim { got: cfg.seq_len });
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

/// Project one spike row `x` (length `in_dim`) through `w` (row-major
/// `[out_dim, in_dim]`) into a pre-allocated `current` buffer; the output
/// dimension is `current.len()`.
fn matvec(w: &[f32], x: &[f32], in_dim: usize, current: &mut [f32]) -> SnnResult<()> {
    for (i, c_i) in current.iter_mut().enumerate() {
        let row_off = i * in_dim;
        let w_row = w.get(row_off..row_off + in_dim).ok_or(SnnError::Internal {
            msg: "feed-forward weight row out of range".into(),
        })?;
        let mut acc = 0.0_f32;
        for (&wij, &xj) in w_row.iter().zip(x.iter()) {
            acc += wij * xj;
        }
        *c_i = acc;
    }
    Ok(())
}

/// Element-wise SEW residual `OR` connect: `out[k] = 1 if a[k] + b[k] >= 1`.
fn sew_residual(a: &[f32], b: &[f32]) -> SnnResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(SnnError::IncompatibleLength {
            a: a.len(),
            b: b.len(),
        });
    }
    let mut out = vec![0.0_f32; a.len()];
    for (o, (&ai, &bi)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = if ai + bi >= 1.0 { 1.0_f32 } else { 0.0_f32 };
    }
    Ok(out)
}

impl SpikformerBlock {
    /// Allocate a new Spikformer block with Kaiming-normal weights.
    ///
    /// The inner [`SpikingSelfAttention`] and the two feed-forward matrices are
    /// all initialised from `rng`. Returns an error when `embed_dim` is not
    /// divisible by `n_heads`, or when any dimension / hyper-parameter is out of
    /// range.
    pub fn new(cfg: SpikformerBlockConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        let mlp_dim = cfg.mlp_dim();
        let ssa_cfg = SsaConfig {
            embed_dim: cfg.embed_dim,
            n_heads: cfg.n_heads,
            n_timesteps: cfg.n_timesteps,
            threshold: cfg.threshold,
            tau: cfg.tau,
            scale: cfg.scale,
        };
        let attn = SpikingSelfAttention::new(ssa_cfg, rng)?;

        let scale1 = (2.0_f32 / cfg.embed_dim as f32).sqrt();
        let mut mlp_fc1 = vec![0.0_f32; mlp_dim * cfg.embed_dim];
        rng.fill_normal(&mut mlp_fc1);
        for v in &mut mlp_fc1 {
            *v *= scale1;
        }

        let scale2 = (2.0_f32 / mlp_dim as f32).sqrt();
        let mut mlp_fc2 = vec![0.0_f32; cfg.embed_dim * mlp_dim];
        rng.fill_normal(&mut mlp_fc2);
        for v in &mut mlp_fc2 {
            *v *= scale2;
        }

        Ok(Self {
            attn,
            mlp_fc1,
            mlp_fc2,
            mlp_lif: lif_cfg_from(&cfg),
            cfg,
            mlp_dim,
        })
    }

    /// Expected flat length of an input/output spike train.
    #[must_use]
    pub fn expected_len(&self) -> usize {
        self.cfg.n_timesteps * self.cfg.seq_len * self.cfg.embed_dim
    }

    /// Run the spiking feed-forward (MLP) sublayer over a full spike train.
    ///
    /// `x` is `[n_timesteps][seq_len][embed_dim]` flat row-major. The two
    /// projections use fresh LIF membrane states (one membrane per channel),
    /// integrated across the `n_timesteps` slices in order, exactly like the
    /// attention projections.
    fn mlp_forward(&self, x: &[f32]) -> SnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let m = self.mlp_dim;
        let l = self.cfg.seq_len;
        let t = self.cfg.n_timesteps;
        let per_t = l * d;

        let mut state1 = LifState::new(m);
        let mut state2 = LifState::new(d);
        let mut cur1 = vec![0.0_f32; m];
        let mut hid = vec![0.0_f32; m];
        let mut cur2 = vec![0.0_f32; d];
        let mut out = vec![0.0_f32; x.len()];

        for ti in 0..t {
            let t_off = ti * per_t;
            for s in 0..l {
                let in_off = t_off + s * d;
                let x_row = x.get(in_off..in_off + d).ok_or(SnnError::Internal {
                    msg: "mlp input row out of range".into(),
                })?;
                // h = SN_1(W_1 · x): in_dim = embed_dim, out_dim = mlp_dim.
                matvec(&self.mlp_fc1, x_row, d, &mut cur1)?;
                lif_step(&mut state1, &cur1, &self.mlp_lif, &mut hid)?;
                // z = SN_2(W_2 · h): in_dim = mlp_dim, out_dim = embed_dim.
                matvec(&self.mlp_fc2, &hid, m, &mut cur2)?;
                let out_row = out.get_mut(in_off..in_off + d).ok_or(SnnError::Internal {
                    msg: "mlp output row out of range".into(),
                })?;
                lif_step(&mut state2, &cur2, &self.mlp_lif, out_row)?;
            }
        }
        Ok(out)
    }

    /// Run the full Spikformer encoder block over a spike train.
    ///
    /// `spikes` is a flat row-major `[n_timesteps][seq_len][embed_dim]` train of
    /// `0/1` spikes; its length must equal [`Self::expected_len`]. Returns a
    /// binary spike train of identical shape.
    pub fn forward(&self, spikes: &[f32]) -> SnnResult<Vec<f32>> {
        if spikes.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let expected = self.expected_len();
        if spikes.len() != expected {
            return Err(SnnError::BadShape {
                expected,
                got: spikes.len(),
            });
        }
        // Self-attention sublayer with SEW residual: u = SEW(SSA(x), x).
        let attn_out = self.attn.forward(spikes)?;
        let u = sew_residual(&attn_out, spikes)?;
        // Feed-forward sublayer with SEW residual: y = SEW(SMLP(u), u).
        let mlp_out = self.mlp_forward(&u)?;
        let y = sew_residual(&mlp_out, &u)?;
        Ok(y)
    }

    /// Mean firing rate of a spike train in `[0, 1]`.
    #[must_use]
    pub fn firing_rate(spikes: &[f32]) -> f32 {
        SpikingSelfAttention::firing_rate(spikes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> SpikformerBlockConfig {
        SpikformerBlockConfig {
            embed_dim: 8,
            n_heads: 2,
            mlp_ratio: 2,
            n_timesteps: 3,
            seq_len: 5,
            threshold: 0.5,
            tau: 2.0,
            scale: 0.5,
        }
    }

    fn random_spike_train(cfg: &SpikformerBlockConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let len = cfg.n_timesteps * cfg.seq_len * cfg.embed_dim;
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
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let input = random_spike_train(&cfg, 7);
        let out = block.forward(&input).expect("forward");
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn outputs_are_binary() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(2);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let input = random_spike_train(&cfg, 11);
        let out = block.forward(&input).expect("forward");
        for &s in &out {
            assert!(s == 0.0 || s == 1.0, "non-binary spike: {s}");
        }
    }

    #[test]
    fn identity_skip_carries_input_spikes_through() {
        // Zero out every weight so both sublayers are silent (SSA fires nothing,
        // SMLP fires nothing). The SEW residual must then pass the input spikes
        // straight through: y == input. This is the residual short-cut in action.
        let cfg = small_cfg();
        let mut rng = LcgRng::new(3);
        let mut block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        for w in &mut block.attn.w_q {
            *w = 0.0;
        }
        for w in &mut block.attn.w_k {
            *w = 0.0;
        }
        for w in &mut block.attn.w_v {
            *w = 0.0;
        }
        for w in &mut block.attn.w_o {
            *w = 0.0;
        }
        for w in &mut block.mlp_fc1 {
            *w = 0.0;
        }
        for w in &mut block.mlp_fc2 {
            *w = 0.0;
        }
        let input = random_spike_train(&cfg, 21);
        let out = block.forward(&input).expect("forward");
        assert_eq!(out, input, "residual must carry input spikes through");
    }

    #[test]
    fn residual_changes_output_vs_no_skip_baseline() {
        // With non-trivial weights, the block output (which includes the residual
        // skips) differs from the bare attention-only output for a spiking input.
        let cfg = small_cfg();
        let mut rng = LcgRng::new(4);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let input = random_spike_train(&cfg, 31);
        let block_out = block.forward(&input).expect("forward");
        let attn_only = block.attn.forward(&input).expect("attn");
        assert_eq!(block_out.len(), attn_only.len());
        assert_ne!(
            block_out, attn_only,
            "residual + MLP sublayer should change the output"
        );
    }

    #[test]
    fn zero_input_with_weights_can_stay_zero_or_fire() {
        // Zero spike input: SSA produces no spikes (threshold>0), so u == 0; the
        // MLP also sees zero current => no spikes; SEW(0,0) == 0. Output all zero.
        let cfg = small_cfg();
        let mut rng = LcgRng::new(5);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let len = cfg.n_timesteps * cfg.seq_len * cfg.embed_dim;
        let input = vec![0.0_f32; len];
        let out = block.forward(&input).expect("forward");
        assert!(out.iter().all(|&s| s == 0.0), "expected all-zero output");
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = small_cfg();
        let input = random_spike_train(&cfg, 99);
        let mut rng_a = LcgRng::new(123);
        let a = SpikformerBlock::new(cfg, &mut rng_a).expect("a");
        let mut rng_b = LcgRng::new(123);
        let b = SpikformerBlock::new(cfg, &mut rng_b).expect("b");
        let out_a = a.forward(&input).expect("a fwd");
        let out_b = b.forward(&input).expect("b fwd");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn stateless_across_calls() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(6);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let input = random_spike_train(&cfg, 17);
        let first = block.forward(&input).expect("first");
        let second = block.forward(&input).expect("second");
        assert_eq!(first, second, "block must be stateless across calls");
    }

    #[test]
    fn high_drive_produces_some_spikes() {
        let cfg = SpikformerBlockConfig {
            threshold: 0.5,
            scale: 2.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(7);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let len = cfg.n_timesteps * cfg.seq_len * cfg.embed_dim;
        let input = vec![1.0_f32; len];
        let out = block.forward(&input).expect("forward");
        assert!(out.contains(&1.0), "expected at least one output spike");
    }

    #[test]
    fn mlp_dim_is_embed_times_ratio() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(8);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        assert_eq!(block.mlp_dim, cfg.embed_dim * cfg.mlp_ratio);
        assert_eq!(block.mlp_fc1.len(), block.mlp_dim * cfg.embed_dim);
        assert_eq!(block.mlp_fc2.len(), cfg.embed_dim * block.mlp_dim);
    }

    #[test]
    fn firing_rate_in_unit_interval() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(9);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let input = random_spike_train(&cfg, 53);
        let out = block.forward(&input).expect("forward");
        let r = SpikformerBlock::firing_rate(&out);
        assert!((0.0..=1.0).contains(&r), "firing rate out of range: {r}");
    }

    #[test]
    fn empty_input_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(10);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let err = block.forward(&[]);
        assert!(matches!(err, Err(SnnError::EmptyInput)));
    }

    #[test]
    fn wrong_length_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(11);
        let block = SpikformerBlock::new(cfg, &mut rng).expect("ctor");
        let bad = vec![0.0_f32; block.expected_len() + 1];
        let err = block.forward(&bad);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn embed_dim_not_divisible_by_heads_errors() {
        let cfg = SpikformerBlockConfig {
            embed_dim: 10,
            n_heads: 3,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(12);
        let err = SpikformerBlock::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn bad_threshold_errors() {
        let cfg = SpikformerBlockConfig {
            threshold: 0.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(13);
        let err = SpikformerBlock::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadThreshold { .. })));
    }

    #[test]
    fn bad_tau_errors() {
        let cfg = SpikformerBlockConfig {
            tau: -1.0,
            ..small_cfg()
        };
        let mut rng = LcgRng::new(14);
        let err = SpikformerBlock::new(cfg, &mut rng);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn zero_dims_error() {
        let mut rng = LcgRng::new(15);
        assert!(matches!(
            SpikformerBlock::new(
                SpikformerBlockConfig {
                    embed_dim: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            SpikformerBlock::new(
                SpikformerBlockConfig {
                    seq_len: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            SpikformerBlock::new(
                SpikformerBlockConfig {
                    mlp_ratio: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            SpikformerBlock::new(
                SpikformerBlockConfig {
                    n_timesteps: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn sew_residual_is_logical_or() {
        let a = vec![0.0, 0.0, 1.0, 1.0];
        let b = vec![0.0, 1.0, 0.0, 1.0];
        let out = sew_residual(&a, &b).expect("sew");
        assert_eq!(out, vec![0.0, 1.0, 1.0, 1.0]);
    }
}
