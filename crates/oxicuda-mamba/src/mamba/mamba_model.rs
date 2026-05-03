//! Full Mamba language model: token embedding, stacked Mamba blocks, final
//! RMSNorm, and an LM head for next-token prediction.
//!
//! # Architecture
//!
//! ```text
//! token_ids: [L]
//!   → embedding lookup: [L, D]
//!   → MambaBlock_0 → MambaBlock_1 → ... → MambaBlock_{n_layers-1}   each: [L, D]
//!   → final RMSNorm [L, D]
//!   → lm_head [D → vocab_size]: logits [L, vocab_size]
//! ```

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::mamba::mamba_block::{MambaBlock, MambaBlockConfig, MambaBlockWeights, rms_norm};

// ─── MambaConfig ─────────────────────────────────────────────────────────────

/// Configuration for a full Mamba language model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MambaConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model (embedding / residual stream) dimension `D`.
    pub d_model: usize,
    /// Number of stacked Mamba blocks.
    pub n_layers: usize,
    /// SSM state size `N` per channel (shared across all layers).
    pub d_state: usize,
    /// Depthwise conv kernel size (shared across all layers).
    pub d_conv: usize,
}

impl MambaConfig {
    /// Tiny config for unit tests: `vocab=256, D=32, 2 layers, N=4, d_conv=4`.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_model: 32,
            n_layers: 2,
            d_state: 4,
            d_conv: 4,
        }
    }

    /// Create a new config, validating all fields.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidVocabSize`]  — if `vocab_size == 0`
    /// - [`MambaError::InvalidModelDim`]   — if `d_model == 0`
    /// - [`MambaError::InvalidLayerCount`] — if `n_layers == 0`
    /// - [`MambaError::InvalidSsmOrder`]   — if `d_state == 0`
    /// - [`MambaError::Internal`]          — if `d_conv == 0`
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        d_state: usize,
        d_conv: usize,
    ) -> MambaResult<Self> {
        if vocab_size == 0 {
            return Err(MambaError::InvalidVocabSize(vocab_size));
        }
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if n_layers == 0 {
            return Err(MambaError::InvalidLayerCount(n_layers));
        }
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(d_state));
        }
        if d_conv == 0 {
            return Err(MambaError::Internal("d_conv must be > 0".into()));
        }
        Ok(Self {
            vocab_size,
            d_model,
            n_layers,
            d_state,
            d_conv,
        })
    }

    /// Compute the `MambaBlockConfig` for a single block in this model.
    fn block_config(&self) -> MambaResult<MambaBlockConfig> {
        MambaBlockConfig::new(self.d_model)?
            .with_d_state(self.d_state)?
            .with_d_conv(self.d_conv)
    }
}

// ─── MambaModelWeights ───────────────────────────────────────────────────────

/// All learnable weights for a full Mamba language model.
pub struct MambaModelWeights {
    /// Token embedding matrix: `[vocab_size, d_model]`.
    pub embedding: Vec<f32>,
    /// Per-layer block weights: `n_layers` entries.
    pub layers: Vec<MambaBlockWeights>,
    /// Final RMSNorm weight: `[d_model]`.
    pub norm_f: Vec<f32>,
    /// LM head: `[vocab_size, d_model]` (not weight-tied to embedding).
    pub lm_head: Vec<f32>,
}

impl MambaModelWeights {
    /// Allocate all weight tensors and zero-initialize them.
    pub fn zeros(config: &MambaConfig) -> Self {
        let block_cfg = config
            .block_config()
            .expect("MambaConfig must be valid to construct weights");
        Self {
            embedding: vec![0.0_f32; config.vocab_size * config.d_model],
            layers: (0..config.n_layers)
                .map(|_| MambaBlockWeights::zeros(&block_cfg))
                .collect(),
            norm_f: vec![0.0_f32; config.d_model],
            lm_head: vec![0.0_f32; config.vocab_size * config.d_model],
        }
    }

    /// Initialize with small random weights from a normal distribution.
    ///
    /// Each block uses `MambaBlockWeights::random`, which sets `a_log` and
    /// `norm_weight` to their paper defaults. The final `norm_f` is set to 1.0
    /// and the embedding / lm_head are drawn from N(0, 1) scaled by 0.02.
    pub fn random(config: &MambaConfig, rng: &mut LcgRng) -> Self {
        let block_cfg = config
            .block_config()
            .expect("MambaConfig must be valid to construct weights");
        let emb_size = config.vocab_size * config.d_model;
        let mut embedding = vec![0.0_f32; emb_size];
        rng.fill_normal(&mut embedding);
        // Scale down to avoid early saturation
        for v in &mut embedding {
            *v *= 0.02;
        }
        let layers: Vec<MambaBlockWeights> = (0..config.n_layers)
            .map(|_| MambaBlockWeights::random(&block_cfg, rng))
            .collect();
        let norm_f = vec![1.0_f32; config.d_model];
        let mut lm_head = vec![0.0_f32; config.vocab_size * config.d_model];
        rng.fill_normal(&mut lm_head);
        for v in &mut lm_head {
            *v *= 0.02;
        }
        Self {
            embedding,
            layers,
            norm_f,
            lm_head,
        }
    }
}

// ─── MambaModel ──────────────────────────────────────────────────────────────

/// Full Mamba language model.
///
/// Supports forward pass over a token sequence and greedy next-token prediction.
pub struct MambaModel {
    config: MambaConfig,
    blocks: Vec<MambaBlock>,
    norm_f: Vec<f32>,
    embedding: Vec<f32>,
    lm_head: Vec<f32>,
}

impl MambaModel {
    /// Construct a new `MambaModel`, consuming the config and weights.
    ///
    /// # Errors
    ///
    /// - [`MambaError::WeightShapeMismatch`] — if any weight tensor has wrong shape.
    /// - Propagated errors from `MambaBlock::new`.
    pub fn new(config: MambaConfig, weights: MambaModelWeights) -> MambaResult<Self> {
        // Validate top-level weight shapes
        let expected_emb = config.vocab_size * config.d_model;
        if weights.embedding.len() != expected_emb {
            return Err(MambaError::WeightShapeMismatch {
                name: "embedding",
                expected: vec![config.vocab_size, config.d_model],
                got: vec![weights.embedding.len()],
            });
        }
        if weights.norm_f.len() != config.d_model {
            return Err(MambaError::WeightShapeMismatch {
                name: "norm_f",
                expected: vec![config.d_model],
                got: vec![weights.norm_f.len()],
            });
        }
        let expected_head = config.vocab_size * config.d_model;
        if weights.lm_head.len() != expected_head {
            return Err(MambaError::WeightShapeMismatch {
                name: "lm_head",
                expected: vec![config.vocab_size, config.d_model],
                got: vec![weights.lm_head.len()],
            });
        }
        if weights.layers.len() != config.n_layers {
            return Err(MambaError::DimensionMismatch {
                expected: config.n_layers,
                got: weights.layers.len(),
            });
        }

        let block_cfg = config.block_config()?;
        let blocks: Vec<MambaBlock> = weights
            .layers
            .into_iter()
            .map(|layer_w| MambaBlock::new(block_cfg.clone(), layer_w))
            .collect::<MambaResult<Vec<_>>>()?;

        Ok(Self {
            config,
            blocks,
            norm_f: weights.norm_f,
            embedding: weights.embedding,
            lm_head: weights.lm_head,
        })
    }

    /// Forward pass over a token sequence.
    ///
    /// # Arguments
    ///
    /// * `token_ids` — Slice of `L` token indices. Each must be `< vocab_size`.
    ///
    /// # Returns
    ///
    /// Logits flat `[L * vocab_size]` (row-major `[L, vocab_size]`).
    ///
    /// # Errors
    ///
    /// - [`MambaError::EmptyInput`] — if `token_ids` is empty.
    /// - [`MambaError::TokenOutOfVocab`] — if any token index >= `vocab_size`.
    /// - Propagated errors from MambaBlock and helper functions.
    pub fn forward(&self, token_ids: &[usize]) -> MambaResult<Vec<f32>> {
        let cfg = &self.config;
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(MambaError::EmptyInput("token_ids"));
        }

        // ── Validate token ids ────────────────────────────────────────────────
        for &id in token_ids {
            if id >= cfg.vocab_size {
                return Err(MambaError::TokenOutOfVocab {
                    id,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        // ── Embedding lookup: [L, D] ──────────────────────────────────────────
        let d = cfg.d_model;
        let mut hidden = vec![0.0_f32; seq_len * d];
        for (t, &id) in token_ids.iter().enumerate() {
            let emb_start = id * d;
            let emb_end = emb_start + d;
            hidden[t * d..(t + 1) * d].copy_from_slice(&self.embedding[emb_start..emb_end]);
        }

        // ── Stacked Mamba blocks ──────────────────────────────────────────────
        for block in &self.blocks {
            hidden = block.forward(&hidden, seq_len)?;
        }

        // ── Final RMSNorm ─────────────────────────────────────────────────────
        hidden = rms_norm(&hidden, &self.norm_f, seq_len, d, 1e-5)?;

        // ── LM head: [L, D] → [L, vocab_size] ────────────────────────────────
        // lm_head: [vocab_size, d_model] — same layout as linear weight W
        let logits_size = seq_len * cfg.vocab_size;
        let mut logits = vec![0.0_f32; logits_size];
        for t in 0..seq_len {
            let h_row = &hidden[t * d..(t + 1) * d];
            for v in 0..cfg.vocab_size {
                let w_row = &self.lm_head[v * d..(v + 1) * d];
                let mut acc = 0.0_f32;
                for k in 0..d {
                    acc += h_row[k] * w_row[k];
                }
                logits[t * cfg.vocab_size + v] = acc;
            }
        }

        Ok(logits)
    }

    /// Greedy next-token: run forward on `context`, return the token with the
    /// highest logit at the last position.
    ///
    /// # Errors
    ///
    /// - Propagated errors from `forward`.
    pub fn next_token(&self, context: &[usize]) -> MambaResult<usize> {
        let logits = self.forward(context)?;
        let vocab = self.config.vocab_size;
        let seq_len = context.len();
        // Logits at the last token position
        let last_start = (seq_len - 1) * vocab;
        let last_logits = &logits[last_start..last_start + vocab];
        // Argmax (ties broken by lowest index)
        let best = last_logits
            .iter()
            .enumerate()
            .fold(
                0_usize,
                |best, (i, &v)| if v > last_logits[best] { i } else { best },
            );
        Ok(best)
    }

    /// Return a reference to the model configuration.
    #[inline]
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MambaConfig ───────────────────────────────────────────────────────────

    #[test]
    fn mamba_config_tiny() {
        let cfg = MambaConfig::tiny();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.d_model, 32);
        assert_eq!(cfg.n_layers, 2);
    }

    #[test]
    fn mamba_config_invalid_vocab() {
        let err = MambaConfig::new(0, 32, 2, 4, 4).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidVocabSize(0)));
    }

    #[test]
    fn mamba_config_invalid_layers() {
        let err = MambaConfig::new(256, 32, 0, 4, 4).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidLayerCount(0)));
    }

    // ── MambaModelWeights ─────────────────────────────────────────────────────

    #[test]
    fn model_weights_zeros_shapes() {
        let cfg = MambaConfig::tiny();
        let w = MambaModelWeights::zeros(&cfg);
        assert_eq!(
            w.embedding.len(),
            cfg.vocab_size * cfg.d_model,
            "embedding shape"
        );
        assert_eq!(w.layers.len(), cfg.n_layers, "layers count");
        assert_eq!(
            w.lm_head.len(),
            cfg.vocab_size * cfg.d_model,
            "lm_head shape"
        );
    }

    // ── MambaModel forward ────────────────────────────────────────────────────

    #[test]
    fn model_forward_shape() {
        let cfg = MambaConfig::tiny();
        let weights = MambaModelWeights::zeros(&cfg);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let token_ids = [0_usize, 1, 2];
        let logits = model.forward(&token_ids).expect("forward");
        assert_eq!(
            logits.len(),
            3 * cfg.vocab_size,
            "logits should have L * vocab_size elements"
        );
    }

    #[test]
    fn model_forward_finite() {
        let cfg = MambaConfig::tiny();
        let mut rng = LcgRng::new(42);
        let weights = MambaModelWeights::random(&cfg, &mut rng);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let token_ids: Vec<usize> = (0..5).map(|i| i % cfg.vocab_size).collect();
        let logits = model.forward(&token_ids).expect("forward");
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}]={v} is not finite");
        }
    }

    #[test]
    fn model_next_token_in_vocab() {
        let cfg = MambaConfig::tiny();
        let mut rng = LcgRng::new(99);
        let weights = MambaModelWeights::random(&cfg, &mut rng);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let context = [1_usize, 2, 3];
        let tok = model.next_token(&context).expect("next_token");
        assert!(
            tok < cfg.vocab_size,
            "next_token={tok} should be < vocab_size={}",
            cfg.vocab_size
        );
    }

    #[test]
    fn model_next_token_deterministic() {
        let cfg = MambaConfig::tiny();
        let mut rng = LcgRng::new(77);
        let weights = MambaModelWeights::random(&cfg, &mut rng);
        let model = MambaModel::new(cfg, weights).expect("valid model");
        let context = [5_usize, 10, 15];
        let tok1 = model.next_token(&context).expect("next_token first");
        let tok2 = model.next_token(&context).expect("next_token second");
        assert_eq!(tok1, tok2, "next_token must be deterministic");
    }

    #[test]
    fn model_greedy_decode_5_steps() {
        let cfg = MambaConfig::tiny();
        let mut rng = LcgRng::new(55);
        let weights = MambaModelWeights::random(&cfg, &mut rng);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let mut context = vec![0_usize];
        for _ in 0..5 {
            let tok = model.next_token(&context).expect("next_token");
            assert!(tok < cfg.vocab_size, "decoded token {tok} out of vocab");
            context.push(tok);
        }
        assert_eq!(
            context.len(),
            6,
            "should have 1 prompt + 5 generated tokens"
        );
    }

    #[test]
    fn model_zero_weights_logits_zero() {
        // With all weights zero (embedding=0, lm_head=0), all logits should be 0.
        let cfg = MambaConfig::tiny();
        let weights = MambaModelWeights::zeros(&cfg);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let token_ids = [0_usize, 1];
        let logits = model.forward(&token_ids).expect("forward");
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "logits[{i}]={v} should be zero for zero weights"
            );
        }
    }

    #[test]
    fn model_config_accessors() {
        let cfg = MambaConfig::tiny();
        let weights = MambaModelWeights::zeros(&cfg);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        assert_eq!(model.config().vocab_size, cfg.vocab_size);
        assert_eq!(model.config().d_model, cfg.d_model);
        assert_eq!(model.config().n_layers, cfg.n_layers);
    }

    #[test]
    fn model_large_context() {
        let cfg = MambaConfig::tiny();
        let mut rng = LcgRng::new(13);
        let weights = MambaModelWeights::random(&cfg, &mut rng);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let token_ids: Vec<usize> = (0..32).map(|i| i % cfg.vocab_size).collect();
        let logits = model.forward(&token_ids).expect("forward on large context");
        assert_eq!(logits.len(), 32 * cfg.vocab_size);
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "logits[{i}]={v} not finite for large context"
            );
        }
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn model_forward_empty_input() {
        let cfg = MambaConfig::tiny();
        let weights = MambaModelWeights::zeros(&cfg);
        let model = MambaModel::new(cfg, weights).expect("valid model");
        let err = model.forward(&[]).expect_err("should fail on empty input");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    #[test]
    fn model_forward_token_out_of_vocab() {
        let cfg = MambaConfig::tiny();
        let weights = MambaModelWeights::zeros(&cfg);
        let model = MambaModel::new(cfg.clone(), weights).expect("valid model");
        let token_ids = [cfg.vocab_size]; // exactly one past the end
        let err = model.forward(&token_ids).expect_err("should fail");
        assert!(matches!(err, MambaError::TokenOutOfVocab { .. }));
    }
}
