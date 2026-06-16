//! BLIP-2 Q-Former — querying transformer that extracts a fixed-length
//! representation from variable-length image features.
//!
//! Reference: Li et al. 2023, "BLIP-2: Bootstrapping Language-Image
//! Pre-training with Frozen Image Encoders and Large Language Models".
//!
//! A fixed set of learnable *query tokens* (`n_query × d_model`) is repeatedly
//! refined through `n_layers` blocks. Each block interleaves:
//!   (a) self-attention among the query tokens,
//!   (b) cross-attention where the queries attend to the (frozen) image
//!       features (queries = Q, image features = K/V),
//!   (c) a position-wise feed-forward network.
//!
//! Every sublayer follows the pre-norm residual layout
//! `x = x + sublayer(LayerNorm(x))`. Because the output is always the
//! transformed query set, the produced representation has a fixed length of
//! `n_query × d_model` regardless of how many image tokens are supplied — the
//! defining property of the Q-Former.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the BLIP-2 Q-Former.
#[derive(Debug, Clone)]
pub struct QFormerConfig {
    /// Number of learnable query tokens (fixed output length).
    pub n_query: usize,
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Number of interleaved self-attn + cross-attn + FFN blocks.
    pub n_layers: usize,
    /// Hidden dimension of the feed-forward network.
    pub ffn_dim: usize,
}

impl QFormerConfig {
    /// Tiny preset for testing: `n_query=4`, `d_model=8`, `n_heads=2`,
    /// `n_layers=2`, `ffn_dim=16`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_query: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_dim: 16,
        }
    }

    /// Validate the configuration, returning the appropriate error variant.
    fn validate(&self) -> MmResult<()> {
        if self.n_query == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if self.d_model == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        if self.ffn_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Per-layer weights ─────────────────────────────────────────────────────────

/// Weights for a single Q-Former block.
#[derive(Debug, Clone)]
pub struct QFormerLayerWeights {
    /// Self-attention projections (Q/K/V/O), each `[d_model × d_model]`.
    pub self_attn: CrossAttnWeights,
    /// Cross-attention projections (queries attend to image features).
    pub cross_attn: CrossAttnWeights,
    /// Feed-forward network (`d_model → ffn_dim → d_model`).
    pub ffn: FeedForward,
    /// LayerNorm applied before the self-attention sublayer.
    pub ln_self: LayerNorm,
    /// LayerNorm applied before the cross-attention sublayer.
    pub ln_cross: LayerNorm,
    /// LayerNorm applied before the feed-forward sublayer.
    pub ln_ffn: LayerNorm,
}

// ─── Model weights ──────────────────────────────────────────────────────────────

/// All learnable parameters of the Q-Former.
#[derive(Debug, Clone)]
pub struct QFormerWeights {
    /// Learnable query tokens, row-major `[n_query × d_model]`.
    pub query_tokens: Vec<f32>,
    /// Per-layer parameters; length == `n_layers`.
    pub layers: Vec<QFormerLayerWeights>,
}

impl QFormerWeights {
    /// Randomly initialise all weights with small Gaussian values.
    fn random(cfg: &QFormerConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.d_model;
        // Standard transformer init scale.
        let attn_scale = (1.0 / d as f32).sqrt();
        let ffn_in_scale = (1.0 / d as f32).sqrt();
        let ffn_out_scale = (1.0 / cfg.ffn_dim as f32).sqrt();

        let query_tokens = gaussian_vec(cfg.n_query * d, attn_scale, rng);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let self_attn = random_attn_weights(d, attn_scale, rng);
            let cross_attn = random_attn_weights(d, attn_scale, rng);
            let ffn = FeedForward {
                w1: gaussian_vec(d * cfg.ffn_dim, ffn_in_scale, rng),
                b1: vec![0.0_f32; cfg.ffn_dim],
                w2: gaussian_vec(cfg.ffn_dim * d, ffn_out_scale, rng),
                b2: vec![0.0_f32; d],
                d_model: d,
                d_ff: cfg.ffn_dim,
            };
            layers.push(QFormerLayerWeights {
                self_attn,
                cross_attn,
                ffn,
                ln_self: LayerNorm::ones(d),
                ln_cross: LayerNorm::ones(d),
                ln_ffn: LayerNorm::ones(d),
            });
        }

        Self {
            query_tokens,
            layers,
        }
    }
}

/// Build a `[d × d]` set of attention projections from Gaussian noise.
fn random_attn_weights(d: usize, scale: f32, rng: &mut LcgRng) -> CrossAttnWeights {
    CrossAttnWeights {
        w_q: gaussian_vec(d * d, scale, rng),
        w_k: gaussian_vec(d * d, scale, rng),
        w_v: gaussian_vec(d * d, scale, rng),
        w_o: gaussian_vec(d * d, scale, rng),
    }
}

/// Allocate a vector of `len` N(0, `scale`²) samples.
fn gaussian_vec(len: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    rng.fill_normal(&mut v);
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

// ─── QFormer ─────────────────────────────────────────────────────────────────

/// BLIP-2 Q-Former module.
#[derive(Debug, Clone)]
pub struct QFormer {
    pub cfg: QFormerConfig,
    pub weights: QFormerWeights,
}

impl QFormer {
    /// Create a new Q-Former with randomly initialised weights.
    pub fn new(cfg: QFormerConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = QFormerWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Forward pass.
    ///
    /// - `image_features`: `[n_image_tokens × d_model]` row-major — the frozen
    ///   image-encoder outputs the queries attend to.
    ///
    /// Returns the transformed query representation, `[n_query × d_model]`,
    /// whose length is independent of `n_image_tokens`.
    pub fn forward(&self, image_features: &[f32], n_image_tokens: usize) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let n_query = self.cfg.n_query;

        if n_image_tokens == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if image_features.len() != n_image_tokens * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_image_tokens * d,
                got: image_features.len(),
            });
        }

        // Cross-attention sublayer config (shared head count / dimension).
        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;

        // Start from the learned query tokens.
        let mut x = self.weights.query_tokens.clone();

        for layer in &self.weights.layers {
            // ── (a) Self-attention among queries (pre-norm + residual) ───────
            let self_attn = CrossAttention::with_weights(attn_cfg.clone(), layer.self_attn.clone());
            let ln_x = layer.ln_self.forward(&x, n_query)?;
            let self_out = self_attn.forward(&ln_x, &ln_x, &ln_x, n_query, n_query)?;
            add_in_place(&mut x, &self_out)?;

            // ── (b) Cross-attention: queries attend to image features ────────
            let cross_attn =
                CrossAttention::with_weights(attn_cfg.clone(), layer.cross_attn.clone());
            let ln_q = layer.ln_cross.forward(&x, n_query)?;
            let cross_out = cross_attn.forward(
                &ln_q,
                image_features,
                image_features,
                n_query,
                n_image_tokens,
            )?;
            add_in_place(&mut x, &cross_out)?;

            // ── (c) Feed-forward network (pre-norm + residual) ───────────────
            let ln_f = layer.ln_ffn.forward(&x, n_query)?;
            let ffn_out = layer.ffn.forward(&ln_f, n_query)?;
            add_in_place(&mut x, &ffn_out)?;
        }

        if x.len() != n_query * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_query * d,
                got: x.len(),
            });
        }
        Ok(x)
    }

    /// Return the learned query tokens, `[n_query × d_model]`.
    #[must_use]
    pub fn query_tokens(&self) -> &[f32] {
        &self.weights.query_tokens
    }

    /// Output dimensionality: `n_query * d_model`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.cfg.n_query * self.cfg.d_model
    }
}

/// Add `delta` into `acc` element-wise (residual connection).
fn add_in_place(acc: &mut [f32], delta: &[f32]) -> MmResult<()> {
    if acc.len() != delta.len() {
        return Err(MultiModalError::DimensionMismatch {
            expected: acc.len(),
            got: delta.len(),
        });
    }
    for (a, d) in acc.iter_mut().zip(delta.iter()) {
        *a += *d;
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qformer(seed: u64) -> QFormer {
        let mut rng = LcgRng::new(seed);
        QFormer::new(QFormerConfig::tiny(), &mut rng).expect("value should be present")
    }

    #[test]
    fn forward_output_length() {
        let qf = make_qformer(1);
        let n_img = 5;
        let img = vec![0.1_f32; n_img * qf.cfg.d_model];
        let out = qf.forward(&img, n_img).expect("forward should succeed");
        assert_eq!(out.len(), qf.cfg.n_query * qf.cfg.d_model);
    }

    #[test]
    fn query_tokens_length() {
        let qf = make_qformer(2);
        assert_eq!(qf.query_tokens().len(), qf.cfg.n_query * qf.cfg.d_model);
    }

    #[test]
    fn output_dim_correct() {
        let qf = make_qformer(3);
        assert_eq!(qf.output_dim(), qf.cfg.n_query * qf.cfg.d_model);
        assert_eq!(qf.output_dim(), 4 * 8);
    }

    #[test]
    fn output_shape_independent_of_n_image_tokens() {
        // THE defining Q-Former property: the output length is fixed at
        // n_query * d_model regardless of how many image tokens are supplied.
        let qf = make_qformer(4);
        let d = qf.cfg.d_model;

        let img3 = vec![0.2_f32; 3 * d];
        let out3 = qf.forward(&img3, 3).expect("forward should succeed");

        let img7 = vec![0.2_f32; 7 * d];
        let out7 = qf.forward(&img7, 7).expect("forward should succeed");

        assert_eq!(out3.len(), qf.cfg.n_query * d);
        assert_eq!(out7.len(), qf.cfg.n_query * d);
        assert_eq!(out3.len(), out7.len());
    }

    #[test]
    fn single_image_token_works() {
        let qf = make_qformer(5);
        let d = qf.cfg.d_model;
        let img = vec![0.3_f32; d];
        let out = qf.forward(&img, 1).expect("forward should succeed");
        assert_eq!(out.len(), qf.cfg.n_query * d);
    }

    #[test]
    fn deterministic_given_seed() {
        let qf_a = make_qformer(7);
        let qf_b = make_qformer(7);
        let d = qf_a.cfg.d_model;
        let img = vec![0.15_f32; 4 * d];
        let out_a = qf_a.forward(&img, 4).expect("forward should succeed");
        let out_b = qf_b.forward(&img, 4).expect("forward should succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn output_finite() {
        let qf = make_qformer(8);
        let d = qf.cfg.d_model;
        let img = vec![0.4_f32; 6 * d];
        let out = qf.forward(&img, 6).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn d_model_not_divisible_by_heads_errors() {
        let mut rng = LcgRng::new(9);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 10,
            n_heads: 3,
            n_layers: 2,
            ffn_dim: 16,
        };
        let err = QFormer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    #[test]
    fn image_features_wrong_length_errors() {
        let qf = make_qformer(10);
        let d = qf.cfg.d_model;
        // Claim 4 tokens but supply 3 tokens' worth of data.
        let img = vec![0.1_f32; 3 * d];
        let err = qf.forward(&img, 4).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn n_layers_one_works() {
        let mut rng = LcgRng::new(11);
        let cfg = QFormerConfig {
            n_query: 3,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
        };
        let qf = QFormer::new(cfg, &mut rng).expect("new should succeed");
        let img = vec![0.2_f32; 5 * 8];
        let out = qf.forward(&img, 5).expect("forward should succeed");
        assert_eq!(out.len(), 3 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn single_head_works() {
        let mut rng = LcgRng::new(12);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 8,
            n_heads: 1,
            n_layers: 2,
            ffn_dim: 16,
        };
        let qf = QFormer::new(cfg, &mut rng).expect("new should succeed");
        let img = vec![0.25_f32; 4 * 8];
        let out = qf.forward(&img, 4).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn changing_image_features_changes_output() {
        // Cross-attention is wired in: different image features must yield
        // a different query representation.
        let qf = make_qformer(13);
        let d = qf.cfg.d_model;
        let img_a = vec![0.1_f32; 5 * d];
        let mut img_b = vec![0.1_f32; 5 * d];
        for (i, v) in img_b.iter_mut().enumerate() {
            *v = 0.1 + (i as f32) * 0.05;
        }
        let out_a = qf.forward(&img_a, 5).expect("forward should succeed");
        let out_b = qf.forward(&img_b, 5).expect("forward should succeed");
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "cross-attention should respond to image features, diff={diff}"
        );
    }

    #[test]
    fn n_query_zero_errors() {
        let mut rng = LcgRng::new(14);
        let cfg = QFormerConfig {
            n_query: 0,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_dim: 16,
        };
        let err = QFormer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn d_model_zero_errors() {
        let mut rng = LcgRng::new(15);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 0,
            n_heads: 1,
            n_layers: 2,
            ffn_dim: 16,
        };
        let err = QFormer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn ffn_dim_zero_errors() {
        let mut rng = LcgRng::new(16);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_dim: 0,
        };
        let err = QFormer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn n_image_tokens_zero_errors() {
        let qf = make_qformer(17);
        let err = qf.forward(&[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn n_layers_zero_errors() {
        let mut rng = LcgRng::new(18);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 0,
            ffn_dim: 16,
        };
        let err = QFormer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidLayerCount));
    }

    #[test]
    fn n_layers_greater_than_one_works() {
        let mut rng = LcgRng::new(19);
        let cfg = QFormerConfig {
            n_query: 4,
            d_model: 8,
            n_heads: 2,
            n_layers: 4,
            ffn_dim: 16,
        };
        let qf = QFormer::new(cfg, &mut rng).expect("new should succeed");
        assert_eq!(qf.weights.layers.len(), 4);
        let img = vec![0.3_f32; 6 * 8];
        let out = qf.forward(&img, 6).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
