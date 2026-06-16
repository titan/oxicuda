//! LLaVA-NeXT (LLaVA-1.6) — AnyRes vision-language model.
//!
//! Reference: Liu, Li, Li, Lee 2024, *LLaVA-NeXT: Improved reasoning, OCR, and
//! world knowledge* (the 1.6 follow-up to *Visual Instruction Tuning*).
//!
//! Compact-but-faithful CPU core of the three ingredients that distinguish
//! LLaVA-NeXT from vanilla LLaVA:
//!
//! 1. **AnyRes image handling** — a high-resolution image is cut into a
//!    `grid_rows × grid_cols` mosaic of base-resolution sub-images, plus a
//!    down-sampled global *thumbnail*. Every tile (and the thumbnail) is run
//!    through the shared [`ViTEncoder`]; the CLS feature of each becomes one
//!    visual token, so a single image yields `grid_rows·grid_cols + 1` tokens.
//! 2. **Projector** — a 2-layer GELU MLP ([`LlavaProjector`]) mapping each
//!    vision feature (`vit.d_model`) into the language embedding space
//!    (`llm_dim`).
//! 3. **Token interleaving + causal LM** — the projected visual tokens replace
//!    a single `<image>` placeholder inside the text-embedding stream, and a
//!    small causal transformer is run over the fused sequence to produce
//!    next-token logits. The visual tokens are *seen by* the language tokens
//!    exactly as if they were ordinary text embeddings.
//!
//! All heavy lifting reuses existing crate primitives: [`ViTEncoder`],
//! [`LlavaProjector`], [`LayerNorm`], [`crate::cross_attn::self_cross_block::FeedForward`] and the shared masked
//! multi-head attention `mha_with_weights`.

use crate::alignment::llava_projector::{LlavaProjector, LlavaProjectorConfig};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;
use crate::vlm::{
    LmLayer, first_layer_causal_attention, gaussian_vec, lm_head_logits, random_lm_layers,
    run_causal_lm,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the LLaVA-NeXT model.
#[derive(Debug, Clone)]
pub struct LlavaNextConfig {
    /// Vision tower configuration; `vit.img_size` is the base tile resolution.
    pub vit: ViTEncoderConfig,
    /// Number of AnyRes tile rows.
    pub grid_rows: usize,
    /// Number of AnyRes tile columns.
    pub grid_cols: usize,
    /// Language-model embedding dimension (projector output width).
    pub llm_dim: usize,
    /// Attention heads of the language model. Must divide `llm_dim`.
    pub n_heads: usize,
    /// Number of causal transformer layers.
    pub n_layers: usize,
    /// Feed-forward hidden width of the language model.
    pub d_ff: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
}

impl LlavaNextConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vit: ViTEncoderConfig::tiny(),
            grid_rows: 2,
            grid_cols: 2,
            llm_dim: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
            vocab_size: 24,
        }
    }

    /// Number of visual tokens produced per image: one per tile plus the
    /// global thumbnail.
    #[must_use]
    pub fn n_visual_tokens(&self) -> usize {
        self.grid_rows * self.grid_cols + 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> MmResult<()> {
        self.vit.validate()?;
        if self.grid_rows == 0 || self.grid_cols == 0 {
            return Err(MultiModalError::InvalidPatchCount { n_patches: 0 });
        }
        if self.llm_dim == 0 || self.n_heads == 0 || self.llm_dim % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.llm_dim,
            });
        }
        if self.n_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        if self.d_ff == 0 || self.vocab_size == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// All learnable parameters of [`LlavaNext`].
#[derive(Debug, Clone)]
pub struct LlavaNextWeights {
    vit: ViTEncoderWeights,
    projector: LlavaProjector,
    layers: Vec<LmLayer>,
    final_ln: LayerNorm,
    /// LM head, `[llm_dim × vocab_size]` row-major.
    lm_head: Vec<f32>,
    /// LM head bias, `[vocab_size]`.
    lm_head_bias: Vec<f32>,
}

impl LlavaNextWeights {
    fn random(cfg: &LlavaNextConfig, rng: &mut LcgRng) -> MmResult<Self> {
        let d = cfg.llm_dim;
        let proj_cfg = LlavaProjectorConfig {
            vision_dim: cfg.vit.d_model,
            llm_dim: d,
            hidden_dim: d,
            mlp_depth: 2,
        };
        let projector = LlavaProjector::new(proj_cfg, rng)?;
        let layers = random_lm_layers(cfg.n_layers, d, cfg.d_ff, cfg.n_heads, rng)?;

        Ok(Self {
            vit: ViTEncoderWeights::random(&cfg.vit, rng),
            projector,
            layers,
            final_ln: LayerNorm::ones(d),
            lm_head: gaussian_vec(d * cfg.vocab_size, 1.0 / (d as f32).sqrt(), rng),
            lm_head_bias: vec![0.0_f32; cfg.vocab_size],
        })
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// LLaVA-NeXT vision-language model.
#[derive(Debug, Clone)]
pub struct LlavaNext {
    cfg: LlavaNextConfig,
    weights: LlavaNextWeights,
}

impl LlavaNext {
    /// Construct a model with deterministically random weights.
    pub fn new(cfg: LlavaNextConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = LlavaNextWeights::random(&cfg, rng)?;
        Ok(Self { cfg, weights })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &LlavaNextConfig {
        &self.cfg
    }

    // ── AnyRes ───────────────────────────────────────────────────────────────

    /// Split a high-resolution image into the AnyRes mosaic.
    ///
    /// `pixels` is the CHW high-resolution image of shape
    /// `[n_channels × (grid_rows·base) × (grid_cols·base)]`. Returns the
    /// `grid_rows·grid_cols` base-resolution sub-images (row-major over the
    /// grid, each CHW `[n_channels × base × base]`) together with a single
    /// average-pooled thumbnail of the same base resolution.
    pub fn split_anyres(&self, pixels: &[f32]) -> MmResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let base = self.cfg.vit.img_size;
        let ch = self.cfg.vit.n_channels;
        let gr = self.cfg.grid_rows;
        let gc = self.cfg.grid_cols;
        let hi_h = gr * base;
        let hi_w = gc * base;
        let expected = ch * hi_h * hi_w;
        if pixels.len() != expected {
            return Err(MultiModalError::DimensionMismatch {
                expected,
                got: pixels.len(),
            });
        }
        let tile_len = ch * base * base;

        // Grid of base-resolution crops.
        let mut sub_images = Vec::with_capacity(gr * gc);
        for tr in 0..gr {
            for tc in 0..gc {
                let mut tile = vec![0.0_f32; tile_len];
                for c in 0..ch {
                    for y in 0..base {
                        for x in 0..base {
                            let gy = tr * base + y;
                            let gx = tc * base + x;
                            tile[c * base * base + y * base + x] =
                                pixels[c * hi_h * hi_w + gy * hi_w + gx];
                        }
                    }
                }
                sub_images.push(tile);
            }
        }

        // Global thumbnail: average-pool the full image down to base × base.
        let inv = 1.0_f32 / (gr * gc) as f32;
        let mut thumbnail = vec![0.0_f32; tile_len];
        for c in 0..ch {
            for y in 0..base {
                for x in 0..base {
                    let mut acc = 0.0_f32;
                    for by in 0..gr {
                        for bx in 0..gc {
                            let gy = y * gr + by;
                            let gx = x * gc + bx;
                            acc += pixels[c * hi_h * hi_w + gy * hi_w + gx];
                        }
                    }
                    thumbnail[c * base * base + y * base + x] = acc * inv;
                }
            }
        }

        Ok((sub_images, thumbnail))
    }

    /// Encode an image into `[n_visual_tokens × vit.d_model]` visual tokens by
    /// running the shared ViT over every AnyRes tile and the thumbnail and
    /// collecting each one's CLS feature.
    pub fn encode_image(&self, pixels: &[f32]) -> MmResult<Vec<f32>> {
        let (sub_images, thumbnail) = self.split_anyres(pixels)?;
        let vd = self.cfg.vit.d_model;
        let mut tokens = Vec::with_capacity(self.cfg.n_visual_tokens() * vd);
        for sub in &sub_images {
            let cls = ViTEncoder::forward(sub, &self.cfg.vit, &self.weights.vit)?;
            tokens.extend_from_slice(&cls);
        }
        let cls = ViTEncoder::forward(&thumbnail, &self.cfg.vit, &self.weights.vit)?;
        tokens.extend_from_slice(&cls);
        Ok(tokens)
    }

    // ── Projector ─────────────────────────────────────────────────────────────

    /// Project `[n × vit.d_model]` visual tokens into `[n × llm_dim]`.
    pub fn project(&self, visual_tokens: &[f32]) -> MmResult<Vec<f32>> {
        let vd = self.cfg.vit.d_model;
        if visual_tokens.is_empty() || visual_tokens.len() % vd != 0 {
            return Err(MultiModalError::DimensionMismatch {
                expected: vd,
                got: visual_tokens.len(),
            });
        }
        let n = visual_tokens.len() / vd;
        self.weights.projector.project_tokens(visual_tokens, n)
    }

    // ── Interleaving ─────────────────────────────────────────────────────────

    /// Build the fused LM input by expanding the `<image>` placeholder at
    /// `image_pos` into the projected visual tokens.
    ///
    /// `text_emb` holds the `n_text` text-token embeddings (`[n_text × llm_dim]`)
    /// and `visual_tokens` the already-projected `[n_visual × llm_dim]`. The
    /// result has `n_text + n_visual` rows: the first `image_pos` text tokens,
    /// then the visual tokens, then the remaining text tokens.
    pub fn build_sequence(
        &self,
        text_emb: &[f32],
        n_text: usize,
        visual_tokens: &[f32],
        image_pos: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.llm_dim;
        if text_emb.len() != n_text * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_text * d,
                got: text_emb.len(),
            });
        }
        if visual_tokens.is_empty() || visual_tokens.len() % d != 0 {
            return Err(MultiModalError::DimensionMismatch {
                expected: d,
                got: visual_tokens.len(),
            });
        }
        let pos = image_pos.min(n_text);
        let mut fused = Vec::with_capacity(text_emb.len() + visual_tokens.len());
        fused.extend_from_slice(&text_emb[..pos * d]);
        fused.extend_from_slice(visual_tokens);
        fused.extend_from_slice(&text_emb[pos * d..]);
        Ok(fused)
    }

    // ── Language model ───────────────────────────────────────────────────────

    /// Causal self-attention weight matrix `[seq × seq]` of the **first** LM
    /// layer over a fused sequence — the head-averaged softmax weights. Exposed
    /// so callers (and tests) can confirm the lower-triangular causal mask.
    pub fn first_layer_attention(&self, fused: &[f32], seq: usize) -> MmResult<Vec<f32>> {
        first_layer_causal_attention(
            &self.weights.layers,
            fused,
            seq,
            self.cfg.llm_dim,
            self.cfg.n_heads,
        )
    }

    /// Full forward pass: encode the image, project, interleave with the text
    /// embeddings, run the causal LM and apply the LM head.
    ///
    /// Returns next-token logits `[(n_text + n_visual) × vocab_size]`.
    pub fn forward(
        &self,
        pixels: &[f32],
        text_emb: &[f32],
        n_text: usize,
        image_pos: usize,
    ) -> MmResult<Vec<f32>> {
        let visual = self.encode_image(pixels)?;
        let projected = self.project(&visual)?;
        let fused = self.build_sequence(text_emb, n_text, &projected, image_pos)?;
        let d = self.cfg.llm_dim;
        let seq = fused.len() / d;
        let hidden = run_causal_lm(
            &self.weights.layers,
            &self.weights.final_ln,
            &fused,
            seq,
            d,
            self.cfg.n_heads,
        )?;
        Ok(lm_head_logits(
            &hidden,
            seq,
            d,
            &self.weights.lm_head,
            &self.weights.lm_head_bias,
        ))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> LlavaNext {
        let mut rng = LcgRng::new(seed);
        LlavaNext::new(LlavaNextConfig::tiny(), &mut rng).expect("construct LlavaNext")
    }

    fn hi_res_image(cfg: &LlavaNextConfig, scale: f32) -> Vec<f32> {
        let base = cfg.vit.img_size;
        let n = cfg.vit.n_channels * (cfg.grid_rows * base) * (cfg.grid_cols * base);
        (0..n).map(|i| (i as f32 * 0.013 * scale).sin()).collect()
    }

    fn text(cfg: &LlavaNextConfig, n_text: usize) -> Vec<f32> {
        (0..n_text * cfg.llm_dim)
            .map(|i| (i as f32 * 0.021).cos() * 0.3)
            .collect()
    }

    // 1 ── AnyRes split yields grid·grid sub-images + a thumbnail, right sizes.
    #[test]
    fn anyres_split_counts_and_sizes() {
        let m = model(1);
        let cfg = m.config();
        let img = hi_res_image(cfg, 1.0);
        let (subs, thumb) = m.split_anyres(&img).expect("split_anyres should succeed");
        assert_eq!(subs.len(), cfg.grid_rows * cfg.grid_cols);
        let tile_len = cfg.vit.n_channels * cfg.vit.img_size * cfg.vit.img_size;
        assert!(subs.iter().all(|s| s.len() == tile_len));
        assert_eq!(thumb.len(), tile_len);
    }

    // 2 ── encode_image produces one CLS token per tile + thumbnail.
    #[test]
    fn encode_image_token_count() {
        let m = model(2);
        let cfg = m.config();
        let img = hi_res_image(cfg, 1.0);
        let tokens = m.encode_image(&img).expect("encode_image should succeed");
        assert_eq!(tokens.len(), cfg.n_visual_tokens() * cfg.vit.d_model);
        assert!(tokens.iter().all(|v| v.is_finite()));
    }

    // 3 ── Projected visual tokens live in the LM embedding dim.
    #[test]
    fn projected_tokens_have_llm_dim() {
        let m = model(3);
        let cfg = m.config();
        let img = hi_res_image(cfg, 1.0);
        let visual = m.encode_image(&img).expect("encode_image should succeed");
        let projected = m.project(&visual).expect("project should succeed");
        assert_eq!(projected.len(), cfg.n_visual_tokens() * cfg.llm_dim);
    }

    // 4 ── Placeholder expands: fused length = n_text + n_visual.
    #[test]
    fn fused_sequence_length_and_placement() {
        let m = model(4);
        let cfg = m.config();
        let n_text = 3;
        let txt = text(cfg, n_text);
        let n_vis = cfg.n_visual_tokens();
        let visual = vec![7.5_f32; n_vis * cfg.llm_dim];
        let image_pos = 2;
        let fused = m
            .build_sequence(&txt, n_text, &visual, image_pos)
            .expect("build_sequence should succeed");
        assert_eq!(fused.len(), (n_text + n_vis) * cfg.llm_dim);
        // The visual block must sit exactly at rows [image_pos, image_pos+n_vis).
        let d = cfg.llm_dim;
        for r in image_pos..image_pos + n_vis {
            assert!(
                fused[r * d..(r + 1) * d]
                    .iter()
                    .all(|&v| (v - 7.5).abs() < 1e-6)
            );
        }
        // The first text row is preserved before the image.
        assert!((fused[0] - txt[0]).abs() < 1e-6);
    }

    // 5 ── Causal attention is lower-triangular and rows sum to 1.
    #[test]
    fn causal_attention_lower_triangular_and_normalised() {
        let m = model(5);
        let cfg = m.config();
        let seq = 5;
        let fused: Vec<f32> = (0..seq * cfg.llm_dim)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let attn = m
            .first_layer_attention(&fused, seq)
            .expect("first_layer_attention should succeed");
        assert_eq!(attn.len(), seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    assert_eq!(attn[i * seq + j], 0.0, "future leak ({i},{j})");
                }
            }
            let s: f32 = attn[i * seq..(i + 1) * seq].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sum {s}");
        }
    }

    // 6 ── Different images produce different logits (real fusion).
    #[test]
    fn changing_image_changes_logits() {
        let m = model(6);
        let cfg = m.config();
        let n_text = 2;
        let txt = text(cfg, n_text);
        let img_a = hi_res_image(cfg, 1.0);
        let img_b = hi_res_image(cfg, 4.0);
        let la = m
            .forward(&img_a, &txt, n_text, 1)
            .expect("forward should succeed");
        let lb = m
            .forward(&img_b, &txt, n_text, 1)
            .expect("forward should succeed");
        let diff: f32 = la.iter().zip(lb.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "image should affect logits, diff={diff}");
    }

    // 7 ── Forward is deterministic and finite, with the expected shape.
    #[test]
    fn forward_deterministic_and_finite() {
        let m1 = model(7);
        let m2 = model(7);
        let cfg = m1.config();
        let n_text = 2;
        let txt = text(cfg, n_text);
        let img = hi_res_image(cfg, 1.0);
        let l1 = m1
            .forward(&img, &txt, n_text, 1)
            .expect("forward should succeed");
        let l2 = m2
            .forward(&img, &txt, n_text, 1)
            .expect("forward should succeed");
        let seq = n_text + cfg.n_visual_tokens();
        assert_eq!(l1.len(), seq * cfg.vocab_size);
        assert_eq!(l1, l2);
        assert!(l1.iter().all(|v| v.is_finite()));
    }

    // 8 ── Moving the image placeholder changes the fused ordering, hence logits.
    #[test]
    fn image_position_matters() {
        let m = model(8);
        let cfg = m.config();
        let n_text = 3;
        let txt = text(cfg, n_text);
        let img = hi_res_image(cfg, 1.0);
        let l0 = m
            .forward(&img, &txt, n_text, 0)
            .expect("forward should succeed");
        let l3 = m
            .forward(&img, &txt, n_text, 3)
            .expect("forward should succeed");
        let diff: f32 = l0.iter().zip(l3.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-4,
            "placeholder position should matter, diff={diff}"
        );
    }

    // 9 ── Wrong high-res image size is rejected.
    #[test]
    fn wrong_image_size_errors() {
        let m = model(9);
        let err = m.split_anyres(&[0.0_f32; 10]).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // 10 ── Invalid config (heads not dividing llm_dim) is rejected.
    #[test]
    fn invalid_config_errors() {
        let mut cfg = LlavaNextConfig::tiny();
        cfg.n_heads = 3; // 16 % 3 != 0
        let mut rng = LcgRng::new(10);
        let err = LlavaNext::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }
}
