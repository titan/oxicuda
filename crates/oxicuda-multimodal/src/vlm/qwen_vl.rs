//! Qwen-VL — vision-language model with a position-aware cross-attention
//! resampler.
//!
//! Reference: Bai, Bai, Yang, Wang et al. 2023, *Qwen-VL: A Versatile
//! Vision-Language Model for Understanding, Localization, Text Reading, and
//! Beyond*.
//!
//! Compact-but-faithful CPU core of the two parts that make Qwen-VL distinct:
//!
//! 1. **Position-aware vision-language adapter (resampler).** A fixed set of
//!    `n_query` learnable query tokens cross-attends to the variable-length
//!    grid of ViT patch features and compresses it to **exactly** `n_query`
//!    visual tokens — independent of how many patches arrive. 2-D sinusoidal
//!    positional encodings are added to the *keys* so the queries know where
//!    each patch sits, which is essential for the localization tasks Qwen-VL
//!    targets.
//! 2. **Marker-delimited insertion.** The resampled visual tokens are spliced
//!    into the text-embedding stream between learnable `<img>` / `</img>`
//!    markers, and a small causal transformer is run over the fused sequence.
//!
//! Vision encoding reuses [`ViTEncoder::forward_tokens`]; the resampler reuses
//! the shared masked multi-head attention; the language model reuses this
//! module's shared causal-LM helpers.

use crate::cross_attn::cross_attention::{CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::masked_mha::{MhaArgs, mha_with_weights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;
use crate::vlm::{
    LmLayer, add_2d_sinusoidal_pos, gaussian_vec, lm_head_logits, random_lm_layers, run_causal_lm,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the Qwen-VL model.
///
/// The resampler, markers and language model all operate at `vit.d_model`, so
/// the resampled visual tokens are drop-in compatible with the text embeddings.
#[derive(Debug, Clone)]
pub struct QwenVlConfig {
    /// Vision tower configuration; `vit.d_model` is the shared model width.
    pub vit: ViTEncoderConfig,
    /// Fixed number of visual query tokens the resampler outputs.
    pub n_query: usize,
    /// Attention heads (resampler + language model). Must divide `vit.d_model`.
    pub n_heads: usize,
    /// Number of causal language-model layers.
    pub n_layers: usize,
    /// Feed-forward hidden width of the language model.
    pub d_ff: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
}

impl QwenVlConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vit: ViTEncoderConfig::tiny(),
            n_query: 4,
            n_heads: 2,
            n_layers: 2,
            d_ff: 16,
            vocab_size: 24,
        }
    }

    /// Shared model width (`vit.d_model`).
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.vit.d_model
    }

    /// Validate the configuration.
    pub fn validate(&self) -> MmResult<()> {
        self.vit.validate()?;
        let d = self.vit.d_model;
        if self.n_query == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if self.n_heads == 0 || d % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: d,
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

/// All learnable parameters of [`QwenVl`].
#[derive(Debug, Clone)]
pub struct QwenVlWeights {
    vit: ViTEncoderWeights,
    /// Learnable resampler query tokens, `[n_query × d_model]`.
    query_tokens: Vec<f32>,
    /// Resampler cross-attention projections.
    resampler_attn: CrossAttnWeights,
    /// Pre-norm on the resampler queries.
    ln_query: LayerNorm,
    /// Pre-norm on the resampler keys/values.
    ln_kv: LayerNorm,
    /// `<img>` begin marker embedding, `[d_model]`.
    img_begin: Vec<f32>,
    /// `</img>` end marker embedding, `[d_model]`.
    img_end: Vec<f32>,
    layers: Vec<LmLayer>,
    final_ln: LayerNorm,
    /// LM head, `[d_model × vocab_size]`.
    lm_head: Vec<f32>,
    /// LM head bias, `[vocab_size]`.
    lm_head_bias: Vec<f32>,
}

impl QwenVlWeights {
    fn random(cfg: &QwenVlConfig, rng: &mut LcgRng) -> MmResult<Self> {
        let d = cfg.d_model();
        let attn_cfg = CrossAttnConfig::new(cfg.n_heads, d, 0.0)?;
        let scale = 1.0_f32 / (d as f32).sqrt();
        let layers = random_lm_layers(cfg.n_layers, d, cfg.d_ff, cfg.n_heads, rng)?;
        Ok(Self {
            vit: ViTEncoderWeights::random(&cfg.vit, rng),
            query_tokens: gaussian_vec(cfg.n_query * d, scale, rng),
            resampler_attn: CrossAttnWeights::random(&attn_cfg, rng),
            ln_query: LayerNorm::ones(d),
            ln_kv: LayerNorm::ones(d),
            img_begin: gaussian_vec(d, scale, rng),
            img_end: gaussian_vec(d, scale, rng),
            layers,
            final_ln: LayerNorm::ones(d),
            lm_head: gaussian_vec(d * cfg.vocab_size, scale, rng),
            lm_head_bias: vec![0.0_f32; cfg.vocab_size],
        })
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Qwen-VL vision-language model.
#[derive(Debug, Clone)]
pub struct QwenVl {
    cfg: QwenVlConfig,
    weights: QwenVlWeights,
}

impl QwenVl {
    /// Construct a model with deterministically random weights.
    pub fn new(cfg: QwenVlConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = QwenVlWeights::random(&cfg, rng)?;
        Ok(Self { cfg, weights })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &QwenVlConfig {
        &self.cfg
    }

    /// The `<img>` begin marker embedding, `[d_model]`.
    #[must_use]
    pub fn image_begin(&self) -> &[f32] {
        &self.weights.img_begin
    }

    /// The `</img>` end marker embedding, `[d_model]`.
    #[must_use]
    pub fn image_end(&self) -> &[f32] {
        &self.weights.img_end
    }

    // ── Vision ────────────────────────────────────────────────────────────────

    /// Encode an image into its grid of ViT patch features
    /// `[n_patches × d_model]` (the CLS token is dropped — the resampler works
    /// on the spatial patch tokens).
    pub fn vision_encode(&self, pixels: &[f32]) -> MmResult<Vec<f32>> {
        let tokens = ViTEncoder::forward_tokens(pixels, &self.cfg.vit, &self.weights.vit)?;
        let d = self.cfg.d_model();
        Ok(tokens[d..].to_vec())
    }

    // ── Resampler ─────────────────────────────────────────────────────────────

    /// Resample `[n_patches × d_model]` patch features into the fixed
    /// `[n_query × d_model]` visual tokens, with 2-D positional encodings added
    /// to the keys. The output length is `n_query` regardless of `n_patches`.
    pub fn resample(&self, patch_features: &[f32], n_patches: usize) -> MmResult<Vec<f32>> {
        Ok(self.resample_inner(patch_features, n_patches, true)?.0)
    }

    /// Like [`Self::resample`] but with the 2-D positional encoding disabled —
    /// used to demonstrate that positions actually change the result.
    pub fn resample_no_pos(&self, patch_features: &[f32], n_patches: usize) -> MmResult<Vec<f32>> {
        Ok(self.resample_inner(patch_features, n_patches, false)?.0)
    }

    /// Resample and also return the cross-attention weights
    /// `[n_query × n_patches]` (head-averaged; each query's row sums to 1).
    pub fn resample_attention(
        &self,
        patch_features: &[f32],
        n_patches: usize,
    ) -> MmResult<(Vec<f32>, Vec<f32>)> {
        self.resample_inner(patch_features, n_patches, true)
    }

    fn resample_inner(
        &self,
        patch_features: &[f32],
        n_patches: usize,
        use_pos: bool,
    ) -> MmResult<(Vec<f32>, Vec<f32>)> {
        let d = self.cfg.d_model();
        let n_query = self.cfg.n_query;
        if n_patches == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if patch_features.len() != n_patches * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_patches * d,
                got: patch_features.len(),
            });
        }

        // Keys/values = patch features, optionally + 2-D positional encoding.
        let mut kv = patch_features.to_vec();
        if use_pos {
            let (rows, cols) = grid_shape(n_patches);
            add_2d_sinusoidal_pos(&mut kv, rows, cols, d);
        }

        // Pre-norm queries and keys, then cross-attend.
        let q = self
            .weights
            .ln_query
            .forward(&self.weights.query_tokens, n_query)?;
        let k = self.weights.ln_kv.forward(&kv, n_patches)?;
        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;
        let args = MhaArgs {
            query: &q,
            key: &k,
            value: &k,
            q_len: n_query,
            kv_len: n_patches,
            causal: false,
        };
        let (attn_out, weights) = mha_with_weights(&args, &attn_cfg, &self.weights.resampler_attn)?;

        // Residual from the learnable query tokens (pre-norm residual).
        let mut tokens = attn_out;
        for (t, q0) in tokens.iter_mut().zip(self.weights.query_tokens.iter()) {
            *t += q0;
        }
        Ok((tokens, weights))
    }

    // ── Interleaving ─────────────────────────────────────────────────────────

    /// Splice the visual tokens into the text stream between the learnable
    /// `<img>` / `</img>` markers.
    ///
    /// Result layout (rows): `[ text[..image_pos], <img>, visual…, </img>,
    /// text[image_pos..] ]`, total `n_text + n_visual + 2` rows. The begin
    /// marker lands at row `image_pos`, the end marker at row
    /// `image_pos + 1 + n_visual`.
    pub fn build_sequence(
        &self,
        text_emb: &[f32],
        n_text: usize,
        visual_tokens: &[f32],
        image_pos: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model();
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
        let mut fused = Vec::with_capacity(text_emb.len() + visual_tokens.len() + 2 * d);
        fused.extend_from_slice(&text_emb[..pos * d]);
        fused.extend_from_slice(&self.weights.img_begin);
        fused.extend_from_slice(visual_tokens);
        fused.extend_from_slice(&self.weights.img_end);
        fused.extend_from_slice(&text_emb[pos * d..]);
        Ok(fused)
    }

    /// Row indices of the `<img>` and `</img>` markers in the fused sequence
    /// produced by [`Self::build_sequence`].
    #[must_use]
    pub fn marker_positions(
        &self,
        n_text: usize,
        n_visual: usize,
        image_pos: usize,
    ) -> (usize, usize) {
        let pos = image_pos.min(n_text);
        (pos, pos + 1 + n_visual)
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /// Full forward pass: encode the image, resample, splice between markers,
    /// run the causal LM and apply the LM head.
    ///
    /// Returns next-token logits `[(n_text + n_query + 2) × vocab_size]`.
    pub fn forward(
        &self,
        pixels: &[f32],
        text_emb: &[f32],
        n_text: usize,
        image_pos: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model();
        let patches = self.vision_encode(pixels)?;
        let n_patches = patches.len() / d;
        let visual = self.resample(&patches, n_patches)?;
        let fused = self.build_sequence(text_emb, n_text, &visual, image_pos)?;
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Factor `n` into a grid `(rows, cols)` with `rows·cols == n` and `rows` the
/// largest divisor not exceeding `√n` (a near-square layout, falling back to
/// `1 × n` for primes). Used to lay out patch features for 2-D positions.
fn grid_shape(n: usize) -> (usize, usize) {
    let mut rows = 1;
    let mut r = 1;
    while r * r <= n {
        if n % r == 0 {
            rows = r;
        }
        r += 1;
    }
    (rows, n / rows)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> QwenVl {
        let mut rng = LcgRng::new(seed);
        QwenVl::new(QwenVlConfig::tiny(), &mut rng).expect("construct QwenVl")
    }

    fn patches(d: usize, n: usize, scale: f32) -> Vec<f32> {
        (0..n * d)
            .map(|i| (i as f32 * 0.017 * scale).sin())
            .collect()
    }

    fn image(cfg: &QwenVlConfig, scale: f32) -> Vec<f32> {
        let n = cfg.vit.n_channels * cfg.vit.img_size * cfg.vit.img_size;
        (0..n).map(|i| (i as f32 * 0.011 * scale).cos()).collect()
    }

    fn text(cfg: &QwenVlConfig, n_text: usize) -> Vec<f32> {
        (0..n_text * cfg.d_model())
            .map(|i| (i as f32 * 0.023).sin() * 0.3)
            .collect()
    }

    // 1 ── The resampler outputs exactly n_query tokens for ANY patch count.
    #[test]
    fn resampler_compresses_to_fixed_count() {
        let m = model(1);
        let cfg = m.config();
        let d = cfg.d_model();
        let out16 = m
            .resample(&patches(d, 16, 1.0), 16)
            .expect("value should be present");
        let out64 = m
            .resample(&patches(d, 64, 1.0), 64)
            .expect("value should be present");
        assert_eq!(out16.len(), cfg.n_query * d);
        assert_eq!(out64.len(), cfg.n_query * d);
        assert_eq!(out16.len(), out64.len());
        assert!(out16.iter().chain(out64.iter()).all(|v| v.is_finite()));
    }

    // 2 ── Cross-attention weights over the patches sum to 1 per query.
    #[test]
    fn resampler_weights_sum_to_one() {
        let m = model(2);
        let cfg = m.config();
        let d = cfg.d_model();
        let n_patches = 16;
        let (_, w) = m
            .resample_attention(&patches(d, n_patches, 1.0), n_patches)
            .expect("value should be present");
        assert_eq!(w.len(), cfg.n_query * n_patches);
        for q in 0..cfg.n_query {
            let s: f32 = w[q * n_patches..(q + 1) * n_patches].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "query {q} sum {s}");
        }
    }

    // 3 ── The 2-D positional encoding changes the resampled tokens.
    #[test]
    fn positional_encoding_changes_output() {
        let m = model(3);
        let cfg = m.config();
        let d = cfg.d_model();
        let n_patches = 16;
        let p = patches(d, n_patches, 1.0);
        let with_pos = m.resample(&p, n_patches).expect("resample should succeed");
        let without_pos = m
            .resample_no_pos(&p, n_patches)
            .expect("resample_no_pos should succeed");
        let diff: f32 = with_pos
            .iter()
            .zip(without_pos.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "2-D positions should change output, diff={diff}"
        );
    }

    // 4 ── Varying the patch features changes the resampled tokens.
    #[test]
    fn varying_patches_changes_tokens() {
        let m = model(4);
        let cfg = m.config();
        let d = cfg.d_model();
        let n_patches = 16;
        let a = m
            .resample(&patches(d, n_patches, 1.0), n_patches)
            .expect("value should be present");
        let b = m
            .resample(&patches(d, n_patches, 5.0), n_patches)
            .expect("value should be present");
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-4, "patches should change tokens, diff={diff}");
    }

    // 5 ── Image markers are placed correctly around the visual tokens.
    #[test]
    fn image_markers_placed_correctly() {
        let m = model(5);
        let cfg = m.config();
        let d = cfg.d_model();
        let n_text = 3;
        let txt = text(cfg, n_text);
        let n_vis = cfg.n_query;
        let visual = vec![2.5_f32; n_vis * d];
        let image_pos = 1;
        let fused = m
            .build_sequence(&txt, n_text, &visual, image_pos)
            .expect("build_sequence should succeed");
        assert_eq!(fused.len(), (n_text + n_vis + 2) * d);
        let (begin_idx, end_idx) = m.marker_positions(n_text, n_vis, image_pos);
        assert_eq!(begin_idx, image_pos);
        assert_eq!(end_idx, image_pos + 1 + n_vis);
        // Begin / end markers match the learnable embeddings.
        assert_eq!(&fused[begin_idx * d..(begin_idx + 1) * d], m.image_begin());
        assert_eq!(&fused[end_idx * d..(end_idx + 1) * d], m.image_end());
        // Visual tokens sit strictly between the markers.
        for r in (begin_idx + 1)..end_idx {
            assert!(
                fused[r * d..(r + 1) * d]
                    .iter()
                    .all(|&v| (v - 2.5).abs() < 1e-6)
            );
        }
    }

    // 6 ── End-to-end forward is deterministic, finite and correctly shaped.
    #[test]
    fn forward_deterministic_finite_shape() {
        let m1 = model(6);
        let m2 = model(6);
        let cfg = m1.config();
        let n_text = 2;
        let txt = text(cfg, n_text);
        let img = image(cfg, 1.0);
        let l1 = m1
            .forward(&img, &txt, n_text, 1)
            .expect("forward should succeed");
        let l2 = m2
            .forward(&img, &txt, n_text, 1)
            .expect("forward should succeed");
        let seq = n_text + cfg.n_query + 2;
        assert_eq!(l1.len(), seq * cfg.vocab_size);
        assert_eq!(l1, l2);
        assert!(l1.iter().all(|v| v.is_finite()));
    }

    // 7 ── Different images yield different logits (vision really feeds in).
    #[test]
    fn changing_image_changes_logits() {
        let m = model(7);
        let cfg = m.config();
        let n_text = 2;
        let txt = text(cfg, n_text);
        let la = m
            .forward(&image(cfg, 1.0), &txt, n_text, 1)
            .expect("value should be present");
        let lb = m
            .forward(&image(cfg, 6.0), &txt, n_text, 1)
            .expect("value should be present");
        let diff: f32 = la.iter().zip(lb.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "image should affect logits, diff={diff}");
    }

    // 8 ── vision_encode yields the ViT patch grid (n_patches × d).
    #[test]
    fn vision_encode_patch_count() {
        let m = model(8);
        let cfg = m.config();
        let patches = m
            .vision_encode(&image(cfg, 1.0))
            .expect("value should be present");
        assert_eq!(patches.len(), cfg.vit.n_patches() * cfg.d_model());
    }

    // 9 ── grid_shape always factors exactly and is near-square.
    #[test]
    fn grid_shape_factors_exactly() {
        for n in [1usize, 4, 5, 12, 16, 36, 64] {
            let (r, c) = grid_shape(n);
            assert_eq!(r * c, n, "n={n}");
            assert!(r <= c, "rows should be the smaller side for n={n}");
        }
    }

    // 10 ── Empty patch input is rejected.
    #[test]
    fn empty_patches_errors() {
        let m = model(10);
        let err = m.resample(&[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    // 11 ── Heads must divide d_model.
    #[test]
    fn invalid_heads_errors() {
        let mut cfg = QwenVlConfig::tiny();
        cfg.n_heads = 3; // 8 % 3 != 0
        let mut rng = LcgRng::new(11);
        let err = QwenVl::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }
}
