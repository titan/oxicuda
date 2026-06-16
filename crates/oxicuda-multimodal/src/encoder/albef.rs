//! ALBEF — "Align before Fuse" vision-language pre-training (Li et al., 2021).
//!
//! ALBEF first **aligns** unimodal image and text representations with an
//! image-text contrastive (ITC) loss on the projected `[CLS]` embeddings, *then*
//! **fuses** them with a multimodal encoder in which the text tokens cross-attend
//! to the image tokens. On top of the fused representation it places an
//! image-text matching (ITM) binary head and a masked-language-modeling (MLM)
//! head. To combat the noise in web image-text pairs it uses **momentum
//! distillation**: an exponential-moving-average (EMA) copy of the model
//! produces soft pseudo-targets for the ITC and MLM objectives.
//!
//! This module assembles those pieces from the crate's existing building blocks:
//!
//! * image encoder — [`ViTEncoder::forward_tokens`] (full patch token sequence),
//! * text encoder — a BERT-style stack producing the full token sequence,
//! * fusion encoder — [`SelfCrossBlock`] (text self-attention + cross-attention
//!   onto the image tokens + FFN),
//! * ITC projection heads, an [`ItmHead`], and a linear MLM head.
//!
//! Everything is flat row-major `Vec<f32>`; randomness uses the deterministic
//! [`LcgRng`].

use crate::alignment::matching::ItmHead;
use crate::cross_attn::cross_attention::{
    CrossAttention, CrossAttnConfig, CrossAttnWeights, softmax_rows_inplace,
};
use crate::cross_attn::self_cross_block::{
    FeedForward, LayerNorm, SelfCrossBlock, SelfCrossBlockWeights,
};
use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Configuration ──────────────────────────────────────────────────────────────

/// ALBEF configuration. Image and text share the model width `d_model`; the ITC
/// projection maps `[CLS]` embeddings to `proj_dim`.
#[derive(Debug, Clone)]
pub struct AlbefConfig {
    /// Image (ViT) encoder configuration.
    pub vit: ViTEncoderConfig,
    /// Text vocabulary size.
    pub vocab_size: usize,
    /// Maximum text sequence length.
    pub max_text_len: usize,
    /// Shared model width.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Number of unimodal text encoder layers.
    pub n_text_layers: usize,
    /// Number of multimodal fusion layers.
    pub n_fusion_layers: usize,
    /// Feed-forward width.
    pub d_ff: usize,
    /// ITC projection dimension.
    pub proj_dim: usize,
    /// Contrastive temperature.
    pub temperature: f32,
    /// Momentum coefficient for the EMA teacher (close to 1, e.g. 0.995).
    pub momentum: f32,
}

impl AlbefConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        let mut vit = ViTEncoderConfig::tiny();
        vit.d_model = 16;
        vit.n_heads = 2;
        vit.d_ff = 32;
        Self {
            vit,
            vocab_size: 40,
            max_text_len: 16,
            d_model: 16,
            n_heads: 2,
            n_text_layers: 2,
            n_fusion_layers: 2,
            d_ff: 32,
            proj_dim: 8,
            temperature: 0.07,
            momentum: 0.995,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> MmResult<()> {
        self.vit.validate()?;
        if self.vit.d_model != self.d_model {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_model,
                got: self.vit.d_model,
            });
        }
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_text_layers == 0 || self.n_fusion_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        if self.proj_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(MultiModalError::InvalidTemperature {
                temp: self.temperature,
            });
        }
        Ok(())
    }
}

// ─── Text-encoder layer weights ─────────────────────────────────────────────────

/// One pre-norm text self-attention encoder layer.
#[derive(Debug, Clone)]
pub struct TextLayer {
    /// Self-attention projections.
    pub attn: CrossAttnWeights,
    /// Feed-forward network.
    pub ffn: FeedForward,
    /// LayerNorm before attention.
    pub ln1: LayerNorm,
    /// LayerNorm before the feed-forward.
    pub ln2: LayerNorm,
}

impl TextLayer {
    fn zeros(cfg: &AlbefConfig) -> Self {
        let d = cfg.d_model;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            attn: CrossAttnWeights::zeros(&attn_cfg),
            ffn: FeedForward::zeros(d, cfg.d_ff),
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
        }
    }
}

// ─── Model weights ──────────────────────────────────────────────────────────────

/// All learnable ALBEF weights (one tower; the EMA teacher holds a second copy).
#[derive(Debug, Clone)]
pub struct AlbefWeights {
    /// Image (ViT) encoder weights.
    pub vit: ViTEncoderWeights,
    /// Text token embedding table `[vocab_size × d_model]`.
    pub text_token_embed: Vec<f32>,
    /// Text positional embedding table `[max_text_len × d_model]`.
    pub text_pos_embed: Vec<f32>,
    /// Unimodal text encoder layers.
    pub text_layers: Vec<TextLayer>,
    /// Final LayerNorm of the unimodal text encoder.
    pub text_final_ln: LayerNorm,
    /// ITC image projection `[d_model × proj_dim]`.
    pub image_proj: Vec<f32>,
    /// ITC text projection `[d_model × proj_dim]`.
    pub text_proj: Vec<f32>,
    /// Multimodal fusion layers (text cross-attends to image tokens).
    pub fusion_layers: Vec<SelfCrossBlockWeights>,
    /// ITM binary head.
    pub itm_head: ItmHead,
    /// MLM head `[d_model × vocab_size]`.
    pub mlm_head: Vec<f32>,
    /// MLM bias `[vocab_size]`.
    pub mlm_bias: Vec<f32>,
}

impl AlbefWeights {
    /// Zero-initialised weights (LayerNorm scales = 1).
    #[must_use]
    pub fn zeros(cfg: &AlbefConfig) -> Self {
        let d = cfg.d_model;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            vit: ViTEncoderWeights::zeros(&cfg.vit),
            text_token_embed: vec![0.0_f32; cfg.vocab_size * d],
            text_pos_embed: vec![0.0_f32; cfg.max_text_len * d],
            text_layers: (0..cfg.n_text_layers)
                .map(|_| TextLayer::zeros(cfg))
                .collect(),
            text_final_ln: LayerNorm::ones(d),
            image_proj: vec![0.0_f32; d * cfg.proj_dim],
            text_proj: vec![0.0_f32; d * cfg.proj_dim],
            fusion_layers: (0..cfg.n_fusion_layers)
                .map(|_| SelfCrossBlockWeights::zeros(&attn_cfg))
                .collect(),
            itm_head: ItmHead::zeros(d, cfg.d_ff),
            mlm_head: vec![0.0_f32; d * cfg.vocab_size],
            mlm_bias: vec![0.0_f32; cfg.vocab_size],
        }
    }

    /// Deterministic Gaussian initialisation so the towers propagate inputs.
    #[must_use]
    pub fn random(cfg: &AlbefConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.d_model;
        let s_d = 1.0_f32 / (d as f32).sqrt();
        let s_ff = 1.0_f32 / (cfg.d_ff as f32).sqrt();
        let mut w = Self::zeros(cfg);

        w.vit = ViTEncoderWeights::random(&cfg.vit, rng);
        fill_scaled(&mut w.text_token_embed, s_d, rng);
        fill_scaled(&mut w.text_pos_embed, s_d, rng);
        for layer in &mut w.text_layers {
            fill_scaled(&mut layer.attn.w_q, s_d, rng);
            fill_scaled(&mut layer.attn.w_k, s_d, rng);
            fill_scaled(&mut layer.attn.w_v, s_d, rng);
            fill_scaled(&mut layer.attn.w_o, s_d, rng);
            fill_scaled(&mut layer.ffn.w1, s_d, rng);
            fill_scaled(&mut layer.ffn.w2, s_ff, rng);
        }
        fill_scaled(&mut w.image_proj, s_d, rng);
        fill_scaled(&mut w.text_proj, s_d, rng);
        for fl in &mut w.fusion_layers {
            fill_scaled(&mut fl.self_attn.w_q, s_d, rng);
            fill_scaled(&mut fl.self_attn.w_k, s_d, rng);
            fill_scaled(&mut fl.self_attn.w_v, s_d, rng);
            fill_scaled(&mut fl.self_attn.w_o, s_d, rng);
            fill_scaled(&mut fl.cross_attn.w_q, s_d, rng);
            fill_scaled(&mut fl.cross_attn.w_k, s_d, rng);
            fill_scaled(&mut fl.cross_attn.w_v, s_d, rng);
            fill_scaled(&mut fl.cross_attn.w_o, s_d, rng);
            fill_scaled(&mut fl.ffn.w1, s_d, rng);
            fill_scaled(&mut fl.ffn.w2, s_ff, rng);
        }
        fill_scaled(&mut w.itm_head.w1, s_d, rng);
        fill_scaled(&mut w.itm_head.w2, 1.0 / (cfg.d_ff as f32).sqrt(), rng);
        fill_scaled(&mut w.mlm_head, s_d, rng);
        w
    }

    /// In-place EMA update toward an `online` set of weights:
    /// `θ_ema ← m·θ_ema + (1−m)·θ_online`. This is the momentum-distillation
    /// teacher update; with `0 < m < 1` the teacher moves *toward* the online
    /// (student) parameters.
    pub fn ema_update(&mut self, online: &AlbefWeights, momentum: f32) {
        let m = momentum;
        ema_vec(&mut self.text_token_embed, &online.text_token_embed, m);
        ema_vec(&mut self.text_pos_embed, &online.text_pos_embed, m);
        ema_vec(&mut self.image_proj, &online.image_proj, m);
        ema_vec(&mut self.text_proj, &online.text_proj, m);
        ema_vec(&mut self.mlm_head, &online.mlm_head, m);
        ema_vec(&mut self.mlm_bias, &online.mlm_bias, m);
        ema_vec(&mut self.vit.cls_token, &online.vit.cls_token, m);
        ema_vec(&mut self.vit.patch_embed, &online.vit.patch_embed, m);
        for (e, o) in self.text_layers.iter_mut().zip(online.text_layers.iter()) {
            ema_vec(&mut e.attn.w_q, &o.attn.w_q, m);
            ema_vec(&mut e.attn.w_k, &o.attn.w_k, m);
            ema_vec(&mut e.attn.w_v, &o.attn.w_v, m);
            ema_vec(&mut e.attn.w_o, &o.attn.w_o, m);
            ema_vec(&mut e.ffn.w1, &o.ffn.w1, m);
            ema_vec(&mut e.ffn.w2, &o.ffn.w2, m);
        }
    }
}

/// Fill `buf` with N(0,1) samples scaled by `scale`.
fn fill_scaled(buf: &mut [f32], scale: f32, rng: &mut LcgRng) {
    rng.fill_normal(buf);
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

/// `dst ← m·dst + (1−m)·src` element-wise (lengths must match; mismatches are
/// skipped defensively, which only matters for hand-constructed weights).
fn ema_vec(dst: &mut [f32], src: &[f32], m: f32) {
    if dst.len() != src.len() {
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = m * *d + (1.0 - m) * s;
    }
}

// ─── Outputs ────────────────────────────────────────────────────────────────────

/// ITC outputs: normalised projected `[CLS]` embeddings and their similarity.
#[derive(Debug, Clone)]
pub struct ItcOutput {
    /// Normalised image embeddings `[batch × proj_dim]`.
    pub image_embed: Vec<f32>,
    /// Normalised text embeddings `[batch × proj_dim]`.
    pub text_embed: Vec<f32>,
    /// `[batch × batch]` image→text similarity matrix.
    pub sim_i2t: Vec<f32>,
    /// `[batch × batch]` text→image similarity matrix (transpose of `sim_i2t`).
    pub sim_t2i: Vec<f32>,
}

// ─── Model ──────────────────────────────────────────────────────────────────────

/// ALBEF model (a single tower; pair with an EMA teacher copy of [`AlbefWeights`]).
pub struct Albef {
    cfg: AlbefConfig,
}

impl Albef {
    /// Construct from a configuration.
    #[must_use]
    pub fn new(cfg: AlbefConfig) -> Self {
        Self { cfg }
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &AlbefConfig {
        &self.cfg
    }

    /// Encode an image to its full patch token sequence `[(1+n_patches) × d_model]`.
    pub fn encode_image_tokens(&self, image: &[f32], w: &AlbefWeights) -> MmResult<Vec<f32>> {
        ViTEncoder::forward_tokens(image, &self.cfg.vit, &w.vit)
    }

    /// Encode a token sequence to the full unimodal text sequence
    /// `[seq_len × d_model]` (CLS at position 0).
    pub fn encode_text_tokens(&self, token_ids: &[u32], w: &AlbefWeights) -> MmResult<Vec<f32>> {
        self.cfg.validate()?;
        let d = self.cfg.d_model;
        let seq = token_ids.len();
        if seq == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        for &tid in token_ids {
            if tid as usize >= self.cfg.vocab_size {
                return Err(MultiModalError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }
        // Token + positional embeddings.
        let mut hidden = vec![0.0_f32; seq * d];
        for (pos, &tid) in token_ids.iter().enumerate() {
            let p = pos.min(self.cfg.max_text_len - 1);
            for i in 0..d {
                hidden[pos * d + i] =
                    w.text_token_embed[tid as usize * d + i] + w.text_pos_embed[p * d + i];
            }
        }
        for layer in &w.text_layers {
            hidden = text_layer_forward(&hidden, seq, &self.cfg, layer)?;
        }
        hidden = w.text_final_ln.forward(&hidden, seq)?;
        Ok(hidden)
    }

    /// Project & L2-normalise a `[CLS]` embedding (position 0 of a token
    /// sequence) with the given `[d_model × proj_dim]` projection.
    fn project_cls(&self, tokens: &[f32], proj: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let p = self.cfg.proj_dim;
        let cls = &tokens[..d];
        let mut out = vec![0.0_f32; p];
        for j in 0..p {
            let mut acc = 0.0_f32;
            for i in 0..d {
                acc += cls[i] * proj[i * p + j];
            }
            out[j] = acc;
        }
        l2_normalise_row(&mut out);
        out
    }

    /// Image-text contrastive forward over a batch.
    ///
    /// `images`: `batch` flat ViT images. `texts`: `batch` token-id sequences
    /// (variable length allowed). Returns the projected normalised `[CLS]`
    /// embeddings and the `[batch × batch]` similarity matrices.
    pub fn itc_forward(
        &self,
        images: &[&[f32]],
        texts: &[&[u32]],
        w: &AlbefWeights,
    ) -> MmResult<ItcOutput> {
        if images.is_empty() || texts.is_empty() {
            return Err(MultiModalError::EmptyInput);
        }
        if images.len() != texts.len() {
            return Err(MultiModalError::DimensionMismatch {
                expected: images.len(),
                got: texts.len(),
            });
        }
        let batch = images.len();
        let p = self.cfg.proj_dim;

        let mut image_embed = vec![0.0_f32; batch * p];
        let mut text_embed = vec![0.0_f32; batch * p];
        for b in 0..batch {
            let img_tokens = self.encode_image_tokens(images[b], w)?;
            let txt_tokens = self.encode_text_tokens(texts[b], w)?;
            let ie = self.project_cls(&img_tokens, &w.image_proj);
            let te = self.project_cls(&txt_tokens, &w.text_proj);
            image_embed[b * p..(b + 1) * p].copy_from_slice(&ie);
            text_embed[b * p..(b + 1) * p].copy_from_slice(&te);
        }

        // Similarity matrices scaled by 1/temperature.
        let t = self.cfg.temperature;
        let mut sim_i2t = vec![0.0_f32; batch * batch];
        let mut sim_t2i = vec![0.0_f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                let mut dot = 0.0_f32;
                for k in 0..p {
                    dot += image_embed[i * p + k] * text_embed[j * p + k];
                }
                sim_i2t[i * batch + j] = dot / t;
                sim_t2i[j * batch + i] = dot / t;
            }
        }
        Ok(ItcOutput {
            image_embed,
            text_embed,
            sim_i2t,
            sim_t2i,
        })
    }

    /// Image-text contrastive loss: symmetric InfoNCE over the (already
    /// temperature-scaled) similarity matrices, with the diagonal as positives.
    ///
    /// `alpha` blends the hard one-hot targets with the EMA-teacher soft targets
    /// (momentum distillation): `target = (1−α)·one_hot + α·softmax(sim_teacher)`.
    /// Pass `alpha = 0` and `sim_*_teacher = None` for the plain ITC loss.
    pub fn itc_loss(
        &self,
        out: &ItcOutput,
        batch: usize,
        teacher: Option<&ItcOutput>,
        alpha: f32,
    ) -> MmResult<f32> {
        let li2t = soft_ce(&out.sim_i2t, batch, teacher.map(|t| &t.sim_i2t[..]), alpha)?;
        let lt2i = soft_ce(&out.sim_t2i, batch, teacher.map(|t| &t.sim_t2i[..]), alpha)?;
        let loss = (li2t + lt2i) / 2.0;
        if !loss.is_finite() {
            return Err(MultiModalError::NanEncountered {
                location: "albef::itc_loss",
            });
        }
        Ok(loss)
    }

    /// Multimodal fusion: the text token sequence self-attends and cross-attends
    /// to the image token sequence through the fusion layers. Returns the fused
    /// text sequence `[text_len × d_model]`.
    pub fn fuse(
        &self,
        text_tokens: &[f32],
        image_tokens: &[f32],
        text_len: usize,
        image_len: usize,
        w: &AlbefWeights,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        if text_tokens.len() != text_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: text_len * d,
                got: text_tokens.len(),
            });
        }
        if image_tokens.len() != image_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: image_len * d,
                got: image_tokens.len(),
            });
        }
        let attn_cfg = CrossAttnConfig {
            n_heads: self.cfg.n_heads,
            d_model: d,
            d_k: d / self.cfg.n_heads,
            d_v: d / self.cfg.n_heads,
            dropout_rate: 0.0,
        };
        let mut x = text_tokens.to_vec();
        for fl in &w.fusion_layers {
            let block = SelfCrossBlock::with_weights(attn_cfg.clone(), fl.clone());
            x = block.forward(&x, image_tokens, text_len, image_len)?;
        }
        Ok(x)
    }

    /// Image-text matching logit from a fused text sequence: pools the fused
    /// `[CLS]` (position 0) and applies the binary ITM head.
    pub fn itm_logit(&self, fused_text: &[f32], w: &AlbefWeights) -> MmResult<f32> {
        let d = self.cfg.d_model;
        if fused_text.len() < d {
            return Err(MultiModalError::DimensionMismatch {
                expected: d,
                got: fused_text.len(),
            });
        }
        w.itm_head.forward_single(&fused_text[..d])
    }

    /// Two-way ITM probabilities `[p_no_match, p_match]` from a single logit
    /// (`p_match = σ(logit)`), so the head is a genuine 2-class output.
    pub fn itm_probs(&self, fused_text: &[f32], w: &AlbefWeights) -> MmResult<[f32; 2]> {
        let logit = self.itm_logit(fused_text, w)?;
        let p_match = 1.0 / (1.0 + (-logit).exp());
        Ok([1.0 - p_match, p_match])
    }

    /// MLM logits over the vocabulary for every fused text position
    /// `[text_len × vocab_size]`.
    pub fn mlm_logits(
        &self,
        fused_text: &[f32],
        text_len: usize,
        w: &AlbefWeights,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        if fused_text.len() != text_len * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: text_len * d,
                got: fused_text.len(),
            });
        }
        let mut out = vec![0.0_f32; text_len * v];
        for t in 0..text_len {
            for j in 0..v {
                let mut acc = w.mlm_bias[j];
                for i in 0..d {
                    acc += fused_text[t * d + i] * w.mlm_head[i * v + j];
                }
                out[t * v + j] = acc;
            }
        }
        Ok(out)
    }

    /// Masked-language-modeling cross-entropy over masked positions only.
    ///
    /// `mlm_logits`: `[text_len × vocab_size]`. `targets`: original token ids.
    /// `mask`: per-position boolean (true = masked, contributes to the loss).
    pub fn mlm_loss(
        &self,
        mlm_logits: &[f32],
        targets: &[u32],
        mask: &[bool],
        text_len: usize,
    ) -> MmResult<f32> {
        let v = self.cfg.vocab_size;
        if mlm_logits.len() != text_len * v {
            return Err(MultiModalError::DimensionMismatch {
                expected: text_len * v,
                got: mlm_logits.len(),
            });
        }
        if targets.len() != text_len || mask.len() != text_len {
            return Err(MultiModalError::DimensionMismatch {
                expected: text_len,
                got: targets.len().min(mask.len()),
            });
        }
        let mut total = 0.0_f32;
        let mut n = 0usize;
        for t in 0..text_len {
            if !mask[t] {
                continue;
            }
            let tgt = targets[t] as usize;
            if tgt >= v {
                return Err(MultiModalError::TokenOutOfRange {
                    token_id: targets[t],
                    vocab_size: v,
                });
            }
            let row = &mlm_logits[t * v..(t + 1) * v];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for &x in row {
                sum += (x - max).exp();
            }
            total += (max + sum.ln()) - row[tgt];
            n += 1;
        }
        if n == 0 {
            return Ok(0.0);
        }
        Ok(total / n as f32)
    }

    /// Argmax-predicted token id for each MLM position (`[text_len]`).
    pub fn mlm_predict(&self, mlm_logits: &[f32], text_len: usize) -> MmResult<Vec<usize>> {
        let v = self.cfg.vocab_size;
        if mlm_logits.len() != text_len * v {
            return Err(MultiModalError::DimensionMismatch {
                expected: text_len * v,
                got: mlm_logits.len(),
            });
        }
        let mut out = vec![0usize; text_len];
        for t in 0..text_len {
            let row = &mlm_logits[t * v..(t + 1) * v];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &x) in row.iter().enumerate() {
                if x > best_v {
                    best_v = x;
                    best = j;
                }
            }
            out[t] = best;
        }
        Ok(out)
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────────

/// L2-normalise a single row in place.
fn l2_normalise_row(row: &mut [f32]) {
    let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Soft cross-entropy over a `[batch × batch]` similarity matrix whose rows are
/// already scaled by `1/temperature`. The per-row target is
/// `(1−α)·one_hot(diag) + α·softmax(teacher_row)`; with `teacher = None` or
/// `α = 0` this reduces to standard InfoNCE against the diagonal positives.
fn soft_ce(sim: &[f32], batch: usize, teacher: Option<&[f32]>, alpha: f32) -> MmResult<f32> {
    let mut loss = 0.0_f32;
    for i in 0..batch {
        let row = &sim[i * batch..(i + 1) * batch];
        // Stable log-softmax of the student row.
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for &x in row {
            sum += (x - max).exp();
        }
        let log_sum = max + sum.ln();

        // Soft teacher target distribution for this row (if provided).
        let teacher_soft = teacher.map(|t| {
            let trow = &t[i * batch..(i + 1) * batch];
            let tmax = trow.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut tsum = 0.0_f32;
            for &x in trow {
                tsum += (x - tmax).exp();
            }
            let mut probs = vec![0.0_f32; batch];
            for (j, &x) in trow.iter().enumerate() {
                probs[j] = (x - tmax).exp() / tsum;
            }
            probs
        });

        for j in 0..batch {
            let log_p = row[j] - log_sum;
            let hard = if j == i { 1.0 } else { 0.0 };
            let target = match &teacher_soft {
                Some(p) => (1.0 - alpha) * hard + alpha * p[j],
                None => hard,
            };
            loss -= target * log_p;
        }
    }
    Ok(loss / batch as f32)
}

/// One pre-norm text self-attention encoder layer.
fn text_layer_forward(
    input: &[f32],
    seq: usize,
    cfg: &AlbefConfig,
    w: &TextLayer,
) -> MmResult<Vec<f32>> {
    let _ = softmax_rows_inplace; // shared softmax kept consistent across the crate.
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k: d / h,
        d_v: d / h,
        dropout_rate: 0.0,
    };
    let normed1 = w.ln1.forward(input, seq)?;
    let attn = CrossAttention::with_weights(attn_cfg, w.attn.clone());
    let sa = attn.forward(&normed1, &normed1, &normed1, seq, seq)?;
    let mut x: Vec<f32> = input.iter().zip(sa.iter()).map(|(a, b)| a + b).collect();
    let normed2 = w.ln2.forward(&x, seq)?;
    let ffn_out = w.ffn.forward(&normed2, seq)?;
    for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += fi;
    }
    Ok(x)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(seed: u64) -> (AlbefConfig, AlbefWeights) {
        let cfg = AlbefConfig::tiny();
        let mut rng = LcgRng::new(seed);
        let w = AlbefWeights::random(&cfg, &mut rng);
        (cfg, w)
    }

    fn make_image(cfg: &AlbefConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let n = cfg.vit.n_channels * cfg.vit.img_size * cfg.vit.img_size;
        let mut img = vec![0.0_f32; n];
        rng.fill_normal(&mut img);
        img
    }

    #[test]
    fn config_tiny_valid() {
        assert!(AlbefConfig::tiny().validate().is_ok());
    }

    // ITC similarity matrix symmetric (sim_t2i == sim_i2t^T).
    #[test]
    fn itc_similarity_is_transpose_symmetric() {
        let (cfg, w) = weights(1);
        let albef = Albef::new(cfg.clone());
        let img0 = make_image(&cfg, 10);
        let img1 = make_image(&cfg, 11);
        let t0: Vec<u32> = vec![1, 2, 3, 4];
        let t1: Vec<u32> = vec![5, 6, 7];
        let out = albef
            .itc_forward(&[&img0, &img1], &[&t0[..], &t1[..]], &w)
            .expect("value should be present");
        let b = 2;
        for i in 0..b {
            for j in 0..b {
                assert!(
                    (out.sim_i2t[i * b + j] - out.sim_t2i[j * b + i]).abs() < 1e-5,
                    "sim_t2i must be the transpose of sim_i2t"
                );
            }
        }
        // Diagonal of i2t equals diagonal of t2i (same pair similarity).
        for i in 0..b {
            assert!((out.sim_i2t[i * b + i] - out.sim_t2i[i * b + i]).abs() < 1e-5);
        }
    }

    // Contrastive loss ≥ 0.
    #[test]
    fn itc_loss_non_negative() {
        let (cfg, w) = weights(2);
        let albef = Albef::new(cfg.clone());
        let img0 = make_image(&cfg, 20);
        let img1 = make_image(&cfg, 21);
        let t0: Vec<u32> = vec![1, 2, 3];
        let t1: Vec<u32> = vec![4, 5, 6];
        let out = albef
            .itc_forward(&[&img0, &img1], &[&t0[..], &t1[..]], &w)
            .expect("value should be present");
        let loss = albef
            .itc_loss(&out, 2, None, 0.0)
            .expect("itc_loss should succeed");
        assert!(loss >= 0.0, "ITC loss must be non-negative, got {loss}");
        assert!(loss.is_finite());
        // Projected embeddings are unit-norm.
        let p = cfg.proj_dim;
        for b in 0..2 {
            let n: f32 = out.image_embed[b * p..(b + 1) * p]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((n - 1.0).abs() < 1e-4, "image embed must be unit-norm");
        }
    }

    // Fusion output shape correct.
    #[test]
    fn fusion_output_shape() {
        let (cfg, w) = weights(3);
        let albef = Albef::new(cfg.clone());
        let img = make_image(&cfg, 30);
        let txt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let img_tokens = albef
            .encode_image_tokens(&img, &w)
            .expect("encode_image_tokens should succeed");
        let txt_tokens = albef
            .encode_text_tokens(&txt, &w)
            .expect("encode_text_tokens should succeed");
        let image_len = 1 + cfg.vit.n_patches();
        let fused = albef
            .fuse(&txt_tokens, &img_tokens, txt.len(), image_len, &w)
            .expect("value should be present");
        assert_eq!(fused.len(), txt.len() * cfg.d_model);
        assert!(fused.iter().all(|v| v.is_finite()));
    }

    // ITM logit 2-way (probabilities sum to 1, both in [0,1]).
    #[test]
    fn itm_two_way_probabilities() {
        let (cfg, w) = weights(4);
        let albef = Albef::new(cfg.clone());
        let img = make_image(&cfg, 40);
        let txt: Vec<u32> = vec![2, 3, 4];
        let img_tokens = albef
            .encode_image_tokens(&img, &w)
            .expect("encode_image_tokens should succeed");
        let txt_tokens = albef
            .encode_text_tokens(&txt, &w)
            .expect("encode_text_tokens should succeed");
        let image_len = 1 + cfg.vit.n_patches();
        let fused = albef
            .fuse(&txt_tokens, &img_tokens, txt.len(), image_len, &w)
            .expect("value should be present");
        let logit = albef
            .itm_logit(&fused, &w)
            .expect("itm_logit should succeed");
        assert!(logit.is_finite());
        let probs = albef
            .itm_probs(&fused, &w)
            .expect("itm_probs should succeed");
        assert!(
            (probs[0] + probs[1] - 1.0).abs() < 1e-5,
            "ITM probs must sum to 1"
        );
        assert!((0.0..=1.0).contains(&probs[0]) && (0.0..=1.0).contains(&probs[1]));
    }

    // MLM predicts masked tokens (head shape + argmax in range); MLM loss ≥ 0.
    #[test]
    fn mlm_head_shape_and_prediction() {
        let (cfg, w) = weights(5);
        let albef = Albef::new(cfg.clone());
        let img = make_image(&cfg, 50);
        let txt: Vec<u32> = vec![1, 2, 3, 4];
        let img_tokens = albef
            .encode_image_tokens(&img, &w)
            .expect("encode_image_tokens should succeed");
        let txt_tokens = albef
            .encode_text_tokens(&txt, &w)
            .expect("encode_text_tokens should succeed");
        let image_len = 1 + cfg.vit.n_patches();
        let fused = albef
            .fuse(&txt_tokens, &img_tokens, txt.len(), image_len, &w)
            .expect("value should be present");
        let logits = albef
            .mlm_logits(&fused, txt.len(), &w)
            .expect("value should be present");
        assert_eq!(logits.len(), txt.len() * cfg.vocab_size);

        let preds = albef
            .mlm_predict(&logits, txt.len())
            .expect("value should be present");
        assert_eq!(preds.len(), txt.len());
        assert!(preds.iter().all(|&p| p < cfg.vocab_size));

        let mask = vec![true, false, true, false];
        let loss = albef
            .mlm_loss(&logits, &txt, &mask, txt.len())
            .expect("value should be present");
        assert!(
            loss >= 0.0 && loss.is_finite(),
            "MLM loss must be >= 0: {loss}"
        );

        // No mask → zero MLM loss.
        let none = vec![false; txt.len()];
        let loss0 = albef
            .mlm_loss(&logits, &txt, &none, txt.len())
            .expect("value should be present");
        assert_eq!(loss0, 0.0);
    }

    // Momentum encoder EMA moves toward the online encoder.
    #[test]
    fn momentum_ema_moves_toward_online() {
        let cfg = AlbefConfig::tiny();
        let mut rng_online = LcgRng::new(100);
        let online = AlbefWeights::random(&cfg, &mut rng_online);
        // Teacher starts as zeros, far from the online weights.
        let mut teacher = AlbefWeights::zeros(&cfg);

        // Distance before vs. after one EMA step (with m < 1 the teacher must
        // move toward the online weights, so distance strictly decreases).
        let before = l2_dist(&teacher.text_proj, &online.text_proj);
        teacher.ema_update(&online, 0.9);
        let after = l2_dist(&teacher.text_proj, &online.text_proj);
        assert!(
            after < before,
            "EMA must reduce teacher↔online distance: {before} -> {after}"
        );

        // Many steps converge the teacher toward the online weights.
        for _ in 0..200 {
            teacher.ema_update(&online, 0.9);
        }
        let converged = l2_dist(&teacher.text_proj, &online.text_proj);
        assert!(
            converged < 1e-3,
            "EMA should converge toward online: {converged}"
        );
    }

    // Momentum-distillation: soft teacher targets change the ITC loss.
    #[test]
    fn momentum_distillation_blends_soft_targets() {
        let (cfg, w) = weights(6);
        let albef = Albef::new(cfg.clone());
        let img0 = make_image(&cfg, 60);
        let img1 = make_image(&cfg, 61);
        let t0: Vec<u32> = vec![1, 2, 3];
        let t1: Vec<u32> = vec![4, 5, 6];
        let out = albef
            .itc_forward(&[&img0, &img1], &[&t0[..], &t1[..]], &w)
            .expect("value should be present");
        // Build a teacher with different weights → different similarities.
        let mut rng_t = LcgRng::new(999);
        let w_teacher = AlbefWeights::random(&cfg, &mut rng_t);
        let out_teacher = albef
            .itc_forward(&[&img0, &img1], &[&t0[..], &t1[..]], &w_teacher)
            .expect("value should be present");

        let hard = albef
            .itc_loss(&out, 2, None, 0.0)
            .expect("itc_loss should succeed");
        let distilled = albef
            .itc_loss(&out, 2, Some(&out_teacher), 0.4)
            .expect("value should be present");
        assert!(hard.is_finite() && distilled.is_finite());
        assert!(
            (hard - distilled).abs() > 1e-6,
            "soft distillation targets must change the loss: {hard} vs {distilled}"
        );
    }

    #[test]
    fn encode_text_rejects_out_of_range_token() {
        let (cfg, w) = weights(7);
        let albef = Albef::new(cfg.clone());
        let txt: Vec<u32> = vec![1, cfg.vocab_size as u32 + 1];
        let err = albef.encode_text_tokens(&txt, &w).unwrap_err();
        assert!(matches!(err, MultiModalError::TokenOutOfRange { .. }));
    }

    #[test]
    fn itc_forward_rejects_mismatched_batch() {
        let (cfg, w) = weights(8);
        let albef = Albef::new(cfg.clone());
        let img = make_image(&cfg, 70);
        let t0: Vec<u32> = vec![1, 2];
        let err = albef
            .itc_forward(&[&img], &[&t0[..], &t0[..]], &w)
            .unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let (cfg, w1) = weights(33);
        let (_, w2) = weights(33);
        let albef = Albef::new(cfg.clone());
        let txt: Vec<u32> = vec![1, 2, 3];
        let a = albef
            .encode_text_tokens(&txt, &w1)
            .expect("encode_text_tokens should succeed");
        let b = albef
            .encode_text_tokens(&txt, &w2)
            .expect("encode_text_tokens should succeed");
        assert_eq!(a, b);
    }

    /// Euclidean distance between two equal-length vectors.
    fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}
