//! SAM — a compact, faithful CPU reference of the *Segment Anything Model*
//! (Kirillov et al. 2023, *"Segment Anything"*).
//!
//! The model has three cooperating parts, all implemented here with real
//! computation (no shape-only stubs):
//!
//! 1. **Image encoder** — a ViT patch-embedder followed by a couple of
//!    transformer blocks, reshaped into a dense image embedding `(C, H', W')`.
//!    Reuses [`crate::patch_embed::PatchEmbed`] and [`crate::vit::ViTBlock`].
//! 2. **Prompt encoder** — points / boxes are turned into **sparse** embeddings
//!    via a random-Fourier positional encoding plus learned per-type embeddings;
//!    a coarse mask is turned into a **dense** embedding added to the image
//!    embedding.
//! 3. **Two-way transformer mask decoder** — output tokens and the image
//!    embedding attend to *each other* in alternating directions (tokens→image
//!    and image→tokens), after which the mask tokens form a spatial filter that
//!    is applied to the up-scaled image embedding to produce masks, and an
//!    IoU-token MLP predicts a quality score per mask.
//!
//! ## Tensor layout
//! Token sequences are flat `[n_tokens · embed_dim]` row-major. Dense maps use
//! the channel-major [`FeatureMap`] layout (`[C, H, W]`).

use crate::{
    error::{VisionError, VisionResult},
    fpn::top_down::FeatureMap,
    handle::LcgRng,
    patch_embed::{PatchEmbed, PatchEmbedConfig, pos_2d_sincos},
    vit::vit_block::{gelu_exact, layer_norm, linear, softmax_rows},
    vit::{ViTBlock, ViTBlockConfig},
};

use std::f32::consts::PI;

// ─── Small reusable parameter blocks ───────────────────────────────────────────

/// Fill a buffer with `N(0, scale)` samples.
fn filled(n: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

/// A learned LayerNorm affine (`γ` ones, `β` zeros), applied per row.
#[derive(Debug, Clone)]
struct LayerNormParams {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNormParams {
    fn new(d: usize) -> Self {
        Self {
            weight: vec![1.0f32; d],
            bias: vec![0.0f32; d],
        }
    }

    fn apply(&self, x: &[f32], n: usize, d: usize) -> Vec<f32> {
        layer_norm(x, &self.weight, &self.bias, n, d, 1e-6)
    }
}

/// A two-layer MLP `Linear → GELU → Linear` (the transformer / hypernetwork FFN).
#[derive(Debug, Clone)]
struct Mlp {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_in: usize,
    hidden: usize,
    d_out: usize,
}

impl Mlp {
    fn new(d_in: usize, hidden: usize, d_out: usize, rng: &mut LcgRng) -> Self {
        Self {
            w1: filled(hidden * d_in, (2.0 / d_in as f32).sqrt(), rng),
            b1: vec![0.0f32; hidden],
            w2: filled(d_out * hidden, (2.0 / hidden as f32).sqrt(), rng),
            b2: vec![0.0f32; d_out],
            d_in,
            hidden,
            d_out,
        }
    }

    /// Apply to a `[batch · d_in]` buffer → `[batch · d_out]`.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut h = linear(x, &self.w1, &self.b1, self.d_in, self.hidden);
        for v in &mut h {
            *v = gelu_exact(*v);
        }
        linear(&h, &self.w2, &self.b2, self.hidden, self.d_out)
    }
}

// ─── Multi-head attention with explicit weights ────────────────────────────────

/// General multi-head attention with **separate** Q / K / V projections so that
/// the query and key/value sequences may differ (cross-attention). The forward
/// pass returns both the output and the per-head attention weights, which the
/// two-way transformer tests use to verify the softmax normalisation.
pub struct MultiHeadAttention {
    wq: Vec<f32>,
    bq: Vec<f32>,
    wk: Vec<f32>,
    bk: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    wo: Vec<f32>,
    bo: Vec<f32>,
    embed_dim: usize,
    n_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Construct an attention module.
    ///
    /// # Errors
    /// - [`VisionError::InvalidNumHeads`] if `n_heads == 0`.
    /// - [`VisionError::HeadDimMismatch`] if `n_heads` does not divide
    ///   `embed_dim`.
    pub fn new(embed_dim: usize, n_heads: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if embed_dim % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch { n_heads, embed_dim });
        }
        let scale = 1.0 / (embed_dim as f32).sqrt();
        Ok(Self {
            wq: filled(embed_dim * embed_dim, scale, rng),
            bq: vec![0.0f32; embed_dim],
            wk: filled(embed_dim * embed_dim, scale, rng),
            bk: vec![0.0f32; embed_dim],
            wv: filled(embed_dim * embed_dim, scale, rng),
            bv: vec![0.0f32; embed_dim],
            wo: filled(embed_dim * embed_dim, scale, rng),
            bo: vec![0.0f32; embed_dim],
            embed_dim,
            n_heads,
            head_dim: embed_dim / n_heads,
        })
    }

    /// Attention forward. `q_in` is `[n_q · e]`; `k_in`, `v_in` are `[n_k · e]`.
    ///
    /// Returns `(output [n_q · e], weights [n_heads · n_q · n_k])`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] on inconsistent input lengths.
    /// - [`VisionError::NonFinite`] if the output is non-finite.
    pub fn forward(
        &self,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        n_q: usize,
        n_k: usize,
    ) -> VisionResult<(Vec<f32>, Vec<f32>)> {
        let e = self.embed_dim;
        if q_in.len() != n_q * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_q * e,
                got: q_in.len(),
            });
        }
        if k_in.len() != n_k * e || v_in.len() != n_k * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_k * e,
                got: k_in.len(),
            });
        }

        let q = linear(q_in, &self.wq, &self.bq, e, e);
        let k = linear(k_in, &self.wk, &self.bk, e, e);
        let v = linear(v_in, &self.wv, &self.bv, e, e);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut concat = vec![0.0f32; n_q * e];
        let mut weights = vec![0.0f32; self.n_heads * n_q * n_k];
        let mut scores = vec![0.0f32; n_q * n_k];

        for h in 0..self.n_heads {
            let off = h * self.head_dim;
            for i in 0..n_q {
                for j in 0..n_k {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[i * e + off + d] * k[j * e + off + d];
                    }
                    scores[i * n_k + j] = dot * scale;
                }
            }
            softmax_rows(&mut scores, n_q, n_k);

            // Record weights and aggregate values.
            for i in 0..n_q {
                let w_row = (h * n_q + i) * n_k;
                let s_row = i * n_k;
                weights[w_row..w_row + n_k].copy_from_slice(&scores[s_row..s_row + n_k]);
                for d in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..n_k {
                        acc += scores[s_row + j] * v[j * e + off + d];
                    }
                    concat[i * e + off + d] = acc;
                }
            }
        }

        let out = linear(&concat, &self.wo, &self.bo, e, e);
        if out.iter().any(|x| !x.is_finite()) {
            return Err(VisionError::NonFinite("SAM attention output"));
        }
        Ok((out, weights))
    }
}

// ─── Two-way attention block ───────────────────────────────────────────────────

/// Output of a [`TwoWayAttentionBlock`] forward pass.
#[derive(Debug, Clone)]
pub struct TwoWayBlockOutput {
    /// Updated token embeddings `[n_tokens · e]`.
    pub tokens: Vec<f32>,
    /// Updated image embeddings `[n_image · e]`.
    pub image: Vec<f32>,
    /// Token self-attention weights `[n_heads · n_tokens · n_tokens]`.
    pub self_weights: Vec<f32>,
    /// Token→image cross-attention weights `[n_heads · n_tokens · n_image]`.
    pub token_to_image_weights: Vec<f32>,
    /// Image→token cross-attention weights `[n_heads · n_image · n_tokens]`.
    pub image_to_token_weights: Vec<f32>,
}

/// A two-way attention block: tokens attend to themselves and to the image, then
/// the image attends back to the tokens — so **both** representations are
/// updated in a single block.
pub struct TwoWayAttentionBlock {
    self_attn: MultiHeadAttention,
    cross_token_to_image: MultiHeadAttention,
    cross_image_to_token: MultiHeadAttention,
    mlp: Mlp,
    norm1: LayerNormParams,
    norm2: LayerNormParams,
    norm3: LayerNormParams,
    norm4: LayerNormParams,
    embed_dim: usize,
}

impl TwoWayAttentionBlock {
    /// Construct a block.
    ///
    /// # Errors
    /// Propagates [`MultiHeadAttention::new`] validation.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        mlp_dim: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(embed_dim, n_heads, rng)?,
            cross_token_to_image: MultiHeadAttention::new(embed_dim, n_heads, rng)?,
            cross_image_to_token: MultiHeadAttention::new(embed_dim, n_heads, rng)?,
            mlp: Mlp::new(embed_dim, mlp_dim, embed_dim, rng),
            norm1: LayerNormParams::new(embed_dim),
            norm2: LayerNormParams::new(embed_dim),
            norm3: LayerNormParams::new(embed_dim),
            norm4: LayerNormParams::new(embed_dim),
            embed_dim,
        })
    }

    /// Forward pass.
    ///
    /// `tokens`/`query_pe` are `[n_t · e]`; `image`/`key_pe` are `[n_i · e]`.
    /// The positional encodings are *added to queries and keys only* (never to
    /// the values), exactly as in SAM.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] on inconsistent lengths.
    /// - Propagates attention errors.
    pub fn forward(
        &self,
        tokens: &[f32],
        image: &[f32],
        query_pe: &[f32],
        key_pe: &[f32],
    ) -> VisionResult<TwoWayBlockOutput> {
        let e = self.embed_dim;
        if tokens.len() != query_pe.len() {
            return Err(VisionError::DimensionMismatch {
                expected: tokens.len(),
                got: query_pe.len(),
            });
        }
        if image.len() != key_pe.len() {
            return Err(VisionError::DimensionMismatch {
                expected: image.len(),
                got: key_pe.len(),
            });
        }
        if tokens.len() % e != 0 || image.len() % e != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: e,
                got: tokens.len() % e,
            });
        }
        let n_t = tokens.len() / e;
        let n_i = image.len() / e;

        // 1) Token self-attention (Q = K = tokens + pe, V = tokens).
        let q = add_vec(tokens, query_pe);
        let (sa, self_w) = self_attn_or_err(&self.self_attn, &q, tokens, n_t)?;
        let mut tokens_cur = add_vec(tokens, &sa);
        tokens_cur = self.norm1.apply(&tokens_cur, n_t, e);

        // 2) Token → image cross-attention.
        let q = add_vec(&tokens_cur, query_pe);
        let k = add_vec(image, key_pe);
        let (ca, t2i_w) = self.cross_token_to_image.forward(&q, &k, image, n_t, n_i)?;
        tokens_cur = add_vec(&tokens_cur, &ca);
        tokens_cur = self.norm2.apply(&tokens_cur, n_t, e);

        // 3) Token MLP.
        let m = self.mlp.apply(&tokens_cur);
        tokens_cur = add_vec(&tokens_cur, &m);
        tokens_cur = self.norm3.apply(&tokens_cur, n_t, e);

        // 4) Image → token cross-attention (image is the query now).
        let q = add_vec(image, key_pe);
        let k = add_vec(&tokens_cur, query_pe);
        let (ca2, i2t_w) = self
            .cross_image_to_token
            .forward(&q, &k, &tokens_cur, n_i, n_t)?;
        let mut image_cur = add_vec(image, &ca2);
        image_cur = self.norm4.apply(&image_cur, n_i, e);

        Ok(TwoWayBlockOutput {
            tokens: tokens_cur,
            image: image_cur,
            self_weights: self_w,
            token_to_image_weights: t2i_w,
            image_to_token_weights: i2t_w,
        })
    }
}

/// Self-attention helper (Q/K share, V is the raw tokens).
fn self_attn_or_err(
    attn: &MultiHeadAttention,
    qk: &[f32],
    v: &[f32],
    n: usize,
) -> VisionResult<(Vec<f32>, Vec<f32>)> {
    attn.forward(qk, qk, v, n, n)
}

// ─── Two-way transformer ───────────────────────────────────────────────────────

/// Stacks several [`TwoWayAttentionBlock`]s, followed by a final token→image
/// attention as in SAM's `TwoWayTransformer`.
pub struct TwoWayTransformer {
    blocks: Vec<TwoWayAttentionBlock>,
    final_attn: MultiHeadAttention,
    final_norm: LayerNormParams,
    embed_dim: usize,
}

impl TwoWayTransformer {
    fn new(
        embed_dim: usize,
        n_heads: usize,
        depth: usize,
        mlp_dim: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<Self> {
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(TwoWayAttentionBlock::new(embed_dim, n_heads, mlp_dim, rng)?);
        }
        Ok(Self {
            blocks,
            final_attn: MultiHeadAttention::new(embed_dim, n_heads, rng)?,
            final_norm: LayerNormParams::new(embed_dim),
            embed_dim,
        })
    }

    /// Run the transformer. `image`/`image_pe` are `[n_i · e]`, `point_tokens`
    /// is `[n_t · e]`. The point tokens double as the query positional encoding,
    /// exactly as in SAM.
    ///
    /// Returns `(tokens [n_t · e], image [n_i · e])`.
    ///
    /// # Errors
    /// Propagates block / attention errors.
    pub fn forward(
        &self,
        image: &[f32],
        image_pe: &[f32],
        point_tokens: &[f32],
    ) -> VisionResult<(Vec<f32>, Vec<f32>)> {
        let e = self.embed_dim;
        let n_t = point_tokens.len() / e;
        let n_i = image.len() / e;
        let query_pe = point_tokens.to_vec();

        let mut tokens = point_tokens.to_vec();
        let mut img = image.to_vec();
        for block in &self.blocks {
            let out = block.forward(&tokens, &img, &query_pe, image_pe)?;
            tokens = out.tokens;
            img = out.image;
        }

        // Final token → image attention.
        let q = add_vec(&tokens, &query_pe);
        let k = add_vec(&img, image_pe);
        let (attn, _w) = self.final_attn.forward(&q, &k, &img, n_t, n_i)?;
        tokens = add_vec(&tokens, &attn);
        tokens = self.final_norm.apply(&tokens, n_t, e);

        Ok((tokens, img))
    }
}

// ─── Image encoder ─────────────────────────────────────────────────────────────

/// ViT image encoder producing a dense image embedding.
pub struct ImageEncoder {
    patch_embed: PatchEmbed,
    pos_embed: Vec<f32>,
    blocks: Vec<ViTBlock>,
    neck_w: Vec<f32>,
    neck_b: Vec<f32>,
    grid: usize,
    embed_dim: usize,
}

impl ImageEncoder {
    fn new(cfg: &SamConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let pe_cfg =
            PatchEmbedConfig::new(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)?;
        let grid = pe_cfg.grid_size();
        let patch_embed = PatchEmbed::new(pe_cfg, rng);
        let pos_embed = pos_2d_sincos(grid, grid, cfg.embed_dim)?;
        let block_cfg = ViTBlockConfig::new(cfg.embed_dim, cfg.enc_heads, cfg.enc_mlp_ratio)?;
        let mut blocks = Vec::with_capacity(cfg.enc_depth);
        for _ in 0..cfg.enc_depth {
            blocks.push(ViTBlock::new(block_cfg.clone(), rng));
        }
        let scale = 1.0 / (cfg.embed_dim as f32).sqrt();
        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            neck_w: filled(cfg.embed_dim * cfg.embed_dim, scale, rng),
            neck_b: vec![0.0f32; cfg.embed_dim],
            grid,
            embed_dim: cfg.embed_dim,
        })
    }

    /// Encode a flat CHW image into a dense `(embed_dim, grid, grid)` map.
    ///
    /// # Errors
    /// Propagates patch-embed / block errors.
    pub fn forward(&self, image: &[f32]) -> VisionResult<FeatureMap> {
        let e = self.embed_dim;
        let n_patches = self.grid * self.grid;
        let mut tokens = self.patch_embed.forward(image)?;
        // Add positional embedding.
        for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
            *t += *p;
        }
        for block in &self.blocks {
            tokens = block.forward(&tokens, n_patches)?;
        }
        // 1×1 neck (per-token linear).
        let tokens = linear(&tokens, &self.neck_w, &self.neck_b, e, e);
        // Reshape [n_patches, e] (patch-major) → (e, grid, grid).
        let chw = tokens_to_chw(&tokens, e, self.grid, self.grid);
        FeatureMap::new(chw, e, self.grid, self.grid)
    }
}

// ─── Positional encoding (random Fourier) ──────────────────────────────────────

/// SAM's `PositionEmbeddingRandom`: a fixed Gaussian matrix `[2 · num_freq]` maps
/// a normalised coordinate `(x, y) ∈ [0, 1]²` to `[sin, cos]` Fourier features.
pub struct PositionEmbeddingRandom {
    gaussian: Vec<f32>,
    num_freq: usize,
}

impl PositionEmbeddingRandom {
    fn new(num_freq: usize, rng: &mut LcgRng) -> Self {
        // Scale 1.0 keeps the projection well-conditioned for the tiny ref.
        Self {
            gaussian: filled(2 * num_freq, 1.0, rng),
            num_freq,
        }
    }

    /// Encode a normalised `(x, y)` into a `[2 · num_freq]` embedding.
    fn encode_point(&self, x: f32, y: f32) -> Vec<f32> {
        let nf = self.num_freq;
        let mut out = vec![0.0f32; 2 * nf];
        for f in 0..nf {
            let proj = 2.0 * PI * (x * self.gaussian[f] + y * self.gaussian[nf + f]);
            out[f] = proj.sin();
            out[nf + f] = proj.cos();
        }
        out
    }

    /// Encode an `h × w` grid into a dense `(2·num_freq, h, w)` positional map,
    /// using cell-centre coordinates normalised to `[0, 1]`.
    fn encode_grid(&self, h: usize, w: usize) -> Vec<f32> {
        let dim = 2 * self.num_freq;
        let mut out = vec![0.0f32; dim * h * w];
        for i in 0..h {
            for j in 0..w {
                let x = (j as f32 + 0.5) / w as f32;
                let y = (i as f32 + 0.5) / h as f32;
                let enc = self.encode_point(x, y);
                for (c, &val) in enc.iter().enumerate() {
                    out[(c * h + i) * w + j] = val;
                }
            }
        }
        out
    }
}

// ─── Prompt encoder ────────────────────────────────────────────────────────────

/// Encodes geometric prompts (points / boxes) into sparse embeddings and a
/// coarse mask into a dense embedding.
pub struct PromptEncoder {
    pe_layer: PositionEmbeddingRandom,
    /// Learned per-label point embeddings: `[2 · e]` (index 0 = background,
    /// 1 = foreground).
    point_embeddings: Vec<f32>,
    /// Learned box-corner embeddings: `[2 · e]` (top-left, bottom-right).
    corner_embeddings: Vec<f32>,
    /// Embedding for a padding / "not a point" entry: `[e]`.
    not_a_point: Vec<f32>,
    /// Embedding used when no mask prompt is supplied: `[e]`.
    no_mask_embed: Vec<f32>,
    /// 1×1 projection `1 → e` for the coarse mask.
    mask_w: Vec<f32>,
    mask_b: Vec<f32>,
    embed_dim: usize,
    grid: usize,
    input_size: f32,
}

impl PromptEncoder {
    fn new(cfg: &SamConfig, rng: &mut LcgRng) -> Self {
        let e = cfg.embed_dim;
        let scale = 1.0 / (e as f32).sqrt();
        Self {
            pe_layer: PositionEmbeddingRandom::new(e / 2, rng),
            point_embeddings: filled(2 * e, 0.1, rng),
            corner_embeddings: filled(2 * e, 0.1, rng),
            not_a_point: filled(e, 0.1, rng),
            no_mask_embed: filled(e, 0.1, rng),
            mask_w: filled(e, scale, rng),
            mask_b: vec![0.0f32; e],
            embed_dim: e,
            grid: cfg.img_size / cfg.patch_size,
            input_size: cfg.img_size as f32,
        }
    }

    /// Image-grid positional encoding used by the decoder, `(e, grid, grid)`.
    #[must_use]
    pub fn dense_positional_encoding(&self) -> Vec<f32> {
        self.pe_layer.encode_grid(self.grid, self.grid)
    }

    /// Encode point prompts.
    ///
    /// `coords` is `[n · 2]` pixel coordinates; `labels[i]` is `1` for a
    /// foreground point, `0` for background, and any negative value for a
    /// padding ("not a point") entry. Returns sparse embeddings `[n · e]`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `coords.len() != 2·labels.len()`.
    pub fn encode_points(&self, coords: &[f32], labels: &[i32]) -> VisionResult<Vec<f32>> {
        let n = labels.len();
        if coords.len() != n * 2 {
            return Err(VisionError::DimensionMismatch {
                expected: n * 2,
                got: coords.len(),
            });
        }
        let e = self.embed_dim;
        let mut out = vec![0.0f32; n * e];
        for p in 0..n {
            let x = coords[p * 2] / self.input_size;
            let y = coords[p * 2 + 1] / self.input_size;
            let pe = self.pe_layer.encode_point(x, y);
            let dst = &mut out[p * e..(p + 1) * e];
            if labels[p] < 0 {
                // Padding point: zeroed PE plus the "not a point" embedding.
                for (d, slot) in dst.iter_mut().enumerate() {
                    *slot = self.not_a_point[d];
                }
            } else {
                let label_off = if labels[p] >= 1 { e } else { 0 };
                for (d, slot) in dst.iter_mut().enumerate() {
                    *slot = pe[d] + self.point_embeddings[label_off + d];
                }
            }
        }
        Ok(out)
    }

    /// Encode a box prompt `[x1, y1, x2, y2]` (pixels) into two corner
    /// embeddings `[2 · e]`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `box4.len() != 4`.
    pub fn encode_box(&self, box4: &[f32]) -> VisionResult<Vec<f32>> {
        if box4.len() != 4 {
            return Err(VisionError::DimensionMismatch {
                expected: 4,
                got: box4.len(),
            });
        }
        let e = self.embed_dim;
        let corners = [(box4[0], box4[1], 0usize), (box4[2], box4[3], 1usize)];
        let mut out = vec![0.0f32; 2 * e];
        for (idx, &(cx, cy, corner)) in corners.iter().enumerate() {
            let pe = self
                .pe_layer
                .encode_point(cx / self.input_size, cy / self.input_size);
            let dst = &mut out[idx * e..(idx + 1) * e];
            for (d, slot) in dst.iter_mut().enumerate() {
                *slot = pe[d] + self.corner_embeddings[corner * e + d];
            }
        }
        Ok(out)
    }

    /// Encode a coarse mask `[grid · grid]` into a dense embedding
    /// `(e, grid, grid)`. When `mask` is `None`, the learned `no_mask` embedding
    /// is broadcast over the grid.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `mask` is `Some` but not
    ///   `grid · grid` long.
    pub fn encode_mask(&self, mask: Option<&[f32]>) -> VisionResult<Vec<f32>> {
        let e = self.embed_dim;
        let hw = self.grid * self.grid;
        let mut out = vec![0.0f32; e * hw];
        match mask {
            None => {
                for c in 0..e {
                    let val = self.no_mask_embed[c];
                    for p in 0..hw {
                        out[c * hw + p] = val;
                    }
                }
            }
            Some(m) => {
                if m.len() != hw {
                    return Err(VisionError::DimensionMismatch {
                        expected: hw,
                        got: m.len(),
                    });
                }
                // 1×1 conv with one input channel: out[c,p] = w[c]·m[p] + b[c].
                for c in 0..e {
                    let w = self.mask_w[c];
                    let b = self.mask_b[c];
                    for p in 0..hw {
                        out[c * hw + p] = w * m[p] + b;
                    }
                }
            }
        }
        Ok(out)
    }
}

// ─── Mask decoder ──────────────────────────────────────────────────────────────

/// Predicted masks plus their IoU quality scores.
#[derive(Debug, Clone)]
pub struct MaskPrediction {
    /// Masks: flat `[n_mask · (2·grid) · (2·grid)]`.
    pub masks: Vec<f32>,
    /// Per-mask predicted IoU scores: `[n_mask]`.
    pub iou: Vec<f32>,
    /// Number of masks.
    pub n_mask: usize,
    /// Mask spatial height (= 2 · grid).
    pub height: usize,
    /// Mask spatial width (= 2 · grid).
    pub width: usize,
}

/// The SAM mask decoder.
pub struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Vec<f32>,
    mask_tokens: Vec<f32>,
    upscale_w: Vec<f32>,
    upscale_b: Vec<f32>,
    hypernets: Vec<Mlp>,
    iou_head: Mlp,
    n_mask: usize,
    embed_dim: usize,
}

impl MaskDecoder {
    fn new(cfg: &SamConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let e = cfg.embed_dim;
        let transformer =
            TwoWayTransformer::new(e, cfg.dec_heads, cfg.dec_depth, cfg.dec_mlp_dim, rng)?;
        let scale = 1.0 / (e as f32).sqrt();
        let hypernets = (0..cfg.n_mask).map(|_| Mlp::new(e, e, e, rng)).collect();
        Ok(Self {
            transformer,
            iou_token: filled(e, 0.02, rng),
            mask_tokens: filled(cfg.n_mask * e, 0.02, rng),
            upscale_w: filled(e * e, scale, rng),
            upscale_b: vec![0.0f32; e],
            hypernets,
            iou_head: Mlp::new(e, e, cfg.n_mask, rng),
            n_mask: cfg.n_mask,
            embed_dim: e,
        })
    }

    /// Decode masks from an image embedding and prompt embeddings.
    ///
    /// - `image_embedding`: dense `(e, grid, grid)`.
    /// - `image_pe`: dense positional encoding `(e, grid, grid)` (flat).
    /// - `sparse_prompt`: `[n_sparse · e]` (may be empty).
    /// - `dense_prompt`: dense `(e, grid, grid)` (flat).
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] on shape mismatches.
    /// - [`VisionError::NonFinite`] if any output is non-finite.
    pub fn forward(
        &self,
        image_embedding: &FeatureMap,
        image_pe: &[f32],
        sparse_prompt: &[f32],
        dense_prompt: &[f32],
    ) -> VisionResult<MaskPrediction> {
        let e = self.embed_dim;
        let (h, w) = (image_embedding.height, image_embedding.width);
        let hw = h * w;
        if image_embedding.channels != e || image_embedding.data.len() != e * hw {
            return Err(VisionError::DimensionMismatch {
                expected: e * hw,
                got: image_embedding.data.len(),
            });
        }
        if image_pe.len() != e * hw || dense_prompt.len() != e * hw {
            return Err(VisionError::DimensionMismatch {
                expected: e * hw,
                got: image_pe.len(),
            });
        }
        if sparse_prompt.len() % e != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: e,
                got: sparse_prompt.len() % e,
            });
        }
        let n_sparse = sparse_prompt.len() / e;

        // Assemble output tokens: [iou_token, mask_tokens..., sparse...].
        let n_tokens = 1 + self.n_mask + n_sparse;
        let mut tokens = Vec::with_capacity(n_tokens * e);
        tokens.extend_from_slice(&self.iou_token);
        tokens.extend_from_slice(&self.mask_tokens);
        tokens.extend_from_slice(sparse_prompt);

        // Inject the dense prompt into the image embedding, then flatten to
        // token order [hw, e].
        let mut src_chw = image_embedding.data.clone();
        for (s, d) in src_chw.iter_mut().zip(dense_prompt.iter()) {
            *s += *d;
        }
        let src_tokens = chw_to_tokens(&src_chw, e, h, w);
        let pe_tokens = chw_to_tokens(image_pe, e, h, w);

        // Two-way transformer.
        let (tokens_out, src_out) = self.transformer.forward(&src_tokens, &pe_tokens, &tokens)?;

        // Reshape the updated image tokens back to (e, h, w) and upscale 2×.
        let src_img = tokens_to_chw(&src_out, e, h, w);
        let up = upsample2x_chw(&src_img, e, h, w);
        let (uh, uw) = (h * 2, w * 2);
        // 1×1 conv on the upscaled features (per-pixel linear e → e).
        let up_tokens = chw_to_tokens(&up, e, uh, uw);
        let up_tokens = linear(&up_tokens, &self.upscale_w, &self.upscale_b, e, e);
        let up = tokens_to_chw(&up_tokens, e, uh, uw);

        // Hypernetwork filters → spatial masks.
        let mut masks = vec![0.0f32; self.n_mask * uh * uw];
        for m in 0..self.n_mask {
            let token = &tokens_out[(1 + m) * e..(2 + m) * e];
            let filter = self.hypernets[m].apply(token); // [e]
            for p in 0..(uh * uw) {
                let mut acc = 0.0f32;
                for c in 0..e {
                    acc += filter[c] * up[c * uh * uw + p];
                }
                masks[m * uh * uw + p] = acc;
            }
        }

        // IoU prediction from the IoU token.
        let iou_token = &tokens_out[0..e];
        let iou = self.iou_head.apply(iou_token);

        if masks.iter().chain(iou.iter()).any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("SAM mask decoder output"));
        }

        Ok(MaskPrediction {
            masks,
            iou,
            n_mask: self.n_mask,
            height: uh,
            width: uw,
        })
    }
}

// ─── Sam (top level) ───────────────────────────────────────────────────────────

/// SAM hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SamConfig {
    /// Input image channels.
    pub in_chans: usize,
    /// Square input size.
    pub img_size: usize,
    /// Patch size of the image encoder.
    pub patch_size: usize,
    /// Embedding dimension (shared across encoder / prompt / decoder).
    pub embed_dim: usize,
    /// Encoder transformer depth.
    pub enc_depth: usize,
    /// Encoder attention heads.
    pub enc_heads: usize,
    /// Encoder MLP ratio.
    pub enc_mlp_ratio: usize,
    /// Decoder two-way transformer depth.
    pub dec_depth: usize,
    /// Decoder attention heads.
    pub dec_heads: usize,
    /// Decoder MLP hidden dimension.
    pub dec_mlp_dim: usize,
    /// Number of output mask tokens.
    pub n_mask: usize,
}

impl SamConfig {
    /// Create and validate a configuration.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `embed_dim` is 0 or odd.
    /// - [`VisionError::InvalidPatchSize`] if `patch_size` does not divide
    ///   `img_size`.
    /// - [`VisionError::HeadDimMismatch`] if a head count does not divide
    ///   `embed_dim`.
    /// - [`VisionError::EmptyInput`] if `n_mask == 0`.
    pub fn new(
        in_chans: usize,
        img_size: usize,
        patch_size: usize,
        embed_dim: usize,
        enc_depth: usize,
        enc_heads: usize,
        enc_mlp_ratio: usize,
        dec_depth: usize,
        dec_heads: usize,
        dec_mlp_dim: usize,
        n_mask: usize,
    ) -> VisionResult<Self> {
        if embed_dim == 0 || embed_dim % 2 != 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if patch_size == 0 || img_size % patch_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size,
                img_size,
            });
        }
        if enc_heads == 0 || embed_dim % enc_heads != 0 {
            return Err(VisionError::HeadDimMismatch {
                n_heads: enc_heads,
                embed_dim,
            });
        }
        if dec_heads == 0 || embed_dim % dec_heads != 0 {
            return Err(VisionError::HeadDimMismatch {
                n_heads: dec_heads,
                embed_dim,
            });
        }
        if n_mask == 0 {
            return Err(VisionError::EmptyInput("sam n_mask"));
        }
        Ok(Self {
            in_chans,
            img_size,
            patch_size,
            embed_dim,
            enc_depth,
            enc_heads,
            enc_mlp_ratio,
            dec_depth,
            dec_heads,
            dec_mlp_dim,
            n_mask,
        })
    }

    /// A tiny configuration for unit tests: 3×32×32 input, patch 8, embed 16,
    /// 2 encoder blocks (2 heads), 2-deep decoder (2 heads), 3 masks.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            in_chans: 3,
            img_size: 32,
            patch_size: 8,
            embed_dim: 16,
            enc_depth: 2,
            enc_heads: 2,
            enc_mlp_ratio: 2,
            dec_depth: 2,
            dec_heads: 2,
            dec_mlp_dim: 32,
            n_mask: 3,
        }
    }
}

/// The full Segment Anything Model.
pub struct Sam {
    cfg: SamConfig,
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
}

impl Sam {
    /// Build SAM with randomly-initialised weights.
    ///
    /// # Errors
    /// Propagates configuration validation.
    pub fn new(cfg: SamConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let image_encoder = ImageEncoder::new(&cfg, rng)?;
        let prompt_encoder = PromptEncoder::new(&cfg, rng);
        let mask_decoder = MaskDecoder::new(&cfg, rng)?;
        Ok(Self {
            cfg,
            image_encoder,
            prompt_encoder,
            mask_decoder,
        })
    }

    /// Read-only configuration access.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &SamConfig {
        &self.cfg
    }

    /// Read-only access to the prompt encoder.
    #[must_use]
    #[inline]
    pub fn prompt_encoder(&self) -> &PromptEncoder {
        &self.prompt_encoder
    }

    /// Encode an image into its dense embedding `(embed_dim, grid, grid)`.
    ///
    /// # Errors
    /// Propagates encoder errors.
    pub fn encode_image(&self, image: &[f32]) -> VisionResult<FeatureMap> {
        self.image_encoder.forward(image)
    }

    /// End-to-end prediction from an image and a set of point prompts (plus an
    /// optional coarse mask prompt).
    ///
    /// # Errors
    /// Propagates encoder / prompt / decoder errors.
    pub fn predict(
        &self,
        image: &[f32],
        point_coords: &[f32],
        point_labels: &[i32],
        mask: Option<&[f32]>,
    ) -> VisionResult<MaskPrediction> {
        let embedding = self.encode_image(image)?;
        let sparse = self
            .prompt_encoder
            .encode_points(point_coords, point_labels)?;
        let dense = self.prompt_encoder.encode_mask(mask)?;
        let image_pe = self.prompt_encoder.dense_positional_encoding();
        self.mask_decoder
            .forward(&embedding, &image_pe, &sparse, &dense)
    }
}

// ─── Shared helpers ────────────────────────────────────────────────────────────

/// Element-wise sum of two equal-length buffers.
fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// `(C, H, W)` → `[H·W, C]` (token-major).
fn chw_to_tokens(chw: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    let mut out = vec![0.0f32; hw * c];
    for ch in 0..c {
        for p in 0..hw {
            out[p * c + ch] = chw[ch * hw + p];
        }
    }
    out
}

/// `[H·W, C]` (token-major) → `(C, H, W)`.
fn tokens_to_chw(tokens: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    let mut out = vec![0.0f32; c * hw];
    for p in 0..hw {
        for ch in 0..c {
            out[ch * hw + p] = tokens[p * c + ch];
        }
    }
    out
}

/// Nearest-neighbour 2× upsampling of a `(C, H, W)` buffer.
fn upsample2x_chw(chw: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h * 2, w * 2);
    let mut out = vec![0.0f32; c * h2 * w2];
    for ch in 0..c {
        for i in 0..h {
            for j in 0..w {
                let v = chw[(ch * h + i) * w + j];
                let oi = i * 2;
                let oj = j * 2;
                out[(ch * h2 + oi) * w2 + oj] = v;
                out[(ch * h2 + oi) * w2 + oj + 1] = v;
                out[(ch * h2 + oi + 1) * w2 + oj] = v;
                out[(ch * h2 + oi + 1) * w2 + oj + 1] = v;
            }
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_image(cfg: &SamConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut img = vec![0.0f32; cfg.in_chans * cfg.img_size * cfg.img_size];
        rng.fill_normal(&mut img);
        img
    }

    // ── Config ─────────────────────────────────────────────────────────────────

    #[test]
    fn config_tiny_valid() {
        let cfg = SamConfig::tiny();
        assert_eq!(cfg.embed_dim, 16);
        assert_eq!(cfg.n_mask, 3);
    }

    #[test]
    fn config_bad_heads_errors() {
        // 16 % 3 != 0
        let r = SamConfig::new(3, 32, 8, 16, 2, 3, 2, 2, 2, 32, 3);
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    // ── Image embedding shape ─────────────────────────────────────────────────

    #[test]
    fn image_embedding_shape() {
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(1);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 2);
        let emb = sam.encode_image(&img).expect("ok");
        let grid = cfg.img_size / cfg.patch_size; // 4
        assert_eq!(
            (emb.channels, emb.height, emb.width),
            (cfg.embed_dim, grid, grid)
        );
        assert!(emb.data.iter().all(|v| v.is_finite()));
    }

    // ── Point prompt encoding ─────────────────────────────────────────────────

    #[test]
    fn different_points_give_different_sparse_embeddings() {
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(3);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let pe = sam.prompt_encoder();
        let a = pe.encode_points(&[4.0, 4.0], &[1]).expect("ok");
        let b = pe.encode_points(&[28.0, 20.0], &[1]).expect("ok");
        assert_eq!(a.len(), cfg.embed_dim);
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff > 1e-3,
            "different points must encode differently, diff={diff}"
        );
        // Foreground vs background label at the same location must also differ
        // (different learned label embedding).
        let fg = pe.encode_points(&[4.0, 4.0], &[1]).expect("ok");
        let bg = pe.encode_points(&[4.0, 4.0], &[0]).expect("ok");
        let label_diff: f32 = fg.iter().zip(bg.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(label_diff > 1e-4, "fg/bg labels must differ");
        // Positional encoding present (not all zeros).
        assert!(a.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn box_prompt_encodes_two_corners() {
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(4);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let emb = sam
            .prompt_encoder()
            .encode_box(&[2.0, 3.0, 20.0, 25.0])
            .expect("ok");
        assert_eq!(emb.len(), 2 * cfg.embed_dim, "box → 2 corner embeddings");
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    // ── Two-way attention updates BOTH and weights sum to 1 ───────────────────

    #[test]
    fn two_way_block_updates_both_and_weights_normalised() {
        let e = 16;
        let n_heads = 2;
        let mut rng = LcgRng::new(5);
        let block = TwoWayAttentionBlock::new(e, n_heads, 32, &mut rng).expect("ok");

        let n_t = 4;
        let n_i = 9;
        let mut tokens = vec![0.0f32; n_t * e];
        let mut image = vec![0.0f32; n_i * e];
        let mut qpe = vec![0.0f32; n_t * e];
        let mut kpe = vec![0.0f32; n_i * e];
        rng.fill_normal(&mut tokens);
        rng.fill_normal(&mut image);
        rng.fill_normal(&mut qpe);
        rng.fill_normal(&mut kpe);

        let out = block.forward(&tokens, &image, &qpe, &kpe).expect("ok");

        // BOTH representations must change.
        let tok_diff: f32 = out
            .tokens
            .iter()
            .zip(tokens.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let img_diff: f32 = out
            .image
            .iter()
            .zip(image.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(tok_diff > 1e-4, "tokens must be updated, diff={tok_diff}");
        assert!(img_diff > 1e-4, "image must be updated, diff={img_diff}");

        // Self-attention weights: each (head, query) row sums to 1 over keys.
        check_rows_sum_to_one(&out.self_weights, n_heads, n_t, n_t);
        // Token→image weights: rows over n_i sum to 1.
        check_rows_sum_to_one(&out.token_to_image_weights, n_heads, n_t, n_i);
        // Image→token weights: rows over n_t sum to 1.
        check_rows_sum_to_one(&out.image_to_token_weights, n_heads, n_i, n_t);
    }

    fn check_rows_sum_to_one(weights: &[f32], n_heads: usize, n_q: usize, n_k: usize) {
        for h in 0..n_heads {
            for i in 0..n_q {
                let row = &weights[(h * n_q + i) * n_k..(h * n_q + i + 1) * n_k];
                let sum: f32 = row.iter().sum();
                assert!(
                    row.iter().all(|&w| w >= 0.0),
                    "weights must be non-negative"
                );
                assert!((sum - 1.0).abs() < 1e-4, "attention row sum {sum} != 1");
            }
        }
    }

    // ── Prompt changes the mask ───────────────────────────────────────────────

    #[test]
    fn changing_prompt_changes_mask() {
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(6);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 7);
        let pred_a = sam.predict(&img, &[4.0, 4.0], &[1], None).expect("ok");
        let pred_b = sam.predict(&img, &[28.0, 26.0], &[1], None).expect("ok");
        let diff: f32 = pred_a
            .masks
            .iter()
            .zip(pred_b.masks.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "different prompts must change the mask, diff={diff}"
        );
    }

    // ── Mask output dims & IoU finite ─────────────────────────────────────────

    #[test]
    fn mask_output_dims_and_iou_finite() {
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(8);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 9);
        let pred = sam.predict(&img, &[10.0, 10.0], &[1], None).expect("ok");
        let grid = cfg.img_size / cfg.patch_size; // 4
        assert_eq!(pred.n_mask, cfg.n_mask);
        assert_eq!((pred.height, pred.width), (grid * 2, grid * 2));
        assert_eq!(pred.masks.len(), cfg.n_mask * (grid * 2) * (grid * 2));
        assert_eq!(pred.iou.len(), cfg.n_mask);
        assert!(
            pred.iou.iter().all(|v| v.is_finite()),
            "IoU scores must be finite"
        );
        assert!(pred.masks.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mask_prompt_changes_output() {
        // Supplying a coarse mask prompt (dense path) must change the result vs.
        // the no-mask broadcast embedding.
        let cfg = SamConfig::tiny();
        let mut rng = LcgRng::new(10);
        let sam = Sam::new(cfg.clone(), &mut rng).expect("ok");
        let img = random_image(&cfg, 11);
        let grid = cfg.img_size / cfg.patch_size;
        let mut coarse = vec![0.0f32; grid * grid];
        let mut mrng = LcgRng::new(12);
        mrng.fill_normal(&mut coarse);
        let with_mask = sam
            .predict(&img, &[10.0, 10.0], &[1], Some(&coarse))
            .expect("ok");
        let without = sam.predict(&img, &[10.0, 10.0], &[1], None).expect("ok");
        let diff: f32 = with_mask
            .masks
            .iter()
            .zip(without.masks.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "mask prompt must influence the output, diff={diff}"
        );
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_seed() {
        let cfg = SamConfig::tiny();
        let img = random_image(&cfg, 13);
        let mut ra = LcgRng::new(77);
        let mut rb = LcgRng::new(77);
        let sa = Sam::new(cfg.clone(), &mut ra).expect("ok");
        let sb = Sam::new(cfg, &mut rb).expect("ok");
        let pa = sa.predict(&img, &[10.0, 10.0], &[1], None).expect("ok");
        let pb = sb.predict(&img, &[10.0, 10.0], &[1], None).expect("ok");
        assert_eq!(pa.masks, pb.masks, "same seed → identical masks");
        assert_eq!(pa.iou, pb.iou, "same seed → identical IoU");
    }
}
