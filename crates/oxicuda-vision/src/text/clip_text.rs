//! CLIP text encoder — a faithful CPU reference of the Transformer text
//! tower from Radford et al. 2021, *"Learning Transferable Visual Models From
//! Natural Language Supervision"* (CLIP).
//!
//! The encoder turns a sequence of integer token ids into a single unit-norm
//! embedding in the joint image-text space:
//!
//! ```text
//! tokens [n_ctx]
//!   → token_embedding[token]            → [n_ctx, width]
//!   → + positional_embedding            → [n_ctx, width]
//!   → N × pre-LN transformer block
//!        (CAUSAL multi-head self-attention + MLP with GELU)
//!   → final LayerNorm                   → [n_ctx, width]
//!   → take row at the EOS / last token  → [width]
//!   → linear projection                 → [embed_dim]
//!   → L2-normalise                      → [embed_dim]
//! ```
//!
//! The two architectural details that distinguish the CLIP text tower from a
//! plain ViT encoder are (1) the **causal attention mask** — a token may only
//! attend to itself and to earlier positions, exactly as in an autoregressive
//! language model — and (2) the **EOS pooling** — the joint embedding is read
//! from the hidden state at the position of the end-of-text token (here, the
//! highest token id in the sequence, matching CLIP's use of the largest BPE id
//! `<|endoftext|>`), not from a prepended CLS token.
//!
//! All weights are flat row-major `Vec<f32>`; no `unsafe`, no external RNG.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::{gelu_exact, layer_norm, linear},
};

// ─── Config ────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the CLIP text Transformer.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipTextConfig {
    /// Vocabulary size (number of distinct token ids).
    pub vocab_size: usize,
    /// Maximum context length (number of positional embeddings).
    pub n_ctx: usize,
    /// Transformer width (token embedding / residual-stream dimension).
    pub width: usize,
    /// Number of transformer blocks.
    pub depth: usize,
    /// Number of attention heads. Must divide `width`.
    pub n_heads: usize,
    /// MLP hidden-dim multiplier (`mlp_dim = mlp_ratio * width`).
    pub mlp_ratio: usize,
    /// Output joint-embedding dimension (after the text projection).
    pub embed_dim: usize,
    /// Id used as the end-of-text marker. When pooling, the position of the
    /// *last* occurrence of this id (or, if absent, the highest id present) is
    /// used. CLIP itself uses `argmax` over ids because `<|endoftext|>` is the
    /// largest BPE id; we follow that convention via [`Self::eot_token`].
    pub eot_token: usize,
}

impl ClipTextConfig {
    /// Validate and construct a config.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `width == 0` or `embed_dim == 0`.
    /// - [`VisionError::InvalidNumHeads`] if `n_heads == 0`.
    /// - [`VisionError::HeadDimMismatch`] if `n_heads` does not divide `width`.
    /// - [`VisionError::Internal`] if `vocab_size`, `n_ctx`, or `depth` is 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        n_ctx: usize,
        width: usize,
        depth: usize,
        n_heads: usize,
        mlp_ratio: usize,
        embed_dim: usize,
        eot_token: usize,
    ) -> VisionResult<Self> {
        if width == 0 {
            return Err(VisionError::InvalidEmbedDim(width));
        }
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if width % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch {
                n_heads,
                embed_dim: width,
            });
        }
        if vocab_size == 0 {
            return Err(VisionError::Internal("vocab_size must be > 0".into()));
        }
        if n_ctx == 0 {
            return Err(VisionError::Internal("n_ctx must be > 0".into()));
        }
        if depth == 0 {
            return Err(VisionError::Internal("depth must be > 0".into()));
        }
        if eot_token >= vocab_size {
            return Err(VisionError::Internal(
                "eot_token must be < vocab_size".into(),
            ));
        }
        Ok(Self {
            vocab_size,
            n_ctx,
            width,
            depth,
            n_heads,
            mlp_ratio,
            embed_dim,
            eot_token,
        })
    }

    /// A tiny config suitable for unit tests.
    ///
    /// `vocab_size = 64`, `n_ctx = 16`, `width = 32`, `depth = 2`,
    /// `n_heads = 4`, `mlp_ratio = 4`, `embed_dim = 24`, `eot_token = 63`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            n_ctx: 16,
            width: 32,
            depth: 2,
            n_heads: 4,
            mlp_ratio: 4,
            embed_dim: 24,
            eot_token: 63,
        }
    }

    /// Per-head dimension.
    #[must_use]
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.width / self.n_heads
    }

    /// MLP hidden dimension.
    #[must_use]
    #[inline]
    pub fn mlp_dim(&self) -> usize {
        self.mlp_ratio * self.width
    }
}

// ─── Per-block weights ──────────────────────────────────────────────────────────

/// Learnable weights for one pre-LN causal transformer block.
struct TextBlockWeights {
    qkv_weight: Vec<f32>,  // [3*width, width]
    qkv_bias: Vec<f32>,    // [3*width]
    out_weight: Vec<f32>,  // [width, width]
    out_bias: Vec<f32>,    // [width]
    mlp1_weight: Vec<f32>, // [mlp_dim, width]
    mlp1_bias: Vec<f32>,   // [mlp_dim]
    mlp2_weight: Vec<f32>, // [width, mlp_dim]
    mlp2_bias: Vec<f32>,   // [width]
    ln1_weight: Vec<f32>,  // [width]
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
}

impl TextBlockWeights {
    fn default_init(cfg: &ClipTextConfig, rng: &mut LcgRng) -> Self {
        let w = cfg.width;
        let mlp = cfg.mlp_dim();
        let scale = 1.0 / (w as f32).sqrt();
        let fill = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };
        Self {
            qkv_weight: fill(rng, 3 * w * w, scale),
            qkv_bias: vec![0.0f32; 3 * w],
            out_weight: fill(rng, w * w, scale),
            out_bias: vec![0.0f32; w],
            mlp1_weight: fill(rng, mlp * w, scale),
            mlp1_bias: vec![0.0f32; mlp],
            mlp2_weight: fill(rng, w * mlp, scale),
            mlp2_bias: vec![0.0f32; w],
            ln1_weight: vec![1.0f32; w],
            ln1_bias: vec![0.0f32; w],
            ln2_weight: vec![1.0f32; w],
            ln2_bias: vec![0.0f32; w],
        }
    }
}

// ─── Causal multi-head self-attention ───────────────────────────────────────────

/// Causal (autoregressive) multi-head self-attention.
///
/// Identical to standard scaled-dot-product attention except that, before the
/// row-wise softmax, every score `S[i, j]` with `j > i` is set to `-∞`, so the
/// softmax weight of any *future* key is exactly zero. Position `i` therefore
/// mixes only positions `0..=i`.
///
/// `tokens` is `[n · e]` row-major; returns `[n · e]`.
#[allow(clippy::too_many_arguments)]
fn causal_mhsa(
    tokens: &[f32],
    n: usize,
    e: usize,
    n_heads: usize,
    head_dim: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
) -> VisionResult<Vec<f32>> {
    // Fused QKV projection → [n, 3e].
    let qkv = linear(tokens, qkv_weight, qkv_bias, e, 3 * e);

    let mut q = vec![0.0f32; n * e];
    let mut k = vec![0.0f32; n * e];
    let mut v = vec![0.0f32; n * e];
    for t in 0..n {
        let src = &qkv[t * 3 * e..(t + 1) * 3 * e];
        q[t * e..(t + 1) * e].copy_from_slice(&src[..e]);
        k[t * e..(t + 1) * e].copy_from_slice(&src[e..2 * e]);
        v[t * e..(t + 1) * e].copy_from_slice(&src[2 * e..]);
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut concat = vec![0.0f32; n * e];

    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..n {
            // Causal window: keys 0..=i only.
            // Stable softmax over the masked row.
            let mut max_score = f32::NEG_INFINITY;
            let mut row_scores = vec![0.0f32; i + 1];
            for (j, slot) in row_scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * e + off + d] * k[j * e + off + d];
                }
                let s = dot * scale;
                *slot = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum = 0.0f32;
            for s in &mut row_scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &sw) in row_scores.iter().enumerate() {
                    acc += sw * inv * v[j * e + off + d];
                }
                concat[i * e + off + d] = acc;
            }
        }
    }

    let out = linear(&concat, out_weight, out_bias, e, e);
    if out.iter().any(|x| !x.is_finite()) {
        return Err(VisionError::NonFinite("clip text attention output"));
    }
    Ok(out)
}

// ─── CLIP text encoder ──────────────────────────────────────────────────────────

/// The CLIP Transformer text tower.
pub struct ClipTextEncoder {
    /// Configuration.
    pub config: ClipTextConfig,
    /// Token embedding table: `[vocab_size · width]` row-major.
    pub token_embedding: Vec<f32>,
    /// Learned positional embedding: `[n_ctx · width]` row-major.
    pub positional_embedding: Vec<f32>,
    /// Per-block weights.
    blocks: Vec<TextBlockWeights>,
    /// Final LayerNorm scale `[width]`.
    final_ln_weight: Vec<f32>,
    /// Final LayerNorm bias `[width]`.
    final_ln_bias: Vec<f32>,
    /// Text projection: `[embed_dim · width]` row-major (maps width → embed_dim).
    text_projection: Vec<f32>,
}

impl ClipTextEncoder {
    /// Construct a CLIP text encoder with Gaussian-initialised weights.
    ///
    /// # Errors
    /// Propagates configuration / sub-component validation errors.
    pub fn new(cfg: ClipTextConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let w = cfg.width;

        // Token & positional embeddings use the canonical CLIP init std of 0.02
        // (small so the residual stream starts near the identity).
        let mut token_embedding = vec![0.0f32; cfg.vocab_size * w];
        rng.fill_normal(&mut token_embedding);
        for v in &mut token_embedding {
            *v *= 0.02;
        }
        let mut positional_embedding = vec![0.0f32; cfg.n_ctx * w];
        rng.fill_normal(&mut positional_embedding);
        for v in &mut positional_embedding {
            *v *= 0.01;
        }

        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(TextBlockWeights::default_init(&cfg, rng));
        }

        let final_ln_weight = vec![1.0f32; w];
        let final_ln_bias = vec![0.0f32; w];

        // Text projection (width → embed_dim), scaled by 1/√width.
        let scale = 1.0 / (w as f32).sqrt();
        let mut text_projection = vec![0.0f32; cfg.embed_dim * w];
        rng.fill_normal(&mut text_projection);
        for v in &mut text_projection {
            *v *= scale;
        }

        Ok(Self {
            config: cfg,
            token_embedding,
            positional_embedding,
            blocks,
            final_ln_weight,
            final_ln_bias,
            text_projection,
        })
    }

    /// Locate the pooling position for a token sequence.
    ///
    /// Following CLIP, the joint embedding is read at the position of the
    /// end-of-text token. We select the position of the **last** occurrence of
    /// `eot_token`; if it never appears, we fall back to CLIP's `argmax`
    /// convention (the position of the highest token id), and as a final
    /// fallback the last index.
    #[must_use]
    pub fn eot_position(&self, tokens: &[usize]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        // Last occurrence of the explicit EOT id.
        for (idx, &tok) in tokens.iter().enumerate().rev() {
            if tok == self.config.eot_token {
                return idx;
            }
        }
        // Fallback: position of the maximum id (CLIP argmax).
        let mut best_idx = tokens.len() - 1;
        let mut best_val = tokens[best_idx];
        for (idx, &tok) in tokens.iter().enumerate() {
            if tok > best_val {
                best_val = tok;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Run the full encoder and return the contextual hidden states *before*
    /// pooling and projection: `[n · width]`, after the final LayerNorm.
    ///
    /// Exposed so causality tests can probe individual token hidden states.
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `tokens` is empty.
    /// - [`VisionError::Internal`] if the sequence is longer than `n_ctx` or a
    ///   token id is out of the vocabulary range.
    pub fn hidden_states(&self, tokens: &[usize]) -> VisionResult<Vec<f32>> {
        let cfg = &self.config;
        let w = cfg.width;
        let n = tokens.len();
        if n == 0 {
            return Err(VisionError::EmptyInput("token sequence"));
        }
        if n > cfg.n_ctx {
            return Err(VisionError::Internal(
                "sequence length exceeds n_ctx".into(),
            ));
        }
        for &tok in tokens {
            if tok >= cfg.vocab_size {
                return Err(VisionError::Internal(
                    "token id out of vocabulary range".into(),
                ));
            }
        }

        // Embed: token_embedding[token] + positional_embedding[position].
        let mut h = vec![0.0f32; n * w];
        for (pos, &tok) in tokens.iter().enumerate() {
            let te = &self.token_embedding[tok * w..(tok + 1) * w];
            let pe = &self.positional_embedding[pos * w..(pos + 1) * w];
            let dst = &mut h[pos * w..(pos + 1) * w];
            for d in 0..w {
                dst[d] = te[d] + pe[d];
            }
        }

        // Pre-LN causal transformer blocks.
        for blk in &self.blocks {
            // Block 1: x = x + Attn(LN1(x))   (causal)
            let normed = layer_norm(&h, &blk.ln1_weight, &blk.ln1_bias, n, w, 1e-5);
            let attn = causal_mhsa(
                &normed,
                n,
                w,
                cfg.n_heads,
                cfg.head_dim(),
                &blk.qkv_weight,
                &blk.qkv_bias,
                &blk.out_weight,
                &blk.out_bias,
            )?;
            for (hv, av) in h.iter_mut().zip(attn.iter()) {
                *hv += av;
            }

            // Block 2: x = x + MLP(LN2(x))
            let normed2 = layer_norm(&h, &blk.ln2_weight, &blk.ln2_bias, n, w, 1e-5);
            let mlp_dim = cfg.mlp_dim();
            let mid = linear(&normed2, &blk.mlp1_weight, &blk.mlp1_bias, w, mlp_dim);
            let mid: Vec<f32> = mid.into_iter().map(gelu_exact).collect();
            let mlp_out = linear(&mid, &blk.mlp2_weight, &blk.mlp2_bias, mlp_dim, w);
            for (hv, mv) in h.iter_mut().zip(mlp_out.iter()) {
                *hv += mv;
            }
        }

        // Final LayerNorm.
        let out = layer_norm(&h, &self.final_ln_weight, &self.final_ln_bias, n, w, 1e-5);
        Ok(out)
    }

    /// Encode a token sequence to a unit-norm joint-space embedding.
    ///
    /// # Returns
    /// `[embed_dim]` L2-normalised text embedding.
    ///
    /// # Errors
    /// Propagates errors from [`Self::hidden_states`].
    pub fn encode(&self, tokens: &[usize]) -> VisionResult<Vec<f32>> {
        let cfg = &self.config;
        let w = cfg.width;
        let hs = self.hidden_states(tokens)?;

        // Pool the hidden state at the EOS / argmax position.
        let pool = self.eot_position(tokens);
        let pooled = &hs[pool * w..(pool + 1) * w];

        // Linear projection width → embed_dim (no bias, like CLIP's text_projection).
        let mut z = vec![0.0f32; cfg.embed_dim];
        for (p, zp) in z.iter_mut().enumerate() {
            let row = &self.text_projection[p * w..(p + 1) * w];
            *zp = row
                .iter()
                .zip(pooled.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
        }

        // L2-normalise.
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv = 1.0 / norm.max(1e-12);
        for v in &mut z {
            *v *= inv;
        }

        if z.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("clip text embedding"));
        }
        Ok(z)
    }

    /// Encode a batch of (independent) token sequences.
    ///
    /// # Returns
    /// One `[embed_dim]` embedding per input sequence.
    ///
    /// # Errors
    /// Propagates the first encoding error.
    pub fn encode_batch(&self, sequences: &[Vec<usize>]) -> VisionResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(sequences.len());
        for seq in sequences {
            out.push(self.encode(seq)?);
        }
        Ok(out)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(seed: u64) -> ClipTextEncoder {
        let mut rng = LcgRng::new(seed);
        ClipTextEncoder::new(ClipTextConfig::tiny(), &mut rng).expect("encoder ok")
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-12)
    }

    // ── Config ──────────────────────────────────────────────────────────────────

    #[test]
    fn config_tiny_valid() {
        let cfg = ClipTextConfig::tiny();
        assert_eq!(cfg.head_dim(), 8);
        assert_eq!(cfg.mlp_dim(), 128);
    }

    #[test]
    fn config_head_mismatch_errors() {
        let r = ClipTextConfig::new(64, 16, 30, 2, 4, 4, 24, 63);
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    #[test]
    fn config_zero_width_errors() {
        let r = ClipTextConfig::new(64, 16, 0, 2, 4, 4, 24, 63);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn config_eot_out_of_range_errors() {
        let r = ClipTextConfig::new(64, 16, 32, 2, 4, 4, 24, 64);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    // ── (a) Output embedding is unit-norm ─────────────────────────────────────────

    #[test]
    fn encode_output_is_unit_norm() {
        let enc = make_encoder(1);
        let tokens = vec![3usize, 7, 12, 5, 63];
        let z = enc.encode(&tokens).expect("encode ok");
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "text embedding must be L2-unit-norm; got {norm}"
        );
    }

    // ── (b) CAUSALITY: a future token cannot change an earlier hidden state ────────

    #[test]
    fn causality_future_token_does_not_affect_earlier_hidden_state() {
        let enc = make_encoder(2);
        // Two sequences identical in positions 0..=2, differing at position 3.
        let seq_a = vec![5usize, 9, 14, 2, 63];
        let seq_b = vec![5usize, 9, 14, 31, 63]; // position 3 changed
        let hs_a = enc.hidden_states(&seq_a).expect("ok");
        let hs_b = enc.hidden_states(&seq_b).expect("ok");
        let w = enc.config.width;
        // Hidden states at positions 0,1,2 must be identical (they only attend
        // to positions ≤ their index, none of which changed).
        for pos in 0..3 {
            for d in 0..w {
                let a = hs_a[pos * w + d];
                let b = hs_b[pos * w + d];
                assert!(
                    (a - b).abs() < 1e-6,
                    "causality violated at pos {pos}, dim {d}: {a} vs {b}"
                );
            }
        }
        // Sanity: position 3 (which saw the changed token) *should* differ.
        let diff_pos3: f32 = (0..w)
            .map(|d| (hs_a[3 * w + d] - hs_b[3 * w + d]).abs())
            .sum();
        assert!(
            diff_pos3 > 1e-6,
            "position 3 should change when its own token changes (diff={diff_pos3})"
        );
    }

    // ── (c) Different sequences → different embeddings ─────────────────────────────

    #[test]
    fn different_sequences_give_different_embeddings() {
        let enc = make_encoder(3);
        let za = enc.encode(&[1usize, 2, 3, 63]).expect("ok");
        let zb = enc.encode(&[10usize, 20, 30, 63]).expect("ok");
        let diff: f32 = za.iter().zip(zb.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-4,
            "distinct token sequences must produce distinct embeddings (diff={diff})"
        );
    }

    // ── (d) Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_input_same_output() {
        let enc = make_encoder(4);
        let tokens = vec![4usize, 8, 15, 16, 23, 42, 63];
        let z1 = enc.encode(&tokens).expect("ok");
        let z2 = enc.encode(&tokens).expect("ok");
        assert_eq!(z1, z2, "encoder must be deterministic");
    }

    // ── (e) Cosine similarity of identical inputs == 1 ────────────────────────────

    #[test]
    fn cosine_of_identical_inputs_is_one() {
        let enc = make_encoder(5);
        let tokens = vec![2usize, 4, 6, 8, 63];
        let z = enc.encode(&tokens).expect("ok");
        let sim = cosine(&z, &z);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "cosine(z, z) must be 1.0; got {sim}"
        );
    }

    // ── (f) Projection output dim == configured joint dim ─────────────────────────

    #[test]
    fn projection_output_dim_matches_config() {
        let enc = make_encoder(6);
        let z = enc.encode(&[1usize, 2, 63]).expect("ok");
        assert_eq!(
            z.len(),
            enc.config.embed_dim,
            "projected embedding dim must equal config.embed_dim"
        );
    }

    // ── (g) EOS / pooling position selection ──────────────────────────────────────

    #[test]
    fn eot_position_selects_last_eot_occurrence() {
        let enc = make_encoder(7);
        // EOT id is 63. It appears at index 4 (last real position before padding).
        let tokens = vec![5usize, 9, 14, 2, 63, 0, 0];
        assert_eq!(
            enc.eot_position(&tokens),
            4,
            "must pool at the last EOT (id=63) position"
        );
    }

    #[test]
    fn eot_position_argmax_fallback_when_no_explicit_eot() {
        let enc = make_encoder(8);
        // No id == 63; highest id is 40 at index 2 → argmax fallback.
        let tokens = vec![5usize, 9, 40, 2, 7];
        assert_eq!(
            enc.eot_position(&tokens),
            2,
            "argmax fallback should pick the highest-id position"
        );
    }

    #[test]
    fn pooling_uses_eot_hidden_state() {
        // The embedding must be derived from the hidden state at the EOT
        // position: changing a token *after* the EOT must not change the
        // embedding (because pooling happens at EOT and causal attention means
        // EOT never sees later tokens).
        let enc = make_encoder(9);
        let base = vec![3usize, 7, 12, 63, 1, 2]; // EOT at index 3
        let changed = vec![3usize, 7, 12, 63, 30, 40]; // tokens after EOT differ
        let z_base = enc.encode(&base).expect("ok");
        let z_changed = enc.encode(&changed).expect("ok");
        let diff: f32 = z_base
            .iter()
            .zip(z_changed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-6,
            "tokens after the EOT must not affect the pooled embedding (diff={diff})"
        );
    }

    // ── Error paths ───────────────────────────────────────────────────────────────

    #[test]
    fn empty_sequence_errors() {
        let enc = make_encoder(10);
        let r = enc.encode(&[]);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn sequence_too_long_errors() {
        let enc = make_encoder(11);
        let too_long: Vec<usize> = (0..enc.config.n_ctx + 1).map(|i| i % 60).collect();
        let r = enc.encode(&too_long);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    #[test]
    fn out_of_vocab_token_errors() {
        let enc = make_encoder(12);
        let r = enc.encode(&[1usize, 9999, 63]);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    // ── Batch ─────────────────────────────────────────────────────────────────────

    #[test]
    fn encode_batch_matches_individual() {
        let enc = make_encoder(13);
        let seqs = vec![vec![1usize, 2, 63], vec![5usize, 9, 14, 63]];
        let batch = enc.encode_batch(&seqs).expect("ok");
        assert_eq!(batch.len(), 2);
        for (i, seq) in seqs.iter().enumerate() {
            let single = enc.encode(seq).expect("ok");
            for (a, b) in batch[i].iter().zip(single.iter()) {
                assert!((a - b).abs() < 1e-6, "batch vs single mismatch");
            }
        }
    }

    // ── Extra causality guard: an EARLY token change DOES propagate forward ────────

    #[test]
    fn early_token_change_propagates_to_later_positions() {
        let enc = make_encoder(14);
        let seq_a = vec![5usize, 9, 14, 2, 63];
        let seq_b = vec![31usize, 9, 14, 2, 63]; // position 0 changed
        let hs_a = enc.hidden_states(&seq_a).expect("ok");
        let hs_b = enc.hidden_states(&seq_b).expect("ok");
        let w = enc.config.width;
        // A later position (e.g. 4) must change because it attends back to pos 0.
        let diff_pos4: f32 = (0..w)
            .map(|d| (hs_a[4 * w + d] - hs_b[4 * w + d]).abs())
            .sum();
        assert!(
            diff_pos4 > 1e-6,
            "changing position 0 must affect later positions (diff={diff_pos4})"
        );
    }
}
