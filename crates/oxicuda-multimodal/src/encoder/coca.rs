//! CoCa — Contrastive Captioners are Image-Text Foundation Models.
//!
//! Reference: Yu et al. 2022, "CoCa: Contrastive Captioners are Image-Text
//! Foundation Models".
//!
//! CoCa unifies two complementary self-supervised objectives in a single
//! image-text foundation model:
//!
//! 1. **Contrastive objective** — image and text encoders project to a shared
//!    embedding space; image/text pairs are pulled together while non-pairs are
//!    pushed apart via a symmetric InfoNCE loss with a learned temperature.
//! 2. **Generative captioning objective** — a multimodal text decoder
//!    cross-attends over the image features and is trained with teacher-forced
//!    language modelling to predict the next text token.
//!
//! The contrastive head consumes a single image embedding per example, obtained
//! by *attentional pooling*: a single learnable query attends over the image
//! patch tokens via cross-attention to summarise them into one vector. Sharing
//! the image encoder between the contrastive and generative heads gives CoCa
//! its "foundation" character — one network, two losses.
//!
//! ```text
//! image_tokens [n_img × d_model]
//!         │
//!         ▼
//! ┌─────────────────────────┐         ┌──────────────────────────────┐
//! │ Attentional pooler:     │         │ Multimodal decoder:          │
//! │ learnable q (1×d_model) │         │   self-attn(text)            │
//! │ cross-attends image     │         │   cross-attn(text→image)     │
//! │ → 1×d_model embedding   │         │   FFN                        │
//! └──────────┬──────────────┘         │   vocab linear (d→V)         │
//!            │                        └─────────────┬────────────────┘
//!            ▼                                      ▼
//!   contrastive projection                  per-token logits (n_text × V)
//! ```
//!
//! Loss aggregation:
//! `L = λ · L_contrastive + (1 − λ) · L_captioning`.

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the CoCa multimodal foundation model.
#[derive(Debug, Clone)]
pub struct CoCaConfig {
    /// Model dimension (shared embedding size).
    pub d_model: usize,
    /// Number of attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Vocabulary size for the captioning head.
    pub vocab_size: usize,
    /// Contrastive softmax temperature; must be `> 0`.
    pub temperature: f32,
}

impl CoCaConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            d_model: 8,
            n_heads: 2,
            vocab_size: 32,
            temperature: 0.07,
        }
    }

    /// Validate the configuration.
    fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.vocab_size == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(MultiModalError::InvalidTemperature {
                temp: self.temperature,
            });
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// All learnable parameters of a CoCa model.
#[derive(Debug, Clone)]
pub struct CoCaWeights {
    /// Learnable pooling query, `[1 × d_model]`.
    pub pool_query: Vec<f32>,
    /// Attentional pooler cross-attention (Q from `pool_query`, K/V from
    /// image tokens).
    pub pool_attn: CrossAttnWeights,
    /// Contrastive projection head `[d_model × d_model]`.
    pub contrast_proj: Vec<f32>,
    /// LayerNorm before the multimodal decoder's self-attention.
    pub ln_self: LayerNorm,
    /// LayerNorm before the multimodal decoder's cross-attention.
    pub ln_cross: LayerNorm,
    /// LayerNorm before the multimodal decoder's FFN.
    pub ln_ffn: LayerNorm,
    /// Multimodal decoder self-attention weights (text → text).
    pub decoder_self_attn: CrossAttnWeights,
    /// Multimodal decoder cross-attention weights (text → image features).
    pub decoder_cross_attn: CrossAttnWeights,
    /// Multimodal decoder feed-forward network.
    pub decoder_ffn: FeedForward,
    /// Vocabulary projection `[d_model × vocab_size]` (row-major).
    pub vocab_head: Vec<f32>,
    /// Vocabulary projection bias `[vocab_size]`.
    pub vocab_bias: Vec<f32>,
}

impl CoCaWeights {
    /// Randomly initialise all weights from N(0, 1/d) Gaussian noise.
    fn random(cfg: &CoCaConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let d_ff = d.max(4) * 4;
        let attn_scale = (1.0 / d as f32).sqrt();
        let ffn_in_scale = (1.0 / d as f32).sqrt();
        let ffn_out_scale = (1.0 / d_ff as f32).sqrt();
        let head_scale = (1.0 / d as f32).sqrt();

        let pool_query = gaussian_vec(d, attn_scale, rng);
        let pool_attn = random_attn_weights(d, attn_scale, rng);
        let contrast_proj = gaussian_vec(d * d, attn_scale, rng);

        let decoder_self_attn = random_attn_weights(d, attn_scale, rng);
        let decoder_cross_attn = random_attn_weights(d, attn_scale, rng);

        let decoder_ffn = FeedForward {
            w1: gaussian_vec(d * d_ff, ffn_in_scale, rng),
            b1: vec![0.0_f32; d_ff],
            w2: gaussian_vec(d_ff * d, ffn_out_scale, rng),
            b2: vec![0.0_f32; d],
            d_model: d,
            d_ff,
        };

        let vocab_head = gaussian_vec(d * v, head_scale, rng);
        let vocab_bias = vec![0.0_f32; v];

        Self {
            pool_query,
            pool_attn,
            contrast_proj,
            ln_self: LayerNorm::ones(d),
            ln_cross: LayerNorm::ones(d),
            ln_ffn: LayerNorm::ones(d),
            decoder_self_attn,
            decoder_cross_attn,
            decoder_ffn,
            vocab_head,
            vocab_bias,
        }
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

/// Build a `[d × d]` set of attention projections from Gaussian noise.
fn random_attn_weights(d: usize, scale: f32, rng: &mut LcgRng) -> CrossAttnWeights {
    CrossAttnWeights {
        w_q: gaussian_vec(d * d, scale, rng),
        w_k: gaussian_vec(d * d, scale, rng),
        w_v: gaussian_vec(d * d, scale, rng),
        w_o: gaussian_vec(d * d, scale, rng),
    }
}

// ─── CoCa ────────────────────────────────────────────────────────────────────

/// CoCa Contrastive Captioner.
#[derive(Debug, Clone)]
pub struct CoCa {
    pub cfg: CoCaConfig,
    pub weights: CoCaWeights,
}

impl CoCa {
    /// Create a new CoCa with randomly initialised weights.
    pub fn new(cfg: CoCaConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = CoCaWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Attentional image pooling.
    ///
    /// The single learnable query (`1 × d_model`) cross-attends over the
    /// `n_tokens × d_model` image tokens to produce one summary embedding of
    /// length `d_model`. The output length is **independent of `n_tokens`**.
    pub fn pool_image(&self, image_tokens: &[f32], n_tokens: usize) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        if n_tokens == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if image_tokens.len() != n_tokens * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_tokens * d,
                got: image_tokens.len(),
            });
        }

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;
        let attn = CrossAttention::with_weights(attn_cfg, self.weights.pool_attn.clone());

        // pool_query is [1 × d_model]; image tokens are K/V.
        let out = attn.forward(
            &self.weights.pool_query,
            image_tokens,
            image_tokens,
            1,
            n_tokens,
        )?;

        if out.len() != d {
            return Err(MultiModalError::DimensionMismatch {
                expected: d,
                got: out.len(),
            });
        }
        Ok(out)
    }

    /// Symmetric InfoNCE contrastive loss for a batch of image / text
    /// embeddings.
    ///
    /// Both inputs are `batch × d_model` row-major. The embeddings are
    /// L2-normalised, then the similarity matrix `S = (I · Tᵀ) / τ` is fed
    /// through a symmetric cross-entropy with diagonal targets:
    /// `L = (CE(S, diag) + CE(Sᵀ, diag)) / 2`.
    pub fn contrastive_loss(
        &self,
        image_embs: &[f32],
        text_embs: &[f32],
        batch: usize,
    ) -> MmResult<f32> {
        let d = self.cfg.d_model;
        if batch == 0 {
            return Err(MultiModalError::InvalidBatchSize);
        }
        if image_embs.len() != batch * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * d,
                got: image_embs.len(),
            });
        }
        if text_embs.len() != batch * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * d,
                got: text_embs.len(),
            });
        }
        let temp = self.cfg.temperature;
        if !temp.is_finite() || temp <= 0.0 {
            return Err(MultiModalError::InvalidTemperature { temp });
        }

        // Project embeddings through the contrastive projection head.
        let img_proj = matmul_seq(image_embs, &self.weights.contrast_proj, batch, d, d)?;
        let txt_proj = matmul_seq(text_embs, &self.weights.contrast_proj, batch, d, d)?;

        // L2-normalise each row.
        let img_n = l2_normalise_rows(&img_proj, batch, d);
        let txt_n = l2_normalise_rows(&txt_proj, batch, d);

        // Similarity S[i, j] = img_n[i] · txt_n[j], scaled by 1/τ.
        let mut sim = vec![0.0_f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                let mut dot = 0.0_f32;
                for k in 0..d {
                    dot += img_n[i * d + k] * txt_n[j * d + k];
                }
                sim[i * batch + j] = dot / temp;
            }
        }

        // Row-wise cross-entropy with diagonal targets (image → text).
        let loss_i2t = ce_diag_loss(&sim, batch);

        // Column-wise cross-entropy: equivalent to row-wise on the transpose
        // (text → image).
        let mut sim_t = vec![0.0_f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                sim_t[i * batch + j] = sim[j * batch + i];
            }
        }
        let loss_t2i = ce_diag_loss(&sim_t, batch);

        let loss = 0.5 * (loss_i2t + loss_t2i);
        if !loss.is_finite() {
            return Err(MultiModalError::NanEncountered {
                location: "coca_contrastive_loss",
            });
        }
        Ok(loss)
    }

    /// Multimodal captioning logits.
    ///
    /// `text_tokens`: `[n_text × d_model]` row-major — already-embedded text
    /// hidden states.
    /// `image_features`: `[n_img × d_model]` row-major — image token features
    /// the decoder cross-attends to (typically the pre-pooling outputs).
    ///
    /// Returns per-position vocabulary logits, shape `[n_text × vocab_size]`.
    pub fn captioning_logits(
        &self,
        text_tokens: &[f32],
        n_text: usize,
        image_features: &[f32],
        n_img: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        if n_text == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if n_img == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if text_tokens.len() != n_text * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_text * d,
                got: text_tokens.len(),
            });
        }
        if image_features.len() != n_img * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_img * d,
                got: image_features.len(),
            });
        }

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;

        // ── Self-attention (text → text), pre-norm + residual ───────────────
        let self_attn =
            CrossAttention::with_weights(attn_cfg.clone(), self.weights.decoder_self_attn.clone());
        let ln_self = self.weights.ln_self.forward(text_tokens, n_text)?;
        let self_out = self_attn.forward(&ln_self, &ln_self, &ln_self, n_text, n_text)?;
        let mut x = add_vecs(text_tokens, &self_out)?;

        // ── Cross-attention (text → image features) ─────────────────────────
        let cross_attn =
            CrossAttention::with_weights(attn_cfg, self.weights.decoder_cross_attn.clone());
        let ln_cross = self.weights.ln_cross.forward(&x, n_text)?;
        let cross_out =
            cross_attn.forward(&ln_cross, image_features, image_features, n_text, n_img)?;
        for (xi, ci) in x.iter_mut().zip(cross_out.iter()) {
            *xi += *ci;
        }

        // ── Feed-forward network ────────────────────────────────────────────
        let ln_ffn = self.weights.ln_ffn.forward(&x, n_text)?;
        let ffn_out = self.weights.decoder_ffn.forward(&ln_ffn, n_text)?;
        for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += *fi;
        }

        // ── Vocabulary projection: [n_text × d] · [d × V] + bias ────────────
        let mut logits = matmul_seq(&x, &self.weights.vocab_head, n_text, d, v)?;
        for r in 0..n_text {
            for j in 0..v {
                logits[r * v + j] += self.weights.vocab_bias[j];
            }
        }

        Ok(logits)
    }

    /// Combine the contrastive and captioning losses with weight `lambda`.
    ///
    /// `L = λ · L_c + (1 − λ) · L_g`, with `λ ∈ [0, 1]`.
    pub fn coca_loss(&self, contrastive: f32, captioning: f32, lambda: f32) -> MmResult<f32> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(MultiModalError::Internal(format!(
                "lambda must be in [0, 1]; got {lambda}"
            )));
        }
        if !contrastive.is_finite() || !captioning.is_finite() {
            return Err(MultiModalError::NanEncountered {
                location: "coca_loss",
            });
        }
        Ok(lambda * contrastive + (1.0 - lambda) * captioning)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Matrix multiply: `A [rows × in_dim] × W [in_dim × out_dim]` → `[rows × out_dim]`.
fn matmul_seq(
    a: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> MmResult<Vec<f32>> {
    if a.len() != rows * in_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: rows * in_dim,
            got: a.len(),
        });
    }
    if w.len() != in_dim * out_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: in_dim * out_dim,
            got: w.len(),
        });
    }
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += a[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    Ok(out)
}

/// L2-normalise every row of an `[rows × cols]` row-major matrix.
fn l2_normalise_rows(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = m.to_vec();
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let mut norm_sq = 0.0_f32;
        for v in &out[start..end] {
            norm_sq += v * v;
        }
        let inv = if norm_sq > 1e-12 {
            1.0 / norm_sq.sqrt()
        } else {
            1.0
        };
        for v in &mut out[start..end] {
            *v *= inv;
        }
    }
    out
}

/// Row-wise cross-entropy of a `[batch × batch]` logits matrix against
/// diagonal targets. Returns the mean negative log-likelihood.
fn ce_diag_loss(logits: &[f32], batch: usize) -> f32 {
    let mut loss = 0.0_f32;
    for i in 0..batch {
        let row = &logits[i * batch..(i + 1) * batch];
        let max_s = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        for &s in row {
            sum_exp += (s - max_s).exp();
        }
        let log_sum = max_s + sum_exp.ln();
        loss += log_sum - row[i];
    }
    loss / batch as f32
}

/// Add two equally-sized vectors element-wise.
fn add_vecs(a: &[f32], b: &[f32]) -> MmResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(MultiModalError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coca(seed: u64) -> CoCa {
        let mut rng = LcgRng::new(seed);
        match CoCa::new(CoCaConfig::tiny(), &mut rng) {
            Ok(c) => c,
            Err(e) => panic!("tiny CoCa should construct: {e:?}"),
        }
    }

    // ── 1: pool_image returns length d_model, independent of n_tokens ───────
    #[test]
    fn pool_image_output_length_is_d_model() {
        let coca = make_coca(1);
        let d = coca.cfg.d_model;
        let img3 = vec![0.1_f32; 3 * d];
        let img7 = vec![0.1_f32; 7 * d];
        let out3 = coca
            .pool_image(&img3, 3)
            .expect("pool_image should succeed");
        let out7 = coca
            .pool_image(&img7, 7)
            .expect("pool_image should succeed");
        assert_eq!(out3.len(), d);
        assert_eq!(out7.len(), d);
    }

    // ── 2: contrastive_loss is non-negative ─────────────────────────────────
    #[test]
    fn contrastive_loss_non_negative() {
        let coca = make_coca(2);
        let d = coca.cfg.d_model;
        let n = 4;
        let img: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let txt: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let loss = coca
            .contrastive_loss(&img, &txt, n)
            .expect("contrastive_loss should succeed");
        assert!(loss.is_finite());
        assert!(loss >= -1e-5);
    }

    // ── 3: contrastive_loss ~0 when image_embs == text_embs & well separated
    #[test]
    fn contrastive_loss_perfect_alignment_near_zero() {
        // Identity projection → l2-normalisation preserves orthogonality.
        // We override the contrastive projection to be the identity so that
        // the similarity matrix is exactly diagonal-dominant for one-hot
        // embeddings.
        let mut coca = make_coca(3);
        let d = coca.cfg.d_model;
        let n = d; // n = d ensures each example can be a distinct basis vector.
        let mut ident = vec![0.0_f32; d * d];
        for i in 0..d {
            ident[i * d + i] = 1.0;
        }
        coca.weights.contrast_proj = ident;

        // Each example is a distinct unit basis vector — well-separated.
        let mut embs = vec![0.0_f32; n * d];
        for i in 0..n {
            embs[i * d + i] = 1.0;
        }
        // With identical embeddings and a low temperature, the diagonal sim
        // (=1/τ) dominates the off-diagonal sims (=0/τ), so CE → 0.
        let loss = coca
            .contrastive_loss(&embs, &embs, n)
            .expect("contrastive_loss should succeed");
        // τ = 0.07, log-sum-exp ≈ 1/0.07 + ln(1 + (n-1) e^{-1/0.07}).
        // The second term is ~e^{-14.28} ≈ 6e-7 per off-diag, so loss is
        // essentially zero.
        assert!(
            loss < 1e-3,
            "perfect alignment loss should be ~0, got {loss}"
        );
    }

    // ── 4: contrastive_loss symmetric in its two args ────────────────────────
    #[test]
    fn contrastive_loss_symmetric_in_args() {
        let coca = make_coca(4);
        let d = coca.cfg.d_model;
        let n = 5;
        let a: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
        let b: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.11).cos()).collect();
        let l1 = coca
            .contrastive_loss(&a, &b, n)
            .expect("contrastive_loss should succeed");
        let l2 = coca
            .contrastive_loss(&b, &a, n)
            .expect("contrastive_loss should succeed");
        assert!((l1 - l2).abs() < 1e-4, "{l1} vs {l2}");
    }

    // ── 5: captioning_logits length == n_text * vocab_size ──────────────────
    #[test]
    fn captioning_logits_output_length() {
        let coca = make_coca(5);
        let d = coca.cfg.d_model;
        let n_text = 3;
        let n_img = 4;
        let text = vec![0.1_f32; n_text * d];
        let img = vec![0.2_f32; n_img * d];
        let out = coca
            .captioning_logits(&text, n_text, &img, n_img)
            .expect("captioning_logits should succeed");
        assert_eq!(out.len(), n_text * coca.cfg.vocab_size);
    }

    // ── 6: coca_loss == lambda·c + (1−lambda)·g ─────────────────────────────
    #[test]
    fn coca_loss_weighted_sum_matches() {
        let coca = make_coca(6);
        let c = 1.2_f32;
        let g = 3.4_f32;
        let lam = 0.3_f32;
        let l = coca.coca_loss(c, g, lam).expect("coca_loss should succeed");
        let expected = lam * c + (1.0 - lam) * g;
        assert!((l - expected).abs() < 1e-6, "{l} vs {expected}");
    }

    // ── 7: lambda = 0 → captioning only ─────────────────────────────────────
    #[test]
    fn coca_loss_lambda_zero_is_captioning() {
        let coca = make_coca(7);
        let l = coca
            .coca_loss(99.0, 7.0, 0.0)
            .expect("coca_loss should succeed");
        assert!((l - 7.0).abs() < 1e-6);
    }

    // ── 8: lambda = 1 → contrastive only ────────────────────────────────────
    #[test]
    fn coca_loss_lambda_one_is_contrastive() {
        let coca = make_coca(8);
        let l = coca
            .coca_loss(2.5, 99.0, 1.0)
            .expect("coca_loss should succeed");
        assert!((l - 2.5).abs() < 1e-6);
    }

    // ── 9: deterministic given seed ─────────────────────────────────────────
    #[test]
    fn deterministic_given_seed() {
        let a = make_coca(9);
        let b = make_coca(9);
        let d = a.cfg.d_model;
        let img = vec![0.3_f32; 4 * d];
        let out_a = a.pool_image(&img, 4).expect("pool_image should succeed");
        let out_b = b.pool_image(&img, 4).expect("pool_image should succeed");
        assert_eq!(out_a, out_b);
    }

    // ── 10: d_model % n_heads != 0 → Err ────────────────────────────────────
    #[test]
    fn d_model_not_divisible_errors() {
        let mut rng = LcgRng::new(10);
        let cfg = CoCaConfig {
            d_model: 10,
            n_heads: 3,
            vocab_size: 16,
            temperature: 0.07,
        };
        let err = CoCa::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    // ── 11: temperature ≤ 0 → Err ───────────────────────────────────────────
    #[test]
    fn non_positive_temperature_errors() {
        let mut rng = LcgRng::new(11);
        let cfg = CoCaConfig {
            d_model: 8,
            n_heads: 2,
            vocab_size: 16,
            temperature: 0.0,
        };
        let err = CoCa::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidTemperature { .. }));
    }

    // ── 12: lambda outside [0, 1] → Err ─────────────────────────────────────
    #[test]
    fn lambda_out_of_range_errors() {
        let coca = make_coca(12);
        assert!(coca.coca_loss(1.0, 1.0, -0.1).is_err());
        assert!(coca.coca_loss(1.0, 1.0, 1.1).is_err());
    }

    // ── 13: length mismatches → Err ─────────────────────────────────────────
    #[test]
    fn image_tokens_length_mismatch_errors() {
        let coca = make_coca(13);
        let d = coca.cfg.d_model;
        let img = vec![0.0_f32; 3 * d]; // pretend 4 tokens
        let err = coca.pool_image(&img, 4).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));

        let err2 = coca
            .contrastive_loss(&vec![0.0_f32; 2 * d], &vec![0.0_f32; 3 * d], 3)
            .unwrap_err();
        assert!(matches!(err2, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 14: single text token works ─────────────────────────────────────────
    #[test]
    fn single_text_token_works() {
        let coca = make_coca(14);
        let d = coca.cfg.d_model;
        let text = vec![0.4_f32; d];
        let img = vec![0.2_f32; 3 * d];
        let out = coca
            .captioning_logits(&text, 1, &img, 3)
            .expect("captioning_logits should succeed");
        assert_eq!(out.len(), coca.cfg.vocab_size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 15: batch = 1 contrastive ───────────────────────────────────────────
    #[test]
    fn batch_one_contrastive_loss_is_zero() {
        // With a single pair the diagonal is the only entry, so cross-entropy
        // is exactly 0.
        let coca = make_coca(15);
        let d = coca.cfg.d_model;
        let a = vec![0.5_f32; d];
        let b = vec![0.3_f32; d];
        let loss = coca
            .contrastive_loss(&a, &b, 1)
            .expect("contrastive_loss should succeed");
        assert!(loss.abs() < 1e-5, "batch=1 loss should be ~0, got {loss}");
    }

    // ── 16: changing image_features changes captioning logits ───────────────
    #[test]
    fn changing_image_features_changes_logits() {
        let coca = make_coca(16);
        let d = coca.cfg.d_model;
        let text = vec![0.4_f32; 3 * d];
        let img_a = vec![0.1_f32; 4 * d];
        let mut img_b = vec![0.1_f32; 4 * d];
        for (i, v) in img_b.iter_mut().enumerate() {
            *v = 0.1 + (i as f32 * 0.05).sin();
        }
        let out_a = coca
            .captioning_logits(&text, 3, &img_a, 4)
            .expect("captioning_logits should succeed");
        let out_b = coca
            .captioning_logits(&text, 3, &img_b, 4)
            .expect("captioning_logits should succeed");
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "cross-attn should respond to image features, diff={diff}"
        );
    }

    // ── 17: vocab_size = 0 → Err ────────────────────────────────────────────
    #[test]
    fn vocab_size_zero_errors() {
        let mut rng = LcgRng::new(17);
        let cfg = CoCaConfig {
            d_model: 8,
            n_heads: 2,
            vocab_size: 0,
            temperature: 0.07,
        };
        let err = CoCa::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 18: d_model = 0 → Err ───────────────────────────────────────────────
    #[test]
    fn d_model_zero_errors() {
        let mut rng = LcgRng::new(18);
        let cfg = CoCaConfig {
            d_model: 0,
            n_heads: 1,
            vocab_size: 16,
            temperature: 0.07,
        };
        let err = CoCa::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 19: pool_image n_tokens = 0 → Err ───────────────────────────────────
    #[test]
    fn pool_image_zero_tokens_errors() {
        let coca = make_coca(19);
        let err = coca.pool_image(&[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    // ── 20: contrastive_loss batch = 0 → Err ────────────────────────────────
    #[test]
    fn contrastive_loss_zero_batch_errors() {
        let coca = make_coca(20);
        let err = coca.contrastive_loss(&[], &[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidBatchSize));
    }

    // ── 21: captioning_logits zero text/image → Err ─────────────────────────
    #[test]
    fn captioning_zero_text_or_image_errors() {
        let coca = make_coca(21);
        let d = coca.cfg.d_model;
        let img = vec![0.1_f32; 2 * d];
        let err = coca.captioning_logits(&[], 0, &img, 2).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));

        let text = vec![0.1_f32; 2 * d];
        let err2 = coca.captioning_logits(&text, 2, &[], 0).unwrap_err();
        assert!(matches!(err2, MultiModalError::EmptyInput));
    }

    // ── 22: output of pool_image is finite ──────────────────────────────────
    #[test]
    fn pool_image_finite() {
        let coca = make_coca(22);
        let d = coca.cfg.d_model;
        let img = vec![0.25_f32; 5 * d];
        let out = coca.pool_image(&img, 5).expect("pool_image should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 23: captioning_logits NaN/Inf-free ──────────────────────────────────
    #[test]
    fn captioning_logits_finite() {
        let coca = make_coca(23);
        let d = coca.cfg.d_model;
        let text = vec![0.1_f32; 4 * d];
        let img = vec![0.2_f32; 5 * d];
        let out = coca
            .captioning_logits(&text, 4, &img, 5)
            .expect("captioning_logits should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
