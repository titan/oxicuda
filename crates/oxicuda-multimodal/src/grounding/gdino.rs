//! Grounding-DINO — open-set detection by marrying DINO with grounded
//! language pre-training.
//!
//! Reference: Liu, Zeng, Ren, Li et al. 2023, *Grounding DINO: Marrying DINO
//! with Grounded Pre-Training for Open-Set Object Detection*.
//!
//! Compact-but-faithful CPU core of the three grounding ingredients:
//!
//! 1. **Cross-modality fusion.** Text token features and image patch features
//!    are fused by *bidirectional* cross-attention — text attends to image
//!    (text→image) and image attends to text (image→text) — so each modality
//!    is conditioned on the other before decoding.
//! 2. **Language-guided query selection.** The image features most similar to
//!    the text (highest dot-product alignment) are selected as the decoder
//!    queries, focusing detection on what the prompt actually mentions.
//! 3. **Box + alignment heads.** Every selected query predicts a normalized box
//!    `(cx, cy, w, h) ∈ [0, 1]` through a sigmoid MLP, plus a text-alignment
//!    score (its best dot product against the text tokens — the contrastive
//!    logit Grounding DINO uses in place of fixed class labels).
//!
//! Fusion reuses the shared masked multi-head attention; the box MLP and the
//! sigmoid reuse existing crate primitives.

use crate::cross_attn::cross_attention::{CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::masked_mha::{MhaArgs, mha_with_weights};
use crate::cross_attn::self_cross_block::LayerNorm;
use crate::error::{MmResult, MultiModalError};
use crate::fusion::gmu::sigmoid;
use crate::handle::LcgRng;

/// Bidirectional fusion output: `(text_out, img_out, t2i_weights, i2t_weights)`,
/// where the two weight matrices are `[n_text × n_img]` and `[n_img × n_text]`.
pub type FusionWithWeights = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`GroundingDino`].
#[derive(Debug, Clone)]
pub struct GroundingDinoConfig {
    /// Shared model width of the text and image features.
    pub d_model: usize,
    /// Cross-attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Number of decoder queries selected from the image features.
    pub n_queries: usize,
    /// Hidden width of the box-regression MLP.
    pub box_hidden: usize,
}

impl GroundingDinoConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            d_model: 8,
            n_heads: 2,
            n_queries: 3,
            box_hidden: 16,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 || self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_queries == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if self.box_hidden == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// All learnable parameters of [`GroundingDino`].
#[derive(Debug, Clone)]
pub struct GroundingDinoWeights {
    /// Text→image cross-attention.
    t2i_attn: CrossAttnWeights,
    /// Image→text cross-attention.
    i2t_attn: CrossAttnWeights,
    /// Pre-norm on text features.
    ln_text: LayerNorm,
    /// Pre-norm on image features.
    ln_img: LayerNorm,
    /// Box MLP layer 1, `[d_model × box_hidden]`.
    box_w1: Vec<f32>,
    /// Box MLP bias 1, `[box_hidden]`.
    box_b1: Vec<f32>,
    /// Box MLP layer 2, `[box_hidden × 4]`.
    box_w2: Vec<f32>,
    /// Box MLP bias 2, `[4]`.
    box_b2: Vec<f32>,
}

impl GroundingDinoWeights {
    fn random(cfg: &GroundingDinoConfig, rng: &mut LcgRng) -> MmResult<Self> {
        let d = cfg.d_model;
        let h = cfg.box_hidden;
        let attn_cfg = CrossAttnConfig::new(cfg.n_heads, d, 0.0)?;
        let gauss = |len: usize, scale: f32, rng: &mut LcgRng| {
            let mut v = vec![0.0_f32; len];
            rng.fill_normal(&mut v);
            for x in v.iter_mut() {
                *x *= scale;
            }
            v
        };
        Ok(Self {
            t2i_attn: CrossAttnWeights::random(&attn_cfg, rng),
            i2t_attn: CrossAttnWeights::random(&attn_cfg, rng),
            ln_text: LayerNorm::ones(d),
            ln_img: LayerNorm::ones(d),
            box_w1: gauss(d * h, 1.0 / (d as f32).sqrt(), rng),
            box_b1: vec![0.0_f32; h],
            box_w2: gauss(h * 4, 1.0 / (h as f32).sqrt(), rng),
            box_b2: vec![0.0_f32; 4],
        })
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Grounding-DINO open-set detector head.
#[derive(Debug, Clone)]
pub struct GroundingDino {
    cfg: GroundingDinoConfig,
    weights: GroundingDinoWeights,
}

impl GroundingDino {
    /// Construct a detector with deterministically random weights.
    pub fn new(cfg: GroundingDinoConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = GroundingDinoWeights::random(&cfg, rng)?;
        Ok(Self { cfg, weights })
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &GroundingDinoConfig {
        &self.cfg
    }

    // ── Cross-modality fusion ─────────────────────────────────────────────────

    /// Bidirectional cross-attention fusion.
    ///
    /// - `text_feat`: `[n_text × d_model]`, `img_feat`: `[n_img × d_model]`.
    /// - Returns `(text_out, img_out)` of the same shapes, each the input plus
    ///   the cross-attended context from the other modality (residual).
    pub fn fuse(
        &self,
        text_feat: &[f32],
        img_feat: &[f32],
        n_text: usize,
        n_img: usize,
    ) -> MmResult<(Vec<f32>, Vec<f32>)> {
        let (text_out, img_out, _, _) = self.fuse_attention(text_feat, img_feat, n_text, n_img)?;
        Ok((text_out, img_out))
    }

    /// Like [`Self::fuse`] but also returns both attention weight matrices:
    /// `t2i_weights` `[n_text × n_img]` (text→image) and `i2t_weights`
    /// `[n_img × n_text]` (image→text). Each row sums to 1.
    pub fn fuse_attention(
        &self,
        text_feat: &[f32],
        img_feat: &[f32],
        n_text: usize,
        n_img: usize,
    ) -> MmResult<FusionWithWeights> {
        let d = self.cfg.d_model;
        if n_text == 0 || n_img == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if text_feat.len() != n_text * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_text * d,
                got: text_feat.len(),
            });
        }
        if img_feat.len() != n_img * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_img * d,
                got: img_feat.len(),
            });
        }

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;
        let text_n = self.weights.ln_text.forward(text_feat, n_text)?;
        let img_n = self.weights.ln_img.forward(img_feat, n_img)?;

        // text → image.
        let t2i_args = MhaArgs {
            query: &text_n,
            key: &img_n,
            value: &img_n,
            q_len: n_text,
            kv_len: n_img,
            causal: false,
        };
        let (t_ctx, t2i_w) = mha_with_weights(&t2i_args, &attn_cfg, &self.weights.t2i_attn)?;

        // image → text.
        let i2t_args = MhaArgs {
            query: &img_n,
            key: &text_n,
            value: &text_n,
            q_len: n_img,
            kv_len: n_text,
            causal: false,
        };
        let (i_ctx, i2t_w) = mha_with_weights(&i2t_args, &attn_cfg, &self.weights.i2t_attn)?;

        let text_out: Vec<f32> = text_feat
            .iter()
            .zip(t_ctx.iter())
            .map(|(a, b)| a + b)
            .collect();
        let img_out: Vec<f32> = img_feat
            .iter()
            .zip(i_ctx.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok((text_out, img_out, t2i_w, i2t_w))
    }

    // ── Language-guided query selection ──────────────────────────────────────

    /// Select the `k` image features most aligned with the text (highest best
    /// dot product against any text token), returning their indices (sorted by
    /// descending alignment, ties broken by ascending index) and the gathered
    /// feature rows `[k' × d_model]`, where `k' = min(k, n_img)`.
    pub fn select_queries(
        &self,
        img_feat: &[f32],
        text_feat: &[f32],
        n_img: usize,
        n_text: usize,
        k: usize,
    ) -> MmResult<(Vec<usize>, Vec<f32>)> {
        let d = self.cfg.d_model;
        if n_img == 0 || n_text == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if img_feat.len() != n_img * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_img * d,
                got: img_feat.len(),
            });
        }
        if text_feat.len() != n_text * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_text * d,
                got: text_feat.len(),
            });
        }

        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(n_img);
        for i in 0..n_img {
            let mut best = f32::NEG_INFINITY;
            for t in 0..n_text {
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += img_feat[i * d + di] * text_feat[t * d + di];
                }
                if dot > best {
                    best = dot;
                }
            }
            scored.push((best, i));
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        let kk = k.min(n_img);
        let indices: Vec<usize> = scored.iter().take(kk).map(|&(_, i)| i).collect();
        let mut feats = vec![0.0_f32; kk * d];
        for (qi, &idx) in indices.iter().enumerate() {
            feats[qi * d..(qi + 1) * d].copy_from_slice(&img_feat[idx * d..(idx + 1) * d]);
        }
        Ok((indices, feats))
    }

    // ── Heads ─────────────────────────────────────────────────────────────────

    /// Predict normalized boxes `[k × 4]` (`cx, cy, w, h ∈ [0, 1]`) from query
    /// features `[k × d_model]` via a 2-layer ReLU MLP with a sigmoid output.
    fn box_head(&self, query_feats: &[f32], k: usize) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let h = self.cfg.box_hidden;
        if query_feats.len() != k * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: k * d,
                got: query_feats.len(),
            });
        }
        let mut boxes = vec![0.0_f32; k * 4];
        for q in 0..k {
            let mut hidden = vec![0.0_f32; h];
            for hi in 0..h {
                let mut acc = self.weights.box_b1[hi];
                for di in 0..d {
                    acc += query_feats[q * d + di] * self.weights.box_w1[di * h + hi];
                }
                hidden[hi] = acc.max(0.0); // ReLU
            }
            for o in 0..4 {
                let mut acc = self.weights.box_b2[o];
                for hi in 0..h {
                    acc += hidden[hi] * self.weights.box_w2[hi * 4 + o];
                }
                boxes[q * 4 + o] = sigmoid(acc); // → [0, 1]
            }
        }
        Ok(boxes)
    }

    /// Per-query text-alignment score: each query's best dot product against
    /// any text token.
    fn alignment_scores(
        &self,
        query_feats: &[f32],
        k: usize,
        text_feat: &[f32],
        n_text: usize,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut scores = vec![0.0_f32; k];
        for q in 0..k {
            let mut best = f32::NEG_INFINITY;
            for t in 0..n_text {
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += query_feats[q * d + di] * text_feat[t * d + di];
                }
                if dot > best {
                    best = dot;
                }
            }
            scores[q] = best;
        }
        scores
    }

    /// Full forward pass: fuse the modalities, select language-guided queries,
    /// and predict boxes + alignment scores.
    ///
    /// Returns `(boxes, scores)` where `boxes` is `[k' × 4]` with every value in
    /// `[0, 1]` and `scores` is `[k']`, with `k' = min(n_queries, n_img)`.
    pub fn forward(
        &self,
        text_feat: &[f32],
        img_feat: &[f32],
        n_text: usize,
        n_img: usize,
    ) -> MmResult<(Vec<f32>, Vec<f32>)> {
        let (text_out, img_out) = self.fuse(text_feat, img_feat, n_text, n_img)?;
        let (_, selected) =
            self.select_queries(&img_out, &text_out, n_img, n_text, self.cfg.n_queries)?;
        let k = selected.len() / self.cfg.d_model;
        let boxes = self.box_head(&selected, k)?;
        let scores = self.alignment_scores(&selected, k, &text_out, n_text);
        Ok((boxes, scores))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> GroundingDino {
        let mut rng = LcgRng::new(seed);
        GroundingDino::new(GroundingDinoConfig::tiny(), &mut rng).expect("construct GroundingDino")
    }

    fn feats(n: usize, d: usize, phase: f32) -> Vec<f32> {
        (0..n * d)
            .map(|i| (i as f32 * 0.019 + phase).sin() * 0.5)
            .collect()
    }

    // 1 ── Both cross-attention directions are normalised per query.
    #[test]
    fn bidirectional_weights_sum_to_one() {
        let m = model(1);
        let d = m.config().d_model;
        let n_text = 3;
        let n_img = 6;
        let txt = feats(n_text, d, 0.1);
        let img = feats(n_img, d, 0.7);
        let (_, _, t2i, i2t) = m
            .fuse_attention(&txt, &img, n_text, n_img)
            .expect("fuse_attention should succeed");
        assert_eq!(t2i.len(), n_text * n_img);
        assert_eq!(i2t.len(), n_img * n_text);
        for q in 0..n_text {
            let s: f32 = t2i[q * n_img..(q + 1) * n_img].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "t2i row {q} sum {s}");
        }
        for q in 0..n_img {
            let s: f32 = i2t[q * n_text..(q + 1) * n_text].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "i2t row {q} sum {s}");
        }
    }

    // 2 ── Query selection returns the genuinely most text-similar features.
    #[test]
    fn select_queries_picks_top_k_handbuilt() {
        let m = model(2);
        let d = m.config().d_model; // 8
        let n_text = 1;
        let mut text = vec![0.0_f32; n_text * d];
        text[0] = 1.0; // text points along axis 0
        let n_img = 6;
        let mut img = vec![0.0_f32; n_img * d];
        // Alignment (dot with text) per row is its channel-0 value.
        img[2 * d] = 10.0; // row 2 — strongest
        img[5 * d] = 6.0; // row 5 — second
        img[d] = 1.0; // row 1 — weak
        let (idx, sel) = m
            .select_queries(&img, &text, n_img, n_text, 2)
            .expect("select_queries should succeed");
        assert_eq!(idx, vec![2, 5], "top-2 by text similarity");
        assert_eq!(sel.len(), 2 * d);
        // The gathered features are exactly rows 2 and 5.
        assert!((sel[0] - 10.0).abs() < 1e-6);
        assert!((sel[d] - 6.0).abs() < 1e-6);
    }

    // 3 ── Changing the text prompt re-routes selection (real grounding).
    #[test]
    fn changing_text_changes_selected_queries() {
        let m = model(3);
        let d = m.config().d_model;
        let n_img = 6;
        let mut img = vec![0.0_f32; n_img * d];
        img[0] = 5.0; // rows 0,1 align with axis 0
        img[d] = 4.0;
        img[2 * d + 1] = 5.0; // rows 2,3 align with axis 1
        img[3 * d + 1] = 4.0;
        let mut text_a = vec![0.0_f32; d];
        text_a[0] = 1.0;
        let mut text_b = vec![0.0_f32; d];
        text_b[1] = 1.0;
        let (idx_a, _) = m
            .select_queries(&img, &text_a, n_img, 1, 2)
            .expect("select_queries should succeed");
        let (idx_b, _) = m
            .select_queries(&img, &text_b, n_img, 1, 2)
            .expect("select_queries should succeed");
        assert_eq!(idx_a, vec![0, 1]);
        assert_eq!(idx_b, vec![2, 3]);
        assert_ne!(idx_a, idx_b);
    }

    // 4 ── Predicted boxes are all within [0, 1] (sigmoid) and shaped [k × 4].
    #[test]
    fn boxes_in_unit_range() {
        let m = model(4);
        let cfg = m.config();
        let d = cfg.d_model;
        let n_text = 2;
        let n_img = 8;
        let txt = feats(n_text, d, 0.3);
        let img = feats(n_img, d, 1.1);
        let (boxes, _) = m
            .forward(&txt, &img, n_text, n_img)
            .expect("forward should succeed");
        let k = cfg.n_queries.min(n_img);
        assert_eq!(boxes.len(), k * 4);
        for &v in &boxes {
            assert!((0.0..=1.0).contains(&v), "box value {v} out of [0,1]");
        }
    }

    // 5 ── Changing the text prompt changes the predicted boxes end-to-end.
    #[test]
    fn changing_text_changes_boxes() {
        let m = model(5);
        let cfg = m.config();
        let d = cfg.d_model;
        let n_text = 2;
        let n_img = 8;
        let img = feats(n_img, d, 0.9);
        let (boxes_a, _) = m
            .forward(&feats(n_text, d, 0.0), &img, n_text, n_img)
            .expect("value should be present");
        let (boxes_b, _) = m
            .forward(&feats(n_text, d, 2.0), &img, n_text, n_img)
            .expect("value should be present");
        let diff: f32 = boxes_a
            .iter()
            .zip(boxes_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "text prompt should change boxes, diff={diff}");
    }

    // 6 ── Forward is deterministic and its scores are finite.
    #[test]
    fn forward_deterministic_and_scores_finite() {
        let m1 = model(6);
        let m2 = model(6);
        let cfg = m1.config();
        let d = cfg.d_model;
        let n_text = 2;
        let n_img = 7;
        let txt = feats(n_text, d, 0.5);
        let img = feats(n_img, d, 1.3);
        let (b1, s1) = m1
            .forward(&txt, &img, n_text, n_img)
            .expect("forward should succeed");
        let (b2, s2) = m2
            .forward(&txt, &img, n_text, n_img)
            .expect("forward should succeed");
        assert_eq!(b1, b2);
        assert_eq!(s1, s2);
        assert!(s1.iter().all(|v| v.is_finite()));
        assert_eq!(s1.len(), cfg.n_queries.min(n_img));
    }

    // 7 ── Fusion preserves shapes, is residual and finite.
    #[test]
    fn fuse_output_shapes_finite() {
        let m = model(7);
        let d = m.config().d_model;
        let n_text = 4;
        let n_img = 5;
        let txt = feats(n_text, d, 0.2);
        let img = feats(n_img, d, 0.8);
        let (text_out, img_out) = m
            .fuse(&txt, &img, n_text, n_img)
            .expect("fuse should succeed");
        assert_eq!(text_out.len(), n_text * d);
        assert_eq!(img_out.len(), n_img * d);
        assert!(text_out.iter().chain(img_out.iter()).all(|v| v.is_finite()));
    }

    // 8 ── select_queries clamps k to n_img.
    #[test]
    fn select_queries_clamps_k() {
        let m = model(8);
        let d = m.config().d_model;
        let n_img = 3;
        let img = feats(n_img, d, 0.4);
        let txt = feats(1, d, 0.0);
        let (idx, sel) = m
            .select_queries(&img, &txt, n_img, 1, 10)
            .expect("select_queries should succeed");
        assert_eq!(idx.len(), n_img);
        assert_eq!(sel.len(), n_img * d);
    }

    // 9 ── Invalid config (heads not dividing d_model) is rejected.
    #[test]
    fn invalid_heads_errors() {
        let mut cfg = GroundingDinoConfig::tiny();
        cfg.n_heads = 3; // 8 % 3 != 0
        let mut rng = LcgRng::new(9);
        let err = GroundingDino::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    // 10 ── Mismatched feature length is rejected.
    #[test]
    fn dimension_mismatch_errors() {
        let m = model(10);
        let d = m.config().d_model;
        let err = m
            .fuse(&vec![0.0_f32; 3 * d], &vec![0.0_f32; 2 * d], 4, 2)
            .unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }
}
