//! OWL-ViT — a faithful CPU reference of the open-vocabulary detector from
//! Minderer et al. 2022, *"Simple Open-Vocabulary Object Detection with Vision
//! Transformers"*.
//!
//! Unlike a closed-set detector with a fixed list of class logits, OWL-ViT
//! casts detection as **per-patch image-text matching**:
//!
//! 1. **Image encoder** — a ViT (patch embed → positional embed → transformer
//!    encoder) produces one **object embedding per patch**. Every patch is a
//!    detection candidate (no separate region proposal step). We reuse
//!    [`crate::patch_embed::PatchEmbed`] and [`crate::vit::ViTEncoder`].
//! 2. **Class head** — each patch embedding is linearly projected into the
//!    image-text joint space (and L2-normalised), then scored against a set of
//!    text *query* embeddings by **cosine similarity**. Because the class set is
//!    supplied as text embeddings at inference time, the vocabulary is open: any
//!    text query can be scored without retraining.
//! 3. **Box head** — a small MLP on each patch embedding predicts a normalised
//!    box `(cx, cy, w, h) ∈ [0, 1]⁴`. Following OWL-ViT, the predicted centre is
//!    an **offset added to the patch's grid-cell centre prior**, so a patch at
//!    grid `(i, j)` defaults to a box centred on its own location and the MLP
//!    only has to learn a small residual.
//!
//! ## Tensor layout
//! Patch tokens are flat `[n_patches · embed_dim]` row-major, patch `t`
//! corresponding to grid cell `(t / grid, t % grid)` — matching the patch
//! ordering of [`crate::patch_embed::PatchEmbed::forward`]. Boxes are returned
//! as `[n_patches · 4]` row-major in `(cx, cy, w, h)` order.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    patch_embed::{PatchEmbed, PatchEmbedConfig},
    vit::vit_block::{gelu_exact, linear},
    vit::{ViTConfig, ViTEncoder, ViTEncoderConfig},
};

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the OWL-ViT detector.
#[derive(Debug, Clone)]
pub struct OwlVitConfig {
    /// Underlying ViT image-encoder hyper-parameters.
    pub vit_config: ViTConfig,
    /// Dimension of the shared image-text joint space (text-query dim).
    pub joint_dim: usize,
    /// Hidden width of the box-regression MLP.
    pub box_hidden: usize,
}

impl OwlVitConfig {
    /// Create and validate a config.
    ///
    /// # Errors
    /// - [`VisionError::InvalidProjDim`] if `joint_dim == 0`.
    /// - [`VisionError::InvalidEmbedDim`] if `box_hidden == 0`.
    pub fn new(vit_config: ViTConfig, joint_dim: usize, box_hidden: usize) -> VisionResult<Self> {
        if joint_dim == 0 {
            return Err(VisionError::InvalidProjDim(joint_dim));
        }
        if box_hidden == 0 {
            return Err(VisionError::InvalidEmbedDim(box_hidden));
        }
        Ok(Self {
            vit_config,
            joint_dim,
            box_hidden,
        })
    }

    /// A tiny config for tests: tiny ViT backbone, joint dim 16, box hidden 32.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vit_config: ViTConfig::tiny(),
            joint_dim: 16,
            box_hidden: 32,
        }
    }

    /// Number of patches (= number of detection candidates).
    #[must_use]
    pub fn n_patches(&self) -> usize {
        self.vit_config.n_patches()
    }

    /// Grid size (patches per spatial dimension).
    #[must_use]
    pub fn grid_size(&self) -> usize {
        self.vit_config.img_size / self.vit_config.patch_size
    }
}

// ─── Detection output ───────────────────────────────────────────────────────────

/// Per-patch detection output.
#[derive(Debug, Clone)]
pub struct OwlVitOutput {
    /// Patch object embeddings projected into the joint space, L2-normalised:
    /// flat `[n_patches · joint_dim]`.
    pub class_embeddings: Vec<f32>,
    /// Predicted boxes `(cx, cy, w, h)` per patch, all coords in `[0, 1]`:
    /// flat `[n_patches · 4]`.
    pub boxes: Vec<f32>,
    /// Number of patches.
    pub n_patches: usize,
    /// Joint-space dimension.
    pub joint_dim: usize,
}

impl OwlVitOutput {
    /// Cosine-similarity scores of every patch against every text query.
    ///
    /// `text_queries` is flat `[n_queries · joint_dim]`. The patch embeddings
    /// are already unit-norm, so each text query is L2-normalised here and the
    /// score reduces to a dot product — i.e. the open-vocabulary class score.
    ///
    /// Returns flat `[n_patches · n_queries]` row-major.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `text_queries.len()` is not a
    ///   multiple of `joint_dim`.
    pub fn score_queries(&self, text_queries: &[f32]) -> VisionResult<Vec<f32>> {
        let d = self.joint_dim;
        if text_queries.is_empty() || text_queries.len() % d != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: d,
                got: text_queries.len() % d,
            });
        }
        let n_q = text_queries.len() / d;

        // Pre-normalise each text query (cosine ⇒ scale-invariant).
        let mut q_unit = vec![0.0f32; n_q * d];
        for q in 0..n_q {
            let src = &text_queries[q * d..(q + 1) * d];
            let norm: f32 = src.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let inv = 1.0 / norm.max(1e-12);
            let dst = &mut q_unit[q * d..(q + 1) * d];
            for k in 0..d {
                dst[k] = src[k] * inv;
            }
        }

        let mut scores = vec![0.0f32; self.n_patches * n_q];
        for p in 0..self.n_patches {
            let pe = &self.class_embeddings[p * d..(p + 1) * d];
            for q in 0..n_q {
                let qe = &q_unit[q * d..(q + 1) * d];
                let dot: f32 = pe.iter().zip(qe.iter()).map(|(&a, &b)| a * b).sum();
                scores[p * n_q + q] = dot;
            }
        }
        Ok(scores)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────────

/// The OWL-ViT open-vocabulary detector.
pub struct OwlVit {
    /// Configuration.
    pub config: OwlVitConfig,
    /// Strided Conv2D patch embedder.
    patch_embed: PatchEmbed,
    /// Learned positional embedding for the patch tokens: `[n_patches · embed_dim]`.
    pos_embed: Vec<f32>,
    /// ViT transformer encoder.
    encoder: ViTEncoder,
    /// Class-head projection `[joint_dim · embed_dim]` (patch emb → joint space).
    class_proj_weight: Vec<f32>,
    /// Class-head projection bias `[joint_dim]`.
    class_proj_bias: Vec<f32>,
    /// Box-head MLP first layer `[box_hidden · embed_dim]`.
    box_w1: Vec<f32>,
    box_b1: Vec<f32>,
    /// Box-head MLP second layer `[4 · box_hidden]` → (cx_off, cy_off, w, h).
    box_w2: Vec<f32>,
    box_b2: Vec<f32>,
}

impl OwlVit {
    /// Construct an OWL-ViT detector with Gaussian-initialised weights.
    ///
    /// # Errors
    /// Propagates patch / encoder / config validation errors.
    pub fn new(cfg: OwlVitConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let vc = &cfg.vit_config;
        let e = vc.embed_dim;

        let pe_cfg = PatchEmbedConfig::new(vc.img_size, vc.patch_size, vc.in_chans, e)?;
        let patch_embed = PatchEmbed::new(pe_cfg, rng);

        let n_patches = cfg.n_patches();
        let mut pos_embed = vec![0.0f32; n_patches * e];
        rng.fill_normal(&mut pos_embed);
        for v in &mut pos_embed {
            *v *= 0.02;
        }

        let enc_cfg = ViTEncoderConfig::new(e, vc.n_heads, vc.mlp_ratio, vc.depth)?;
        let encoder = ViTEncoder::new(enc_cfg, rng)?;

        let fill = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };

        let j = cfg.joint_dim;
        let bh = cfg.box_hidden;
        let scale_e = 1.0 / (e as f32).sqrt();
        let scale_bh = 1.0 / (bh as f32).sqrt();

        let class_proj_weight = fill(rng, j * e, scale_e);
        let class_proj_bias = vec![0.0f32; j];
        let box_w1 = fill(rng, bh * e, scale_e);
        let box_b1 = vec![0.0f32; bh];
        let box_w2 = fill(rng, 4 * bh, scale_bh);
        // Initialise box-head output bias to zero so that, before training, the
        // predicted centre offset is ~0 and the box centre equals the grid
        // prior, and (w, h) start near the default sigmoid(0) = 0.5.
        let box_b2 = vec![0.0f32; 4];

        Ok(Self {
            config: cfg,
            patch_embed,
            pos_embed,
            encoder,
            class_proj_weight,
            class_proj_bias,
            box_w1,
            box_b1,
            box_w2,
            box_b2,
        })
    }

    /// Run the backbone and produce per-patch object embeddings `[n_patches · e]`.
    fn patch_object_embeddings(&self, image: &[f32]) -> VisionResult<Vec<f32>> {
        let e = self.config.vit_config.embed_dim;
        let n_patches = self.config.n_patches();

        // Patch embed → [n_patches, e].
        let mut tokens = self.patch_embed.forward(image)?;
        if tokens.len() != n_patches * e {
            return Err(VisionError::DimensionMismatch {
                expected: n_patches * e,
                got: tokens.len(),
            });
        }
        // Add positional embedding (no CLS token: every patch is a candidate).
        for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
            *t += p;
        }
        // Transformer encoder → [n_patches, e].
        self.encoder.forward(&tokens, n_patches)
    }

    /// Forward pass: produce per-patch class embeddings and boxes.
    ///
    /// `image` is flat CHW `[in_chans · img_size · img_size]`.
    ///
    /// # Errors
    /// Propagates backbone / dimension errors.
    pub fn forward(&self, image: &[f32]) -> VisionResult<OwlVitOutput> {
        let cfg = &self.config;
        let e = cfg.vit_config.embed_dim;
        let j = cfg.joint_dim;
        let bh = cfg.box_hidden;
        let n_patches = cfg.n_patches();
        let grid = cfg.grid_size();

        let feats = self.patch_object_embeddings(image)?;

        // ── Class head: project to joint space, L2-normalise. ──
        let proj = linear(&feats, &self.class_proj_weight, &self.class_proj_bias, e, j);
        let mut class_embeddings = proj;
        for p in 0..n_patches {
            let row = &mut class_embeddings[p * j..(p + 1) * j];
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let inv = 1.0 / norm.max(1e-12);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }

        // ── Box head: MLP → (cx_off, cy_off, w, h_logits), add grid prior. ──
        let hidden = linear(&feats, &self.box_w1, &self.box_b1, e, bh);
        let hidden: Vec<f32> = hidden.into_iter().map(gelu_exact).collect();
        let raw = linear(&hidden, &self.box_w2, &self.box_b2, bh, 4);

        let mut boxes = vec![0.0f32; n_patches * 4];
        let inv_grid = 1.0 / grid as f32;
        for p in 0..n_patches {
            let gy = p / grid;
            let gx = p % grid;
            // Grid-cell centre prior in normalised coords.
            let prior_cx = (gx as f32 + 0.5) * inv_grid;
            let prior_cy = (gy as f32 + 0.5) * inv_grid;

            let r = &raw[p * 4..(p + 1) * 4];
            // Centre = prior + tanh(offset) * (half a cell), then clamp to [0,1].
            // tanh keeps the learned offset bounded so the prior dominates at
            // init (offset ≈ 0 ⇒ centre ≈ prior).
            let cx = (prior_cx + (r[0].tanh()) * (0.5 * inv_grid)).clamp(0.0, 1.0);
            let cy = (prior_cy + (r[1].tanh()) * (0.5 * inv_grid)).clamp(0.0, 1.0);
            // Width / height via sigmoid ⇒ guaranteed in (0, 1).
            let bw = sigmoid(r[2]);
            let bh_ = sigmoid(r[3]);
            let o = &mut boxes[p * 4..(p + 1) * 4];
            o[0] = cx;
            o[1] = cy;
            o[2] = bw;
            o[3] = bh_;
        }

        if class_embeddings.iter().any(|v| !v.is_finite()) || boxes.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("owl-vit output"));
        }

        Ok(OwlVitOutput {
            class_embeddings,
            boxes,
            n_patches,
            joint_dim: j,
        })
    }
}

/// Numerically-stable logistic sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector(seed: u64) -> OwlVit {
        let mut rng = LcgRng::new(seed);
        OwlVit::new(OwlVitConfig::tiny(), &mut rng).expect("detector ok")
    }

    fn image(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0f32; len];
        rng.fill_normal(&mut v);
        v
    }

    // ── (a) Box shape & coords in [0,1] ──────────────────────────────────────────

    #[test]
    fn boxes_shape_and_range() {
        let det = make_detector(1);
        let vc = &det.config.vit_config;
        let img = image(2, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("forward ok");
        assert_eq!(
            out.boxes.len(),
            out.n_patches * 4,
            "box tensor must be [n_patches,4]"
        );
        assert_eq!(out.n_patches, det.config.n_patches());
        for (i, &c) in out.boxes.iter().enumerate() {
            assert!((0.0..=1.0).contains(&c), "box coord {i} out of [0,1]: {c}");
        }
    }

    // ── (b) Class score == manual normalised dot product (cosine) ────────────────

    #[test]
    fn score_equals_manual_cosine() {
        let det = make_detector(3);
        let vc = &det.config.vit_config;
        let img = image(4, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let d = out.joint_dim;
        // One arbitrary (non-unit) text query.
        let mut rng = LcgRng::new(99);
        let mut query = vec![0.0f32; d];
        rng.fill_normal(&mut query);

        let scores = out.score_queries(&query).expect("ok");
        assert_eq!(scores.len(), out.n_patches);

        // Manually compute cosine for patch 0.
        let pe = &out.class_embeddings[0..d]; // already unit-norm
        let qn: f32 = query.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let dot: f32 = pe.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
        let manual = dot / qn; // |pe| == 1
        assert!(
            (scores[0] - manual).abs() < 1e-5,
            "score must equal cosine(patch, query); got {} vs {manual}",
            scores[0]
        );
    }

    // ── (c) A query aligned to a patch yields the max score at that patch ────────

    #[test]
    fn aligned_query_maximised_at_its_patch() {
        let det = make_detector(5);
        let vc = &det.config.vit_config;
        let img = image(6, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let d = out.joint_dim;
        let target_patch = 7usize.min(out.n_patches - 1);

        // Build a text query equal to the target patch's (unit) embedding.
        let query = out.class_embeddings[target_patch * d..(target_patch + 1) * d].to_vec();
        let scores = out.score_queries(&query).expect("ok");

        // The aligned patch should attain the maximum (its self-cosine == 1).
        let mut best = 0usize;
        for p in 1..out.n_patches {
            if scores[p] > scores[best] {
                best = p;
            }
        }
        assert_eq!(
            best, target_patch,
            "query built from patch {target_patch} should score highest there"
        );
        assert!(
            (scores[target_patch] - 1.0).abs() < 1e-4,
            "self-cosine of the aligned patch must be ≈1; got {}",
            scores[target_patch]
        );
    }

    // ── (d) An orthogonal query yields low scores everywhere ─────────────────────

    #[test]
    fn orthogonal_query_low_scores_everywhere() {
        let det = make_detector(7);
        let vc = &det.config.vit_config;
        let img = image(8, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let d = out.joint_dim;

        // Construct a query orthogonal to patch 0's embedding via Gram-Schmidt
        // against a random seed vector.
        let p0 = out.class_embeddings[0..d].to_vec();
        let mut rng = LcgRng::new(123);
        let mut q = vec![0.0f32; d];
        rng.fill_normal(&mut q);
        let proj: f32 = q.iter().zip(p0.iter()).map(|(&a, &b)| a * b).sum();
        for k in 0..d {
            q[k] -= proj * p0[k];
        }
        // Cosine of q with patch 0 must now be ≈ 0.
        let scores = out.score_queries(&q).expect("ok");
        assert!(
            scores[0].abs() < 1e-4,
            "query orthogonalised against patch 0 must score ≈0 there; got {}",
            scores[0]
        );
        // And every score is a valid cosine in [-1, 1].
        for &s in &scores {
            assert!(
                (-1.0 - 1e-4..=1.0 + 1e-4).contains(&s),
                "cosine out of range: {s}"
            );
        }
    }

    // ── (e) Box centre prior: at init, centre defaults near the grid location ────

    #[test]
    fn box_center_prior_near_grid_cell() {
        // With box_b2 = 0 and bounded tanh offsets, the predicted centre must
        // lie within half a cell of the grid-cell centre prior for every patch.
        let det = make_detector(9);
        let vc = &det.config.vit_config;
        let grid = det.config.grid_size();
        let img = image(10, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let inv_grid = 1.0 / grid as f32;
        for p in 0..out.n_patches {
            let gy = p / grid;
            let gx = p % grid;
            let prior_cx = (gx as f32 + 0.5) * inv_grid;
            let prior_cy = (gy as f32 + 0.5) * inv_grid;
            let cx = out.boxes[p * 4];
            let cy = out.boxes[p * 4 + 1];
            // Offset is tanh(·)*0.5*cell ⇒ |centre - prior| ≤ 0.5*cell (+eps).
            let tol = 0.5 * inv_grid + 1e-5;
            assert!(
                (cx - prior_cx).abs() <= tol,
                "patch {p}: cx {cx} too far from prior {prior_cx}"
            );
            assert!(
                (cy - prior_cy).abs() <= tol,
                "patch {p}: cy {cy} too far from prior {prior_cy}"
            );
        }
    }

    #[test]
    fn box_centers_track_distinct_grid_cells() {
        // Two patches in different grid columns must have different default
        // centre priors, so their predicted centres differ.
        let det = make_detector(11);
        let vc = &det.config.vit_config;
        let img = image(12, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let grid = det.config.grid_size();
        // Patch (0,0) vs patch (0, grid-1): far apart in x.
        let p_left = 0usize;
        let p_right = grid - 1;
        let cx_left = out.boxes[p_left * 4];
        let cx_right = out.boxes[p_right * 4];
        assert!(
            cx_right > cx_left,
            "rightmost-column patch centre x ({cx_right}) should exceed leftmost ({cx_left})"
        );
    }

    // ── (f) Scoring invariant to text-embedding scale (cosine) ───────────────────

    #[test]
    fn scoring_invariant_to_query_scale() {
        let det = make_detector(13);
        let vc = &det.config.vit_config;
        let img = image(14, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let d = out.joint_dim;
        let mut rng = LcgRng::new(55);
        let mut q = vec![0.0f32; d];
        rng.fill_normal(&mut q);
        let q_scaled: Vec<f32> = q.iter().map(|&v| v * 17.0).collect();

        let s1 = out.score_queries(&q).expect("ok");
        let s2 = out.score_queries(&q_scaled).expect("ok");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "cosine score must be invariant to query scale: {a} vs {b}"
            );
        }
    }

    // ── Determinism & error paths ────────────────────────────────────────────────

    #[test]
    fn forward_deterministic() {
        let det = make_detector(15);
        let vc = &det.config.vit_config;
        let img = image(16, vc.in_chans * vc.img_size * vc.img_size);
        let a = det.forward(&img).expect("ok");
        let b = det.forward(&img).expect("ok");
        assert_eq!(a.boxes, b.boxes);
        assert_eq!(a.class_embeddings, b.class_embeddings);
    }

    #[test]
    fn forward_wrong_image_size_errors() {
        let det = make_detector(17);
        let r = det.forward(&[0.0f32; 7]);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn score_queries_bad_dim_errors() {
        let det = make_detector(19);
        let vc = &det.config.vit_config;
        let img = image(20, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        // length not a multiple of joint_dim
        let bad = vec![0.0f32; out.joint_dim + 1];
        let r = out.score_queries(&bad);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn class_embeddings_are_unit_norm() {
        let det = make_detector(21);
        let vc = &det.config.vit_config;
        let img = image(22, vc.in_chans * vc.img_size * vc.img_size);
        let out = det.forward(&img).expect("ok");
        let d = out.joint_dim;
        for p in 0..out.n_patches {
            let row = &out.class_embeddings[p * d..(p + 1) * d];
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "patch {p} embedding not unit-norm: {norm}"
            );
        }
    }

    #[test]
    fn config_zero_joint_dim_errors() {
        let r = OwlVitConfig::new(ViTConfig::tiny(), 0, 32);
        assert!(matches!(r, Err(VisionError::InvalidProjDim(0))));
    }
}
