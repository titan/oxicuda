//! I-JEPA — Assran et al. 2023 — Image Joint Embedding Predictive Architecture.
//!
//! A context encoder processes the **visible** (unmasked) patches; a lightweight
//! predictor receives the context representations and predicts the representations
//! of **target** (masked) patches produced by a separate target encoder (EMA of
//! the context encoder). No pixel-level reconstruction is needed — the loss is
//! mean-L2 distance in the representation space.
//!
//! ```text
//!   context patches  ──► context encoder ──► h_ctx   [n_visible × d]
//!   (mean-pool h_ctx) ──► predictor MLP ──► ĥ_tgt   [1 × d, replicated n_target]
//!   target patches   ──► target encoder ──► h_tgt    [n_target × d]
//!   loss = mean_i ||ĥ_tgt_i − h_tgt_i||_2
//! ```
//!
//! Reference: "Self-Supervised Learning from Images with a Joint-Embedding
//! Predictive Architecture", Assran et al., CVPR 2023.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// Convenience alias — same concrete type used throughout this module.
pub type SslRng = LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for I-JEPA.
#[derive(Debug, Clone)]
pub struct IJepaConfig {
    /// Total number of image patches (e.g. 196 for ViT-B/16 on 224×224).
    pub n_patches: usize,
    /// Token / feature dimension.
    pub d_model: usize,
    /// Number of attention heads (stored for bookkeeping; not used in the
    /// pure-MLP CPU implementation).
    pub n_heads: usize,
    /// Depth of the context-encoder MLP tower.
    pub n_context_blocks: usize,
    /// Depth of the predictor MLP.
    pub n_predictor_layers: usize,
    /// Fraction of patches designated as *target* patches (default 0.25).
    /// Must be in `(0, 1)`.
    pub mask_ratio: f32,
}

impl Default for IJepaConfig {
    fn default() -> Self {
        Self {
            n_patches: 196,
            d_model: 64,
            n_heads: 8,
            n_context_blocks: 2,
            n_predictor_layers: 2,
            mask_ratio: 0.25,
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// I-JEPA model: context encoder + predictor, both implemented as MLP towers.
#[derive(Debug)]
pub struct IJepa {
    /// Weight matrices for each context-encoder layer `[d_model × d_model]`.
    context_encoder_w: Vec<Vec<f32>>,
    /// Bias vectors for each context-encoder layer `[d_model]`.
    context_encoder_b: Vec<Vec<f32>>,
    /// Weight matrices for each predictor layer `[d_model × d_model]`.
    predictor_w: Vec<Vec<f32>>,
    /// Bias vectors for each predictor layer `[d_model]`.
    predictor_b: Vec<Vec<f32>>,
    config: IJepaConfig,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Apply a single dense (linear) layer with optional ReLU activation.
///
/// * `input`  — `[n_tokens × d_in]` row-major
/// * `w`      — `[d_in × d_out]` row-major
/// * `b`      — `[d_out]`
/// * returns  — `[n_tokens × d_out]`
fn apply_mlp_layer(
    input: &[f32],
    w: &[f32],
    b: &[f32],
    n_tokens: usize,
    d_in: usize,
    d_out: usize,
    relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_tokens * d_out];
    for t in 0..n_tokens {
        for j in 0..d_out {
            let mut val = b[j];
            for i in 0..d_in {
                val += input[t * d_in + i] * w[i * d_out + j];
            }
            out[t * d_out + j] = if relu { val.max(0.0) } else { val };
        }
    }
    out
}

/// Xavier-uniform initialisation: scale = `1 / sqrt(fan_in)`.
fn xavier_init(buf: &mut [f32], fan_in: usize, rng: &mut SslRng) {
    let scale = if fan_in > 0 {
        1.0_f32 / (fan_in as f32).sqrt()
    } else {
        1.0
    };
    for v in buf.iter_mut() {
        // Map [0,1) → [-scale, scale)
        *v = (rng.next_f32() * 2.0 - 1.0) * scale;
    }
}

impl IJepa {
    // ─── Constructor ─────────────────────────────────────────────────────────

    /// Create a new [`IJepa`] with randomly initialised weights.
    ///
    /// # Errors
    /// - [`SslError::InvalidParameter`] — `n_patches == 0` or
    ///   `n_context_blocks == 0`.
    /// - [`SslError::InvalidFeatureDim`] — `d_model == 0`.
    /// - [`SslError::InvalidMaskRatio`] — `mask_ratio` not in `(0, 1)`.
    pub fn new(config: IJepaConfig, rng: &mut SslRng) -> SslResult<Self> {
        if config.n_patches == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_patches".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.d_model == 0 {
            return Err(SslError::InvalidFeatureDim);
        }
        if config.n_context_blocks == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_context_blocks".into(),
                reason: "must be > 0".into(),
            });
        }
        if !(config.mask_ratio > 0.0 && config.mask_ratio < 1.0) {
            return Err(SslError::InvalidMaskRatio {
                ratio: config.mask_ratio,
            });
        }

        let d = config.d_model;
        let n_ctx = config.n_context_blocks;
        let n_pred = config.n_predictor_layers.max(1);

        let init_layer = |rng: &mut SslRng| -> (Vec<f32>, Vec<f32>) {
            let mut w = vec![0.0_f32; d * d];
            xavier_init(&mut w, d, rng);
            let b = vec![0.0_f32; d];
            (w, b)
        };

        let mut context_encoder_w = Vec::with_capacity(n_ctx);
        let mut context_encoder_b = Vec::with_capacity(n_ctx);
        for _ in 0..n_ctx {
            let (w, b) = init_layer(rng);
            context_encoder_w.push(w);
            context_encoder_b.push(b);
        }

        let mut predictor_w = Vec::with_capacity(n_pred);
        let mut predictor_b = Vec::with_capacity(n_pred);
        for _ in 0..n_pred {
            let (w, b) = init_layer(rng);
            predictor_w.push(w);
            predictor_b.push(b);
        }

        Ok(Self {
            context_encoder_w,
            context_encoder_b,
            predictor_w,
            predictor_b,
            config,
        })
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Return the feature dimension `d_model`.
    #[must_use]
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    // ─── Forward passes ──────────────────────────────────────────────────────

    /// Run the context encoder over a set of visible patches.
    ///
    /// `patches` — `[n_visible × d_model]` row-major patch embeddings.
    /// `patch_ids` — indices of the visible patches (length `n_visible`).
    ///
    /// Returns `[n_visible × d_model]` context representations.
    ///
    /// # Errors
    /// - [`SslError::EmptyInput`] — no visible patches.
    /// - [`SslError::DimensionMismatch`] — `patches.len() != patch_ids.len() *
    ///   d_model`.
    pub fn encode_context(&self, patches: &[f32], patch_ids: &[usize]) -> SslResult<Vec<f32>> {
        let d = self.config.d_model;
        let n_visible = patch_ids.len();

        if n_visible == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_visible * d;
        if patches.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: patches.len(),
            });
        }

        let n_layers = self.context_encoder_w.len();
        let mut h = patches.to_vec();

        for layer in 0..n_layers {
            let relu = layer + 1 < n_layers; // ReLU on all but the last layer
            h = apply_mlp_layer(
                &h,
                &self.context_encoder_w[layer],
                &self.context_encoder_b[layer],
                n_visible,
                d,
                d,
                relu,
            );
        }
        Ok(h)
    }

    /// Predict target representations from context representations.
    ///
    /// Strategy: mean-pool `context_repr` (shape `[n_visible × d_model]`) to a
    /// single `d_model` vector, push it through the predictor MLP, then tile the
    /// result `n_target` times to produce `[n_target × d_model]`.
    ///
    /// # Errors
    /// - [`SslError::EmptyInput`] — `n_visible == 0` or `n_target == 0`.
    /// - [`SslError::DimensionMismatch`] — `context_repr.len() != n_visible *
    ///   d_model`.
    pub fn predict_targets(
        &self,
        context_repr: &[f32],
        _target_ids: &[usize],
        n_visible: usize,
        n_target: usize,
    ) -> SslResult<Vec<f32>> {
        let d = self.config.d_model;

        if n_visible == 0 || n_target == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_visible * d;
        if context_repr.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: context_repr.len(),
            });
        }

        // Mean-pool context representations → [1 × d]
        let inv_n = 1.0_f32 / n_visible as f32;
        let mut pooled = vec![0.0_f32; d];
        for t in 0..n_visible {
            for j in 0..d {
                pooled[j] += context_repr[t * d + j];
            }
        }
        for v in pooled.iter_mut() {
            *v *= inv_n;
        }

        // Run pooled token through predictor MLP — treated as single token
        let n_layers = self.predictor_w.len();
        let mut h = pooled;

        for layer in 0..n_layers {
            let relu = layer + 1 < n_layers;
            let next_h = apply_mlp_layer(
                &h,
                &self.predictor_w[layer],
                &self.predictor_b[layer],
                1,
                d,
                d,
                relu,
            );
            h = next_h;
        }

        // Tile result n_target times → [n_target × d]
        let mut out = Vec::with_capacity(n_target * d);
        for _ in 0..n_target {
            out.extend_from_slice(&h);
        }
        Ok(out)
    }

    /// Compute the I-JEPA mean-L2 loss between predicted and target patches.
    ///
    /// Loss = mean over `i` in `[0, n_target)` of `||predicted_i - target_i||_2`.
    ///
    /// # Errors
    /// - [`SslError::EmptyInput`] — `n_target == 0`.
    /// - [`SslError::DimensionMismatch`] — length mismatch.
    pub fn loss(&self, predicted: &[f32], target: &[f32], n_target: usize) -> SslResult<f32> {
        let d = self.config.d_model;

        if n_target == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_target * d;
        if predicted.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: predicted.len(),
            });
        }
        if target.len() != predicted.len() {
            return Err(SslError::DimensionMismatch {
                expected: predicted.len(),
                got: target.len(),
            });
        }

        let mut total = 0.0_f64;
        for i in 0..n_target {
            let mut sq = 0.0_f64;
            for j in 0..d {
                let diff = predicted[i * d + j] as f64 - target[i * d + j] as f64;
                sq += diff * diff;
            }
            total += sq.sqrt();
        }
        Ok((total / n_target as f64) as f32)
    }

    /// Sample a random partition of patch indices into (context, target) sets.
    ///
    /// `n_target = max(1, floor(mask_ratio × n_patches))`
    /// `n_context = n_patches − n_target`
    ///
    /// Returns `(context_ids, target_ids)` — non-overlapping, together covering
    /// all `n_patches` indices.
    ///
    /// # Errors
    /// - [`SslError::InvalidMaskRatio`] — `n_patches < 2` (cannot split into
    ///   two non-empty groups).
    pub fn sample_masks(&self, rng: &mut SslRng) -> SslResult<(Vec<usize>, Vec<usize>)> {
        let n = self.config.n_patches;
        if n < 2 {
            return Err(SslError::InvalidMaskRatio {
                ratio: self.config.mask_ratio,
            });
        }

        let n_target = (self.config.mask_ratio * n as f32).floor() as usize;
        let n_target = n_target.max(1).min(n - 1); // ensure at least 1 context patch
        let n_context = n - n_target;

        // Fisher-Yates shuffle over [0, n)
        let mut indices: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut indices);

        let target_ids = indices[..n_target].to_vec();
        let context_ids = indices[n_target..n_target + n_context].to_vec();

        Ok((context_ids, target_ids))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> IJepaConfig {
        IJepaConfig {
            n_patches: 16,
            d_model: 8,
            n_heads: 2,
            n_context_blocks: 2,
            n_predictor_layers: 2,
            mask_ratio: 0.25,
        }
    }

    fn make_rng() -> SslRng {
        LcgRng::new(42)
    }

    // ── 1. encode_output_shape ───────────────────────────────────────────────
    #[test]
    fn encode_output_shape() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_visible = 12_usize;
        let patches = vec![0.1_f32; n_visible * d];
        let ids: Vec<usize> = (0..n_visible).collect();
        let out = model
            .encode_context(&patches, &ids)
            .expect("encode_context should succeed");
        assert_eq!(out.len(), n_visible * d);
    }

    // ── 2. encode_output_finite ──────────────────────────────────────────────
    #[test]
    fn encode_output_finite() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_visible = 6_usize;
        let patches: Vec<f32> = (0..n_visible * d)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let ids: Vec<usize> = (0..n_visible).collect();
        let out = model
            .encode_context(&patches, &ids)
            .expect("encode_context should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in encode output"
        );
    }

    // ── 3. predict_output_shape ──────────────────────────────────────────────
    #[test]
    fn predict_output_shape() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_visible = 12_usize;
        let n_target = 4_usize;
        let ctx = vec![0.2_f32; n_visible * d];
        let target_ids: Vec<usize> = (n_visible..n_visible + n_target).collect();
        let out = model
            .predict_targets(&ctx, &target_ids, n_visible, n_target)
            .expect("value should be present");
        assert_eq!(out.len(), n_target * d);
    }

    // ── 4. predict_output_finite ─────────────────────────────────────────────
    #[test]
    fn predict_output_finite() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_visible = 10_usize;
        let n_target = 4_usize;
        let ctx: Vec<f32> = (0..n_visible * d)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let target_ids: Vec<usize> = (0..n_target).collect();
        let out = model
            .predict_targets(&ctx, &target_ids, n_visible, n_target)
            .expect("value should be present");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite in predict output"
        );
    }

    // ── 5. loss_nonneg ───────────────────────────────────────────────────────
    #[test]
    fn loss_nonneg() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_target = 4_usize;
        let pred: Vec<f32> = (0..n_target * d).map(|i| (i as f32 * 0.11).sin()).collect();
        let tgt: Vec<f32> = (0..n_target * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let l = model
            .loss(&pred, &tgt, n_target)
            .expect("loss should succeed");
        assert!(l >= 0.0, "loss must be non-negative, got {l}");
        assert!(l.is_finite());
    }

    // ── 6. loss_zero_for_identical ───────────────────────────────────────────
    #[test]
    fn loss_zero_for_identical() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n_target = 4_usize;
        let v: Vec<f32> = (0..n_target * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let l = model.loss(&v, &v, n_target).expect("loss should succeed");
        assert!(
            l.abs() < 1e-5,
            "loss for identical inputs should be ~0, got {l}"
        );
    }

    // ── 7. sample_masks_no_overlap ───────────────────────────────────────────
    #[test]
    fn sample_masks_no_overlap() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let (ctx, tgt) = model
            .sample_masks(&mut rng)
            .expect("sample_masks should succeed");

        // Confirm no index appears in both sets
        for &c in &ctx {
            assert!(
                !tgt.contains(&c),
                "index {c} appears in both context and target"
            );
        }
    }

    // ── 8. sample_masks_total_eq_n_patches ───────────────────────────────────
    #[test]
    fn sample_masks_total_eq_n_patches() {
        let mut rng = make_rng();
        let n_patches = 16_usize;
        let mut cfg = default_config();
        cfg.n_patches = n_patches;
        let model = IJepa::new(cfg, &mut rng).expect("new should succeed");
        let (ctx, tgt) = model
            .sample_masks(&mut rng)
            .expect("sample_masks should succeed");
        assert_eq!(
            ctx.len() + tgt.len(),
            n_patches,
            "context + target must cover all patches"
        );
    }

    // ── 9. n_patches_zero_error ──────────────────────────────────────────────
    #[test]
    fn n_patches_zero_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.n_patches = 0;
        let result = IJepa::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for n_patches == 0");
        assert!(matches!(
            result.unwrap_err(),
            SslError::InvalidParameter { .. }
        ));
    }

    // ── 10. d_model_zero_error ───────────────────────────────────────────────
    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.d_model = 0;
        let result = IJepa::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for d_model == 0");
        assert!(matches!(result.unwrap_err(), SslError::InvalidFeatureDim));
    }

    // ── 11. n_context_blocks_zero_error ──────────────────────────────────────
    #[test]
    fn n_context_blocks_zero_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.n_context_blocks = 0;
        let result = IJepa::new(cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SslError::InvalidParameter { .. }
        ));
    }

    // ── 12. mask_ratio_out_of_range_error ────────────────────────────────────
    #[test]
    fn mask_ratio_out_of_range_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.mask_ratio = 0.0;
        assert!(IJepa::new(cfg.clone(), &mut rng).is_err());
        cfg.mask_ratio = 1.0;
        assert!(IJepa::new(cfg, &mut rng).is_err());
    }

    // ── 13. encode_empty_patches_error ───────────────────────────────────────
    #[test]
    fn encode_empty_patches_error() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let result = model.encode_context(&[], &[]);
        assert!(matches!(result.unwrap_err(), SslError::EmptyInput));
    }

    // ── 14. predict_n_target_zero_error ──────────────────────────────────────
    #[test]
    fn predict_n_target_zero_error() {
        let mut rng = make_rng();
        let model = IJepa::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let ctx = vec![0.1_f32; 4 * d];
        let result = model.predict_targets(&ctx, &[], 4, 0);
        assert!(matches!(result.unwrap_err(), SslError::EmptyInput));
    }
}
