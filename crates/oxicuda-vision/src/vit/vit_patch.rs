//! Vision Transformer (ViT) patch embedding — Dosovitskiy et al. 2021.
//!
//! "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"
//! (Dosovitskiy et al., ICLR 2021) splits an image into a regular grid of
//! non-overlapping square patches, flattens each patch into a vector, and
//! projects it linearly into a `d_model`-dimensional embedding. A learnable
//! `[CLS]` token is prepended, and a learnable positional embedding is added to
//! the resulting `n_patches + 1` token sequence.
//!
//! ## Pipeline
//! ```text
//! image [C × H × W]
//!   ──split──▶ patches [(H/P)·(W/P) × (C·P·P)]
//!   ──proj───▶ tokens  [n_patches × d_model]   (linear: W·patch + b)
//!   ──cls────▶        [(n_patches+1) × d_model] (prepend [CLS] token)
//!   ──pos────▶        [(n_patches+1) × d_model] (+ learnable position embed)
//! ```
//!
//! Unlike [`crate::patch_embed::PatchEmbed`] — which exposes the strided-conv
//! view and matches the `patch_embed_ptx` kernel layout — this module follows
//! the original ViT formulation literally: an explicit per-patch flatten
//! (`C·P·P`) followed by a dense projection, the `[CLS]` prepend, and the
//! positional embedding fused into a single [`VitPatchEmbed::forward`] call.
//!
//! All tensors use flat row-major `Vec<f32>` layouts and the forward pass runs
//! on the CPU.

use crate::{
    blocks::VisionRng,
    error::{VisionError, VisionResult},
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`VitPatchEmbed`].
///
/// The image is `[n_channels × image_size × image_size]`. It is tiled into
/// `(image_size / patch_size)²` non-overlapping patches, each flattened to
/// `patch_size² · n_channels` values and projected to `d_model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VitPatchConfig {
    /// Square image spatial size (height = width).
    pub image_size: usize,
    /// Square patch spatial size (stride = `patch_size`).
    pub patch_size: usize,
    /// Number of input channels (e.g. 3 for RGB).
    pub n_channels: usize,
    /// Output token embedding dimension.
    pub d_model: usize,
}

impl VitPatchConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`VisionError::InvalidPatchSize`] if `patch_size == 0` or `image_size`
    ///   is not an exact multiple of `patch_size`.
    /// - [`VisionError::InvalidEmbedDim`] if `d_model == 0`.
    /// - [`VisionError::InvalidImageSize`] if `image_size == 0` or
    ///   `n_channels == 0`.
    pub fn validate(&self) -> VisionResult<()> {
        if self.patch_size == 0 || self.image_size % self.patch_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: self.patch_size,
                img_size: self.image_size,
            });
        }
        if self.d_model == 0 {
            return Err(VisionError::InvalidEmbedDim(self.d_model));
        }
        if self.image_size == 0 || self.n_channels == 0 {
            return Err(VisionError::InvalidImageSize {
                height: self.image_size,
                width: self.image_size,
                channels: self.n_channels,
            });
        }
        Ok(())
    }

    /// Patches along one spatial axis: `image_size / patch_size`.
    #[must_use]
    #[inline]
    pub fn grid_size(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Flattened patch length: `patch_size² · n_channels`.
    #[must_use]
    #[inline]
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.n_channels
    }
}

// ─── VitPatchEmbed ───────────────────────────────────────────────────────────

/// ViT patch-embedding layer with `[CLS]` token and learnable positions.
pub struct VitPatchEmbed {
    /// Projection weight, flat `[d_model × (patch_size² · n_channels)]`,
    /// row-major (output-major).
    proj_w: Vec<f32>,
    /// Projection bias, flat `[d_model]`.
    proj_b: Vec<f32>,
    /// Learnable class token, flat `[d_model]`.
    cls_token: Vec<f32>,
    /// Learnable positional embedding, flat `[(n_patches + 1) × d_model]`.
    pos_emb: Vec<f32>,
    /// Validated configuration.
    config: VitPatchConfig,
}

impl VitPatchEmbed {
    /// Construct a new patch embedder with random parameters.
    ///
    /// The projection weight is initialised `N(0, 1/√patch_dim)` (the fan-in
    /// scaling used by the reference implementations); biases are near-zero,
    /// the class token is `N(0, 0.02)` (ViT's truncated-normal scale), and the
    /// positional embedding is `N(0, 0.02)`.
    ///
    /// # Errors
    /// Propagates [`VitPatchConfig::validate`] failures.
    pub fn new(config: VitPatchConfig, rng: &mut VisionRng) -> VisionResult<Self> {
        config.validate()?;

        let patch_dim = config.patch_dim();
        let d_model = config.d_model;
        let n_tokens = config.grid_size() * config.grid_size() + 1;

        let scale = 1.0 / (patch_dim as f32).sqrt();
        let mut proj_w = vec![0.0_f32; d_model * patch_dim];
        rng.fill_normal(&mut proj_w);
        for w in &mut proj_w {
            *w *= scale;
        }

        let mut proj_b = vec![0.0_f32; d_model];
        rng.fill_normal(&mut proj_b);
        for b in &mut proj_b {
            *b *= 0.01;
        }

        let mut cls_token = vec![0.0_f32; d_model];
        rng.fill_normal(&mut cls_token);
        for c in &mut cls_token {
            *c *= 0.02;
        }

        let mut pos_emb = vec![0.0_f32; n_tokens * d_model];
        rng.fill_normal(&mut pos_emb);
        for p in &mut pos_emb {
            *p *= 0.02;
        }

        Ok(Self {
            proj_w,
            proj_b,
            cls_token,
            pos_emb,
            config,
        })
    }

    /// Read-only access to the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &VitPatchConfig {
        &self.config
    }

    /// Number of image patches: `(image_size / patch_size)²`.
    #[must_use]
    #[inline]
    pub fn n_patches(&self) -> usize {
        self.config.grid_size() * self.config.grid_size()
    }

    /// Embed an image into a token sequence.
    ///
    /// `image` must be `[n_channels × image_size × image_size]` row-major (CHW).
    /// The returned sequence is `[(n_patches + 1) × d_model]`: a prepended
    /// `[CLS]` token followed by the `n_patches` projected patch tokens, with
    /// the learnable positional embedding added element-wise.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `image.len()` does not equal
    ///   `n_channels · image_size²`.
    /// - [`VisionError::NonFinite`] if a non-finite value is produced.
    pub fn forward(&self, image: &[f32]) -> VisionResult<Vec<f32>> {
        let cfg = &self.config;
        let expected = cfg.n_channels * cfg.image_size * cfg.image_size;
        if image.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: image.len(),
            });
        }

        let grid = cfg.grid_size();
        let patch = cfg.patch_size;
        let n_patches = grid * grid;
        let patch_dim = cfg.patch_dim();
        let d_model = cfg.d_model;
        let img = cfg.image_size;
        let plane = img * img;

        // Output: [(n_patches + 1) × d_model]; token 0 is CLS.
        let mut out = vec![0.0_f32; (n_patches + 1) * d_model];

        // ── CLS token (row 0) ────────────────────────────────────────────────
        out[..d_model].copy_from_slice(&self.cls_token);

        // ── Patch projection (rows 1..=n_patches) ────────────────────────────
        // Scratch buffer for one flattened patch in (C, ph, pw) order so the
        // projection weight rows line up with `[c · P² + ph · P + pw]`.
        let mut flat = vec![0.0_f32; patch_dim];

        for gy in 0..grid {
            for gx in 0..grid {
                // Flatten patch (gy, gx) → `flat[c·P² + ph·P + pw]`.
                for c in 0..cfg.n_channels {
                    let chan_base = c * plane;
                    let dst_chan = c * patch * patch;
                    for ph in 0..patch {
                        let row = gy * patch + ph;
                        let src_row = chan_base + row * img + gx * patch;
                        let dst_row = dst_chan + ph * patch;
                        flat[dst_row..dst_row + patch]
                            .copy_from_slice(&image[src_row..src_row + patch]);
                    }
                }

                // Linear projection: token = W · flat + b.
                let patch_idx = gy * grid + gx;
                let out_base = (patch_idx + 1) * d_model;
                for o in 0..d_model {
                    let w_row = &self.proj_w[o * patch_dim..(o + 1) * patch_dim];
                    let mut acc = self.proj_b[o];
                    for (wv, fv) in w_row.iter().zip(flat.iter()) {
                        acc += wv * fv;
                    }
                    out[out_base + o] = acc;
                }
            }
        }

        // ── Positional embedding (element-wise add over all tokens) ──────────
        for (o, p) in out.iter_mut().zip(self.pos_emb.iter()) {
            *o += *p;
        }

        if out.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("ViT patch embedding output"));
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg() -> VitPatchConfig {
        VitPatchConfig {
            image_size: 16,
            patch_size: 4,
            n_channels: 3,
            d_model: 8,
        }
    }

    #[test]
    fn n_patches_correct() {
        let mut rng = LcgRng::new(1);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        // (16/4)² = 16
        assert_eq!(pe.n_patches(), 16);
    }

    #[test]
    fn forward_shape() {
        let mut rng = LcgRng::new(2);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        let image = vec![0.5_f32; 3 * 16 * 16];
        let out = pe.forward(&image).expect("forward ok");
        // (16 patches + 1 CLS) × d_model(8)
        assert_eq!(out.len(), 17 * 8);
    }

    #[test]
    fn forward_finite() {
        let mut rng = LcgRng::new(3);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        let mut image = vec![0.0_f32; 3 * 16 * 16];
        rng.fill_normal(&mut image);
        let out = pe.forward(&image).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cls_token_prepended() {
        // With a zero image, the projection of every patch is just proj_b, but
        // the CLS row equals cls_token + pos_emb[0]; the patch rows equal
        // proj_b + pos_emb[row]. Verify the CLS row differs from the patch rows
        // (cls_token is not generally equal to proj_b) and that subtracting the
        // positional embedding recovers cls_token at row 0.
        let mut rng = LcgRng::new(4);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        let image = vec![0.0_f32; 3 * 16 * 16];
        let out = pe.forward(&image).expect("ok");
        let d = pe.config().d_model;
        for (o, &out_o) in out.iter().enumerate().take(d) {
            let recovered = out_o - pe.pos_emb[o];
            assert!(
                (recovered - pe.cls_token[o]).abs() < 1e-5,
                "CLS token not recovered at dim {o}"
            );
        }
    }

    #[test]
    fn image_size_not_divisible_error() {
        let bad = VitPatchConfig {
            image_size: 15,
            patch_size: 4,
            n_channels: 3,
            d_model: 8,
        };
        let mut rng = LcgRng::new(5);
        let r = VitPatchEmbed::new(bad, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn patch_size_0_error() {
        let bad = VitPatchConfig {
            image_size: 16,
            patch_size: 0,
            n_channels: 3,
            d_model: 8,
        };
        let mut rng = LcgRng::new(6);
        let r = VitPatchEmbed::new(bad, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn different_images_different_embeds() {
        let mut rng = LcgRng::new(7);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        let img_a = vec![0.2_f32; 3 * 16 * 16];
        let mut img_b = vec![0.2_f32; 3 * 16 * 16];
        img_b[0] = 5.0; // perturb a single pixel
        let out_a = pe.forward(&img_a).expect("ok");
        let out_b = pe.forward(&img_b).expect("ok");
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "embeddings should differ for different images");
    }

    #[test]
    fn pos_emb_added() {
        // Build a layer, then zero the projection + cls + bias and re-run on a
        // zero image: the output must equal exactly the positional embedding.
        let mut rng = LcgRng::new(8);
        let mut pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        for w in &mut pe.proj_w {
            *w = 0.0;
        }
        for b in &mut pe.proj_b {
            *b = 0.0;
        }
        for c in &mut pe.cls_token {
            *c = 0.0;
        }
        let image = vec![3.0_f32; 3 * 16 * 16];
        let out = pe.forward(&image).expect("ok");
        for (o, p) in out.iter().zip(pe.pos_emb.iter()) {
            assert!((o - p).abs() < 1e-6, "output must equal pos_emb");
        }
    }

    #[test]
    fn d_model_0_error() {
        let bad = VitPatchConfig {
            image_size: 16,
            patch_size: 4,
            n_channels: 3,
            d_model: 0,
        };
        let mut rng = LcgRng::new(9);
        let r = VitPatchEmbed::new(bad, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn single_patch() {
        // image_size == patch_size ⇒ exactly one patch.
        let single = VitPatchConfig {
            image_size: 8,
            patch_size: 8,
            n_channels: 3,
            d_model: 4,
        };
        let mut rng = LcgRng::new(10);
        let pe = VitPatchEmbed::new(single, &mut rng).expect("ok");
        assert_eq!(pe.n_patches(), 1);
        let image = vec![0.5_f32; 3 * 8 * 8];
        let out = pe.forward(&image).expect("ok");
        assert_eq!(out.len(), 2 * 4); // (1 patch + CLS) × 4
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn n_channels_0_error() {
        let bad = VitPatchConfig {
            image_size: 16,
            patch_size: 4,
            n_channels: 0,
            d_model: 8,
        };
        let mut rng = LcgRng::new(11);
        let r = VitPatchEmbed::new(bad, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn forward_wrong_image_len_error() {
        let mut rng = LcgRng::new(12);
        let pe = VitPatchEmbed::new(cfg(), &mut rng).expect("ok");
        let image = vec![0.5_f32; 10]; // wrong length
        let r = pe.forward(&image);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn patch_flatten_matches_manual() {
        // d_model = 1, identity-like weight: set proj_w to select the first
        // element of the flattened patch and zero bias/cls/pos. Then token 1's
        // value must equal image[0] (top-left pixel of channel 0).
        let c = VitPatchConfig {
            image_size: 4,
            patch_size: 2,
            n_channels: 1,
            d_model: 1,
        };
        let mut rng = LcgRng::new(13);
        let mut pe = VitPatchEmbed::new(c, &mut rng).expect("ok");
        // patch_dim = 2*2*1 = 4; weight row picks element 0.
        pe.proj_w = vec![1.0, 0.0, 0.0, 0.0];
        pe.proj_b = vec![0.0];
        pe.cls_token = vec![0.0];
        for p in &mut pe.pos_emb {
            *p = 0.0;
        }
        let mut image = vec![0.0_f32; 16];
        image[0] = 7.0; // top-left pixel → first element of patch (0,0)
        let out = pe.forward(&image).expect("ok");
        // out[0] = CLS = 0; out[1] = token of patch (0,0) = image[0] = 7.
        assert!((out[1] - 7.0).abs() < 1e-6, "got {}", out[1]);
    }
}
