//! Patch embedder: strided Conv2D producing `[N_patches, embed_dim]`.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the patch embedder.
///
/// The image is assumed to have shape `[in_chans, img_size, img_size]`.
/// It is split into `(img_size / patch_size)²` non-overlapping patches,
/// each of size `in_chans × patch_size × patch_size`, which are projected
/// linearly to `embed_dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchEmbedConfig {
    /// Square image spatial dimension (H = W).
    pub img_size: usize,
    /// Patch spatial dimension (P × P windows, stride = P).
    pub patch_size: usize,
    /// Number of input channels (e.g., 3 for RGB).
    pub in_chans: usize,
    /// Output token embedding dimension.
    pub embed_dim: usize,
}

impl PatchEmbedConfig {
    /// Create and validate a `PatchEmbedConfig`.
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> VisionResult<Self> {
        if patch_size == 0 || img_size % patch_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size,
                img_size,
            });
        }
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if img_size == 0 || in_chans == 0 {
            return Err(VisionError::InvalidImageSize {
                height: img_size,
                width: img_size,
                channels: in_chans,
            });
        }
        Ok(Self {
            img_size,
            patch_size,
            in_chans,
            embed_dim,
        })
    }

    /// Number of patches along one spatial dimension.
    #[must_use]
    pub fn grid_size(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Total number of patches (CLS token not counted here).
    #[must_use]
    pub fn n_patches(&self) -> usize {
        self.grid_size() * self.grid_size()
    }

    /// Kernel volume for one filter: `in_chans × patch_size²`.
    #[must_use]
    pub fn kernel_vol(&self) -> usize {
        self.in_chans * self.patch_size * self.patch_size
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learnable weights for the patch embedder.
///
/// `kernel` has layout `[embed_dim, in_chans, patch_size, patch_size]`
/// (row-major, C-contiguous): filter `e` occupies
/// `kernel[e * kernel_vol .. (e+1) * kernel_vol]`.
pub struct PatchEmbedWeights {
    /// Conv2D kernel: flat `[embed_dim × in_chans × P × P]`.
    pub kernel: Vec<f32>,
    /// Bias: flat `[embed_dim]`.
    pub bias: Vec<f32>,
    /// CLS token: flat `[embed_dim]`.
    pub cls_token: Vec<f32>,
}

impl PatchEmbedWeights {
    /// Xavier/He-style default init: N(0, 1/√(kernel_vol)).
    pub fn default_init(cfg: &PatchEmbedConfig, rng: &mut LcgRng) -> Self {
        let kv = cfg.kernel_vol();
        let scale = 1.0 / (kv as f32).sqrt();
        let n_kernel = cfg.embed_dim * kv;

        let mut kernel = vec![0.0f32; n_kernel];
        rng.fill_normal(&mut kernel);
        for v in &mut kernel {
            *v *= scale;
        }

        let mut bias = vec![0.0f32; cfg.embed_dim];
        rng.fill_normal(&mut bias);
        for v in &mut bias {
            *v *= 0.01;
        }

        let mut cls_token = vec![0.0f32; cfg.embed_dim];
        rng.fill_normal(&mut cls_token);
        for v in &mut cls_token {
            *v *= 0.02;
        }

        Self {
            kernel,
            bias,
            cls_token,
        }
    }
}

// ─── PatchEmbed ──────────────────────────────────────────────────────────────

/// Patch embedder: converts a CHW image to a `[N_patches, embed_dim]` token
/// sequence via a strided Conv2D with `stride = kernel_size = patch_size`.
pub struct PatchEmbed {
    pub config: PatchEmbedConfig,
    pub weights: PatchEmbedWeights,
}

impl PatchEmbed {
    /// Create a new `PatchEmbed` with Xavier-initialised weights.
    pub fn new(cfg: PatchEmbedConfig, rng: &mut LcgRng) -> Self {
        let weights = PatchEmbedWeights::default_init(&cfg, rng);
        Self {
            config: cfg,
            weights,
        }
    }

    /// Forward pass: `image` is flat `[in_chans, img_size, img_size]` CHW.
    ///
    /// Returns `[n_patches, embed_dim]` flat row-major.
    pub fn forward(&self, image: &[f32]) -> VisionResult<Vec<f32>> {
        let cfg = &self.config;
        let expected = cfg.in_chans * cfg.img_size * cfg.img_size;
        if image.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: image.len(),
            });
        }

        let n_patches = cfg.n_patches();
        let grid = cfg.grid_size();
        let p = cfg.patch_size;
        let c = cfg.in_chans;
        let e = cfg.embed_dim;
        let kv = cfg.kernel_vol(); // c * p * p

        let mut out = vec![0.0f32; n_patches * e];

        // For each patch (ph, pw) and each output channel ed:
        //   out[ph*grid + pw, ed] = bias[ed] + Σ_{ci,pi,pj} kernel[ed, ci, pi, pj] * image[ci, ph*p+pi, pw*p+pj]
        for ph in 0..grid {
            for pw in 0..grid {
                let patch_idx = ph * grid + pw;
                for ed in 0..e {
                    let mut acc = self.weights.bias[ed];
                    // Kernel for output channel `ed`: slice [ed*kv .. (ed+1)*kv]
                    let k_off = ed * kv;
                    for ci in 0..c {
                        for pi in 0..p {
                            for pj in 0..p {
                                let k_idx = k_off + ci * p * p + pi * p + pj;
                                let img_row = ph * p + pi;
                                let img_col = pw * p + pj;
                                let img_idx = ci * cfg.img_size * cfg.img_size
                                    + img_row * cfg.img_size
                                    + img_col;
                                acc += self.weights.kernel[k_idx] * image[img_idx];
                            }
                        }
                    }
                    out[patch_idx * e + ed] = acc;
                }
            }
        }

        Ok(out)
    }
}

// ─── CLS prepend ─────────────────────────────────────────────────────────────

/// Prepend the CLS token to a `[n_patches, embed_dim]` token sequence.
///
/// Returns `[(n_patches+1) * embed_dim]` flat, with the CLS token at index 0.
pub fn prepend_cls(tokens: &[f32], cls: &[f32], embed_dim: usize) -> VisionResult<Vec<f32>> {
    let n_tok = tokens.len() / embed_dim;
    if tokens.len() != n_tok * embed_dim {
        return Err(VisionError::DimensionMismatch {
            expected: n_tok * embed_dim,
            got: tokens.len(),
        });
    }
    if cls.len() != embed_dim {
        return Err(VisionError::DimensionMismatch {
            expected: embed_dim,
            got: cls.len(),
        });
    }
    let mut out = Vec::with_capacity((n_tok + 1) * embed_dim);
    out.extend_from_slice(cls);
    out.extend_from_slice(tokens);
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_cfg() -> PatchEmbedConfig {
        PatchEmbedConfig::new(16, 4, 3, 8).expect("valid config")
    }

    #[test]
    fn config_valid() {
        let cfg = make_cfg();
        assert_eq!(cfg.n_patches(), 16); // (16/4)^2
        assert_eq!(cfg.grid_size(), 4);
        assert_eq!(cfg.kernel_vol(), 3 * 4 * 4); // c*p*p = 48
    }

    #[test]
    fn config_invalid_patch_size_not_dividing() {
        let r = PatchEmbedConfig::new(16, 5, 3, 8);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn config_invalid_patch_size_zero() {
        let r = PatchEmbedConfig::new(16, 0, 3, 8);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn config_invalid_embed_dim_zero() {
        let r = PatchEmbedConfig::new(16, 4, 3, 0);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn forward_output_shape() {
        let cfg = make_cfg(); // 16×16×3, p=4 → 16 patches, embed=8
        let mut rng = LcgRng::new(1);
        let pe = PatchEmbed::new(cfg.clone(), &mut rng);
        let image = vec![0.5f32; 3 * 16 * 16];
        let out = pe.forward(&image).expect("forward ok");
        assert_eq!(out.len(), cfg.n_patches() * cfg.embed_dim);
    }

    #[test]
    fn forward_wrong_image_size_errors() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(2);
        let pe = PatchEmbed::new(cfg, &mut rng);
        let image = vec![0.5f32; 10]; // wrong
        let r = pe.forward(&image);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_zero_image_is_bias() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(3);
        let pe = PatchEmbed::new(cfg.clone(), &mut rng);
        let image = vec![0.0f32; 3 * 16 * 16];
        let out = pe.forward(&image).expect("forward ok");
        // With zero input, output = bias, so patch 0 channel 0 = bias[0]
        let diff = (out[0] - pe.weights.bias[0]).abs();
        assert!(
            diff < 1e-6,
            "expected bias={}, got {}",
            pe.weights.bias[0],
            out[0]
        );
    }

    #[test]
    fn forward_finite_random_input() {
        let cfg = PatchEmbedConfig::new(32, 4, 3, 64).expect("valid");
        let mut rng = LcgRng::new(7);
        let pe = PatchEmbed::new(cfg.clone(), &mut rng);
        let mut image = vec![0.0f32; 3 * 32 * 32];
        rng.fill_normal(&mut image);
        let out = pe.forward(&image).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output contains non-finite"
        );
    }

    #[test]
    fn prepend_cls_shape() {
        let tokens = vec![1.0f32; 16 * 8]; // 16 patches, embed=8
        let cls = vec![0.0f32; 8];
        let out = prepend_cls(&tokens, &cls, 8).expect("ok");
        assert_eq!(out.len(), 17 * 8);
        // First row is the CLS token
        assert!(out[..8].iter().all(|&v| v == 0.0));
        // Next row is the first patch token
        assert_eq!(out[8..16], tokens[..8]);
    }

    #[test]
    fn prepend_cls_wrong_cls_dim_errors() {
        let tokens = vec![1.0f32; 16 * 8];
        let cls = vec![0.0f32; 4]; // wrong
        let r = prepend_cls(&tokens, &cls, 8);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn weights_default_init_correct_size() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(42);
        let w = PatchEmbedWeights::default_init(&cfg, &mut rng);
        assert_eq!(w.kernel.len(), cfg.embed_dim * cfg.kernel_vol());
        assert_eq!(w.bias.len(), cfg.embed_dim);
        assert_eq!(w.cls_token.len(), cfg.embed_dim);
    }

    #[test]
    fn weights_default_init_finite() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(99);
        let w = PatchEmbedWeights::default_init(&cfg, &mut rng);
        assert!(w.kernel.iter().all(|v| v.is_finite()));
        assert!(w.bias.iter().all(|v| v.is_finite()));
        assert!(w.cls_token.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn patch_embed_different_seeds_differ() {
        let cfg = make_cfg();
        let image = vec![0.5f32; 3 * 16 * 16];
        let mut rng1 = LcgRng::new(1);
        let mut rng2 = LcgRng::new(2);
        let pe1 = PatchEmbed::new(cfg.clone(), &mut rng1);
        let pe2 = PatchEmbed::new(cfg, &mut rng2);
        let out1 = pe1.forward(&image).expect("ok");
        let out2 = pe2.forward(&image).expect("ok");
        // Different kernels should yield different outputs
        assert!(
            out1.iter()
                .zip(out2.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }
}
