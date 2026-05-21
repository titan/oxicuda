//! Adversarial Patch attack.
//!
//! Structured, bounded-support threat model from
//! Brown, Mané, Roy, Abadi & Gilmer (2017),
//! *"Adversarial Patch"*, NeurIPS Workshop.
//!
//! Rather than constraining the perturbation to a small Lp ball over the whole
//! image, the patch attack confines an *unbounded* perturbation to a small
//! rectangular region (a "sticker") and leaves every pixel outside that region
//! untouched. The patch contents are optimised by PGD-style sign-gradient
//! ascent on the loss, restricted to the patch pixels.
//!
//! # Image layout
//!
//! Images are flat `(channels, img_h, img_w)` row-major buffers — i.e. the
//! channel-major (`CHW`) convention used by the rest of the crate's reference
//! paths. The element at channel `c`, row `r`, column `col` lives at index
//!
//! ```text
//! c * (img_h * img_w) + r * img_w + col
//! ```
//!
//! The patch occupies rows `[pos_row, pos_row + patch_h)` and columns
//! `[pos_col, pos_col + patch_w)` across *all* channels.
//!
//! # Model interface
//!
//! Following the crate convention the loss gradient is supplied as a closure
//! `grad: Fn(&[f32]) -> Vec<f32>` returning `∂loss / ∂image` of identical
//! length to the image.

use crate::error::{AdvError, AdvResult};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Hyperparameters for [`PatchAttack`].
///
/// Construct with [`PatchAttack::new`] which validates every field up-front.
#[derive(Debug, Clone, Copy)]
pub struct PatchConfig {
    /// Image height in pixels (rows).
    pub img_h: usize,
    /// Image width in pixels (columns).
    pub img_w: usize,
    /// Number of channels (must be `>= 1`).
    pub channels: usize,
    /// Patch height in pixels (must be `>= 1` and fit inside the image).
    pub patch_h: usize,
    /// Patch width in pixels (must be `>= 1` and fit inside the image).
    pub patch_w: usize,
    /// Top row of the patch (`pos_row + patch_h <= img_h`).
    pub pos_row: usize,
    /// Left column of the patch (`pos_col + patch_w <= img_w`).
    pub pos_col: usize,
    /// PGD-style step size (must be `> 0` and finite).
    pub step_size: f32,
    /// Number of ascent steps (must be `>= 1`).
    pub n_steps: usize,
    /// Box lower bound (inclusive).
    pub clamp_min: f32,
    /// Box upper bound (inclusive).
    pub clamp_max: f32,
}

// ─── Attack ──────────────────────────────────────────────────────────────────

/// Adversarial Patch attack.
#[derive(Debug, Clone)]
pub struct PatchAttack {
    cfg: PatchConfig,
}

impl PatchAttack {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::Internal`]          — zero channels, zero patch extent, or
    ///   a patch that does not fit inside the image at the given position.
    /// * [`AdvError::InvalidAlpha`]      — non-finite or non-positive `step_size`.
    /// * [`AdvError::InvalidNumSteps`]   — `n_steps == 0`.
    /// * [`AdvError::InvalidLossWeight`] — degenerate box (`clamp_min >= clamp_max`
    ///   or non-finite bounds).
    pub fn new(cfg: PatchConfig) -> AdvResult<Self> {
        if cfg.channels == 0 {
            return Err(AdvError::Internal(
                "patch: channels must be >= 1".to_owned(),
            ));
        }
        if cfg.patch_h == 0 || cfg.patch_w == 0 {
            return Err(AdvError::Internal(
                "patch: patch_h and patch_w must be >= 1".to_owned(),
            ));
        }
        if cfg.img_h == 0 || cfg.img_w == 0 {
            return Err(AdvError::Internal(
                "patch: img_h and img_w must be >= 1".to_owned(),
            ));
        }
        // Patch must fit entirely inside the image at the configured position.
        if cfg.pos_row + cfg.patch_h > cfg.img_h || cfg.pos_col + cfg.patch_w > cfg.img_w {
            return Err(AdvError::Internal(
                "patch: patch region exceeds image bounds".to_owned(),
            ));
        }
        if !(cfg.step_size.is_finite() && cfg.step_size > 0.0) {
            return Err(AdvError::InvalidAlpha {
                alpha: cfg.step_size,
            });
        }
        if cfg.n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if !(cfg.clamp_min.is_finite() && cfg.clamp_max.is_finite())
            || cfg.clamp_min >= cfg.clamp_max
        {
            return Err(AdvError::InvalidLossWeight {
                weight: cfg.clamp_max - cfg.clamp_min,
            });
        }
        Ok(Self { cfg })
    }

    /// Borrow the validated configuration.
    #[must_use]
    pub fn config(&self) -> &PatchConfig {
        &self.cfg
    }

    /// Total number of elements in a valid image: `channels * img_h * img_w`.
    #[must_use]
    #[inline]
    fn image_len(&self) -> usize {
        self.cfg.channels * self.cfg.img_h * self.cfg.img_w
    }

    /// Number of elements in a valid patch: `patch_h * patch_w * channels`.
    #[must_use]
    #[inline]
    fn patch_len(&self) -> usize {
        self.cfg.patch_h * self.cfg.patch_w * self.cfg.channels
    }

    /// Boolean mask of length `channels * img_h * img_w`; `true` exactly for the
    /// pixels inside the patch region across all channels.
    #[must_use]
    pub fn patch_mask(&self) -> Vec<bool> {
        let plane = self.cfg.img_h * self.cfg.img_w;
        let mut mask = vec![false; self.image_len()];
        for c in 0..self.cfg.channels {
            let chan_off = c * plane;
            for r in self.cfg.pos_row..(self.cfg.pos_row + self.cfg.patch_h) {
                let row_off = chan_off + r * self.cfg.img_w;
                for col in self.cfg.pos_col..(self.cfg.pos_col + self.cfg.patch_w) {
                    mask[row_off + col] = true;
                }
            }
        }
        mask
    }

    /// Run the patch attack on `image`.
    ///
    /// `x = image.clone()`; for each of `n_steps` iterations the loss gradient
    /// `g = grad(&x)` is evaluated and, for every patch index `i`,
    /// `x[i] += step_size · sign(g[i])` followed by a clamp into
    /// `[clamp_min, clamp_max]`. Pixels outside the patch region are never
    /// touched.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `image.len()` or any `grad(&x)`
    ///   output has the wrong length.
    /// * [`AdvError::NanEncountered`]    — a gradient entry is non-finite.
    pub fn attack<G>(&self, image: &[f32], grad: G) -> AdvResult<Vec<f32>>
    where
        G: Fn(&[f32]) -> Vec<f32>,
    {
        let expected = self.image_len();
        if image.len() != expected {
            return Err(AdvError::DimensionMismatch {
                expected,
                got: image.len(),
            });
        }

        // Precompute the flat indices inside the patch once.
        let mask = self.patch_mask();
        let patch_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        let mut x = image.to_vec();
        for _ in 0..self.cfg.n_steps {
            let g = grad(&x);
            if g.len() != expected {
                return Err(AdvError::DimensionMismatch {
                    expected,
                    got: g.len(),
                });
            }
            // Validate only the gradient entries we actually consume.
            for &i in &patch_indices {
                if !g[i].is_finite() {
                    return Err(AdvError::NanEncountered {
                        location: "patch:attack:grad",
                    });
                }
            }
            for &i in &patch_indices {
                let s = sign(g[i]);
                x[i] =
                    (x[i] + self.cfg.step_size * s).clamp(self.cfg.clamp_min, self.cfg.clamp_max);
            }
        }
        Ok(x)
    }

    /// Overlay the supplied `patch` content (length `patch_h * patch_w *
    /// channels`, laid out channel-major to match the image) onto a copy of
    /// `image` at the configured position, clamping each written value into
    /// `[clamp_min, clamp_max]`. Pixels outside the patch region are copied
    /// verbatim.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `image.len()` or `patch.len()` has
    ///   the wrong length.
    pub fn apply_patch(&self, image: &[f32], patch: &[f32]) -> AdvResult<Vec<f32>> {
        let img_expected = self.image_len();
        if image.len() != img_expected {
            return Err(AdvError::DimensionMismatch {
                expected: img_expected,
                got: image.len(),
            });
        }
        let patch_expected = self.patch_len();
        if patch.len() != patch_expected {
            return Err(AdvError::DimensionMismatch {
                expected: patch_expected,
                got: patch.len(),
            });
        }

        let plane = self.cfg.img_h * self.cfg.img_w;
        let mut out = image.to_vec();
        // Patch content is channel-major (CHW): channel c, row pr, col pc maps to
        // patch index ((c * patch_h) + pr) * patch_w + pc.
        for c in 0..self.cfg.channels {
            let chan_off = c * plane;
            let patch_chan_off = c * self.cfg.patch_h * self.cfg.patch_w;
            for pr in 0..self.cfg.patch_h {
                let img_row_off = chan_off + (self.cfg.pos_row + pr) * self.cfg.img_w;
                let patch_row_off = patch_chan_off + pr * self.cfg.patch_w;
                for pc in 0..self.cfg.patch_w {
                    let value = patch[patch_row_off + pc];
                    out[img_row_off + self.cfg.pos_col + pc] =
                        value.clamp(self.cfg.clamp_min, self.cfg.clamp_max);
                }
            }
        }
        Ok(out)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Element-wise sign with `0.0` for zeros (avoids `f32::signum` returning `±0`).
#[inline]
fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Default valid config: 3-channel 4×4 image, 2×2 patch at (1, 1).
    fn cfg_default() -> PatchConfig {
        PatchConfig {
            img_h: 4,
            img_w: 4,
            channels: 3,
            patch_h: 2,
            patch_w: 2,
            pos_row: 1,
            pos_col: 1,
            step_size: 0.1,
            n_steps: 3,
            clamp_min: 0.0,
            clamp_max: 1.0,
        }
    }

    /// Constant-gradient closure of a fixed length.
    fn const_grad(g: Vec<f32>) -> impl Fn(&[f32]) -> Vec<f32> {
        move |_x: &[f32]| g.clone()
    }

    // ── mask geometry ──────────────────────────────────────────────────────────

    #[test]
    fn patch_mask_length_and_count() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let mask = p.patch_mask();
        assert_eq!(mask.len(), 3 * 4 * 4);
        let true_count = mask.iter().filter(|&&b| b).count();
        assert_eq!(true_count, 2 * 2 * 3);
    }

    #[test]
    fn patch_mask_marks_correct_region() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let mask = p.patch_mask();
        let plane = 4 * 4;
        // Channel 0, rows 1..3, cols 1..3 should be true; (0,0) false.
        for c in 0..3 {
            for r in 0..4 {
                for col in 0..4 {
                    let idx = c * plane + r * 4 + col;
                    let inside = (1..3).contains(&r) && (1..3).contains(&col);
                    assert_eq!(mask[idx], inside, "c={c} r={r} col={col}");
                }
            }
        }
    }

    #[test]
    fn patch_mask_matches_apply_patch_region() {
        // The pixels apply_patch overwrites must be exactly the masked pixels.
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.3_f32; 3 * 4 * 4];
        // Distinct patch content (all 0.9) so overwrites are detectable.
        let patch = vec![0.9_f32; 2 * 2 * 3];
        let out = p.apply_patch(&image, &patch).unwrap();
        let mask = p.patch_mask();
        for (i, (&before, &after)) in image.iter().zip(out.iter()).enumerate() {
            if mask[i] {
                assert!((after - 0.9).abs() < 1e-6, "masked idx {i} not overwritten");
            } else {
                assert!((after - before).abs() < 1e-9, "non-masked idx {i} changed");
            }
        }
    }

    // ── attack behaviour ────────────────────────────────────────────────────────

    #[test]
    fn attack_output_length_equals_image() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        let g = vec![1.0_f32; 3 * 4 * 4];
        let out = p.attack(&image, const_grad(g)).unwrap();
        assert_eq!(out.len(), image.len());
    }

    #[test]
    fn attack_leaves_non_patch_pixels_exactly_unchanged() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image: Vec<f32> = (0..3 * 4 * 4).map(|i| (i as f32) * 0.01).collect();
        // Large positive gradient everywhere; only patch pixels may move.
        let g = vec![5.0_f32; 3 * 4 * 4];
        let out = p.attack(&image, const_grad(g)).unwrap();
        let mask = p.patch_mask();
        for (i, (&before, &after)) in image.iter().zip(out.iter()).enumerate() {
            if !mask[i] {
                assert_eq!(before, after, "non-patch pixel {i} changed");
            }
        }
    }

    #[test]
    fn attack_clamps_patch_pixels() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.95_f32; 3 * 4 * 4];
        let g = vec![1.0_f32; 3 * 4 * 4]; // ascend → hits upper clamp
        let out = p.attack(&image, const_grad(g)).unwrap();
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "out of [0,1]: {v}");
        }
    }

    #[test]
    fn attack_constant_gradient_increments_by_step_times_nsteps() {
        // step_size * n_steps = 0.1 * 3 = 0.3; image at 0.0 stays well below clamp.
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.0_f32; 3 * 4 * 4];
        let g = vec![2.0_f32; 3 * 4 * 4]; // positive sign → +step each step
        let out = p.attack(&image, const_grad(g)).unwrap();
        let mask = p.patch_mask();
        for (i, &after) in out.iter().enumerate() {
            if mask[i] {
                assert!((after - 0.3).abs() < 1e-5, "patch idx {i} = {after}");
            } else {
                assert!((after).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn attack_negative_gradient_decrements() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        let g = vec![-1.0_f32; 3 * 4 * 4]; // negative sign → −step each step
        let out = p.attack(&image, const_grad(g)).unwrap();
        let mask = p.patch_mask();
        for (i, &after) in out.iter().enumerate() {
            if mask[i] {
                // 0.5 - 0.3 = 0.2
                assert!((after - 0.2).abs() < 1e-5, "patch idx {i} = {after}");
            }
        }
    }

    #[test]
    fn attack_deterministic() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.4_f32; 3 * 4 * 4];
        let g = vec![1.0_f32; 3 * 4 * 4];
        let a = p.attack(&image, const_grad(g.clone())).unwrap();
        let b = p.attack(&image, const_grad(g)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn attack_corner_position_works() {
        let cfg = PatchConfig {
            pos_row: 2,
            pos_col: 2,
            ..cfg_default()
        };
        let p = PatchAttack::new(cfg).unwrap();
        let image = vec![0.0_f32; 3 * 4 * 4];
        let g = vec![1.0_f32; 3 * 4 * 4];
        let out = p.attack(&image, const_grad(g)).unwrap();
        // Bottom-right corner pixel (channel 0, row 3, col 3) is inside patch.
        let idx = 3 * 4 + 3;
        assert!((out[idx] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn attack_full_image_patch_works() {
        // patch covers the whole image.
        let cfg = PatchConfig {
            img_h: 2,
            img_w: 2,
            channels: 1,
            patch_h: 2,
            patch_w: 2,
            pos_row: 0,
            pos_col: 0,
            ..cfg_default()
        };
        let p = PatchAttack::new(cfg).unwrap();
        let image = vec![0.0_f32; 4];
        let g = vec![1.0_f32; 4];
        let out = p.attack(&image, const_grad(g)).unwrap();
        // Every pixel is in the patch, so all incremented by 0.3.
        for &v in &out {
            assert!((v - 0.3).abs() < 1e-5);
        }
        // Mask is all true.
        assert!(p.patch_mask().iter().all(|&b| b));
    }

    #[test]
    fn attack_single_channel_image() {
        let cfg = PatchConfig {
            channels: 1,
            ..cfg_default()
        };
        let p = PatchAttack::new(cfg).unwrap();
        let image = vec![0.0_f32; 4 * 4];
        let g = vec![1.0_f32; 4 * 4];
        let out = p.attack(&image, const_grad(g)).unwrap();
        let mask = p.patch_mask();
        assert_eq!(mask.iter().filter(|&&b| b).count(), 2 * 2);
        for (i, &after) in out.iter().enumerate() {
            if mask[i] {
                assert!((after - 0.3).abs() < 1e-5);
            }
        }
    }

    // ── apply_patch behaviour ────────────────────────────────────────────────────

    #[test]
    fn apply_patch_overwrites_only_region() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.2_f32; 3 * 4 * 4];
        let patch = vec![0.8_f32; 2 * 2 * 3];
        let out = p.apply_patch(&image, &patch).unwrap();
        let mask = p.patch_mask();
        for (i, &after) in out.iter().enumerate() {
            if mask[i] {
                assert!((after - 0.8).abs() < 1e-6);
            } else {
                assert!((after - 0.2).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn apply_patch_clamps_content() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        // Patch content exceeds the box → must be clamped to [0, 1].
        let patch = vec![5.0_f32; 2 * 2 * 3];
        let out = p.apply_patch(&image, &patch).unwrap();
        let mask = p.patch_mask();
        for (i, &after) in out.iter().enumerate() {
            if mask[i] {
                assert!((after - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn apply_patch_channel_major_placement() {
        // 1-channel 3×3 image, 1×1 patch at (2, 0): only that one pixel changes.
        let cfg = PatchConfig {
            img_h: 3,
            img_w: 3,
            channels: 1,
            patch_h: 1,
            patch_w: 1,
            pos_row: 2,
            pos_col: 0,
            ..cfg_default()
        };
        let p = PatchAttack::new(cfg).unwrap();
        let image = vec![0.0_f32; 9];
        let patch = vec![0.7_f32];
        let out = p.apply_patch(&image, &patch).unwrap();
        let idx = 2 * 3; // row 2, col 0
        assert!((out[idx] - 0.7).abs() < 1e-6);
        // Everything else stays 0.
        for (i, &v) in out.iter().enumerate() {
            if i != idx {
                assert!(v.abs() < 1e-9);
            }
        }
    }

    // ── error paths ────────────────────────────────────────────────────────────

    #[test]
    fn err_patch_out_of_bounds() {
        let cfg = PatchConfig {
            pos_row: 3,
            pos_col: 0,
            patch_h: 2,
            patch_w: 2,
            ..cfg_default()
        };
        // pos_row(3) + patch_h(2) = 5 > img_h(4).
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn err_channels_zero() {
        let cfg = PatchConfig {
            channels: 0,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn err_patch_h_zero() {
        let cfg = PatchConfig {
            patch_h: 0,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn err_step_size_non_positive() {
        let cfg = PatchConfig {
            step_size: 0.0,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::InvalidAlpha { .. }
        ));
        let cfg_neg = PatchConfig {
            step_size: -0.1,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg_neg).unwrap_err(),
            AdvError::InvalidAlpha { .. }
        ));
    }

    #[test]
    fn err_n_steps_zero() {
        let cfg = PatchConfig {
            n_steps: 0,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::InvalidNumSteps
        ));
    }

    #[test]
    fn err_degenerate_box() {
        let cfg = PatchConfig {
            clamp_min: 1.0,
            clamp_max: 1.0,
            ..cfg_default()
        };
        assert!(matches!(
            PatchAttack::new(cfg).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn err_image_wrong_length() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 10]; // expected 48
        assert!(matches!(
            p.attack(&image, const_grad(vec![1.0; 10])).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            p.apply_patch(&image, &[0.0; 2 * 2 * 3]).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn err_patch_wrong_length() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        let patch = vec![0.8_f32; 5]; // expected 12
        assert!(matches!(
            p.apply_patch(&image, &patch).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn err_grad_wrong_length_during_attack() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        // Gradient closure returns the wrong length.
        let bad = |_x: &[f32]| vec![1.0_f32; 7];
        assert!(matches!(
            p.attack(&image, bad).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn err_nan_gradient_during_attack() {
        let p = PatchAttack::new(cfg_default()).unwrap();
        let image = vec![0.5_f32; 3 * 4 * 4];
        // Put a NaN at a patch index (channel 0, row 1, col 1 = index 5).
        let mut g = vec![1.0_f32; 3 * 4 * 4];
        g[5] = f32::NAN;
        assert!(matches!(
            p.attack(&image, const_grad(g)).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn config_accessor_round_trips() {
        let cfg = cfg_default();
        let p = PatchAttack::new(cfg).unwrap();
        assert_eq!(p.config().patch_h, cfg.patch_h);
        assert_eq!(p.config().channels, cfg.channels);
    }
}
