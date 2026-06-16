//! Mathematical morphology on single-channel `f32` images.
//!
//! Operates on a **single-channel** (grayscale or binary) `[h × w]` row-major
//! `f32` image, matching the [`crate::imgproc::edges`] convention. Binary inputs
//! are simply images whose values are restricted to `{0.0, 1.0}`; because the
//! flat (non-weighted) morphological operators reduce to per-pixel `min` / `max`
//! over a neighbourhood, the same grayscale implementation handles both the
//! binary (set) and grayscale (function) cases.
//!
//! * [`StructuringElement`] — a binary neighbourhood mask with an anchor, built
//!   via [`StructuringElement::rect`], [`StructuringElement::square`],
//!   [`StructuringElement::cross`], or [`StructuringElement::new`].
//! * [`erode`] — local minimum over the structuring element `B`:
//!   `(f ⊖ B)(x) = min_{b∈B} f(x + b)`.
//! * [`dilate`] — local maximum over the *reflected* structuring element:
//!   `(f ⊕ B)(x) = max_{b∈B} f(x − b)`. A single foreground point therefore
//!   dilates to a translated copy of `B`.
//! * [`open`] — [`erode`] followed by [`dilate`]; removes small bright details
//!   (idempotent).
//! * [`close`] — [`dilate`] followed by [`erode`]; fills small dark holes
//!   (idempotent).
//! * [`morphological_gradient`] — `dilate − erode`; highlights object
//!   boundaries.
//!
//! Out-of-bounds samples use **replicate** (clamp-to-edge) border handling,
//! consistent with the Sobel / Canny operators in this crate.

use crate::error::{VisionError, VisionResult};

/// A binary structuring element (neighbourhood mask) with an anchor.
///
/// The mask is a row-major `[height × width]` boolean grid; `true` cells are
/// members of the neighbourhood. The anchor (`anchor_y`, `anchor_x`) marks the
/// cell that aligns with the output pixel — typically the geometric centre.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuringElement {
    /// Element width (number of columns).
    pub width: usize,
    /// Element height (number of rows).
    pub height: usize,
    /// Anchor column (origin of the element), `< width`.
    pub anchor_x: usize,
    /// Anchor row (origin of the element), `< height`.
    pub anchor_y: usize,
    /// Row-major `[height × width]` membership mask (`true` = member).
    pub mask: Vec<bool>,
}

impl StructuringElement {
    /// Build a structuring element from an explicit `[height × width]` mask,
    /// anchored at its geometric centre (`height / 2`, `width / 2`).
    ///
    /// # Errors
    /// Returns [`VisionError::EmptyInput`] if the element is degenerate (zero
    /// dimension or no member cells) and [`VisionError::DimensionMismatch`] if
    /// `mask.len() != width * height`.
    pub fn new(width: usize, height: usize, mask: Vec<bool>) -> VisionResult<Self> {
        let se = Self {
            width,
            height,
            anchor_x: width / 2,
            anchor_y: height / 2,
            mask,
        };
        validate_se(&se)?;
        Ok(se)
    }

    /// Build a solid (all-member) `width × height` rectangular element anchored
    /// at its centre.
    ///
    /// # Errors
    /// Returns [`VisionError::EmptyInput`] if either dimension is zero.
    pub fn rect(width: usize, height: usize) -> VisionResult<Self> {
        if width == 0 || height == 0 {
            return Err(VisionError::EmptyInput("structuring element"));
        }
        Ok(Self {
            width,
            height,
            anchor_x: width / 2,
            anchor_y: height / 2,
            mask: vec![true; width * height],
        })
    }

    /// Build a solid `size × size` square element.
    ///
    /// # Errors
    /// Returns [`VisionError::EmptyInput`] if `size == 0`.
    pub fn square(size: usize) -> VisionResult<Self> {
        Self::rect(size, size)
    }

    /// Build a plus-shaped (4-connected) cross element of the given `radius`.
    ///
    /// The element is `(2·radius + 1)²` in size with the central row and column
    /// set. A `radius` of `0` yields the `1 × 1` identity element.
    #[must_use]
    pub fn cross(radius: usize) -> Self {
        let size = 2 * radius + 1;
        let mut mask = vec![false; size * size];
        for i in 0..size {
            mask[radius * size + i] = true;
            mask[i * size + radius] = true;
        }
        Self {
            width: size,
            height: size,
            anchor_x: radius,
            anchor_y: radius,
            mask,
        }
    }

    /// Collect the active `(dy, dx)` offsets relative to the anchor.
    fn active_offsets(&self) -> Vec<(isize, isize)> {
        let mut offsets = Vec::new();
        for row in 0..self.height {
            for col in 0..self.width {
                if self.mask[row * self.width + col] {
                    offsets.push((
                        row as isize - self.anchor_y as isize,
                        col as isize - self.anchor_x as isize,
                    ));
                }
            }
        }
        offsets
    }
}

/// Validate a structuring element's shape and non-emptiness.
fn validate_se(se: &StructuringElement) -> VisionResult<()> {
    if se.width == 0 || se.height == 0 {
        return Err(VisionError::EmptyInput("structuring element"));
    }
    if se.mask.len() != se.width * se.height {
        return Err(VisionError::DimensionMismatch {
            expected: se.width * se.height,
            got: se.mask.len(),
        });
    }
    if se.anchor_x >= se.width || se.anchor_y >= se.height {
        return Err(VisionError::Internal(format!(
            "structuring-element anchor ({}, {}) out of bounds for {}×{}",
            se.anchor_y, se.anchor_x, se.height, se.width
        )));
    }
    if !se.mask.iter().any(|&m| m) {
        return Err(VisionError::EmptyInput("structuring element"));
    }
    Ok(())
}

/// Validate a single-channel image buffer.
#[inline]
fn validate_gray(img: &[f32], h: usize, w: usize) -> VisionResult<()> {
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels: 1,
        });
    }
    if img.len() != h * w {
        return Err(VisionError::DimensionMismatch {
            expected: h * w,
            got: img.len(),
        });
    }
    Ok(())
}

/// Validate the image and structuring element together for an operation.
fn validate_op(img: &[f32], h: usize, w: usize, se: &StructuringElement) -> VisionResult<()> {
    validate_gray(img, h, w)?;
    validate_se(se)?;
    if se.height > h || se.width > w {
        return Err(VisionError::Internal(format!(
            "structuring element {}×{} larger than image {}×{}",
            se.height, se.width, h, w
        )));
    }
    Ok(())
}

/// Clamp a signed coordinate into `[0, n)` (replicate / clamp-to-edge border).
#[inline]
fn clamp_idx(v: isize, n: usize) -> usize {
    if v < 0 {
        0
    } else if v as usize >= n {
        n - 1
    } else {
        v as usize
    }
}

/// Grayscale / binary **erosion**: per-pixel minimum over the structuring
/// element, `(f ⊖ B)(x) = min_{b∈B} f(x + b)`.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] / [`VisionError::DimensionMismatch`]
/// for shape problems, [`VisionError::EmptyInput`] for a degenerate element, and
/// [`VisionError::Internal`] if the element is larger than the image.
pub fn erode(img: &[f32], h: usize, w: usize, se: &StructuringElement) -> VisionResult<Vec<f32>> {
    validate_op(img, h, w, se)?;
    let offsets = se.active_offsets();
    let mut out = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = f32::INFINITY;
            for &(dy, dx) in &offsets {
                let sy = clamp_idx(y as isize + dy, h);
                let sx = clamp_idx(x as isize + dx, w);
                let v = img[sy * w + sx];
                if v < acc {
                    acc = v;
                }
            }
            out[y * w + x] = acc;
        }
    }
    Ok(out)
}

/// Grayscale / binary **dilation**: per-pixel maximum over the reflected
/// structuring element, `(f ⊕ B)(x) = max_{b∈B} f(x − b)`.
///
/// A single foreground point dilates to a translated copy of `B`, so the
/// operator "grows" bright regions by the element's shape.
///
/// # Errors
/// Same as [`erode`].
pub fn dilate(img: &[f32], h: usize, w: usize, se: &StructuringElement) -> VisionResult<Vec<f32>> {
    validate_op(img, h, w, se)?;
    let offsets = se.active_offsets();
    let mut out = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = f32::NEG_INFINITY;
            for &(dy, dx) in &offsets {
                // Reflected element: sample at x − b.
                let sy = clamp_idx(y as isize - dy, h);
                let sx = clamp_idx(x as isize - dx, w);
                let v = img[sy * w + sx];
                if v > acc {
                    acc = v;
                }
            }
            out[y * w + x] = acc;
        }
    }
    Ok(out)
}

/// Morphological **opening**: [`erode`] then [`dilate`]. Removes small bright
/// structures (smaller than the element) while preserving the shape and size of
/// larger ones. Opening is idempotent: `open(open(f)) = open(f)`.
///
/// # Errors
/// Same as [`erode`].
pub fn open(img: &[f32], h: usize, w: usize, se: &StructuringElement) -> VisionResult<Vec<f32>> {
    let eroded = erode(img, h, w, se)?;
    dilate(&eroded, h, w, se)
}

/// Morphological **closing**: [`dilate`] then [`erode`]. Fills small dark holes
/// and gaps (smaller than the element) while preserving larger structures.
/// Closing is idempotent: `close(close(f)) = close(f)`.
///
/// # Errors
/// Same as [`erode`].
pub fn close(img: &[f32], h: usize, w: usize, se: &StructuringElement) -> VisionResult<Vec<f32>> {
    let dilated = dilate(img, h, w, se)?;
    erode(&dilated, h, w, se)
}

/// Morphological **gradient**: `dilate(f) − erode(f)`. Produces a thin response
/// along object boundaries (the difference between the grown and shrunk image).
///
/// # Errors
/// Same as [`erode`].
pub fn morphological_gradient(
    img: &[f32],
    h: usize,
    w: usize,
    se: &StructuringElement,
) -> VisionResult<Vec<f32>> {
    let dilated = dilate(img, h, w, se)?;
    let eroded = erode(img, h, w, se)?;
    let out = dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d - e)
        .collect();
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Image with a single foreground pixel at `(y, x)`.
    fn single_pixel(h: usize, w: usize, y: usize, x: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        img[y * w + x] = 1.0;
        img
    }

    /// Image with a solid `[r0..r1] × [c0..c1]` block of foreground.
    fn block(h: usize, w: usize, r0: usize, r1: usize, c0: usize, c1: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        for y in r0..r1 {
            for x in c0..c1 {
                img[y * w + x] = 1.0;
            }
        }
        img
    }

    #[test]
    fn erode_removes_isolated_pixel() {
        let img = single_pixel(7, 7, 3, 3);
        let se = StructuringElement::square(3).expect("se");
        let out = erode(&img, 7, 7, &se).expect("erode");
        let total: f32 = out.iter().sum();
        assert_eq!(total, 0.0, "erosion must remove an isolated pixel");
    }

    #[test]
    fn dilate_grows_single_pixel_to_se() {
        let img = single_pixel(5, 5, 2, 2);
        let se = StructuringElement::square(3).expect("se");
        let out = dilate(&img, 5, 5, &se).expect("dilate");
        for y in 0..5 {
            for x in 0..5 {
                let expected = if (1..=3).contains(&y) && (1..=3).contains(&x) {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    out[y * 5 + x],
                    expected,
                    "dilation of a point must reproduce the 3×3 element at ({y},{x})"
                );
            }
        }
    }

    #[test]
    fn dilate_constant_image_unchanged() {
        let img = vec![0.37_f32; 6 * 6];
        let se = StructuringElement::square(3).expect("se");
        let out = dilate(&img, 6, 6, &se).expect("dilate");
        assert!(out.iter().all(|&v| (v - 0.37).abs() < 1e-6));
    }

    #[test]
    fn erode_constant_image_unchanged() {
        let img = vec![0.42_f32; 6 * 6];
        let se = StructuringElement::square(3).expect("se");
        let out = erode(&img, 6, 6, &se).expect("erode");
        assert!(out.iter().all(|&v| (v - 0.42).abs() < 1e-6));
    }

    #[test]
    fn cross_se_dilation_shape() {
        let img = single_pixel(5, 5, 2, 2);
        let se = StructuringElement::cross(1);
        let out = dilate(&img, 5, 5, &se).expect("dilate");
        let total: f32 = out.iter().sum();
        assert_eq!(total, 5.0, "plus-shaped element has 5 members");
        // The plus arms.
        for &(y, x) in &[(2usize, 2usize), (1, 2), (3, 2), (2, 1), (2, 3)] {
            assert_eq!(out[y * 5 + x], 1.0);
        }
        // Diagonal corner must remain background.
        assert_eq!(out[3 * 5 + 3], 0.0);
    }

    #[test]
    fn opening_removes_noise_keeps_blob() {
        // 8×8: a solid 4×4 block (rows 2..6, cols 2..6) plus a single noise pixel.
        let mut img = block(8, 8, 2, 6, 2, 6);
        img[0] = 1.0; // isolated noise at (0, 0)
        let se = StructuringElement::square(3).expect("se");
        let out = open(&img, 8, 8, &se).expect("open");
        // Noise gone.
        assert_eq!(out[0], 0.0, "opening must remove the isolated noise pixel");
        // Block restored exactly (16 pixels, rows/cols 2..6).
        let total: f32 = out.iter().sum();
        assert_eq!(total, 16.0, "opening must preserve the 4×4 blob");
        for y in 2..6 {
            for x in 2..6 {
                assert_eq!(out[y * 8 + x], 1.0);
            }
        }
    }

    #[test]
    fn closing_fills_small_hole() {
        // Solid 4×4 block with a single-pixel hole punched at its centre.
        let mut img = block(8, 8, 2, 6, 2, 6);
        img[3 * 8 + 3] = 0.0; // hole
        let se = StructuringElement::square(3).expect("se");
        let out = close(&img, 8, 8, &se).expect("close");
        assert_eq!(out[3 * 8 + 3], 1.0, "closing must fill the interior hole");
        // The block is fully solid afterwards, no spill outside.
        let total: f32 = out.iter().sum();
        assert_eq!(total, 16.0);
    }

    #[test]
    fn opening_is_idempotent() {
        // Interior features with generous margin so border effects vanish.
        let mut img = block(12, 12, 5, 9, 5, 9);
        img[3 * 12 + 3] = 1.0; // interior noise, margin ≥ 3 from every border
        let se = StructuringElement::square(3).expect("se");
        let once = open(&img, 12, 12, &se).expect("open");
        let twice = open(&once, 12, 12, &se).expect("open");
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-6, "opening must be idempotent");
        }
    }

    #[test]
    fn closing_is_idempotent() {
        let mut img = block(12, 12, 4, 9, 4, 9);
        img[6 * 12 + 6] = 0.0; // interior hole
        let se = StructuringElement::square(3).expect("se");
        let once = close(&img, 12, 12, &se).expect("close");
        let twice = close(&once, 12, 12, &se).expect("close");
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-6, "closing must be idempotent");
        }
    }

    #[test]
    fn morph_gradient_outlines_point() {
        let img = single_pixel(5, 5, 2, 2);
        let se = StructuringElement::square(3).expect("se");
        let grad = morphological_gradient(&img, 5, 5, &se).expect("gradient");
        // erode → all zero, dilate → 3×3 block ⇒ gradient is the 3×3 block.
        let total: f32 = grad.iter().sum();
        assert_eq!(total, 9.0);
        assert!(grad.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn se_larger_than_image_errors() {
        let img = vec![0.0_f32; 4 * 4];
        let se = StructuringElement::square(7).expect("se");
        assert!(matches!(
            erode(&img, 4, 4, &se),
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            dilate(&img, 4, 4, &se),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn empty_se_errors() {
        // All-false mask → no members.
        let r = StructuringElement::new(3, 3, vec![false; 9]);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
        // Zero dimension.
        assert!(matches!(
            StructuringElement::rect(0, 3),
            Err(VisionError::EmptyInput(_))
        ));
    }

    #[test]
    fn se_mask_length_mismatch_errors() {
        let r = StructuringElement::new(3, 3, vec![true; 8]);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn wrong_image_size_errors() {
        let img = vec![0.0_f32; 10];
        let se = StructuringElement::square(3).expect("se");
        assert!(matches!(
            erode(&img, 8, 8, &se),
            Err(VisionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn zero_dim_image_errors() {
        let img: Vec<f32> = vec![];
        let se = StructuringElement::square(3).expect("se");
        assert!(matches!(
            dilate(&img, 0, 8, &se),
            Err(VisionError::InvalidImageSize { .. })
        ));
    }

    #[test]
    fn erode_is_anti_extensive_dilate_extensive() {
        // For any input, erosion ≤ f ≤ dilation pointwise (with a centred SE).
        let img = block(8, 8, 2, 6, 1, 5);
        let se = StructuringElement::square(3).expect("se");
        let er = erode(&img, 8, 8, &se).expect("erode");
        let di = dilate(&img, 8, 8, &se).expect("dilate");
        for ((&e, &f), &d) in er.iter().zip(img.iter()).zip(di.iter()) {
            assert!(e <= f + 1e-6, "erosion must be anti-extensive");
            assert!(d + 1e-6 >= f, "dilation must be extensive");
        }
    }
}
