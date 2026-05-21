//! Multi-scale anchor generator + greedy / Soft-NMS for two-stage detectors.
//!
//! A two-stage detector (Faster R-CNN, Mask R-CNN, RetinaNet, …) tiles each
//! feature-pyramid level with axis-aligned **anchor boxes** of multiple scales
//! and aspect ratios. The Region-Proposal Network (RPN) classifies and
//! regresses these anchors, then **Non-Maximum Suppression** (NMS) collapses
//! the overlapping proposals into a sparse set of detections.
//!
//! This module provides:
//! - [`AnchorGenerator`] — produces a flat `[n_anchors × 4]` table of
//!   `(x1, y1, x2, y2)` boxes for a list of feature-map levels.
//! - [`iou`] — Jaccard IoU between two axis-aligned boxes.
//! - [`nms`]      — greedy hard NMS (Neubeck & Van Gool 2006-style).
//! - [`soft_nms`] — linear / gaussian decay Soft-NMS (Bodla et al. 2017).
//!
//! ## Anchor convention
//! For a `base_size = b` and `aspect_ratio = r`:
//! ```text
//! width  = b · √r
//! height = b / √r
//! ```
//! The anchor is centred at the **cell centre** in image coordinates, i.e.
//! `(cx, cy) = (gx · stride + stride / 2, gy · stride + stride / 2)` where
//! `(gx, gy)` is the grid cell index and `stride` is the level stride.
//!
//! Per-level anchor count: `grid_h · grid_w · |base_sizes| · |aspect_ratios|`.
//! Total: `Σ_level` of the per-level count.
//!
//! ## Soft-NMS decay
//! ```text
//! linear  : s_new = s · (1 − iou)              if iou > iou_threshold
//! gaussian: s_new = s · exp(−iou² / sigma)     unconditional
//! ```
//! Boxes whose decayed score falls below `score_threshold` are dropped.
//!
//! All public functions are pure: they take row-major `Vec<f32>` /`[f32]`
//! inputs and return owned `Vec<…>` outputs.

use crate::error::{VisionError, VisionResult};

// ─── AnchorConfig ─────────────────────────────────────────────────────────────

/// Configuration for the [`AnchorGenerator`].
///
/// `base_sizes` and `aspect_ratios` define the anchor templates that are tiled
/// at **every** feature-map level. `feature_sizes[l]` is the `(grid_h, grid_w)`
/// of level `l`, and `strides[l]` is the level's spatial stride (e.g. 8, 16,
/// 32, 64, 128 for an FPN P3–P7 stack).
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorConfig {
    /// Base anchor edge lengths (in image-pixel units, ≥ 1 entry).
    pub base_sizes: Vec<f32>,
    /// Anchor aspect ratios (width / height, ≥ 1 entry, all > 0).
    pub aspect_ratios: Vec<f32>,
    /// Per-level grid size `(grid_h, grid_w)`. Must align with `strides`.
    pub feature_sizes: Vec<(usize, usize)>,
    /// Per-level stride in image pixels. Must align with `feature_sizes`.
    pub strides: Vec<usize>,
}

// ─── AnchorGenerator ──────────────────────────────────────────────────────────

/// Multi-scale anchor box generator.
#[derive(Debug, Clone)]
pub struct AnchorGenerator {
    cfg: AnchorConfig,
}

impl AnchorGenerator {
    /// Validate `cfg` and construct an `AnchorGenerator`.
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `base_sizes` or `aspect_ratios` is
    ///   empty.
    /// - [`VisionError::DimensionMismatch`] if `feature_sizes.len() !=
    ///   strides.len()`, or if any aspect-ratio / base-size / stride / grid
    ///   dim is non-positive.
    pub fn new(cfg: AnchorConfig) -> VisionResult<Self> {
        if cfg.base_sizes.is_empty() {
            return Err(VisionError::EmptyInput("anchor base_sizes"));
        }
        if cfg.aspect_ratios.is_empty() {
            return Err(VisionError::EmptyInput("anchor aspect_ratios"));
        }
        if cfg.feature_sizes.len() != cfg.strides.len() {
            return Err(VisionError::DimensionMismatch {
                expected: cfg.strides.len(),
                got: cfg.feature_sizes.len(),
            });
        }
        if cfg.feature_sizes.is_empty() {
            return Err(VisionError::EmptyInput("anchor feature_sizes"));
        }
        for &b in &cfg.base_sizes {
            if b <= 0.0 || !b.is_finite() {
                return Err(VisionError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
        }
        for &r in &cfg.aspect_ratios {
            if r <= 0.0 || !r.is_finite() {
                return Err(VisionError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
        }
        for &s in &cfg.strides {
            if s == 0 {
                return Err(VisionError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
        }
        for &(gh, gw) in &cfg.feature_sizes {
            if gh == 0 || gw == 0 {
                return Err(VisionError::InvalidImageSize {
                    height: gh,
                    width: gw,
                    channels: 1,
                });
            }
        }
        Ok(Self { cfg })
    }

    /// Read-only access to the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &AnchorConfig {
        &self.cfg
    }

    /// Total anchor count across all levels:
    /// `Σ_level grid_h · grid_w · |base_sizes| · |aspect_ratios|`.
    #[must_use]
    pub fn n_anchors(&self) -> usize {
        let templates_per_cell = self.cfg.base_sizes.len() * self.cfg.aspect_ratios.len();
        let mut total = 0usize;
        for &(gh, gw) in &self.cfg.feature_sizes {
            total += gh * gw * templates_per_cell;
        }
        total
    }

    /// Generate the flat anchor table.
    ///
    /// Layout: row-major `[n_anchors × 4]` with each row `(x1, y1, x2, y2)`
    /// in **image-pixel coordinates**. The traversal order is:
    /// ```text
    /// for level in 0..L:
    ///   for gy in 0..grid_h(level):
    ///     for gx in 0..grid_w(level):
    ///       for base_size in base_sizes:
    ///         for ratio in aspect_ratios:
    ///           emit (x1, y1, x2, y2)
    /// ```
    ///
    /// # Errors
    /// Only failure mode is exhausting allocator; on a well-formed config
    /// this returns `Ok`.
    pub fn generate(&self) -> VisionResult<Vec<f32>> {
        let n_total = self.n_anchors();
        let mut out = Vec::with_capacity(n_total * 4);

        for (level, &(grid_h, grid_w)) in self.cfg.feature_sizes.iter().enumerate() {
            // Caller-validated: strides.len() == feature_sizes.len(), so this
            // index is always in range.
            let stride = match self.cfg.strides.get(level) {
                Some(&s) => s as f32,
                None => {
                    return Err(VisionError::Internal(format!(
                        "stride index {level} out of range"
                    )));
                }
            };
            let half_stride = stride * 0.5;

            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let cx = gx as f32 * stride + half_stride;
                    let cy = gy as f32 * stride + half_stride;
                    for &base in &self.cfg.base_sizes {
                        for &ratio in &self.cfg.aspect_ratios {
                            let sqrt_r = ratio.sqrt();
                            let w = base * sqrt_r;
                            let h = base / sqrt_r;
                            let half_w = w * 0.5;
                            let half_h = h * 0.5;
                            out.push(cx - half_w);
                            out.push(cy - half_h);
                            out.push(cx + half_w);
                            out.push(cy + half_h);
                        }
                    }
                }
            }
        }

        Ok(out)
    }
}

// ─── IoU ──────────────────────────────────────────────────────────────────────

/// Jaccard Intersection-over-Union of two axis-aligned boxes `[x1, y1, x2, y2]`.
///
/// Degenerate boxes (zero area, or `x2 ≤ x1`, `y2 ≤ y1`) produce `0.0`. The
/// result is clamped to `[0, 1]`.
#[must_use]
pub fn iou(box_a: &[f32; 4], box_b: &[f32; 4]) -> f32 {
    let area_a = box_area(box_a);
    let area_b = box_area(box_b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let ix1 = box_a[0].max(box_b[0]);
    let iy1 = box_a[1].max(box_b[1]);
    let ix2 = box_a[2].min(box_b[2]);
    let iy2 = box_a[3].min(box_b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        return 0.0;
    }
    (inter / union).clamp(0.0, 1.0)
}

/// Positive area of a box, or 0 if degenerate.
#[inline]
fn box_area(b: &[f32; 4]) -> f32 {
    let w = b[2] - b[0];
    let h = b[3] - b[1];
    if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h }
}

/// Read box `i` out of a flat `[n × 4]` table.
#[inline]
fn read_box(boxes: &[f32], i: usize) -> [f32; 4] {
    let base = i * 4;
    [
        boxes[base],
        boxes[base + 1],
        boxes[base + 2],
        boxes[base + 3],
    ]
}

// ─── NMS ──────────────────────────────────────────────────────────────────────

/// Greedy non-maximum suppression.
///
/// Sorts the `n` candidates by `scores` in descending order, then sweeps
/// through them and keeps a candidate iff its IoU with every previously kept
/// box is **≤** `iou_threshold`. The result is capped at `max_keep` and
/// returned as **indices into the original `boxes` / `scores` arrays**, in
/// descending-score order.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `n == 0`.
/// - [`VisionError::DimensionMismatch`] if `boxes.len() != n * 4` or
///   `scores.len() != n`.
/// - [`VisionError::NonFinite`] if `iou_threshold` is not in `[0, 1]`.
pub fn nms(
    boxes: &[f32],
    scores: &[f32],
    n: usize,
    iou_threshold: f32,
    max_keep: usize,
) -> VisionResult<Vec<usize>> {
    if n == 0 {
        return Err(VisionError::EmptyInput("nms boxes"));
    }
    if boxes.len() != n * 4 {
        return Err(VisionError::DimensionMismatch {
            expected: n * 4,
            got: boxes.len(),
        });
    }
    if scores.len() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: scores.len(),
        });
    }
    if !(0.0..=1.0).contains(&iou_threshold) || !iou_threshold.is_finite() {
        return Err(VisionError::NonFinite("nms iou_threshold"));
    }

    // Stable descending sort by score (ties broken by lower index, which
    // makes the routine deterministic).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let cap = if max_keep == 0 { n } else { max_keep };
    let mut kept: Vec<usize> = Vec::with_capacity(cap.min(n));
    let mut kept_boxes: Vec<[f32; 4]> = Vec::with_capacity(cap.min(n));

    for &idx in &order {
        if kept.len() >= cap {
            break;
        }
        let candidate = read_box(boxes, idx);
        let mut suppress = false;
        for kb in &kept_boxes {
            if iou(&candidate, kb) > iou_threshold {
                suppress = true;
                break;
            }
        }
        if !suppress {
            kept.push(idx);
            kept_boxes.push(candidate);
        }
    }

    Ok(kept)
}

// ─── Soft-NMS ─────────────────────────────────────────────────────────────────

/// Soft-NMS (Bodla et al. 2017): instead of hard-suppressing overlapping
/// boxes, **decay** their scores and keep everything above
/// `score_threshold`.
///
/// This implementation uses the Gaussian decay variant
/// `s ← s · exp(−iou² / sigma)` (unconditional, applied for every overlap).
/// The "pivot" box (currently the highest remaining score) is appended to
/// the output, then every other surviving box has its score multiplied by
/// the decay factor before the next iteration.
///
/// Returns `(original_index, decayed_score)` pairs in **descending decayed
/// score order**. Pairs whose decayed score has dropped at or below
/// `score_threshold` are omitted.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `n == 0`.
/// - [`VisionError::DimensionMismatch`] if `boxes.len() != n * 4` or
///   `scores.len() != n`.
/// - [`VisionError::NonFinite`] if `sigma <= 0`.
pub fn soft_nms(
    boxes: &[f32],
    scores: &[f32],
    n: usize,
    sigma: f32,
    score_threshold: f32,
) -> VisionResult<Vec<(usize, f32)>> {
    if n == 0 {
        return Err(VisionError::EmptyInput("soft_nms boxes"));
    }
    if boxes.len() != n * 4 {
        return Err(VisionError::DimensionMismatch {
            expected: n * 4,
            got: boxes.len(),
        });
    }
    if scores.len() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: scores.len(),
        });
    }
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(VisionError::NonFinite("soft_nms sigma"));
    }
    if !score_threshold.is_finite() {
        return Err(VisionError::NonFinite("soft_nms score_threshold"));
    }

    // Working buffers — (original_index, current_score, box).
    let mut pool: Vec<(usize, f32, [f32; 4])> =
        (0..n).map(|i| (i, scores[i], read_box(boxes, i))).collect();

    let inv_sigma = 1.0_f32 / sigma;
    let mut out: Vec<(usize, f32)> = Vec::new();

    while !pool.is_empty() {
        // Find the index of the maximum-scored entry remaining in the pool.
        let (max_pos, max_score) = pool.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best_i, best_s), (i, e)| {
                if e.1 > best_s {
                    (i, e.1)
                } else {
                    (best_i, best_s)
                }
            },
        );

        // Pop pivot.
        let pivot = pool.swap_remove(max_pos);

        if max_score <= score_threshold {
            // No remaining entry can beat threshold (all remaining scores
            // ≤ current max ≤ threshold), so we stop.
            break;
        }
        out.push((pivot.0, pivot.1));

        // Decay every remaining entry by Gaussian overlap factor.
        for entry in pool.iter_mut() {
            let ov = iou(&pivot.2, &entry.2);
            let decay = (-(ov * ov) * inv_sigma).exp();
            entry.1 *= decay;
        }
    }

    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a tiny single-level anchor config.
    fn cfg_single_level() -> AnchorConfig {
        AnchorConfig {
            base_sizes: vec![32.0],
            aspect_ratios: vec![1.0],
            feature_sizes: vec![(2, 2)],
            strides: vec![16],
        }
    }

    fn cfg_two_levels() -> AnchorConfig {
        AnchorConfig {
            base_sizes: vec![16.0, 32.0],
            aspect_ratios: vec![0.5, 1.0, 2.0],
            feature_sizes: vec![(2, 3), (1, 2)],
            strides: vec![8, 16],
        }
    }

    // ── IoU ───────────────────────────────────────────────────────────────────

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let v = iou(&a, &a);
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = [0.0_f32, 0.0, 1.0, 1.0];
        let b = [2.0_f32, 2.0, 3.0, 3.0];
        assert!(iou(&a, &b).abs() < 1e-7);
    }

    #[test]
    fn iou_half_overlap_known_value() {
        // Two 10×10 boxes whose right/left edges meet at x=5:
        // a = [0,0,10,10] (area 100), b = [5,0,15,10] (area 100)
        // inter = 5*10 = 50, union = 100 + 100 - 50 = 150
        // iou = 50 / 150 = 1/3
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let b = [5.0_f32, 0.0, 15.0, 10.0];
        let v = iou(&a, &b);
        assert!((v - (1.0 / 3.0)).abs() < 1e-5, "expected 1/3, got {v}");
    }

    #[test]
    fn iou_degenerate_box_is_zero() {
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let b = [5.0_f32, 5.0, 5.0, 5.0]; // zero area
        assert!(iou(&a, &b).abs() < 1e-7);
    }

    // ── AnchorGenerator construction ──────────────────────────────────────────

    #[test]
    fn anchor_generator_construction_ok() {
        let g = AnchorGenerator::new(cfg_single_level()).expect("ok");
        assert_eq!(g.config().base_sizes, vec![32.0]);
    }

    #[test]
    fn anchor_count_matches_sigma_formula() {
        let cfg = cfg_two_levels();
        let level0 = 2 * 3 * 2 * 3; // 6 cells × 6 templates = 36
        let level1 = 2 * 2 * 3; // 2 cells × 6 templates = 12
        let expected = level0 + level1;
        let g = AnchorGenerator::new(cfg).expect("ok");
        assert_eq!(g.n_anchors(), expected);
    }

    #[test]
    fn anchor_generate_output_length_matches_n_anchors() {
        let g = AnchorGenerator::new(cfg_two_levels()).expect("ok");
        let out = g.generate().expect("generate ok");
        assert_eq!(out.len(), g.n_anchors() * 4);
    }

    #[test]
    fn anchor_boxes_have_positive_extent() {
        let g = AnchorGenerator::new(cfg_two_levels()).expect("ok");
        let out = g.generate().expect("ok");
        let n = g.n_anchors();
        for i in 0..n {
            let b = read_box(&out, i);
            assert!(b[2] > b[0], "anchor {i} x2 <= x1: {b:?}");
            assert!(b[3] > b[1], "anchor {i} y2 <= y1: {b:?}");
        }
    }

    #[test]
    fn anchor_center_at_cell_centre() {
        // Single-level, single-template, 2×2 grid, stride 16.
        // Centres should be (8,8), (24,8), (8,24), (24,24).
        let g = AnchorGenerator::new(cfg_single_level()).expect("ok");
        let out = g.generate().expect("ok");
        // Anchor 0: cell (0,0), centre (8,8), size 32 → (-8,-8,24,24)
        let b0 = read_box(&out, 0);
        let cx0 = 0.5 * (b0[0] + b0[2]);
        let cy0 = 0.5 * (b0[1] + b0[3]);
        assert!((cx0 - 8.0).abs() < 1e-5);
        assert!((cy0 - 8.0).abs() < 1e-5);
        // Anchor 3: cell (gy=1,gx=1), centre (24,24)
        let b3 = read_box(&out, 3);
        let cx3 = 0.5 * (b3[0] + b3[2]);
        let cy3 = 0.5 * (b3[1] + b3[3]);
        assert!((cx3 - 24.0).abs() < 1e-5, "got cx={cx3}");
        assert!((cy3 - 24.0).abs() < 1e-5, "got cy={cy3}");
    }

    #[test]
    fn anchor_size_follows_sqrt_ratio() {
        // base = 32, ratio = 4.0 ⇒ w = 32·2 = 64, h = 32/2 = 16.
        let cfg = AnchorConfig {
            base_sizes: vec![32.0],
            aspect_ratios: vec![4.0],
            feature_sizes: vec![(1, 1)],
            strides: vec![16],
        };
        let g = AnchorGenerator::new(cfg).expect("ok");
        let out = g.generate().expect("ok");
        let b = read_box(&out, 0);
        let w = b[2] - b[0];
        let h = b[3] - b[1];
        assert!((w - 64.0).abs() < 1e-4, "expected w=64, got {w}");
        assert!((h - 16.0).abs() < 1e-4, "expected h=16, got {h}");
    }

    // ── AnchorGenerator validation errors ─────────────────────────────────────

    #[test]
    fn err_empty_base_sizes() {
        let cfg = AnchorConfig {
            base_sizes: vec![],
            aspect_ratios: vec![1.0],
            feature_sizes: vec![(2, 2)],
            strides: vec![16],
        };
        let r = AnchorGenerator::new(cfg);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn err_empty_aspect_ratios() {
        let cfg = AnchorConfig {
            base_sizes: vec![1.0],
            aspect_ratios: vec![],
            feature_sizes: vec![(2, 2)],
            strides: vec![16],
        };
        let r = AnchorGenerator::new(cfg);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn err_feature_size_strides_length_mismatch() {
        let cfg = AnchorConfig {
            base_sizes: vec![1.0],
            aspect_ratios: vec![1.0],
            feature_sizes: vec![(2, 2), (1, 1)],
            strides: vec![16],
        };
        let r = AnchorGenerator::new(cfg);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_zero_grid_size() {
        let cfg = AnchorConfig {
            base_sizes: vec![1.0],
            aspect_ratios: vec![1.0],
            feature_sizes: vec![(0, 2)],
            strides: vec![16],
        };
        let r = AnchorGenerator::new(cfg);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn err_zero_base_size() {
        let cfg = AnchorConfig {
            base_sizes: vec![0.0],
            aspect_ratios: vec![1.0],
            feature_sizes: vec![(2, 2)],
            strides: vec![16],
        };
        let r = AnchorGenerator::new(cfg);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── NMS ───────────────────────────────────────────────────────────────────

    #[test]
    fn nms_keeps_highest_score_and_suppresses_overlap() {
        // Two heavily overlapping boxes; the second has the higher score.
        let boxes = vec![0.0_f32, 0.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0];
        let scores = vec![0.4_f32, 0.9];
        let kept = nms(&boxes, &scores, 2, 0.3, 0).expect("ok");
        // Only the higher-scored box (index 1) should survive.
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn nms_keeps_two_disjoint_boxes() {
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
        let scores = vec![0.9_f32, 0.8];
        let kept = nms(&boxes, &scores, 2, 0.5, 0).expect("ok");
        assert_eq!(kept.len(), 2);
        // Returned in descending-score order: index 0 first (0.9), then 1.
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn nms_max_keep_caps_output() {
        let boxes = vec![
            0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        ];
        let scores = vec![0.9_f32, 0.7, 0.5];
        // All three are disjoint, but we cap at 2.
        let kept = nms(&boxes, &scores, 3, 0.5, 2).expect("ok");
        assert_eq!(kept.len(), 2);
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn nms_threshold_one_keeps_all() {
        // Even identical boxes are kept when threshold == 1.0 (only strictly
        // greater overlaps suppress).
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let scores = vec![0.9_f32, 0.8];
        let kept = nms(&boxes, &scores, 2, 1.0, 0).expect("ok");
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_single_box_is_kept() {
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0];
        let scores = vec![0.5_f32];
        let kept = nms(&boxes, &scores, 1, 0.5, 0).expect("ok");
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn nms_returns_indices_in_descending_score_order() {
        let boxes = vec![
            0.0_f32, 0.0, 1.0, 1.0, // box 0
            5.0_f32, 0.0, 6.0, 1.0, // box 1
            10.0_f32, 0.0, 11.0, 1.0, // box 2
        ];
        // Scores intentionally ordered so the sort matters.
        let scores = vec![0.3_f32, 0.9, 0.5];
        let kept = nms(&boxes, &scores, 3, 0.5, 0).expect("ok");
        assert_eq!(kept, vec![1, 2, 0]);
    }

    #[test]
    fn nms_err_empty_boxes() {
        let r = nms(&[], &[], 0, 0.5, 0);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn nms_err_length_mismatch() {
        // boxes has 4 entries (1 box) but n=2
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0];
        let scores = vec![0.9_f32, 0.8];
        let r = nms(&boxes, &scores, 2, 0.5, 0);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn nms_err_iou_out_of_range() {
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0];
        let scores = vec![0.5_f32];
        let r = nms(&boxes, &scores, 1, 1.5, 0);
        assert!(matches!(r, Err(VisionError::NonFinite(_))));
        let r2 = nms(&boxes, &scores, 1, -0.1, 0);
        assert!(matches!(r2, Err(VisionError::NonFinite(_))));
    }

    #[test]
    fn nms_deterministic() {
        let boxes = vec![
            0.0_f32, 0.0, 1.0, 1.0, // 0
            0.5, 0.5, 1.5, 1.5, // 1 — overlaps 0
            5.0, 5.0, 6.0, 6.0, // 2 — disjoint
        ];
        let scores = vec![0.4_f32, 0.7, 0.6];
        let k1 = nms(&boxes, &scores, 3, 0.3, 0).expect("ok");
        let k2 = nms(&boxes, &scores, 3, 0.3, 0).expect("ok");
        assert_eq!(k1, k2);
    }

    // ── Soft-NMS ──────────────────────────────────────────────────────────────

    #[test]
    fn soft_nms_decays_overlapping_score() {
        // Two heavily overlapping boxes. The lower-scored one must have its
        // score multiplied by exp(-iou^2 / sigma) (< 1).
        let boxes = vec![0.0_f32, 0.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0];
        let scores = vec![0.9_f32, 0.8];
        let out = soft_nms(&boxes, &scores, 2, 0.5, 0.0).expect("ok");
        assert_eq!(out.len(), 2);
        // First pivot is the highest-scored (index 0).
        assert_eq!(out[0].0, 0);
        // The second is index 1 with a decayed score < 0.8.
        assert_eq!(out[1].0, 1);
        assert!(out[1].1 < 0.8, "expected decay, got {}", out[1].1);
    }

    #[test]
    fn soft_nms_drops_below_score_threshold() {
        // With a high threshold, the second overlapping box should drop out.
        let boxes = vec![0.0_f32, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0];
        let scores = vec![1.0_f32, 0.5];
        // Both boxes are identical → iou=1, decay = exp(-1/sigma). With
        // sigma=0.5 this is exp(-2) ≈ 0.135 — well below 0.4.
        let out = soft_nms(&boxes, &scores, 2, 0.5, 0.4).expect("ok");
        // Only the pivot survives.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
    }

    #[test]
    fn soft_nms_disjoint_no_decay() {
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
        let scores = vec![0.7_f32, 0.6];
        let out = soft_nms(&boxes, &scores, 2, 0.5, 0.0).expect("ok");
        assert_eq!(out.len(), 2);
        // No decay because IoU is 0 → exp(0) = 1.
        assert!((out[0].1 - 0.7).abs() < 1e-5);
        assert!((out[1].1 - 0.6).abs() < 1e-5);
    }

    #[test]
    fn soft_nms_deterministic() {
        let boxes = vec![
            0.0_f32, 0.0, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5, 5.0, 5.0, 6.0, 6.0,
        ];
        let scores = vec![0.4_f32, 0.7, 0.6];
        let a = soft_nms(&boxes, &scores, 3, 0.5, 0.1).expect("ok");
        let b = soft_nms(&boxes, &scores, 3, 0.5, 0.1).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn soft_nms_err_invalid_sigma() {
        let boxes = vec![0.0_f32, 0.0, 1.0, 1.0];
        let scores = vec![0.5_f32];
        let r = soft_nms(&boxes, &scores, 1, 0.0, 0.0);
        assert!(matches!(r, Err(VisionError::NonFinite(_))));
        let r2 = soft_nms(&boxes, &scores, 1, -1.0, 0.0);
        assert!(matches!(r2, Err(VisionError::NonFinite(_))));
    }

    #[test]
    fn soft_nms_err_empty() {
        let r = soft_nms(&[], &[], 0, 0.5, 0.0);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }
}
