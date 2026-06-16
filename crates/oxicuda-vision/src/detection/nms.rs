//! Non-Maximum Suppression over scored bounding boxes.
//!
//! Object detectors emit many overlapping candidate boxes for each ground-truth
//! object. **Non-Maximum Suppression** (NMS) collapses these into a sparse set
//! by greedily keeping the highest-scoring box and discarding any later box
//! whose Intersection-over-Union (IoU) with an already-kept box exceeds a
//! threshold.
//!
//! **Soft-NMS** (Bodla et al., 2017) replaces the hard discard with a Gaussian
//! score *decay*: an overlapping box keeps its identity but has its score
//! multiplied by `exp(−IoU² / sigma)`, so heavily-overlapping low-confidence
//! boxes fade out gradually rather than being deleted outright. Boxes whose
//! decayed score falls at or below `score_threshold` are dropped.
//!
//! This module operates on the ergonomic struct-of-fields [`BBox`] type
//! (`x1, y1, x2, y2, score`). The flat `[n × 4]` array-based variants used by
//! the anchor / RPN pipeline live in [`crate::detection::anchor_nms`].

use crate::error::{VisionError, VisionResult};

// ─── BBox ────────────────────────────────────────────────────────────────────

/// An axis-aligned bounding box with a detection confidence score.
///
/// Coordinates follow the half-open convention `[x1, x2) × [y1, y2)`; a box is
/// **degenerate** (zero area) when `x2 ≤ x1` or `y2 ≤ y1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Left edge (inclusive).
    pub x1: f32,
    /// Top edge (inclusive).
    pub y1: f32,
    /// Right edge (exclusive).
    pub x2: f32,
    /// Bottom edge (exclusive).
    pub y2: f32,
    /// Detection confidence score.
    pub score: f32,
}

impl BBox {
    /// Construct a new box.
    #[must_use]
    #[inline]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            score,
        }
    }

    /// Positive area of the box, or `0.0` if degenerate.
    #[must_use]
    #[inline]
    pub fn area(&self) -> f32 {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h }
    }
}

// ─── IoU ─────────────────────────────────────────────────────────────────────

/// Jaccard Intersection-over-Union of two boxes.
///
/// Degenerate boxes (zero area) yield `0.0`. The result is clamped to `[0, 1]`.
#[must_use]
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let area_a = a.area();
    let area_b = b.area();
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        return 0.0;
    }
    (inter / union).clamp(0.0, 1.0)
}

// ─── NMS ─────────────────────────────────────────────────────────────────────

/// Greedy hard Non-Maximum Suppression.
///
/// Sorts the input boxes by descending score (ties broken by ascending input
/// index for determinism), then sweeps through them keeping a box iff its IoU
/// with every previously-kept box is **≤** `iou_threshold`. Returns the kept
/// **indices into the original `boxes` slice**, in descending-score order.
///
/// An empty input returns an empty `Vec` (no error). With
/// `iou_threshold == 1.0`, only IoU strictly greater than 1 suppresses — i.e.
/// nothing — so every box is kept.
///
/// # Errors
/// - [`VisionError::NonFinite`] if `iou_threshold` is NaN/∞ or outside `[0, 1]`.
pub fn nms(boxes: &[BBox], iou_threshold: f32) -> VisionResult<Vec<usize>> {
    if !(0.0..=1.0).contains(&iou_threshold) || !iou_threshold.is_finite() {
        return Err(VisionError::NonFinite("nms iou_threshold"));
    }
    if boxes.is_empty() {
        return Ok(Vec::new());
    }

    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut kept: Vec<usize> = Vec::new();
    for &idx in &order {
        let candidate = &boxes[idx];
        let mut suppressed = false;
        for &k in &kept {
            if iou(candidate, &boxes[k]) > iou_threshold {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            kept.push(idx);
        }
    }
    Ok(kept)
}

// ─── Soft-NMS ────────────────────────────────────────────────────────────────

/// Gaussian Soft-NMS (Bodla et al., 2017).
///
/// Instead of hard-suppressing overlapping boxes, repeatedly select the
/// highest-scoring remaining box as the *pivot*, emit it, then multiply every
/// other remaining box's score by the Gaussian decay `exp(−IoU(pivot, ·)² /
/// sigma)`. Boxes whose decayed score drops **at or below** `score_threshold`
/// are removed and never emitted.
///
/// Returns `(original_index, decayed_score)` pairs in descending decayed-score
/// (emission) order. An empty input returns an empty `Vec`.
///
/// # Errors
/// - [`VisionError::NonFinite`] if `sigma ≤ 0`, or if `sigma` /
///   `score_threshold` is non-finite.
pub fn soft_nms(
    boxes: &[BBox],
    sigma: f32,
    score_threshold: f32,
) -> VisionResult<Vec<(usize, f32)>> {
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(VisionError::NonFinite("soft_nms sigma"));
    }
    if !score_threshold.is_finite() {
        return Err(VisionError::NonFinite("soft_nms score_threshold"));
    }
    if boxes.is_empty() {
        return Ok(Vec::new());
    }

    let inv_sigma = 1.0_f32 / sigma;
    // Pool of (original index, current score) — boxes themselves are read by
    // index from `boxes`.
    let mut pool: Vec<(usize, f32)> = boxes.iter().map(|b| (b.score, b)).enumerate().fold(
        Vec::with_capacity(boxes.len()),
        |mut acc, (i, (s, _))| {
            acc.push((i, s));
            acc
        },
    );

    let mut out: Vec<(usize, f32)> = Vec::new();

    while !pool.is_empty() {
        // Highest-scoring remaining entry becomes the pivot.
        let (max_pos, max_score) = pool.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best_i, best_s), (i, &(_, s))| {
                if s > best_s { (i, s) } else { (best_i, best_s) }
            },
        );

        if max_score <= score_threshold {
            // Every remaining score ≤ max_score ≤ threshold: stop.
            break;
        }

        let pivot = pool.swap_remove(max_pos);
        out.push(pivot);

        let pivot_box = &boxes[pivot.0];
        for entry in pool.iter_mut() {
            let ov = iou(pivot_box, &boxes[entry.0]);
            let decay = (-(ov * ov) * inv_sigma).exp();
            entry.1 *= decay;
        }
    }

    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x1: f32, y1: f32, x2: f32, y2: f32, s: f32) -> BBox {
        BBox::new(x1, y1, x2, y2, s)
    }

    #[test]
    fn iou_identical_is_1() {
        let a = b(0.0, 0.0, 10.0, 10.0, 0.9);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_is_0() {
        let a = b(0.0, 0.0, 1.0, 1.0, 0.9);
        let c = b(2.0, 2.0, 3.0, 3.0, 0.8);
        assert!(iou(&a, &c).abs() < 1e-7);
    }

    #[test]
    fn iou_half_overlap() {
        // a = [0,0,10,10] (area 100), c = [5,0,15,10] (area 100)
        // inter = 5·10 = 50, union = 150 ⇒ iou = 1/3.
        let a = b(0.0, 0.0, 10.0, 10.0, 0.9);
        let c = b(5.0, 0.0, 15.0, 10.0, 0.8);
        assert!((iou(&a, &c) - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn nms_keeps_highest() {
        // Two heavily overlapping boxes; index 1 has the higher score.
        let boxes = vec![b(0.0, 0.0, 10.0, 10.0, 0.4), b(1.0, 1.0, 10.0, 10.0, 0.9)];
        let kept = nms(&boxes, 0.3).expect("ok");
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn nms_suppresses_overlap() {
        // Three boxes: 0 and 1 overlap heavily, 2 is disjoint.
        let boxes = vec![
            b(0.0, 0.0, 10.0, 10.0, 0.9),
            b(0.5, 0.5, 10.5, 10.5, 0.8),
            b(50.0, 50.0, 60.0, 60.0, 0.7),
        ];
        let kept = nms(&boxes, 0.3).expect("ok");
        // Box 1 suppressed by 0; 0 and 2 survive.
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn nms_keeps_disjoint() {
        let boxes = vec![b(0.0, 0.0, 1.0, 1.0, 0.9), b(5.0, 5.0, 6.0, 6.0, 0.8)];
        let kept = nms(&boxes, 0.5).expect("ok");
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn soft_nms_decays_scores() {
        // Two overlapping boxes; the lower-scored survivor must be decayed.
        let boxes = vec![b(0.0, 0.0, 10.0, 10.0, 0.9), b(1.0, 1.0, 10.0, 10.0, 0.8)];
        let out = soft_nms(&boxes, 0.5, 0.0).expect("ok");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[1].0, 1);
        assert!(out[1].1 < 0.8, "expected decay, got {}", out[1].1);
    }

    #[test]
    fn empty_boxes() {
        let boxes: Vec<BBox> = Vec::new();
        assert_eq!(nms(&boxes, 0.5).expect("ok"), Vec::<usize>::new());
        assert_eq!(
            soft_nms(&boxes, 0.5, 0.0).expect("ok"),
            Vec::<(usize, f32)>::new()
        );
    }

    #[test]
    fn threshold_1_keeps_all() {
        // Even identical boxes are kept when threshold == 1.0.
        let boxes = vec![b(0.0, 0.0, 1.0, 1.0, 0.9), b(0.0, 0.0, 1.0, 1.0, 0.8)];
        let kept = nms(&boxes, 1.0).expect("ok");
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn soft_nms_threshold_filters() {
        // Two identical boxes ⇒ iou = 1, decay = exp(-1/0.5) = exp(-2) ≈ 0.135.
        // The second box's decayed score 0.5·0.135 ≈ 0.068 < 0.4 ⇒ dropped.
        let boxes = vec![b(0.0, 0.0, 10.0, 10.0, 1.0), b(0.0, 0.0, 10.0, 10.0, 0.5)];
        let out = soft_nms(&boxes, 0.5, 0.4).expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
    }

    // ── Additional coverage ──────────────────────────────────────────────────

    #[test]
    fn nms_invalid_threshold_errors() {
        let boxes = vec![b(0.0, 0.0, 1.0, 1.0, 0.9)];
        assert!(matches!(nms(&boxes, 1.5), Err(VisionError::NonFinite(_))));
        assert!(matches!(nms(&boxes, -0.1), Err(VisionError::NonFinite(_))));
        assert!(matches!(
            nms(&boxes, f32::NAN),
            Err(VisionError::NonFinite(_))
        ));
    }

    #[test]
    fn soft_nms_invalid_sigma_errors() {
        let boxes = vec![b(0.0, 0.0, 1.0, 1.0, 0.9)];
        assert!(matches!(
            soft_nms(&boxes, 0.0, 0.0),
            Err(VisionError::NonFinite(_))
        ));
        assert!(matches!(
            soft_nms(&boxes, -1.0, 0.0),
            Err(VisionError::NonFinite(_))
        ));
    }

    #[test]
    fn nms_returns_descending_score_order() {
        let boxes = vec![
            b(0.0, 0.0, 1.0, 1.0, 0.3),
            b(5.0, 0.0, 6.0, 1.0, 0.9),
            b(10.0, 0.0, 11.0, 1.0, 0.5),
        ];
        let kept = nms(&boxes, 0.5).expect("ok");
        assert_eq!(kept, vec![1, 2, 0]);
    }

    #[test]
    fn soft_nms_disjoint_no_decay() {
        let boxes = vec![b(0.0, 0.0, 1.0, 1.0, 0.7), b(5.0, 5.0, 6.0, 6.0, 0.6)];
        let out = soft_nms(&boxes, 0.5, 0.0).expect("ok");
        assert_eq!(out.len(), 2);
        assert!((out[0].1 - 0.7).abs() < 1e-5);
        assert!((out[1].1 - 0.6).abs() < 1e-5);
    }

    #[test]
    fn iou_degenerate_is_0() {
        let a = b(0.0, 0.0, 10.0, 10.0, 0.9);
        let degenerate = b(5.0, 5.0, 5.0, 5.0, 0.8);
        assert!(iou(&a, &degenerate).abs() < 1e-7);
    }
}
