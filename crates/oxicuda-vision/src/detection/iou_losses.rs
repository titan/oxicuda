//! IoU-family bounding-box regression losses.
//!
//! Object-detection regression heads benefit from losses defined directly on the
//! Intersection-over-Union (IoU) metric rather than on raw coordinate offsets,
//! because IoU is scale-invariant and matches the evaluation metric. This module
//! implements the four canonical members of the family:
//!
//! * **IoU loss** — `1 − IoU` (Yu et al. 2016, *UnitBox*).
//! * **GIoU loss** — Generalised IoU (Rezatofighi et al. 2019, CVPR): adds a
//!   penalty for the area of the smallest enclosing box not covered by the
//!   union, so the gradient is non-zero even for non-overlapping boxes.
//! * **DIoU loss** — Distance IoU (Zheng et al. 2020, AAAI): penalises the
//!   squared centre distance normalised by the enclosing-box diagonal, giving
//!   faster convergence than GIoU.
//! * **CIoU loss** — Complete IoU (Zheng et al. 2020, AAAI): augments DIoU with
//!   an aspect-ratio consistency term `α·v`.
//!
//! ## Box formats
//! Both centre format `(cx, cy, w, h)` and corner format `(x1, y1, x2, y2)` are
//! supported through [`IouBox`]. Internally every box is normalised to corner
//! format. Degenerate (non-positive width/height) boxes are rejected.
//!
//! ## Loss conventions
//! Each loss returns a non-negative scalar in `f32`:
//! ```text
//! iou_loss  = 1 − IoU                        ∈ [0, 1]
//! giou_loss = 1 − GIoU                       ∈ [0, 2]
//! diou_loss = 1 − DIoU                       ∈ [0, 2]
//! ciou_loss = 1 − CIoU                       ∈ [0, 2 + …]
//! ```
//! The `*_pairs` helpers average the per-box loss over a batch.

use crate::error::{VisionError, VisionResult};

/// Axis-aligned bounding box in corner format `(x1, y1, x2, y2)`.
///
/// `x2 > x1` and `y2 > y1` are required for a valid (positive-area) box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IouBox {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
}

impl IouBox {
    /// Build a box from corner coordinates `(x1, y1, x2, y2)`.
    #[must_use]
    #[inline]
    pub fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Build a box from centre format `(cx, cy, w, h)`.
    #[must_use]
    #[inline]
    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let hw = 0.5 * w;
        let hh = 0.5 * h;
        Self {
            x1: cx - hw,
            y1: cy - hh,
            x2: cx + hw,
            y2: cy + hh,
        }
    }

    /// Width `x2 − x1` (may be non-positive for a degenerate box).
    #[must_use]
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Height `y2 − y1` (may be non-positive for a degenerate box).
    #[must_use]
    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Positive area, or `0.0` if degenerate.
    #[must_use]
    #[inline]
    pub fn area(&self) -> f32 {
        let w = self.width();
        let h = self.height();
        if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h }
    }

    /// Centre `(cx, cy)`.
    #[must_use]
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    }

    /// Validate that the box has positive width and height.
    fn validate(&self, name: &'static str) -> VisionResult<()> {
        if self.width() <= 0.0 || self.height() <= 0.0 {
            return Err(VisionError::InvalidRoiBox {
                x1: self.x1,
                y1: self.y1,
                x2: self.x2,
                y2: self.y2,
            });
        }
        if !(self.x1.is_finite()
            && self.y1.is_finite()
            && self.x2.is_finite()
            && self.y2.is_finite())
        {
            return Err(VisionError::NonFinite(name));
        }
        Ok(())
    }
}

/// Geometric quantities shared by every IoU-family loss.
struct IouGeometry {
    iou: f32,
    /// Area of the union of the two boxes.
    union: f32,
    /// Area of the smallest axis-aligned box enclosing both inputs.
    enclosing_area: f32,
    /// Squared length of the enclosing-box diagonal.
    enclosing_diag_sq: f32,
    /// Squared distance between the two box centres.
    center_dist_sq: f32,
}

/// Compute the shared IoU geometry for a predicted/target box pair.
fn iou_geometry(pred: &IouBox, target: &IouBox) -> IouGeometry {
    let area_p = pred.area();
    let area_t = target.area();

    let ix1 = pred.x1.max(target.x1);
    let iy1 = pred.y1.max(target.y1);
    let ix2 = pred.x2.min(target.x2);
    let iy2 = pred.y2.min(target.y2);
    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let inter = inter_w * inter_h;

    let union = (area_p + area_t - inter).max(0.0);
    let iou = if union > 1e-12 { inter / union } else { 0.0 };

    // Smallest enclosing box.
    let ex1 = pred.x1.min(target.x1);
    let ey1 = pred.y1.min(target.y1);
    let ex2 = pred.x2.max(target.x2);
    let ey2 = pred.y2.max(target.y2);
    let enc_w = (ex2 - ex1).max(0.0);
    let enc_h = (ey2 - ey1).max(0.0);
    let enclosing_area = enc_w * enc_h;
    let enclosing_diag_sq = enc_w * enc_w + enc_h * enc_h;

    let (pcx, pcy) = pred.center();
    let (tcx, tcy) = target.center();
    let dx = pcx - tcx;
    let dy = pcy - tcy;
    let center_dist_sq = dx * dx + dy * dy;

    IouGeometry {
        iou: iou.clamp(0.0, 1.0),
        union,
        enclosing_area,
        enclosing_diag_sq,
        center_dist_sq,
    }
}

/// Raw IoU (Jaccard overlap) of two boxes, in `[0, 1]`.
pub fn iou(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    pred.validate("iou pred")?;
    target.validate("iou target")?;
    Ok(iou_geometry(pred, target).iou)
}

/// IoU regression loss `1 − IoU` (Yu et al. 2016), in `[0, 1]`.
pub fn iou_loss(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    Ok(1.0 - iou(pred, target)?)
}

/// Generalised IoU value (Rezatofighi et al. 2019), in `[−1, 1]`.
///
/// `GIoU = IoU − (enclosing − union) / enclosing`.
pub fn giou(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    pred.validate("giou pred")?;
    target.validate("giou target")?;
    let g = iou_geometry(pred, target);
    let value = if g.enclosing_area > 1e-12 {
        g.iou - (g.enclosing_area - g.union) / g.enclosing_area
    } else {
        g.iou
    };
    Ok(value)
}

/// GIoU regression loss `1 − GIoU`, in `[0, 2]`.
pub fn giou_loss(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    Ok(1.0 - giou(pred, target)?)
}

/// Distance IoU value (Zheng et al. 2020), in `[−1, 1]`.
///
/// `DIoU = IoU − ρ²(centres) / c²` where `c` is the enclosing-box diagonal.
pub fn diou(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    pred.validate("diou pred")?;
    target.validate("diou target")?;
    let g = iou_geometry(pred, target);
    let penalty = if g.enclosing_diag_sq > 1e-12 {
        g.center_dist_sq / g.enclosing_diag_sq
    } else {
        0.0
    };
    Ok(g.iou - penalty)
}

/// DIoU regression loss `1 − DIoU`, in `[0, 2]`.
pub fn diou_loss(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    Ok(1.0 - diou(pred, target)?)
}

/// Complete IoU value (Zheng et al. 2020), in `(−∞, 1]` (practically near `[−1, 1]`).
///
/// `CIoU = IoU − ρ²/c² − α·v`, where
/// `v = (4/π²)·(atan(w_t/h_t) − atan(w_p/h_p))²` measures aspect-ratio
/// inconsistency and `α = v / ((1 − IoU) + v)` is a positive trade-off weight.
pub fn ciou(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    pred.validate("ciou pred")?;
    target.validate("ciou target")?;
    let g = iou_geometry(pred, target);

    let dist_penalty = if g.enclosing_diag_sq > 1e-12 {
        g.center_dist_sq / g.enclosing_diag_sq
    } else {
        0.0
    };

    let wp = pred.width();
    let hp = pred.height();
    let wt = target.width();
    let ht = target.height();
    let inv_pi2 = 4.0 / (std::f32::consts::PI * std::f32::consts::PI);
    let angle_t = (wt / ht).atan();
    let angle_p = (wp / hp).atan();
    let diff = angle_t - angle_p;
    let v = inv_pi2 * diff * diff;

    // α is treated as a constant w.r.t. backprop in the original paper, but the
    // forward value is identical; we compute it directly.
    let denom = (1.0 - g.iou) + v;
    let alpha = if denom > 1e-12 { v / denom } else { 0.0 };

    Ok(g.iou - dist_penalty - alpha * v)
}

/// CIoU regression loss `1 − CIoU`, in `[0, ≈2]`.
pub fn ciou_loss(pred: &IouBox, target: &IouBox) -> VisionResult<f32> {
    Ok(1.0 - ciou(pred, target)?)
}

/// Which IoU-family loss to apply in a batched reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IouLossKind {
    /// `1 − IoU`.
    Iou,
    /// `1 − GIoU`.
    Giou,
    /// `1 − DIoU`.
    Diou,
    /// `1 − CIoU`.
    Ciou,
}

/// Mean IoU-family loss over a batch of `(pred, target)` box pairs.
///
/// Returns [`VisionError::EmptyInput`] if `pairs` is empty.
pub fn iou_loss_pairs(pairs: &[(IouBox, IouBox)], kind: IouLossKind) -> VisionResult<f32> {
    if pairs.is_empty() {
        return Err(VisionError::EmptyInput("iou_loss_pairs"));
    }
    let mut acc = 0.0f32;
    for (pred, target) in pairs {
        acc += match kind {
            IouLossKind::Iou => iou_loss(pred, target)?,
            IouLossKind::Giou => giou_loss(pred, target)?,
            IouLossKind::Diou => diou_loss(pred, target)?,
            IouLossKind::Ciou => ciou_loss(pred, target)?,
        };
    }
    Ok(acc / pairs.len() as f32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn unit() -> IouBox {
        IouBox::from_xyxy(0.0, 0.0, 1.0, 1.0)
    }

    #[test]
    fn cxcywh_matches_xyxy() {
        let a = IouBox::from_cxcywh(0.5, 0.5, 1.0, 1.0);
        let b = unit();
        assert!((a.x1 - b.x1).abs() < TOL);
        assert!((a.y2 - b.y2).abs() < TOL);
        assert!((a.area() - 1.0).abs() < TOL);
    }

    #[test]
    fn identical_boxes_have_zero_losses() {
        let a = unit();
        assert!(iou_loss(&a, &a).expect("iou") < TOL);
        assert!(giou_loss(&a, &a).expect("giou") < TOL);
        assert!(diou_loss(&a, &a).expect("diou") < TOL);
        assert!(ciou_loss(&a, &a).expect("ciou") < TOL);
    }

    #[test]
    fn iou_half_overlap() {
        // Two unit boxes shifted by 0.5 in x: intersection 0.5, union 1.5 → IoU 1/3.
        let a = unit();
        let b = IouBox::from_xyxy(0.5, 0.0, 1.5, 1.0);
        let v = iou(&a, &b).expect("iou");
        assert!((v - 1.0 / 3.0).abs() < 1e-4, "iou={v}");
        let l = iou_loss(&a, &b).expect("loss");
        assert!((l - (1.0 - 1.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn giou_non_overlapping_is_negative() {
        let a = unit();
        let b = IouBox::from_xyxy(5.0, 5.0, 6.0, 6.0);
        let g = giou(&a, &b).expect("giou");
        assert!(g < 0.0, "giou={g}");
        // Loss in (1, 2] for fully separated boxes.
        let l = giou_loss(&a, &b).expect("loss");
        assert!(l > 1.0 && l <= 2.0 + TOL, "loss={l}");
    }

    #[test]
    fn giou_le_iou() {
        // GIoU is always ≤ IoU.
        let a = unit();
        let b = IouBox::from_xyxy(0.3, 0.4, 1.3, 1.4);
        let g = giou(&a, &b).expect("giou");
        let i = iou(&a, &b).expect("iou");
        assert!(g <= i + TOL, "giou={g} iou={i}");
    }

    #[test]
    fn diou_penalises_center_distance() {
        // Same IoU geometry but DIoU < IoU when centres differ.
        let a = unit();
        let b = IouBox::from_xyxy(0.5, 0.0, 1.5, 1.0);
        let d = diou(&a, &b).expect("diou");
        let i = iou(&a, &b).expect("iou");
        assert!(d < i, "diou={d} iou={i}");
    }

    #[test]
    fn diou_concentric_equals_iou() {
        // Concentric boxes: centre distance is zero → DIoU == IoU.
        let a = unit();
        let b = IouBox::from_cxcywh(0.5, 0.5, 0.5, 0.5);
        let d = diou(&a, &b).expect("diou");
        let i = iou(&a, &b).expect("iou");
        assert!((d - i).abs() < TOL, "diou={d} iou={i}");
    }

    #[test]
    fn ciou_aspect_ratio_term_nonpositive_contribution() {
        // CIoU ≤ DIoU because the aspect-ratio term α·v ≥ 0 is subtracted.
        let a = unit();
        let b = IouBox::from_cxcywh(0.5, 0.5, 0.5, 2.0); // different aspect ratio
        let c = ciou(&a, &b).expect("ciou");
        let d = diou(&a, &b).expect("diou");
        assert!(c <= d + TOL, "ciou={c} diou={d}");
    }

    #[test]
    fn ciou_same_aspect_ratio_equals_diou() {
        // Boxes with identical aspect ratio: v = 0 → CIoU == DIoU.
        let a = unit();
        let b = IouBox::from_cxcywh(0.7, 0.7, 0.5, 0.5); // square like a
        let c = ciou(&a, &b).expect("ciou");
        let d = diou(&a, &b).expect("diou");
        assert!((c - d).abs() < 1e-4, "ciou={c} diou={d}");
    }

    #[test]
    fn losses_are_nonnegative() {
        let a = unit();
        let cases = [
            IouBox::from_xyxy(0.2, 0.1, 1.1, 0.9),
            IouBox::from_xyxy(2.0, 2.0, 3.0, 3.5),
            IouBox::from_cxcywh(0.5, 0.5, 2.0, 0.5),
        ];
        for b in &cases {
            assert!(iou_loss(&a, b).expect("iou") >= -TOL);
            assert!(giou_loss(&a, b).expect("giou") >= -TOL);
            assert!(diou_loss(&a, b).expect("diou") >= -TOL);
            assert!(ciou_loss(&a, b).expect("ciou") >= -TOL);
        }
    }

    #[test]
    fn degenerate_box_is_rejected() {
        let good = unit();
        let bad = IouBox::from_xyxy(1.0, 1.0, 0.5, 2.0); // x2 < x1
        assert!(iou(&bad, &good).is_err());
        assert!(giou_loss(&good, &bad).is_err());
        assert!(ciou_loss(&bad, &bad).is_err());
    }

    #[test]
    fn nonfinite_box_is_rejected() {
        let good = unit();
        let nan = IouBox::from_xyxy(0.0, 0.0, f32::NAN, 1.0);
        assert!(iou(&nan, &good).is_err());
    }

    #[test]
    fn batched_mean_matches_manual() {
        let a = unit();
        let b = IouBox::from_xyxy(0.5, 0.0, 1.5, 1.0);
        let pairs = vec![(a, a), (a, b)];
        let mean = iou_loss_pairs(&pairs, IouLossKind::Iou).expect("mean");
        let manual = 0.5 * (iou_loss(&a, &a).expect("l0") + iou_loss(&a, &b).expect("l1"));
        assert!((mean - manual).abs() < TOL, "mean={mean} manual={manual}");
    }

    #[test]
    fn batched_empty_is_error() {
        let pairs: Vec<(IouBox, IouBox)> = Vec::new();
        assert!(iou_loss_pairs(&pairs, IouLossKind::Giou).is_err());
    }

    #[test]
    fn ordering_giou_diou_ciou_bounded() {
        // For overlapping boxes all values lie in [-1, 1].
        let a = unit();
        let b = IouBox::from_xyxy(0.25, 0.25, 1.25, 1.1);
        for v in [
            giou(&a, &b).expect("g"),
            diou(&a, &b).expect("d"),
            ciou(&a, &b).expect("c"),
        ] {
            assert!((-1.0..=1.0).contains(&v), "value out of range: {v}");
        }
    }
}
