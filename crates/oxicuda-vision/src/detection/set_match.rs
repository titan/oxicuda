//! Hungarian-style bipartite matching for DETR set prediction loss.
//!
//! Implements a greedy + 2-opt bipartite matching algorithm for ≤32 queries,
//! and utilities to build the cost matrix from predicted class logits, predicted
//! boxes, and ground-truth targets.
//!
//! ## Cost components
//! - **Class cost**: cross-entropy loss for the predicted class probability
//!   of the ground-truth class.
//! - **L1 box cost**: L1 distance between predicted and target boxes (CxCyWH).
//! - **GIoU cost**: GIoU distance (1 − GIoU) between predicted and target boxes.

use crate::error::{VisionError, VisionResult};

// ─── MatchCost ────────────────────────────────────────────────────────────────

/// Individual cost components for one (query, target) pair.
#[derive(Debug, Clone)]
pub struct MatchCost {
    /// Negative log-probability of the true class: −log p(target_class).
    pub class_cost: f32,
    /// L1 distance between predicted and target boxes.
    pub l1_box_cost: f32,
    /// GIoU-based distance: 1 − GIoU(pred_box, target_box).
    pub giou_cost: f32,
}

// ─── bipartite_match ─────────────────────────────────────────────────────────

/// Greedy + 2-opt bipartite matching.
///
/// Works well for small instances (DETR typically uses ≤100 queries; the
/// 2-opt improvement pass handles near-optimal matching).
///
/// # Parameters
/// - `cost_matrix`: flat `[n_queries × n_targets]` (lower cost = better match).
/// - `n_queries`, `n_targets`: dimensions of the cost matrix.
///
/// # Returns
/// `Vec<(query_idx, target_idx)>` with exactly `min(n_queries, n_targets)` pairs.
/// Each index appears at most once in the returned pairs.
///
/// # Errors
/// - `DimensionMismatch` if `cost_matrix.len() != n_queries * n_targets`.
/// - `EmptyInput` if `n_queries == 0` or `n_targets == 0`.
pub fn bipartite_match(
    cost_matrix: &[f32],
    n_queries: usize,
    n_targets: usize,
) -> VisionResult<Vec<(usize, usize)>> {
    if n_queries == 0 || n_targets == 0 {
        return Err(VisionError::EmptyInput(
            "bipartite_match: empty queries or targets",
        ));
    }

    let expected = n_queries * n_targets;
    if cost_matrix.len() != expected {
        return Err(VisionError::DimensionMismatch {
            expected,
            got: cost_matrix.len(),
        });
    }

    let n_pairs = n_queries.min(n_targets);

    // ── Phase 1: greedy matching ───────────────────────────────────────────────
    // Enumerate all (q, t) pairs and sort by ascending cost.
    let mut candidates: Vec<(usize, usize, f32)> = Vec::with_capacity(n_queries * n_targets);
    for q in 0..n_queries {
        for t in 0..n_targets {
            candidates.push((q, t, cost_matrix[q * n_targets + t]));
        }
    }
    // Sort by cost ascending (using total order: NaN sorts last).
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Greater));

    let mut query_used = vec![false; n_queries];
    let mut target_used = vec![false; n_targets];
    let mut matches: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);

    for (q, t, _cost) in &candidates {
        if matches.len() == n_pairs {
            break;
        }
        if !query_used[*q] && !target_used[*t] {
            matches.push((*q, *t));
            query_used[*q] = true;
            target_used[*t] = true;
        }
    }

    // ── Phase 2: 2-opt improvement ────────────────────────────────────────────
    // Repeatedly try all pairwise swaps of two assignments; accept if total
    // cost decreases.  Repeat until no improvement in a full pass.
    loop {
        let mut improved = false;
        let m = matches.len();
        'outer: for i in 0..m {
            for j in (i + 1)..m {
                let (qi, ti) = matches[i];
                let (qj, tj) = matches[j];
                let cost_before =
                    cost_matrix[qi * n_targets + ti] + cost_matrix[qj * n_targets + tj];
                // Option A: swap targets (qi↔tj, qj↔ti)
                let cost_swap_t =
                    cost_matrix[qi * n_targets + tj] + cost_matrix[qj * n_targets + ti];
                if cost_swap_t < cost_before - 1e-8 {
                    matches[i] = (qi, tj);
                    matches[j] = (qj, ti);
                    improved = true;
                    break 'outer; // restart pass after any improvement
                }
            }
        }
        if !improved {
            break;
        }
    }

    Ok(matches)
}

// ─── build_cost_matrix ────────────────────────────────────────────────────────

/// Build the matching cost matrix from predictions and ground-truth targets.
///
/// # Parameters
/// - `pred_logits`:    `[n_queries × n_classes]` raw class logits.
/// - `n_queries`:      number of object query vectors.
/// - `n_classes`:      number of object classes.
/// - `pred_boxes`:     `[n_queries × 4]` predicted boxes in `(cx, cy, w, h)` format.
/// - `target_labels`:  `[n_targets]` ground-truth class indices.
/// - `target_boxes`:   `[n_targets × 4]` ground-truth boxes in `(cx, cy, w, h)` format.
/// - `n_targets`:      number of ground-truth targets.
/// - `class_weight`:   scalar weight for the class cost term.
/// - `l1_weight`:      scalar weight for the L1 box cost term.
/// - `giou_weight`:    scalar weight for the GIoU cost term.
///
/// # Returns
/// Flat `[n_queries × n_targets]` cost matrix.
///
/// # Errors
/// - `DimensionMismatch` if tensor sizes do not match the declared shapes.
/// - `EmptyInput` if `n_queries == 0` or `n_targets == 0`.
#[allow(clippy::too_many_arguments)]
pub fn build_cost_matrix(
    pred_logits: &[f32],
    n_queries: usize,
    n_classes: usize,
    pred_boxes: &[f32],
    target_labels: &[usize],
    target_boxes: &[f32],
    n_targets: usize,
    class_weight: f32,
    l1_weight: f32,
    giou_weight: f32,
) -> VisionResult<Vec<f32>> {
    if n_queries == 0 {
        return Err(VisionError::EmptyInput("build_cost_matrix: n_queries=0"));
    }
    if n_targets == 0 {
        return Err(VisionError::EmptyInput("build_cost_matrix: n_targets=0"));
    }
    if n_classes == 0 {
        return Err(VisionError::EmptyInput("build_cost_matrix: n_classes=0"));
    }

    let expected_logits = n_queries * n_classes;
    if pred_logits.len() != expected_logits {
        return Err(VisionError::DimensionMismatch {
            expected: expected_logits,
            got: pred_logits.len(),
        });
    }
    let expected_boxes = n_queries * 4;
    if pred_boxes.len() != expected_boxes {
        return Err(VisionError::DimensionMismatch {
            expected: expected_boxes,
            got: pred_boxes.len(),
        });
    }
    if target_labels.len() != n_targets {
        return Err(VisionError::DimensionMismatch {
            expected: n_targets,
            got: target_labels.len(),
        });
    }
    let expected_tgt_boxes = n_targets * 4;
    if target_boxes.len() != expected_tgt_boxes {
        return Err(VisionError::DimensionMismatch {
            expected: expected_tgt_boxes,
            got: target_boxes.len(),
        });
    }

    // Pre-compute softmax probabilities for all queries: [n_queries × n_classes]
    let probs = softmax_rows_2d(pred_logits, n_queries, n_classes);

    let mut cost = vec![0.0f32; n_queries * n_targets];

    for q in 0..n_queries {
        let pb: [f32; 4] = [
            pred_boxes[q * 4],
            pred_boxes[q * 4 + 1],
            pred_boxes[q * 4 + 2],
            pred_boxes[q * 4 + 3],
        ];
        let q_probs = &probs[q * n_classes..(q + 1) * n_classes];

        for t in 0..n_targets {
            let cls = target_labels[t];
            // Class cost: −log p(target_class) (clamped to avoid log(0))
            let prob_cls = q_probs.get(cls).copied().unwrap_or(0.0).max(1e-10);
            let class_c = -prob_cls.ln();

            // L1 box cost
            let tb: [f32; 4] = [
                target_boxes[t * 4],
                target_boxes[t * 4 + 1],
                target_boxes[t * 4 + 2],
                target_boxes[t * 4 + 3],
            ];
            let l1_c = (pb[0] - tb[0]).abs()
                + (pb[1] - tb[1]).abs()
                + (pb[2] - tb[2]).abs()
                + (pb[3] - tb[3]).abs();

            // GIoU cost: 1 − GIoU
            let giou_val = giou(&pb, &tb);
            let giou_c = 1.0 - giou_val;

            cost[q * n_targets + t] =
                class_weight * class_c + l1_weight * l1_c + giou_weight * giou_c;
        }
    }

    Ok(cost)
}

// ─── giou ─────────────────────────────────────────────────────────────────────

/// Compute the GIoU (Generalised Intersection over Union) for two boxes.
///
/// Both boxes are in `[cx, cy, w, h]` (centre-format) with values normalised
/// to `[0, 1]`.  Returns a value in `[−1, 1]` where 1 = perfect overlap.
///
/// Formula:
/// ```text
/// IoU = intersection / union
/// GIoU = IoU − (enclosing_area − union) / enclosing_area
/// ```
pub fn giou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    // Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    let (ax1, ay1, ax2, ay2) = cxcywh_to_xyxy(b1);
    let (bx1, by1, bx2, by2) = cxcywh_to_xyxy(b2);

    // Intersection
    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let intersection = inter_w * inter_h;

    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - intersection;

    let iou = if union > 1e-10 {
        intersection / union
    } else {
        0.0
    };

    // Enclosing box
    let ex1 = ax1.min(bx1);
    let ey1 = ay1.min(by1);
    let ex2 = ax2.max(bx2);
    let ey2 = ay2.max(by2);
    let enclosing = ((ex2 - ex1).max(0.0)) * ((ey2 - ey1).max(0.0));

    if enclosing > 1e-10 {
        iou - (enclosing - union) / enclosing
    } else {
        iou
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Convert `[cx, cy, w, h]` to `[x1, y1, x2, y2]`.
#[inline]
fn cxcywh_to_xyxy(b: &[f32; 4]) -> (f32, f32, f32, f32) {
    let (cx, cy, w, h) = (b[0], b[1], b[2], b[3]);
    (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)
}

/// Row-wise softmax returning a new `Vec<f32>` of the same shape.
fn softmax_rows_2d(logits: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = logits.to_vec();
    for i in 0..n_rows {
        let row = &mut out[i * n_cols..(i + 1) * n_cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── giou ──────────────────────────────────────────────────────────────────

    #[test]
    fn giou_identical_boxes_is_one() {
        let b = [0.5f32, 0.5, 0.4, 0.4];
        let v = giou(&b, &b);
        assert!((v - 1.0).abs() < 1e-5, "identical boxes: GIoU={v}");
    }

    #[test]
    fn giou_non_overlapping_boxes_negative() {
        // Two boxes far apart should give negative GIoU.
        let b1 = [0.1f32, 0.1, 0.1, 0.1];
        let b2 = [0.9f32, 0.9, 0.1, 0.1];
        let v = giou(&b1, &b2);
        assert!(v < 0.0, "non-overlapping: GIoU={v} should be negative");
    }

    #[test]
    fn giou_partially_overlapping_between_neg1_and_1() {
        let b1 = [0.5f32, 0.5, 0.6, 0.6];
        let b2 = [0.6f32, 0.6, 0.6, 0.6];
        let v = giou(&b1, &b2);
        assert!((-1.0..=1.0).contains(&v), "GIoU={v} out of range");
    }

    #[test]
    fn giou_degenerate_zero_area_box() {
        // Zero-area box: should not panic, GIoU should be in valid range.
        let b1 = [0.5f32, 0.5, 0.0, 0.0];
        let b2 = [0.5f32, 0.5, 0.4, 0.4];
        let v = giou(&b1, &b2);
        assert!(v.is_finite(), "GIoU should be finite for zero-area box");
    }

    // ── bipartite_match ───────────────────────────────────────────────────────

    #[test]
    fn bipartite_match_identity_diagonal() {
        // 3×3 cost matrix: diagonal = 0, off-diagonal = 1
        // Expected: (0,0), (1,1), (2,2)
        #[rustfmt::skip]
        let cost = vec![
            0.0f32, 1.0, 1.0,
            1.0,    0.0, 1.0,
            1.0,    1.0, 0.0,
        ];
        let pairs = bipartite_match(&cost, 3, 3).expect("match ok");
        assert_eq!(pairs.len(), 3, "should produce 3 pairs");

        let mut matched_queries: Vec<usize> = pairs.iter().map(|&(q, _)| q).collect();
        let mut matched_targets: Vec<usize> = pairs.iter().map(|&(_, t)| t).collect();
        matched_queries.sort_unstable();
        matched_targets.sort_unstable();

        assert_eq!(matched_queries, vec![0, 1, 2], "all queries matched");
        assert_eq!(matched_targets, vec![0, 1, 2], "all targets matched");

        // Verify total cost is 0 (optimal: diagonal).
        let total: f32 = pairs.iter().map(|&(q, t)| cost[q * 3 + t]).sum();
        assert!(total.abs() < 1e-6, "expected total cost 0, got {total}");
    }

    #[test]
    fn bipartite_match_more_queries_than_targets() {
        // 4 queries, 2 targets: only 2 pairs returned.
        #[rustfmt::skip]
        let cost = vec![
            0.1f32, 0.9,
            0.8,    0.2,
            0.5,    0.5,
            0.7,    0.3,
        ];
        let pairs = bipartite_match(&cost, 4, 2).expect("match ok");
        assert_eq!(pairs.len(), 2, "should produce min(4,2)=2 pairs");
        // Each target should appear at most once.
        let t0_count = pairs.iter().filter(|&&(_, t)| t == 0).count();
        let t1_count = pairs.iter().filter(|&&(_, t)| t == 1).count();
        assert!(t0_count <= 1 && t1_count <= 1, "no duplicate targets");
    }

    #[test]
    fn bipartite_match_empty_errors() {
        let cost = vec![1.0f32];
        assert!(bipartite_match(&cost, 0, 1).is_err());
        assert!(bipartite_match(&cost, 1, 0).is_err());
    }

    #[test]
    fn bipartite_match_wrong_matrix_size_errors() {
        let cost = vec![0.0f32; 5]; // should be 3*3=9
        let r = bipartite_match(&cost, 3, 3);
        assert!(
            matches!(
                r,
                Err(VisionError::DimensionMismatch {
                    expected: 9,
                    got: 5
                })
            ),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn bipartite_match_1x1() {
        let cost = vec![0.5f32];
        let pairs = bipartite_match(&cost, 1, 1).expect("1x1 match ok");
        assert_eq!(pairs, vec![(0, 0)]);
    }

    // ── build_cost_matrix ─────────────────────────────────────────────────────

    #[test]
    fn build_cost_matrix_shape() {
        let n_queries = 4;
        let n_classes = 3;
        let n_targets = 2;

        let logits = vec![0.0f32; n_queries * n_classes];
        let boxes = vec![0.5f32; n_queries * 4];
        let target_labels = vec![0usize, 1];
        let target_boxes = vec![0.5f32; n_targets * 4];

        let cost = build_cost_matrix(
            &logits,
            n_queries,
            n_classes,
            &boxes,
            &target_labels,
            &target_boxes,
            n_targets,
            1.0,
            1.0,
            1.0,
        )
        .expect("build_cost_matrix ok");

        assert_eq!(
            cost.len(),
            n_queries * n_targets,
            "cost matrix shape [n_queries × n_targets]"
        );
    }

    #[test]
    fn build_cost_matrix_all_values_finite() {
        let n_queries = 6;
        let n_classes = 4;
        let n_targets = 3;
        let mut rng = LcgRng::new(99);
        let mut logits = vec![0.0f32; n_queries * n_classes];
        rng.fill_normal(&mut logits);
        let mut boxes = vec![0.0f32; n_queries * 4];
        for b in boxes.iter_mut() {
            *b = rng.next_f32();
        }
        let target_labels: Vec<usize> = (0..n_targets).map(|i| i % n_classes).collect();
        let mut target_boxes = vec![0.0f32; n_targets * 4];
        for b in target_boxes.iter_mut() {
            *b = rng.next_f32();
        }

        let cost = build_cost_matrix(
            &logits,
            n_queries,
            n_classes,
            &boxes,
            &target_labels,
            &target_boxes,
            n_targets,
            1.0,
            5.0,
            2.0,
        )
        .expect("build_cost_matrix ok");

        assert!(
            cost.iter().all(|v| v.is_finite()),
            "all cost entries should be finite"
        );
    }

    #[test]
    fn build_cost_matrix_empty_queries_errors() {
        let logits: Vec<f32> = vec![];
        let boxes: Vec<f32> = vec![];
        let r = build_cost_matrix(
            &logits,
            0,
            3,
            &boxes,
            &[0],
            &[0.5, 0.5, 0.2, 0.2],
            1,
            1.0,
            1.0,
            1.0,
        );
        assert!(r.is_err());
    }

    // ── Integration: match on cost matrix built from predictions ──────────────

    #[test]
    fn bipartite_match_on_cost_matrix_no_duplicates() {
        let n_queries = 4;
        let n_classes = 2;
        let n_targets = 2;
        let logits = vec![0.0f32; n_queries * n_classes];
        let boxes: Vec<f32> = (0..n_queries)
            .flat_map(|q| {
                let cx = 0.2 + 0.2 * q as f32;
                vec![cx, 0.5f32, 0.1, 0.1]
            })
            .collect();
        let target_labels = vec![0usize, 1];
        let target_boxes = vec![0.3f32, 0.5, 0.1, 0.1, 0.7, 0.5, 0.1, 0.1];

        let cost = build_cost_matrix(
            &logits,
            n_queries,
            n_classes,
            &boxes,
            &target_labels,
            &target_boxes,
            n_targets,
            1.0,
            5.0,
            2.0,
        )
        .expect("cost matrix ok");

        let pairs = bipartite_match(&cost, n_queries, n_targets).expect("match ok");
        assert_eq!(pairs.len(), 2, "exactly min(4,2)=2 pairs");

        // No duplicate query or target indices.
        let qs: std::collections::HashSet<usize> = pairs.iter().map(|&(q, _)| q).collect();
        let ts: std::collections::HashSet<usize> = pairs.iter().map(|&(_, t)| t).collect();
        assert_eq!(qs.len(), 2, "all assigned queries distinct");
        assert_eq!(ts.len(), 2, "all assigned targets distinct");
    }
}
