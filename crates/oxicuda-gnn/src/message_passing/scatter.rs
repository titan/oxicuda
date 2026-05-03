//! Scatter and gather operations for message passing.

use crate::error::{GnnError, GnnResult};

fn validate_scatter(src: &[f32], idx: &[usize], n_out: usize, feat_dim: usize) -> GnnResult<usize> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feat_dim must be > 0".to_string(),
        ));
    }
    let n = idx.len();
    if src.len() != n * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n * feat_dim,
            got: src.len(),
        });
    }
    for &i in idx {
        if i >= n_out {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: i,
                n_nodes: n_out,
            });
        }
    }
    Ok(n)
}

/// Scatter-add: `out[idx[i], k] += src[i, k]` for all `i`.
///
/// Returns `[n_out × feat_dim]` initialized to 0.
pub fn scatter_add(
    src: &[f32],
    idx: &[usize],
    n_out: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n = validate_scatter(src, idx, n_out, feat_dim)?;
    let mut out = vec![0.0_f32; n_out * feat_dim];
    for i in 0..n {
        let o = idx[i];
        for k in 0..feat_dim {
            out[o * feat_dim + k] += src[i * feat_dim + k];
        }
    }
    Ok(out)
}

/// Scatter-max: `out[idx[i], k] = max(out[idx[i], k], src[i, k])`.
///
/// Returns `(values [n_out × feat_dim], argmax [n_out × feat_dim])`.
/// Positions with no contributing elements get value 0 and `None` argmax.
pub fn scatter_max(
    src: &[f32],
    idx: &[usize],
    n_out: usize,
    feat_dim: usize,
) -> GnnResult<(Vec<f32>, Vec<Option<usize>>)> {
    let n = validate_scatter(src, idx, n_out, feat_dim)?;
    let mut out = vec![f32::NEG_INFINITY; n_out * feat_dim];
    let mut argmax: Vec<Option<usize>> = vec![None; n_out * feat_dim];

    for i in 0..n {
        let o = idx[i];
        for k in 0..feat_dim {
            let v = src[i * feat_dim + k];
            let pos = o * feat_dim + k;
            if v > out[pos] {
                out[pos] = v;
                argmax[pos] = Some(i);
            }
        }
    }
    // Replace -Inf (empty bins) with 0
    for val in &mut out {
        if *val == f32::NEG_INFINITY {
            *val = 0.0;
        }
    }
    Ok((out, argmax))
}

/// Scatter-min: `out[idx[i], k] = min(out[idx[i], k], src[i, k])`.
///
/// Returns `(values [n_out × feat_dim], argmin [n_out × feat_dim])`.
pub fn scatter_min(
    src: &[f32],
    idx: &[usize],
    n_out: usize,
    feat_dim: usize,
) -> GnnResult<(Vec<f32>, Vec<Option<usize>>)> {
    let n = validate_scatter(src, idx, n_out, feat_dim)?;
    let mut out = vec![f32::INFINITY; n_out * feat_dim];
    let mut argmin: Vec<Option<usize>> = vec![None; n_out * feat_dim];

    for i in 0..n {
        let o = idx[i];
        for k in 0..feat_dim {
            let v = src[i * feat_dim + k];
            let pos = o * feat_dim + k;
            if v < out[pos] {
                out[pos] = v;
                argmin[pos] = Some(i);
            }
        }
    }
    for val in &mut out {
        if *val == f32::INFINITY {
            *val = 0.0;
        }
    }
    Ok((out, argmin))
}

/// Scatter-multiply: `out[idx[i], k] *= src[i, k]`.
///
/// Positions with no contributing elements are initialized to 1 (identity for multiply)
/// then reset to 0 after.  This uses a presence mask.
pub fn scatter_mul(
    src: &[f32],
    idx: &[usize],
    n_out: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n = validate_scatter(src, idx, n_out, feat_dim)?;
    let mut out = vec![1.0_f32; n_out * feat_dim];
    let mut has_value = vec![false; n_out * feat_dim];

    for i in 0..n {
        let o = idx[i];
        for k in 0..feat_dim {
            let pos = o * feat_dim + k;
            out[pos] *= src[i * feat_dim + k];
            has_value[pos] = true;
        }
    }
    for pos in 0..out.len() {
        if !has_value[pos] {
            out[pos] = 0.0;
        }
    }
    Ok(out)
}

/// Gather: `out[i, k] = src[idx[i], k]` for all `i`.
///
/// Returns `[idx.len() × feat_dim]`.
pub fn gather(src: &[f32], idx: &[usize], feat_dim: usize) -> GnnResult<Vec<f32>> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feat_dim must be > 0".to_string(),
        ));
    }
    let n_src = src.len() / feat_dim;
    if src.len() != n_src * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_src * feat_dim,
            got: src.len(),
        });
    }
    for &i in idx {
        if i >= n_src {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: i,
                n_nodes: n_src,
            });
        }
    }
    let mut out = Vec::with_capacity(idx.len() * feat_dim);
    for &i in idx {
        out.extend_from_slice(&src[i * feat_dim..(i + 1) * feat_dim]);
    }
    Ok(out)
}

/// Segment softmax: normalise `scores` within each segment defined by `segment_ids`.
///
/// `scores`: `[n]`, `segment_ids`: `[n]` (each in `[0, n_segments)`).
/// Returns `[n]` with each segment summing to 1.
pub fn segment_softmax(
    scores: &[f32],
    segment_ids: &[usize],
    n_segments: usize,
) -> GnnResult<Vec<f32>> {
    if scores.len() != segment_ids.len() {
        return Err(GnnError::DimensionMismatch {
            expected: scores.len(),
            got: segment_ids.len(),
        });
    }
    for &s in segment_ids {
        if s >= n_segments {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: s,
                n_nodes: n_segments,
            });
        }
    }
    let n = scores.len();

    // Pass 1: per-segment max (numerically stable)
    let mut seg_max = vec![f32::NEG_INFINITY; n_segments];
    for i in 0..n {
        let s = segment_ids[i];
        if scores[i] > seg_max[s] {
            seg_max[s] = scores[i];
        }
    }
    // Clamp -inf to 0 for empty segments
    for m in &mut seg_max {
        if m.is_infinite() {
            *m = 0.0;
        }
    }

    // Pass 2: exp(score - max), sum per segment
    let mut exps = vec![0.0_f32; n];
    let mut seg_sum = vec![0.0_f32; n_segments];
    for i in 0..n {
        let s = segment_ids[i];
        let e = (scores[i] - seg_max[s]).exp();
        exps[i] = e;
        seg_sum[s] += e;
    }

    // Pass 3: normalise
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let s = segment_ids[i];
        if seg_sum[s] > 0.0 {
            out[i] = exps[i] / seg_sum[s];
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_add_basic() {
        // 3 items with feat_dim=2 to 2 buckets
        let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx = vec![0usize, 1, 0];
        let out = scatter_add(&src, &idx, 2, 2).expect("test invariant: value must be valid");
        // bucket 0: [1+5, 2+6] = [6, 8]
        assert!((out[0] - 6.0).abs() < 1e-6);
        assert!((out[1] - 8.0).abs() < 1e-6);
        // bucket 1: [3, 4]
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn scatter_add_empty_bucket() {
        let src = vec![1.0_f32, 2.0];
        let idx = vec![0usize];
        let out = scatter_add(&src, &idx, 3, 2).expect("test invariant: value must be valid");
        // bucket 1 and 2 are empty
        assert!((out[2]).abs() < 1e-6);
        assert!((out[4]).abs() < 1e-6);
    }

    #[test]
    fn scatter_max_selects_maximum() {
        let src = vec![1.0_f32, 5.0, 3.0, 2.0];
        let idx = vec![0usize, 0];
        let (vals, argmax) =
            scatter_max(&src, &idx, 1, 2).expect("test invariant: value must be valid");
        // max([1,5], [3,2]) = [3,5]
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 5.0).abs() < 1e-6);
        // argmax[0] = 1 (item index 1 contributed max for feature 0... actually item 1 has [3,2])
        assert!(argmax[0].is_some());
        assert!(argmax[1].is_some());
    }

    #[test]
    fn scatter_max_empty_bucket_zero() {
        let src = vec![2.0_f32];
        let idx = vec![0usize];
        let (vals, _) = scatter_max(&src, &idx, 2, 1).expect("test invariant: value must be valid");
        // bucket 1 is empty → 0
        assert!((vals[1]).abs() < 1e-6);
    }

    #[test]
    fn scatter_min_selects_minimum() {
        let src = vec![3.0_f32, 1.0, 5.0, 2.0];
        let idx = vec![0usize, 0];
        let (vals, _) = scatter_min(&src, &idx, 1, 2).expect("test invariant: value must be valid");
        // min([3,1],[5,2]) = [3,1]
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scatter_mul_product() {
        let src = vec![2.0_f32, 3.0_f32];
        let idx = vec![0usize, 0];
        let out = scatter_mul(&src, &idx, 1, 1).expect("test invariant: value must be valid");
        // 2 * 3 = 6
        assert!((out[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn scatter_mul_empty_bucket_zero() {
        let src = vec![5.0_f32];
        let idx = vec![1usize];
        let out = scatter_mul(&src, &idx, 2, 1).expect("test invariant: value must be valid");
        // bucket 0 is empty → 0
        assert!((out[0]).abs() < 1e-6);
    }

    #[test]
    fn gather_basic() {
        let src = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // 3 nodes × feat_dim=2
        let idx = vec![2usize, 0];
        let out = gather(&src, &idx, 2).expect("test invariant: value must be valid");
        // gather node 2: [50, 60], node 0: [10, 20]
        assert!((out[0] - 50.0).abs() < 1e-6);
        assert!((out[1] - 60.0).abs() < 1e-6);
        assert!((out[2] - 10.0).abs() < 1e-6);
        assert!((out[3] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn gather_out_of_range_error() {
        let src = vec![1.0_f32, 2.0]; // 2 nodes × fd=1
        let err = gather(&src, &[5], 1);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn segment_softmax_sums_to_one() {
        let scores = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let seg_ids = vec![0usize, 0, 1, 1, 1];
        let out =
            segment_softmax(&scores, &seg_ids, 2).expect("test invariant: value must be valid");
        let sum0: f32 = out[0] + out[1];
        let sum1: f32 = out[2] + out[3] + out[4];
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn segment_softmax_single_element() {
        let scores = vec![42.0_f32];
        let seg_ids = vec![0usize];
        let out =
            segment_softmax(&scores, &seg_ids, 1).expect("test invariant: value must be valid");
        // Single element segment: softmax = 1
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scatter_add_out_of_range_error() {
        let src = vec![1.0_f32];
        let idx = vec![10usize];
        let err = scatter_add(&src, &idx, 2, 1);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn scatter_add_dimension_mismatch_error() {
        let src = vec![1.0_f32, 2.0, 3.0]; // 3 elements but idx has 2 with feat_dim=2 → expected 4
        let idx = vec![0usize, 1];
        let err = scatter_add(&src, &idx, 2, 2);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }
}
