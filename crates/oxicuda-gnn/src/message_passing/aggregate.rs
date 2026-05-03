//! Message aggregation functions for graph neural networks.

use crate::error::{GnnError, GnnResult};

/// Aggregation strategy for neighbourhood messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    /// Sum all incoming messages.
    Sum,
    /// Average (mean) of all incoming messages.
    Mean,
    /// Element-wise maximum over incoming messages.
    Max,
    /// Element-wise minimum over incoming messages.
    Min,
    /// Attention-weighted aggregation (requires separate weights).
    SoftmaxWeighted,
}

/// Aggregate messages from neighbours into per-node representations.
///
/// - `messages`: flattened `[n_edges × feat_dim]` array
/// - `target_idx`: `[n_edges]`, the destination node for each edge message
/// - Returns `[n_nodes × feat_dim]`
pub fn aggregate(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
    agg_type: AggregationType,
) -> GnnResult<Vec<f32>> {
    match agg_type {
        AggregationType::Sum => aggregate_sum(messages, target_idx, n_nodes, feat_dim),
        AggregationType::Mean => aggregate_mean(messages, target_idx, n_nodes, feat_dim),
        AggregationType::Max => aggregate_max(messages, target_idx, n_nodes, feat_dim),
        AggregationType::Min => aggregate_min(messages, target_idx, n_nodes, feat_dim),
        AggregationType::SoftmaxWeighted => Err(GnnError::InvalidAggregation(
            "SoftmaxWeighted requires explicit weights; use aggregate_softmax instead",
        )),
    }
}

fn validate_messages(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<usize> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feat_dim must be > 0".to_string(),
        ));
    }
    let n_edges = target_idx.len();
    if messages.len() != n_edges * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_edges * feat_dim,
            got: messages.len(),
        });
    }
    for &idx in target_idx {
        if idx >= n_nodes {
            return Err(GnnError::NodeIndexOutOfRange { idx, n_nodes });
        }
    }
    Ok(n_edges)
}

/// Sum-aggregate: `out[i, k] = Σ_{e: target[e]=i} messages[e, k]`
pub fn aggregate_sum(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n_edges = validate_messages(messages, target_idx, n_nodes, feat_dim)?;
    let mut out = vec![0.0_f32; n_nodes * feat_dim];
    for e in 0..n_edges {
        let t = target_idx[e];
        for k in 0..feat_dim {
            out[t * feat_dim + k] += messages[e * feat_dim + k];
        }
    }
    Ok(out)
}

/// Mean-aggregate: `out[i, k] = (1/deg_in[i]) * Σ_{e: target[e]=i} messages[e, k]`
pub fn aggregate_mean(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n_edges = validate_messages(messages, target_idx, n_nodes, feat_dim)?;
    let mut out = vec![0.0_f32; n_nodes * feat_dim];
    let mut counts = vec![0usize; n_nodes];

    for e in 0..n_edges {
        let t = target_idx[e];
        counts[t] += 1;
        for k in 0..feat_dim {
            out[t * feat_dim + k] += messages[e * feat_dim + k];
        }
    }
    // Normalise by in-degree (nodes with zero in-degree stay 0)
    for i in 0..n_nodes {
        if counts[i] > 0 {
            let inv = 1.0 / counts[i] as f32;
            for k in 0..feat_dim {
                out[i * feat_dim + k] *= inv;
            }
        }
    }
    Ok(out)
}

/// Max-aggregate: `out[i, k] = max_{e: target[e]=i} messages[e, k]`
///
/// Nodes with no incoming messages are left as `f32::NEG_INFINITY` (caller
/// should handle isolated nodes downstream; for GNN training nodes always
/// receive at least their self-feature).
pub fn aggregate_max(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n_edges = validate_messages(messages, target_idx, n_nodes, feat_dim)?;
    let mut out = vec![f32::NEG_INFINITY; n_nodes * feat_dim];
    let mut has_msg = vec![false; n_nodes];

    for e in 0..n_edges {
        let t = target_idx[e];
        has_msg[t] = true;
        for k in 0..feat_dim {
            let v = messages[e * feat_dim + k];
            if v > out[t * feat_dim + k] {
                out[t * feat_dim + k] = v;
            }
        }
    }
    // Nodes with no messages → 0
    for i in 0..n_nodes {
        if !has_msg[i] {
            for k in 0..feat_dim {
                out[i * feat_dim + k] = 0.0;
            }
        }
    }
    Ok(out)
}

/// Min-aggregate: `out[i, k] = min_{e: target[e]=i} messages[e, k]`
pub fn aggregate_min(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n_edges = validate_messages(messages, target_idx, n_nodes, feat_dim)?;
    let mut out = vec![f32::INFINITY; n_nodes * feat_dim];
    let mut has_msg = vec![false; n_nodes];

    for e in 0..n_edges {
        let t = target_idx[e];
        has_msg[t] = true;
        for k in 0..feat_dim {
            let v = messages[e * feat_dim + k];
            if v < out[t * feat_dim + k] {
                out[t * feat_dim + k] = v;
            }
        }
    }
    for i in 0..n_nodes {
        if !has_msg[i] {
            for k in 0..feat_dim {
                out[i * feat_dim + k] = 0.0;
            }
        }
    }
    Ok(out)
}

/// Attention-weighted aggregation (used in GAT).
///
/// `out[i, k] = Σ_{e: target[e]=i} weights[e] * messages[e, k]`
///
/// The weights are assumed to already be normalised (e.g. by softmax per source node).
pub fn aggregate_softmax(
    messages: &[f32],
    weights: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    let n_edges = validate_messages(messages, target_idx, n_nodes, feat_dim)?;
    if weights.len() != n_edges {
        return Err(GnnError::DimensionMismatch {
            expected: n_edges,
            got: weights.len(),
        });
    }
    let mut out = vec![0.0_f32; n_nodes * feat_dim];
    for e in 0..n_edges {
        let t = target_idx[e];
        let w = weights[e];
        for k in 0..feat_dim {
            out[t * feat_dim + k] += w * messages[e * feat_dim + k];
        }
    }
    Ok(out)
}

/// Degree-normalised aggregation.
///
/// Same as mean but uses the out-degree of the source (from `target_idx`) rather
/// than in-degree of the destination, i.e. `out[i] = sum / degree_in[i]`.
/// This is equivalent to `aggregate_mean` when all edges point to the target.
pub fn aggregate_degree_norm(
    messages: &[f32],
    target_idx: &[usize],
    n_nodes: usize,
    feat_dim: usize,
) -> GnnResult<Vec<f32>> {
    // Equivalent to mean aggregate; delegates to it.
    aggregate_mean(messages, target_idx, n_nodes, feat_dim)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 3 edges from 3 messages going to nodes: edge 0→node 0, edge 1→node 1, edge 2→node 0
    fn small_setup() -> (Vec<f32>, Vec<usize>, usize, usize) {
        // feat_dim = 2
        // msg 0 = [1, 2] → node 0
        // msg 1 = [3, 4] → node 1
        // msg 2 = [5, 6] → node 0
        let messages = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let target_idx = vec![0, 1, 0];
        (messages, target_idx, 2, 2) // n_nodes=2, feat_dim=2
    }

    #[test]
    fn sum_aggregate_correct() {
        let (msg, idx, n, d) = small_setup();
        let out = aggregate_sum(&msg, &idx, n, d).expect("test invariant: value must be valid");
        // node 0: [1+5, 2+6] = [6, 8]
        assert!((out[0] - 6.0).abs() < 1e-6);
        assert!((out[1] - 8.0).abs() < 1e-6);
        // node 1: [3, 4]
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn mean_aggregate_correct() {
        let (msg, idx, n, d) = small_setup();
        let out = aggregate_mean(&msg, &idx, n, d).expect("test invariant: value must be valid");
        // node 0: [6/2, 8/2] = [3, 4]
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        // node 1: [3/1, 4/1] = [3, 4]
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn max_aggregate_correct() {
        let (msg, idx, n, d) = small_setup();
        let out = aggregate_max(&msg, &idx, n, d).expect("test invariant: value must be valid");
        // node 0: max([1,2],[5,6]) = [5,6]
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn min_aggregate_correct() {
        let (msg, idx, n, d) = small_setup();
        let out = aggregate_min(&msg, &idx, n, d).expect("test invariant: value must be valid");
        // node 0: min([1,2],[5,6]) = [1,2]
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn isolated_node_produces_zero_sum() {
        let messages = vec![1.0_f32, 2.0];
        let target_idx = vec![0usize]; // only node 0 gets a message
        let out = aggregate_sum(&messages, &target_idx, 3, 2)
            .expect("test invariant: value must be valid");
        // node 2 is isolated
        assert!((out[4]).abs() < 1e-6);
        assert!((out[5]).abs() < 1e-6);
    }

    #[test]
    fn isolated_node_produces_zero_max() {
        let messages = vec![1.0_f32, 2.0];
        let target_idx = vec![1usize];
        let out = aggregate_max(&messages, &target_idx, 3, 2)
            .expect("test invariant: value must be valid");
        // node 0 and 2 are isolated → 0
        assert!((out[0]).abs() < 1e-6);
        assert!((out[4]).abs() < 1e-6);
    }

    #[test]
    fn softmax_aggregate_weighted() {
        // 2 edges to node 0 with weights 0.3 and 0.7
        let messages = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weights = vec![0.3_f32, 0.7];
        let target_idx = vec![0, 0];
        let out = aggregate_softmax(&messages, &weights, &target_idx, 1, 2)
            .expect("test invariant: value must be valid");
        // [0.3*1+0.7*3, 0.3*2+0.7*4] = [2.4, 3.4]
        assert!((out[0] - 2.4).abs() < 1e-5);
        assert!((out[1] - 3.4).abs() < 1e-5);
    }

    #[test]
    fn aggregate_dispatch_sum() {
        let (msg, idx, n, d) = small_setup();
        let out = aggregate(&msg, &idx, n, d, AggregationType::Sum)
            .expect("test invariant: value must be valid");
        assert!((out[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn aggregate_dispatch_softmax_weighted_error() {
        let (msg, idx, n, d) = small_setup();
        let err = aggregate(&msg, &idx, n, d, AggregationType::SoftmaxWeighted);
        assert!(err.is_err());
    }

    #[test]
    fn dimension_mismatch_error() {
        // wrong message length
        let err = aggregate_sum(&[1.0_f32, 2.0], &[0, 1], 2, 2);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn degree_norm_equals_mean() {
        let (msg, idx, n, d) = small_setup();
        let mean_out =
            aggregate_mean(&msg, &idx, n, d).expect("test invariant: value must be valid");
        let deg_out =
            aggregate_degree_norm(&msg, &idx, n, d).expect("test invariant: value must be valid");
        for (a, b) in mean_out.iter().zip(deg_out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn out_of_range_target_error() {
        let messages = vec![1.0_f32, 2.0];
        let target_idx = vec![10usize]; // out of range
        let err = aggregate_sum(&messages, &target_idx, 3, 2);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }
}
