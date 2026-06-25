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

// ─── Edge-conditioned message construction ────────────────────────────────────

/// Build per-edge messages from source-node features and **optional** per-edge
/// features — the message-passing `message(x_src, edge_attr)` step.
///
/// This threads edge features through the message function so that a message
/// can depend on the edge attribute, exactly as in the general
/// message-passing / MPNN framework (Gilmer et al. 2017):
///
/// ```text
/// m_e = W_x · x_{src(e)}                    (when edge_features is None)
/// m_e = W_x · x_{src(e)} + W_e · edge_e     (when edge_features is Some)
/// ```
///
/// The resulting `[n_edges × out_dim]` messages can then be fed to any of the
/// `aggregate_*` / `scatter_*` reducers keyed by the edge's destination.
///
/// # Arguments
///
/// - `node_features`: `[n_nodes × node_dim]` row-major.
/// - `src_idx`: `[n_edges]` — the source node of each edge.
/// - `node_weight`: `[out_dim × node_dim]` — `W_x` source-feature projection.
/// - `edge`: when `Some((edge_features, edge_weight))`, `edge_features` is
///   `[n_edges × edge_dim]` and `edge_weight` is `[out_dim × edge_dim]` (`W_e`);
///   `edge_dim` is inferred from `edge_weight.len() / out_dim`. When `None`, the
///   message depends only on the source node (backward-compatible path).
/// - `node_dim`, `out_dim`: source-feature and message dimensions.
///
/// # Returns
///
/// `[n_edges × out_dim]` row-major messages.
pub fn build_edge_messages(
    node_features: &[f32],
    src_idx: &[usize],
    node_weight: &[f32],
    edge: Option<(&[f32], &[f32])>,
    node_dim: usize,
    out_dim: usize,
) -> GnnResult<Vec<f32>> {
    if node_dim == 0 || out_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "node_dim and out_dim must be > 0".to_string(),
        ));
    }
    if node_weight.len() != out_dim * node_dim {
        return Err(GnnError::WeightShapeMismatch {
            r: out_dim,
            c: node_dim,
            d: node_dim,
        });
    }
    let n_edges = src_idx.len();
    let n_nodes = node_features.len() / node_dim;
    if node_features.len() != n_nodes * node_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_nodes * node_dim,
            got: node_features.len(),
        });
    }
    for &s in src_idx {
        if s >= n_nodes {
            return Err(GnnError::NodeIndexOutOfRange { idx: s, n_nodes });
        }
    }

    // Validate and infer the edge-feature shape when present.
    let edge_dim = match edge {
        None => 0,
        Some((edge_features, edge_weight)) => {
            if edge_weight.len() % out_dim != 0 {
                return Err(GnnError::WeightShapeMismatch {
                    r: out_dim,
                    c: 0,
                    d: edge_weight.len(),
                });
            }
            let ed = edge_weight.len() / out_dim;
            if ed == 0 {
                return Err(GnnError::InvalidLayerConfig(
                    "edge_dim must be > 0 when edge features are supplied".to_string(),
                ));
            }
            if edge_features.len() != n_edges * ed {
                return Err(GnnError::EdgeFeatureMismatch(
                    n_edges,
                    edge_features.len() / ed.max(1),
                ));
            }
            ed
        }
    };

    let mut messages = vec![0.0_f32; n_edges * out_dim];
    for e in 0..n_edges {
        let s = src_idx[e];
        let x_off = s * node_dim;
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            // W_x · x_src
            for j in 0..node_dim {
                acc += node_weight[o * node_dim + j] * node_features[x_off + j];
            }
            messages[e * out_dim + o] = acc;
        }
        // + W_e · edge  (only when edge features were supplied)
        if let Some((edge_features, edge_weight)) = edge {
            let ef_off = e * edge_dim;
            for o in 0..out_dim {
                let mut acc = 0.0_f32;
                for d in 0..edge_dim {
                    acc += edge_weight[o * edge_dim + d] * edge_features[ef_off + d];
                }
                messages[e * out_dim + o] += acc;
            }
        }
    }
    Ok(messages)
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

    // ── Edge-conditioned message construction ─────────────────────────────────

    #[test]
    fn build_messages_no_edge_features() {
        // 2 nodes × node_dim=2; identity projection ⇒ message == source feature.
        let nf = vec![1.0_f32, 2.0, 3.0, 4.0];
        let src = vec![0usize, 1, 0];
        let wx = vec![1.0_f32, 0.0, 0.0, 1.0]; // I_2
        let msgs = build_edge_messages(&nf, &src, &wx, None, 2, 2)
            .expect("test invariant: value must be valid");
        // edge 0 from node0 = [1,2], edge1 from node1 = [3,4], edge2 from node0 = [1,2]
        assert_eq!(msgs, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn build_messages_edge_features_add_in() {
        // Source projection identity, edge projection identity ⇒ m = x_src + edge.
        let nf = vec![1.0_f32, 1.0, 2.0, 2.0]; // 2 nodes × 2
        let src = vec![0usize, 1];
        let wx = vec![1.0_f32, 0.0, 0.0, 1.0];
        let ef = vec![10.0_f32, 20.0, 30.0, 40.0]; // 2 edges × edge_dim=2
        let we = vec![1.0_f32, 0.0, 0.0, 1.0];
        let msgs = build_edge_messages(&nf, &src, &wx, Some((&ef, &we)), 2, 2)
            .expect("test invariant: value must be valid");
        // edge0: [1,1]+[10,20]=[11,21]; edge1: [2,2]+[30,40]=[32,42]
        assert!((msgs[0] - 11.0).abs() < 1e-6);
        assert!((msgs[1] - 21.0).abs() < 1e-6);
        assert!((msgs[2] - 32.0).abs() < 1e-6);
        assert!((msgs[3] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn build_messages_edge_features_change_aggregate() {
        // The same edge can carry different attributes; the aggregated result
        // must change when edge features are threaded through the message.
        let nf = vec![1.0_f32, 0.5]; // 1 node × node_dim=2
        let src = vec![0usize, 0]; // two parallel edges from node 0
        let dst = vec![0usize, 0];
        let wx = vec![1.0_f32, 1.0]; // out_dim=1: sum of node features
        let no_edge = build_edge_messages(&nf, &src, &wx, None, 2, 1)
            .expect("test invariant: value must be valid");
        let agg_no_edge =
            aggregate_sum(&no_edge, &dst, 1, 1).expect("test invariant: value must be valid");

        let ef = vec![2.0_f32, -3.0]; // 2 edges × edge_dim=1
        let we = vec![1.0_f32]; // out_dim=1
        let with_edge = build_edge_messages(&nf, &src, &wx, Some((&ef, &we)), 2, 1)
            .expect("test invariant: value must be valid");
        let agg_with_edge =
            aggregate_sum(&with_edge, &dst, 1, 1).expect("test invariant: value must be valid");

        // node-only message per edge = 1.0+0.5 = 1.5 ⇒ sum over 2 edges = 3.0
        assert!((agg_no_edge[0] - 3.0).abs() < 1e-6);
        // with edges: (1.5+2.0)+(1.5-3.0) = 3.5 + (-1.5) = 2.0
        assert!((agg_with_edge[0] - 2.0).abs() < 1e-6);
        assert!((agg_no_edge[0] - agg_with_edge[0]).abs() > 1e-4);
    }

    #[test]
    fn build_messages_projection_shape() {
        // out_dim != node_dim path.
        let nf = vec![1.0_f32, 2.0, 3.0]; // 1 node × node_dim=3
        let src = vec![0usize];
        let wx = vec![1.0_f32, 1.0, 1.0, 2.0, 0.0, 0.0]; // [out_dim=2 × node_dim=3]
        let msgs = build_edge_messages(&nf, &src, &wx, None, 3, 2)
            .expect("test invariant: value must be valid");
        // out[0] = 1+2+3 = 6; out[1] = 2*1 = 2
        assert_eq!(msgs.len(), 2);
        assert!((msgs[0] - 6.0).abs() < 1e-6);
        assert!((msgs[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn build_messages_edge_mismatch_error() {
        let nf = vec![1.0_f32, 2.0]; // 1 node × 2
        let src = vec![0usize, 0]; // 2 edges
        let wx = vec![1.0_f32, 0.0, 0.0, 1.0];
        let we = vec![1.0_f32, 0.0, 0.0, 1.0]; // out_dim=2, edge_dim=2
        let bad_ef = vec![1.0_f32; 3 * 2]; // 3 edge rows but only 2 edges
        let err = build_edge_messages(&nf, &src, &wx, Some((&bad_ef, &we)), 2, 2);
        assert!(matches!(err, Err(GnnError::EdgeFeatureMismatch(..))));
    }

    #[test]
    fn build_messages_node_weight_shape_error() {
        let nf = vec![1.0_f32, 2.0];
        let src = vec![0usize];
        let bad_wx = vec![1.0_f32, 0.0, 0.0]; // not out_dim*node_dim
        let err = build_edge_messages(&nf, &src, &bad_wx, None, 2, 2);
        assert!(matches!(err, Err(GnnError::WeightShapeMismatch { .. })));
    }
}
