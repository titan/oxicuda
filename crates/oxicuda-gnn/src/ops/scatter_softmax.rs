//! Scatter softmax and scatter aggregation operations for graph attention networks.
//!
//! Computes per-node (per-destination) softmax over incoming edge attention
//! scores, as used in GAT / GATv2 / Transformer-based GNN layers.

use crate::error::{GnnError, GnnResult};

// ─── Scatter softmax ──────────────────────────────────────────────────────────

/// Compute per-destination-node softmax over raw edge attention scores.
///
/// # Arguments
///
/// - `scores`    : `[n_edges]` raw (unnormalized) attention logits.
/// - `dst_nodes` : `[n_edges]` destination node for each edge.
/// - `n_nodes`   : total number of nodes in the graph.
///
/// # Returns
///
/// `[n_edges]` softmax-normalized weights, one distribution per destination.
///
/// # Algorithm
///
/// For each destination node `d`:
///   1. Collect all edges `e` where `dst_nodes[e] == d`.
///   2. Compute `max_score = max(scores[e])` (for numerical stability).
///   3. `exp_val[e] = exp(scores[e] - max_score)`.
///   4. `sum_exp = sum(exp_val[e])`.
///   5. `softmax[e] = exp_val[e] / sum_exp`.
///
/// # Errors
///
/// - [`GnnError::DimensionMismatch`] if `scores.len() != dst_nodes.len()`.
/// - [`GnnError::NodeIndexOutOfRange`] if any `dst_nodes[e] >= n_nodes`.
pub fn scatter_softmax(scores: &[f32], dst_nodes: &[usize], n_nodes: usize) -> GnnResult<Vec<f32>> {
    let n_edges = scores.len();
    if n_edges != dst_nodes.len() {
        return Err(GnnError::DimensionMismatch {
            expected: n_edges,
            got: dst_nodes.len(),
        });
    }
    for &d in dst_nodes {
        if d >= n_nodes {
            return Err(GnnError::NodeIndexOutOfRange { idx: d, n_nodes });
        }
    }

    if n_edges == 0 {
        return Ok(vec![]);
    }

    // Pass 1: find max score per destination node (for numerical stability).
    let mut max_per_dst = vec![f32::NEG_INFINITY; n_nodes];
    for (e, &d) in dst_nodes.iter().enumerate() {
        if scores[e] > max_per_dst[d] {
            max_per_dst[d] = scores[e];
        }
    }
    // Clamp -Inf (nodes with no incoming edges) to 0 to avoid NaN in exp.
    for m in &mut max_per_dst {
        if m.is_infinite() {
            *m = 0.0;
        }
    }

    // Pass 2: compute exp(score - max) per edge.
    let mut exp_scores: Vec<f32> = Vec::with_capacity(n_edges);
    for (e, &d) in dst_nodes.iter().enumerate() {
        exp_scores.push((scores[e] - max_per_dst[d]).exp());
    }

    // Pass 3: sum exp-values per destination node.
    let mut sum_exp = vec![0.0_f32; n_nodes];
    for (e, &d) in dst_nodes.iter().enumerate() {
        sum_exp[d] += exp_scores[e];
    }

    // Pass 4: normalize.
    let mut out = vec![0.0_f32; n_edges];
    for (e, &d) in dst_nodes.iter().enumerate() {
        let s = sum_exp[d];
        out[e] = if s > 0.0 { exp_scores[e] / s } else { 0.0 };
    }

    Ok(out)
}

// ─── Scatter-add ─────────────────────────────────────────────────────────────

/// Scatter-add: aggregate per-edge values into destination nodes by sum.
///
/// # Arguments
///
/// - `src_vals`  : `[n_edges × d]` per-edge feature vectors (row-major).
/// - `dst_nodes` : `[n_edges]` destination node index for each edge.
/// - `n_nodes`   : total number of destination nodes.
/// - `d`         : feature dimension.
///
/// # Returns
///
/// `[n_nodes × d]` initialized to zero; `out[dst_nodes[e], k] += src_vals[e, k]`.
///
/// # Errors
///
/// - [`GnnError::InvalidLayerConfig`] if `d == 0`.
/// - [`GnnError::DimensionMismatch`] if `src_vals.len() != n_edges * d`.
/// - [`GnnError::NodeIndexOutOfRange`] if any `dst_nodes[e] >= n_nodes`.
pub fn scatter_add(
    src_vals: &[f32],
    dst_nodes: &[usize],
    n_nodes: usize,
    d: usize,
) -> GnnResult<Vec<f32>> {
    if d == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feature dimension d must be > 0".to_string(),
        ));
    }
    let n_edges = dst_nodes.len();
    if src_vals.len() != n_edges * d {
        return Err(GnnError::DimensionMismatch {
            expected: n_edges * d,
            got: src_vals.len(),
        });
    }
    for &dst in dst_nodes {
        if dst >= n_nodes {
            return Err(GnnError::NodeIndexOutOfRange { idx: dst, n_nodes });
        }
    }

    let mut out = vec![0.0_f32; n_nodes * d];
    for (e, &dst) in dst_nodes.iter().enumerate() {
        for k in 0..d {
            out[dst * d + k] += src_vals[e * d + k];
        }
    }
    Ok(out)
}

// ─── Scatter-mean ────────────────────────────────────────────────────────────

/// Scatter-mean: aggregate per-edge values into destination nodes by average.
///
/// Same as [`scatter_add`] but divides each row by the number of incoming edges.
///
/// # Returns
///
/// `[n_nodes × d]`; nodes with no incoming edges remain `0.0`.
///
/// # Errors
///
/// Same as [`scatter_add`].
pub fn scatter_mean(
    src_vals: &[f32],
    dst_nodes: &[usize],
    n_nodes: usize,
    d: usize,
) -> GnnResult<Vec<f32>> {
    let mut out = scatter_add(src_vals, dst_nodes, n_nodes, d)?;

    // Count incoming edges per node.
    let mut counts = vec![0_u32; n_nodes];
    for &dst in dst_nodes {
        counts[dst] += 1;
    }

    // Divide by count (skip zero-count nodes — they stay 0).
    for (node, &cnt) in counts.iter().enumerate() {
        if cnt > 0 {
            let inv = 1.0 / cnt as f32;
            for k in 0..d {
                out[node * d + k] *= inv;
            }
        }
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // helper: build edge list for a trivial graph
    // edges: 0→2, 1→2, 2→3, 3→3
    fn simple_edges() -> (Vec<f32>, Vec<usize>) {
        let scores = vec![1.0_f32, 2.0, 0.5, -0.5];
        let dst = vec![2_usize, 2, 3, 3];
        (scores, dst)
    }

    // 1 ─ softmax sums to 1 per destination node
    #[test]
    fn softmax_sums_to_1_per_node() {
        let (scores, dst) = simple_edges();
        let sm = scatter_softmax(&scores, &dst, 4).expect("scatter_softmax should succeed");
        // node 2: edges 0,1
        let s2 = sm[0] + sm[1];
        assert!((s2 - 1.0).abs() < 1e-6, "node2 sum={s2}");
        // node 3: edges 2,3
        let s3 = sm[2] + sm[3];
        assert!((s3 - 1.0).abs() < 1e-6, "node3 sum={s3}");
    }

    // 2 ─ softmax values are non-negative
    #[test]
    fn softmax_nonneg() {
        let (scores, dst) = simple_edges();
        let sm = scatter_softmax(&scores, &dst, 4).expect("scatter_softmax should succeed");
        for &v in &sm {
            assert!(v >= 0.0, "negative softmax: {v}");
        }
    }

    // 3 ─ softmax is monotone: higher score → higher softmax within group
    #[test]
    fn softmax_monotone() {
        // scores for node 2: edge0=1.0 < edge1=2.0 → sm[0] < sm[1]
        let (scores, dst) = simple_edges();
        let sm = scatter_softmax(&scores, &dst, 4).expect("scatter_softmax should succeed");
        assert!(
            sm[0] < sm[1],
            "lower score should have lower softmax: sm[0]={}, sm[1]={}",
            sm[0],
            sm[1]
        );
    }

    // 4 ─ scatter_add shape
    #[test]
    fn scatter_add_shape() {
        let d = 3_usize;
        let n_nodes = 5_usize;
        let src = vec![1.0_f32; 4 * d];
        let dst = vec![0_usize, 1, 2, 3];
        let out = scatter_add(&src, &dst, n_nodes, d).expect("scatter_add should succeed");
        assert_eq!(out.len(), n_nodes * d);
    }

    // 5 ─ scatter_add sum correct
    #[test]
    fn scatter_add_sum_correct() {
        // Two edges both going to node 1, feature dim=2
        let src = vec![1.0_f32, 2.0, 3.0, 4.0]; // edge0=[1,2], edge1=[3,4]
        let dst = vec![1_usize, 1];
        let out = scatter_add(&src, &dst, 3, 2).expect("scatter_add should succeed");
        assert!((out[2] - 4.0).abs() < 1e-6, "out[1,0]={}", out[2]); // node1, feat0
        assert!((out[3] - 6.0).abs() < 1e-6, "out[1,1]={}", out[3]); // node1, feat1
    }

    // 6 ─ scatter_mean shape
    #[test]
    fn scatter_mean_shape() {
        let d = 4_usize;
        let n_nodes = 6_usize;
        let src = vec![0.5_f32; 3 * d];
        let dst = vec![0_usize, 2, 4];
        let out = scatter_mean(&src, &dst, n_nodes, d).expect("scatter_mean should succeed");
        assert_eq!(out.len(), n_nodes * d);
    }

    // 7 ─ scatter_mean average correct
    #[test]
    fn scatter_mean_average_correct() {
        // 3 edges to node 0, values: 3.0, 6.0, 9.0 → mean = 6.0; feat dim=1
        let src = vec![3.0_f32, 6.0, 9.0];
        let dst = vec![0_usize, 0, 0];
        let out = scatter_mean(&src, &dst, 2, 1).expect("scatter_mean should succeed");
        assert!((out[0] - 6.0).abs() < 1e-6, "mean={}", out[0]);
    }

    // 8 ─ single edge per node → softmax = 1.0
    #[test]
    fn single_edge_per_node_softmax_is_1() {
        let scores = vec![3.5_f32, -1.2, 0.0];
        let dst = vec![0_usize, 1, 2];
        let sm = scatter_softmax(&scores, &dst, 3).expect("scatter_softmax should succeed");
        for (i, &v) in sm.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "edge {i} sm={v}");
        }
    }

    // 9 ─ no edges returns empty vec
    #[test]
    fn no_edges_returns_empty() {
        let sm = scatter_softmax(&[], &[], 5).expect("scatter_softmax should succeed");
        assert!(sm.is_empty());
    }

    // 10 ─ dst out of range returns error
    #[test]
    fn dst_out_of_range_error() {
        let scores = vec![1.0_f32];
        let dst = vec![5_usize]; // n_nodes = 3
        let result = scatter_softmax(&scores, &dst, 3);
        assert!(result.is_err());
    }

    // 11 ─ scatter_add with multiple edges per node
    #[test]
    fn scatter_add_multiple_edges() {
        let src = vec![1.0_f32, 2.0, 3.0]; // 3 edges, d=1
        let dst = vec![0_usize, 0, 0]; // all to node 0
        let out = scatter_add(&src, &dst, 2, 1).expect("scatter_add should succeed");
        assert!((out[0] - 6.0).abs() < 1e-6, "sum={}", out[0]);
        assert!((out[1]).abs() < 1e-6, "node1 should be 0");
    }

    // 12 ─ scatter_mean with isolated node stays zero
    #[test]
    fn scatter_mean_isolated_node_zero() {
        let src = vec![5.0_f32];
        let dst = vec![0_usize];
        let out = scatter_mean(&src, &dst, 3, 1).expect("scatter_mean should succeed");
        assert!((out[1]).abs() < 1e-6, "node1 isolated, should be 0");
        assert!((out[2]).abs() < 1e-6, "node2 isolated, should be 0");
    }
}
