//! SGC — Simple Graph Convolution.
//!
//! Wu, Souza, Zhang, Fifty, Yu & Weinberger "Simplifying Graph Convolutional Networks",
//! ICML 2019.
//!
//! Removes all nonlinearities between GCN propagation steps and reduces the model to a
//! single linear classifier applied to the K-hop aggregated feature matrix S^K X:
//!
//! ```text
//! Y = softmax(S^K X W)
//! ```
//!
//! where `S = D̃^{−1/2} Ã D̃^{−1/2}` is the symmetric normalised adjacency with
//! self-loops.  The K-hop precomputation is done once offline; this module provides:
//!
//! * [`sgc_propagate`]  — compute S^K X
//! * [`sgc_linear`]     — apply a linear classifier to propagated features
//! * [`sgc_forward`]    — combined propagate + classify

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── K-hop propagation ────────────────────────────────────────────────────────

/// Precompute K-hop normalised graph propagation: S^K X.
///
/// `S = D̃^{−1/2} Ã D̃^{−1/2}` (normalised adjacency with self-loops).
///
/// Applies K consecutive sparse matrix multiplications:
/// `X_1 = S X`, `X_2 = S X_1`, …, `X_K = S X_{K−1}`.
///
/// This is typically computed once **offline** before training.
///
/// # Arguments
///
/// * `graph`    — CSR graph.
/// * `x`        — `[n_nodes × feat_dim]` input node features.
/// * `feat_dim` — feature dimension (must match `x.len() / n_nodes`).
/// * `k`        — number of propagation steps (≥ 1).
///
/// # Returns
///
/// `[n_nodes × feat_dim]` K-hop propagated features.
///
/// # Errors
///
/// * [`GnnError::InvalidLayerConfig`]  if `feat_dim == 0` or `k == 0`.
/// * [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * feat_dim`.
/// * [`GnnError::NonFiniteOutput`]     if any output value is NaN or infinite.
pub fn sgc_propagate(
    graph: &CsrGraph,
    x: &[f32],
    feat_dim: usize,
    k: usize,
) -> GnnResult<Vec<f32>> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "SGC: feat_dim must be > 0".to_string(),
        ));
    }
    if k == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "SGC: k must be >= 1".to_string(),
        ));
    }
    let n = graph.n_nodes();
    if x.len() != n * feat_dim {
        return Err(GnnError::NodeFeatureMismatch(n, x.len() / feat_dim));
    }

    // Pre-compute the normalised adjacency in COO form once.
    let (rows, cols, vals) = graph.normalized_adjacency();

    let mut x_cur = x.to_vec();

    for _step in 0..k {
        let mut x_next = vec![0.0_f32; n * feat_dim];
        for idx in 0..rows.len() {
            let i = rows[idx];
            let j = cols[idx];
            let v = vals[idx];
            for d in 0..feat_dim {
                x_next[i * feat_dim + d] += v * x_cur[j * feat_dim + d];
            }
        }
        x_cur = x_next;
    }

    if x_cur.iter().any(|v| !v.is_finite()) {
        return Err(GnnError::NonFiniteOutput("sgc_propagate"));
    }

    Ok(x_cur)
}

// ─── Linear classifier ────────────────────────────────────────────────────────

/// Apply a linear classifier to propagated features.
///
/// Computes `logits[i, c] = Σ_j x_prop[i, j] · weight[j, c]  (+  bias[c])`.
///
/// # Arguments
///
/// * `x_prop`  — `[n_nodes × in_dim]` propagated features.
/// * `weight`  — `[in_dim × out_dim]` weight matrix (row-major).
/// * `bias`    — optional `[out_dim]` bias vector.
/// * `n_nodes` — number of nodes.
/// * `in_dim`  — input feature dimension.
/// * `out_dim` — number of output classes.
///
/// # Returns
///
/// `[n_nodes × out_dim]` raw logits.
///
/// # Errors
///
/// * [`GnnError::DimensionMismatch`]   if `x_prop.len() != n_nodes * in_dim`.
/// * [`GnnError::WeightShapeMismatch`] if `weight.len() != in_dim * out_dim`.
/// * [`GnnError::DimensionMismatch`]   if `bias.len() != out_dim`.
pub fn sgc_linear(
    x_prop: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n_nodes: usize,
    in_dim: usize,
    out_dim: usize,
) -> GnnResult<Vec<f32>> {
    if x_prop.len() != n_nodes * in_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_nodes * in_dim,
            got: x_prop.len(),
        });
    }
    if weight.len() != in_dim * out_dim {
        return Err(GnnError::WeightShapeMismatch {
            r: in_dim,
            c: out_dim,
            d: in_dim,
        });
    }
    if let Some(b) = bias {
        if b.len() != out_dim {
            return Err(GnnError::DimensionMismatch {
                expected: out_dim,
                got: b.len(),
            });
        }
    }

    let mut logits = vec![0.0_f32; n_nodes * out_dim];

    // logits[i, c] = Σ_j x_prop[i, j] * weight[j, c]
    for i in 0..n_nodes {
        for c in 0..out_dim {
            let mut acc = 0.0_f32;
            for j in 0..in_dim {
                acc += x_prop[i * in_dim + j] * weight[j * out_dim + c];
            }
            logits[i * out_dim + c] = acc;
        }
    }

    // Add bias if provided.
    if let Some(b) = bias {
        for i in 0..n_nodes {
            for c in 0..out_dim {
                logits[i * out_dim + c] += b[c];
            }
        }
    }

    Ok(logits)
}

// ─── Full forward pass ────────────────────────────────────────────────────────

/// Full SGC forward pass: K-hop propagation followed by linear classification.
///
/// Equivalent to:
/// ```rust,ignore
/// sgc_linear(sgc_propagate(graph, x, feat_dim, k)?, weight, bias, n_nodes, feat_dim, n_classes)
/// ```
///
/// # Arguments
///
/// * `graph`     — CSR graph.
/// * `x`         — `[n_nodes × feat_dim]` input node features.
/// * `feat_dim`  — feature dimension.
/// * `k`         — number of propagation hops.
/// * `weight`    — `[feat_dim × n_classes]` weight matrix (row-major).
/// * `bias`      — optional `[n_classes]` bias.
/// * `n_classes` — number of output classes.
///
/// # Returns
///
/// `[n_nodes × n_classes]` raw logits.
pub fn sgc_forward(
    graph: &CsrGraph,
    x: &[f32],
    feat_dim: usize,
    k: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    n_classes: usize,
) -> GnnResult<Vec<f32>> {
    let n = graph.n_nodes();
    let x_prop = sgc_propagate(graph, x, feat_dim, k)?;
    sgc_linear(&x_prop, weight, bias, n, feat_dim, n_classes)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph() -> CsrGraph {
        CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid")
    }

    fn triangle_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
            .expect("test invariant: value must be valid")
    }

    fn identity_weight(d: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    // ── Propagation ───────────────────────────────────────────────────────────

    #[test]
    fn propagate_output_shape() {
        let g = line_graph();
        let x = vec![1.0_f32; 4 * 3];
        let out = sgc_propagate(&g, &x, 3, 2).expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn propagate_k1_applies_once() {
        // For a 2-node bidirected graph with feat_dim=1, verify k=1 output
        // matches the manual COO sparse matmul.
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let x = vec![4.0_f32, 2.0_f32];
        let out = sgc_propagate(&g, &x, 1, 1).expect("test invariant: value must be valid");
        // out_deg[0]=1, so d_inv_sqrt[0]=1/sqrt(2).
        // Self-loop val = (1/sqrt(2))^2 = 0.5.
        // Off-diag val  = (1/sqrt(2))*1*(1/sqrt(2)) = 0.5.
        // node 0: 0.5*x[0] + 0.5*x[1] = 0.5*4 + 0.5*2 = 3.0
        // node 1: 0.5*x[1] + 0.5*x[0] = 3.0
        assert!((out[0] - 3.0_f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 3.0_f32).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn propagate_k0_invalid() {
        let g = line_graph();
        let x = vec![1.0_f32; 4];
        assert!(sgc_propagate(&g, &x, 1, 0).is_err());
    }

    #[test]
    fn propagate_feat_dim_zero_invalid() {
        let g = line_graph();
        let x: Vec<f32> = vec![];
        assert!(sgc_propagate(&g, &x, 0, 1).is_err());
    }

    #[test]
    fn propagate_node_feature_mismatch() {
        let g = line_graph(); // 4 nodes
        let x = vec![1.0_f32; 3 * 2]; // only 3 nodes' worth
        assert!(matches!(
            sgc_propagate(&g, &x, 2, 1),
            Err(GnnError::NodeFeatureMismatch(..))
        ));
    }

    #[test]
    fn propagate_k2_differs_from_k1() {
        let g = triangle_graph();
        let x: Vec<f32> = (0..3 * 2).map(|i| i as f32).collect();
        let out1 = sgc_propagate(&g, &x, 2, 1).expect("test invariant: value must be valid");
        let out2 = sgc_propagate(&g, &x, 2, 2).expect("test invariant: value must be valid");
        let diff: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "k=1 and k=2 should differ, diff={diff}");
    }

    #[test]
    fn propagate_isolate_node_has_self_loop() {
        // Isolated node (degree 0) gets a self-loop with normalized weight 1.0
        // because out_deg=0 → d_plus_1=1 → d_inv_sqrt=1 → self-loop val=1.
        let g = CsrGraph::from_edges(3, &[(1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let x = vec![5.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0]; // node 0 feat=[5,0]
        let out = sgc_propagate(&g, &x, 2, 1).expect("test invariant: value must be valid");
        // Node 0 is isolated (deg=0); normalized self-loop weight = 1/(0+1)=1.
        assert!(
            (out[0] - 5.0_f32).abs() < 1e-5,
            "isolated node preserved, out[0]={}",
            out[0]
        );
        assert!((out[1] - 0.0_f32).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn propagate_single_node() {
        // n=1, self-loop: d_inv_sqrt=1/sqrt(1+0)=1 (no outgoing edges) — wait:
        // out_deg[0]=1 (self-loop is an edge), d_plus_1=2 → d_inv_sqrt=1/sqrt(2).
        // Actually from_edges with (0,0): out_deg=1, d_plus_1=2, val=0.5.
        // With val=0.5 per step: x_k = 0.5^k * x_0.
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let x = vec![8.0_f32, 4.0_f32];
        let out = sgc_propagate(&g, &x, 2, 3).expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Linear classifier ─────────────────────────────────────────────────────

    #[test]
    fn linear_output_shape() {
        let n = 5;
        let in_dim = 3;
        let out_dim = 4;
        let x = vec![1.0_f32; n * in_dim];
        let w = vec![0.1_f32; in_dim * out_dim];
        let out = sgc_linear(&x, &w, None, n, in_dim, out_dim)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), n * out_dim);
    }

    #[test]
    fn linear_zero_weight_gives_bias() {
        let n = 3;
        let in_dim = 2;
        let out_dim = 2;
        let x = vec![1.0_f32; n * in_dim];
        let w = vec![0.0_f32; in_dim * out_dim];
        let bias = vec![3.0_f32, 7.0_f32];
        let out = sgc_linear(&x, &w, Some(&bias), n, in_dim, out_dim)
            .expect("test invariant: value must be valid");
        // All rows should equal the bias.
        for i in 0..n {
            assert!((out[i * out_dim] - 3.0_f32).abs() < 1e-6);
            assert!((out[i * out_dim + 1] - 7.0_f32).abs() < 1e-6);
        }
    }

    #[test]
    fn linear_no_bias_zero_weight_zero_output() {
        let x = vec![5.0_f32; 4 * 3];
        let w = vec![0.0_f32; 3 * 2];
        let out = sgc_linear(&x, &w, None, 4, 3, 2).expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn linear_identity_weight_no_bias() {
        // in_dim == out_dim with identity weight → output == input.
        let d = 3;
        let n = 2;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = identity_weight(d);
        let out = sgc_linear(&x, &w, None, n, d, d).expect("test invariant: value must be valid");
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-6, "o={o} xi={xi}");
        }
    }

    #[test]
    fn linear_bias_mismatch_err() {
        let x = vec![1.0_f32; 4 * 3];
        let w = vec![0.1_f32; 3 * 2];
        let bias = vec![1.0_f32; 5]; // wrong: should be 2
        assert!(matches!(
            sgc_linear(&x, &w, Some(&bias), 4, 3, 2),
            Err(GnnError::DimensionMismatch { .. })
        ));
    }

    // ── Full forward pipeline ─────────────────────────────────────────────────

    #[test]
    fn forward_full_pipeline() {
        let g = line_graph();
        let feat_dim = 3;
        let n_classes = 2;
        let x = vec![1.0_f32; 4 * feat_dim];
        let w = vec![0.1_f32; feat_dim * n_classes];
        let out = sgc_forward(&g, &x, feat_dim, 2, &w, None, n_classes)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * n_classes);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_k1_matches_manual() {
        // 2-node bidirected, feat_dim=1, k=1, n_classes=1, no bias.
        // k=1 propagate: out=[3.0, 3.0] (from propagate_k1_applies_once).
        // Linear with weight=[2.0]: logits=[6.0, 6.0].
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let x = vec![4.0_f32, 2.0_f32];
        let w = vec![2.0_f32];
        let out =
            sgc_forward(&g, &x, 1, 1, &w, None, 1).expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 6.0_f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 6.0_f32).abs() < 1e-5, "out[1]={}", out[1]);
    }

    // ── Self-loop dominates isolated node ─────────────────────────────────────

    #[test]
    fn propagate_self_loop_dominates_isolated() {
        // For an isolated node (degree 0 + normalized self-loop → weight 1.0),
        // the feature does not change over propagation.
        let g = CsrGraph::from_edges(2, &[(1, 1)]).expect("test invariant: value must be valid");
        // Node 0 is isolated (no edges, no self-loop in the edge list,
        // but normalized_adjacency adds implicit self-loop with weight 1/(0+1)=1).
        let x = vec![9.0_f32, 3.0_f32, 1.0_f32, 5.0_f32]; // 2 nodes × feat_dim=2
        let out = sgc_propagate(&g, &x, 2, 3).expect("test invariant: value must be valid");
        // Node 0 (isolated): feature should remain approximately x[0]=[9,3].
        assert!((out[0] - 9.0_f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 3.0_f32).abs() < 1e-5, "out[1]={}", out[1]);
    }

    // ── Uniform features stay uniform ─────────────────────────────────────────

    #[test]
    fn propagate_uniform_features_stays_uniform() {
        // If all nodes have the same feature vector, after propagation they
        // should still all have the same feature vector (rows of Â sum to 1
        // for symmetric normalised adjacency).
        let g = triangle_graph();
        let val = 5.0_f32;
        let x = vec![val; 3 * 2];
        let out = sgc_propagate(&g, &x, 2, 3).expect("test invariant: value must be valid");
        // All outputs should equal val (within floating-point tolerance).
        for (idx, &o) in out.iter().enumerate() {
            assert!((o - val).abs() < 1e-4, "out[{idx}]={o} expected={val}");
        }
    }
}
