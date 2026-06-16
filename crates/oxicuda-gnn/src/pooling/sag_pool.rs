//! Self-Attention Graph Pooling (SAGPool) — Lee, Lee & Kang 2019, ICML.
//!
//! Where Top-K pooling scores each node by a raw projection of its *own*
//! features, SAGPool scores nodes with a **graph-convolution self-attention**
//! head, so that a node's importance reflects its neighbourhood. The score uses
//! one GCN layer collapsing to a scalar:
//!
//! ```text
//! score = σ_gcn( D̃^{-1/2} Ã D̃^{-1/2} X Θ )      (Θ ∈ ℝ^{d × 1})
//! ```
//!
//! The top `⌈ratio · n⌉` nodes by score are retained, their features gated by
//! `tanh(score)`, and the induced subgraph is returned — making SAGPool a
//! learnable, topology-aware hierarchical pooling operator.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Self-attention graph pooling layer.
#[derive(Debug, Clone)]
pub struct SagPool {
    feat_dim: usize,
    ratio: f32,
    k: Option<usize>,
}

/// Result of a SAGPool forward pass.
#[derive(Debug, Clone)]
pub struct SagPoolResult {
    /// Original node indices of the selected nodes (sorted ascending).
    pub node_indices: Vec<usize>,
    /// Gated node features `[k × feat_dim]` (each row scaled by `tanh(score)`).
    pub x: Vec<f32>,
    /// Per-selected-node attention scores (post-`tanh`), aligned to `node_indices`.
    pub scores: Vec<f32>,
    /// Induced subgraph over the selected nodes (local IDs).
    pub graph: CsrGraph,
}

impl SagPoolResult {
    /// Number of retained nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_indices.len()
    }
}

impl SagPool {
    /// Construct a ratio-based SAGPool (keep `ceil(ratio · n)` nodes).
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if `feat_dim == 0` or `ratio ∉ (0, 1]`.
    pub fn new_ratio(feat_dim: usize, ratio: f32) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SAGPool: feat_dim must be > 0".to_string(),
            ));
        }
        if !(0.0 < ratio && ratio <= 1.0) {
            return Err(GnnError::InvalidLayerConfig(
                "SAGPool: ratio must be in (0, 1]".to_string(),
            ));
        }
        Ok(Self {
            feat_dim,
            ratio,
            k: None,
        })
    }

    /// Construct a fixed-`k` SAGPool.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if `feat_dim == 0`.
    pub fn new_k(feat_dim: usize, k: usize) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SAGPool: feat_dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            feat_dim,
            ratio: 1.0,
            k: Some(k),
        })
    }

    /// Compute raw (pre-`tanh`) self-attention scores for every node:
    /// `(Ŝ X θ)[i]`, where `Ŝ` is the symmetric-normalised adjacency.
    ///
    /// # Errors
    ///
    /// [`GnnError::NodeFeatureMismatch`] / [`GnnError::DimensionMismatch`] on
    /// shape errors.
    pub fn attention_scores(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        theta: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let fd = self.feat_dim;
        if x.len() != n * fd {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / fd.max(1)));
        }
        if theta.len() != fd {
            return Err(GnnError::DimensionMismatch {
                expected: fd,
                got: theta.len(),
            });
        }

        // Project each node onto θ → scalar per node: p[i] = x_i · θ.
        let proj: Vec<f32> = (0..n)
            .map(|i| (0..fd).map(|k| x[i * fd + k] * theta[k]).sum())
            .collect();

        // Aggregate with the normalised adjacency: score = Ŝ p.
        let (rows, cols, vals) = graph.normalized_adjacency();
        let mut score = vec![0.0_f32; n];
        for idx in 0..rows.len() {
            score[rows[idx]] += vals[idx] * proj[cols[idx]];
        }
        Ok(score)
    }

    /// Forward pass: score → select top-k → gate → induced subgraph.
    ///
    /// # Arguments
    ///
    /// * `graph`: CSR graph.
    /// * `x`: `[n × feat_dim]` node features.
    /// * `theta`: `[feat_dim]` attention projection vector.
    ///
    /// # Errors
    ///
    /// Shape errors and [`GnnError::TopKExceedsGraphSize`] when `k > n`.
    pub fn forward(&self, graph: &CsrGraph, x: &[f32], theta: &[f32]) -> GnnResult<SagPoolResult> {
        let n = graph.n_nodes();
        let fd = self.feat_dim;

        let raw = self.attention_scores(graph, x, theta)?;
        let gated: Vec<f32> = raw.iter().map(|&s| s.tanh()).collect();

        let k = match self.k {
            Some(fixed) => fixed,
            None => ((n as f32 * self.ratio).ceil() as usize).max(1),
        };
        if k > n {
            return Err(GnnError::TopKExceedsGraphSize { k, n });
        }

        // Select top-k node indices by gated score (descending).
        let mut indexed: Vec<(usize, f32)> =
            gated.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut node_indices: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
        node_indices.sort_unstable();

        let global_to_local: std::collections::HashMap<usize, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // Gated features and per-node scores in selection order.
        let mut new_x = vec![0.0_f32; k * fd];
        let mut scores = vec![0.0_f32; k];
        for (local, &global) in node_indices.iter().enumerate() {
            let s = gated[global];
            scores[local] = s;
            for d in 0..fd {
                new_x[local * fd + d] = x[global * fd + d] * s;
            }
        }

        // Induced subgraph: keep edges with both endpoints selected.
        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        for &global_i in &node_indices {
            let neighbors = graph.neighbors(global_i)?;
            let local_i = global_to_local[&global_i];
            for &global_j in neighbors {
                if let Some(&local_j) = global_to_local.get(&global_j) {
                    new_edges.push((local_i, local_j));
                }
            }
        }
        let new_graph = if new_edges.is_empty() {
            CsrGraph::new(k, vec![0usize; k + 1], vec![])?
        } else {
            CsrGraph::from_edges(k, &new_edges)?
        };

        Ok(SagPoolResult {
            node_indices,
            x: new_x,
            scores,
            graph: new_graph,
        })
    }

    /// Feature dimension.
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> =
            (0..n - 1).flat_map(|i| [(i, i + 1), (i + 1, i)]).collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn build_ratio_and_k() {
        let p = SagPool::new_ratio(4, 0.5).expect("build");
        assert_eq!(p.feat_dim(), 4);
        let p2 = SagPool::new_k(3, 2).expect("build");
        assert_eq!(p2.feat_dim(), 3);
    }

    #[test]
    fn invalid_config_errors() {
        assert!(SagPool::new_ratio(0, 0.5).is_err());
        assert!(SagPool::new_ratio(4, 0.0).is_err());
        assert!(SagPool::new_ratio(4, 1.5).is_err());
        assert!(SagPool::new_k(0, 2).is_err());
    }

    #[test]
    fn ratio_selects_ceil_fraction() {
        let g = line(5);
        let p = SagPool::new_ratio(2, 0.6).expect("build"); // ceil(5*0.6)=3
        let x = vec![1.0_f32; 5 * 2];
        let theta = vec![1.0_f32, 0.0];
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert_eq!(res.n_nodes(), 3);
    }

    #[test]
    fn fixed_k_returns_exactly_k() {
        let g = line(6);
        let p = SagPool::new_k(2, 2).expect("build");
        let x: Vec<f32> = (0..6 * 2).map(|i| i as f32 * 0.1).collect();
        let theta = vec![1.0_f32, 0.5];
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert_eq!(res.n_nodes(), 2);
        assert_eq!(res.graph.n_nodes(), 2);
    }

    #[test]
    fn output_feature_length() {
        let g = line(5);
        let fd = 3;
        let p = SagPool::new_k(fd, 3).expect("build");
        let x = vec![0.5_f32; 5 * fd];
        let theta = vec![1.0_f32; fd];
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert_eq!(res.x.len(), 3 * fd);
        assert_eq!(res.scores.len(), 3);
    }

    #[test]
    fn selected_indices_sorted() {
        let g = line(5);
        let p = SagPool::new_k(2, 3).expect("build");
        let x: Vec<f32> = (0..5).flat_map(|i| [i as f32, 0.0]).collect();
        let theta = vec![1.0_f32, 0.0];
        let res = p.forward(&g, &x, &theta).expect("forward");
        let mut sorted = res.node_indices.clone();
        sorted.sort_unstable();
        assert_eq!(res.node_indices, sorted);
    }

    #[test]
    fn attention_uses_neighborhood() {
        // SAGPool aggregates neighbours, so a high-feature node lifts its
        // neighbours' scores too — unlike a pure per-node projection.
        let g = line(4);
        let p = SagPool::new_k(1, 4).expect("build");
        // node 0 has large feature, others zero
        let x = vec![10.0_f32, 0.0, 0.0, 0.0];
        let theta = vec![1.0_f32];
        let raw = p.attention_scores(&g, &x, &theta).expect("scores");
        // node 1 (neighbour of 0) should get nonzero score from aggregation
        assert!(
            raw[1].abs() > 1e-6,
            "neighbour score should be nonzero: {}",
            raw[1]
        );
    }

    #[test]
    fn scores_are_tanh_bounded() {
        let g = line(5);
        let p = SagPool::new_k(1, 5).expect("build");
        let x: Vec<f32> = (0..5).map(|i| (i as f32) * 100.0).collect();
        let theta = vec![1.0_f32];
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert!(res.scores.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn k_exceeds_n_errors() {
        let g = line(3);
        let p = SagPool::new_k(2, 9).expect("build");
        let x = vec![0.1_f32; 3 * 2];
        let theta = vec![1.0_f32, 0.0];
        let err = p.forward(&g, &x, &theta);
        assert!(matches!(err, Err(GnnError::TopKExceedsGraphSize { .. })));
    }

    #[test]
    fn feature_mismatch_errors() {
        let g = line(4);
        let p = SagPool::new_k(3, 2).expect("build");
        let err = p.attention_scores(&g, &[1.0_f32; 5], &[1.0_f32; 3]);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn theta_dim_mismatch_errors() {
        let g = line(4);
        let p = SagPool::new_k(3, 2).expect("build");
        let x = vec![1.0_f32; 4 * 3];
        let err = p.attention_scores(&g, &x, &[1.0_f32, 2.0]); // len 2 != 3
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn induced_edges_within_selected() {
        let g = line(4);
        let p = SagPool::new_k(2, 2).expect("build");
        let x: Vec<f32> = (0..4 * 2).map(|i| i as f32).collect();
        let theta = vec![1.0_f32, 0.0];
        let res = p.forward(&g, &x, &theta).expect("forward");
        let k = res.n_nodes();
        for e in 0..res.graph.n_edges() {
            assert!(res.graph.col_idx()[e] < k);
        }
    }

    #[test]
    fn output_features_finite() {
        let g = line(6);
        let p = SagPool::new_ratio(3, 0.5).expect("build");
        let x: Vec<f32> = (0..6 * 3).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let theta = vec![0.5_f32, -0.5, 1.0];
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert!(res.x.iter().all(|v| v.is_finite()));
        assert!(res.scores.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn higher_score_node_retained() {
        // The node with the largest aggregated score should survive a k=1 pool.
        // Use a small feature so tanh stays in its non-saturated region and the
        // score ordering is preserved through the gating.
        let g = line(4);
        let p = SagPool::new_k(2, 1).expect("build");
        // make node 3 dominant (feat_dim = 2)
        let x = vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
        let theta = vec![1.0_f32, 0.0];
        let raw = p.attention_scores(&g, &x, &theta).expect("scores");
        // identify the argmax of the raw (pre-tanh) score
        let argmax = (0..raw.len())
            .max_by(|&a, &b| {
                raw[a]
                    .partial_cmp(&raw[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("nonempty");
        let res = p.forward(&g, &x, &theta).expect("forward");
        assert_eq!(res.n_nodes(), 1);
        assert_eq!(
            res.node_indices[0], argmax,
            "k=1 SAGPool must retain the highest-scoring node"
        );
    }
}
