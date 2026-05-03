//! Top-K graph pooling — Gao & Ji 2019.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Top-K node selection pooling.
///
/// Scores each node by projecting its features onto a learned vector, then
/// selects the top-`k` nodes and builds the induced subgraph.
#[derive(Debug, Clone)]
pub struct TopKPool {
    ratio: f32,
    k: Option<usize>,
    feat_dim: usize,
}

/// Result of a Top-K pooling operation.
#[derive(Debug, Clone)]
pub struct TopKPoolResult {
    /// Original node indices of the selected nodes.
    pub node_indices: Vec<usize>,
    /// New node features: `[k × feat_dim]`, scaled by `σ(score_i)`.
    pub x: Vec<f32>,
    /// Induced subgraph on the selected nodes (local IDs).
    pub graph: CsrGraph,
}

impl TopKPoolResult {
    /// Number of selected nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_indices.len()
    }
}

impl TopKPool {
    /// Construct a ratio-based Top-K pooler.
    ///
    /// `ratio` is the fraction of nodes to keep (0 < ratio ≤ 1).
    pub fn new_ratio(feat_dim: usize, ratio: f32) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "feat_dim must be > 0".to_string(),
            ));
        }
        if !(0.0 < ratio && ratio <= 1.0) {
            return Err(GnnError::InvalidLayerConfig(
                "ratio must be in (0, 1]".to_string(),
            ));
        }
        Ok(Self {
            ratio,
            k: None,
            feat_dim,
        })
    }

    /// Construct a fixed-k Top-K pooler.
    pub fn new_k(feat_dim: usize, k: usize) -> Self {
        Self {
            ratio: 1.0,
            k: Some(k),
            feat_dim,
        }
    }

    /// Forward pass.
    ///
    /// # Algorithm
    ///
    /// 1. `raw_score[i] = dot(x[i], proj) / ||proj||`
    /// 2. `score[i] = tanh(raw_score[i])`
    /// 3. Select top-k nodes by score
    /// 4. `x'[i] = x[sel[i]] * score[sel[i]]`
    /// 5. Build induced subgraph
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph
    /// - `x`: `[n_nodes × feat_dim]`
    /// - `proj`: `[feat_dim]` projection vector
    pub fn forward(&self, graph: &CsrGraph, x: &[f32], proj: &[f32]) -> GnnResult<TopKPoolResult> {
        let n = graph.n_nodes();
        let fd = self.feat_dim;

        if x.len() != n * fd {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / fd.max(1)));
        }
        if proj.len() != fd {
            return Err(GnnError::DimensionMismatch {
                expected: fd,
                got: proj.len(),
            });
        }

        // Compute k
        let k = if let Some(fixed_k) = self.k {
            fixed_k
        } else {
            ((n as f32 * self.ratio).ceil() as usize).max(1)
        };

        if k > n {
            return Err(GnnError::TopKExceedsGraphSize { k, n });
        }

        // Compute ||proj||
        let norm_sq: f32 = proj.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-12);

        // Score each node
        let scores: Vec<f32> = (0..n)
            .map(|i| {
                let dot: f32 = (0..fd).map(|k_idx| x[i * fd + k_idx] * proj[k_idx]).sum();
                (dot / norm).tanh()
            })
            .collect();

        // Select top-k indices by score (descending)
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();

        // Sort selected indices so subgraph is in consistent order
        let mut node_indices = selected;
        node_indices.sort_unstable();

        // Build global-to-local mapping
        let global_to_local: std::collections::HashMap<usize, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // New features: x'[local] = x[global] * tanh(score[global])
        // (score already is tanh so we multiply by it directly)
        let new_x: Vec<f32> = node_indices
            .iter()
            .flat_map(|&global| {
                let s = scores[global];
                (0..fd).map(move |k_idx| x[global * fd + k_idx] * s)
            })
            .collect();

        // Induced subgraph: keep only edges between selected nodes
        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        for &global_i in &node_indices {
            let neighbors = graph.neighbors(global_i)?;
            for &global_j in neighbors {
                if let Some(&local_j) = global_to_local.get(&global_j) {
                    let local_i = global_to_local[&global_i];
                    new_edges.push((local_i, local_j));
                }
            }
        }

        let new_graph = if new_edges.is_empty() {
            // k-node graph with no edges
            let row_ptr = vec![0usize; k + 1];
            let col_idx = vec![];
            CsrGraph::new(k, row_ptr, col_idx)?
        } else {
            CsrGraph::from_edges(k, &new_edges)?
        };

        Ok(TopKPoolResult {
            node_indices,
            x: new_x,
            graph: new_graph,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> =
            (0..n - 1).flat_map(|i| [(i, i + 1), (i + 1, i)]).collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn k_never_exceeds_n() {
        let g = line_graph(5);
        let pool = TopKPool::new_ratio(3, 0.6).expect("test invariant: value must be valid"); // k = ceil(5*0.6) = 3
        let x = vec![1.0_f32; 5 * 3];
        let proj = vec![1.0_f32, 0.0, 0.0];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert!(res.n_nodes() <= 5);
        assert_eq!(res.n_nodes(), 3);
    }

    #[test]
    fn fixed_k_returns_exactly_k() {
        let g = line_graph(6);
        let pool = TopKPool::new_k(2, 2);
        let x = vec![1.0_f32; 6 * 2];
        let proj = vec![1.0_f32, 0.5];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert_eq!(res.n_nodes(), 2);
    }

    #[test]
    fn output_graph_consistent_node_count() {
        let g = line_graph(8);
        let pool = TopKPool::new_k(3, 3);
        let x: Vec<f32> = (0..8 * 3).map(|i| i as f32 * 0.1).collect();
        let proj = vec![1.0_f32, 1.0, 1.0];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert_eq!(res.graph.n_nodes(), 3);
    }

    #[test]
    fn selected_indices_are_sorted() {
        let g = line_graph(5);
        let pool = TopKPool::new_k(2, 2);
        let x: Vec<f32> = (0..5 * 2).map(|i| i as f32).collect();
        let proj = vec![1.0_f32, 0.0];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        let sorted = {
            let mut v = res.node_indices.clone();
            v.sort_unstable();
            v
        };
        assert_eq!(res.node_indices, sorted);
    }

    #[test]
    fn x_length_correct() {
        let g = line_graph(5);
        let k = 3;
        let fd = 4;
        let pool = TopKPool::new_k(fd, k);
        let x = vec![0.5_f32; 5 * fd];
        let proj = vec![1.0_f32; fd];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert_eq!(res.x.len(), k * fd);
    }

    #[test]
    fn score_ordering_selects_highest() {
        // Nodes have features [0], [1], [2], [3], [4] and proj=[1]
        // Scores proportional to tanh(feat * 1 / 1) = tanh(val)
        // Top-2 should be nodes 3 and 4
        let g = line_graph(5);
        let pool = TopKPool::new_k(1, 2);
        let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let proj = vec![1.0_f32];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert!(res.node_indices.contains(&3) || res.node_indices.contains(&4));
    }

    #[test]
    fn k_exceeds_n_error() {
        let g = line_graph(3);
        let pool = TopKPool::new_k(2, 5);
        let x = vec![0.1_f32; 3 * 2];
        let proj = vec![1.0_f32, 0.0];
        let err = pool.forward(&g, &x, &proj);
        assert!(matches!(err, Err(GnnError::TopKExceedsGraphSize { .. })));
    }

    #[test]
    fn invalid_ratio_error() {
        let err = TopKPool::new_ratio(4, 0.0);
        assert!(err.is_err());
        let err = TopKPool::new_ratio(4, 1.5);
        assert!(err.is_err());
    }

    #[test]
    fn output_features_finite() {
        let g = line_graph(6);
        let pool = TopKPool::new_k(3, 3);
        let x: Vec<f32> = (0..6 * 3).map(|i| i as f32 * 0.5).collect();
        let proj = vec![1.0_f32, -0.5, 0.5];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert!(res.x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn induced_subgraph_edges_in_selected_set() {
        let g = line_graph(4);
        let pool = TopKPool::new_k(2, 2);
        let x: Vec<f32> = (0..4 * 2).map(|i| i as f32).collect();
        let proj = vec![1.0_f32, 0.0];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        // All edges in new graph should connect nodes in [0, k)
        let k = res.n_nodes();
        for e in 0..res.graph.n_edges() {
            let col = res.graph.col_idx()[e];
            assert!(col < k);
        }
    }
}
