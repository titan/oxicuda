//! PinSage — GraphSAGE with importance-based random-walk neighbor sampling.
//!
//! Reference: Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai,
//! William L. Hamilton, Jure Leskovec, "Graph Convolutional Neural Networks
//! for Web-Scale Recommender Systems", KDD 2018.
//!
//! Architecture:
//!   For each target node, PinSage estimates neighbor importance via short
//!   random walks: nodes visited frequently across multiple walks are
//!   considered the most influential neighbors. The top-k neighbors by visit
//!   count are retrieved and their feature vectors are mean-pooled. The pool
//!   is concatenated with the target node's own features and passed through
//!   a single linear layer followed by ReLU activation.
//!
//!   Walk-based importance sampling replaces the full neighbourhood
//!   enumeration of vanilla GraphSAGE, making the approach tractable for
//!   billion-node graphs.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ─── public types ─────────────────────────────────────────────────────────────

/// Hyper-parameters for [`PinSage`].
#[derive(Debug)]
pub struct PinSageConfig {
    /// Total number of nodes in the graph.
    pub n_nodes: usize,
    /// Number of random walks to run per target node.
    pub n_walks: usize,
    /// Length of each random walk (number of steps).
    pub walk_len: usize,
    /// Number of top-importance neighbors to aggregate (k).
    pub n_neighbors: usize,
    /// Input node feature dimension.
    pub d_input: usize,
    /// Output (aggregated) feature dimension.
    pub d_output: usize,
}

/// PinSage graph convolution layer with random-walk importance sampling.
#[derive(Debug)]
pub struct PinSage {
    /// Weight matrix `W`: `[d_output × (2 * d_input)]` (row-major).
    w: Vec<f32>,
    /// Bias vector `b`: `[d_output]`.
    b: Vec<f32>,
    /// Frozen configuration.
    config: PinSageConfig,
}

impl PinSage {
    /// Construct a PinSage layer with Xavier-initialised weights.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidConfig`] when `n_walks == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `d_input == 0` or `d_output == 0`.
    pub fn new(config: PinSageConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if config.n_walks == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_walks must be > 0".into(),
            });
        }
        if config.d_input == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if config.d_output == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }

        let d_in2 = 2 * config.d_input;
        let d_out = config.d_output;
        // Xavier init: scale = sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (d_in2 + d_out) as f32).sqrt();

        let w: Vec<f32> = (0..d_out * d_in2)
            .map(|_| rng.next_normal() * scale)
            .collect();

        let b: Vec<f32> = vec![0.0_f32; d_out];

        Ok(Self { w, b, config })
    }

    /// Sample top-`n_neighbors` important neighbors for `node_id` using
    /// random walks on the adjacency list `adj`.
    ///
    /// Returns a vector of at most `n_neighbors` unique node indices sorted
    /// by descending visit frequency.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] when `node_id >= n_nodes`.
    pub fn sample_neighbors(
        &self,
        node_id: usize,
        adj: &[Vec<usize>],
        rng: &mut LcgRng,
    ) -> RecsysResult<Vec<usize>> {
        if node_id >= self.config.n_nodes {
            return Err(RecsysError::ItemOutOfBounds {
                idx: node_id,
                n: self.config.n_nodes,
            });
        }

        // Gracefully handle adjacency lists shorter than n_nodes.
        let neighbors_of = |node: usize| -> &[usize] {
            if node < adj.len() {
                adj[node].as_slice()
            } else {
                &[]
            }
        };

        let start_neighbors = neighbors_of(node_id);
        if start_neighbors.is_empty() {
            return Ok(Vec::new());
        }

        // Count visit frequencies via n_walks random walks of length walk_len.
        // We use a flat array indexed by node id (bounded by n_nodes).
        let mut visit_count: Vec<u32> = vec![0_u32; self.config.n_nodes];

        for _ in 0..self.config.n_walks {
            let mut current = node_id;
            for _ in 0..self.config.walk_len {
                let nbrs = neighbors_of(current);
                if nbrs.is_empty() {
                    break;
                }
                let next = nbrs[rng.next_usize(nbrs.len())];
                // Only count nodes that are not the source itself.
                if next != node_id && next < self.config.n_nodes {
                    visit_count[next] = visit_count[next].saturating_add(1);
                }
                current = next;
            }
        }

        // Collect unique visited nodes (visit_count > 0) and sort by count desc.
        let mut visited: Vec<(u32, usize)> = visit_count
            .iter()
            .enumerate()
            .filter_map(
                |(node, &cnt)| {
                    if cnt > 0 { Some((cnt, node)) } else { None }
                },
            )
            .collect();
        visited.sort_unstable_by_key(|&(cnt, _)| std::cmp::Reverse(cnt));

        let top_k: Vec<usize> = visited
            .iter()
            .take(self.config.n_neighbors)
            .map(|&(_, node)| node)
            .collect();

        Ok(top_k)
    }

    /// Compute the PinSage convolution for `node_id`.
    ///
    /// Steps:
    /// 1. Sample top-k neighbors via importance-weighted random walks.
    /// 2. Mean-pool neighbor features (zeros if isolated node).
    /// 3. Concatenate self-features (d_input) with pooled (d_input).
    /// 4. Linear projection (d_output × 2*d_input) + bias.
    /// 5. ReLU activation.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] when `node_id >= n_nodes`.
    /// - [`RecsysError::DimensionMismatch`] when `node_feats.len() != n_nodes * d_input`.
    pub fn forward(
        &self,
        node_id: usize,
        node_feats: &[f32],
        adj: &[Vec<usize>],
        rng: &mut LcgRng,
    ) -> RecsysResult<Vec<f32>> {
        if node_id >= self.config.n_nodes {
            return Err(RecsysError::ItemOutOfBounds {
                idx: node_id,
                n: self.config.n_nodes,
            });
        }
        let expected = self.config.n_nodes * self.config.d_input;
        if node_feats.len() != expected {
            return Err(RecsysError::DimensionMismatch {
                expected,
                got: node_feats.len(),
            });
        }

        let d_in = self.config.d_input;
        let d_out = self.config.d_output;

        // Sample top-k neighbors.
        let neighbors = self.sample_neighbors(node_id, adj, rng)?;

        // Mean-pool neighbor features.
        let pooled: Vec<f32> = if neighbors.is_empty() {
            vec![0.0_f32; d_in]
        } else {
            let mut pool = vec![0.0_f32; d_in];
            for &nb in &neighbors {
                let feat = &node_feats[nb * d_in..(nb + 1) * d_in];
                for k in 0..d_in {
                    pool[k] += feat[k];
                }
            }
            let inv_n = 1.0 / neighbors.len() as f32;
            pool.iter_mut().for_each(|v| *v *= inv_n);
            pool
        };

        // Concatenate self-features and pooled features.
        let self_feat = &node_feats[node_id * d_in..(node_id + 1) * d_in];
        let mut concat = Vec::with_capacity(2 * d_in);
        concat.extend_from_slice(self_feat);
        concat.extend_from_slice(&pooled);

        // Linear: W (d_out × 2*d_in) * concat (2*d_in) + b (d_out)
        let d_in2 = 2 * d_in;
        let mut out: Vec<f32> = (0..d_out)
            .map(|row| {
                self.w[row * d_in2..(row + 1) * d_in2]
                    .iter()
                    .zip(concat.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum::<f32>()
                    + self.b[row]
            })
            .collect();

        // ReLU activation.
        for v in out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        Ok(out)
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_config(n_nodes: usize, n_walks: usize, walk_len: usize) -> PinSageConfig {
        PinSageConfig {
            n_nodes,
            n_walks,
            walk_len,
            n_neighbors: 3,
            d_input: 4,
            d_output: 8,
        }
    }

    fn make_model(n_nodes: usize, n_walks: usize, walk_len: usize) -> PinSage {
        let mut rng = LcgRng::new(42);
        PinSage::new(make_config(n_nodes, n_walks, walk_len), &mut rng)
            .expect("model construction must succeed")
    }

    /// Fully connected graph: every node connects to every other node.
    fn dense_adj(n: usize) -> Vec<Vec<usize>> {
        (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect()
    }

    /// Build node feature matrix with distinct values per node.
    fn make_feats(n_nodes: usize, d_input: usize) -> Vec<f32> {
        (0..n_nodes * d_input)
            .map(|i| (i as f32) * 0.01 + 0.5)
            .collect()
    }

    // 1. forward returns vec of size d_output
    #[test]
    fn forward_shape() {
        let n = 6;
        let model = make_model(n, 4, 3);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let out = model
            .forward(0, &feats, &adj, &mut rng)
            .expect("forward must succeed");
        assert_eq!(out.len(), 8, "forward output must have d_output elements");
    }

    // 2. all forward output values are finite
    #[test]
    fn forward_finite() {
        let n = 6;
        let model = make_model(n, 4, 3);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let out = model
            .forward(2, &feats, &adj, &mut rng)
            .expect("forward must succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} must be finite");
        }
    }

    // 3. sample_neighbors returns at most n_neighbors
    #[test]
    fn sample_neighbors_returns_at_most_n() {
        let n = 10;
        let model = make_model(n, 8, 4);
        let adj = dense_adj(n);
        let mut rng = LcgRng::new(42);
        let neighbors = model
            .sample_neighbors(0, &adj, &mut rng)
            .expect("sample must succeed");
        assert!(
            neighbors.len() <= 3,
            "sample_neighbors must return at most n_neighbors, got {}",
            neighbors.len()
        );
    }

    // 4. isolated node (empty adjacency) works
    #[test]
    fn isolated_node_works() {
        let n = 4;
        let model = make_model(n, 4, 3);
        // All nodes isolated.
        let adj: Vec<Vec<usize>> = vec![vec![]; n];
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let out = model
            .forward(1, &feats, &adj, &mut rng)
            .expect("isolated forward must succeed");
        assert_eq!(
            out.len(),
            8,
            "isolated node forward must return d_output elements"
        );
        for &v in &out {
            assert!(v >= 0.0, "ReLU output must be non-negative; got {v}");
        }
    }

    // 5. n_walks=0 returns Err from new()
    #[test]
    fn n_walks_zero_error() {
        let mut rng = LcgRng::new(42);
        let cfg = PinSageConfig {
            n_nodes: 6,
            n_walks: 0,
            walk_len: 3,
            n_neighbors: 2,
            d_input: 4,
            d_output: 8,
        };
        let result = PinSage::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::InvalidConfig { .. })),
            "expected InvalidConfig, got: {result:?}"
        );
    }

    // 6. node_id >= n_nodes returns Err
    #[test]
    fn node_out_of_range_error() {
        let n = 4;
        let model = make_model(n, 4, 2);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let result = model.forward(99, &feats, &adj, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::ItemOutOfBounds { idx: 99, n: 4 })),
            "expected ItemOutOfBounds, got: {result:?}"
        );
    }

    // 7. different nodes with different features give different output
    #[test]
    fn different_nodes_different_output() {
        let n = 8;
        let model = make_model(n, 6, 3);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let out_0 = model
            .forward(0, &feats, &adj, &mut rng_a)
            .expect("forward node 0");
        let out_7 = model
            .forward(7, &feats, &adj, &mut rng_b)
            .expect("forward node 7");
        let diff: f32 = out_0
            .iter()
            .zip(out_7.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different nodes should have different forward output (diff={diff})"
        );
    }

    // 8. forward output values are all >= 0 (ReLU)
    #[test]
    fn output_relu_nonneg() {
        let n = 6;
        let model = make_model(n, 4, 3);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let out = model
            .forward(3, &feats, &adj, &mut rng)
            .expect("forward must succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "output[{i}] = {v} must be >= 0 after ReLU");
        }
    }

    // 9. walk_len=1 works fine
    #[test]
    fn walk_len_1_works() {
        let n = 5;
        let model = make_model(n, 4, 1);
        let adj = dense_adj(n);
        let feats = make_feats(n, 4);
        let mut rng = LcgRng::new(42);
        let out = model
            .forward(2, &feats, &adj, &mut rng)
            .expect("walk_len=1 must work");
        assert_eq!(out.len(), 8);
    }

    // 10. d_input=0 returns Err from new()
    #[test]
    fn d_input_zero_error() {
        let mut rng = LcgRng::new(42);
        let cfg = PinSageConfig {
            n_nodes: 4,
            n_walks: 2,
            walk_len: 2,
            n_neighbors: 2,
            d_input: 0,
            d_output: 4,
        };
        let result = PinSage::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::InvalidEmbeddingDim { d: 0 })),
            "expected InvalidEmbeddingDim, got: {result:?}"
        );
    }

    // 11. sample_neighbors respects node validity
    #[test]
    fn sample_neighbors_node_out_of_range() {
        let n = 4;
        let model = make_model(n, 2, 2);
        let adj = dense_adj(n);
        let mut rng = LcgRng::new(42);
        let result = model.sample_neighbors(99, &adj, &mut rng);
        assert!(
            matches!(result, Err(RecsysError::ItemOutOfBounds { idx: 99, n: 4 })),
            "expected ItemOutOfBounds, got: {result:?}"
        );
    }
}
