//! GraphSAGE-style neighbor sampling for mini-batch GNN training.
//!
//! Hamilton et al. 2017 "Inductive Representation Learning on Large Graphs".
//! Samples a fixed number of neighbors at each GNN layer so that each mini-batch
//! operates on a compact subgraph rather than the full graph.

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;

// Re-export the type alias so downstream can reference it concisely.
/// Type alias used for the RNG passed to sampling operations.
pub type GnnRng = LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for multi-hop neighbor sampling.
#[derive(Debug, Clone)]
pub struct NeighborSampleConfig {
    /// Number of GNN layers (hops).
    pub n_hops: usize,
    /// `n_neighbors[h]` = how many neighbors to sample at hop `h`.
    ///
    /// Must have length equal to `n_hops`.
    pub n_neighbors: Vec<usize>,
}

// ─── Sampled subgraph ─────────────────────────────────────────────────────────

/// A sampled subgraph produced by [`NeighborSampler::sample`].
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    /// All node global IDs in this subgraph (deduplicated, seed nodes included).
    pub node_ids: Vec<usize>,
    /// Source node indices in *local* (subgraph) id space for each edge.
    pub edge_src: Vec<usize>,
    /// Destination node indices in *local* (subgraph) id space for each edge.
    pub edge_dst: Vec<usize>,
    /// The seed nodes (original query nodes) — subsets of `node_ids`.
    pub seed_nodes: Vec<usize>,
}

impl SampledSubgraph {
    /// Map a global node id to its local index, if it is in the subgraph.
    #[must_use]
    pub fn global_to_local(&self, global_id: usize) -> Option<usize> {
        self.node_ids.iter().position(|&n| n == global_id)
    }

    /// Number of nodes in the subgraph.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.node_ids.len()
    }

    /// Number of edges in the subgraph.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edge_src.len()
    }
}

// ─── NeighborSampler ─────────────────────────────────────────────────────────

/// GraphSAGE-style multi-hop neighbor sampler over an adjacency-list graph.
pub struct NeighborSampler {
    /// Adjacency list: `adj[v]` = list of neighbors of node `v`.
    adj: Vec<Vec<usize>>,
    config: NeighborSampleConfig,
}

impl NeighborSampler {
    /// Create a new sampler.
    ///
    /// # Errors
    ///
    /// - [`GnnError::EmptyGraph`] if `adj` is empty.
    /// - [`GnnError::InvalidLayerConfig`] if config is inconsistent.
    /// - [`GnnError::NodeIndexOutOfRange`] if any neighbor index is out of bounds.
    pub fn new(adj: Vec<Vec<usize>>, config: NeighborSampleConfig) -> GnnResult<Self> {
        if adj.is_empty() {
            return Err(GnnError::EmptyGraph);
        }
        if config.n_hops == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "n_hops must be at least 1".to_string(),
            ));
        }
        if config.n_neighbors.len() != config.n_hops {
            return Err(GnnError::InvalidLayerConfig(format!(
                "n_neighbors length {} != n_hops {}",
                config.n_neighbors.len(),
                config.n_hops
            )));
        }
        for &k in &config.n_neighbors {
            if k == 0 {
                return Err(GnnError::InvalidLayerConfig(
                    "n_neighbors per hop must be > 0".to_string(),
                ));
            }
        }
        let n = adj.len();
        for nbrs in adj.iter() {
            for &u in nbrs {
                if u >= n {
                    return Err(GnnError::NodeIndexOutOfRange { idx: u, n_nodes: n });
                }
            }
        }
        Ok(Self { adj, config })
    }

    /// Total number of nodes in the full graph.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.adj.len()
    }

    /// Sample a subgraph for a batch of seed nodes.
    ///
    /// # Algorithm
    ///
    /// 1. Start with `seeds` as the initial frontier.
    /// 2. For each hop `h`:
    ///    - For every node in the current frontier: sample
    ///      `min(n_neighbors[h], degree(v))` neighbors *without replacement*.
    ///    - Collect sampled neighbors as the next frontier (deduplicated across nodes).
    /// 3. The node set is the union of seeds + all sampled nodes.
    /// 4. Build the edge list with local indices for every edge (u → v) where
    ///    both endpoints are in the node set and u was actually sampled as a
    ///    neighbor of v during the relevant hop.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `seeds` is empty.
    /// - [`GnnError::NodeIndexOutOfRange`] if any seed is out of bounds.
    pub fn sample(&self, seeds: &[usize], rng: &mut GnnRng) -> GnnResult<SampledSubgraph> {
        if seeds.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "seeds must be non-empty".to_string(),
            ));
        }
        let n = self.adj.len();
        for &s in seeds {
            if s >= n {
                return Err(GnnError::NodeIndexOutOfRange { idx: s, n_nodes: n });
            }
        }

        // Collected node set (insertion-ordered for reproducibility).
        // We use a Vec as a lightweight ordered set; presence is checked via a
        // boolean membership array since the graph can be large.
        let mut in_set = vec![false; n];
        let mut node_ids: Vec<usize> = Vec::new();

        // Helper: add a node to the set if not already present.
        let add_node = |node: usize, in_set: &mut Vec<bool>, node_ids: &mut Vec<usize>| {
            if !in_set[node] {
                in_set[node] = true;
                node_ids.push(node);
            }
        };

        // Seed nodes are always included.
        for &s in seeds {
            add_node(s, &mut in_set, &mut node_ids);
        }

        // Edges collected as (global_src, global_dst) pairs.
        let mut raw_edges: Vec<(usize, usize)> = Vec::new();

        // Multi-hop BFS with sampling.
        let mut frontier: Vec<usize> = seeds.to_vec();

        for hop in 0..self.config.n_hops {
            let k = self.config.n_neighbors[hop];
            let mut next_frontier_set: Vec<bool> = vec![false; n];
            let mut next_frontier: Vec<usize> = Vec::new();

            for &v in &frontier {
                let nbrs = &self.adj[v];
                if nbrs.is_empty() {
                    continue;
                }
                // Sample min(k, degree) neighbors without replacement.
                let sample_count = k.min(nbrs.len());
                let sampled = sample_without_replacement(nbrs, sample_count, rng);

                for &u in &sampled {
                    // Edge u → v (u is neighbor sending to v).
                    raw_edges.push((u, v));
                    add_node(u, &mut in_set, &mut node_ids);
                    if !next_frontier_set[u] {
                        next_frontier_set[u] = true;
                        next_frontier.push(u);
                    }
                }
            }
            frontier = next_frontier;
        }

        // Build local-index lookup: global_id → local_idx.
        let mut global_to_local = vec![usize::MAX; n];
        for (local_idx, &global_id) in node_ids.iter().enumerate() {
            global_to_local[global_id] = local_idx;
        }

        // Convert raw edges to local indices; deduplicate.
        let mut seen_edges: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        let mut edge_src: Vec<usize> = Vec::with_capacity(raw_edges.len());
        let mut edge_dst: Vec<usize> = Vec::with_capacity(raw_edges.len());

        for (gsrc, gdst) in raw_edges {
            let lsrc = global_to_local[gsrc];
            let ldst = global_to_local[gdst];
            debug_assert!(lsrc != usize::MAX, "sampled node not in set");
            debug_assert!(ldst != usize::MAX, "seed node not in set");
            if seen_edges.insert((lsrc, ldst)) {
                edge_src.push(lsrc);
                edge_dst.push(ldst);
            }
        }

        // Seed nodes as local indices.
        let seed_nodes: Vec<usize> = seeds.iter().map(|&s| global_to_local[s]).collect();

        Ok(SampledSubgraph {
            node_ids,
            edge_src,
            edge_dst,
            seed_nodes,
        })
    }
}

// ─── Reservoir sampling without replacement ───────────────────────────────────

/// Sample exactly `k` items from `items` without replacement using
/// the Fisher-Yates partial shuffle (O(k) swaps on a temporary copy).
fn sample_without_replacement(items: &[usize], k: usize, rng: &mut LcgRng) -> Vec<usize> {
    debug_assert!(k <= items.len());
    let mut buf: Vec<usize> = items.to_vec();
    let n = buf.len();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        buf.swap(i, j);
    }
    buf[..k].to_vec()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_ring(n: usize) -> Vec<Vec<usize>> {
        // Ring graph: v → (v+1) % n
        (0..n).map(|v| vec![(v + 1) % n]).collect()
    }

    fn make_star(n: usize) -> Vec<Vec<usize>> {
        // Hub (0) is neighbor of all others; each leaf only has hub.
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for v in 1..n {
            adj[0].push(v);
            adj[v].push(0);
        }
        adj
    }

    fn make_complete(n: usize) -> Vec<Vec<usize>> {
        (0..n)
            .map(|v| (0..n).filter(|&u| u != v).collect())
            .collect()
    }

    fn rng() -> LcgRng {
        LcgRng::new(42)
    }

    // 1 ─ sample returns seeds
    #[test]
    fn sample_returns_seeds() {
        let adj = make_ring(10);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![2],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let seeds = vec![3, 7];
        let sg = sampler
            .sample(&seeds, &mut rng())
            .expect("value should be present");
        assert!(sg.node_ids.contains(&3));
        assert!(sg.node_ids.contains(&7));
    }

    // 2 ─ subgraph includes all seeds
    #[test]
    fn subgraph_includes_all_seeds() {
        let adj = make_star(8);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![3],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let seeds: Vec<usize> = (1..5).collect();
        let sg = sampler
            .sample(&seeds, &mut rng())
            .expect("value should be present");
        for &s in &seeds {
            assert!(sg.node_ids.contains(&s), "seed {s} missing");
        }
    }

    // 3 ─ node_ids include seeds (via seed_nodes)
    #[test]
    fn node_ids_include_seeds() {
        let adj = make_complete(6);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![2],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let seeds = vec![0, 5];
        let sg = sampler
            .sample(&seeds, &mut rng())
            .expect("value should be present");
        // seed_nodes are local indices; check they map back to seeds
        let local_0 = sg.global_to_local(0).expect("node 0 in subgraph");
        let local_5 = sg.global_to_local(5).expect("node 5 in subgraph");
        assert!(sg.seed_nodes.contains(&local_0));
        assert!(sg.seed_nodes.contains(&local_5));
    }

    // 4 ─ edge_src in range
    #[test]
    fn edge_src_in_range() {
        let adj = make_complete(5);
        let config = NeighborSampleConfig {
            n_hops: 2,
            n_neighbors: vec![2, 2],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let sg = sampler
            .sample(&[0], &mut rng())
            .expect("value should be present");
        let n_local = sg.n_nodes();
        for &s in &sg.edge_src {
            assert!(s < n_local, "edge_src {s} >= n_nodes {n_local}");
        }
    }

    // 5 ─ edge_dst in range
    #[test]
    fn edge_dst_in_range() {
        let adj = make_complete(5);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![3],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let sg = sampler
            .sample(&[2], &mut rng())
            .expect("value should be present");
        let n_local = sg.n_nodes();
        for &d in &sg.edge_dst {
            assert!(d < n_local, "edge_dst {d} >= n_nodes {n_local}");
        }
    }

    // 6 ─ n_neighbors bounded: sampled ≤ min(k, degree)
    #[test]
    fn n_neighbors_bounded() {
        // Each node has exactly 1 neighbor (ring)
        let adj = make_ring(5);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![10],
        }; // k > degree
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let sg = sampler
            .sample(&[0], &mut rng())
            .expect("value should be present");
        // Only 1 neighbor can be sampled from node 0
        assert!(sg.n_edges() <= 1, "expected ≤ 1 edge, got {}", sg.n_edges());
    }

    // 7 ─ isolated node works
    #[test]
    fn isolated_node_works() {
        let adj = vec![vec![], vec![], vec![]]; // 3 isolated nodes
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![5],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let sg = sampler
            .sample(&[1], &mut rng())
            .expect("value should be present");
        assert_eq!(sg.n_nodes(), 1);
        assert_eq!(sg.n_edges(), 0);
    }

    // 8 ─ empty seeds returns error
    #[test]
    fn empty_seeds_error() {
        let adj = make_ring(4);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![2],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let result = sampler.sample(&[], &mut rng());
        assert!(result.is_err());
    }

    // 9 ─ 1-hop sample
    #[test]
    fn one_hop_sample() {
        let adj = make_star(10); // hub=0, leaves=1..9
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![3],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        let sg = sampler
            .sample(&[0], &mut rng())
            .expect("value should be present");
        // Node 0 (hub) is seed; up to 3 of its 9 neighbors should be sampled
        assert!(sg.n_nodes() <= 4, "seed + ≤3 neighbors");
        assert!(sg.n_nodes() >= 1);
    }

    // 10 ─ 2-hop sample is larger than 1-hop (with enough graph)
    #[test]
    fn two_hop_sample_larger_than_one_hop() {
        let adj = make_complete(8);
        let config1 = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![2],
        };
        let config2 = NeighborSampleConfig {
            n_hops: 2,
            n_neighbors: vec![2, 2],
        };
        let s1 = NeighborSampler::new(adj.clone(), config1).expect("value should be present");
        let s2 = NeighborSampler::new(adj, config2).expect("new should succeed");
        let sg1 = s1
            .sample(&[0], &mut LcgRng::new(7))
            .expect("value should be present");
        let sg2 = s2
            .sample(&[0], &mut LcgRng::new(7))
            .expect("value should be present");
        assert!(
            sg2.n_nodes() >= sg1.n_nodes(),
            "2-hop ({}) should have ≥ nodes than 1-hop ({})",
            sg2.n_nodes(),
            sg1.n_nodes()
        );
    }

    // 11 ─ n_nodes accessor
    #[test]
    fn n_nodes_accessor() {
        let adj = make_ring(12);
        let config = NeighborSampleConfig {
            n_hops: 1,
            n_neighbors: vec![1],
        };
        let sampler = NeighborSampler::new(adj, config).expect("new should succeed");
        assert_eq!(sampler.n_nodes(), 12);
    }

    // 12 ─ invalid n_hops=0
    #[test]
    fn invalid_n_hops_zero() {
        let adj = make_ring(5);
        let config = NeighborSampleConfig {
            n_hops: 0,
            n_neighbors: vec![],
        };
        let result = NeighborSampler::new(adj, config);
        assert!(result.is_err());
    }
}
