//! Graph sampling algorithms: neighborhood sampling, random walks, biased walks.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::handle::LcgRng;

/// k-hop neighborhood sampler.
///
/// For each seed node, samples up to `max_neighbors[hop]` neighbors at each hop,
/// building a subgraph for mini-batch GNN training.
pub struct NeighborhoodSampler {
    max_neighbors: Vec<usize>,
    n_hops: usize,
}

/// Result of neighborhood sampling.
#[derive(Debug, Clone)]
pub struct SampledGraph {
    /// Original node IDs used as seeds.
    pub seed_nodes: Vec<usize>,
    /// All unique node IDs in the sampled subgraph (including seeds).
    pub sampled_nodes: Vec<usize>,
    /// Edge source indices in local (subgraph) ID space.
    pub src: Vec<usize>,
    /// Edge destination indices in local (subgraph) ID space.
    pub dst: Vec<usize>,
    /// Maps local ID → original global node ID.
    pub local_to_global: Vec<usize>,
}

impl SampledGraph {
    /// Number of nodes in the sampled subgraph.
    pub fn n_nodes(&self) -> usize {
        self.sampled_nodes.len()
    }

    /// Number of edges in the sampled subgraph.
    pub fn n_edges(&self) -> usize {
        self.src.len()
    }

    /// Look up the local (subgraph) ID for a global node ID, if present.
    pub fn global_to_local(&self, global_id: usize) -> Option<usize> {
        self.local_to_global.iter().position(|&g| g == global_id)
    }
}

impl NeighborhoodSampler {
    /// Construct a sampler with the given per-hop fanout list.
    ///
    /// `max_neighbors.len()` determines the number of hops.
    /// Each entry must be > 0.
    pub fn new(max_neighbors: Vec<usize>) -> GnnResult<Self> {
        if max_neighbors.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "max_neighbors must have at least one hop".to_string(),
            ));
        }
        for &m in &max_neighbors {
            if m == 0 {
                return Err(GnnError::InvalidLayerConfig(
                    "max_neighbors per hop must be > 0".to_string(),
                ));
            }
        }
        let n_hops = max_neighbors.len();
        Ok(Self {
            max_neighbors,
            n_hops,
        })
    }

    /// BFS neighborhood sampling from seed nodes.
    ///
    /// For each seed, samples up to `max_neighbors[0]` one-hop neighbors,
    /// then `max_neighbors[1]` two-hop neighbors from each one-hop node, etc.
    /// Returns a `SampledGraph` with edges in local ID space.
    pub fn sample(
        &self,
        graph: &CsrGraph,
        seed_nodes: &[usize],
        rng: &mut LcgRng,
    ) -> GnnResult<SampledGraph> {
        if seed_nodes.is_empty() {
            return Err(GnnError::InvalidAggregation("seed_nodes must be non-empty"));
        }
        for &s in seed_nodes {
            if s >= graph.n_nodes() {
                return Err(GnnError::NodeIndexOutOfRange {
                    idx: s,
                    n_nodes: graph.n_nodes(),
                });
            }
        }

        // global_id → local_id
        let mut global_to_local: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut local_to_global: Vec<usize> = Vec::new();

        // Helper: intern a global node id
        let intern = |g: usize,
                      g2l: &mut std::collections::HashMap<usize, usize>,
                      l2g: &mut Vec<usize>|
         -> usize {
            if let Some(&lid) = g2l.get(&g) {
                lid
            } else {
                let lid = l2g.len();
                l2g.push(g);
                g2l.insert(g, lid);
                lid
            }
        };

        // Initialise with seed nodes
        for &s in seed_nodes {
            intern(s, &mut global_to_local, &mut local_to_global);
        }

        let mut frontier: Vec<usize> = seed_nodes.to_vec();
        let mut all_sampled_edges: Vec<(usize, usize)> = Vec::new(); // (global_src, global_dst)

        for hop in 0..self.n_hops {
            let fanout = self.max_neighbors[hop];
            let mut next_frontier: Vec<usize> = Vec::new();

            for &node in &frontier {
                let neighbors = graph.neighbors(node)?;
                if neighbors.is_empty() {
                    continue;
                }
                // Sample up to fanout neighbors without replacement (reservoir)
                let sampled = sample_k_from_slice(neighbors, fanout, rng);
                for &nb in &sampled {
                    intern(nb, &mut global_to_local, &mut local_to_global);
                    all_sampled_edges.push((nb, node)); // message flows from nb to node
                    next_frontier.push(nb);
                }
            }
            frontier = next_frontier;
        }

        // Convert edges to local IDs
        let mut src_local = Vec::with_capacity(all_sampled_edges.len());
        let mut dst_local = Vec::with_capacity(all_sampled_edges.len());
        for (gs, gd) in &all_sampled_edges {
            if let (Some(&ls), Some(&ld)) = (global_to_local.get(gs), global_to_local.get(gd)) {
                src_local.push(ls);
                dst_local.push(ld);
            }
        }

        Ok(SampledGraph {
            seed_nodes: seed_nodes.to_vec(),
            sampled_nodes: local_to_global.clone(),
            src: src_local,
            dst: dst_local,
            local_to_global,
        })
    }
}

/// Sample up to `k` elements from `slice` without replacement using reservoir sampling.
fn sample_k_from_slice<T: Copy>(slice: &[T], k: usize, rng: &mut LcgRng) -> Vec<T> {
    let n = slice.len();
    if k >= n {
        return slice.to_vec();
    }
    // Reservoir sampling (Algorithm R)
    let mut reservoir: Vec<T> = slice[..k].to_vec();
    for (i, &item) in slice.iter().enumerate().skip(k) {
        let j = rng.next_usize(i + 1);
        if j < k {
            reservoir[j] = item;
        }
    }
    reservoir
}

/// Uniform random walk on the graph.
///
/// Starts at `start`, takes `length` steps, choosing a uniformly random
/// neighbor at each step.  If a node has no outgoing neighbors the walk
/// stays at the current node.
pub fn random_walk(
    graph: &CsrGraph,
    start: usize,
    length: usize,
    rng: &mut LcgRng,
) -> GnnResult<Vec<usize>> {
    if start >= graph.n_nodes() {
        return Err(GnnError::NodeIndexOutOfRange {
            idx: start,
            n_nodes: graph.n_nodes(),
        });
    }
    let mut walk = Vec::with_capacity(length + 1);
    walk.push(start);
    let mut current = start;

    for _ in 0..length {
        let neighbors = graph.neighbors(current)?;
        if neighbors.is_empty() {
            walk.push(current);
        } else {
            let next = neighbors[rng.next_usize(neighbors.len())];
            walk.push(next);
            current = next;
        }
    }
    Ok(walk)
}

/// Node2Vec-style biased random walk.
///
/// Transition probabilities are unnormalized:
/// - Return to the previous node: weight `1/p`
/// - Move to a common neighbor of prev and current: weight `1`
/// - Move to a new node (not prev, not common): weight `1/q`
///
/// `p` (return parameter) and `q` (in-out parameter) must be > 0.
pub fn biased_walk(
    graph: &CsrGraph,
    start: usize,
    length: usize,
    p: f32,
    q: f32,
    rng: &mut LcgRng,
) -> GnnResult<Vec<usize>> {
    if p <= 0.0 {
        return Err(GnnError::InvalidLayerConfig("p must be > 0".to_string()));
    }
    if q <= 0.0 {
        return Err(GnnError::InvalidLayerConfig("q must be > 0".to_string()));
    }
    if start >= graph.n_nodes() {
        return Err(GnnError::NodeIndexOutOfRange {
            idx: start,
            n_nodes: graph.n_nodes(),
        });
    }

    let mut walk = Vec::with_capacity(length + 1);
    walk.push(start);

    let neighbors_0 = graph.neighbors(start)?;
    if neighbors_0.is_empty() || length == 0 {
        return Ok(walk);
    }
    // First step: uniform
    let first = neighbors_0[rng.next_usize(neighbors_0.len())];
    walk.push(first);

    let mut prev = start;
    let mut current = first;

    for _ in 1..length {
        let neighbors = graph.neighbors(current)?;
        if neighbors.is_empty() {
            walk.push(current);
            continue;
        }
        // Compute transition weights
        let prev_neighbors: std::collections::HashSet<usize> =
            graph.neighbors(prev)?.iter().copied().collect();

        let weights: Vec<f32> = neighbors
            .iter()
            .map(|&nb| {
                if nb == prev {
                    1.0 / p
                } else if prev_neighbors.contains(&nb) {
                    1.0
                } else {
                    1.0 / q
                }
            })
            .collect();

        let next = weighted_sample(neighbors, &weights, rng);
        walk.push(next);
        prev = current;
        current = next;
    }
    Ok(walk)
}

/// Sample one element from `items` with probability proportional to `weights`.
fn weighted_sample(items: &[usize], weights: &[f32], rng: &mut LcgRng) -> usize {
    let total: f32 = weights.iter().sum();
    let mut r = rng.next_f32() * total;
    for (&item, &w) in items.iter().zip(weights.iter()) {
        r -= w;
        if r <= 0.0 {
            return item;
        }
    }
    *items.last().unwrap_or(&0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    fn complete_graph(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn sampler_new_empty_hops_error() {
        let err = NeighborhoodSampler::new(vec![]);
        assert!(err.is_err());
    }

    #[test]
    fn sampler_new_zero_fanout_error() {
        let err = NeighborhoodSampler::new(vec![5, 0]);
        assert!(err.is_err());
    }

    #[test]
    fn sample_1hop_complete_graph() {
        let g = complete_graph(5);
        let sampler =
            NeighborhoodSampler::new(vec![2]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(42);
        let result = sampler
            .sample(&g, &[0], &mut rng)
            .expect("test invariant: value must be valid");
        // seed + at most 2 neighbors
        assert!(result.n_nodes() >= 1);
        assert!(result.n_nodes() <= 3);
    }

    #[test]
    fn sample_seed_nodes_included() {
        let g = complete_graph(5);
        let sampler =
            NeighborhoodSampler::new(vec![3]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(1);
        let result = sampler
            .sample(&g, &[0, 2], &mut rng)
            .expect("test invariant: value must be valid");
        // Both seeds must appear
        assert!(result.local_to_global.contains(&0));
        assert!(result.local_to_global.contains(&2));
    }

    #[test]
    fn sample_isolated_node_no_edges() {
        // Node 3 has no outgoing edges
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        let sampler =
            NeighborhoodSampler::new(vec![2]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(7);
        let result = sampler
            .sample(&g, &[3], &mut rng)
            .expect("test invariant: value must be valid");
        assert_eq!(result.seed_nodes, vec![3]);
        assert_eq!(result.n_edges(), 0);
    }

    #[test]
    fn sample_2hop() {
        let g = chain_graph(6); // 0→1→2→3→4→5
        let sampler =
            NeighborhoodSampler::new(vec![1, 1]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(99);
        let result = sampler
            .sample(&g, &[0], &mut rng)
            .expect("test invariant: value must be valid");
        // Should include 0, 1, 2
        assert!(result.n_nodes() >= 1);
    }

    #[test]
    fn random_walk_length() {
        let g = complete_graph(5);
        let mut rng = LcgRng::new(55);
        let walk = random_walk(&g, 0, 10, &mut rng).expect("test invariant: value must be valid");
        assert_eq!(walk.len(), 11); // start + 10 steps
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn random_walk_all_nodes_valid() {
        let g = complete_graph(5);
        let mut rng = LcgRng::new(42);
        let walk = random_walk(&g, 2, 50, &mut rng).expect("test invariant: value must be valid");
        for &n in &walk {
            assert!(n < 5);
        }
    }

    #[test]
    fn random_walk_isolated_node_stays_put() {
        let g = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(1);
        let walk = random_walk(&g, 2, 5, &mut rng).expect("test invariant: value must be valid");
        // Node 2 has no outgoing edges; walk stays at 2
        assert!(walk.iter().all(|&n| n == 2));
    }

    #[test]
    fn biased_walk_length() {
        let g = complete_graph(6);
        let mut rng = LcgRng::new(13);
        let walk =
            biased_walk(&g, 0, 8, 1.0, 1.0, &mut rng).expect("test invariant: value must be valid");
        assert_eq!(walk.len(), 9); // start + 8 steps
    }

    #[test]
    fn biased_walk_invalid_p() {
        let g = complete_graph(3);
        let mut rng = LcgRng::new(1);
        let err = biased_walk(&g, 0, 5, 0.0, 1.0, &mut rng);
        assert!(err.is_err());
    }

    #[test]
    fn sample_k_from_slice_all_when_k_large() {
        let mut rng = LcgRng::new(1);
        let data = vec![10usize, 20, 30, 40, 50];
        let result = sample_k_from_slice(&data, 10, &mut rng);
        assert_eq!(result.len(), 5);
    }
}
