//! Cluster-GCN — "Cluster-GCN: An Efficient Algorithm for Training Deep and
//! Large Graph Convolutional Networks" (Chiang et al., KDD 2019).
//!
//! Cluster-GCN makes full-batch GCN training scale to large graphs by exploiting
//! graph clustering. The graph is partitioned **once** into `c` densely-connected
//! clusters; a stochastic training step then draws `q` clusters, unions their
//! nodes into a batch, and restricts message passing to the **batch subgraph** —
//! every edge whose two endpoints both fall inside the batch is kept, and every
//! inter-batch edge is dropped. A standard GCN forward then runs on this compact
//! subgraph. Because well-clustered partitions retain most edges *within*
//! clusters, the dropped inter-cluster edges are few and the per-batch
//! computation closely approximates full-graph propagation while costing a
//! fraction of the memory.
//!
//! Production Cluster-GCN uses METIS for the partition. METIS is out of scope
//! here; instead we provide a deterministic, dependency-free **balanced BFS
//! growth** partitioner: clusters are grown breadth-first from spread-out seeds
//! so the result is a disjoint cover with near-equal cluster sizes. The partition
//! quality only affects how many edges survive a batch — never correctness — so a
//! lightweight heuristic is sufficient for the CPU simulation path.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::handle::LcgRng;

/// Type alias for the RNG used when assembling a training batch.
pub type GnnRng = LcgRng;

// ─── Partition ────────────────────────────────────────────────────────────────

/// A disjoint clustering of the graph nodes into `n_clusters` parts.
///
/// `assignment[v]` is the cluster id of node `v`; `clusters[c]` lists the
/// (ascending) global node ids assigned to cluster `c`. Together they form a
/// disjoint cover: every node appears in exactly one cluster.
#[derive(Debug, Clone)]
pub struct Partition {
    n_nodes: usize,
    n_clusters: usize,
    assignment: Vec<usize>,
    clusters: Vec<Vec<usize>>,
}

impl Partition {
    /// Cluster id of node `v`.
    #[inline]
    pub fn cluster_of(&self, v: usize) -> usize {
        self.assignment[v]
    }

    /// Node ids belonging to cluster `c` (ascending).
    #[inline]
    pub fn cluster(&self, c: usize) -> &[usize] {
        &self.clusters[c]
    }

    /// Number of clusters.
    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Number of nodes covered.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Per-cluster sizes.
    #[must_use]
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.clusters.iter().map(Vec::len).collect()
    }

    /// Full assignment vector (`assignment[v]` = cluster of `v`).
    #[inline]
    pub fn assignment(&self) -> &[usize] {
        &self.assignment
    }
}

// ─── Batch subgraph ───────────────────────────────────────────────────────────

/// The induced subgraph over a batch of clusters.
///
/// `nodes` lists the batch's global node ids in **ascending** order; the batch
/// CSR graph uses local indices `0..nodes.len()` aligned to that order. Only
/// intra-batch edges are retained.
#[derive(Debug, Clone)]
pub struct BatchSubgraph {
    /// Global node ids of the batch, ascending. `nodes[local] = global`.
    pub nodes: Vec<usize>,
    /// Induced CSR subgraph in local-index space.
    pub graph: CsrGraph,
    /// Batch node features `[nodes.len() × feat_dim]`, gathered from the input.
    pub features: Vec<f32>,
}

impl BatchSubgraph {
    /// Number of nodes in the batch.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Map a global node id to its local batch index, if present.
    #[must_use]
    pub fn global_to_local(&self, global: usize) -> Option<usize> {
        // `nodes` is sorted ascending; binary search keeps lookup cheap.
        self.nodes.binary_search(&global).ok()
    }
}

// ─── Cluster-GCN ──────────────────────────────────────────────────────────────

/// Cluster-GCN trainer: owns the partition and assembles per-batch subgraphs.
pub struct ClusterGcn {
    n_clusters: usize,
    partition: Partition,
}

impl ClusterGcn {
    /// Partition `graph` into `n_clusters` balanced clusters via deterministic
    /// balanced BFS growth.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `n_clusters == 0` or
    ///   `n_clusters > n_nodes`.
    pub fn new(graph: &CsrGraph, n_clusters: usize) -> GnnResult<Self> {
        if n_clusters == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "Cluster-GCN: n_clusters must be >= 1".to_string(),
            ));
        }
        if n_clusters > graph.n_nodes() {
            return Err(GnnError::InvalidLayerConfig(format!(
                "Cluster-GCN: n_clusters {} exceeds n_nodes {}",
                n_clusters,
                graph.n_nodes()
            )));
        }
        let partition = balanced_bfs_partition(graph, n_clusters)?;
        Ok(Self {
            n_clusters,
            partition,
        })
    }

    /// Read-only access to the computed partition.
    #[inline]
    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Number of clusters.
    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Assemble the batch subgraph from a fixed set of cluster ids (no RNG).
    ///
    /// Useful for deterministic enumeration (e.g. the whole-graph batch). The
    /// cluster ids are deduplicated; the batch node set is the union of those
    /// clusters' nodes, sorted ascending.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `cluster_ids` is empty.
    /// - [`GnnError::NodeIndexOutOfRange`] if any cluster id is out of range.
    /// - [`GnnError::NodeFeatureMismatch`] if `features` is the wrong length.
    pub fn batch_from_clusters(
        &self,
        graph: &CsrGraph,
        features: &[f32],
        feat_dim: usize,
        cluster_ids: &[usize],
    ) -> GnnResult<BatchSubgraph> {
        if cluster_ids.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "Cluster-GCN: batch must contain >= 1 cluster".to_string(),
            ));
        }
        let n = graph.n_nodes();
        if features.len() != n * feat_dim {
            return Err(GnnError::NodeFeatureMismatch(
                n,
                features.len() / feat_dim.max(1),
            ));
        }
        // Membership mask over global node ids.
        let mut in_batch = vec![false; n];
        for &c in cluster_ids {
            if c >= self.n_clusters {
                return Err(GnnError::NodeIndexOutOfRange {
                    idx: c,
                    n_nodes: self.n_clusters,
                });
            }
            for &v in self.partition.cluster(c) {
                in_batch[v] = true;
            }
        }
        self.build_subgraph(graph, features, feat_dim, &in_batch)
    }

    /// Draw `q` distinct clusters uniformly at random and assemble the batch.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `q == 0` or `q > n_clusters`.
    /// - Propagates errors from [`ClusterGcn::batch_from_clusters`].
    pub fn sample_batch(
        &self,
        graph: &CsrGraph,
        features: &[f32],
        feat_dim: usize,
        q: usize,
        rng: &mut GnnRng,
    ) -> GnnResult<BatchSubgraph> {
        if q == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "Cluster-GCN: q must be >= 1".to_string(),
            ));
        }
        if q > self.n_clusters {
            return Err(GnnError::InvalidLayerConfig(format!(
                "Cluster-GCN: q {} exceeds n_clusters {}",
                q, self.n_clusters
            )));
        }
        // Sample q distinct cluster ids without replacement (partial shuffle).
        let mut ids: Vec<usize> = (0..self.n_clusters).collect();
        for i in 0..q {
            let j = i + rng.next_usize(self.n_clusters - i);
            ids.swap(i, j);
        }
        let chosen = &ids[..q];
        self.batch_from_clusters(graph, features, feat_dim, chosen)
    }

    /// Build the induced subgraph for a membership mask, keeping only edges whose
    /// **both** endpoints lie in the batch.
    fn build_subgraph(
        &self,
        graph: &CsrGraph,
        features: &[f32],
        feat_dim: usize,
        in_batch: &[bool],
    ) -> GnnResult<BatchSubgraph> {
        let n = graph.n_nodes();
        // Ascending batch node list ⇒ deterministic local ordering.
        let mut nodes: Vec<usize> = Vec::new();
        for (v, &member) in in_batch.iter().enumerate() {
            if member {
                nodes.push(v);
            }
        }
        // global → local map.
        let mut global_to_local = vec![usize::MAX; n];
        for (local, &g) in nodes.iter().enumerate() {
            global_to_local[g] = local;
        }
        // Gather intra-batch edges in local space.
        let mut local_edges: Vec<(usize, usize)> = Vec::new();
        for (local_u, &u) in nodes.iter().enumerate() {
            for &v in graph.neighbors(u)? {
                if in_batch[v] {
                    let local_v = global_to_local[v];
                    local_edges.push((local_u, local_v));
                }
            }
        }
        let sub_graph = CsrGraph::from_edges(nodes.len().max(1), &local_edges)?;
        // Gather features in local order.
        let mut sub_feats = vec![0.0_f32; nodes.len() * feat_dim];
        for (local, &g) in nodes.iter().enumerate() {
            sub_feats[local * feat_dim..(local + 1) * feat_dim]
                .copy_from_slice(&features[g * feat_dim..(g + 1) * feat_dim]);
        }
        Ok(BatchSubgraph {
            nodes,
            graph: sub_graph,
            features: sub_feats,
        })
    }
}

// ─── Balanced BFS partition ───────────────────────────────────────────────────

/// Deterministic balanced BFS-growth clustering.
///
/// Seeds are placed at evenly-spaced node ids; clusters then grow in lock-step
/// breadth-first rounds (each round, every cluster claims one unassigned frontier
/// node) so sizes stay within one element of each other. Any nodes never reached
/// (disconnected components, or once a cluster's frontier is exhausted) are
/// assigned greedily to the currently-smallest cluster, preserving balance and
/// guaranteeing a full disjoint cover.
fn balanced_bfs_partition(graph: &CsrGraph, n_clusters: usize) -> GnnResult<Partition> {
    let n = graph.n_nodes();
    let mut assignment = vec![usize::MAX; n];
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    // Evenly spaced deterministic seeds.
    let mut frontiers: Vec<std::collections::VecDeque<usize>> =
        vec![std::collections::VecDeque::new(); n_clusters];
    for (c, cluster) in clusters.iter_mut().enumerate() {
        let seed = (c * n) / n_clusters; // in [0, n)
        if assignment[seed] == usize::MAX {
            assignment[seed] = c;
            cluster.push(seed);
            frontiers[c].push_back(seed);
        }
    }
    let mut remaining = n - clusters.iter().map(Vec::len).sum::<usize>();

    // Lock-step BFS rounds: each cluster claims at most one new node per round.
    while remaining > 0 {
        let mut progressed = false;
        for c in 0..n_clusters {
            if remaining == 0 {
                break;
            }
            // Pop frontier nodes until we find one with an unassigned neighbour.
            while let Some(&front) = frontiers[c].front() {
                // Find first unassigned neighbour of `front`.
                let mut next_node = None;
                for &nb in graph.neighbors(front)? {
                    if assignment[nb] == usize::MAX {
                        next_node = Some(nb);
                        break;
                    }
                }
                match next_node {
                    Some(nb) => {
                        assignment[nb] = c;
                        clusters[c].push(nb);
                        frontiers[c].push_back(nb);
                        remaining -= 1;
                        progressed = true;
                        break;
                    }
                    None => {
                        // `front` exhausted; drop it from the frontier.
                        frontiers[c].pop_front();
                    }
                }
            }
        }
        if !progressed {
            break; // no cluster could grow via BFS; fall through to greedy fill.
        }
    }

    // Greedy balanced fill for any nodes BFS could not reach (disconnected, or
    // empty-seed clusters). Always extend the smallest cluster to keep balance.
    if remaining > 0 {
        // Min-size cluster order via a simple repeated scan; n_clusters is small.
        for (v, slot) in assignment.iter_mut().enumerate() {
            if *slot == usize::MAX {
                let target = (0..n_clusters)
                    .min_by_key(|&c| clusters[c].len())
                    .unwrap_or(0);
                *slot = target;
                clusters[target].push(v);
            }
        }
    }

    // Sort each cluster's node ids ascending for deterministic downstream order.
    for cl in &mut clusters {
        cl.sort_unstable();
    }

    Ok(Partition {
        n_nodes: n,
        n_clusters,
        assignment,
        clusters,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::gcn::{GcnConfig, GcnLayer};

    fn ring(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            edges.push((i, j));
            edges.push((j, i));
        }
        CsrGraph::from_edges(n, &edges).expect("ring")
    }

    fn grid_like(n: usize) -> CsrGraph {
        // Path plus a few long-range edges to create non-trivial structure.
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push((i, i + 1));
            edges.push((i + 1, i));
        }
        for i in 0..n {
            let j = (i + 3) % n;
            edges.push((i, j));
            edges.push((j, i));
        }
        CsrGraph::from_edges(n, &edges).expect("grid")
    }

    fn feats(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = LcgRng::new(seed);
        (0..n * dim).map(|_| r.next_f32() * 2.0 - 1.0).collect()
    }

    // (a) partition is a DISJOINT cover: every node in exactly one cluster,
    //     union = all nodes.
    #[test]
    fn partition_is_disjoint_cover() {
        let g = grid_like(20);
        let cg = ClusterGcn::new(&g, 4).expect("cg");
        let part = cg.partition();
        // Each node counted exactly once across clusters.
        let mut seen = vec![0u32; 20];
        for c in 0..part.n_clusters() {
            for &v in part.cluster(c) {
                seen[v] += 1;
            }
        }
        assert!(
            seen.iter().all(|&s| s == 1),
            "not a disjoint cover: {seen:?}"
        );
        // assignment agrees with cluster membership.
        for c in 0..part.n_clusters() {
            for &v in part.cluster(c) {
                assert_eq!(part.cluster_of(v), c);
            }
        }
        // Total coverage.
        let total: usize = part.cluster_sizes().iter().sum();
        assert_eq!(total, 20);
    }

    // (b) cluster sizes are roughly balanced (max − min ≤ small bound).
    #[test]
    fn cluster_sizes_balanced() {
        let g = grid_like(21);
        let cg = ClusterGcn::new(&g, 3).expect("cg");
        let sizes = cg.partition().cluster_sizes();
        let max = *sizes.iter().max().expect("nonempty");
        let min = *sizes.iter().min().expect("nonempty");
        assert!(
            max - min <= 2,
            "unbalanced sizes {sizes:?} (max-min={})",
            max - min
        );
    }

    // (c) batch subgraph contains ONLY intra-batch edges: no edge crosses to an
    //     outside node, and every endpoint is in the batch.
    #[test]
    fn batch_subgraph_only_intra_batch_edges() {
        let g = grid_like(24);
        let dim = 3;
        let x = feats(24, dim, 9);
        let cg = ClusterGcn::new(&g, 6).expect("cg");
        let mut rng = LcgRng::new(42);
        let batch = cg.sample_batch(&g, &x, dim, 2, &mut rng).expect("batch");

        // Build a membership set from the batch nodes.
        let in_batch: std::collections::HashSet<usize> = batch.nodes.iter().copied().collect();

        // Every local edge must map to two global nodes that are both in batch,
        // and must correspond to a real edge of the original graph.
        let sub = &batch.graph;
        for lu in 0..sub.n_nodes() {
            let gu = batch.nodes[lu];
            for &lv in sub.neighbors(lu).expect("nbrs") {
                let gv = batch.nodes[lv];
                assert!(in_batch.contains(&gu));
                assert!(in_batch.contains(&gv));
                // (gu, gv) is a genuine edge of the original graph.
                let real = g.neighbors(gu).expect("real").contains(&gv);
                assert!(real, "fabricated edge ({gu},{gv})");
            }
        }
    }

    // (d) batch == WHOLE graph ⇒ Cluster-GCN forward EQUALS full-graph GCN
    //     forward (same outputs to ~1e-5).
    #[test]
    fn whole_graph_batch_equals_full_gcn() {
        let g = grid_like(18);
        let in_f = 4;
        let out_f = 5;
        let x = feats(18, in_f, 17);
        let w = feats(in_f * out_f, 1, 71);

        let cg = ClusterGcn::new(&g, 5).expect("cg");
        let all_clusters: Vec<usize> = (0..cg.n_clusters()).collect();
        let batch = cg
            .batch_from_clusters(&g, &x, in_f, &all_clusters)
            .expect("batch");

        // Whole-graph batch ⇒ nodes are exactly 0..n in order.
        assert_eq!(batch.n_nodes(), 18);
        assert!(batch.nodes.iter().enumerate().all(|(i, &v)| i == v));

        let layer = GcnLayer::new(GcnConfig {
            in_features: in_f,
            out_features: out_f,
            bias: false,
            normalize: true,
        })
        .expect("layer");

        let out_full = layer.forward(&g, &x, &w, None).expect("full");
        let out_batch = layer
            .forward(&batch.graph, &batch.features, &w, None)
            .expect("batch fwd");

        assert_eq!(out_full.len(), out_batch.len());
        for (a, b) in out_full.iter().zip(out_batch.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    // (e) deterministic under fixed seed.
    #[test]
    fn deterministic_under_fixed_seed() {
        let g = grid_like(24);
        let dim = 2;
        let x = feats(24, dim, 5);
        let cg = ClusterGcn::new(&g, 6).expect("cg");

        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let b1 = cg.sample_batch(&g, &x, dim, 3, &mut r1).expect("b1");
        let b2 = cg.sample_batch(&g, &x, dim, 3, &mut r2).expect("b2");
        assert_eq!(b1.nodes, b2.nodes);
        assert_eq!(b1.features, b2.features);
        assert_eq!(b1.graph.n_edges(), b2.graph.n_edges());
        assert_eq!(b1.graph.col_idx(), b2.graph.col_idx());

        // Partition itself is deterministic across constructions.
        let cg2 = ClusterGcn::new(&g, 6).expect("cg2");
        assert_eq!(cg.partition().assignment(), cg2.partition().assignment());
    }

    // (f) a node's batch-restricted neighbourhood ⊆ its true neighbourhood.
    #[test]
    fn batch_neighborhood_subset_of_true() {
        let g = grid_like(30);
        let dim = 2;
        let x = feats(30, dim, 8);
        let cg = ClusterGcn::new(&g, 5).expect("cg");
        let mut rng = LcgRng::new(77);
        let batch = cg.sample_batch(&g, &x, dim, 3, &mut rng).expect("batch");

        let sub = &batch.graph;
        for lu in 0..sub.n_nodes() {
            let gu = batch.nodes[lu];
            let true_nbrs: std::collections::HashSet<usize> =
                g.neighbors(gu).expect("true").iter().copied().collect();
            for &lv in sub.neighbors(lu).expect("sub") {
                let gv = batch.nodes[lv];
                assert!(
                    true_nbrs.contains(&gv),
                    "batch nbr {gv} of {gu} not a true neighbour"
                );
            }
        }
    }

    // Extra: single-cluster partition is the whole graph; q must be valid.
    #[test]
    fn validation_and_single_cluster() {
        let g = ring(8);
        assert!(ClusterGcn::new(&g, 0).is_err());
        assert!(ClusterGcn::new(&g, 9).is_err());

        let cg = ClusterGcn::new(&g, 1).expect("cg");
        assert_eq!(cg.partition().cluster(0).len(), 8);

        let x = feats(8, 2, 1);
        let mut rng = LcgRng::new(1);
        assert!(cg.sample_batch(&g, &x, 2, 0, &mut rng).is_err());
        assert!(cg.sample_batch(&g, &x, 2, 2, &mut rng).is_err()); // q > n_clusters
        // bad feature length
        assert!(cg.batch_from_clusters(&g, &x[..4], 2, &[0]).is_err());
    }
}
