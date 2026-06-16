//! GraphSAINT — "GraphSAINT: Graph Sampling Based Inductive Learning Method"
//! (Zeng et al., ICLR 2020).
//!
//! GraphSAINT trains a GNN on **sampled subgraphs** rather than the full graph,
//! and corrects the bias this introduces with normalisation coefficients
//! estimated from how often each node and edge appears across the sampled
//! subgraphs. The estimator is *unbiased*: the expected normalised aggregation of
//! a node over the sampling distribution equals the full-graph aggregation.
//!
//! Three samplers are provided (Zeng §3.2):
//!
//! * **Node sampler** — draw `budget` root nodes (with replacement, importance
//!   ∝ degree) and induce the subgraph on the distinct sampled nodes.
//! * **Edge sampler** — draw `budget` edges with probability proportional to
//!   `1/deg(u) + 1/deg(v)`; the subgraph is induced on the endpoints.
//! * **Random-walk sampler** — draw `n_roots` roots and grow a walk of
//!   `walk_length` steps from each; the visited nodes induce the subgraph.
//!
//! After `R` subgraphs are drawn the **normalisation coefficients** are formed
//! from the appearance counts:
//!
//! ```text
//!   C_v      = #subgraphs containing node v
//!   C_{u,v}  = #subgraphs containing edge (u,v)
//!   α_{u,v}  = C_{u,v} / C_v          (aggregator / edge norm)
//!   λ_v      = R / C_v   ( ∝ 1 / P(v) ) (loss / node norm)
//! ```
//!
//! and the per-subgraph aggregation of node `v` is de-biased by dividing every
//! incoming edge `(u,v)` by `α_{u,v}`:
//!
//! ```text
//!   ĥ_v = Σ_{(u,v) ∈ subgraph}  (1 / α_{u,v}) · x_u
//! ```
//!
//! whose expectation (conditioned on `v` being sampled) approaches the full-graph
//! aggregation `Σ_{u ∈ N(v)} x_u`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::handle::LcgRng;
use std::collections::HashMap;

/// Type alias for the RNG used by GraphSAINT samplers.
pub type GnnRng = LcgRng;

// ─── Sampled subgraph ─────────────────────────────────────────────────────────

/// A subgraph produced by one of the GraphSAINT samplers.
///
/// `nodes` holds the distinct sampled global node ids in ascending order; `edges`
/// holds the induced intra-subgraph edges as **global** `(src, dst)` pairs
/// (directed, mirroring the original graph's stored direction).
#[derive(Debug, Clone)]
pub struct SaintSubgraph {
    /// Distinct sampled global node ids, ascending.
    pub nodes: Vec<usize>,
    /// Induced edges as global `(src, dst)` pairs.
    pub edges: Vec<(usize, usize)>,
}

impl SaintSubgraph {
    /// Number of sampled nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of induced edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Whether global node `v` is present in this subgraph.
    #[must_use]
    pub fn contains_node(&self, v: usize) -> bool {
        self.nodes.binary_search(&v).is_ok()
    }
}

// ─── Sampler configuration ────────────────────────────────────────────────────

/// Which GraphSAINT sampler to use and its budget.
#[derive(Debug, Clone, Copy)]
pub enum SaintSampler {
    /// Node sampler with the given node budget.
    Node {
        /// Number of root nodes drawn per subgraph.
        budget: usize,
    },
    /// Edge sampler with the given edge budget.
    Edge {
        /// Number of edges drawn per subgraph.
        budget: usize,
    },
    /// Random-walk sampler: `n_roots` walkers of length `walk_length`.
    RandomWalk {
        /// Number of independent walk roots.
        n_roots: usize,
        /// Steps per walk.
        walk_length: usize,
    },
}

// ─── GraphSAINT engine ────────────────────────────────────────────────────────

/// GraphSAINT sampler + normalisation-coefficient estimator over a fixed graph.
pub struct GraphSaint<'g> {
    graph: &'g CsrGraph,
    /// Out-degree per node (number of stored neighbours).
    degree: Vec<usize>,
}

impl<'g> GraphSaint<'g> {
    /// Build a GraphSAINT engine over `graph`.
    ///
    /// # Errors
    ///
    /// [`GnnError::EmptyGraph`] if the graph has no nodes.
    pub fn new(graph: &'g CsrGraph) -> GnnResult<Self> {
        if graph.n_nodes() == 0 {
            return Err(GnnError::EmptyGraph);
        }
        let degree = graph.degrees();
        Ok(Self { graph, degree })
    }

    /// Draw a single subgraph with the given sampler.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] for a zero budget / length.
    pub fn sample(&self, sampler: SaintSampler, rng: &mut GnnRng) -> GnnResult<SaintSubgraph> {
        match sampler {
            SaintSampler::Node { budget } => self.sample_node(budget, rng),
            SaintSampler::Edge { budget } => self.sample_edge(budget, rng),
            SaintSampler::RandomWalk {
                n_roots,
                walk_length,
            } => self.sample_random_walk(n_roots, walk_length, rng),
        }
    }

    // ── Node sampler ──────────────────────────────────────────────────────────

    fn sample_node(&self, budget: usize, rng: &mut GnnRng) -> GnnResult<SaintSubgraph> {
        if budget == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT node sampler: budget must be >= 1".to_string(),
            ));
        }
        let n = self.graph.n_nodes();
        // Degree-importance sampling with replacement via a prefix-sum table.
        // Fall back to uniform when the graph is edgeless.
        let total_deg: usize = self.degree.iter().sum();
        let mut in_set = vec![false; n];
        for _ in 0..budget {
            let v = if total_deg == 0 {
                rng.next_usize(n)
            } else {
                let target = rng.next_usize(total_deg);
                let mut acc = 0usize;
                let mut pick = n - 1;
                for (i, &d) in self.degree.iter().enumerate() {
                    acc += d;
                    if target < acc {
                        pick = i;
                        break;
                    }
                }
                pick
            };
            in_set[v] = true;
        }
        self.induce(&in_set)
    }

    // ── Edge sampler ──────────────────────────────────────────────────────────

    fn sample_edge(&self, budget: usize, rng: &mut GnnRng) -> GnnResult<SaintSubgraph> {
        if budget == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT edge sampler: budget must be >= 1".to_string(),
            ));
        }
        let n = self.graph.n_nodes();
        // Flatten edges and build importance weights p(u,v) ∝ 1/deg(u)+1/deg(v).
        let mut edge_list: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        for u in 0..n {
            let du = self.degree[u].max(1) as f32;
            for &v in self.graph.neighbors(u)? {
                let dv = self.degree[v].max(1) as f32;
                edge_list.push((u, v));
                weights.push(1.0 / du + 1.0 / dv);
            }
        }
        let mut in_set = vec![false; n];
        if edge_list.is_empty() {
            // No edges: sample isolated nodes uniformly so the subgraph is valid.
            for _ in 0..budget {
                in_set[rng.next_usize(n)] = true;
            }
            return self.induce(&in_set);
        }
        let total_w: f32 = weights.iter().sum();
        for _ in 0..budget {
            let target = rng.next_f32() * total_w;
            let mut acc = 0.0_f32;
            let mut pick = edge_list.len() - 1;
            for (i, &w) in weights.iter().enumerate() {
                acc += w;
                if target < acc {
                    pick = i;
                    break;
                }
            }
            let (u, v) = edge_list[pick];
            in_set[u] = true;
            in_set[v] = true;
        }
        self.induce(&in_set)
    }

    // ── Random-walk sampler ─────────────────────────────────────────────────────

    fn sample_random_walk(
        &self,
        n_roots: usize,
        walk_length: usize,
        rng: &mut GnnRng,
    ) -> GnnResult<SaintSubgraph> {
        if n_roots == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT random-walk sampler: n_roots must be >= 1".to_string(),
            ));
        }
        if walk_length == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT random-walk sampler: walk_length must be >= 1".to_string(),
            ));
        }
        let n = self.graph.n_nodes();
        let mut in_set = vec![false; n];
        for _ in 0..n_roots {
            let mut current = rng.next_usize(n);
            in_set[current] = true;
            for _ in 0..walk_length {
                let nbrs = self.graph.neighbors(current)?;
                if nbrs.is_empty() {
                    break; // dead end; stop this walk early.
                }
                current = nbrs[rng.next_usize(nbrs.len())];
                in_set[current] = true;
            }
        }
        self.induce(&in_set)
    }

    /// Produce a single random walk (its visited node sequence) of the requested
    /// length from a uniformly-chosen root. The walk stops early on a dead end.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `walk_length == 0`.
    pub fn random_walk_path(&self, walk_length: usize, rng: &mut GnnRng) -> GnnResult<Vec<usize>> {
        if walk_length == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT random walk: walk_length must be >= 1".to_string(),
            ));
        }
        let n = self.graph.n_nodes();
        let mut path = Vec::with_capacity(walk_length + 1);
        let mut current = rng.next_usize(n);
        path.push(current);
        for _ in 0..walk_length {
            let nbrs = self.graph.neighbors(current)?;
            if nbrs.is_empty() {
                break;
            }
            current = nbrs[rng.next_usize(nbrs.len())];
            path.push(current);
        }
        Ok(path)
    }

    /// Induce the subgraph on a node-membership mask, keeping edges whose both
    /// endpoints are in the set.
    fn induce(&self, in_set: &[bool]) -> GnnResult<SaintSubgraph> {
        let mut nodes = Vec::new();
        for (v, &m) in in_set.iter().enumerate() {
            if m {
                nodes.push(v);
            }
        }
        let mut edges = Vec::new();
        for &u in &nodes {
            for &v in self.graph.neighbors(u)? {
                if in_set[v] {
                    edges.push((u, v));
                }
            }
        }
        Ok(SaintSubgraph { nodes, edges })
    }

    // ── Normalisation coefficients ─────────────────────────────────────────────

    /// Estimate GraphSAINT normalisation coefficients over `n_subgraphs` draws.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `n_subgraphs == 0` or the sampler is
    ///   mis-configured.
    pub fn estimate_norm(
        &self,
        sampler: SaintSampler,
        n_subgraphs: usize,
        rng: &mut GnnRng,
    ) -> GnnResult<SaintNorm> {
        if n_subgraphs == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphSAINT: n_subgraphs must be >= 1".to_string(),
            ));
        }
        let n = self.graph.n_nodes();
        let mut node_count = vec![0u64; n];
        let mut edge_count: HashMap<(usize, usize), u64> = HashMap::new();
        for _ in 0..n_subgraphs {
            let sg = self.sample(sampler, rng)?;
            for &v in &sg.nodes {
                node_count[v] += 1;
            }
            for &e in &sg.edges {
                *edge_count.entry(e).or_insert(0) += 1;
            }
        }
        Ok(SaintNorm {
            n_subgraphs: n_subgraphs as u64,
            node_count,
            edge_count,
        })
    }
}

// ─── Normalisation coefficients ───────────────────────────────────────────────

/// Appearance counts and the derived GraphSAINT normalisation coefficients.
#[derive(Debug, Clone)]
pub struct SaintNorm {
    n_subgraphs: u64,
    /// `node_count[v]` = number of subgraphs containing node `v`.
    node_count: Vec<u64>,
    /// `edge_count[(u,v)]` = number of subgraphs containing edge `(u,v)`.
    edge_count: HashMap<(usize, usize), u64>,
}

impl SaintNorm {
    /// Number of subgraphs `R` used to build these statistics.
    #[inline]
    pub fn n_subgraphs(&self) -> u64 {
        self.n_subgraphs
    }

    /// Raw node appearance count `C_v`.
    #[inline]
    pub fn node_count(&self, v: usize) -> u64 {
        self.node_count[v]
    }

    /// Raw edge appearance count `C_{u,v}`.
    #[must_use]
    pub fn edge_count(&self, u: usize, v: usize) -> u64 {
        self.edge_count.get(&(u, v)).copied().unwrap_or(0)
    }

    /// Aggregator / edge normalisation `α_{u,v} = C_{u,v} / C_v`.
    ///
    /// Returns `0.0` if the destination `v` was never sampled.
    #[must_use]
    pub fn alpha(&self, u: usize, v: usize) -> f32 {
        let cv = self.node_count[v];
        if cv == 0 {
            return 0.0;
        }
        self.edge_count(u, v) as f32 / cv as f32
    }

    /// Loss / node normalisation `λ_v = R / C_v` (∝ 1 / P(v)).
    ///
    /// Returns `0.0` if the node `v` was never sampled.
    #[must_use]
    pub fn lambda(&self, v: usize) -> f32 {
        let cv = self.node_count[v];
        if cv == 0 {
            return 0.0;
        }
        self.n_subgraphs as f32 / cv as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            edges.push((i, j));
            edges.push((j, i));
        }
        CsrGraph::from_edges(n, &edges).expect("ring")
    }

    fn complete(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        CsrGraph::from_edges(n, &edges).expect("complete")
    }

    fn feats(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = LcgRng::new(seed);
        (0..n * dim).map(|_| r.next_f32() * 2.0 - 1.0).collect()
    }

    /// Full-graph aggregation of a destination node: Σ_{u∈N(v)} x_u.
    fn full_aggregate(g: &CsrGraph, x: &[f32], dim: usize, v: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; dim];
        for &u in g.neighbors(v).expect("nbrs") {
            for k in 0..dim {
                out[k] += x[u * dim + k];
            }
        }
        out
    }

    // (a) sampled-subgraph node/edge counts stay within the configured budget.
    #[test]
    fn budgets_respected() {
        let g = complete(12);
        let saint = GraphSaint::new(&g).expect("saint");
        let mut rng = LcgRng::new(1);

        // Node sampler: distinct nodes ≤ budget.
        let sg = saint
            .sample(SaintSampler::Node { budget: 5 }, &mut rng)
            .expect("node");
        assert!(sg.n_nodes() <= 5, "node count {} > budget", sg.n_nodes());

        // Edge sampler: distinct endpoint nodes ≤ 2*budget.
        let sg = saint
            .sample(SaintSampler::Edge { budget: 4 }, &mut rng)
            .expect("edge");
        assert!(
            sg.n_nodes() <= 8,
            "edge endpoints {} > 2*budget",
            sg.n_nodes()
        );

        // Random walk: distinct nodes ≤ n_roots*(walk_length+1).
        let sg = saint
            .sample(
                SaintSampler::RandomWalk {
                    n_roots: 2,
                    walk_length: 3,
                },
                &mut rng,
            )
            .expect("rw");
        assert!(sg.n_nodes() <= 2 * 4, "rw nodes {} too many", sg.n_nodes());
    }

    // (b) the normalisation coefficient for an edge equals
    //     (appearance count)/(endpoint appearance count).
    #[test]
    fn alpha_equals_edge_over_node_count() {
        let g = ring(10);
        let saint = GraphSaint::new(&g).expect("saint");
        let mut rng = LcgRng::new(2);
        let norm = saint
            .estimate_norm(SaintSampler::Edge { budget: 6 }, 200, &mut rng)
            .expect("norm");
        // Check the definition holds for several edges that were actually seen.
        let mut checked = 0;
        for u in 0..10 {
            for &v in g.neighbors(u).expect("nbrs") {
                let cv = norm.node_count(v);
                let cuv = norm.edge_count(u, v);
                let alpha = norm.alpha(u, v);
                if cv > 0 {
                    let expected = cuv as f32 / cv as f32;
                    assert!(
                        (alpha - expected).abs() < 1e-6,
                        "α mismatch {alpha} vs {expected}"
                    );
                    if cuv > 0 {
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no edges were sampled to validate α");
    }

    // (c) UNBIASEDNESS — the NORMALISED aggregate of a fixed node, averaged over
    //     many subgraphs, approaches the full-graph aggregate; the UN-normalised
    //     average does NOT.
    #[test]
    fn normalized_aggregate_is_unbiased() {
        let g = ring(8);
        let dim = 3;
        let x = feats(8, dim, 123);
        let target = 0usize; // node whose aggregation we estimate
        let r = 6000;
        let sampler = SaintSampler::Edge { budget: 5 };

        let saint = GraphSaint::new(&g).expect("saint");

        // First pass: estimate normalisation counts.
        let mut rng_norm = LcgRng::new(777);
        let norm = saint
            .estimate_norm(sampler, r, &mut rng_norm)
            .expect("norm");

        // Second pass: Monte-Carlo average of the (normalised and raw) per-
        // subgraph aggregation of `target`, over subgraphs containing it.
        let mut rng_mc = LcgRng::new(777); // same seed ⇒ same subgraph stream
        let mut sum_norm = vec![0.0_f32; dim];
        let mut sum_raw = vec![0.0_f32; dim];
        let mut hits = 0u64;
        for _ in 0..r {
            let sg = saint.sample(sampler, &mut rng_mc).expect("sg");
            if !sg.contains_node(target) {
                continue;
            }
            hits += 1;
            for &(u, v) in &sg.edges {
                if v != target {
                    continue;
                }
                let alpha = norm.alpha(u, v);
                for k in 0..dim {
                    sum_raw[k] += x[u * dim + k];
                    if alpha > 0.0 {
                        sum_norm[k] += x[u * dim + k] / alpha;
                    }
                }
            }
        }
        assert!(hits > 100, "target sampled too rarely ({hits})");

        let full = full_aggregate(&g, &x, dim, target);
        let est_norm: Vec<f32> = sum_norm.iter().map(|&s| s / hits as f32).collect();
        let est_raw: Vec<f32> = sum_raw.iter().map(|&s| s / hits as f32).collect();

        // Normalised estimator is close to the full aggregation.
        let err_norm: f32 = est_norm
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dim as f32;
        // Raw estimator is biased (systematically too small: edges sampled with
        // prob < 1 are not re-weighted).
        let err_raw: f32 = est_raw
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dim as f32;

        assert!(
            err_norm < 0.20,
            "normalised estimator not unbiased: err={err_norm}, est={est_norm:?}, full={full:?}"
        );
        assert!(
            err_raw > err_norm,
            "raw estimator should be more biased than normalised (raw={err_raw}, norm={err_norm})"
        );
    }

    // (d) the random-walk sampler yields connected walks of the requested length.
    #[test]
    fn random_walk_is_connected() {
        let g = ring(12); // every node has degree 2 ⇒ no dead ends.
        let saint = GraphSaint::new(&g).expect("saint");
        let mut rng = LcgRng::new(9);
        let len = 6;
        for _ in 0..50 {
            let path = saint.random_walk_path(len, &mut rng).expect("walk");
            // On a ring (no dead ends) the walk reaches full requested length.
            assert_eq!(path.len(), len + 1, "walk truncated: {path:?}");
            // Consecutive nodes are adjacent in the original graph.
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                assert!(
                    g.neighbors(a).expect("nbrs").contains(&b),
                    "walk step {a}->{b} not an edge"
                );
            }
        }
    }

    // (e) deterministic under fixed seed.
    #[test]
    fn deterministic_under_fixed_seed() {
        let g = complete(10);
        let saint = GraphSaint::new(&g).expect("saint");
        let sampler = SaintSampler::Node { budget: 4 };

        let mut r1 = LcgRng::new(555);
        let mut r2 = LcgRng::new(555);
        let a = saint.sample(sampler, &mut r1).expect("a");
        let b = saint.sample(sampler, &mut r2).expect("b");
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.edges, b.edges);

        let mut rn1 = LcgRng::new(11);
        let mut rn2 = LcgRng::new(11);
        let n1 = saint.estimate_norm(sampler, 50, &mut rn1).expect("n1");
        let n2 = saint.estimate_norm(sampler, 50, &mut rn2).expect("n2");
        for v in 0..10 {
            assert_eq!(n1.node_count(v), n2.node_count(v));
            assert!((n1.lambda(v) - n2.lambda(v)).abs() < 1e-9);
        }
    }

    // (f) every sampled node/edge belongs to the original graph.
    #[test]
    fn sampled_elements_belong_to_graph() {
        let g = ring(15);
        let n = g.n_nodes();
        let saint = GraphSaint::new(&g).expect("saint");
        let mut rng = LcgRng::new(31);
        for sampler in [
            SaintSampler::Node { budget: 5 },
            SaintSampler::Edge { budget: 5 },
            SaintSampler::RandomWalk {
                n_roots: 2,
                walk_length: 4,
            },
        ] {
            let sg = saint.sample(sampler, &mut rng).expect("sg");
            for &v in &sg.nodes {
                assert!(v < n, "node {v} out of range");
            }
            for &(u, v) in &sg.edges {
                assert!(u < n && v < n, "edge ({u},{v}) out of range");
                // Edge must be a real edge of the original graph.
                assert!(
                    g.neighbors(u).expect("nbrs").contains(&v),
                    "fabricated edge ({u},{v})"
                );
            }
            // induced edges only connect sampled nodes.
            let set: std::collections::HashSet<usize> = sg.nodes.iter().copied().collect();
            for &(u, v) in &sg.edges {
                assert!(set.contains(&u) && set.contains(&v));
            }
        }
    }

    // Extra: λ_v = R / C_v and validation errors.
    #[test]
    fn lambda_and_validation() {
        let g = ring(6);
        let saint = GraphSaint::new(&g).expect("saint");
        let mut rng = LcgRng::new(3);
        let norm = saint
            .estimate_norm(SaintSampler::Node { budget: 3 }, 100, &mut rng)
            .expect("norm");
        for v in 0..6 {
            let cv = norm.node_count(v);
            if cv > 0 {
                assert!((norm.lambda(v) - 100.0 / cv as f32).abs() < 1e-6);
            }
        }
        // Validation.
        assert!(
            saint
                .sample(SaintSampler::Node { budget: 0 }, &mut rng)
                .is_err()
        );
        assert!(
            saint
                .sample(SaintSampler::Edge { budget: 0 }, &mut rng)
                .is_err()
        );
        assert!(
            saint
                .sample(
                    SaintSampler::RandomWalk {
                        n_roots: 0,
                        walk_length: 3
                    },
                    &mut rng
                )
                .is_err()
        );
        assert!(
            saint
                .estimate_norm(SaintSampler::Node { budget: 2 }, 0, &mut rng)
                .is_err()
        );
    }
}
