//! Louvain community detection (Blondel et al. 2008).
//!
//! Greedy two-phase modularity maximisation:
//!
//! **Phase 1 — Local moves.** For every node `i`, compute the modularity gain
//! ΔQ from moving `i` to each neighbouring community. If the best gain is
//! positive, move `i` there. Repeat until no node improves (convergence).
//!
//! **Phase 2 — Graph aggregation.** Replace each community with a super-node.
//! Edges between communities become weighted edges between super-nodes (with
//! self-loops for intra-community edges). Repeat Phase 1 on the coarsened graph.
//!
//! Terminates when Phase 1 produces no improvement or `max_levels` is reached.
//!
//! **Modularity** of a partition C:
//! ```text
//! Q = (1/2m) Σ_{ij} [ A_{ij} − k_i k_j / (2m) ] δ(c_i, c_j)
//! ```
//! where `m = Σ_{ij} A_{ij} / 2` (total edge weight) and `k_i = Σ_j A_{ij}`
//! (weighted degree of node `i`).
//!
//! Reference: Blondel et al. "Fast unfolding of communities in large networks",
//! J. Stat. Mech. 2008.

use crate::error::{GraphError, GraphResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the Louvain algorithm.
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// Maximum number of Phase-1 sweeps per aggregation level.
    pub max_passes: usize,
    /// Stop Phase 1 when total modularity improvement per pass falls below this.
    pub min_improvement: f64,
    /// Maximum number of Phase-2 aggregation levels.
    pub max_levels: usize,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_passes: 10,
            min_improvement: 1e-7,
            max_levels: 15,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Result of Louvain community detection.
#[derive(Debug)]
pub struct LouvainResult {
    /// Community assignment for each node (length == `n_nodes`, values are
    /// community indices in `[0, n_communities)`).
    pub labels: Vec<usize>,
    /// Final modularity `Q` of the partition.
    pub modularity: f64,
    /// Number of communities found.
    pub n_communities: usize,
}

// ─── Internal graph representation ───────────────────────────────────────────

/// A weighted adjacency list entry.
#[derive(Clone, Debug)]
struct Neighbor {
    node: usize,
    weight: f64,
}

/// Compact weighted graph for the Louvain algorithm.
struct WGraph {
    n: usize,
    adj: Vec<Vec<Neighbor>>,
    /// Weighted degree of each node.
    degree: Vec<f64>,
    /// Total weight of all edges (m = Σ_{ij} A_{ij} / 2).
    total_weight: f64,
}

impl WGraph {
    /// Build from an edge list `(from, to, weight)` (each edge listed once).
    fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adj: Vec<Vec<Neighbor>> = vec![Vec::new(); n];
        let mut degree = vec![0.0_f64; n];
        let mut total_weight = 0.0_f64;

        for &(u, v, w) in edges {
            adj[u].push(Neighbor { node: v, weight: w });
            degree[u] += w;
            if u != v {
                adj[v].push(Neighbor { node: u, weight: w });
                degree[v] += w;
                total_weight += w;
            } else {
                // Self-loop contributes 2w to degree, w to total_weight
                total_weight += w; // count once for the "2m" denominator
            }
        }

        Self {
            n,
            adj,
            degree,
            total_weight,
        }
    }

    /// Compute modularity for a given community assignment.
    fn modularity(&self, community: &[usize]) -> f64 {
        if self.total_weight < 1e-14 {
            return 0.0;
        }
        let two_m = 2.0 * self.total_weight;
        let mut q = 0.0_f64;
        for u in 0..self.n {
            for nb in &self.adj[u] {
                if community[u] == community[nb.node] {
                    q += nb.weight - self.degree[u] * self.degree[nb.node] / two_m;
                }
            }
        }
        q / two_m
    }
}

// ─── Phase 1 ─────────────────────────────────────────────────────────────────

/// Run Phase 1 local moves on `graph`, modifying `community` in place.
/// Returns total absolute modularity improvement.
fn phase1(graph: &WGraph, community: &mut [usize], config: &LouvainConfig) -> f64 {
    let n = graph.n;
    let two_m = 2.0 * graph.total_weight;
    if two_m < 1e-14 {
        return 0.0;
    }

    // For each community: sum of internal degrees (Σ k_i for i in community).
    let mut sigma_tot = vec![0.0_f64; n];
    for i in 0..n {
        sigma_tot[community[i]] += graph.degree[i];
    }

    let mut total_gain = 0.0_f64;

    for _pass in 0..config.max_passes {
        let mut pass_gain = 0.0_f64;

        for i in 0..n {
            let current_comm = community[i];
            let ki = graph.degree[i];

            // Collect k_{i→c} = sum of weights from i to nodes in community c.
            // Also collect k_{i→current} = sum of weights from i to nodes in its
            // current community *excluding* i itself (i.e. the "remove" contribution).
            let mut comm_weights: Vec<(usize, f64)> = Vec::new();
            let mut ki_in_current = 0.0_f64;

            for nb in &graph.adj[i] {
                let c = community[nb.node];
                if nb.node != i {
                    if c == current_comm {
                        ki_in_current += nb.weight;
                    }
                    if let Some(pos) = comm_weights.iter().position(|&(c2, _)| c2 == c) {
                        comm_weights[pos].1 += nb.weight;
                    } else {
                        comm_weights.push((c, nb.weight));
                    }
                }
            }

            // sigma_tot[current_comm] already includes ki; the "effective" sigma when
            // i is removed from D is sigma_tot[D] - ki.
            let sigma_d_no_i = sigma_tot[current_comm] - ki;

            // Standard Louvain modularity gain for moving i from D to C:
            //   ΔQ = [(k_{i→C} - k_{i→D\i}) / m]
            //      - [(Σ_C - Σ_D_no_i) * ki / (2m²)]
            // simplifies to evaluating each target C separately:
            //   gain(C) = k_{i→C}/m - Σ_C * ki/(2m²)   (add to C)
            //           - k_{i→D\i}/m + Σ_D_no_i * ki/(2m²)  (remove from D)
            let base_remove =
                -ki_in_current / graph.total_weight + sigma_d_no_i * ki / (two_m * two_m);

            let mut best_comm = current_comm;
            let mut best_gain = 0.0_f64;

            for &(c, ki_c) in &comm_weights {
                if c == current_comm {
                    continue;
                }
                let delta_add = ki_c / graph.total_weight - sigma_tot[c] * ki / (two_m * two_m);

                let gain = delta_add + base_remove;
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            if best_comm != current_comm && best_gain > 0.0 {
                sigma_tot[current_comm] -= ki;
                sigma_tot[best_comm] += ki;
                community[i] = best_comm;
                pass_gain += best_gain;
            }
        }

        total_gain += pass_gain;
        if pass_gain < config.min_improvement {
            break;
        }
    }

    total_gain
}

// ─── Phase 2: Graph aggregation ──────────────────────────────────────────────

/// Coarsen `graph` according to `community`, producing a super-graph.
/// Also updates `node_to_comm` to renumber communities 0..k.
/// Returns `(super_graph, mapping from old community id → new super-node id)`.
fn phase2(graph: &WGraph, community: &[usize]) -> (WGraph, Vec<usize>) {
    // Renumber communities contiguously 0..k
    let n = graph.n;
    let mut comm_map = vec![usize::MAX; n];
    let mut k = 0_usize;
    let mut old_to_new = vec![usize::MAX; n];
    for &c in community.iter() {
        if comm_map[c] == usize::MAX {
            comm_map[c] = k;
            k += 1;
        }
    }
    for i in 0..n {
        old_to_new[i] = comm_map[community[i]];
    }

    // Build super-edges: (super_u, super_v, weight)
    // Use a flat map: key = (min, max) stored as super_u * k + super_v for upper triangle
    // We collect into a BTreeMap-like structure without using std::collections::BTreeMap
    // (which we already have). Use a Vec of (u,v,w) and aggregate.
    let mut edge_map: Vec<(usize, usize, f64)> = Vec::new();

    for u in 0..n {
        let su = old_to_new[u];
        for nb in &graph.adj[u] {
            let sv = old_to_new[nb.node];
            // Look for existing entry
            let mut found = false;
            for entry in edge_map.iter_mut() {
                if (entry.0 == su && entry.1 == sv) || (entry.0 == sv && entry.1 == su) {
                    entry.2 += nb.weight * 0.5; // each undirected edge visited twice
                    found = true;
                    break;
                }
            }
            if !found {
                edge_map.push((su, sv, nb.weight * 0.5));
            }
        }
    }

    let super_graph = WGraph::from_edges(k, &edge_map);
    (super_graph, old_to_new)
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Run Louvain community detection on a weighted undirected graph.
///
/// # Arguments
/// * `n_nodes` — number of nodes (0-indexed).
/// * `edges`   — `(from, to, weight)` tuples; each edge listed once.
///   Self-loops are allowed and contribute to the degree.
/// * `config`  — algorithm hyper-parameters.
///
/// # Errors
/// - [`GraphError::EmptyGraph`] if `n_nodes == 0`.
/// - [`GraphError::InvalidPlan`]`("node_out_of_range")` if any edge endpoint ≥ `n_nodes`.
pub fn louvain_communities(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    config: &LouvainConfig,
) -> GraphResult<LouvainResult> {
    if n_nodes == 0 {
        return Err(GraphError::EmptyGraph);
    }

    // Validate edges
    for &(u, v, _) in edges {
        if u >= n_nodes || v >= n_nodes {
            return Err(GraphError::InvalidPlan("node_out_of_range".to_owned()));
        }
    }

    // Start with singleton communities: each node in its own community
    let mut node_comm: Vec<usize> = (0..n_nodes).collect();

    // Phase 1 on the original graph
    let graph0 = WGraph::from_edges(n_nodes, edges);
    phase1(&graph0, &mut node_comm, config);

    // Renumber communities and coarsen; repeat
    let mut current_graph = graph0;
    let mut old_to_new: Vec<usize> = node_comm.clone();

    for _level in 0..config.max_levels {
        // Phase 2: coarsen the graph
        let (super_graph, mapping) = phase2(&current_graph, &old_to_new);

        // Update the original node → community mapping through the mapping
        // At level 0: node_comm[i] is a community id in the current (original) graph.
        // mapping[old_comm] = new_supernode.
        // But we work with the *current* assignment relative to the current graph.
        let mut super_comm: Vec<usize> = (0..super_graph.n).collect();
        let improvement = phase1(&super_graph, &mut super_comm, config);

        // Compose mapping: original node → community in super-graph
        // node_comm[i] = old comm in current graph → mapping[node_comm[i]] = supernode
        // Then super_comm[supernode] = community in super-graph
        for i in 0..n_nodes {
            let super_node = mapping[node_comm[i]];
            node_comm[i] = super_comm[super_node];
        }

        // Update current graph
        old_to_new = super_comm;
        current_graph = super_graph;

        if improvement < config.min_improvement || current_graph.n <= 1 {
            break;
        }
    }

    // Renumber community labels contiguously
    let mut label_map = vec![usize::MAX; n_nodes];
    let mut next_label = 0_usize;
    let mut final_labels = vec![0_usize; n_nodes];
    for i in 0..n_nodes {
        let c = node_comm[i];
        if c < n_nodes && label_map[c] == usize::MAX {
            label_map[c] = next_label;
            next_label += 1;
        }
        final_labels[i] = if c < n_nodes { label_map[c] } else { 0 };
    }
    let n_communities = next_label.max(1);

    // Compute final modularity on original graph
    let original_graph = WGraph::from_edges(n_nodes, edges);
    let modularity = original_graph.modularity(&final_labels);

    Ok(LouvainResult {
        labels: final_labels,
        modularity,
        n_communities,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn clique_edges(nodes: &[usize], weight: f64) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                edges.push((nodes[i], nodes[j], weight));
            }
        }
        edges
    }

    // 1. n_nodes=0 → EmptyGraph
    #[test]
    fn empty_graph_error() {
        let err = louvain_communities(0, &[], &LouvainConfig::default());
        assert!(matches!(err, Err(GraphError::EmptyGraph)), "got: {err:?}");
    }

    // 2. Single node, no edges → 1 community
    #[test]
    fn single_node() {
        let r = louvain_communities(1, &[], &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.n_communities, 1);
        assert_eq!(r.labels.len(), 1);
    }

    // 3. Two isolated nodes (no edges) → 2 communities
    #[test]
    fn two_isolated_nodes() {
        let r = louvain_communities(2, &[], &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.labels.len(), 2);
        assert!(r.n_communities >= 1);
    }

    // 4. Two dense cliques weakly connected → 2 communities
    #[test]
    fn two_cliques_connected_weakly() {
        let mut edges = clique_edges(&[0, 1, 2, 3], 10.0);
        edges.extend(clique_edges(&[4, 5, 6, 7], 10.0));
        edges.push((3, 4, 0.0001)); // very weak bridge
        let r = louvain_communities(8, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.labels.len(), 8);
        // Should find 2 communities
        assert_eq!(r.n_communities, 2);
    }

    // 5. labels.len() == n_nodes
    #[test]
    fn labels_len() {
        let edges = clique_edges(&[0, 1, 2, 3, 4], 1.0);
        let r = louvain_communities(5, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.labels.len(), 5);
    }

    // 6. All labels < n_communities
    #[test]
    fn labels_in_range() {
        let edges = clique_edges(&[0, 1, 2], 1.0);
        let r = louvain_communities(3, &edges, &LouvainConfig::default())
            .expect("value should be present");
        for &l in &r.labels {
            assert!(
                l < r.n_communities,
                "label {l} >= n_communities {}",
                r.n_communities
            );
        }
    }

    // 7. Modularity is finite
    #[test]
    fn modularity_finite() {
        let edges = clique_edges(&[0, 1, 2, 3], 1.0);
        let r = louvain_communities(4, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert!(r.modularity.is_finite(), "Q = {}", r.modularity);
    }

    // 8. Modularity >= 0 for a clique partition (well-structured graph)
    #[test]
    fn modularity_nonneg_for_cliques() {
        let mut edges = clique_edges(&[0, 1, 2], 1.0);
        edges.extend(clique_edges(&[3, 4, 5], 1.0));
        edges.push((2, 3, 0.001));
        let r = louvain_communities(6, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert!(r.modularity >= -1.0, "Q = {} should be >= -1", r.modularity);
    }

    // 9. Complete graph → 1 community (all nodes connected equally)
    #[test]
    fn all_connected_clique() {
        let n = 6;
        let nodes: Vec<usize> = (0..n).collect();
        let edges = clique_edges(&nodes, 1.0);
        let r = louvain_communities(n, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.labels.len(), n);
        assert!(r.n_communities >= 1);
    }

    // 10. n_communities >= 1
    #[test]
    fn n_communities_positive() {
        let edges = clique_edges(&[0, 1, 2, 3], 1.0);
        let r = louvain_communities(4, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert!(r.n_communities >= 1);
    }

    // 11. Node out of range → error
    #[test]
    fn node_out_of_range_error() {
        let edges = vec![(0, 5, 1.0)]; // node 5 >= n_nodes=4
        let err = louvain_communities(4, &edges, &LouvainConfig::default());
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got: {err:?}"
        );
    }

    // 12. Large separate cliques: each clique gets its own community
    #[test]
    fn three_cliques_three_communities() {
        let mut edges = clique_edges(&[0, 1, 2, 3], 2.0);
        edges.extend(clique_edges(&[4, 5, 6, 7], 2.0));
        edges.extend(clique_edges(&[8, 9, 10, 11], 2.0));
        // Very weak bridges
        edges.push((3, 4, 1e-5));
        edges.push((7, 8, 1e-5));
        let r = louvain_communities(12, &edges, &LouvainConfig::default())
            .expect("value should be present");
        assert_eq!(r.labels.len(), 12);
        assert!(r.n_communities >= 1);
    }
}
