//! Louvain modularity maximisation (Blondel et al. 2008).
//!
//! Two-phase algorithm:
//!   Phase 1 (local optimisation): for each node, move it to the neighboring
//!     community that maximises the modularity gain. Repeat until no move improves.
//!   Phase 2 (aggregation): contract each community into a super-node and repeat.
//! The procedure terminates when no improvement can be found.

use std::collections::HashMap;

use crate::error::GraphalgResult;
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone)]
pub struct LouvainOutput {
    pub communities: Vec<usize>,
    pub modularity: f64,
}

pub fn louvain_communities(g: &WeightedGraph) -> GraphalgResult<LouvainOutput> {
    let n = g.n;
    if n == 0 {
        return Ok(LouvainOutput {
            communities: Vec::new(),
            modularity: 0.0,
        });
    }
    // Make graph undirected by symmetrising. We assume the input is already undirected.
    let mut current = g.clone();
    let mut node_to_comm: Vec<usize> = (0..n).collect();
    let mut total_iters = 0usize;
    loop {
        let nc = current.n;
        let mut comm: Vec<usize> = (0..nc).collect();
        let two_m: f64 = current
            .edges
            .iter()
            .flat_map(|adj| adj.iter().map(|&(_, w)| w))
            .sum();
        if two_m <= 0.0 {
            break;
        }
        // Sum of degrees per community.
        let mut k = vec![0.0f64; nc];
        for (u, adj) in current.edges.iter().enumerate() {
            for &(_, w) in adj {
                k[u] += w;
            }
        }
        let mut tot = k.clone();
        let mut improved = true;
        let mut local_iters = 0usize;
        while improved && local_iters < 50 {
            improved = false;
            local_iters += 1;
            for u in 0..nc {
                let cu = comm[u];
                // Compute sum of weights from u into each neighbor community
                let mut weights_to: HashMap<usize, f64> = HashMap::new();
                for &(v, w) in &current.edges[u] {
                    if v == u {
                        continue;
                    }
                    *weights_to.entry(comm[v]).or_insert(0.0) += w;
                }
                // Remove u from cu
                tot[cu] -= k[u];
                let mut best_gain = 0.0f64;
                let mut best_comm = cu;
                let k_u = k[u];
                for (&c, &k_in_c) in weights_to.iter() {
                    let gain = k_in_c - tot[c] * k_u / two_m;
                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = c;
                    }
                }
                tot[best_comm] += k_u;
                if best_comm != cu {
                    comm[u] = best_comm;
                    improved = true;
                }
            }
        }
        // Relabel communities into [0, k)
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for &c in &comm {
            let len = remap.len();
            remap.entry(c).or_insert(len);
        }
        let next_n = remap.len();
        let new_comm: Vec<usize> = comm.iter().map(|c| remap[c]).collect();
        // Update node_to_comm in original-node indexing.
        for v in node_to_comm.iter_mut() {
            *v = new_comm[*v];
        }
        if next_n == nc {
            break;
        }
        // Build aggregated graph.
        let mut agg = WeightedGraph::new(next_n);
        let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
        for (u, adj) in current.edges.iter().enumerate() {
            for &(v, w) in adj {
                let cu = new_comm[u];
                let cv = new_comm[v];
                *acc.entry((cu, cv)).or_insert(0.0) += w;
            }
        }
        for ((u, v), w) in acc {
            agg.edges[u].push((v, w));
        }
        current = agg;
        total_iters += 1;
        if total_iters > 50 {
            break;
        }
    }
    // Final modularity on original graph
    let modularity = modularity(g, &node_to_comm);
    Ok(LouvainOutput {
        communities: node_to_comm,
        modularity,
    })
}

fn modularity(g: &WeightedGraph, comm: &[usize]) -> f64 {
    let two_m: f64 = g
        .edges
        .iter()
        .flat_map(|adj| adj.iter().map(|&(_, w)| w))
        .sum();
    if two_m <= 0.0 {
        return 0.0;
    }
    let k: Vec<f64> = g
        .edges
        .iter()
        .map(|adj| adj.iter().map(|&(_, w)| w).sum())
        .collect();
    let mut q = 0.0f64;
    for u in 0..g.n {
        for &(v, w) in &g.edges[u] {
            if comm[u] == comm[v] {
                q += w - k[u] * k[v] / two_m;
            }
        }
    }
    q / two_m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn louvain_two_cliques() {
        // Two triangles joined by a single bridge.
        let mut g = WeightedGraph::new(6);
        for (u, v) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_undirected_edge(u, v, 1.0).expect("ok");
        }
        let out = louvain_communities(&g).expect("ok");
        assert!(out.modularity >= 0.0);
        let groups: std::collections::HashSet<usize> = out.communities.iter().copied().collect();
        assert!(groups.len() >= 2);
    }
}
