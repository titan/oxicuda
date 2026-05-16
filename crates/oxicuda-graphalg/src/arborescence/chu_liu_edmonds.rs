//! Chu-Liu-Edmonds minimum spanning arborescence (directed MST rooted at the given root).
//!
//! Returns the parent of each node (`parent[root] = root`). Excludes the root from incoming edges.

use std::collections::HashMap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

pub fn chu_liu_edmonds(g: &WeightedGraph, root: usize) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    if root >= n {
        return Err(GraphalgError::SourceOutOfRange { node: root, n });
    }
    // Build list of all directed edges.
    let edges: Vec<(usize, usize, f64)> = g.all_edges();
    cle_internal(n, root, &edges)
}

fn cle_internal(
    n: usize,
    root: usize,
    edges: &[(usize, usize, f64)],
) -> GraphalgResult<Vec<usize>> {
    // Step 1: for each non-root node, find min-cost incoming edge.
    let mut min_in_edge: Vec<Option<(usize, usize, f64)>> = vec![None; n];
    for (idx, &(u, v, w)) in edges.iter().enumerate() {
        if v == root {
            continue;
        }
        match min_in_edge[v] {
            None => min_in_edge[v] = Some((u, idx, w)),
            Some((_, _, cw)) if w < cw => {
                min_in_edge[v] = Some((u, idx, w));
            }
            _ => {}
        }
    }
    for v in 0..n {
        if v == root {
            continue;
        }
        if min_in_edge[v].is_none() {
            return Err(GraphalgError::NoSolution(format!(
                "node {v} has no incoming edge"
            )));
        }
    }
    // Step 2: detect cycles by following min_in_edge chains.
    let mut comp = vec![usize::MAX; n];
    let mut visited = vec![usize::MAX; n];
    let mut comp_id = 0usize;
    let mut in_cycle = vec![false; n];
    for start in 0..n {
        if visited[start] != usize::MAX {
            continue;
        }
        let mut node = start;
        while node != root && visited[node] == usize::MAX && comp[node] == usize::MAX {
            visited[node] = start;
            let (u, _idx, _w) = match min_in_edge[node] {
                Some(t) => t,
                None => break,
            };
            node = u;
        }
        if node != root && visited[node] == start {
            // Found a cycle: walk it once and mark a component id.
            let mut cur = node;
            loop {
                comp[cur] = comp_id;
                in_cycle[cur] = true;
                let (u, _idx, _w) = match min_in_edge[cur] {
                    Some(t) => t,
                    None => break,
                };
                cur = u;
                if cur == node {
                    break;
                }
            }
            comp_id += 1;
        }
    }
    if comp_id == 0 {
        // No cycle: solution is min_in_edge for each non-root.
        let mut parent = vec![root; n];
        for v in 0..n {
            if v == root {
                continue;
            }
            if let Some((u, _, _)) = min_in_edge[v] {
                parent[v] = u;
            }
        }
        return Ok(parent);
    }
    // Step 3: contract cycles into super-nodes. Assign each non-cycle node its own id too.
    let mut new_id = vec![0usize; n];
    let mut next_id = comp_id;
    for v in 0..n {
        if in_cycle[v] {
            new_id[v] = comp[v];
        } else {
            new_id[v] = next_id;
            next_id += 1;
        }
    }
    let new_root = new_id[root];
    // Build new edge list.
    let mut new_edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut edge_map: HashMap<(usize, usize), (usize, usize, f64)> = HashMap::new();
    for &(u, v, w) in edges {
        if v == root {
            continue;
        }
        let nu = new_id[u];
        let nv = new_id[v];
        if nu == nv {
            continue;
        }
        // Reweight if v is in a cycle: subtract min in-edge weight to v.
        let adj_w = if in_cycle[v] {
            let cw = match min_in_edge[v] {
                Some((_, _, x)) => x,
                None => 0.0,
            };
            w - cw
        } else {
            w
        };
        // Keep cheapest reweighted edge per (nu, nv); remember original (u, v, w).
        match edge_map.get_mut(&(nu, nv)) {
            None => {
                edge_map.insert((nu, nv), (u, v, adj_w));
            }
            Some(slot) => {
                if adj_w < slot.2 {
                    *slot = (u, v, adj_w);
                }
            }
        }
    }
    for ((nu, nv), (_, _, w)) in &edge_map {
        new_edges.push((*nu, *nv, *w));
    }
    let new_parents_super = cle_internal(next_id, new_root, &new_edges)?;
    // Decode: parent[v] = original tail of the edge that brought v's super-node in.
    let mut parent = vec![root; n];
    // For each non-cycle node v != root: use min_in_edge[v] from the recursion.
    // To recover original endpoints we replay edge_map and the cycle structure.
    // Default fill in-cycle nodes by their original min in-edge.
    for v in 0..n {
        if v == root {
            continue;
        }
        if let Some((u, _, _)) = min_in_edge[v] {
            parent[v] = u;
        }
    }
    // Now patch the entry edge of each cycle: the cycle has one node whose parent comes from outside.
    for c in 0..comp_id {
        // Find the super-node id of this cycle: it's c.
        // In new_parents_super, the chosen incoming edge for super-node c
        // corresponds to some original (u, v, _) inside cycle c.
        let super_in = new_parents_super[c];
        // Look up an original edge with new id (super_in, c) using stored mapping.
        let key = (super_in, c);
        if let Some(&(orig_u, orig_v, _)) = edge_map.get(&key) {
            parent[orig_v] = orig_u;
        }
    }
    Ok(parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cle_no_cycle() {
        // Simple tree-like: 0->1, 0->2, 1->3
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 1.0).expect("ok");
        g.add_edge(1, 3, 1.0).expect("ok");
        let p = chu_liu_edmonds(&g, 0).expect("ok");
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
        assert_eq!(p[3], 1);
    }
}
