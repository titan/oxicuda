//! Girvan-Newman community detection by iterative edge-betweenness removal.
//!
//! Cost: O(K * V * E) for K removals, since betweenness is O(VE) per iteration.

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

/// Compute Girvan-Newman communities. `k_communities` is the target number of components.
pub fn girvan_newman_communities(
    g: &AdjacencyList,
    k_communities: usize,
) -> GraphalgResult<Vec<usize>> {
    let mut work = g.clone();
    let n = g.n;
    let mut labels = vec![0usize; n];
    let mut iters = 0usize;
    let cap = n.saturating_mul(n);
    loop {
        // Compute connected components.
        let comp = connected_components(&work)?;
        let num_comp = comp.iter().copied().max().map(|x| x + 1).unwrap_or(0);
        if num_comp >= k_communities || iters >= cap {
            labels = comp;
            break;
        }
        // Compute edge betweenness.
        let eb = edge_betweenness(&work)?;
        if eb.is_empty() {
            labels = comp;
            break;
        }
        let mut best = (0usize, 0usize);
        let mut best_v = -1.0f64;
        for &(u, v, b) in &eb {
            if b > best_v {
                best_v = b;
                best = (u, v);
            }
        }
        // Remove edge u-v from both sides.
        let (u, v) = best;
        work.edges[u].retain(|&x| x != v);
        work.edges[v].retain(|&x| x != u);
        iters += 1;
    }
    Ok(labels)
}

fn connected_components(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut comp = vec![usize::MAX; n];
    let mut id = 0usize;
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        comp[start] = id;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            for &v in g.neighbors(u)? {
                if comp[v] == usize::MAX {
                    comp[v] = id;
                    q.push_back(v);
                }
            }
        }
        id += 1;
    }
    Ok(comp)
}

fn edge_betweenness(g: &AdjacencyList) -> GraphalgResult<Vec<(usize, usize, f64)>> {
    let n = g.n;
    let mut eb: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    for s in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            stack.push(v);
            for &w in g.neighbors(v)? {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    q.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    preds[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &preds[w] {
                if sigma[w] > 0.0 {
                    let c = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += c;
                    let key = (v.min(w), v.max(w));
                    *eb.entry(key).or_insert(0.0) += c;
                }
            }
        }
    }
    Ok(eb.into_iter().map(|((u, v), b)| (u, v, b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gn_two_components_after_bridge_removal() {
        let mut g = AdjacencyList::new(6);
        for (u, v) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_undirected_edge(u, v).expect("ok");
        }
        let labels = girvan_newman_communities(&g, 2).expect("ok");
        let groups: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(groups.len(), 2);
    }
}
