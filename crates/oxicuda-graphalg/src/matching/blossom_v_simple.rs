//! Simplified blossom-based maximum matching on general (non-bipartite) graphs.
//!
//! Implements Edmonds' tree-based algorithm with explicit blossom contraction
//! for small unweighted graphs.

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const NIL: usize = usize::MAX;

/// Maximum matching on a general undirected graph. Returns `mate[v]` = matched vertex or NIL.
pub fn blossom_match_unweighted(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut mate = vec![NIL; n];
    for u in 0..n {
        if mate[u] == NIL {
            find_aug(u, g, &mut mate)?;
        }
    }
    Ok(mate)
}

fn find_aug(start: usize, g: &AdjacencyList, mate: &mut [usize]) -> GraphalgResult<bool> {
    let n = g.n;
    let mut parent = vec![NIL; n];
    let mut base: Vec<usize> = (0..n).collect();
    let mut blossom = vec![false; n];
    let mut q: VecDeque<usize> = VecDeque::new();
    let mut in_queue = vec![false; n];
    q.push_back(start);
    in_queue[start] = true;
    let mut visited = vec![false; n];
    visited[start] = true;
    while let Some(u) = q.pop_front() {
        for &v in g.neighbors(u)? {
            if base[u] == base[v] || mate[u] == v {
                continue;
            }
            if v == start || (mate[v] != NIL && parent[mate[v]] != NIL) {
                let cur_base = lca(base[u], base[v], &base, mate, &parent);
                mark_path(u, cur_base, v, &mut blossom, &base, mate, &parent);
                mark_path(v, cur_base, u, &mut blossom, &base, mate, &parent);
                for i in 0..n {
                    if blossom[base[i]] {
                        base[i] = cur_base;
                        if !in_queue[i] {
                            q.push_back(i);
                            in_queue[i] = true;
                        }
                    }
                }
            } else if parent[v] == NIL {
                parent[v] = u;
                if mate[v] == NIL {
                    // Augmenting path
                    augment(v, mate, &parent);
                    return Ok(true);
                }
                let w = mate[v];
                visited[w] = true;
                if !in_queue[w] {
                    q.push_back(w);
                    in_queue[w] = true;
                }
            }
        }
    }
    Ok(false)
}

fn lca(mut u: usize, mut v: usize, base: &[usize], mate: &[usize], parent: &[usize]) -> usize {
    let mut seen = vec![false; base.len()];
    loop {
        u = base[u];
        seen[u] = true;
        if mate[u] == NIL {
            break;
        }
        u = parent[mate[u]];
    }
    loop {
        v = base[v];
        if seen[v] {
            return v;
        }
        if mate[v] == NIL {
            break;
        }
        v = parent[mate[v]];
    }
    v
}

fn mark_path(
    mut u: usize,
    b: usize,
    child: usize,
    blossom: &mut [bool],
    base: &[usize],
    mate: &[usize],
    parent: &[usize],
) {
    let mut prev_child = child;
    while base[u] != b {
        blossom[base[u]] = true;
        blossom[base[mate[u]]] = true;
        let pm = parent[mate[u]];
        let _ = prev_child;
        prev_child = u;
        u = pm;
    }
}

fn augment(start: usize, mate: &mut [usize], parent: &[usize]) {
    let mut v = start;
    while v != NIL {
        let pv = parent[v];
        if pv == NIL {
            break;
        }
        let ppv = mate[pv];
        mate[v] = pv;
        mate[pv] = v;
        v = ppv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blossom_bipartite_match() {
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 2).expect("ok");
        g.add_undirected_edge(1, 3).expect("ok");
        let m = blossom_match_unweighted(&g).expect("ok");
        assert_eq!(m[0], 2);
        assert_eq!(m[2], 0);
    }

    #[test]
    fn blossom_triangle() {
        // Triangle: max matching = 1
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let m = blossom_match_unweighted(&g).expect("ok");
        let matched = m.iter().filter(|&&x| x != NIL).count();
        assert_eq!(matched, 2);
    }
}
