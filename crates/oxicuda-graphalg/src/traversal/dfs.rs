//! Depth-First Search (iterative + pre/post order helpers).

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// Iterative DFS that returns the visited order from `source`.
pub fn dfs_iterative(g: &AdjacencyList, source: usize) -> GraphalgResult<Vec<usize>> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut visited = vec![false; g.n];
    let mut order = Vec::with_capacity(g.n);
    let mut stack: Vec<usize> = vec![source];
    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        order.push(u);
        // Iterate neighbors in reverse so smaller indices are visited first.
        let adj = g.neighbors(u)?;
        for &v in adj.iter().rev() {
            if !visited[v] {
                stack.push(v);
            }
        }
    }
    Ok(order)
}

/// Recursive-style preorder via explicit stack. Visits each node exactly once.
pub fn dfs_preorder(g: &AdjacencyList, source: usize) -> GraphalgResult<Vec<usize>> {
    dfs_iterative(g, source)
}

/// Post-order DFS: emits nodes when their subtree is finished.
pub fn dfs_postorder(g: &AdjacencyList, source: usize) -> GraphalgResult<Vec<usize>> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut visited = vec![false; g.n];
    let mut order = Vec::with_capacity(g.n);
    // Stack of (node, child_index)
    let mut stack: Vec<(usize, usize)> = Vec::new();
    visited[source] = true;
    stack.push((source, 0));
    while let Some(&(u, i)) = stack.last() {
        let adj = g.neighbors(u)?;
        if i < adj.len() {
            let v = adj[i];
            let last = stack.len() - 1;
            stack[last].1 = i + 1;
            if !visited[v] {
                visited[v] = true;
                stack.push((v, 0));
            }
        } else {
            order.push(u);
            stack.pop();
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> AdjacencyList {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(0, 2).expect("ok");
        g.add_edge(1, 3).expect("ok");
        g.add_edge(2, 3).expect("ok");
        g
    }

    #[test]
    fn dfs_visits_all() {
        let g = small();
        let v = dfs_iterative(&g, 0).expect("ok");
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dfs_starts_at_source() {
        let g = small();
        let v = dfs_iterative(&g, 0).expect("ok");
        assert_eq!(v[0], 0);
    }

    #[test]
    fn dfs_postorder_leaves_first() {
        let g = small();
        let v = dfs_postorder(&g, 0).expect("ok");
        assert_eq!(*v.last().expect("ok"), 0);
    }

    #[test]
    fn dfs_oob_source() {
        let g = small();
        assert!(dfs_iterative(&g, 7).is_err());
    }
}
