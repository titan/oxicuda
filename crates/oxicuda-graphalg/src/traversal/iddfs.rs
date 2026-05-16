//! Iterative-Deepening Depth-First Search.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// IDDFS search for `target` from `source` up to `max_depth`. Returns the path if found.
pub fn iddfs_search(
    g: &AdjacencyList,
    source: usize,
    target: usize,
    max_depth: usize,
) -> GraphalgResult<Vec<usize>> {
    if source >= g.n || target >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(target),
            n: g.n,
        });
    }
    for depth in 0..=max_depth {
        let mut path = Vec::new();
        let mut visited = vec![false; g.n];
        if dls(g, source, target, depth, &mut path, &mut visited)? {
            return Ok(path);
        }
    }
    Err(GraphalgError::NoSolution(format!(
        "target {target} not reachable from {source} within depth {max_depth}"
    )))
}

fn dls(
    g: &AdjacencyList,
    u: usize,
    target: usize,
    depth: usize,
    path: &mut Vec<usize>,
    visited: &mut [bool],
) -> GraphalgResult<bool> {
    path.push(u);
    visited[u] = true;
    if u == target {
        return Ok(true);
    }
    if depth == 0 {
        path.pop();
        visited[u] = false;
        return Ok(false);
    }
    for &v in g.neighbors(u)? {
        if !visited[v] && dls(g, v, target, depth - 1, path, visited)? {
            return Ok(true);
        }
    }
    path.pop();
    visited[u] = false;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iddfs_finds_path() {
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        let path = iddfs_search(&g, 0, 4, 5).expect("ok");
        assert_eq!(path.first().copied().expect("ok"), 0usize);
        assert_eq!(path.last().copied().expect("ok"), 4usize);
    }

    #[test]
    fn iddfs_depth_limit() {
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        assert!(iddfs_search(&g, 0, 4, 2).is_err());
    }
}
