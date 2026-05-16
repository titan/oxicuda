//! Breadth-First Search.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// BFS levels from `source`. Unreachable nodes have level `usize::MAX`.
pub fn bfs_levels(g: &AdjacencyList, source: usize) -> GraphalgResult<Vec<usize>> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut level = vec![usize::MAX; g.n];
    level[source] = 0;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(source);
    while let Some(u) = q.pop_front() {
        let lvl = level[u];
        for &v in g.neighbors(u)? {
            if level[v] == usize::MAX {
                level[v] = lvl + 1;
                q.push_back(v);
            }
        }
    }
    Ok(level)
}

/// BFS predecessors from `source`. `parent[source]` is `source` itself; unreachable -> `usize::MAX`.
pub fn bfs_parents(g: &AdjacencyList, source: usize) -> GraphalgResult<Vec<usize>> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    let mut parent = vec![usize::MAX; g.n];
    parent[source] = source;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(source);
    while let Some(u) = q.pop_front() {
        for &v in g.neighbors(u)? {
            if parent[v] == usize::MAX {
                parent[v] = u;
                q.push_back(v);
            }
        }
    }
    Ok(parent)
}

/// Reconstruct a shortest path from `source` to `target` from a parents vector.
pub fn reconstruct_path(
    parents: &[usize],
    source: usize,
    target: usize,
) -> GraphalgResult<Vec<usize>> {
    if target >= parents.len() {
        return Err(GraphalgError::IndexOutOfBounds {
            index: target,
            len: parents.len(),
        });
    }
    if parents[target] == usize::MAX {
        return Err(GraphalgError::NoSolution(format!(
            "node {target} unreachable from source {source}"
        )));
    }
    let mut path = Vec::new();
    let mut cur = target;
    loop {
        path.push(cur);
        if cur == source {
            break;
        }
        let p = parents[cur];
        if p == cur || p == usize::MAX {
            return Err(GraphalgError::NoSolution(format!(
                "path reconstruction failed at {cur}"
            )));
        }
        cur = p;
    }
    path.reverse();
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for i in 0..n - 1 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        g
    }

    #[test]
    fn bfs_line_levels() {
        let g = line(5);
        let lv = bfs_levels(&g, 0).expect("ok");
        assert_eq!(lv, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn bfs_source_oob() {
        let g = line(3);
        assert!(bfs_levels(&g, 99).is_err());
    }

    #[test]
    fn bfs_parents_line() {
        let g = line(4);
        let p = bfs_parents(&g, 0).expect("ok");
        assert_eq!(p, vec![0, 0, 1, 2]);
    }

    #[test]
    fn bfs_unreachable() {
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        let lv = bfs_levels(&g, 0).expect("ok");
        assert_eq!(lv[2], usize::MAX);
        assert_eq!(lv[3], usize::MAX);
    }

    #[test]
    fn reconstruct_simple_path() {
        let g = line(5);
        let p = bfs_parents(&g, 0).expect("ok");
        let path = reconstruct_path(&p, 0, 4).expect("ok");
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }
}
