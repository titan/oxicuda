//! Bidirectional BFS for unweighted shortest path.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// Bidirectional BFS shortest path on an unweighted graph (directed or undirected).
/// For directed graphs the backward search uses the reverse graph.
pub fn bidirectional_bfs(
    g: &AdjacencyList,
    source: usize,
    target: usize,
) -> GraphalgResult<Vec<usize>> {
    if source >= g.n || target >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(target),
            n: g.n,
        });
    }
    if source == target {
        return Ok(vec![source]);
    }
    let rev = g.reverse();
    let mut parent_f = vec![usize::MAX; g.n];
    let mut parent_b = vec![usize::MAX; g.n];
    parent_f[source] = source;
    parent_b[target] = target;
    let mut q_f: VecDeque<usize> = VecDeque::new();
    let mut q_b: VecDeque<usize> = VecDeque::new();
    q_f.push_back(source);
    q_b.push_back(target);
    let mut meeting = usize::MAX;
    while !q_f.is_empty() && !q_b.is_empty() {
        // expand forward layer
        let lf = q_f.len();
        for _ in 0..lf {
            let u = match q_f.pop_front() {
                Some(x) => x,
                None => break,
            };
            for &v in g.neighbors(u)? {
                if parent_f[v] == usize::MAX {
                    parent_f[v] = u;
                    if parent_b[v] != usize::MAX {
                        meeting = v;
                        break;
                    }
                    q_f.push_back(v);
                }
            }
            if meeting != usize::MAX {
                break;
            }
        }
        if meeting != usize::MAX {
            break;
        }
        // expand backward layer
        let lb = q_b.len();
        for _ in 0..lb {
            let u = match q_b.pop_front() {
                Some(x) => x,
                None => break,
            };
            for &v in rev.neighbors(u)? {
                if parent_b[v] == usize::MAX {
                    parent_b[v] = u;
                    if parent_f[v] != usize::MAX {
                        meeting = v;
                        break;
                    }
                    q_b.push_back(v);
                }
            }
            if meeting != usize::MAX {
                break;
            }
        }
    }
    if meeting == usize::MAX {
        return Err(GraphalgError::NoSolution(format!(
            "no path from {source} to {target}"
        )));
    }
    // reconstruct
    let mut left = Vec::new();
    let mut cur = meeting;
    while cur != source {
        left.push(cur);
        let p = parent_f[cur];
        if p == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "forward parent missing".to_string(),
            ));
        }
        cur = p;
    }
    left.push(source);
    left.reverse();
    let mut right = Vec::new();
    let mut cur = meeting;
    while cur != target {
        let p = parent_b[cur];
        if p == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "backward parent missing".to_string(),
            ));
        }
        cur = p;
        right.push(cur);
    }
    left.extend(right);
    Ok(left)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bidir_line() {
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        let p = bidirectional_bfs(&g, 0, 4).expect("ok");
        assert_eq!(p, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn bidir_self() {
        let g = AdjacencyList::new(3);
        let p = bidirectional_bfs(&g, 1, 1).expect("ok");
        assert_eq!(p, vec![1]);
    }
}
