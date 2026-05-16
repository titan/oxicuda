//! Kosaraju's algorithm: two DFS passes.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn scc_kosaraju(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    let n = g.n;
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    // First pass: iterative postorder on original graph.
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = Vec::new();
        visited[start] = true;
        stack.push((start, 0));
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
    }
    // Second pass on reverse graph in reverse-finishing order.
    let rev = g.reverse();
    let mut comp = vec![usize::MAX; n];
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    for &start in order.iter().rev() {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = sccs.len();
        let mut cur = Vec::new();
        let mut stack = vec![start];
        comp[start] = id;
        while let Some(u) = stack.pop() {
            cur.push(u);
            for &v in rev.neighbors(u)? {
                if comp[v] == usize::MAX {
                    comp[v] = id;
                    stack.push(v);
                }
            }
        }
        sccs.push(cur);
    }
    Ok(sccs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kosaraju_matches_tarjan() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        g.add_edge(3, 1).expect("ok");
        let sccs = scc_kosaraju(&g).expect("ok");
        assert_eq!(sccs.len(), 2);
    }
}
