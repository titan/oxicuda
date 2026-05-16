//! Bridges (cut edges) via Tarjan's low-link algorithm on undirected graphs.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const UNVISITED: usize = usize::MAX;

pub fn bridges_tarjan(g: &AdjacencyList) -> GraphalgResult<Vec<(usize, usize)>> {
    let n = g.n;
    let mut disc = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![UNVISITED; n];
    let mut time = 0usize;
    let mut bridges: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if disc[start] != UNVISITED {
            continue;
        }
        // Iterative DFS with explicit edge index.
        let mut stack: Vec<(usize, usize)> = Vec::new();
        disc[start] = time;
        low[start] = time;
        time += 1;
        parent[start] = start;
        stack.push((start, 0));
        while let Some(&(u, i)) = stack.last() {
            let adj = g.neighbors(u)?;
            if i < adj.len() {
                let v = adj[i];
                let last = stack.len() - 1;
                stack[last].1 = i + 1;
                if disc[v] == UNVISITED {
                    parent[v] = u;
                    disc[v] = time;
                    low[v] = time;
                    time += 1;
                    stack.push((v, 0));
                } else if v != parent[u] {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                // finish: propagate to parent and possibly emit bridge
                let par = parent[u];
                if par != u {
                    low[par] = low[par].min(low[u]);
                    if low[u] > disc[par] {
                        bridges.push((par.min(u), par.max(u)));
                    }
                }
                stack.pop();
            }
        }
    }
    Ok(bridges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_line() {
        // Line graph 0-1-2-3 has 3 bridges.
        let mut g = AdjacencyList::new(4);
        for i in 0..3 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        let b = bridges_tarjan(&g).expect("ok");
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn bridge_triangle() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let b = bridges_tarjan(&g).expect("ok");
        assert_eq!(b.len(), 0);
    }
}
