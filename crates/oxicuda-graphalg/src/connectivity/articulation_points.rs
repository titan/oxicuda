//! Articulation points (cut vertices) via Tarjan's low-link algorithm.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const UNVISITED: usize = usize::MAX;

pub fn articulation_points(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut disc = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![UNVISITED; n];
    let mut child_count = vec![0usize; n];
    let mut is_ap = vec![false; n];
    let mut time = 0usize;
    for start in 0..n {
        if disc[start] != UNVISITED {
            continue;
        }
        disc[start] = time;
        low[start] = time;
        time += 1;
        parent[start] = start;
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
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
                    child_count[u] += 1;
                    stack.push((v, 0));
                } else if v != parent[u] {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                let par = parent[u];
                if par != u {
                    low[par] = low[par].min(low[u]);
                    if par != start && low[u] >= disc[par] {
                        is_ap[par] = true;
                    }
                }
                stack.pop();
            }
        }
        if child_count[start] > 1 {
            is_ap[start] = true;
        }
    }
    Ok((0..n).filter(|&i| is_ap[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ap_line_middle() {
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        let ap = articulation_points(&g).expect("ok");
        assert_eq!(ap, vec![1, 2, 3]);
    }

    #[test]
    fn ap_triangle_none() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let ap = articulation_points(&g).expect("ok");
        assert!(ap.is_empty());
    }
}
