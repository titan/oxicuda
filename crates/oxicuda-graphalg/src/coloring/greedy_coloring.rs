//! Greedy coloring with natural vertex ordering.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn greedy_coloring(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut color = vec![usize::MAX; n];
    for u in 0..n {
        let mut used = vec![false; n + 1];
        for &v in g.neighbors(u)? {
            if color[v] != usize::MAX && color[v] < used.len() {
                used[color[v]] = true;
            }
        }
        let mut c = 0;
        while c < used.len() && used[c] {
            c += 1;
        }
        color[u] = c;
    }
    Ok(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_bipartite() {
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        g.add_undirected_edge(1, 3).expect("ok");
        let c = greedy_coloring(&g).expect("ok");
        for u in 0..4 {
            for &v in g.neighbors(u).expect("ok") {
                assert_ne!(c[u], c[v]);
            }
        }
    }
}
