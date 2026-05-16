//! Welsh-Powell coloring: order vertices by degree descending and color greedily.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn welsh_powell_coloring(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let deg = g.out_degrees();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| deg[b].cmp(&deg[a]).then(a.cmp(&b)));
    let mut color = vec![usize::MAX; n];
    for &u in &order {
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
    fn wp_proper() {
        let mut g = AdjacencyList::new(5);
        for (u, v) in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)] {
            g.add_undirected_edge(u, v).expect("ok");
        }
        let c = welsh_powell_coloring(&g).expect("ok");
        for u in 0..5 {
            for &v in g.neighbors(u).expect("ok") {
                assert_ne!(c[u], c[v]);
            }
        }
    }
}
