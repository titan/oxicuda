//! DSATUR coloring: select next vertex with highest degree-of-saturation.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn dsatur_coloring(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut color = vec![usize::MAX; n];
    if n == 0 {
        return Ok(color);
    }
    let deg: Vec<usize> = g.out_degrees();
    let mut sat: Vec<usize> = vec![0; n];
    for _ in 0..n {
        // Pick uncolored vertex with max saturation; ties -> max degree.
        let mut chosen = usize::MAX;
        let mut best_sat = -1i64;
        let mut best_deg = -1i64;
        for u in 0..n {
            if color[u] != usize::MAX {
                continue;
            }
            let s = sat[u] as i64;
            let d = deg[u] as i64;
            if s > best_sat || (s == best_sat && d > best_deg) {
                best_sat = s;
                best_deg = d;
                chosen = u;
            }
        }
        if chosen == usize::MAX {
            break;
        }
        // Assign lowest color not used by neighbors.
        let mut used = vec![false; n + 1];
        for &v in g.neighbors(chosen)? {
            if color[v] != usize::MAX && color[v] < used.len() {
                used[color[v]] = true;
            }
        }
        let mut c = 0;
        while c < used.len() && used[c] {
            c += 1;
        }
        color[chosen] = c;
        // Update saturation of uncolored neighbors.
        for &v in g.neighbors(chosen)? {
            if color[v] == usize::MAX {
                let mut seen = vec![false; n + 1];
                for &w in g.neighbors(v)? {
                    if color[w] != usize::MAX && color[w] < seen.len() {
                        seen[color[w]] = true;
                    }
                }
                sat[v] = seen.iter().filter(|x| **x).count();
            }
        }
    }
    Ok(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsatur_proper() {
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        let c = dsatur_coloring(&g).expect("ok");
        for u in 0..4 {
            for &v in g.neighbors(u).expect("ok") {
                assert_ne!(c[u], c[v]);
            }
        }
    }
}
