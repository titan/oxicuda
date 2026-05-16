//! Label propagation community detection (synchronous variant, deterministic tie-break).

use std::collections::HashMap;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn label_propagation(g: &AdjacencyList, max_iter: usize) -> GraphalgResult<Vec<usize>> {
    let n = g.n;
    let mut labels: Vec<usize> = (0..n).collect();
    for _ in 0..max_iter {
        let mut changed = false;
        let mut new_labels = labels.clone();
        for u in 0..n {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &v in g.neighbors(u)? {
                *counts.entry(labels[v]).or_insert(0) += 1;
            }
            // Pick label with max count; ties broken by min label.
            let mut best_lbl = labels[u];
            let mut best_count = 0usize;
            for (&lbl, &c) in &counts {
                if c > best_count || (c == best_count && lbl < best_lbl) {
                    best_count = c;
                    best_lbl = lbl;
                }
            }
            if best_lbl != labels[u] {
                new_labels[u] = best_lbl;
                changed = true;
            }
        }
        labels = new_labels;
        if !changed {
            break;
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_two_triangles() {
        let mut g = AdjacencyList::new(6);
        for (u, v) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_undirected_edge(u, v).expect("ok");
        }
        let lbl = label_propagation(&g, 100).expect("ok");
        assert_eq!(lbl.len(), 6);
    }
}
