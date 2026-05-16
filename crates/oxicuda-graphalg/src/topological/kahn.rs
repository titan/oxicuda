//! Kahn's topological sort (queue of in-degree-zero nodes).

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// Returns a topological order of the directed graph or an error if a cycle exists.
pub fn topo_sort_kahn(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let mut in_deg = g.in_degrees();
    let mut q: VecDeque<usize> = VecDeque::new();
    for (i, &d) in in_deg.iter().enumerate() {
        if d == 0 {
            q.push_back(i);
        }
    }
    let mut order = Vec::with_capacity(g.n);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &v in g.neighbors(u)? {
            in_deg[v] -= 1;
            if in_deg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    if order.len() != g.n {
        return Err(GraphalgError::NotADag);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_kahn_chain() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 3).expect("ok");
        let o = topo_sort_kahn(&g).expect("ok");
        assert_eq!(o, vec![0, 1, 2, 3]);
    }

    #[test]
    fn topo_kahn_cycle_err() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        assert!(topo_sort_kahn(&g).is_err());
    }

    #[test]
    fn topo_kahn_disconnected() {
        let g = AdjacencyList::new(3);
        let o = topo_sort_kahn(&g).expect("ok");
        assert_eq!(o.len(), 3);
    }
}
