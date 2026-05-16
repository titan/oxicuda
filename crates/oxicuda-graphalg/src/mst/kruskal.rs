//! Kruskal's MST: sort edges, then Union-Find.

use crate::error::GraphalgResult;
use crate::mst::union_find::UnionFind;
use crate::repr::weighted_graph::WeightedGraph;

pub fn kruskal_mst(g: &WeightedGraph) -> GraphalgResult<Vec<(usize, usize, f64)>> {
    // Collect unique undirected edges (u < v).
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            if u < v {
                edges.push((u, v, w));
            }
        }
    }
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut uf = UnionFind::new(g.n);
    let mut mst = Vec::new();
    for (u, v, w) in edges {
        if uf.union(u, v) {
            mst.push((u, v, w));
            if mst.len() == g.n - 1 {
                break;
            }
        }
    }
    Ok(mst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kruskal_total_matches_prim() {
        let mut g = WeightedGraph::new(4);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(0, 2, 4.0).expect("ok");
        g.add_undirected_edge(1, 2, 2.0).expect("ok");
        g.add_undirected_edge(1, 3, 5.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.0).expect("ok");
        let mst = kruskal_mst(&g).expect("ok");
        let total: f64 = mst.iter().map(|e| e.2).sum();
        assert!((total - 4.0).abs() < 1e-12);
    }
}
