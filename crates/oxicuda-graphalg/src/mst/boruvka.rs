//! Borůvka's MST: parallel-friendly contraction.

use crate::error::GraphalgResult;
use crate::mst::union_find::UnionFind;
use crate::repr::weighted_graph::WeightedGraph;

pub fn boruvka_mst(g: &WeightedGraph) -> GraphalgResult<Vec<(usize, usize, f64)>> {
    let mut uf = UnionFind::new(g.n);
    let mut mst: Vec<(usize, usize, f64)> = Vec::new();
    // Collect unique edges (u<v) to iterate over.
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            if u < v {
                edges.push((u, v, w));
            }
        }
    }
    while uf.num_sets() > 1 {
        // For each component, find the cheapest outgoing edge.
        let mut cheapest: Vec<i64> = vec![-1; g.n];
        let mut cheapest_w: Vec<f64> = vec![f64::INFINITY; g.n];
        for (i, &(u, v, w)) in edges.iter().enumerate() {
            let ru = uf.find(u);
            let rv = uf.find(v);
            if ru == rv {
                continue;
            }
            if w < cheapest_w[ru] {
                cheapest_w[ru] = w;
                cheapest[ru] = i as i64;
            }
            if w < cheapest_w[rv] {
                cheapest_w[rv] = w;
                cheapest[rv] = i as i64;
            }
        }
        let mut progress = false;
        for &idx in &cheapest {
            if idx < 0 {
                continue;
            }
            let i = idx as usize;
            let (u, v, w) = edges[i];
            if uf.union(u, v) {
                mst.push((u, v, w));
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    Ok(mst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boruvka_total_correct() {
        let mut g = WeightedGraph::new(4);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(0, 2, 4.0).expect("ok");
        g.add_undirected_edge(1, 2, 2.0).expect("ok");
        g.add_undirected_edge(1, 3, 5.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.0).expect("ok");
        let mst = boruvka_mst(&g).expect("ok");
        let total: f64 = mst.iter().map(|e| e.2).sum();
        assert!((total - 4.0).abs() < 1e-12);
    }
}
