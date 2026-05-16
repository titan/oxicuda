//! Closeness centrality `1 / sum d(v, u)` (unweighted graph).

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn closeness_centrality(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    let mut cc = vec![0.0f64; n];
    for s in 0..n {
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for &v in g.neighbors(u)? {
                if dist[v] < 0 {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        let sum: i64 = dist.iter().filter(|&&d| d > 0).sum();
        if sum > 0 {
            cc[s] = (n - 1) as f64 / sum as f64;
        }
    }
    Ok(cc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closeness_path() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        let c = closeness_centrality(&g).expect("ok");
        // For node 1: avg dist 1, closeness = 2 / (1+1) = 1
        assert!((c[1] - 1.0).abs() < 1e-12);
    }
}
