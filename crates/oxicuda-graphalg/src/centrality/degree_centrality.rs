//! Degree centrality (normalised by `n - 1`).

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn degree_centrality(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    if n <= 1 {
        return Ok(vec![0.0; n]);
    }
    let denom = (n - 1) as f64;
    Ok(g.out_degrees()
        .into_iter()
        .map(|d| d as f64 / denom)
        .collect())
}

pub fn in_degree_centrality(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    if n <= 1 {
        return Ok(vec![0.0; n]);
    }
    let denom = (n - 1) as f64;
    Ok(g.in_degrees()
        .into_iter()
        .map(|d| d as f64 / denom)
        .collect())
}

pub fn out_degree_centrality(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    degree_centrality(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_degree() {
        let mut g = AdjacencyList::new(4);
        for v in 1..4 {
            g.add_undirected_edge(0, v).expect("ok");
        }
        let dc = degree_centrality(&g).expect("ok");
        assert!((dc[0] - 1.0).abs() < 1e-12);
    }
}
