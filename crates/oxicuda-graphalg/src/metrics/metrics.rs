//! Diameter, radius, density, clustering coefficient, transitivity.

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

fn bfs_eccentricities(g: &AdjacencyList) -> GraphalgResult<Vec<i64>> {
    let n = g.n;
    let mut ecc = vec![0i64; n];
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
        let m = dist.iter().copied().filter(|&d| d >= 0).max().unwrap_or(0);
        ecc[s] = m;
    }
    Ok(ecc)
}

/// Graph diameter: max eccentricity.
pub fn diameter(g: &AdjacencyList) -> GraphalgResult<i64> {
    let ecc = bfs_eccentricities(g)?;
    Ok(ecc.into_iter().max().unwrap_or(0))
}

/// Graph radius: min eccentricity.
pub fn radius(g: &AdjacencyList) -> GraphalgResult<i64> {
    let ecc = bfs_eccentricities(g)?;
    Ok(ecc.into_iter().min().unwrap_or(0))
}

/// Density: `m / (n * (n - 1))` for directed; for undirected double the edges first.
pub fn density(g: &AdjacencyList, undirected: bool) -> GraphalgResult<f64> {
    let n = g.n;
    if n < 2 {
        return Ok(0.0);
    }
    let m = g.num_edges() as f64;
    let denom = (n as f64) * ((n - 1) as f64);
    let d = if undirected { m / denom } else { m / denom };
    Ok(d)
}

/// Global clustering coefficient = `3 * triangles / triplets` for undirected graphs.
pub fn clustering_coefficient_global(g: &AdjacencyList) -> GraphalgResult<f64> {
    let n = g.n;
    let mut tri = 0u64;
    let mut triplets = 0u64;
    // Compute via neighbor-set intersection.
    let mut nset: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];
    for (u, adj) in g.edges.iter().enumerate() {
        for &v in adj {
            if v != u {
                nset[u].insert(v);
            }
        }
    }
    for u in 0..n {
        let deg = nset[u].len() as u64;
        if deg >= 2 {
            triplets += deg * (deg - 1) / 2;
            let neigh: Vec<usize> = nset[u].iter().copied().collect();
            for i in 0..neigh.len() {
                for j in i + 1..neigh.len() {
                    if nset[neigh[i]].contains(&neigh[j]) {
                        tri += 1;
                    }
                }
            }
        }
    }
    // Each triangle counted 3 times (once per vertex).
    let tri3 = tri; // already counted once per vertex per triangle; final factor below.
    if triplets == 0 {
        return Ok(0.0);
    }
    Ok(tri3 as f64 / triplets as f64)
}

/// Transitivity: `3 * triangles / paths-of-length-two` (same as global clustering for undirected simple graphs).
pub fn transitivity(g: &AdjacencyList) -> GraphalgResult<f64> {
    clustering_coefficient_global(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diameter_path() {
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        assert_eq!(diameter(&g).expect("ok"), 4);
        assert_eq!(radius(&g).expect("ok"), 2);
    }

    #[test]
    fn cc_triangle_one() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let c = clustering_coefficient_global(&g).expect("ok");
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn density_complete() {
        let mut g = AdjacencyList::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_undirected_edge(i, j).expect("ok");
            }
        }
        // 4 nodes, 12 directed edges, density = 12 / (4*3) = 1.0
        assert!((density(&g, true).expect("ok") - 1.0).abs() < 1e-9);
    }
}
