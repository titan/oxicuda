//! Nearest-neighbor TSP heuristic on a metric distance matrix `n x n` (row-major).

use crate::error::{GraphalgError, GraphalgResult};

#[derive(Debug, Clone)]
pub struct TspTour {
    pub order: Vec<usize>,
    pub cost: f64,
}

pub fn nearest_neighbor_tour(dist: &[f64], n: usize, start: usize) -> GraphalgResult<TspTour> {
    if dist.len() != n * n {
        return Err(GraphalgError::DimensionMismatch {
            a: dist.len(),
            b: n * n,
        });
    }
    if n == 0 {
        return Ok(TspTour {
            order: Vec::new(),
            cost: 0.0,
        });
    }
    if start >= n {
        return Err(GraphalgError::SourceOutOfRange { node: start, n });
    }
    let mut visited = vec![false; n];
    visited[start] = true;
    let mut order = vec![start];
    let mut cost = 0.0;
    let mut cur = start;
    while order.len() < n {
        let mut best = usize::MAX;
        let mut bestd = f64::INFINITY;
        for v in 0..n {
            if visited[v] {
                continue;
            }
            let d = dist[cur * n + v];
            if d < bestd {
                bestd = d;
                best = v;
            }
        }
        if best == usize::MAX {
            return Err(GraphalgError::NoSolution(
                "no unvisited node available".to_string(),
            ));
        }
        visited[best] = true;
        order.push(best);
        cost += bestd;
        cur = best;
    }
    // Close the tour.
    cost += dist[cur * n + start];
    Ok(TspTour { order, cost })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_4x4_square() {
        // Cities at (0,0), (1,0), (1,1), (0,1).
        let pts: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let n = 4;
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i].0 - pts[j].0;
                let dy = pts[i].1 - pts[j].1;
                d[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let t = nearest_neighbor_tour(&d, n, 0).expect("ok");
        assert_eq!(t.order.len(), 4);
        assert!((t.cost - 4.0).abs() < 1e-9);
    }
}
