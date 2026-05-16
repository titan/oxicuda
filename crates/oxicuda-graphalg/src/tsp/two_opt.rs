//! 2-opt local search improvement for TSP tours.

use crate::error::{GraphalgError, GraphalgResult};

pub fn two_opt_improve(
    tour: &[usize],
    dist: &[f64],
    n: usize,
    max_iter: usize,
) -> GraphalgResult<Vec<usize>> {
    if dist.len() != n * n {
        return Err(GraphalgError::DimensionMismatch {
            a: dist.len(),
            b: n * n,
        });
    }
    if tour.len() != n {
        return Err(GraphalgError::DimensionMismatch {
            a: tour.len(),
            b: n,
        });
    }
    let mut t = tour.to_vec();
    let mut iters = 0usize;
    let mut improved = true;
    while improved && iters < max_iter {
        improved = false;
        iters += 1;
        for i in 0..n - 1 {
            for j in i + 1..n {
                let a = t[i];
                let b = t[(i + 1) % n];
                let c = t[j];
                let d = t[(j + 1) % n];
                if a == c || b == d {
                    continue;
                }
                let old_cost = dist[a * n + b] + dist[c * n + d];
                let new_cost = dist[a * n + c] + dist[b * n + d];
                if new_cost + 1e-12 < old_cost {
                    t[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_opt_runs() {
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
        let init = vec![0, 2, 1, 3];
        let t = two_opt_improve(&init, &d, n, 50).expect("ok");
        let cost: f64 = (0..n).map(|i| d[t[i] * n + t[(i + 1) % n]]).sum();
        assert!(cost <= 4.0 + 1e-9);
    }
}
