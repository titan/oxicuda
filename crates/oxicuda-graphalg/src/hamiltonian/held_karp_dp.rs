//! Held-Karp dynamic programming for exact TSP. O(n^2 * 2^n).

use crate::error::{GraphalgError, GraphalgResult};

#[derive(Debug, Clone)]
pub struct HeldKarpTour {
    pub order: Vec<usize>,
    pub cost: f64,
}

pub fn held_karp_tsp(dist: &[f64], n: usize) -> GraphalgResult<HeldKarpTour> {
    if dist.len() != n * n {
        return Err(GraphalgError::DimensionMismatch {
            a: dist.len(),
            b: n * n,
        });
    }
    if n == 0 {
        return Ok(HeldKarpTour {
            order: Vec::new(),
            cost: 0.0,
        });
    }
    if n == 1 {
        return Ok(HeldKarpTour {
            order: vec![0],
            cost: 0.0,
        });
    }
    if n > 20 {
        return Err(GraphalgError::InvalidParameter(
            "Held-Karp tractable only for n <= 20".to_string(),
        ));
    }
    let size = 1usize << n;
    let mut dp = vec![f64::INFINITY; size * n];
    let mut par = vec![usize::MAX; size * n];
    let start = 0usize;
    dp[(1usize << start) * n + start] = 0.0;
    for mask in 1..size {
        if (mask & (1 << start)) == 0 {
            continue;
        }
        for u in 0..n {
            if (mask & (1 << u)) == 0 {
                continue;
            }
            let base = dp[mask * n + u];
            if !base.is_finite() {
                continue;
            }
            for v in 0..n {
                if (mask & (1 << v)) != 0 {
                    continue;
                }
                let next_mask = mask | (1 << v);
                let cand = base + dist[u * n + v];
                let slot = &mut dp[next_mask * n + v];
                if cand < *slot {
                    *slot = cand;
                    par[next_mask * n + v] = u;
                }
            }
        }
    }
    let full_mask = size - 1;
    let mut best_cost = f64::INFINITY;
    let mut end = usize::MAX;
    for v in 0..n {
        if v == start {
            continue;
        }
        let cand = dp[full_mask * n + v] + dist[v * n + start];
        if cand < best_cost {
            best_cost = cand;
            end = v;
        }
    }
    if !best_cost.is_finite() {
        return Err(GraphalgError::NoSolution("no Hamiltonian tour".to_string()));
    }
    // Reconstruct
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut mask = full_mask;
    let mut cur = end;
    while cur != usize::MAX {
        order.push(cur);
        let p = par[mask * n + cur];
        mask &= !(1 << cur);
        if p == usize::MAX || cur == start {
            break;
        }
        cur = p;
    }
    order.reverse();
    if order.first().copied() != Some(start) {
        order.insert(0, start);
    }
    Ok(HeldKarpTour {
        order,
        cost: best_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hk_square() {
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
        let t = held_karp_tsp(&d, n).expect("ok");
        assert!((t.cost - 4.0).abs() < 1e-9);
    }
}
