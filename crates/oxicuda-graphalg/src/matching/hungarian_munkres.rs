//! Hungarian (Kuhn-Munkres) algorithm for min-cost assignment in O(n^3).
//!
//! Implements the classic potentials-based variant for square cost matrices.

use crate::error::{GraphalgError, GraphalgResult};

/// Solve the assignment problem on a square cost matrix `cost` of size `n x n` (row-major).
/// Returns `assign[i] = j` giving column j assigned to row i, and the total cost.
pub fn hungarian_assignment(cost: &[f64], n: usize) -> GraphalgResult<(Vec<usize>, f64)> {
    if cost.len() != n * n {
        return Err(GraphalgError::DimensionMismatch {
            a: cost.len(),
            b: n * n,
        });
    }
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assign = vec![0usize; n];
    let mut total = 0.0f64;
    for j in 1..=n {
        let i = p[j];
        if i == 0 {
            continue;
        }
        assign[i - 1] = j - 1;
        total += cost[(i - 1) * n + (j - 1)];
    }
    Ok((assign, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_3x3() {
        // Cost: rows agents, cols tasks.
        let c = vec![4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let (a, t) = hungarian_assignment(&c, 3).expect("ok");
        assert!((t - 5.0).abs() < 1e-9);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn hungarian_4x4_brute() {
        let c = vec![
            7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 4.0, 8.0, 1.0, 5.0, 3.0, 1.0, 9.0, 2.0, 7.0,
        ];
        let (_a, t) = hungarian_assignment(&c, 4).expect("ok");
        // Brute-force min
        let perms = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ];
        let mut best = f64::INFINITY;
        for perm in perms {
            let s: f64 = (0..4).map(|i| c[i * 4 + perm[i]]).sum();
            if s < best {
                best = s;
            }
        }
        assert!((t - best).abs() < 1e-9);
    }
}
