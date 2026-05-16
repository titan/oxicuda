//! Metrics for evaluating manifold embeddings:
//! - Trustworthiness (Venna & Kaski 2001).
//! - Continuity.
//! - KL divergence (for t-SNE-style P vs Q).
//! - Neighbourhood preservation.

use crate::error::{ManifoldError, ManifoldResult};

/// Compute the full `n x n` pairwise Euclidean distance matrix of row-major data.
pub fn pairwise_distances(x: &[f64], n: usize, dim: usize) -> ManifoldResult<Vec<f64>> {
    if x.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![x.len()],
        });
    }
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..dim {
                let v = x[i * dim + k] - x[j * dim + k];
                s += v * v;
            }
            let s = s.sqrt();
            d[i * n + j] = s;
            d[j * n + i] = s;
        }
    }
    Ok(d)
}

fn rank_matrix(d: &[f64], n: usize) -> Vec<Vec<usize>> {
    // For each row, return permutation of indices sorted by distance ascending (excluding self).
    let mut ranks: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut pairs: Vec<(f64, usize)> = (0..n)
            .filter(|j| *j != i)
            .map(|j| (d[i * n + j], j))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        ranks.push(pairs.into_iter().map(|t| t.1).collect());
    }
    ranks
}

fn rank_lookup(r: &[Vec<usize>], i: usize, j: usize) -> usize {
    for (pos, &idx) in r[i].iter().enumerate() {
        if idx == j {
            return pos + 1;
        }
    }
    r[i].len() + 1
}

/// Trustworthiness `T(k)`.
///
/// Penalises low-d neighbours that are *not* high-d neighbours.
pub fn trustworthiness(
    x_high: &[f64],
    x_low: &[f64],
    n: usize,
    high_dim: usize,
    low_dim: usize,
    k: usize,
) -> ManifoldResult<f64> {
    let dh = pairwise_distances(x_high, n, high_dim)?;
    let dl = pairwise_distances(x_low, n, low_dim)?;
    let rh = rank_matrix(&dh, n);
    let rl = rank_matrix(&dl, n);
    if k == 0 || k > n - 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: format!("must be in 1..{n}"),
        });
    }
    let mut sum = 0.0;
    for i in 0..n {
        // Low-d neighbours of i
        let lows: &[usize] = &rl[i][..k];
        // High-d neighbours
        let highs: &[usize] = &rh[i][..k];
        for &j in lows {
            if !highs.contains(&j) {
                let r_h = rank_lookup(&rh, i, j) as f64;
                sum += r_h - k as f64;
            }
        }
    }
    let nk = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0).max(1.0);
    Ok(1.0 - 2.0 / nk * sum)
}

/// Continuity `C(k)` — dual of trustworthiness, penalises high-d neighbours lost in low-d.
pub fn continuity(
    x_high: &[f64],
    x_low: &[f64],
    n: usize,
    high_dim: usize,
    low_dim: usize,
    k: usize,
) -> ManifoldResult<f64> {
    let dh = pairwise_distances(x_high, n, high_dim)?;
    let dl = pairwise_distances(x_low, n, low_dim)?;
    let rh = rank_matrix(&dh, n);
    let rl = rank_matrix(&dl, n);
    if k == 0 || k > n - 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: format!("must be in 1..{n}"),
        });
    }
    let mut sum = 0.0;
    for i in 0..n {
        let lows: &[usize] = &rl[i][..k];
        let highs: &[usize] = &rh[i][..k];
        for &j in highs {
            if !lows.contains(&j) {
                let r_l = rank_lookup(&rl, i, j) as f64;
                sum += r_l - k as f64;
            }
        }
    }
    let nk = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0).max(1.0);
    Ok(1.0 - 2.0 / nk * sum)
}

/// KL divergence `KL(P || Q) = sum_ij p_ij log(p_ij / q_ij)`.
pub fn kl_pq(p: &[f64], q: &[f64]) -> ManifoldResult<f64> {
    if p.len() != q.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: p.len(),
            b: q.len(),
        });
    }
    let mut s = 0.0;
    for (pi, qi) in p.iter().zip(q) {
        if *pi > 1e-12 && *qi > 1e-12 {
            s += pi * (pi / qi).ln();
        }
    }
    Ok(s)
}

/// Neighbourhood preservation: fraction of high-d kNN preserved in low-d.
pub fn neighborhood_preservation(
    x_high: &[f64],
    x_low: &[f64],
    n: usize,
    high_dim: usize,
    low_dim: usize,
    k: usize,
) -> ManifoldResult<f64> {
    let dh = pairwise_distances(x_high, n, high_dim)?;
    let dl = pairwise_distances(x_low, n, low_dim)?;
    let rh = rank_matrix(&dh, n);
    let rl = rank_matrix(&dl, n);
    if k == 0 || k > n - 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: format!("must be in 1..{n}"),
        });
    }
    let mut hits = 0usize;
    for i in 0..n {
        let highs: &[usize] = &rh[i][..k];
        let lows: &[usize] = &rl[i][..k];
        for &j in highs {
            if lows.contains(&j) {
                hits += 1;
            }
        }
    }
    Ok(hits as f64 / (n * k) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_zero_diagonal() {
        let n = 3;
        let dim = 2;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = pairwise_distances(&x, n, dim).expect("ok");
        for i in 0..n {
            assert!(d[i * n + i].abs() < 1e-12);
        }
    }

    #[test]
    fn trustworthiness_identity_one() {
        // If low-d copies high-d, T(k) = 1.
        let n = 6;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = 2.0 * i as f64;
        }
        let t = trustworthiness(&x, &x, n, dim, dim, 2).expect("ok");
        assert!((t - 1.0).abs() < 1e-6);
    }

    #[test]
    fn continuity_identity_one() {
        let n = 6;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = 2.0 * i as f64;
        }
        let c = continuity(&x, &x, n, dim, dim, 2).expect("ok");
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn kl_self_zero() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let kl = kl_pq(&p, &p).expect("ok");
        assert!(kl.abs() < 1e-12);
    }

    #[test]
    fn neighborhood_preservation_identity_one() {
        let n = 5;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = i as f64;
        }
        let p = neighborhood_preservation(&x, &x, n, dim, dim, 2).expect("ok");
        assert!((p - 1.0).abs() < 1e-9);
    }
}
