//! Penalised matrix decomposition for Sparse PCA (Witten, Tibshirani & Hastie 2009).
//!
//! For a centred data matrix `X` (`n × p`), find rank-1 components `u`, `v` minimising
//! `½ ||X − σ u v^T||²` subject to `||u||₂ ≤ 1`, `||v||₂ ≤ 1`, `||v||₁ ≤ c₁`.
//!
//! Solved by alternation:
//!   u = X v / ||X v||
//!   v = S_λ(X^T u) / ||S_λ(X^T u)||  (with λ chosen by binary search to enforce ||v||₁ ≤ c₁)
//!
//! For multiple components, deflate `X ← X − σ u v^T` and repeat.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec};
use crate::thresholding::iht::soft_threshold;

/// Sparse PCA result containing components and singular values.
#[derive(Debug, Clone)]
pub struct SparsePcaResult {
    /// `u` columns stacked as `n × k` row-major (one column per component).
    pub u: Vec<f64>,
    /// `v` columns stacked as `p × k` row-major.
    pub v: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub iterations: Vec<usize>,
}

fn l2_normalise(x: &mut [f64]) {
    let nrm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-300);
    for v in x.iter_mut() {
        *v /= nrm;
    }
}

fn l1_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

/// Compute `k` sparse principal components with L1 budget `c1` on each `v`.
pub fn sparse_pca_witten(
    x: &[f64],
    n: usize,
    p: usize,
    k: usize,
    c1: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<SparsePcaResult> {
    if x.len() != n * p {
        return Err(CsError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![x.len()],
        });
    }
    if k == 0 || k > n.min(p) {
        return Err(CsError::InvalidRank(k));
    }
    if c1 <= 0.0 {
        return Err(CsError::InvalidParameter("c1 must be > 0".into()));
    }
    let mut residual = x.to_vec();
    let mut u_cols = vec![0.0_f64; n * k];
    let mut v_cols = vec![0.0_f64; p * k];
    let mut sigmas = vec![0.0_f64; k];
    let mut iters = vec![0_usize; k];
    for comp in 0..k {
        // Initialise v as the first row of `residual` (normalised) — could randomise instead.
        let mut v = vec![0.0_f64; p];
        v[..p].copy_from_slice(&residual[..p]);
        l2_normalise(&mut v);
        let mut u = vec![0.0_f64; n];
        let mut sigma = 0.0_f64;
        for it in 0..max_iter {
            // u update: u = X v / ||X v||
            u = mat_vec(&residual, n, p, &v)?;
            l2_normalise(&mut u);
            // v update: pick lambda s.t. ||S_lambda(X^T u)||_1 ≤ c1; binary-search.
            let xtu = mat_t_vec(&residual, n, p, &u)?;
            let max_abs = xtu.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let mut lo = 0.0_f64;
            let mut hi = max_abs + 1.0;
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                let cand = soft_threshold(&xtu, mid);
                let nrm = cand.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-300);
                let v_cand: Vec<f64> = cand.iter().map(|v| v / nrm).collect();
                let l1 = l1_norm(&v_cand);
                if l1 > c1 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let cand = soft_threshold(&xtu, hi);
            let nrm = cand.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-300);
            let v_new: Vec<f64> = cand.iter().map(|v| v / nrm).collect();
            // Compute sigma.
            sigma = mat_vec(&residual, n, p, &v_new)?
                .iter()
                .zip(u.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            // Convergence check.
            let mut delta = 0.0_f64;
            for j in 0..p {
                let d = v_new[j] - v[j];
                delta += d * d;
            }
            v = v_new;
            iters[comp] = it + 1;
            if delta.sqrt() < tol {
                break;
            }
        }
        // Store and deflate.
        for i in 0..n {
            u_cols[i * k + comp] = u[i];
        }
        for j in 0..p {
            v_cols[j * k + comp] = v[j];
        }
        sigmas[comp] = sigma;
        // residual = residual - sigma * u v^T
        for i in 0..n {
            for j in 0..p {
                residual[i * p + j] -= sigma * u[i] * v[j];
            }
        }
    }
    Ok(SparsePcaResult {
        u: u_cols,
        v: v_cols,
        sigmas,
        iterations: iters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_pca_rank1_runs() {
        // Rank-1 data: x = u v^T with sparse v.
        let n = 8;
        let p = 6;
        let mut data = vec![0.0_f64; n * p];
        for i in 0..n {
            data[i * p] = (i as f64) - 3.5; // column 0 only
        }
        let r = sparse_pca_witten(&data, n, p, 1, 1.5, 50, 1.0e-7).expect("ok");
        // v should put most mass on j=0.
        assert!(r.v[0].abs() > 0.5);
    }
}
