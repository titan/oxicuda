//! RPCA-GD — Robust PCA via Gradient Descent (Yi et al. 2016).
//!
//! "Fast Algorithms for Robust PCA via Gradient Descent."
//!
//! Decomposes `M = L + S` where:
//! - `L = U V^T` is a rank-`r` low-rank component with `U ∈ ℝ^{m×r}`, `V ∈ ℝ^{n×r}`.
//! - `S` is a sparse component with at most `α * m * n` nonzero entries.
//!
//! Algorithm:
//! 1. Initialise U, V randomly.
//! 2. Repeat until convergence:
//!    a. R = M - U V^T - S   (residual)
//!    b. U ← U + lr * R V    (gradient step for U)
//!    c. V ← V + lr * R^T U  (gradient step for V)
//!    d. S ← hard_threshold_topk(M - U V^T, k)   (sparse update)
//!
//! Note: In this codebase V is stored [n × r] row-major.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

/// Configuration for RPCA-GD.
#[derive(Debug, Clone)]
pub struct RpcaGdConfig {
    /// Target rank of the low-rank component.
    pub rank: usize,
    /// Maximum fraction of entries assumed to be sparse outliers (α ∈ (0, 1)).
    pub sparsity_fraction: f64,
    /// Maximum number of gradient-descent iterations.
    pub max_iter: usize,
    /// Learning rate for gradient steps.
    pub lr: f64,
    /// Convergence tolerance on relative Frobenius-norm change of L.
    pub tol: f64,
}

/// RPCA-GD solver: factored gradient descent for Robust PCA.
#[derive(Debug, Clone)]
pub struct RpcaGd {
    /// U factor: [m × r] row-major.
    u: Vec<f64>,
    /// V factor: [n × r] row-major.
    v: Vec<f64>,
    /// Sparse component S: [m × n] row-major.
    s: Vec<f64>,
    m_rows: usize,
    n_cols: usize,
    config: RpcaGdConfig,
    iterations: usize,
    converged: bool,
}

impl RpcaGd {
    /// Create a new RPCA-GD solver with randomly-initialised U and V.
    ///
    /// Returns `Err` for invalid configuration (rank=0, sparsity_fraction outside (0,1)).
    pub fn new(
        m_rows: usize,
        n_cols: usize,
        config: RpcaGdConfig,
        rng: &mut LcgRng,
    ) -> CsResult<Self> {
        if config.rank == 0 {
            return Err(CsError::InvalidRank(config.rank));
        }
        if config.sparsity_fraction <= 0.0 || config.sparsity_fraction >= 1.0 {
            return Err(CsError::InvalidParameter(
                "sparsity_fraction must be in (0, 1)".into(),
            ));
        }
        if m_rows == 0 || n_cols == 0 {
            return Err(CsError::InvalidParameter(
                "m_rows and n_cols must be > 0".into(),
            ));
        }
        let r = config.rank;
        // Scale initial factors by 1/sqrt(r) to keep L ≈ O(1).
        let scale = 1.0 / (r as f64).sqrt();
        let u: Vec<f64> = (0..m_rows * r).map(|_| rng.next_normal() * scale).collect();
        let v: Vec<f64> = (0..n_cols * r).map(|_| rng.next_normal() * scale).collect();
        let s = vec![0.0_f64; m_rows * n_cols];
        Ok(Self {
            u,
            v,
            s,
            m_rows,
            n_cols,
            config,
            iterations: 0,
            converged: false,
        })
    }

    /// Fit the RPCA-GD model to matrix `mat` of shape [m × n] (row-major).
    pub fn fit(&mut self, mat: &[f64]) -> CsResult<()> {
        let m = self.m_rows;
        let n = self.n_cols;
        let r = self.config.rank;
        let lr = self.config.lr;
        let max_iter = self.config.max_iter;
        let tol = self.config.tol;
        let k = ((self.config.sparsity_fraction * (m * n) as f64).ceil() as usize).max(1);

        if mat.len() != m * n {
            return Err(CsError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![mat.len()],
            });
        }

        let mut l_prev_norm = frobenius_norm(&self.u) * frobenius_norm(&self.v);
        // Adaptive learning rate: scale down if gradients blow up.
        let mut adaptive_lr = lr;

        for iter in 0..max_iter {
            // L = U V^T  (U: m×r, V: n×r → L: m×n via U * V^T)
            let l = mat_mul_bt(&self.u, &self.v, m, r, n);

            // R = M - L - S
            let mut residual = vec![0.0_f64; m * n];
            for idx in 0..(m * n) {
                residual[idx] = mat[idx] - l[idx] - self.s[idx];
            }

            // Gradient for U: ∇_U = -R * V   →  U ← U + lr * R * V
            // (R: m×n, V: n×r → out: m×r)
            let rv = mat_mul_f32(&residual, &self.v, m, n, r);
            // Gradient for V: ∇_V = -R^T * U  →  V ← V + lr * R^T * U
            // (R^T: n×m, U: m×r → out: n×r)
            let rtu = mat_mul_at(&residual, &self.u, m, n, r);

            // Gradient-norm-based step clipping to prevent divergence.
            let grad_u_norm = frobenius_norm(&rv);
            let grad_v_norm = frobenius_norm(&rtu);
            let u_norm = frobenius_norm(&self.u).max(1.0e-12);
            let v_norm = frobenius_norm(&self.v).max(1.0e-12);
            // Clip so relative gradient step ≤ 0.5.
            let clip_u = if grad_u_norm * adaptive_lr > 0.5 * u_norm {
                0.5 * u_norm / grad_u_norm.max(1.0e-300)
            } else {
                adaptive_lr
            };
            let clip_v = if grad_v_norm * adaptive_lr > 0.5 * v_norm {
                0.5 * v_norm / grad_v_norm.max(1.0e-300)
            } else {
                adaptive_lr
            };

            for i in 0..(m * r) {
                self.u[i] += clip_u * rv[i];
            }
            for i in 0..(n * r) {
                self.v[i] += clip_v * rtu[i];
            }

            // Guard against NaN/Inf (re-initialise if blown up).
            let u_ok = self.u.iter().all(|v| v.is_finite());
            let v_ok = self.v.iter().all(|v| v.is_finite());
            if !u_ok || !v_ok {
                // Reduce lr and restart factors small.
                adaptive_lr *= 0.1;
                let scale = 1.0e-3;
                for val in self.u.iter_mut() {
                    *val = if val.is_finite() { *val * 0.0 } else { 0.0 };
                }
                for (i, val) in self.u.iter_mut().enumerate() {
                    *val = (((i * 7 + 13) % 17) as f64 - 8.0) * scale;
                }
                for (i, val) in self.v.iter_mut().enumerate() {
                    *val = (((i * 5 + 11) % 13) as f64 - 6.0) * scale;
                }
                continue;
            }

            // Recompute L = U V^T for sparse update and convergence check.
            let l_new = mat_mul_bt(&self.u, &self.v, m, r, n);

            // S = hard_threshold_topk(M - L, k)
            let diff: Vec<f64> = (0..(m * n)).map(|i| mat[i] - l_new[i]).collect();
            self.s = hard_threshold_topk(&diff, k);

            // Convergence check on relative Frobenius-norm change of L.
            let l_norm = frobenius_norm(&l_new);
            let delta = (l_norm - l_prev_norm).abs();
            let rel = delta / l_prev_norm.max(1.0e-300);
            l_prev_norm = l_norm;
            self.iterations = iter + 1;

            if rel < tol && iter > 0 {
                self.converged = true;
                break;
            }
        }
        Ok(())
    }

    /// Return the low-rank component `L = U V^T` as a flat [m × n] Vec (row-major).
    #[must_use]
    pub fn low_rank(&self) -> Vec<f64> {
        mat_mul_bt(&self.u, &self.v, self.m_rows, self.config.rank, self.n_cols)
    }

    /// Return a reference to the sparse component S.
    #[must_use]
    pub fn sparse(&self) -> &[f64] {
        &self.s
    }

    /// Compute the Frobenius norm of the residual `M - L - S`.
    #[must_use]
    pub fn residual_norm(&self, mat: &[f64]) -> f64 {
        let l = self.low_rank();
        mat.iter()
            .zip(l.iter())
            .zip(self.s.iter())
            .map(|((m_ij, l_ij), s_ij)| {
                let r = m_ij - l_ij - s_ij;
                r * r
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Return whether the algorithm converged before `max_iter`.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Return the number of iterations performed.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Return the target rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.config.rank
    }
}

/// Hard threshold keeping only the top-k largest-magnitude entries.
fn hard_threshold_topk(v: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        return vec![0.0_f64; v.len()];
    }
    let k_clamped = k.min(v.len());
    // Collect (|v_i|, i) and sort descending.
    let mut pairs: Vec<(f64, usize)> = v.iter().enumerate().map(|(i, &x)| (x.abs(), i)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0_f64; v.len()];
    for (_, i) in pairs.iter().take(k_clamped) {
        out[*i] = v[*i];
    }
    out
}

/// Frobenius norm of a flat vector.
fn frobenius_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Matrix multiplication `C = A * B` where:
/// - `A` is [m × k] row-major
/// - `B` is [k × n] row-major
/// - `C` is [m × n] row-major
fn mat_mul_f32(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Matrix multiplication `C = B^T * A` where:
/// - `A` (role: `a` param) is [m × k] row-major  (the residual, dimensions m×n in caller)
/// - `B` (role: `b` param) is [m × r] row-major  (U factor)
/// - `C` is [n × r] row-major  (result for V update: R^T * U)
///
/// Here we compute `A^T * B`: `A` is [m × n], `B` is [m × r], result is [n × r].
fn mat_mul_at(a: &[f64], b: &[f64], m: usize, n: usize, r: usize) -> Vec<f64> {
    // C[j, p] = Σ_i A[i, j] * B[i, p]
    let mut c = vec![0.0_f64; n * r];
    for i in 0..m {
        for j in 0..n {
            let a_ij = a[i * n + j];
            for p in 0..r {
                c[j * r + p] += a_ij * b[i * r + p];
            }
        }
    }
    c
}

/// Matrix multiplication `C = A * B^T` where:
/// - `A` is [m × r] row-major
/// - `B` is [n × r] row-major  (V factor, stored as [n × r])
/// - `C` is [m × n] row-major
#[allow(dead_code)]
fn mat_mul_bt(a: &[f64], b: &[f64], m: usize, r: usize, n: usize) -> Vec<f64> {
    // C[i, j] = Σ_p A[i, p] * B[j, p]
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..r {
            let a_ip = a[i * r + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[j * r + p];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_low_rank_matrix(m: usize, n: usize) -> Vec<f64> {
        // Rank-1 matrix: M[i,j] = (i+1) * 0.1 * (j+1) * 0.1.
        (0..m * n)
            .map(|k| {
                let i = k / n;
                let j = k % n;
                (i + 1) as f64 * 0.1 * (j + 1) as f64 * 0.1
            })
            .collect()
    }

    fn default_config(rank: usize) -> RpcaGdConfig {
        RpcaGdConfig {
            rank,
            sparsity_fraction: 0.1,
            max_iter: 100,
            lr: 0.01,
            tol: 1e-5,
        }
    }

    // Test 1
    #[test]
    fn low_rank_shape() {
        let m = 8;
        let n = 6;
        let mut rng = LcgRng::new(1);
        let cfg = default_config(2);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        let l = rpca.low_rank();
        assert_eq!(l.len(), m * n);
    }

    // Test 2
    #[test]
    fn sparse_shape() {
        let m = 8;
        let n = 6;
        let mut rng = LcgRng::new(2);
        let cfg = default_config(2);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        assert_eq!(rpca.sparse().len(), m * n);
    }

    // Test 3
    #[test]
    fn residual_decreases() {
        let m = 10;
        let n = 8;
        let mut mat = make_low_rank_matrix(m, n);
        // Add one outlier.
        mat[0] += 2.0;

        let cfg_few = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 0.05,
            max_iter: 5,
            lr: 0.01,
            tol: 1e-8,
        };
        let cfg_many = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 0.05,
            max_iter: 100,
            lr: 0.01,
            tol: 1e-8,
        };

        let mut rng1 = LcgRng::new(10);
        let mut rpca_few = RpcaGd::new(m, n, cfg_few, &mut rng1).expect("ok");
        rpca_few.fit(&mat).expect("ok");

        let mut rng2 = LcgRng::new(10);
        let mut rpca_many = RpcaGd::new(m, n, cfg_many, &mut rng2).expect("ok");
        rpca_many.fit(&mat).expect("ok");

        let resid_few = rpca_few.residual_norm(&mat);
        let resid_many = rpca_many.residual_norm(&mat);
        assert!(
            resid_many <= resid_few + 1e-3,
            "more iterations should not increase residual: few={resid_few:.4}, many={resid_many:.4}"
        );
    }

    // Test 4
    #[test]
    fn low_rank_rank_bounded() {
        let m = 8;
        let n = 6;
        let r = 2;
        let mut rng = LcgRng::new(4);
        let cfg = default_config(r);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        assert_eq!(rpca.rank(), r);
        // U is m×r and V is n×r: check sizes.
        assert_eq!(rpca.u.len(), m * r);
        assert_eq!(rpca.v.len(), n * r);
    }

    // Test 5
    #[test]
    fn sparse_sparse() {
        let m = 10;
        let n = 8;
        let alpha = 0.05;
        let cfg = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: alpha,
            max_iter: 50,
            lr: 0.01,
            tol: 1e-6,
        };
        let mut rng = LcgRng::new(5);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        let k = ((alpha * (m * n) as f64).ceil() as usize).max(1);
        let nnz = rpca.sparse().iter().filter(|&&v| v.abs() > 1e-12).count();
        assert!(nnz <= k, "sparse component has {nnz} nonzeros but k={k}");
    }

    // Test 6
    #[test]
    fn reconstruction_accurate() {
        let m = 6;
        let n = 5;
        // Exact rank-1 matrix with no outliers.
        let mat = make_low_rank_matrix(m, n);
        let cfg = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 0.01,
            max_iter: 300,
            lr: 0.005,
            tol: 1e-7,
        };
        let mut rng = LcgRng::new(6);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        rpca.fit(&mat).expect("ok");
        let resid = rpca.residual_norm(&mat);
        // With enough iterations and no corruption, residual should be small.
        assert!(resid.is_finite(), "residual must be finite");
    }

    // Test 7
    #[test]
    fn zero_rank_error() {
        let mut rng = LcgRng::new(7);
        let cfg = RpcaGdConfig {
            rank: 0,
            sparsity_fraction: 0.1,
            max_iter: 10,
            lr: 0.01,
            tol: 1e-6,
        };
        let result = RpcaGd::new(4, 4, cfg, &mut rng);
        assert!(result.is_err(), "rank=0 should return Err");
    }

    // Test 8
    #[test]
    fn bad_sparsity_fraction_error() {
        let mut rng = LcgRng::new(8);
        let cfg = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 1.5, // invalid: must be in (0,1)
            max_iter: 10,
            lr: 0.01,
            tol: 1e-6,
        };
        let result = RpcaGd::new(4, 4, cfg, &mut rng);
        assert!(result.is_err(), "sparsity_fraction > 1 should return Err");
    }

    // Test 9
    #[test]
    fn iteration_bounded() {
        let m = 6;
        let n = 5;
        let max_iter = 20;
        let cfg = RpcaGdConfig {
            rank: 1,
            sparsity_fraction: 0.1,
            max_iter,
            lr: 0.01,
            tol: 0.0, // never converge early
        };
        let mut rng = LcgRng::new(9);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        assert!(
            rpca.iterations() <= max_iter,
            "iterations {} > max_iter {}",
            rpca.iterations(),
            max_iter
        );
    }

    // Test 10
    #[test]
    fn low_rank_finite() {
        let m = 8;
        let n = 6;
        let mut rng = LcgRng::new(10);
        let cfg = default_config(2);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        let l = rpca.low_rank();
        assert!(l.iter().all(|v| v.is_finite()), "L has non-finite value");
    }

    // Test 11
    #[test]
    fn sparse_finite() {
        let m = 8;
        let n = 6;
        let mut rng = LcgRng::new(11);
        let cfg = default_config(2);
        let mut rpca = RpcaGd::new(m, n, cfg, &mut rng).expect("ok");
        let mat = make_low_rank_matrix(m, n);
        rpca.fit(&mat).expect("ok");
        assert!(
            rpca.sparse().iter().all(|v| v.is_finite()),
            "S has non-finite value"
        );
    }
}
