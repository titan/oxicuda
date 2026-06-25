//! Shampoo optimizer — Gupta, Koren & Singer, 2018.
//!
//! "Shampoo: Preconditioned Stochastic Tensor Optimization" (arXiv:1802.09568).
//!
//! Shampoo is a structure-aware full-matrix adaptive method.  For a 2-D
//! parameter (gradient) `G ∈ ℝ^{m×n}` it maintains **two** preconditioners,
//! one per tensor dimension, instead of a single `mn × mn` matrix:
//!
//! ```text
//! L ← L + G·Gᵀ        (m × m, left / row statistics)
//! R ← R + Gᵀ·G        (n × n, right / column statistics)
//! Ĝ = L^(−1/4) · G · R^(−1/4)        (preconditioned gradient)
//! W ← W − η·Ĝ
//! ```
//!
//! The inverse fourth roots `L^(−1/4)`, `R^(−1/4)` are computed via symmetric
//! eigendecomposition (Jacobi rotations) on the small per-dimension matrices,
//! which is exact for the symmetric-PSD preconditioners.  A damping term
//! `ε·I` is added before the root to keep the matrices strictly positive
//! definite.
//!
//! For a 1-D parameter, both preconditioners collapse to the diagonal and
//! Shampoo degenerates to AdaGrad-style `Ĝ = G / √(Σ G²)` per coordinate.
//!
//! All preconditioner state is stored in `f64` for the eigendecomposition's
//! numerical stability.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Shampoo`] optimizer.
#[derive(Debug, Clone)]
pub struct ShampooConfig {
    /// Learning rate `η` (must be > 0).
    pub lr: f64,
    /// Damping `ε` added as `ε·I` to each preconditioner before taking the
    /// inverse root (default 1e-4; must be > 0).
    pub epsilon: f64,
    /// Momentum coefficient on the preconditioned update (default 0; `[0, 1)`).
    pub momentum: f64,
    /// Decoupled weight-decay coefficient `λ` (default 0; ≥ 0).
    pub weight_decay: f64,
    /// Recompute the inverse roots every `precond_interval` steps (the
    /// statistics `L`, `R` are still accumulated every step).  Must be ≥ 1;
    /// default 1.
    pub precond_interval: usize,
    /// Number of Jacobi sweeps for the symmetric eigendecomposition
    /// (default 12; must be ≥ 1).
    pub jacobi_sweeps: usize,
}

impl Default for ShampooConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            epsilon: 1e-4,
            momentum: 0.0,
            weight_decay: 0.0,
            precond_interval: 1,
            jacobi_sweeps: 12,
        }
    }
}

impl ShampooConfig {
    /// Validate every field.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if `lr ≤ 0`.
    /// * [`TrainError::Internal`] for any other out-of-range field.
    fn validate(&self) -> TrainResult<()> {
        if self.lr <= 0.0 || self.lr.is_nan() {
            return Err(TrainError::InvalidLearningRate { lr: self.lr });
        }
        if self.epsilon <= 0.0 || self.epsilon.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("epsilon must be positive, got {}", self.epsilon),
            });
        }
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(TrainError::Internal {
                msg: format!("momentum must be in [0, 1), got {}", self.momentum),
            });
        }
        if self.weight_decay < 0.0 || self.weight_decay.is_nan() {
            return Err(TrainError::Internal {
                msg: format!(
                    "weight_decay must be non-negative, got {}",
                    self.weight_decay
                ),
            });
        }
        if self.precond_interval == 0 {
            return Err(TrainError::Internal {
                msg: "precond_interval must be >= 1".into(),
            });
        }
        if self.jacobi_sweeps == 0 {
            return Err(TrainError::Internal {
                msg: "jacobi_sweeps must be >= 1".into(),
            });
        }
        Ok(())
    }
}

// ─── Linear-algebra helpers (row-major dense f64) ─────────────────────────────

/// Symmetric eigendecomposition of an `n×n` matrix `a` (row-major, symmetric)
/// by the cyclic Jacobi method.  Returns `(eigenvalues, eigenvectors)` where
/// the eigenvectors are stored column-wise: column `k` is the eigenvector for
/// `eigenvalues[k]`, laid out row-major in the returned `n×n` matrix.
fn jacobi_eigh(a: &[f64], n: usize, sweeps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    // V starts as the identity; accumulates rotations (column = eigenvector).
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 1 {
        return (vec![m[0]], v);
    }
    for _ in 0..sweeps {
        // Off-diagonal magnitude — early-out once symmetric matrix is diagonal.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                // Jacobi rotation angle.
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Apply rotation to rows/cols p, q of M.
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                // Accumulate eigenvectors: V ← V·J.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    (eig, v)
}

/// Inverse `p`-th root of a symmetric PSD matrix `a` (`n×n`, row-major) with
/// damping `eps·I`: returns `(a + eps·I)^(−1/p)` as an `n×n` row-major matrix.
fn inverse_pth_root(a: &[f64], n: usize, p: f64, eps: f64, sweeps: usize) -> Vec<f64> {
    // Damp then eigendecompose.
    let mut damped = a.to_vec();
    for i in 0..n {
        damped[i * n + i] += eps;
    }
    let (eig, vecs) = jacobi_eigh(&damped, n, sweeps);
    // D^{-1/p} on eigenvalues (clamped to a small positive floor).
    let floor = eps.max(1e-30);
    let inv_root: Vec<f64> = eig
        .iter()
        .map(|&lam| {
            let l = lam.max(floor);
            l.powf(-1.0 / p)
        })
        .collect();
    // result = V · diag(inv_root) · Vᵀ.
    let mut result = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += vecs[i * n + k] * inv_root[k] * vecs[j * n + k];
            }
            result[i * n + j] = acc;
        }
    }
    result
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Per-parameter Shampoo preconditioner storage.
#[derive(Debug, Clone)]
enum Precond {
    /// 2-D parameter `(rows, cols)`: left `L` (rows×rows), right `R` (cols×cols)
    /// statistics plus their cached inverse-fourth-root preconditioners.
    Matrix {
        rows: usize,
        cols: usize,
        left: Vec<f64>,
        right: Vec<f64>,
        left_root: Vec<f64>,
        right_root: Vec<f64>,
    },
    /// 1-D parameter: diagonal AdaGrad accumulator.
    Diagonal { acc: Vec<f64> },
}

/// Shampoo optimizer operating on a single flat parameter tensor.
#[derive(Debug, Clone)]
pub struct Shampoo {
    config: ShampooConfig,
    precond: Precond,
    /// Optional momentum buffer on the preconditioned update.
    momentum_buf: Option<Vec<f64>>,
    dim: usize,
    t: u64,
}

impl Shampoo {
    /// Create a Shampoo optimizer for a parameter of `dim` elements.
    ///
    /// * `shape = Some((rows, cols))` enables the two-sided matrix
    ///   preconditioner and requires `rows * cols == dim`.
    /// * `shape = None` uses a diagonal (AdaGrad-equivalent) preconditioner.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `dim == 0`.
    /// * Any error from `ShampooConfig::validate`.
    /// * [`TrainError::ShapeMismatch`] if `rows * cols != dim`.
    pub fn new(
        dim: usize,
        shape: Option<(usize, usize)>,
        config: ShampooConfig,
    ) -> TrainResult<Self> {
        if dim == 0 {
            return Err(TrainError::EmptyParams);
        }
        config.validate()?;
        let precond = match shape {
            Some((rows, cols)) => {
                if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(dim) {
                    return Err(TrainError::ShapeMismatch {
                        expected: vec![dim],
                        got: vec![rows, cols],
                    });
                }
                // Initialise roots to identity so the first step (before the
                // statistics build up) reduces to plain SGD scaled by ε^{-1/4}.
                let mut left_root = vec![0.0_f64; rows * rows];
                let mut right_root = vec![0.0_f64; cols * cols];
                let lr_scale = config.epsilon.powf(-0.25);
                for i in 0..rows {
                    left_root[i * rows + i] = lr_scale;
                }
                for j in 0..cols {
                    right_root[j * cols + j] = lr_scale;
                }
                Precond::Matrix {
                    rows,
                    cols,
                    left: vec![0.0; rows * rows],
                    right: vec![0.0; cols * cols],
                    left_root,
                    right_root,
                }
            }
            None => Precond::Diagonal {
                acc: vec![0.0; dim],
            },
        };
        let momentum_buf = if config.momentum > 0.0 {
            Some(vec![0.0_f64; dim])
        } else {
            None
        };
        Ok(Self {
            config,
            precond,
            momentum_buf,
            dim,
            t: 0,
        })
    }

    /// Perform one Shampoo update in-place.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ShapeMismatch`] if `params`/`grads` length mismatch.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> TrainResult<()> {
        if params.len() != self.dim || grads.len() != self.dim {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![params.len(), grads.len()],
            });
        }
        self.t += 1;
        let recompute = (self.t as usize - 1) % self.config.precond_interval == 0;
        let eps = self.config.epsilon;
        let sweeps = self.config.jacobi_sweeps;

        let mut update = vec![0.0_f64; self.dim];

        match &mut self.precond {
            Precond::Matrix {
                rows,
                cols,
                left,
                right,
                left_root,
                right_root,
            } => {
                let rows = *rows;
                let cols = *cols;
                // Accumulate L += G·Gᵀ and R += Gᵀ·G.
                for i in 0..rows {
                    for k in 0..rows {
                        let mut acc = 0.0;
                        for j in 0..cols {
                            acc += f64::from(grads[i * cols + j]) * f64::from(grads[k * cols + j]);
                        }
                        left[i * rows + k] += acc;
                    }
                }
                for i in 0..cols {
                    for k in 0..cols {
                        let mut acc = 0.0;
                        for j in 0..rows {
                            acc += f64::from(grads[j * cols + i]) * f64::from(grads[j * cols + k]);
                        }
                        right[i * cols + k] += acc;
                    }
                }
                // Refresh the inverse fourth-roots periodically.
                if recompute {
                    *left_root = inverse_pth_root(left, rows, 4.0, eps, sweeps);
                    *right_root = inverse_pth_root(right, cols, 4.0, eps, sweeps);
                }
                // Ĝ = L^{-1/4} · G · R^{-1/4}.
                // Intermediate tmp = L^{-1/4} · G  (rows × cols).
                let mut tmp = vec![0.0_f64; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        let mut acc = 0.0;
                        for k in 0..rows {
                            acc += left_root[i * rows + k] * f64::from(grads[k * cols + j]);
                        }
                        tmp[i * cols + j] = acc;
                    }
                }
                for i in 0..rows {
                    for j in 0..cols {
                        let mut acc = 0.0;
                        for k in 0..cols {
                            acc += tmp[i * cols + k] * right_root[k * cols + j];
                        }
                        update[i * cols + j] = acc;
                    }
                }
            }
            Precond::Diagonal { acc } => {
                for (idx, a) in acc.iter_mut().enumerate() {
                    let g = f64::from(grads[idx]);
                    *a += g * g;
                    update[idx] = g / (a.sqrt() + eps);
                }
            }
        }

        // Optional heavy-ball momentum on the preconditioned update.
        if let (Some(mu), Some(buf)) = (
            (self.config.momentum > 0.0).then_some(self.config.momentum),
            self.momentum_buf.as_mut(),
        ) {
            for (b, u) in buf.iter_mut().zip(update.iter_mut()) {
                *b = mu * *b + *u;
                *u = *b;
            }
        }

        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        for (p, u) in params.iter_mut().zip(update.iter()) {
            let mut val = f64::from(*p);
            if wd > 0.0 {
                val -= lr * wd * val;
            }
            val -= lr * *u;
            *p = val as f32;
        }
        Ok(())
    }

    /// Reset all preconditioner state and the step counter.
    pub fn reset(&mut self) {
        match &mut self.precond {
            Precond::Matrix {
                rows,
                cols,
                left,
                right,
                left_root,
                right_root,
            } => {
                left.iter_mut().for_each(|x| *x = 0.0);
                right.iter_mut().for_each(|x| *x = 0.0);
                let lr_scale = self.config.epsilon.powf(-0.25);
                left_root.iter_mut().for_each(|x| *x = 0.0);
                right_root.iter_mut().for_each(|x| *x = 0.0);
                for i in 0..*rows {
                    left_root[i * *rows + i] = lr_scale;
                }
                for j in 0..*cols {
                    right_root[j * *cols + j] = lr_scale;
                }
            }
            Precond::Diagonal { acc } => acc.iter_mut().for_each(|x| *x = 0.0),
        }
        if let Some(buf) = self.momentum_buf.as_mut() {
            buf.iter_mut().for_each(|x| *x = 0.0);
        }
        self.t = 0;
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Whether the optimizer uses the two-sided matrix preconditioner.
    #[must_use]
    pub fn is_matrix(&self) -> bool {
        matches!(self.precond, Precond::Matrix { .. })
    }

    /// Parameter dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg(lr: f64) -> ShampooConfig {
        ShampooConfig {
            lr,
            ..Default::default()
        }
    }

    #[test]
    fn rejects_bad_shape() {
        assert!(matches!(
            Shampoo::new(7, Some((2, 2)), ShampooConfig::default()),
            Err(TrainError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_bad_lr() {
        assert!(matches!(
            Shampoo::new(4, None, cfg(0.0)),
            Err(TrainError::InvalidLearningRate { .. })
        ));
    }

    #[test]
    fn matrix_flag() {
        assert!(
            Shampoo::new(6, Some((2, 3)), cfg(1e-2))
                .expect("valid")
                .is_matrix()
        );
        assert!(!Shampoo::new(6, None, cfg(1e-2)).expect("valid").is_matrix());
    }

    /// Jacobi eigendecomposition reconstructs a known symmetric matrix:
    /// A = V·diag(λ)·Vᵀ should hold to high precision.
    #[test]
    fn jacobi_reconstructs() {
        // Symmetric 3×3.
        let a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let n = 3;
        let (eig, v) = jacobi_eigh(&a, n, 30);
        // Reconstruct.
        let mut recon = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += v[i * n + k] * eig[k] * v[j * n + k];
                }
                recon[i * n + j] = acc;
            }
        }
        for (r, orig) in recon.iter().zip(a.iter()) {
            assert!((r - orig).abs() < 1e-9, "recon {r} vs {orig}");
        }
    }

    /// Inverse fourth root squared-twice returns the (damped) inverse:
    /// (A^{-1/4})^4 · (A + εI) ≈ I.
    #[test]
    fn inverse_pth_root_correct() {
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let n = 2;
        let eps = 1e-6;
        let root = inverse_pth_root(&a, n, 4.0, eps, 30);
        // Compute root^4.
        let matmul = |x: &[f64], y: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..n {
                        acc += x[i * n + k] * y[k * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            out
        };
        let r2 = matmul(&root, &root);
        let r4 = matmul(&r2, &r2);
        // damped A.
        let mut damped = a.clone();
        damped[0] += eps;
        damped[3] += eps;
        let prod = matmul(&r4, &damped);
        // prod ≈ identity.
        for i in 0..n {
            for j in 0..n {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i * n + j] - expect).abs() < 1e-6,
                    "prod[{i},{j}] = {} != {expect}",
                    prod[i * n + j]
                );
            }
        }
    }

    #[test]
    fn diagonal_converges_quadratic() {
        let dim = 6;
        let mut opt = Shampoo::new(dim, None, cfg(0.3)).expect("valid");
        let mut params = vec![1.0_f32; dim];
        for _ in 0..400 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(max_abs < 1e-2, "diagonal Shampoo not converged: {max_abs}");
    }

    /// Two-sided matrix Shampoo minimises the convex quadratic f(W)=Σ Wᵢ²
    /// below tolerance from a random initialisation.
    #[test]
    fn matrix_converges_quadratic() {
        let (rows, cols) = (3, 4);
        let dim = rows * cols;
        let mut opt = Shampoo::new(dim, Some((rows, cols)), cfg(0.2)).expect("valid");
        let mut rng = LcgRng::new(7);
        let mut params: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        for _ in 0..500 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(max_abs < 5e-2, "matrix Shampoo not converged: {max_abs}");
    }

    #[test]
    fn momentum_converges() {
        let mut c = cfg(0.2);
        c.momentum = 0.9;
        let dim = 4;
        let mut opt = Shampoo::new(dim, None, c).expect("valid");
        let mut params = vec![1.0_f32; dim];
        for _ in 0..600 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(max_abs < 5e-2, "momentum Shampoo not converged: {max_abs}");
    }

    #[test]
    fn wrong_length_errors() {
        let mut opt = Shampoo::new(4, None, cfg(1e-2)).expect("valid");
        let mut p = vec![1.0_f32; 4];
        assert!(matches!(
            opt.step(&mut p, &[0.1]),
            Err(TrainError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn reset_clears_state() {
        let mut opt = Shampoo::new(6, Some((2, 3)), cfg(1e-2)).expect("valid");
        let mut p = vec![1.0_f32; 6];
        opt.step(&mut p, &[0.5_f32; 6]).expect("ok");
        assert_eq!(opt.step_count(), 1);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
    }

    /// The `precond_interval` path: refreshing the roots every 5 steps should
    /// still converge (statistics accumulate every step regardless).
    #[test]
    fn lazy_precond_converges() {
        let mut c = cfg(0.2);
        c.precond_interval = 5;
        let (rows, cols) = (2, 3);
        let dim = rows * cols;
        let mut opt = Shampoo::new(dim, Some((rows, cols)), c).expect("valid");
        let mut params = vec![0.8_f32; dim];
        for _ in 0..600 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(
            max_abs < 5e-2,
            "lazy-precond Shampoo not converged: {max_abs}"
        );
    }
}
