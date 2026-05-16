//! Kernel Principal Component Analysis.
//!
//! Steps:
//! 1. Build kernel Gram matrix `K_ij = k(x_i, x_j)`.
//! 2. Centre `K` via the double-centering operator.
//! 3. Eigendecompose `K_centered`.
//! 4. Sort by descending eigenvalue, take top `n_components`.
//! 5. Embedding is `alpha_k * sqrt(lambda_k)`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Kernel type.
#[derive(Debug, Clone, Copy)]
pub enum KernelKind {
    /// Gaussian RBF: `k(x,y) = exp(-||x-y||^2 / (2 sigma^2))`.
    Gaussian { sigma: f64 },
    /// Polynomial: `k(x,y) = (x . y + c)^d`.
    Polynomial { degree: usize, coef0: f64 },
    /// Linear: `k(x,y) = x . y`.
    Linear,
}

/// Result of a kernel PCA fit.
pub struct KernelPcaResult {
    /// Centered Gram matrix (n x n).
    pub gram_centered: Vec<f64>,
    /// Eigenvalues sorted descending (length n_components).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors stored row-major as (n x n_components).
    pub eigenvectors: Vec<f64>,
    /// Embedding (n_samples x n_components).
    pub projection: Vec<f64>,
}

/// Fit kernel PCA on row-major data `x` of shape `n_samples x dim`.
pub fn kernel_pca(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_components: usize,
    kernel: KernelKind,
) -> ManifoldResult<KernelPcaResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components > n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..={n_samples}, got {n_components}"),
        });
    }
    let n = n_samples;
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let val = kernel_eval(
                &x[i * dim..i * dim + dim],
                &x[j * dim..j * dim + dim],
                kernel,
            );
            k[i * n + j] = val;
            k[j * n + i] = val;
        }
    }
    // Centre: K_c = K - 1_n K - K 1_n + 1_n K 1_n where 1_n is the matrix of 1/n
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            row_mean[i] += k[i * n + j];
            col_mean[j] += k[i * n + j];
            total += k[i * n + j];
        }
    }
    for v in &mut row_mean {
        *v /= n as f64;
    }
    for v in &mut col_mean {
        *v /= n as f64;
    }
    total /= (n * n) as f64;
    let mut k_c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            k_c[i * n + j] = k[i * n + j] - row_mean[i] - col_mean[j] + total;
        }
    }
    // Eigendecompose (symmetric)
    let (mut w, mut v) = jacobi_eigh(&k_c, n)?;
    sort_eigen_descending(&mut w, &mut v, n);
    // Take top n_components
    let mut eigenvalues = vec![0.0; n_components];
    let mut eigenvectors = vec![0.0; n * n_components];
    for c in 0..n_components {
        eigenvalues[c] = w[c];
        for r in 0..n {
            eigenvectors[r * n_components + c] = v[r * n + c];
        }
    }
    // Embedding: Y_ic = alpha_ic * sqrt(lambda_c) when lambda > 0
    let mut projection = vec![0.0; n * n_components];
    for c in 0..n_components {
        let sqrt_lam = eigenvalues[c].max(0.0).sqrt();
        for r in 0..n {
            projection[r * n_components + c] = eigenvectors[r * n_components + c] * sqrt_lam;
        }
    }
    Ok(KernelPcaResult {
        gram_centered: k_c,
        eigenvalues,
        eigenvectors,
        projection,
    })
}

fn kernel_eval(a: &[f64], b: &[f64], kind: KernelKind) -> f64 {
    match kind {
        KernelKind::Gaussian { sigma } => {
            let mut s = 0.0;
            for (x, y) in a.iter().zip(b) {
                let d = x - y;
                s += d * d;
            }
            let denom = 2.0 * sigma * sigma;
            (-s / denom).exp()
        }
        KernelKind::Polynomial { degree, coef0 } => {
            let mut s = 0.0;
            for (x, y) in a.iter().zip(b) {
                s += x * y;
            }
            (s + coef0).powi(degree as i32)
        }
        KernelKind::Linear => {
            let mut s = 0.0;
            for (x, y) in a.iter().zip(b) {
                s += x * y;
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_kernel_matches_pca_topcomp() {
        // For linear kernel, Kernel PCA recovers the standard PCA scores up to sign.
        let n = 5;
        let dim = 2;
        let x = vec![-2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let r = kernel_pca(&x, n, dim, 1, KernelKind::Linear).expect("ok");
        let vals: Vec<f64> = (0..n).map(|i| r.projection[i].abs()).collect();
        // Largest projection magnitude corresponds to the most extreme point
        assert!(vals[0] > vals[2]);
        assert!(vals[4] > vals[2]);
    }

    #[test]
    fn gaussian_kernel_runs() {
        let n = 6;
        let dim = 2;
        let x: Vec<f64> = (0..n * dim).map(|k| (k % 4) as f64 - 1.5).collect();
        let r = kernel_pca(&x, n, dim, 2, KernelKind::Gaussian { sigma: 1.0 }).expect("ok");
        assert_eq!(r.projection.len(), n * 2);
        assert!(r.eigenvalues[0] >= r.eigenvalues[1]);
    }

    #[test]
    fn polynomial_kernel_runs() {
        let n = 4;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|k| 0.1 * k as f64).collect();
        let r = kernel_pca(
            &x,
            n,
            dim,
            2,
            KernelKind::Polynomial {
                degree: 2,
                coef0: 1.0,
            },
        )
        .expect("ok");
        assert_eq!(r.eigenvalues.len(), 2);
    }
}
