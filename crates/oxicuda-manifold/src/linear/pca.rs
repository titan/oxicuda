//! Principal Component Analysis via covariance eigendecomposition.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// PCA fit result.
pub struct PcaResult {
    /// Mean vector (length `dim`).
    pub mean: Vec<f64>,
    /// Principal components — rows are components (`n_components x dim`).
    pub components: Vec<f64>,
    /// Explained variance per component (length `n_components`).
    pub explained_variance: Vec<f64>,
    /// Projected data (`n_samples x n_components`).
    pub projection: Vec<f64>,
}

/// Fit PCA on row-major data of shape `n_samples x dim`.
///
/// Pipeline:
/// 1. Subtract column means (centering).
/// 2. Compute covariance `Sigma = X^T X / (n - 1)` (bias-corrected).
/// 3. Eigendecompose `Sigma`.
/// 4. Sort by descending eigenvalue and take top `k = n_components`.
/// 5. Project: `Y = X_centered * V_top`.
pub fn pca_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_components: usize,
) -> ManifoldResult<PcaResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components > dim {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..={dim}, got {n_components}"),
        });
    }

    // 1. Compute column means
    let mut mean = vec![0.0; dim];
    for i in 0..n_samples {
        for j in 0..dim {
            mean[j] += x[i * dim + j];
        }
    }
    for m in &mut mean {
        *m /= n_samples as f64;
    }

    // 2. Center the data into a new buffer
    let mut centered = vec![0.0; n_samples * dim];
    for i in 0..n_samples {
        for j in 0..dim {
            centered[i * dim + j] = x[i * dim + j] - mean[j];
        }
    }

    // 3. Covariance Sigma = (X^T X) / (n - 1)
    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let mut sigma = vec![0.0; dim * dim];
    for j in 0..dim {
        for k in j..dim {
            let mut acc = 0.0;
            for i in 0..n_samples {
                acc += centered[i * dim + j] * centered[i * dim + k];
            }
            let v = acc / denom;
            sigma[j * dim + k] = v;
            sigma[k * dim + j] = v;
        }
    }

    // 4. Eigendecomposition (Sigma = V diag(w) V^T)
    let (mut w, mut v) = jacobi_eigh(&sigma, dim)?;
    sort_eigen_descending(&mut w, &mut v, dim);

    // 5. Take top n_components rows of components (transpose first columns of V)
    let mut components = vec![0.0; n_components * dim];
    let mut explained_variance = vec![0.0; n_components];
    for c in 0..n_components {
        explained_variance[c] = w[c];
        for r in 0..dim {
            components[c * dim + r] = v[r * dim + c];
        }
    }

    // 6. Project: Y = X_centered * V_top
    let mut projection = vec![0.0; n_samples * n_components];
    for i in 0..n_samples {
        for c in 0..n_components {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += centered[i * dim + j] * components[c * dim + j];
            }
            projection[i * n_components + c] = acc;
        }
    }

    Ok(PcaResult {
        mean,
        components,
        explained_variance,
        projection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_axis_aligned() {
        // Data spread along axis 0 only
        let n = 5;
        let dim = 2;
        let x = vec![-2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let r = pca_fit(&x, n, dim, 1).expect("ok");
        // First principal component should be axis 0
        assert!((r.components[0].abs() - 1.0).abs() < 1e-8);
        assert!(r.components[1].abs() < 1e-8);
        assert!(r.explained_variance[0] > 0.0);
    }

    #[test]
    fn pca_mean_centered() {
        let n = 3;
        let dim = 2;
        let x = vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0];
        let r = pca_fit(&x, n, dim, 1).expect("ok");
        assert!((r.mean[0] - 4.0).abs() < 1e-10);
        assert!((r.mean[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn pca_projection_dims() {
        let n = 6;
        let dim = 3;
        let x: Vec<f64> = (0..n * dim).map(|k| (k % 5) as f64 - 2.5).collect();
        let r = pca_fit(&x, n, dim, 2).expect("ok");
        assert_eq!(r.projection.len(), n * 2);
        assert_eq!(r.components.len(), 2 * dim);
    }

    #[test]
    fn pca_explained_variance_descending() {
        let n = 8;
        let dim = 4;
        let mut x = vec![0.0; n * dim];
        // Strong variance on axis 0, less on axis 1, etc.
        for i in 0..n {
            let v = i as f64 - 3.5;
            x[i * dim] = 3.0 * v;
            x[i * dim + 1] = 1.5 * v;
            x[i * dim + 2] = 0.5 * v;
            x[i * dim + 3] = 0.1 * v;
        }
        let r = pca_fit(&x, n, dim, 4).expect("ok");
        for w in 0..3 {
            assert!(
                r.explained_variance[w] >= r.explained_variance[w + 1] - 1e-9,
                "{} >= {} ?",
                r.explained_variance[w],
                r.explained_variance[w + 1]
            );
        }
    }
}
