//! FastICA fixed-point algorithm for Independent Component Analysis.
//!
//! Pipeline:
//! 1. Centre data (subtract column means).
//! 2. Whiten via PCA: `X_w = D^{-1/2} E^T X` where `Sigma = E D E^T`.
//! 3. Iterate fixed-point: `w <- E[g(w^T x) x] - E[g'(w^T x)] w`.
//! 4. Symmetric orthogonalisation: `W <- (W W^T)^{-1/2} W`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::linalg::householder_qr::polar_orthogonal;
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Non-linearity used in FastICA contrast function.
#[derive(Debug, Clone, Copy)]
pub enum IcaNonlinearity {
    /// `g(u) = tanh(u)`, `g'(u) = 1 - tanh^2(u)`.
    Tanh,
    /// `g(u) = u * exp(-u^2/2)`, `g'(u) = (1 - u^2) exp(-u^2/2)`.
    Gauss,
}

/// FastICA result.
pub struct IcaResult {
    /// Column means (length `dim`).
    pub mean: Vec<f64>,
    /// Whitening matrix (`n_components x dim`).
    pub whitening: Vec<f64>,
    /// Unmixing matrix (`n_components x n_components`) acting on whitened data.
    pub w_matrix: Vec<f64>,
    /// Recovered independent sources (`n_samples x n_components`).
    pub sources: Vec<f64>,
}

/// Fit FastICA on data of shape `n_samples x dim`.
pub fn fast_ica(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    nonlin: IcaNonlinearity,
    rng: &mut LcgRng,
) -> ManifoldResult<IcaResult> {
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
            reason: format!("must be in 1..={dim}"),
        });
    }
    // Centre
    let mut mean = vec![0.0; dim];
    for i in 0..n_samples {
        for j in 0..dim {
            mean[j] += x[i * dim + j];
        }
    }
    for m in &mut mean {
        *m /= n_samples as f64;
    }
    let mut centered = vec![0.0; n_samples * dim];
    for i in 0..n_samples {
        for j in 0..dim {
            centered[i * dim + j] = x[i * dim + j] - mean[j];
        }
    }
    // Covariance
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
    let (mut w, mut v) = jacobi_eigh(&sigma, dim)?;
    sort_eigen_descending(&mut w, &mut v, dim);
    // Whitening matrix: K = D^{-1/2} E^T (top n_components rows)
    let mut whitening = vec![0.0; n_components * dim];
    for c in 0..n_components {
        let inv_sqrt = 1.0 / w[c].max(1e-14).sqrt();
        for r in 0..dim {
            whitening[c * dim + r] = inv_sqrt * v[r * dim + c];
        }
    }
    // Whitened data X_w = X_centered * K^T -> (n_samples x n_components)
    let mut x_w = vec![0.0; n_samples * n_components];
    for i in 0..n_samples {
        for c in 0..n_components {
            let mut acc = 0.0;
            for j in 0..dim {
                acc += centered[i * dim + j] * whitening[c * dim + j];
            }
            x_w[i * n_components + c] = acc;
        }
    }
    // Initial W: random orthogonal n_components x n_components
    let mut w_mat = vec![0.0; n_components * n_components];
    for r in 0..n_components {
        for c in 0..n_components {
            w_mat[r * n_components + c] = rng.next_normal();
        }
    }
    w_mat = polar_orthogonal(&w_mat, n_components)?;

    // Fixed-point iteration
    for _ in 0..max_iter {
        // S = X_w * W^T -> (n_samples x n_components)
        let mut s = vec![0.0; n_samples * n_components];
        for i in 0..n_samples {
            for c in 0..n_components {
                let mut acc = 0.0;
                for k in 0..n_components {
                    acc += x_w[i * n_components + k] * w_mat[c * n_components + k];
                }
                s[i * n_components + c] = acc;
            }
        }
        // g(s) and g'(s)
        let mut g = vec![0.0; n_samples * n_components];
        let mut gp = vec![0.0; n_samples * n_components];
        for i in 0..n_samples * n_components {
            let u = s[i];
            let (gv, gpv) = match nonlin {
                IcaNonlinearity::Tanh => {
                    let t = u.tanh();
                    (t, 1.0 - t * t)
                }
                IcaNonlinearity::Gauss => {
                    let e = (-0.5 * u * u).exp();
                    (u * e, (1.0 - u * u) * e)
                }
            };
            g[i] = gv;
            gp[i] = gpv;
        }
        // W_new[c, k] = (1/n) sum_i g(s_ic) * x_w[i,k] - (1/n) sum_i g'(s_ic) * w_ck
        let mut w_new = vec![0.0; n_components * n_components];
        for c in 0..n_components {
            let mut mean_gp = 0.0;
            for i in 0..n_samples {
                mean_gp += gp[i * n_components + c];
            }
            mean_gp /= n_samples as f64;
            for k in 0..n_components {
                let mut acc = 0.0;
                for i in 0..n_samples {
                    acc += g[i * n_components + c] * x_w[i * n_components + k];
                }
                acc /= n_samples as f64;
                w_new[c * n_components + k] = acc - mean_gp * w_mat[c * n_components + k];
            }
        }
        // Symmetric orthogonalisation
        let w_orth = polar_orthogonal(&w_new, n_components)?;
        // Check convergence: max |<w_old_c, w_new_c>| close to 1
        let mut max_diff = 0.0_f64;
        for c in 0..n_components {
            let mut dot = 0.0;
            for k in 0..n_components {
                dot += w_orth[c * n_components + k] * w_mat[c * n_components + k];
            }
            max_diff = max_diff.max((dot.abs() - 1.0).abs());
        }
        w_mat = w_orth;
        if max_diff < tol {
            break;
        }
    }
    // Sources = X_w * W^T
    let mut sources = vec![0.0; n_samples * n_components];
    for i in 0..n_samples {
        for c in 0..n_components {
            let mut acc = 0.0;
            for k in 0..n_components {
                acc += x_w[i * n_components + k] * w_mat[c * n_components + k];
            }
            sources[i * n_components + c] = acc;
        }
    }
    Ok(IcaResult {
        mean,
        whitening,
        w_matrix: w_mat,
        sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ica_runs_tanh() {
        let mut rng = LcgRng::new(7);
        let n = 20;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for xi in x.iter_mut() {
            *xi = rng.next_normal();
        }
        let r = fast_ica(&x, n, dim, 2, 200, 1e-5, IcaNonlinearity::Tanh, &mut rng).expect("ok");
        assert_eq!(r.sources.len(), n * 2);
    }

    #[test]
    fn ica_sources_unit_variance() {
        let mut rng = LcgRng::new(9);
        let n = 50;
        let dim = 3;
        // Mix two independent uniforms
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            let s1 = rng.next_range(-1.0, 1.0);
            let s2 = rng.next_range(0.0, std::f64::consts::TAU).sin();
            x[i * dim] = s1 + 0.5 * s2;
            x[i * dim + 1] = 0.5 * s1 + s2;
            x[i * dim + 2] = 0.3 * s1 - 0.7 * s2;
        }
        let r = fast_ica(&x, n, dim, 2, 200, 1e-5, IcaNonlinearity::Tanh, &mut rng).expect("ok");
        // Computed sources should have approximate unit variance
        let mut var = [0.0_f64; 2];
        let mut mean = [0.0_f64; 2];
        for i in 0..n {
            for (c, m) in mean.iter_mut().enumerate() {
                *m += r.sources[i * 2 + c];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        for i in 0..n {
            for (c, v) in var.iter_mut().enumerate() {
                let d = r.sources[i * 2 + c] - mean[c];
                *v += d * d;
            }
        }
        for v in var.iter_mut() {
            *v /= (n - 1) as f64;
            assert!(*v > 0.1 && *v < 10.0);
        }
    }
}
