//! Diffusion Maps embedding.
//!
//! 1. Build kernel `K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))`.
//! 2. Density-normalise to handle non-uniform sampling: `K_alpha = D^{-alpha} K D^{-alpha}`.
//! 3. Row-normalise to a Markov transition matrix `P = D'^{-1} K_alpha`.
//! 4. Symmetric conjugate: `P_sym = D'^{1/2} P D'^{-1/2}`, eigendecompose.
//! 5. Embedding `Psi_i = (lambda_1^t psi_1(i), lambda_2^t psi_2(i), ...)`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Diffusion map result.
pub struct DiffusionMapResult {
    pub embedding: Vec<f64>,
    pub eigenvalues: Vec<f64>,
}

/// Fit diffusion map.
pub fn diffusion_map_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_components: usize,
    sigma: f64,
    alpha: f64,
    t: usize,
) -> ManifoldResult<DiffusionMapResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components + 1 > n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n_samples}"),
        });
    }
    let n = n_samples;
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for d in 0..dim {
                let v = x[i * dim + d] - x[j * dim + d];
                s += v * v;
            }
            let val = (-s / two_sigma2).exp();
            k[i * n + j] = val;
            k[j * n + i] = val;
        }
    }
    // Density-normalisation
    let mut deg = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += k[i * n + j];
        }
        deg[i] = s.max(1e-300);
    }
    if alpha > 0.0 {
        for i in 0..n {
            let di_a = deg[i].powf(alpha);
            for j in 0..n {
                let dj_a = deg[j].powf(alpha);
                k[i * n + j] /= di_a * dj_a;
            }
        }
        // Recompute degree
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += k[i * n + j];
            }
            deg[i] = s.max(1e-300);
        }
    }
    // Symmetric conjugate of P: P_sym = D^{-1/2} K D^{-1/2}
    let mut p_sym = vec![0.0; n * n];
    for i in 0..n {
        let inv_i = 1.0 / deg[i].sqrt();
        for j in 0..n {
            let inv_j = 1.0 / deg[j].sqrt();
            p_sym[i * n + j] = k[i * n + j] * inv_i * inv_j;
        }
    }
    // Eigendecompose P_sym
    let (mut w, mut v) = jacobi_eigh(&p_sym, n)?;
    sort_eigen_descending(&mut w, &mut v, n);
    // Recover right eigenvectors of P: phi_k = D^{-1/2} v_k
    let mut embedding = vec![0.0; n * n_components];
    let mut eigenvalues = vec![0.0; n_components];
    for c in 0..n_components {
        let lam = w[c + 1];
        eigenvalues[c] = lam;
        let lam_t = lam.powi(t as i32);
        for r in 0..n {
            let phi = v[r * n + (c + 1)] / deg[r].sqrt();
            embedding[r * n_components + c] = lam_t * phi;
        }
    }
    Ok(DiffusionMapResult {
        embedding,
        eigenvalues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusion_map_runs() {
        let n = 10;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = (i as f64).sin();
        }
        let r = diffusion_map_fit(&x, n, dim, 2, 1.0, 0.5, 1).expect("ok");
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diffusion_eigenvalues_descending() {
        let n = 8;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = (i % 3) as f64;
            x[i * dim + 1] = (i % 5) as f64;
        }
        let r = diffusion_map_fit(&x, n, dim, 3, 1.0, 0.5, 1).expect("ok");
        for w in 0..2 {
            assert!(r.eigenvalues[w] >= r.eigenvalues[w + 1] - 1e-9);
        }
    }
}
