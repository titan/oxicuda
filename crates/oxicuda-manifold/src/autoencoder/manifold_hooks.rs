//! Manifold hooks for auto-encoder-style embedding and gradient export.
//!
//! Provides the interface between manifold learning algorithms and neural network training:
//! - An **embedding** Z ∈ ℝ^{n×d} (low-dimensional representation)
//! - The **gradient of the reconstruction loss** with respect to the embedding coordinates
//! - A **reconstruction error** metric
//!
//! # Linear Decoder
//!
//! Given embedding Z, reconstruct via `X̂ = Z W + b` (W ∈ ℝ^{d×D}).
//!
//! Reconstruction loss: `L = (1/n) ||X - X̂||_F²`
//!
//! Gradients:
//! - `∂L/∂Z = (2/n) (X̂ - X) W^T`
//! - `∂L/∂W = (2/n) Z^T (X̂ - X)`
//! - `∂L/∂b = (2/n) Σ_i (X̂_i - X_i)`

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::linear::pca::pca_fit;
use crate::tsne::perplexity::compute_perplexity_p_matrix;

// ──────────────────────────────────────────────────────────────────────────────
// Trait definition
// ──────────────────────────────────────────────────────────────────────────────

/// Core interface for manifold-learning hooks that expose embedding + gradient information
/// to neural network training loops (e.g. oxicuda-dnn encoders).
pub trait ManifoldHook {
    /// Forward pass: compute embedding Z ∈ ℝ^{n×d_out} from X ∈ ℝ^{n×dim}.
    ///
    /// Returns a flattened row-major vector of shape `[n * d_out]`.
    fn embed(&self, x: &[f64], n: usize, dim: usize) -> ManifoldResult<Vec<f64>>;

    /// Backward pass: compute `∂L/∂X ∈ ℝ^{n×dim}` given `∂L/∂Z ∈ ℝ^{n×d_out}`.
    ///
    /// Uses the chain rule through the (linear) embedding map.
    fn embedding_gradient(
        &self,
        grad_z: &[f64],
        n: usize,
        d_in: usize,
        d_out: usize,
    ) -> ManifoldResult<Vec<f64>>;

    /// Reconstruction loss: decode Z → X̂ and compute `||X - X̂||_F² / n`.
    fn reconstruction_loss(
        &self,
        x: &[f64],
        z: &[f64],
        n: usize,
        d_in: usize,
        d_out: usize,
    ) -> ManifoldResult<f64>;
}

// ──────────────────────────────────────────────────────────────────────────────
// PCA-based manifold hook
// ──────────────────────────────────────────────────────────────────────────────

/// PCA-based manifold hook.
///
/// Stores the fitted principal components and mean vector so that:
/// - `encode(X) = (X - mean) @ components^T`
/// - `decode(Z) = Z @ components + mean`
///
/// This makes the hook a perfect linear autoencoder with an orthonormal decoder.
#[derive(Debug, Clone)]
pub struct PcaManifoldHook {
    pub n_components: usize,
    /// Row-major [n_components × d_in] principal component matrix.
    pub components: Option<Vec<f64>>,
    /// Mean vector of length d_in (for centering).
    pub mean: Option<Vec<f64>>,
    /// Stored d_in after fitting.
    d_in: usize,
}

impl PcaManifoldHook {
    /// Construct an unfitted hook with the given number of components.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            mean: None,
            d_in: 0,
        }
    }

    /// Fit PCA on `x` of shape `[n × d_in]`.  Stores `components` and `mean`.
    pub fn fit(&mut self, x: &[f64], n: usize, d_in: usize) -> ManifoldResult<()> {
        if n == 0 || d_in == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if x.len() != n * d_in {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, d_in],
                got: vec![x.len()],
            });
        }
        if self.n_components == 0 || self.n_components > d_in {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: format!("must be in 1..={d_in}, got {}", self.n_components),
            });
        }
        let result = pca_fit(x, n, d_in, self.n_components)?;
        self.components = Some(result.components);
        self.mean = Some(result.mean);
        self.d_in = d_in;
        Ok(())
    }

    /// Encode `x` of shape `[n × d_in]` → `Z` of shape `[n × n_components]`.
    pub fn encode(&self, x: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
        let comps = self.components.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let mean = self.mean.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let d_in = self.d_in;
        let k = self.n_components;
        if x.len() != n * d_in {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, d_in],
                got: vec![x.len()],
            });
        }
        // Z[i, c] = sum_j (x[i,j] - mean[j]) * comps[c, j]
        let mut z = vec![0.0_f64; n * k];
        for i in 0..n {
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..d_in {
                    acc += (x[i * d_in + j] - mean[j]) * comps[c * d_in + j];
                }
                z[i * k + c] = acc;
            }
        }
        Ok(z)
    }

    /// Decode `z` of shape `[n × n_components]` → `X̂` of shape `[n × d_in]`.
    ///
    /// `X̂ = Z @ components + mean`
    pub fn decode(&self, z: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
        let comps = self.components.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let mean = self.mean.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let d_in = self.d_in;
        let k = self.n_components;
        if z.len() != n * k {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, k],
                got: vec![z.len()],
            });
        }
        // x_hat[i, j] = sum_c z[i,c] * comps[c,j] + mean[j]
        let mut x_hat = vec![0.0_f64; n * d_in];
        for i in 0..n {
            for j in 0..d_in {
                let mut acc = mean[j];
                for c in 0..k {
                    acc += z[i * k + c] * comps[c * d_in + j];
                }
                x_hat[i * d_in + j] = acc;
            }
        }
        Ok(x_hat)
    }

    /// Compute `∂L/∂X` of shape `[n × d_in]` from `∂L/∂Z` of shape `[n × n_components]`.
    ///
    /// Via chain rule through the PCA encoding:
    /// `∂L/∂X[i,j] = sum_c ∂L/∂Z[i,c] * comps[c,j]`
    pub fn grad_encode(&self, grad_z: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
        let comps = self.components.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let d_in = self.d_in;
        let k = self.n_components;
        if grad_z.len() != n * k {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, k],
                got: vec![grad_z.len()],
            });
        }
        let mut grad_x = vec![0.0_f64; n * d_in];
        for i in 0..n {
            for j in 0..d_in {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += grad_z[i * k + c] * comps[c * d_in + j];
                }
                grad_x[i * d_in + j] = acc;
            }
        }
        Ok(grad_x)
    }

    /// Compute reconstruction gradients:
    /// - `∂L/∂Z` of shape `[n × n_components]`
    /// - `∂L/∂W` of shape `[n_components × d_in]`  (W = components here)
    ///
    /// Given: `L = (1/n) ||X - (Z W + mean)||_F²`
    /// Residual `R = X̂ - X = Z W + mean - X`
    /// `∂L/∂Z = (2/n) R W^T`
    /// `∂L/∂W = (2/n) Z^T R`
    pub fn grad_decode(
        &self,
        x: &[f64],
        z: &[f64],
        n: usize,
        d_in: usize,
    ) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
        let comps = self.components.as_ref().ok_or_else(|| {
            ManifoldError::InvalidConfiguration("PcaManifoldHook not fitted".into())
        })?;
        let k = self.n_components;
        if x.len() != n * d_in || z.len() != n * k {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n * d_in + n * k],
                got: vec![x.len() + z.len()],
            });
        }
        // Decode to get X̂
        let x_hat = self.decode(z, n)?;
        // Residual R = X̂ - X,  shape [n × d_in]
        let mut residual = vec![0.0_f64; n * d_in];
        for idx in 0..n * d_in {
            residual[idx] = x_hat[idx] - x[idx];
        }
        let scale = 2.0 / n as f64;
        // ∂L/∂Z[i,c] = scale * sum_j R[i,j] * comps[c,j]
        let mut grad_z = vec![0.0_f64; n * k];
        for i in 0..n {
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..d_in {
                    acc += residual[i * d_in + j] * comps[c * d_in + j];
                }
                grad_z[i * k + c] = scale * acc;
            }
        }
        // ∂L/∂W[c,j] = scale * sum_i Z[i,c] * R[i,j]
        let mut grad_w = vec![0.0_f64; k * d_in];
        for c in 0..k {
            for j in 0..d_in {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += z[i * k + c] * residual[i * d_in + j];
                }
                grad_w[c * d_in + j] = scale * acc;
            }
        }
        Ok((grad_z, grad_w))
    }
}

impl ManifoldHook for PcaManifoldHook {
    fn embed(&self, x: &[f64], n: usize, _dim: usize) -> ManifoldResult<Vec<f64>> {
        self.encode(x, n)
    }

    fn embedding_gradient(
        &self,
        grad_z: &[f64],
        n: usize,
        _d_in: usize,
        _d_out: usize,
    ) -> ManifoldResult<Vec<f64>> {
        self.grad_encode(grad_z, n)
    }

    fn reconstruction_loss(
        &self,
        x: &[f64],
        z: &[f64],
        n: usize,
        d_in: usize,
        _d_out: usize,
    ) -> ManifoldResult<f64> {
        let x_hat = self.decode(z, n)?;
        if x.len() != n * d_in {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, d_in],
                got: vec![x.len()],
            });
        }
        let mut loss = 0.0_f64;
        for idx in 0..n * d_in {
            let r = x_hat[idx] - x[idx];
            loss += r * r;
        }
        Ok(loss / n as f64)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// t-SNE regularised gradient hook
// ──────────────────────────────────────────────────────────────────────────────

/// t-SNE regularized gradient hook.
///
/// Combines a PCA-based linear reconstruction gradient with a t-SNE manifold gradient.
///
/// `grad_total = alpha * grad_recon + (1 - alpha) * grad_tsne`
///
/// where `grad_recon` is `∂L_recon/∂Z` and `grad_tsne` is `∂KL(P||Q)/∂Z`.
pub struct TsneRegHook {
    pub n_components: usize,
    pub perplexity: f64,
    /// Weight on the t-SNE gradient (reconstruction weight = 1 - alpha).
    pub alpha: f64,
    pub pca_hook: PcaManifoldHook,
}

impl TsneRegHook {
    /// Create a new t-SNE regularized hook.
    ///
    /// - `n_components`: embedding dimensionality
    /// - `perplexity`: t-SNE perplexity parameter
    /// - `alpha`: weight on t-SNE gradient (in `[0,1]`)
    #[must_use]
    pub fn new(n_components: usize, perplexity: f64, alpha: f64) -> Self {
        Self {
            n_components,
            perplexity,
            alpha: alpha.clamp(0.0, 1.0),
            pca_hook: PcaManifoldHook::new(n_components),
        }
    }

    /// Fit PCA and run t-SNE-style embedding.  Returns embedding Z of shape `[n × n_components]`.
    ///
    /// Steps:
    /// 1. Fit PCA (for the linear decoder part)
    /// 2. Use PCA projection as the initial embedding
    /// 3. Run t-SNE-style gradient descent on Z using the perplexity-based P matrix
    pub fn fit(
        &mut self,
        x: &[f64],
        n: usize,
        d_in: usize,
        rng: &mut LcgRng,
    ) -> ManifoldResult<Vec<f64>> {
        if n == 0 || d_in == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if x.len() != n * d_in {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, d_in],
                got: vec![x.len()],
            });
        }
        let k = self.n_components;
        if k == 0 || k > d_in {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: format!("must be in 1..={d_in}"),
            });
        }
        // 1. Fit PCA
        self.pca_hook.fit(x, n, d_in)?;
        // 2. Initial embedding from PCA projection (small scale)
        let pca_z = self.pca_hook.encode(x, n)?;
        // Scale to small values like t-SNE initialisation
        let mut z = vec![0.0_f64; n * k];
        for (i, v) in pca_z.iter().enumerate() {
            z[i] = v * 0.0001 + rng.next_normal() * 0.0001;
        }
        // 3. Build pairwise squared distances for perplexity P matrix
        let mut d2 = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0;
                for kk in 0..d_in {
                    let v = x[i * d_in + kk] - x[j * d_in + kk];
                    s += v * v;
                }
                d2[i * n + j] = s;
                d2[j * n + i] = s;
            }
        }
        let perp = self.perplexity.min((n - 1) as f64 / 3.0).max(1.0);
        let p = compute_perplexity_p_matrix(&d2, n, perp, 60, 1e-5)?;
        // 4. t-SNE gradient descent (simplified, 100 iterations)
        let mut dy_prev = vec![0.0_f64; n * k];
        let mut gains = vec![1.0_f64; n * k];
        let lr = 200.0_f64;
        let momentum = 0.5_f64;
        let min_gain = 0.01_f64;
        for _iter in 0..100 {
            // Compute Q matrix
            let mut q = vec![0.0_f64; n * n];
            let mut z_sum = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut d2ij = 0.0;
                    for kk in 0..k {
                        let v = z[i * k + kk] - z[j * k + kk];
                        d2ij += v * v;
                    }
                    let qval = 1.0 / (1.0 + d2ij);
                    q[i * n + j] = qval;
                    z_sum += qval;
                }
            }
            let z_sum = z_sum.max(1e-300);
            for v in &mut q {
                *v /= z_sum;
            }
            for v in q.iter_mut() {
                if *v < 1e-12 {
                    *v = 1e-12;
                }
            }
            // Gradient
            let mut grad = vec![0.0_f64; n * k];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut d2ij = 0.0;
                    for kk in 0..k {
                        let v = z[i * k + kk] - z[j * k + kk];
                        d2ij += v * v;
                    }
                    let qkernel = 1.0 / (1.0 + d2ij);
                    let pij = p[i * n + j];
                    let qij = q[i * n + j];
                    let mult = 4.0 * (pij - qij) * qkernel;
                    for kk in 0..k {
                        grad[i * k + kk] += mult * (z[i * k + kk] - z[j * k + kk]);
                    }
                }
            }
            // Update with momentum
            for i in 0..n * k {
                let same_sign = grad[i].signum() == dy_prev[i].signum();
                if same_sign {
                    gains[i] *= 0.8;
                } else {
                    gains[i] += 0.2;
                }
                gains[i] = gains[i].max(min_gain);
                let new_dy = momentum * dy_prev[i] - lr * gains[i] * grad[i];
                dy_prev[i] = new_dy;
                z[i] += new_dy;
            }
            // Re-centre
            for kk in 0..k {
                let mut m = 0.0;
                for i in 0..n {
                    m += z[i * k + kk];
                }
                m /= n as f64;
                for i in 0..n {
                    z[i * k + kk] -= m;
                }
            }
        }
        Ok(z)
    }

    /// Compute combined gradient = alpha * grad_tsne + (1 - alpha) * grad_recon.
    ///
    /// - `x`: original data `[n × d_in]`
    /// - `z`: current embedding `[n × n_components]`
    /// - `p`: t-SNE joint probability matrix `[n × n]`
    /// - Returns: combined gradient `[n × n_components]`
    pub fn combined_gradient(
        &self,
        x: &[f64],
        z: &[f64],
        p: &[f64],
        n: usize,
        d_in: usize,
    ) -> ManifoldResult<Vec<f64>> {
        let k = self.n_components;
        if x.len() != n * d_in {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, d_in],
                got: vec![x.len()],
            });
        }
        if z.len() != n * k {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, k],
                got: vec![z.len()],
            });
        }
        if p.len() != n * n {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![p.len()],
            });
        }
        // Reconstruction gradient ∂L_recon/∂Z
        let (grad_recon, _grad_w) = self.pca_hook.grad_decode(x, z, n, d_in)?;
        // t-SNE gradient ∂KL(P||Q)/∂Z
        let mut q = vec![0.0_f64; n * n];
        let mut z_sum = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d2ij = 0.0;
                for kk in 0..k {
                    let v = z[i * k + kk] - z[j * k + kk];
                    d2ij += v * v;
                }
                let qval = 1.0 / (1.0 + d2ij);
                q[i * n + j] = qval;
                z_sum += qval;
            }
        }
        let z_sum = z_sum.max(1e-300);
        for v in &mut q {
            *v /= z_sum;
        }
        for v in q.iter_mut() {
            if *v < 1e-12 {
                *v = 1e-12;
            }
        }
        let mut grad_tsne = vec![0.0_f64; n * k];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d2ij = 0.0;
                for kk in 0..k {
                    let v = z[i * k + kk] - z[j * k + kk];
                    d2ij += v * v;
                }
                let qkernel = 1.0 / (1.0 + d2ij);
                let pij = p[i * n + j];
                let qij = q[i * n + j];
                let mult = 4.0 * (pij - qij) * qkernel;
                for kk in 0..k {
                    grad_tsne[i * k + kk] += mult * (z[i * k + kk] - z[j * k + kk]);
                }
            }
        }
        // Combined: alpha * grad_tsne + (1 - alpha) * grad_recon
        let alpha = self.alpha;
        let mut combined = vec![0.0_f64; n * k];
        for i in 0..n * k {
            combined[i] = alpha * grad_tsne[i] + (1.0 - alpha) * grad_recon[i];
        }
        Ok(combined)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Embedding export
// ──────────────────────────────────────────────────────────────────────────────

/// Complete embedding export for use by downstream crates (e.g. oxicuda-dnn).
///
/// Contains all information needed by a neural-network encoder to back-propagate
/// through the manifold embedding.
pub struct EmbeddingExport {
    /// Embedding coordinates Z of shape `[n_samples × d_out]`.
    pub z: Vec<f64>,
    /// Linear reconstruction X̂ of shape `[n_samples × d_in]`.
    pub reconstruction: Vec<f64>,
    /// Mean squared reconstruction loss `(1/n) ||X - X̂||_F²`.
    pub recon_loss: f64,
    /// Reconstruction gradient `∂L/∂Z` of shape `[n_samples × d_out]`.
    pub z_gradient: Vec<f64>,
    pub n_samples: usize,
    pub d_in: usize,
    pub d_out: usize,
}

/// Encode data using a fitted `PcaManifoldHook` and export all embedding information.
///
/// This is the primary function for integrating manifold hooks into a neural-network
/// training loop.  The returned [`EmbeddingExport`] contains the embedding, reconstruction,
/// loss, and gradient — everything a DNN encoder needs for one forward + backward step.
pub fn manifold_encode_and_export(
    hook: &PcaManifoldHook,
    x: &[f64],
    n: usize,
    d_in: usize,
) -> ManifoldResult<EmbeddingExport> {
    if n == 0 || d_in == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n * d_in {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d_in],
            got: vec![x.len()],
        });
    }
    let d_out = hook.n_components;
    // Forward: encode
    let z = hook.encode(x, n)?;
    // Decode to get reconstruction
    let reconstruction = hook.decode(&z, n)?;
    // Reconstruction loss
    let mut recon_loss = 0.0_f64;
    for idx in 0..n * d_in {
        let r = reconstruction[idx] - x[idx];
        recon_loss += r * r;
    }
    recon_loss /= n as f64;
    // Gradient ∂L/∂Z
    let (z_gradient, _grad_w) = hook.grad_decode(x, &z, n, d_in)?;
    Ok(EmbeddingExport {
        z,
        reconstruction,
        recon_loss,
        z_gradient,
        n_samples: n,
        d_in,
        d_out,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize, rng: &mut LcgRng) -> Vec<f64> {
        (0..n * d).map(|_| rng.next_normal()).collect()
    }

    fn make_low_rank_data(n: usize, d: usize, rank: usize, rng: &mut LcgRng) -> Vec<f64> {
        // X = U @ V where U is [n × rank], V is [rank × d]
        let u: Vec<f64> = (0..n * rank).map(|_| rng.next_normal()).collect();
        let v: Vec<f64> = (0..rank * d).map(|_| rng.next_normal()).collect();
        let mut x = vec![0.0_f64; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut acc = 0.0;
                for r in 0..rank {
                    acc += u[i * rank + r] * v[r * d + j];
                }
                x[i * d + j] = acc;
            }
        }
        x
    }

    // 1. pca_hook_fit_runs
    #[test]
    fn pca_hook_fit_runs() {
        let mut rng = LcgRng::new(1);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8)
            .expect("fit should succeed on 20×8 data");
        assert!(hook.components.is_some());
        assert!(hook.mean.is_some());
    }

    // 2. pca_hook_embed_shape
    #[test]
    fn pca_hook_embed_shape() {
        let mut rng = LcgRng::new(2);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8).expect("fit");
        let z = hook.embed(&x, 20, 8).expect("embed");
        assert_eq!(z.len(), 20 * 3, "embedding should be [n × k]");
    }

    // 3. pca_hook_decode_shape
    #[test]
    fn pca_hook_decode_shape() {
        let mut rng = LcgRng::new(3);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8).expect("fit");
        let z = hook.encode(&x, 20).expect("encode");
        let x_hat = hook.decode(&z, 20).expect("decode");
        assert_eq!(x_hat.len(), 20 * 8, "reconstruction should be [n × d_in]");
    }

    // 4. pca_hook_reconstruction_loss_finite
    #[test]
    fn pca_hook_reconstruction_loss_finite() {
        let mut rng = LcgRng::new(4);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8).expect("fit");
        let z = hook.encode(&x, 20).expect("encode");
        let loss = hook.reconstruction_loss(&x, &z, 20, 8, 3).expect("loss");
        assert!(loss.is_finite(), "reconstruction loss must be finite");
        assert!(loss >= 0.0, "reconstruction loss must be non-negative");
    }

    // 5. pca_hook_recon_loss_zero_on_pca_data
    #[test]
    fn pca_hook_recon_loss_zero_on_pca_data() {
        let mut rng = LcgRng::new(5);
        // Build data that lives in a rank-2 subspace of ℝ^4
        // When we use 2 components, the PCA decoder should perfectly reconstruct
        let x = make_low_rank_data(20, 4, 2, &mut rng);
        let mut hook = PcaManifoldHook::new(2);
        hook.fit(&x, 20, 4).expect("fit");
        let z = hook.encode(&x, 20).expect("encode");
        let loss = hook.reconstruction_loss(&x, &z, 20, 4, 2).expect("loss");
        // Low-rank data: PCA with rank components → reconstruction error ≈ 0
        assert!(
            loss < 1e-8,
            "reconstruction loss should be ~0 for rank-2 data with 2 components, got {loss}"
        );
    }

    // 6. pca_hook_grad_encode_shape
    #[test]
    fn pca_hook_grad_encode_shape() {
        let mut rng = LcgRng::new(6);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8).expect("fit");
        // Fake upstream gradient ∂L/∂X
        let grad_z: Vec<f64> = (0..20 * 3).map(|i| (i as f64) * 0.01).collect();
        let grad_x = hook
            .embedding_gradient(&grad_z, 20, 8, 3)
            .expect("embedding_gradient");
        assert_eq!(grad_x.len(), 20 * 8, "∂L/∂X should be [n × d_in]");
    }

    // 7. pca_hook_grad_decode_shape
    #[test]
    fn pca_hook_grad_decode_shape() {
        let mut rng = LcgRng::new(7);
        let x = make_data(20, 8, &mut rng);
        let mut hook = PcaManifoldHook::new(3);
        hook.fit(&x, 20, 8).expect("fit");
        let z = hook.encode(&x, 20).expect("encode");
        let (grad_z, _grad_w) = hook.grad_decode(&x, &z, 20, 8).expect("grad_decode");
        assert_eq!(grad_z.len(), 20 * 3, "∂L/∂Z should be [n × d_out]");
    }

    // 8. tsne_hook_fit_runs
    #[test]
    fn tsne_hook_fit_runs() {
        let mut rng = LcgRng::new(8);
        let x = make_data(20, 4, &mut rng);
        let mut hook = TsneRegHook::new(2, 3.0, 0.5);
        let z = hook.fit(&x, 20, 4, &mut rng).expect("TsneRegHook::fit");
        assert_eq!(z.len(), 20 * 2, "embedding should be [n × n_components]");
        assert!(
            z.iter().all(|v| v.is_finite()),
            "all embedding values must be finite"
        );
    }

    // 9. tsne_hook_combined_grad_shape
    #[test]
    fn tsne_hook_combined_grad_shape() {
        let mut rng = LcgRng::new(9);
        let x = make_data(20, 4, &mut rng);
        let mut hook = TsneRegHook::new(2, 3.0, 0.5);
        let z = hook.fit(&x, 20, 4, &mut rng).expect("fit");
        // Build a dummy P matrix
        let p: Vec<f64> = (0..20 * 20)
            .map(|idx| {
                let i = idx / 20;
                let j = idx % 20;
                if i == j { 0.0 } else { 1.0 / (20.0 * 19.0) }
            })
            .collect();
        let grad = hook
            .combined_gradient(&x, &z, &p, 20, 4)
            .expect("combined_gradient");
        assert_eq!(grad.len(), 20 * 2, "gradient should be [n × n_components]");
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "all gradient values must be finite"
        );
    }

    // 10. embedding_export_consistent_shapes
    #[test]
    fn embedding_export_consistent_shapes() {
        let mut rng = LcgRng::new(10);
        let n = 20;
        let d_in = 8;
        let k = 3;
        let x = make_data(n, d_in, &mut rng);
        let mut hook = PcaManifoldHook::new(k);
        hook.fit(&x, n, d_in).expect("fit");
        let export =
            manifold_encode_and_export(&hook, &x, n, d_in).expect("manifold_encode_and_export");
        assert_eq!(export.z.len(), n * k, "z shape");
        assert_eq!(
            export.reconstruction.len(),
            n * d_in,
            "reconstruction shape"
        );
        assert_eq!(export.z_gradient.len(), n * k, "z_gradient shape");
        assert_eq!(export.n_samples, n);
        assert_eq!(export.d_in, d_in);
        assert_eq!(export.d_out, k);
        assert!(
            export.recon_loss.is_finite() && export.recon_loss >= 0.0,
            "recon_loss must be finite and non-negative"
        );
    }

    // 11. pca_hook_orthogonal_components
    #[test]
    fn pca_hook_orthogonal_components() {
        let mut rng = LcgRng::new(11);
        let x = make_data(30, 8, &mut rng);
        let k = 4;
        let mut hook = PcaManifoldHook::new(k);
        hook.fit(&x, 30, 8).expect("fit");
        let comps = hook.components.as_ref().expect("components");
        // Check that rows are orthonormal: C @ C^T ≈ I_k
        let d_in = 8;
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for jj in 0..d_in {
                    dot += comps[i * d_in + jj] * comps[j * d_in + jj];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "component rows not orthonormal: C[{i},{j}] dot = {dot}, expected {expected}"
                );
            }
        }
    }

    // 12. pca_hook_roundtrip
    #[test]
    fn pca_hook_roundtrip() {
        let mut rng = LcgRng::new(12);
        let n = 30;
        let d_in = 8;
        let k = 4;
        let x = make_data(n, d_in, &mut rng);
        let mut hook = PcaManifoldHook::new(k);
        hook.fit(&x, n, d_in).expect("fit");
        // PCA roundtrip loss
        let z_pca = hook.encode(&x, n).expect("encode");
        let pca_loss = hook
            .reconstruction_loss(&x, &z_pca, n, d_in, k)
            .expect("loss");
        // Random projection baseline
        // Use a random Z and decode through the same (PCA) decoder
        let z_random: Vec<f64> = (0..n * k).map(|_| rng.next_normal()).collect();
        let random_loss = hook
            .reconstruction_loss(&x, &z_random, n, d_in, k)
            .expect("random loss");
        assert!(
            pca_loss <= random_loss,
            "PCA roundtrip loss {pca_loss} should be <= random projection loss {random_loss}"
        );
    }
}
