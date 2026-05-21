//! Incremental PCA — Ross, Lim, Lin & Yang (2008).
//!
//! Processes data in chunks of size `batch_size`, maintaining a rank-k sketch
//! `(s, V)` and a running mean `μ` without storing the full dataset.
//!
//! Each chunk update stacks a small matrix `M` of shape `(k + b + 1) × d`
//! and extracts its thin SVD via eigendecomposition of `M Mᵀ` (a `(k+b+1)`
//! square matrix), keeping only the top-k singular triplets.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`IncrementalPca`].
#[derive(Debug, Clone)]
pub struct IncrementalPcaConfig {
    /// Number of principal components to retain.
    pub n_components: usize,
    /// Number of rows per mini-batch (chunk size).
    pub batch_size: usize,
}

impl Default for IncrementalPcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            batch_size: 10,
        }
    }
}

// ─── Struct ───────────────────────────────────────────────────────────────────

/// Incremental PCA model (Ross-Lim-Lin-Yang 2008).
///
/// State summary:
/// - `mean`: running column mean, length `n_features`.
/// - `components`: row-major `(n_components × n_features)` right singular
///   vectors (the "sketch" `V`).
/// - `singular_values`: length `n_components`, kept in descending order.
/// - `n_samples_seen`: total number of rows processed so far.
/// - `fitted`: becomes `true` after the first [`IncrementalPca::partial_fit`] call.
#[derive(Debug, Clone)]
pub struct IncrementalPca {
    /// Column means, length `n_features`.
    pub mean: Vec<f64>,
    /// Top-k right singular vectors, row-major `(n_components × n_features)`.
    pub components: Vec<f64>,
    /// Top-k singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Dimensionality of the input space (set on first `partial_fit`).
    pub n_features: usize,
    /// Total number of rows seen across all `partial_fit` calls.
    pub n_samples_seen: usize,
    /// Number of components to retain.
    pub n_components: usize,
    /// Chunk size used by [`IncrementalPca::fit`].
    pub batch_size: usize,
    /// Whether at least one `partial_fit` has been applied.
    pub fitted: bool,
}

impl IncrementalPca {
    /// Construct a fresh (unfitted) model from `config`.
    #[must_use]
    pub fn new(config: &IncrementalPcaConfig) -> Self {
        Self {
            mean: Vec::new(),
            components: Vec::new(),
            singular_values: Vec::new(),
            n_features: 0,
            n_samples_seen: 0,
            n_components: config.n_components,
            batch_size: config.batch_size.max(1),
            fitted: false,
        }
    }

    // ─── Core update ──────────────────────────────────────────────────────────

    /// Update the model with one chunk of `n_rows × n_features` data (row-major).
    ///
    /// Implements a single step of the Ross et al. (2008) incremental SVD:
    ///
    /// 1. Validate inputs.
    /// 2. Update the running mean and compute the mean-shift correction.
    /// 3. Build the stacking matrix
    ///    `M = [ √n · diag(s) · V ;  X_centered ;  correction ]`.
    /// 4. Compute a thin SVD of M by eigendecomposing `M Mᵀ` (small square).
    /// 5. Retain the top-k singular triplets.
    pub fn partial_fit(
        &mut self,
        chunk: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> ManifoldResult<()> {
        // ── validation ────────────────────────────────────────────────────────
        if n_rows == 0 || n_features == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if chunk.len() != n_rows * n_features {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n_rows, n_features],
                got: vec![chunk.len()],
            });
        }
        if self.n_components == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be >= 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "must be <= n_features ({n_features}), got {}",
                    self.n_components
                ),
            });
        }

        // ── on first call, initialise dimensions ──────────────────────────────
        if !self.fitted {
            self.n_features = n_features;
            self.mean = vec![0.0; n_features];
            self.components = vec![0.0; self.n_components * n_features];
            self.singular_values = vec![0.0; self.n_components];
        } else if n_features != self.n_features {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![self.n_features],
                got: vec![n_features],
            });
        }

        let n_old = self.n_samples_seen;
        let b = n_rows;
        let n_new = n_old + b;
        let d = n_features;
        let k = self.n_components;

        // ── chunk mean ────────────────────────────────────────────────────────
        let mut chunk_mean = vec![0.0; d];
        for i in 0..b {
            for j in 0..d {
                chunk_mean[j] += chunk[i * d + j];
            }
        }
        let b_f = b as f64;
        for m in &mut chunk_mean {
            *m /= b_f;
        }

        // ── updated running mean μ_{n+b} ──────────────────────────────────────
        let n_old_f = n_old as f64;
        let n_new_f = n_new as f64;
        let mut new_mean = vec![0.0; d];
        for j in 0..d {
            new_mean[j] = (n_old_f * self.mean[j] + b_f * chunk_mean[j]) / n_new_f;
        }

        // ── centre the chunk with the updated mean ───────────────────────────
        let mut x_centered = vec![0.0; b * d];
        for i in 0..b {
            for j in 0..d {
                x_centered[i * d + j] = chunk[i * d + j] - new_mean[j];
            }
        }

        // ── mean-shift correction row ─────────────────────────────────────────
        // row = sqrt(n_old * b / n_new) * (old_mean - new_mean)
        let correction_scale = if n_old > 0 {
            (n_old_f * b_f / n_new_f).sqrt()
        } else {
            0.0
        };
        let mut correction_row = vec![0.0; d];
        for j in 0..d {
            correction_row[j] = correction_scale * (self.mean[j] - new_mean[j]);
        }

        // ── number of rows in stacking matrix M ───────────────────────────────
        // M rows: k (existing components, scaled) + b (chunk) + 1 (correction)
        let m_rows = k + b + 1;

        // ── build M (m_rows × d, row-major) ──────────────────────────────────
        //
        // Block 0: rows 0..k  — sqrt(n_old) * diag(s) * V
        //   Row c = sqrt(n_old) * s[c] * components[c, :]
        //   (If n_old == 0 these rows are all zero — correct initial state.)
        //
        // Block 1: rows k..k+b — X_centered
        //
        // Block 2: row  k+b    — correction_row
        let mut m_mat = vec![0.0; m_rows * d];

        let sqrt_n_old = n_old_f.sqrt();
        for c in 0..k {
            let sv = self.singular_values[c];
            let scale = sqrt_n_old * sv;
            for j in 0..d {
                m_mat[c * d + j] = scale * self.components[c * d + j];
            }
        }
        for i in 0..b {
            for j in 0..d {
                m_mat[(k + i) * d + j] = x_centered[i * d + j];
            }
        }
        for j in 0..d {
            m_mat[(k + b) * d + j] = correction_row[j];
        }

        // ── thin SVD of M via eigendecomposition of G = M Mᵀ  ────────────────
        // G is (m_rows × m_rows) symmetric positive semi-definite.
        // eigenvalues λ_i of G  →  singular values  s_i = sqrt(max(λ_i, 0))
        // eigenvectors u_i of G →  left  singular vectors of M
        // right singular vectors: v_i = Mᵀ u_i / s_i  (normalised)
        let gram = mat_gram(&m_mat, m_rows, d);
        let (mut eig_vals, mut eig_vecs) = jacobi_eigh(&gram, m_rows)?;
        sort_eigen_descending(&mut eig_vals, &mut eig_vecs, m_rows);

        // ── retain top-k ─────────────────────────────────────────────────────
        let mut new_singular = vec![0.0; k];
        let mut new_components = vec![0.0; k * d];

        for c in 0..k {
            let lambda = eig_vals[c].max(0.0);
            let sv = lambda.sqrt();
            new_singular[c] = sv;

            if sv > 1e-14 {
                // v_c = Mᵀ u_c / sv  (length d)
                // eig_vecs is (m_rows × m_rows) row-major; column c: eig_vecs[r*m_rows+c]
                for j in 0..d {
                    let mut acc = 0.0;
                    for r in 0..m_rows {
                        acc += m_mat[r * d + j] * eig_vecs[r * m_rows + c];
                    }
                    new_components[c * d + j] = acc / sv;
                }
            }
            // else: zero singular value — component stays zero (already initialised).
        }

        // ── commit ────────────────────────────────────────────────────────────
        self.mean = new_mean;
        self.singular_values = new_singular;
        self.components = new_components;
        self.n_samples_seen = n_new;
        self.fitted = true;

        Ok(())
    }

    // ─── Batch fit ────────────────────────────────────────────────────────────

    /// Fit by iterating `partial_fit` over non-overlapping chunks of `batch_size` rows.
    ///
    /// The last chunk may be smaller than `batch_size` if `n_samples` is not
    /// divisible by `batch_size`.
    pub fn fit(&mut self, x: &[f64], n_samples: usize, n_features: usize) -> ManifoldResult<()> {
        if n_samples == 0 || n_features == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if x.len() != n_samples * n_features {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n_samples, n_features],
                got: vec![x.len()],
            });
        }
        if self.n_components == 0 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be >= 1".into(),
            });
        }
        let bs = self.batch_size;
        let mut start = 0;
        while start < n_samples {
            let end = (start + bs).min(n_samples);
            let rows = end - start;
            let slice = &x[start * n_features..end * n_features];
            self.partial_fit(slice, rows, n_features)?;
            start = end;
        }
        Ok(())
    }

    // ─── Transform ────────────────────────────────────────────────────────────

    /// Project `n_samples` rows of `x` onto the top-k components.
    ///
    /// Returns a `(n_samples × n_components)` row-major matrix:
    /// `Y = (X - mean) Vᵀ`   where each row of `V` is a principal component.
    pub fn transform(&self, x: &[f64], n_samples: usize) -> ManifoldResult<Vec<f64>> {
        if !self.fitted {
            return Err(ManifoldError::InvalidParameter {
                name: "model".into(),
                reason: "call fit or partial_fit before transform".into(),
            });
        }
        if n_samples == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        let d = self.n_features;
        let k = self.n_components;
        if x.len() != n_samples * d {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n_samples, d],
                got: vec![x.len()],
            });
        }

        let mut y = vec![0.0; n_samples * k];
        for i in 0..n_samples {
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += (x[i * d + j] - self.mean[j]) * self.components[c * d + j];
                }
                y[i * k + c] = acc;
            }
        }
        Ok(y)
    }

    // ─── Explained variance ratio ─────────────────────────────────────────────

    /// Per-component explained-variance ratio: `s_i² / Σ_j s_j²`.
    ///
    /// Returns a vector of zeros when the model has not been fitted yet.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total < 1e-30 {
            return vec![0.0; self.n_components];
        }
        self.singular_values.iter().map(|s| s * s / total).collect()
    }
}

// ─── Private matrix helpers ───────────────────────────────────────────────────

/// Compute the Gram matrix G = M Mᵀ for an (m × d) row-major matrix M.
/// Returns a row-major (m × m) symmetric matrix.
fn mat_gram(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut g = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in i..rows {
            let mut dot = 0.0;
            for k in 0..cols {
                dot += m[i * cols + k] * m[j * cols + k];
            }
            g[i * rows + j] = dot;
            g[j * rows + i] = dot;
        }
    }
    g
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::pca::pca_fit;

    /// Build data with variance concentrated on axis 0.
    fn axis_data(n: usize, d: usize) -> Vec<f64> {
        let mut x = vec![0.0; n * d];
        for i in 0..n {
            x[i * d] = (i as f64) - (n as f64 - 1.0) / 2.0;
        }
        x
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }

    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn partial_fit_single_chunk_matches_batch_pca() {
        // With one chunk = full data, IPCA should recover the same leading
        // component as batch PCA (up to sign).
        let n = 8;
        let d = 3;
        let x = axis_data(n, d);

        let cfg = IncrementalPcaConfig {
            n_components: 1,
            batch_size: n,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.partial_fit(&x, n, d).expect("partial_fit ok");

        let batch = pca_fit(&x, n, d, 1).expect("batch pca ok");

        // Leading component alignment — |cos θ| must be close to 1.
        let ipca_v0: Vec<f64> = (0..d).map(|j| ipca.components[j]).collect();
        let pca_v0: Vec<f64> = (0..d).map(|j| batch.components[j]).collect();
        let cos = dot(&ipca_v0, &pca_v0).abs();
        assert!(cos > 0.98, "alignment cos={cos}");
    }

    #[test]
    fn fit_many_chunks_convergence() {
        // 100 samples in 5D, processed in chunks of 10.
        // Explained variance ratios must sum to <= 1 and be positive.
        let n = 100;
        let d = 5;
        let mut x = vec![0.0; n * d];
        for i in 0..n {
            let v = (i as f64 - 50.0) * 0.1;
            x[i * d] = 4.0 * v;
            x[i * d + 1] = 2.0 * v;
            x[i * d + 2] = v;
            x[i * d + 3] = 0.5 * v;
            x[i * d + 4] = 0.1 * v;
        }

        let cfg = IncrementalPcaConfig {
            n_components: 2,
            batch_size: 10,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.fit(&x, n, d).expect("fit ok");

        let evr = ipca.explained_variance_ratio();
        let sum: f64 = evr.iter().sum();
        assert!(sum <= 1.0 + 1e-10, "EVR sum={sum} > 1");
        assert!(sum > 0.0, "EVR sum must be positive");
    }

    #[test]
    fn incremental_components_orthonormal() {
        // After fitting, each pair of rows of V must be orthonormal.
        let n = 50;
        let d = 4;
        let k = 3;
        let mut x = vec![0.0; n * d];
        for i in 0..n {
            let t = i as f64 / n as f64;
            x[i * d] = t;
            x[i * d + 1] = 2.0 * t - 1.0;
            x[i * d + 2] = (t * std::f64::consts::PI).sin();
            x[i * d + 3] = (t * std::f64::consts::PI).cos();
        }

        let cfg = IncrementalPcaConfig {
            n_components: k,
            batch_size: 10,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.fit(&x, n, d).expect("fit ok");

        // V Vᵀ must equal Iₖ  (components rows are unit vectors, pairwise orthogonal).
        for a in 0..k {
            for b in 0..k {
                let gram_ab: f64 = (0..d)
                    .map(|j| ipca.components[a * d + j] * ipca.components[b * d + j])
                    .sum();
                let target = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (gram_ab - target).abs() < 1e-8,
                    "VVᵀ[{a},{b}]={gram_ab:.6e}, expected {target}"
                );
            }
        }
    }

    #[test]
    fn transform_output_shape() {
        let n = 20;
        let d = 4;
        let k = 2;
        let x = axis_data(n, d);

        let cfg = IncrementalPcaConfig {
            n_components: k,
            batch_size: 5,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.fit(&x, n, d).expect("fit ok");

        let y = ipca.transform(&x, n).expect("transform ok");
        assert_eq!(
            y.len(),
            n * k,
            "output shape must be n_samples x n_components"
        );
    }

    #[test]
    fn mean_updates_correctly() {
        // Running mean after 3 partial_fit calls must equal the global column mean.
        let d = 3;
        let c1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]; // 4 rows
        let c2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]; // 3 rows
        let c3: Vec<f64> = (1..=15).map(|v| v as f64 * 0.5).collect(); // 5 rows

        let cfg = IncrementalPcaConfig {
            n_components: 1,
            batch_size: 4,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.partial_fit(&c1, 4, d).expect("c1 ok");
        ipca.partial_fit(&c2, 3, d).expect("c2 ok");
        ipca.partial_fit(&c3, 5, d).expect("c3 ok");

        // Compute global mean manually.
        let all: Vec<f64> = c1
            .iter()
            .chain(c2.iter())
            .chain(c3.iter())
            .copied()
            .collect();
        let n_total = 4 + 3 + 5;
        let mut global_mean = vec![0.0; d];
        for i in 0..n_total {
            for j in 0..d {
                global_mean[j] += all[i * d + j];
            }
        }
        for m in &mut global_mean {
            *m /= n_total as f64;
        }

        for (j, &gm) in global_mean.iter().enumerate().take(d) {
            assert!(
                (ipca.mean[j] - gm).abs() < 1e-10,
                "mean[{j}]: got {}, expected {}",
                ipca.mean[j],
                gm
            );
        }
    }

    #[test]
    fn explained_variance_ratio_sums_correctly() {
        let n = 40;
        let d = 4;
        let k = 3;
        let mut x = vec![0.0; n * d];
        for i in 0..n {
            let v = i as f64 - 20.0;
            x[i * d] = 5.0 * v;
            x[i * d + 1] = 2.0 * v;
            x[i * d + 2] = 0.8 * v;
            x[i * d + 3] = 0.1 * v;
        }

        let cfg = IncrementalPcaConfig {
            n_components: k,
            batch_size: 8,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        ipca.fit(&x, n, d).expect("fit ok");

        let evr = ipca.explained_variance_ratio();
        assert_eq!(evr.len(), k);

        for (i, &r) in evr.iter().enumerate() {
            assert!(r > 0.0, "evr[{i}]={r} must be positive");
        }

        // Ratios sorted descending (up to floating-point noise).
        assert!(evr[0] >= evr[1] - 1e-10, "EVR not sorted descending at 0,1");
        assert!(evr[1] >= evr[2] - 1e-10, "EVR not sorted descending at 1,2");

        let sum: f64 = evr.iter().sum();
        assert!(sum <= 1.0 + 1e-10, "EVR sum={sum} > 1");
        assert!(
            sum > 0.5,
            "EVR sum={sum} suspiciously small for structured data"
        );
    }

    #[test]
    fn empty_chunk_returns_error() {
        let cfg = IncrementalPcaConfig::default();
        let mut ipca = IncrementalPca::new(&cfg);
        let result = ipca.partial_fit(&[], 0, 3);
        assert!(
            matches!(result, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput, got {result:?}"
        );
    }

    #[test]
    fn mismatched_features_returns_error() {
        let cfg = IncrementalPcaConfig {
            n_components: 1,
            batch_size: 5,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        // First chunk: d=4, 5 rows.
        let c1: Vec<f64> = (0..20).map(|v| v as f64).collect();
        ipca.partial_fit(&c1, 5, 4).expect("first chunk ok");
        // Second chunk: d=3 — feature dimension mismatch, must fail.
        let c2: Vec<f64> = (0..15).map(|v| v as f64).collect();
        let result = ipca.partial_fit(&c2, 5, 3);
        assert!(
            matches!(result, Err(ManifoldError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {result:?}"
        );
    }

    #[test]
    fn zero_components_returns_error() {
        let cfg = IncrementalPcaConfig {
            n_components: 0,
            batch_size: 5,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        let chunk: Vec<f64> = (0..20).map(|v| v as f64).collect();
        let result = ipca.partial_fit(&chunk, 5, 4);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "expected InvalidParameter, got {result:?}"
        );
    }

    #[test]
    fn single_sample_chunk_works() {
        // A chunk of exactly 1 row must succeed and produce a valid model.
        let d = 3;
        let k = 1;
        let cfg = IncrementalPcaConfig {
            n_components: k,
            batch_size: 1,
        };
        let mut ipca = IncrementalPca::new(&cfg);
        let row = vec![1.0, 2.0, 3.0];
        ipca.partial_fit(&row, 1, d).expect("single row ok");
        assert!(ipca.fitted);
        assert_eq!(ipca.n_samples_seen, 1);
        // Transform must succeed.
        let y = ipca.transform(&row, 1).expect("transform ok");
        assert_eq!(y.len(), k);
    }

    #[test]
    fn transform_before_fit_returns_error() {
        let cfg = IncrementalPcaConfig::default();
        let ipca = IncrementalPca::new(&cfg);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = ipca.transform(&x, 2);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "expected InvalidParameter, got {result:?}"
        );
    }
}
