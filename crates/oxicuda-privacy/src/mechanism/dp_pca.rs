//! Differentially-Private Principal Component Analysis (DP-PCA) via the
//! Analyze-Gauss mechanism (Dwork, Talwar, Thakurta, Zhang, STOC 2014,
//! <https://arxiv.org/abs/1405.7085>).
//!
//! # Algorithm
//! Given `X ∈ ℝ^{n × d}`:
//! 1. **Clip** each row `xᵢ` to L2 norm `≤ clip_norm` (default 1).  This
//!    bounds the Frobenius norm of the rank-1 update `xᵢxᵢᵀ` by `clip_norm²`.
//! 2. Form **Gram-like** `C = (1/n) · XᵀX  ∈  ℝ^{d × d}` (symmetric).
//! 3. Add **symmetric Gaussian noise**: for `i ≤ j` draw `e_{ij} ~ N(0, σ²)`
//!    with `σ = (Δ / n) · √(2 ln(1.25 / δ)) / ε`, `Δ = clip_norm²`.  Mirror
//!    the noise to keep `C + Ñ` symmetric.
//! 4. Eigendecompose using **cyclic Jacobi rotations** (Golub-Van Loan 4th ed.,
//!    §8.4) up to `max_iter` sweeps with off-diagonal tolerance `tol`.
//! 5. Sort eigenpairs descending and emit the top `k` eigenvectors.
//!
//! Calibration follows the standard Gaussian-mechanism analysis (Dwork-Roth
//! 2014, Theorem A.1).  All RNG draws flow through `LcgRng::normal_pair`
//! (Box-Muller) — no `rand` crate.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Analyze-Gauss DP-PCA mechanism.
#[derive(Debug, Clone)]
pub struct DpPcaConfig {
    /// Number of principal components to return (`1 ≤ k ≤ d`).
    pub n_components: usize,
    /// Privacy parameter `ε > 0`.
    pub epsilon: f64,
    /// Failure probability `δ ∈ (0, 1)`.
    pub delta: f64,
    /// Per-row L2 clipping bound (defines the per-row contribution norm).
    pub clip_norm: f64,
    /// Maximum number of Jacobi sweeps.  100 is the canonical default
    /// (Golub-Van Loan §8.4).
    pub max_iter: usize,
    /// Off-diagonal convergence tolerance for Jacobi (typically `1e-10`).
    pub tol: f64,
}

impl DpPcaConfig {
    /// Sensible defaults: `clip_norm = 1.0`, `max_iter = 100`, `tol = 1e-10`.
    #[must_use]
    pub fn new(n_components: usize, epsilon: f64, delta: f64) -> Self {
        Self {
            n_components,
            epsilon,
            delta,
            clip_norm: 1.0,
            max_iter: 100,
            tol: 1e-10,
        }
    }
}

/// Result of a DP-PCA call.
#[derive(Debug, Clone)]
pub struct DpPcaResult {
    /// Top-k principal components, row-major `[k × d]` (each row a unit vector).
    pub components: Vec<f64>,
    /// Corresponding eigenvalues (sorted descending).  Length `= k`.
    pub eigenvalues: Vec<f64>,
    /// Noise scale `σ` used for the symmetric Gaussian perturbation.
    pub sigma: f64,
}

impl DpPcaResult {
    /// Access the components view (`[k × d]` row-major).
    #[must_use]
    #[inline]
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Access the eigenvalues (length `k`).
    #[must_use]
    #[inline]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Noise-scale `σ` actually applied.
    #[must_use]
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

/// Validate a DP-PCA configuration against the data shape.
fn validate(cfg: &DpPcaConfig, n_rows: usize, n_cols: usize, x_len: usize) -> PrivacyResult<()> {
    if n_rows == 0 || n_cols == 0 {
        return Err(PrivacyError::EmptyInput);
    }
    if n_rows.checked_mul(n_cols) != Some(x_len) {
        return Err(PrivacyError::DimensionMismatch {
            expected: n_rows.saturating_mul(n_cols),
            got: x_len,
        });
    }
    if cfg.n_components == 0 {
        return Err(PrivacyError::InvalidParameter(
            "n_components must be ≥ 1".into(),
        ));
    }
    if cfg.n_components > n_cols {
        return Err(PrivacyError::InvalidParameter(format!(
            "n_components ({}) exceeds n_cols ({})",
            cfg.n_components, n_cols
        )));
    }
    if !(cfg.epsilon.is_finite() && cfg.epsilon > 0.0) {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if !(cfg.delta > 0.0 && cfg.delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(cfg.delta));
    }
    if !(cfg.clip_norm.is_finite() && cfg.clip_norm > 0.0) {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.clip_norm));
    }
    if cfg.max_iter == 0 {
        return Err(PrivacyError::InvalidParameter(
            "max_iter must be ≥ 1".into(),
        ));
    }
    if !(cfg.tol.is_finite() && cfg.tol >= 0.0) {
        return Err(PrivacyError::InvalidParameter(format!(
            "tol must be finite and ≥ 0, got {}",
            cfg.tol
        )));
    }
    Ok(())
}

/// In-place L2 clip of each row to `clip_norm`.  Rows already within the
/// ball are unchanged.
fn clip_rows(x: &mut [f64], n_rows: usize, n_cols: usize, clip_norm: f64) {
    for i in 0..n_rows {
        let row = &mut x[i * n_cols..(i + 1) * n_cols];
        let mut sq: f64 = 0.0;
        for &v in row.iter() {
            sq += v * v;
        }
        let norm = sq.sqrt();
        if norm > clip_norm && norm > 0.0 {
            let s = clip_norm / norm;
            for v in row.iter_mut() {
                *v *= s;
            }
        }
    }
}

/// Compute `C = (1/n) · XᵀX`.  Returns a `d × d` symmetric matrix in row-major.
fn gram_matrix(x: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n_cols * n_cols];
    let inv_n = 1.0 / n_rows as f64;
    for i in 0..n_rows {
        let row = &x[i * n_cols..(i + 1) * n_cols];
        for a in 0..n_cols {
            let ra = row[a];
            for b in a..n_cols {
                c[a * n_cols + b] += ra * row[b];
            }
        }
    }
    // Scale and mirror to the lower triangle.
    for a in 0..n_cols {
        for b in a..n_cols {
            let v = c[a * n_cols + b] * inv_n;
            c[a * n_cols + b] = v;
            c[b * n_cols + a] = v;
        }
    }
    c
}

/// Add symmetric Gaussian noise to a `d × d` matrix in-place.
fn add_symmetric_gaussian_noise(c: &mut [f64], d: usize, sigma: f64, rng: &mut LcgRng) {
    if sigma == 0.0 {
        return;
    }
    let mut held: Option<f64> = None;
    let mut draw = |rng: &mut LcgRng| -> f64 {
        if let Some(v) = held.take() {
            return v;
        }
        let (a, b) = rng.normal_pair();
        held = Some(b);
        a
    };
    for i in 0..d {
        for j in i..d {
            let z = draw(rng) * sigma;
            c[i * d + j] += z;
            if i != j {
                c[j * d + i] += z;
            }
        }
    }
}

/// Symmetric cyclic Jacobi eigendecomposition.
///
/// Input: symmetric `d × d` matrix `a` in row-major.  Output:
/// - `eigvals[k]`  — `a[k, k]` after rotations
/// - `eigvecs[k·d + j]`  — `j`-th component of the `k`-th eigenvector
///
/// The routine sweeps over pairs `(p, q)` with `p < q`, rotating to zero the
/// largest off-diagonal magnitude until either `max_sweeps` is reached or the
/// off-diagonal Frobenius norm falls below `tol`.
fn jacobi_eigendecomp(
    a: &mut [f64],
    d: usize,
    max_sweeps: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    // Initialise the eigenvector matrix V to the identity.
    let mut v = vec![0.0_f64; d * d];
    for k in 0..d {
        v[k * d + k] = 1.0;
    }

    let threshold = tol.max(0.0);

    for _sweep in 0..max_sweeps {
        // Sum of squares of off-diagonal entries.
        let mut off = 0.0_f64;
        for p in 0..d {
            for q in (p + 1)..d {
                let x = a[p * d + q];
                off += x * x;
            }
        }
        if off.sqrt() <= threshold {
            break;
        }

        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() <= f64::EPSILON {
                    continue;
                }
                let app = a[p * d + p];
                let aqq = a[q * d + q];

                // Compute rotation angle (Golub-Van Loan §8.4.1, formula 8.4.4).
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1.0e150 {
                    0.5 / theta
                } else if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal.
                a[p * d + p] = app - t * apq;
                a[q * d + q] = aqq + t * apq;
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;

                // Update remaining rows/cols.
                for r in 0..d {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * d + p];
                    let arq = a[r * d + q];
                    let new_arp = c * arp - s * arq;
                    let new_arq = s * arp + c * arq;
                    a[r * d + p] = new_arp;
                    a[p * d + r] = new_arp;
                    a[r * d + q] = new_arq;
                    a[q * d + r] = new_arq;
                }

                // Update eigenvector matrix V (right-multiply).
                for r in 0..d {
                    let vrp = v[r * d + p];
                    let vrq = v[r * d + q];
                    v[r * d + p] = c * vrp - s * vrq;
                    v[r * d + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalues from the diagonal and eigenvectors as columns of V.
    let mut eigvals = vec![0.0_f64; d];
    let mut eigvecs = vec![0.0_f64; d * d];
    for k in 0..d {
        eigvals[k] = a[k * d + k];
        for j in 0..d {
            eigvecs[k * d + j] = v[j * d + k];
        }
    }
    (eigvals, eigvecs)
}

/// Sort eigenpairs descending by eigenvalue and pick the top `k`.
fn select_top_k(eigvals: &[f64], eigvecs: &[f64], d: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top_vals = Vec::with_capacity(k);
    let mut top_vecs = vec![0.0_f64; k * d];
    for (out_i, &src) in idx.iter().take(k).enumerate() {
        top_vals.push(eigvals[src]);
        let dst = &mut top_vecs[out_i * d..(out_i + 1) * d];
        let src_vec = &eigvecs[src * d..(src + 1) * d];
        // Normalise (Jacobi already gives unit vectors but renormalise for safety).
        let mut sq = 0.0_f64;
        for &v in src_vec.iter() {
            sq += v * v;
        }
        let norm = sq.sqrt().max(f64::MIN_POSITIVE);
        for (a, &b) in dst.iter_mut().zip(src_vec.iter()) {
            *a = b / norm;
        }
    }
    (top_vals, top_vecs)
}

/// Run differentially-private PCA on a row-major `n × d` data matrix.
///
/// # Errors
/// Returns one of `EmptyInput`, `DimensionMismatch`, `InvalidParameter`,
/// `NonPositiveEpsilon`, `InvalidDelta`, or `NonPositiveSensitivity` if
/// inputs are malformed.
pub fn dp_pca(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    cfg: &DpPcaConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<DpPcaResult> {
    validate(cfg, n_rows, n_cols, x.len())?;

    let mut clipped = x.to_vec();
    clip_rows(&mut clipped, n_rows, n_cols, cfg.clip_norm);

    let mut c = gram_matrix(&clipped, n_rows, n_cols);

    // Frobenius sensitivity: ‖Δ_i‖_F ≤ clip_norm² for the rank-1 update
    // xᵢxᵢᵀ.  Dividing by n gives the contribution to C.
    let delta_f = cfg.clip_norm * cfg.clip_norm;
    let sigma = (delta_f / n_rows as f64) * (2.0 * (1.25 / cfg.delta).ln()).sqrt() / cfg.epsilon;

    add_symmetric_gaussian_noise(&mut c, n_cols, sigma, rng);

    let (eigvals, eigvecs) = jacobi_eigendecomp(&mut c, n_cols, cfg.max_iter, cfg.tol);
    let (top_vals, top_vecs) = select_top_k(&eigvals, &eigvecs, n_cols, cfg.n_components);

    Ok(DpPcaResult {
        components: top_vecs,
        eigenvalues: top_vals,
        sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute cosine similarity between two equally-sized vectors.
    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
    }

    fn make_aligned_data(n: usize) -> Vec<f64> {
        // Each row is [1, 0] with tiny perturbation.
        let mut x = Vec::with_capacity(n * 2);
        for i in 0..n {
            let p = (i as f64) * 1e-4;
            x.push(1.0 - p);
            x.push(p);
        }
        x
    }

    #[test]
    fn test_clip_behaviour_rejects_oversized_row() {
        // Row with norm 5 must be clipped to norm 1.
        let mut data = vec![3.0, 4.0, 1.0, 0.0]; // 2 rows × 2 cols
        clip_rows(&mut data, 2, 2, 1.0);
        let n0 = (data[0] * data[0] + data[1] * data[1]).sqrt();
        let n1 = (data[2] * data[2] + data[3] * data[3]).sqrt();
        assert!((n0 - 1.0).abs() < 1e-12, "row 0 norm = {n0}");
        assert!((n1 - 1.0).abs() < 1e-12, "row 1 norm = {n1}");
    }

    #[test]
    fn test_no_noise_recovers_top_component() {
        // Huge ε ⇒ negligible noise; top-1 should align with [1, 0].
        let x = make_aligned_data(200);
        let mut cfg = DpPcaConfig::new(1, 1.0, 1e-6);
        cfg.epsilon = 1e9; // effectively no noise
        let mut rng = LcgRng::new(7);
        let res = dp_pca(&x, 200, 2, &cfg, &mut rng).expect("ok");
        assert_eq!(res.components.len(), 2);
        let target = [1.0, 0.0];
        let cos = cosine(&res.components, &target).abs();
        assert!(cos > 0.95, "cosine = {cos}");
    }

    #[test]
    fn test_deterministic_with_seed() {
        let x = make_aligned_data(50);
        let cfg = DpPcaConfig::new(1, 1.0, 1e-3);
        let mut rng_a = LcgRng::new(1234);
        let mut rng_b = LcgRng::new(1234);
        let a = dp_pca(&x, 50, 2, &cfg, &mut rng_a).expect("ok");
        let b = dp_pca(&x, 50, 2, &cfg, &mut rng_b).expect("ok");
        assert_eq!(a.components, b.components);
        assert_eq!(a.eigenvalues, b.eigenvalues);
    }

    #[test]
    fn test_variance_recovery_aligned_data() {
        let x = make_aligned_data(500);
        let mut cfg = DpPcaConfig::new(1, 1.0, 1e-3);
        cfg.epsilon = 100.0; // light noise
        let mut rng = LcgRng::new(99);
        let res = dp_pca(&x, 500, 2, &cfg, &mut rng).expect("ok");
        let cos = cosine(&res.components[0..2], &[1.0, 0.0]).abs();
        assert!(cos > 0.9, "cosine = {cos}");
    }

    #[test]
    fn test_sigma_scales_inversely_with_epsilon() {
        let x = make_aligned_data(100);
        let cfg_lo = DpPcaConfig::new(1, 0.5, 1e-3);
        let cfg_hi = DpPcaConfig::new(1, 5.0, 1e-3);
        let mut rng = LcgRng::new(7);
        let lo = dp_pca(&x, 100, 2, &cfg_lo, &mut rng).expect("ok");
        let hi = dp_pca(&x, 100, 2, &cfg_hi, &mut rng).expect("ok");
        // 10× ε ⇒ σ scales by 1/10.
        assert!(
            (lo.sigma / hi.sigma - 10.0).abs() < 1e-9,
            "ratio = {}",
            lo.sigma / hi.sigma
        );
    }

    #[test]
    fn test_n_components_zero_rejected() {
        let x = vec![0.0; 6];
        let cfg = DpPcaConfig::new(0, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_n_components_exceeds_d_rejected() {
        let x = vec![0.0; 6];
        let cfg = DpPcaConfig::new(5, 1.0, 1e-3); // d = 3
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_epsilon_nonpositive_rejected() {
        let x = vec![0.0; 6];
        let cfg = DpPcaConfig::new(1, 0.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&x, 2, 3, &cfg, &mut rng).is_err());
        let cfg2 = DpPcaConfig::new(1, -1.0, 1e-3);
        assert!(dp_pca(&x, 2, 3, &cfg2, &mut rng).is_err());
    }

    #[test]
    fn test_delta_out_of_range_rejected() {
        let x = vec![0.0; 6];
        let cfg_lo = DpPcaConfig::new(1, 1.0, 0.0);
        let cfg_hi = DpPcaConfig::new(1, 1.0, 1.0);
        let cfg_neg = DpPcaConfig::new(1, 1.0, -0.1);
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&x, 2, 3, &cfg_lo, &mut rng).is_err());
        assert!(dp_pca(&x, 2, 3, &cfg_hi, &mut rng).is_err());
        assert!(dp_pca(&x, 2, 3, &cfg_neg, &mut rng).is_err());
    }

    #[test]
    fn test_dim_mismatch_rejected() {
        let x = vec![0.0; 5]; // 5 elements, not 2*3 = 6
        let cfg = DpPcaConfig::new(1, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&x, 2, 3, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_components_are_unit_norm() {
        let x = make_aligned_data(50);
        let cfg = DpPcaConfig::new(2, 1.0, 1e-3);
        let mut rng = LcgRng::new(11);
        let res = dp_pca(&x, 50, 2, &cfg, &mut rng).expect("ok");
        for k in 0..2 {
            let row = &res.components[k * 2..(k + 1) * 2];
            let n = (row[0] * row[0] + row[1] * row[1]).sqrt();
            assert!((n - 1.0).abs() < 1e-9, "comp {k} norm = {n}");
        }
    }

    #[test]
    fn test_eigenvalues_sorted_descending() {
        let x = make_aligned_data(50);
        let cfg = DpPcaConfig::new(2, 1.0, 1e-3);
        let mut rng = LcgRng::new(22);
        let res = dp_pca(&x, 50, 2, &cfg, &mut rng).expect("ok");
        for k in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[k - 1] >= res.eigenvalues[k],
                "{} < {} at index {k}",
                res.eigenvalues[k - 1],
                res.eigenvalues[k]
            );
        }
    }

    #[test]
    fn test_one_dim_data() {
        // d = 1, k must be 1.
        let x = vec![0.5_f64; 10];
        let cfg = DpPcaConfig::new(1, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        let res = dp_pca(&x, 10, 1, &cfg, &mut rng).expect("ok");
        assert_eq!(res.components.len(), 1);
        assert!((res.components[0].abs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_input_rejected() {
        let cfg = DpPcaConfig::new(1, 1.0, 1e-3);
        let mut rng = LcgRng::new(0);
        assert!(dp_pca(&[], 0, 3, &cfg, &mut rng).is_err());
        assert!(dp_pca(&[], 3, 0, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_jacobi_diagonal_matrix() {
        // Eigendecomposition of a diagonal matrix is trivial.
        let mut a = vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let (vals, vecs) = jacobi_eigendecomp(&mut a, 3, 100, 1e-12);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 3.0).abs() < 1e-9);
        assert!((sorted[1] - 2.0).abs() < 1e-9);
        assert!((sorted[2] - 1.0).abs() < 1e-9);
        // Eigenvectors are columns of an identity (in some permutation).
        for k in 0..3 {
            let mut sq = 0.0;
            for j in 0..3 {
                sq += vecs[k * 3 + j] * vecs[k * 3 + j];
            }
            assert!((sq - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        // Verify the helper produces a symmetric matrix.
        let x = vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0];
        let c = gram_matrix(&x, 2, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((c[i * 3 + j] - c[j * 3 + i]).abs() < 1e-12);
            }
        }
    }
}
