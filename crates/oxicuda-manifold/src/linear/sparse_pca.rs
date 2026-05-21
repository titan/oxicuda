//! Sparse PCA via Penalized Matrix Decomposition (Witten-Tibshirani-Hastie 2009).
//!
//! Solves:  max u^T X v   s.t.  ||u||₂ = 1,  ||v||₂ ≤ 1,  ||v||₁ ≤ c_v
//!
//! by alternating updates:
//!  1. u ← X v / ||X v||₂
//!  2. v ← soft_threshold(X^T u, λ) / ||soft_threshold(X^T u, λ)||₂
//!
//! where λ is chosen via bisection so that ||v||₁ = c_v (or 0 if already satisfied).
//! Multiple components are extracted by PMD deflation: after finding (u_k, v_k),
//! deflate X ← X − (X v_k) v_k^T and repeat.

use crate::error::{ManifoldError, ManifoldResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public config / result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Sparse PCA (PMD).
#[derive(Debug, Clone)]
pub struct SparsePcaConfig {
    /// Number of sparse components to extract.
    pub n_components: usize,
    /// L1 norm bound on the right singular vector v (the c_v parameter).
    ///
    /// When `l1_bound ≥ sqrt(n_features)` the constraint is inactive and the
    /// algorithm reduces to ordinary PCA on the first component.
    /// Must be strictly positive.
    pub l1_bound: f64,
    /// Maximum number of alternating-update iterations per component.
    pub max_iter: usize,
    /// Convergence tolerance for max |v_new − v_old|.
    pub tol: f64,
}

impl Default for SparsePcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            l1_bound: 1.0,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

/// Result of a Sparse PCA fit.
#[derive(Debug)]
pub struct SparsePcaResult {
    /// Sparse right singular vectors, row-major `(n_components, n_features)`.
    ///
    /// Each row is the v_k vector; entries are zero wherever sparsity was imposed.
    pub components: Vec<f64>,
    /// Score (loading) vectors, row-major `(n_components, n_samples)`.
    ///
    /// Each row is u_k (the left singular vector) so projections are already
    /// available without a separate matrix multiply.
    pub loadings: Vec<f64>,
    /// Number of components extracted.
    pub n_components: usize,
    /// Number of features (columns) in the input.
    pub n_features: usize,
    /// Per-component criterion value u_k^T X v_k.
    pub variances: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (private to this module)
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc + x * x).sqrt()
}

/// L1 norm of a slice.
#[inline]
fn l1_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x.abs()).sum()
}

/// Soft-threshold operator:  sign(z) * max(|z| - lambda, 0).
#[inline]
fn soft_threshold(z: f64, lambda: f64) -> f64 {
    let az = z.abs();
    if az <= lambda {
        0.0
    } else {
        z.signum() * (az - lambda)
    }
}

/// Apply soft-threshold to every element of `w` with threshold `lambda`,
/// returning the result as a new Vec.
fn soft_threshold_vec(w: &[f64], lambda: f64) -> Vec<f64> {
    w.iter().map(|&z| soft_threshold(z, lambda)).collect()
}

/// Compute X v  where X is `(n_samples × n_features)` row-major and v is length
/// `n_features`.  Returns a Vec of length `n_samples`.
fn mat_vec(x: &[f64], n_samples: usize, n_features: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_samples];
    for i in 0..n_samples {
        let mut acc = 0.0_f64;
        for j in 0..n_features {
            acc += x[i * n_features + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// Compute X^T u  where X is `(n_samples × n_features)` row-major and u is length
/// `n_samples`.  Returns a Vec of length `n_features`.
fn mat_t_vec(x: &[f64], n_samples: usize, n_features: usize, u: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_features];
    for j in 0..n_features {
        let mut acc = 0.0_f64;
        for i in 0..n_samples {
            acc += x[i * n_features + j] * u[i];
        }
        out[j] = acc;
    }
    out
}

/// Find the Lagrange multiplier λ ≥ 0 such that after soft-thresholding and
/// L2 normalisation the L1 norm of the resulting v satisfies:
///
///   ||soft_threshold(w, λ)||₁ / ||soft_threshold(w, λ)||₂ = target
///
/// i.e. the *normalized* v vector satisfies `||v||₁ = target`.
/// If the unconstrained normalized L1 is already ≤ target, returns 0 (no
/// thresholding needed).
///
/// Bisection bounds: λ ∈ [0, max|w|).  At λ → max|w| only a single component
/// survives, giving normalized L1 = 1 (minimum). At λ = 0 the normalized L1
/// is ||w||₁ / ||w||₂ (maximum).
fn bisect_lambda(w: &[f64], target: f64) -> f64 {
    // Evaluate normalized L1 at λ = 0
    let l2_w = l2_norm(w);
    if l2_w < 1e-300 {
        return 0.0;
    }
    let l1_normalized_unconstrained = l1_norm(w) / l2_w;
    if l1_normalized_unconstrained <= target {
        return 0.0;
    }
    let max_w = w.iter().map(|&z| z.abs()).fold(0.0_f64, f64::max);
    if max_w == 0.0 {
        return 0.0;
    }
    let mut lo = 0.0_f64;
    let mut hi = max_w * (1.0 - 1e-12); // avoid zeroing all entries
    // 60 bisection iterations comfortably reach machine precision.
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let st = soft_threshold_vec(w, mid);
        let l2_st = l2_norm(&st);
        if l2_st < 1e-300 {
            // All entries zeroed — move hi inward
            hi = mid;
            continue;
        }
        let l1_st_normalized = l1_norm(&st) / l2_st;
        if l1_st_normalized > target {
            lo = mid; // need more thresholding
        } else {
            hi = mid; // need less thresholding
        }
    }
    0.5 * (lo + hi)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-component PMD (rank-1 solve on the working matrix X_work)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract one rank-1 PMD component from `x_work`.
///
/// Returns `(u, v, converged)` where `u` has length `n_samples` and `v` has
/// length `n_features`, both normalised to unit L2 norm.
fn pmd_rank1(
    x_work: &[f64],
    n_samples: usize,
    n_features: usize,
    l1_bound: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, bool) {
    // ── Initialise v as uniform unit vector 1/√d ──────────────────────────
    let inv_sqrt_d = 1.0 / (n_features as f64).sqrt();
    let mut v = vec![inv_sqrt_d; n_features];

    let mut converged = false;
    for _iter in 0..max_iter {
        // ── Step 1: update u ─────────────────────────────────────────────────
        // u ← X v / ||X v||₂
        let xv = mat_vec(x_work, n_samples, n_features, &v);
        let xv_norm = l2_norm(&xv);
        let u = if xv_norm < 1e-300 {
            // Degenerate — keep previous or uniform; just return
            let uv = vec![1.0 / (n_samples as f64).sqrt(); n_samples];
            return (uv, v, false);
        } else {
            xv.iter().map(|&z| z / xv_norm).collect::<Vec<_>>()
        };

        // ── Step 2: update v ─────────────────────────────────────────────────
        // w ← X^T u
        let w = mat_t_vec(x_work, n_samples, n_features, &u);
        // find λ via bisection so that ||soft_threshold(w, λ)||₁ = l1_bound
        let lambda = bisect_lambda(&w, l1_bound);
        let st = soft_threshold_vec(&w, lambda);
        let st_norm = l2_norm(&st);
        let v_new: Vec<f64> = if st_norm < 1e-300 {
            // All entries zeroed out by thresholding — return uniform
            vec![inv_sqrt_d; n_features]
        } else {
            st.iter().map(|&z| z / st_norm).collect()
        };

        // ── Convergence check ────────────────────────────────────────────────
        let max_delta = v_new
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        v = v_new;
        if max_delta < tol {
            converged = true;
            // Final u update with the converged v
            let xv_final = mat_vec(x_work, n_samples, n_features, &v);
            let xv_norm_final = l2_norm(&xv_final);
            let u_final: Vec<f64> = if xv_norm_final < 1e-300 {
                vec![1.0 / (n_samples as f64).sqrt(); n_samples]
            } else {
                xv_final.iter().map(|&z| z / xv_norm_final).collect()
            };
            return (u_final, v, converged);
        }
        let _ = u; // u is recomputed next iteration; v is the convergence variable
    }

    // max_iter exceeded — compute u from the current v and return
    let xv = mat_vec(x_work, n_samples, n_features, &v);
    let xv_norm = l2_norm(&xv);
    let u_final: Vec<f64> = if xv_norm < 1e-300 {
        vec![1.0 / (n_samples as f64).sqrt(); n_samples]
    } else {
        xv.iter().map(|&z| z / xv_norm).collect()
    };
    (u_final, v, converged)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit Sparse PCA (PMD) on row-major data `x` of shape `n_samples × n_features`.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] if `n_samples == 0` or `n_features == 0`.
/// - [`ManifoldError::ShapeMismatch`] if `x.len() != n_samples * n_features`.
/// - [`ManifoldError::InvalidParameter`] if `n_components`, `l1_bound`, `max_iter`,
///   or `tol` are out of range.
/// - [`ManifoldError::NotConverged`] if *none* of the requested components converge.
pub fn sparse_pca(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    config: &SparsePcaConfig,
) -> ManifoldResult<SparsePcaResult> {
    // ── Input validation ─────────────────────────────────────────────────────
    if n_samples == 0 || n_features == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * n_features {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if config.n_components == 0 || config.n_components > n_features.min(n_samples) {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!(
                "must be in 1..={}, got {}",
                n_features.min(n_samples),
                config.n_components
            ),
        });
    }
    if config.l1_bound <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "l1_bound".into(),
            reason: format!("must be strictly positive, got {}", config.l1_bound),
        });
    }
    if config.max_iter == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "max_iter".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if config.tol <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "tol".into(),
            reason: format!("must be strictly positive, got {}", config.tol),
        });
    }

    // ── Centre data by subtracting column means ──────────────────────────────
    let mut col_means = vec![0.0_f64; n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            col_means[j] += x[i * n_features + j];
        }
    }
    for m in &mut col_means {
        *m /= n_samples as f64;
    }
    let mut x_work: Vec<f64> = x
        .chunks_exact(n_features)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &v)| v - col_means[j])
                .collect::<Vec<_>>()
        })
        .collect();

    let k = config.n_components;
    let mut components = vec![0.0_f64; k * n_features]; // v vectors
    let mut loadings = vec![0.0_f64; k * n_samples]; // u vectors
    let mut variances = vec![0.0_f64; k];

    let mut any_converged = false;

    for comp in 0..k {
        let (u, v, converged) = pmd_rank1(
            &x_work,
            n_samples,
            n_features,
            config.l1_bound,
            config.max_iter,
            config.tol,
        );

        if converged {
            any_converged = true;
        }

        // Criterion value: u^T X v  (since u = X v / ||X v||, this equals ||X v||)
        let xv = mat_vec(&x_work, n_samples, n_features, &v);
        let criterion: f64 = u.iter().zip(xv.iter()).map(|(&ui, &xvi)| ui * xvi).sum();
        variances[comp] = criterion;

        // Store component (v) and loading (u)
        components[comp * n_features..(comp + 1) * n_features].copy_from_slice(&v);
        loadings[comp * n_samples..(comp + 1) * n_samples].copy_from_slice(&u);

        // ── PMD deflation: X ← X − (X v) v^T ───────────────────────────────
        // x_work[i, j] -= xv[i] * v[j]
        for i in 0..n_samples {
            for j in 0..n_features {
                x_work[i * n_features + j] -= xv[i] * v[j];
            }
        }
    }

    if !any_converged {
        return Err(ManifoldError::NotConverged {
            iter: config.max_iter,
        });
    }

    Ok(SparsePcaResult {
        components,
        loadings,
        n_components: k,
        n_features,
        variances,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ───────────────────────────────────────────────────────────────

    /// Build a simple dataset where all variance is in dimension 0:
    ///   x[:, 0] ~ linspace(-n/2, n/2)
    ///   x[:, 1] = 0
    fn axis0_data(n: usize) -> Vec<f64> {
        let mut data = vec![0.0_f64; n * 2];
        for i in 0..n {
            data[i * 2] = i as f64 - (n as f64 - 1.0) / 2.0;
            // column 1 stays 0
        }
        data
    }

    // ── 1. recovers sparse direction ─────────────────────────────────────────

    #[test]
    fn sparse_pca_recovers_sparse_direction() {
        let n = 20;
        let data = axis0_data(n);
        let cfg = SparsePcaConfig {
            n_components: 1,
            l1_bound: 1.0,
            ..Default::default()
        };
        let res = sparse_pca(&data, n, 2, &cfg).expect("should converge");
        // First component should load heavily on dimension 0, not dimension 1
        let v0 = res.components[0].abs();
        let v1 = res.components[1].abs();
        assert!(
            v0 > v1,
            "expected |v[0]| > |v[1]|, got v0={v0:.6}, v1={v1:.6}"
        );
    }

    // ── 2. components are sparse with tight l1_bound ─────────────────────────

    #[test]
    fn sparse_components_are_sparse() {
        // 20-dimensional data; l1_bound = 0.5 forces most entries to zero
        let n = 30;
        let d = 20;
        // Random-like deterministic data
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            for j in 0..d {
                let v = ((i * d + j) as f64 * 0.37).sin() * 3.0;
                data[i * d + j] = if j == 0 { v * 5.0 } else { v };
            }
        }
        let cfg = SparsePcaConfig {
            n_components: 1,
            l1_bound: 0.5,
            max_iter: 500,
            tol: 1e-8,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("should converge");
        let zero_count = res.components.iter().filter(|&&z| z == 0.0).count();
        // With l1_bound=0.5 on a 20-d vector we expect ≥ 10 zeros
        assert!(
            zero_count >= 10,
            "expected ≥ 10 zeros with l1_bound=0.5 in 20-d, got {zero_count}"
        );
    }

    // ── 3. unconstrained sparse PCA ≈ regular PCA first component ────────────

    #[test]
    fn sparse_pca_k1_vs_full_pca_unconstrained() {
        let n = 30;
        let d = 5;
        // structured data: dominant direction is (1,1,1,1,1)/√5
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            let t = i as f64 - (n as f64 - 1.0) / 2.0;
            for j in 0..d {
                // Strong shared component + small noise
                data[i * d + j] = t + ((i * d + j) as f64 * 0.1).sin() * 0.05;
            }
        }
        // l1_bound = sqrt(d) makes the L1 constraint inactive
        let l1_bound = (d as f64).sqrt();
        let cfg = SparsePcaConfig {
            n_components: 1,
            l1_bound,
            max_iter: 500,
            tol: 1e-9,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        // All loadings should have similar magnitude (uniform direction)
        let v = &res.components[..d];
        let mean_abs: f64 = v.iter().map(|&z| z.abs()).sum::<f64>() / d as f64;
        for &vi in v {
            assert!(
                (vi.abs() - mean_abs).abs() < 0.15,
                "expected approx uniform direction, got {vi:.4}"
            );
        }
    }

    // ── 4. L1 norm within bound ──────────────────────────────────────────────

    #[test]
    fn components_l1_within_bound() {
        let n = 25;
        let d = 8;
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            for j in 0..d {
                data[i * d + j] = ((i + j * 3) as f64 * 0.5).cos();
            }
        }
        let l1_bound = 1.5_f64;
        let cfg = SparsePcaConfig {
            n_components: 3,
            l1_bound,
            max_iter: 300,
            tol: 1e-7,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        for k in 0..3 {
            let v = &res.components[k * d..(k + 1) * d];
            let l1 = l1_norm(v);
            assert!(
                l1 <= l1_bound + 1e-4,
                "component {k}: ||v||₁ = {l1:.6} > l1_bound + 1e-4 = {}",
                l1_bound + 1e-4
            );
        }
    }

    // ── 5. L2 unit norm ──────────────────────────────────────────────────────

    #[test]
    fn components_l2_unit_norm() {
        let n = 20;
        let d = 6;
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            for j in 0..d {
                data[i * d + j] = ((i * 7 + j * 3) as f64 * 0.3).sin();
            }
        }
        let cfg = SparsePcaConfig {
            n_components: 2,
            l1_bound: 1.2,
            max_iter: 300,
            tol: 1e-8,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        for k in 0..2 {
            let v = &res.components[k * d..(k + 1) * d];
            let norm = l2_norm(v);
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "component {k}: ||v||₂ = {norm:.8}, expected 1.0"
            );
        }
    }

    // ── 6. output shapes ─────────────────────────────────────────────────────

    #[test]
    fn output_shapes() {
        let n = 15;
        let d = 7;
        let k = 3;
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64).sin()).collect();
        let cfg = SparsePcaConfig {
            n_components: k,
            l1_bound: 1.0,
            ..Default::default()
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        assert_eq!(res.components.len(), k * d, "components shape");
        assert_eq!(res.loadings.len(), k * n, "loadings shape");
        assert_eq!(res.variances.len(), k, "variances length");
        assert_eq!(res.n_components, k);
        assert_eq!(res.n_features, d);
    }

    // ── 7. variances are positive ─────────────────────────────────────────────

    #[test]
    fn variances_positive() {
        let n = 20;
        let d = 5;
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            data[i * d] = i as f64 - 9.5; // strong first column
            for j in 1..d {
                data[i * d + j] = ((i * j) as f64 * 0.2).cos() * 0.1;
            }
        }
        let cfg = SparsePcaConfig {
            n_components: 2,
            l1_bound: 1.0,
            max_iter: 300,
            tol: 1e-7,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        for (k, &var) in res.variances.iter().enumerate() {
            assert!(var > 0.0, "variance[{k}] = {var} is not positive");
        }
    }

    // ── 8. deflation gives approximately orthogonal loadings ─────────────────

    #[test]
    fn deflation_gives_orthogonal_loadings() {
        let n = 40;
        let d = 6;
        // Two independent directions of variance
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            let t = i as f64 - (n as f64 - 1.0) / 2.0;
            data[i * d] = 3.0 * t; // direction 0
            data[i * d + 1] = 2.0 * t; // direction 1 (correlated)
            data[i * d + 2] = -1.5 * t;
            // remaining columns small noise
            for j in 3..d {
                data[i * d + j] = ((i * j + 17) as f64 * 0.23).sin() * 0.1;
            }
        }
        let cfg = SparsePcaConfig {
            n_components: 2,
            l1_bound: 2.0, // loose bound to allow multi-feature loading
            max_iter: 500,
            tol: 1e-9,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        // u_1 and u_2 should be approximately orthogonal after deflation
        let u1 = &res.loadings[..n];
        let u2 = &res.loadings[n..2 * n];
        let dot: f64 = u1.iter().zip(u2).map(|(&a, &b)| a * b).sum();
        assert!(
            dot.abs() < 0.3,
            "u1 · u2 = {dot:.4}, expected approximately 0 (orthogonality from deflation)"
        );
    }

    // ── 9. empty input returns error ──────────────────────────────────────────

    #[test]
    fn empty_input_returns_error() {
        let cfg = SparsePcaConfig::default();
        // n_samples = 0
        let err = sparse_pca(&[], 0, 5, &cfg);
        assert!(
            matches!(err, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            err
        );
        // n_features = 0
        let err2 = sparse_pca(&[], 5, 0, &cfg);
        assert!(
            matches!(err2, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            err2
        );
    }

    // ── 10. l1_bound ≤ 0 returns InvalidParameter ────────────────────────────

    #[test]
    fn l1_bound_zero_returns_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg_zero = SparsePcaConfig {
            n_components: 1,
            l1_bound: 0.0,
            ..Default::default()
        };
        let err = sparse_pca(&data, 2, 3, &cfg_zero);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { ref name, .. }) if name == "l1_bound"),
            "expected InvalidParameter for l1_bound=0, got {:?}",
            err
        );
        let cfg_neg = SparsePcaConfig {
            n_components: 1,
            l1_bound: -1.0,
            ..Default::default()
        };
        let err2 = sparse_pca(&data, 2, 3, &cfg_neg);
        assert!(
            matches!(err2, Err(ManifoldError::InvalidParameter { ref name, .. }) if name == "l1_bound"),
            "expected InvalidParameter for l1_bound=-1, got {:?}",
            err2
        );
    }

    // ── 11. shape mismatch is caught ─────────────────────────────────────────

    #[test]
    fn shape_mismatch_error() {
        let cfg = SparsePcaConfig {
            n_components: 1,
            l1_bound: 1.0,
            ..Default::default()
        };
        // supply 5 elements but claim n_samples=3, n_features=3 → expect 9
        let err = sparse_pca(&[1.0; 5], 3, 3, &cfg);
        assert!(
            matches!(err, Err(ManifoldError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {:?}",
            err
        );
    }

    // ── 12. single sample / single feature edge cases ────────────────────────

    #[test]
    fn single_feature_single_component() {
        // 1 feature: v must be ±1 (only possibility with ||v||₂ = 1)
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let cfg = SparsePcaConfig {
            n_components: 1,
            l1_bound: 1.5,
            ..Default::default()
        };
        let res = sparse_pca(&data, 5, 1, &cfg).expect("ok");
        assert_eq!(res.components.len(), 1);
        assert!((res.components[0].abs() - 1.0).abs() < 1e-10);
    }

    // ── 13. criterion monotonically non-increasing across components ──────────

    #[test]
    fn variances_non_increasing_per_deflation() {
        // The deflation peels off variance: first component criterion ≥ second.
        let n = 30;
        let d = 10;
        let mut data = vec![0.0_f64; n * d];
        for i in 0..n {
            let t = i as f64 - (n as f64 - 1.0) / 2.0;
            data[i * d] = 4.0 * t;
            data[i * d + 1] = 2.0 * t;
            for j in 2..d {
                data[i * d + j] = ((i + j * 5) as f64 * 0.17).sin() * 0.2;
            }
        }
        let cfg = SparsePcaConfig {
            n_components: 3,
            l1_bound: 2.0,
            max_iter: 400,
            tol: 1e-8,
        };
        let res = sparse_pca(&data, n, d, &cfg).expect("ok");
        assert!(
            res.variances[0] >= res.variances[1] - 1e-6,
            "variance[0]={} < variance[1]={}",
            res.variances[0],
            res.variances[1]
        );
    }
}
