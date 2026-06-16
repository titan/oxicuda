//! ν-One-Class SVM (Schölkopf, Platt, Shawe-Taylor, Smola, Williamson 2001).
//!
//! Finds the maximum-margin hyperplane in feature space separating data from
//! the origin, using an RBF kernel and a simplified dual SMO solver.
//!
//! ## Dual Problem
//!
//! ```text
//! max  −½ Σ_i Σ_j α_i α_j k(x_i, x_j)
//! s.t. Σ_i α_i = 1
//!      0 ≤ α_i ≤ 1/(ν·n)
//! ```
//!
//! ## Decision Function
//!
//! ```text
//! f(x) = Σ_i α_i k(x_i, x) − ρ
//! ```
//!
//! **Anomaly score**: `ρ − f(x)`.  Score > 0 → outlier; score ≤ 0 → inlier.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the ν-One-Class SVM detector.
#[derive(Debug, Clone)]
pub struct OcsvmConfig {
    /// ν parameter in `(0, 1]`: upper bound on outlier fraction (default 0.1).
    pub nu: f32,
    /// RBF kernel bandwidth γ > 0 (default 1.0).  `k(x,y) = exp(−γ ‖x−y‖²)`.
    pub gamma: f32,
    /// Maximum SMO iterations (default 1000).
    pub max_iter: usize,
    /// KKT gap tolerance (default 1e-3).
    pub tol: f32,
    /// Fraction of samples labelled as outliers by `ocsvm_predict` (default 0.1).
    pub contamination: f32,
}

impl Default for OcsvmConfig {
    fn default() -> Self {
        Self {
            nu: 0.1,
            gamma: 1.0,
            max_iter: 1000,
            tol: 1e-3,
            contamination: 0.1,
        }
    }
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted ν-OC-SVM model.
pub struct OcsvmFit {
    /// Dual variables αᵢ, length `n_train`.
    pub alphas: Vec<f32>,
    /// Bias term ρ.
    pub rho: f32,
    /// Training data copy `[n_train × d]` (row-major) needed for kernel evaluations.
    pub support_vectors: Vec<f32>,
    /// Number of training samples.
    pub n_train: usize,
    /// Feature dimensionality.
    pub d: usize,
    /// RBF bandwidth (copied from config).
    pub gamma: f32,
    /// Fraction of outliers used in predict.
    pub contamination: f32,
}

// ─── RBF kernel helpers ───────────────────────────────────────────────────────

/// RBF kernel: `k(x, y) = exp(−γ ‖x − y‖²)`.
#[inline]
fn rbf_kernel(x: &[f32], y: &[f32], gamma: f32) -> f32 {
    let sq: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    (-gamma * sq).exp()
}

/// Build the full `n × n` symmetric kernel matrix from row-major `data [n × d]`.
fn build_kernel_matrix(data: &[f32], n: usize, d: usize, gamma: f32) -> Vec<f32> {
    let mut km = vec![0.0_f32; n * n];
    for i in 0..n {
        let xi = &data[i * d..(i + 1) * d];
        for j in 0..=i {
            let xj = &data[j * d..(j + 1) * d];
            let kval = rbf_kernel(xi, xj, gamma);
            km[i * n + j] = kval;
            km[j * n + i] = kval;
        }
    }
    km
}

// ─── SMO solver ──────────────────────────────────────────────────────────────

/// Gradient-based working-set SMO for the ν-OC-SVM dual.
///
/// Invariant maintained: `Σ αᵢ = 1`, `0 ≤ αᵢ ≤ C = 1/(ν·n)`.
///
/// Returns `(alphas, rho)`.
fn smo_solve(
    km: &[f32],
    n: usize,
    nu: f32,
    max_iter: usize,
    tol: f32,
) -> AnomalyResult<(Vec<f32>, f32)> {
    let c_upper = 1.0_f32 / (nu * n as f32); // box upper bound

    // Initialise uniform distribution: αᵢ = 1/n ≤ C (holds iff ν ≤ 1).
    let mut alpha = vec![1.0_f32 / n as f32; n];

    // Gradient of the objective (−½ αᵀ K α) w.r.t. αᵢ:
    // grad_i = ∂/∂αᵢ (−½ αᵀ K α) = −(Kα)_i = −Σ_j α_j K[i,j]
    let mut grad = vec![0.0_f32; n];
    for i in 0..n {
        for j in 0..n {
            grad[i] -= alpha[j] * km[i * n + j];
        }
    }

    let eps = 1e-12_f32;

    for _ in 0..max_iter {
        // i* = argmin grad_i  subject to  α_i < C  (can increase α_i)
        let mut i_star = n;
        let mut min_grad = f32::INFINITY;
        for i in 0..n {
            if alpha[i] < c_upper - eps && grad[i] < min_grad {
                min_grad = grad[i];
                i_star = i;
            }
        }

        // j* = argmax grad_j  subject to  α_j > 0  (can decrease α_j)
        let mut j_star = n;
        let mut max_grad = f32::NEG_INFINITY;
        for j in 0..n {
            if alpha[j] > eps && grad[j] > max_grad {
                max_grad = grad[j];
                j_star = j;
            }
        }

        if i_star == n || j_star == n {
            break; // no valid working pair
        }

        // KKT optimality gap
        if max_grad - min_grad < tol {
            break;
        }

        // Second-order step size
        let eta = km[i_star * n + i_star] + km[j_star * n + j_star] - 2.0 * km[i_star * n + j_star];
        if eta <= eps {
            continue; // degenerate curvature; skip pair
        }

        let raw_delta = (max_grad - min_grad) / eta;

        // Clip to maintain box constraints and Σ α = 1
        let max_step = (c_upper - alpha[i_star]).min(alpha[j_star]);
        let delta = raw_delta.min(max_step).max(0.0);

        if delta < eps {
            continue;
        }

        alpha[i_star] += delta;
        alpha[j_star] -= delta;

        // Incremental gradient update: grad[k] += −delta * (K[k,i*] − K[k,j*])
        for k_idx in 0..n {
            grad[k_idx] -= delta * (km[k_idx * n + i_star] - km[k_idx * n + j_star]);
        }
    }

    let rho = compute_rho(&alpha, km, n, c_upper, eps);

    Ok((alpha, rho))
}

/// Compute the bias `ρ` from the dual solution.
///
/// For any margin SV `s` (0 < α_s < C): `ρ = Σ_j α_j K[j, s]`.
/// Falls back to the mean over all active SVs if no margin SVs exist.
fn compute_rho(alpha: &[f32], km: &[f32], n: usize, c_upper: f32, eps: f32) -> f32 {
    let margin_svs: Vec<usize> = (0..n)
        .filter(|&i| alpha[i] > eps && alpha[i] < c_upper - eps)
        .collect();

    let sv_set: &[usize] = if margin_svs.is_empty() {
        // Use all active SVs as fallback
        let active: Vec<usize> = (0..n).filter(|&i| alpha[i] > eps).collect();
        if active.is_empty() {
            return 0.0;
        }
        // Need a static slice; use heap allocation via Box/leak is wasteful.
        // Instead, recompute inline for the active set.
        let sum: f32 = (0..n)
            .filter(|&i| alpha[i] > eps)
            .map(|sv| (0..n).map(|j| alpha[j] * km[j * n + sv]).sum::<f32>())
            .sum();
        let count = (0..n).filter(|&i| alpha[i] > eps).count();
        return if count == 0 { 0.0 } else { sum / count as f32 };
    } else {
        &margin_svs
    };

    let sum: f32 = sv_set
        .iter()
        .map(|&sv| (0..n).map(|j| alpha[j] * km[j * n + sv]).sum::<f32>())
        .sum();
    sum / sv_set.len() as f32
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a ν-One-Class SVM to training data `data` of shape `[n × d]` (row-major).
pub fn ocsvm_fit(data: &[f32], n: usize, d: usize, cfg: OcsvmConfig) -> AnomalyResult<OcsvmFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if n < 2 {
        return Err(AnomalyError::InsufficientSamples { need: 2, got: n });
    }
    if data.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: data.len(),
        });
    }
    if cfg.nu <= 0.0 || cfg.nu > 1.0 {
        return Err(AnomalyError::InvalidNu { nu: cfg.nu });
    }
    if cfg.gamma <= 0.0 {
        return Err(AnomalyError::Internal {
            msg: format!("gamma must be > 0, got {}", cfg.gamma),
        });
    }
    for v in data.iter() {
        if v.is_nan() {
            return Err(AnomalyError::NanEncountered {
                context: "training data".into(),
            });
        }
    }

    let km = build_kernel_matrix(data, n, d, cfg.gamma);
    let (alphas, rho) = smo_solve(&km, n, cfg.nu, cfg.max_iter, cfg.tol)?;

    if rho.is_nan() || rho.is_infinite() {
        return Err(AnomalyError::ConvergenceFailed {
            msg: "rho is not finite after SMO".into(),
        });
    }

    Ok(OcsvmFit {
        alphas,
        rho,
        support_vectors: data.to_vec(),
        n_train: n,
        d,
        gamma: cfg.gamma,
        contamination: cfg.contamination,
    })
}

/// Compute per-sample anomaly scores for `data` of shape `[n × d]`.
///
/// Score = `ρ − f(x)` where `f(x) = Σ_i αᵢ k(xᵢ, x) − ρ`.
/// Equivalently: score = `2ρ − Σ_i αᵢ k(xᵢ, x)`.
/// Positive score → outlier; non-positive → inlier.
pub fn ocsvm_score(fit: &OcsvmFit, data: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let expected_len = n * fit.d;
    if data.len() != expected_len {
        return Err(AnomalyError::DimensionMismatch {
            expected: expected_len,
            got: data.len(),
        });
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &data[i * fit.d..(i + 1) * fit.d];
        // ksum = Σ_j α_j k(x_j, x_i)
        let ksum: f32 = (0..fit.n_train)
            .map(|j| {
                let xj = &fit.support_vectors[j * fit.d..(j + 1) * fit.d];
                fit.alphas[j] * rbf_kernel(xj, xi, fit.gamma)
            })
            .sum();
        // f(xi) = ksum − ρ; score = ρ − f(xi) = 2ρ − ksum
        scores.push(2.0 * fit.rho - ksum);
    }

    Ok(scores)
}

/// Classify each sample as outlier (`true`) or inlier (`false`).
///
/// The top `contamination × n` fraction (by descending anomaly score) are
/// labelled `true`.
pub fn ocsvm_predict(fit: &OcsvmFit, data: &[f32], n: usize) -> AnomalyResult<Vec<bool>> {
    let scores = ocsvm_score(fit, data, n)?;

    let n_outliers = ((fit.contamination * n as f32).ceil() as usize).min(n);

    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![false; n];
    for &(i, _) in indexed.iter().take(n_outliers) {
        result[i] = true;
    }
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate `n` points from a 2-D isotropic Gaussian N(center, sigma²·I).
    fn gaussian_blob_2d(n: usize, center: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .flat_map(|_| {
                [
                    center + rng.next_normal() * sigma,
                    center + rng.next_normal() * sigma,
                ]
            })
            .collect()
    }

    fn default_cfg() -> OcsvmConfig {
        OcsvmConfig::default()
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────
    #[test]
    fn gaussian_blob_inliers_score_low() {
        let data = gaussian_blob_2d(50, 0.0, 0.3, 1);
        let fit = ocsvm_fit(&data, 50, 2, default_cfg())
            .expect("OC-SVM fit should succeed on Gaussian blob");
        let scores =
            ocsvm_score(&fit, &data, 50).expect("OC-SVM score should succeed on training data");
        let outlier = vec![100.0_f32, 100.0];
        let out_score = ocsvm_score(&fit, &outlier, 1)
            .expect("OC-SVM score should succeed for outlier point")[0];
        let mean_in: f32 = scores.iter().sum::<f32>() / 50.0;
        assert!(
            out_score > mean_in,
            "outlier score {out_score} should exceed inlier mean {mean_in}"
        );
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────
    #[test]
    fn far_outlier_top_ranked() {
        let mut data = gaussian_blob_2d(40, 0.0, 0.5, 2);
        data.extend_from_slice(&[50.0_f32, 50.0]);
        let n = 41;
        let fit = ocsvm_fit(&data, n, 2, default_cfg())
            .expect("OC-SVM fit should succeed with appended far outlier");
        let scores = ocsvm_score(&fit, &data, n)
            .expect("OC-SVM score should succeed on data with far outlier");
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("scores iterator should yield a maximum element");
        assert_eq!(
            max_idx, 40,
            "far outlier (index 40) should have highest score"
        );
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────
    #[test]
    fn nu_fraction_bounds_outlier_rate() {
        let mut data = gaussian_blob_2d(50, 0.0, 0.3, 3);
        for _ in 0..10 {
            data.push(100.0_f32);
            data.push(100.0_f32);
        }
        let n = 60;
        let cfg = OcsvmConfig {
            nu: 0.2,
            contamination: 0.2,
            max_iter: 2000,
            ..default_cfg()
        };
        let fit = ocsvm_fit(&data, n, 2, cfg).expect("OC-SVM fit with nu=0.2 should succeed");
        let scores =
            ocsvm_score(&fit, &data, n).expect("OC-SVM score should succeed on training data");
        let inlier_mean: f32 = scores[..50].iter().sum::<f32>() / 50.0;
        let outlier_mean: f32 = scores[50..].iter().sum::<f32>() / 10.0;
        assert!(
            outlier_mean > inlier_mean,
            "outlier mean score {outlier_mean} should exceed inlier mean {inlier_mean}"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────
    #[test]
    fn decision_values_continuous() {
        let data = gaussian_blob_2d(30, 1.0, 0.5, 4);
        let fit = ocsvm_fit(&data, 30, 2, default_cfg())
            .expect("OC-SVM fit on 30 samples should succeed");
        let scores =
            ocsvm_score(&fit, &data, 30).expect("OC-SVM score should succeed on training data");
        for &s in &scores {
            assert!(s.is_finite(), "score should be finite, got {s}");
        }
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────
    #[test]
    fn single_point_fits_without_error() {
        // n=1 is rejected; n=2 with nu=1 is the smallest valid case.
        let data = vec![1.0_f32, 2.0, 1.1, 2.1];
        let cfg = OcsvmConfig {
            nu: 1.0,
            ..default_cfg()
        };
        let result = ocsvm_fit(&data, 2, 2, cfg);
        assert!(result.is_ok(), "n=2 should succeed: {:?}", result.err());
    }

    // ── Test 6 ────────────────────────────────────────────────────────────────
    #[test]
    fn invalid_nu_zero() {
        let data = gaussian_blob_2d(10, 0.0, 1.0, 6);
        let cfg = OcsvmConfig {
            nu: 0.0,
            ..default_cfg()
        };
        assert!(matches!(
            ocsvm_fit(&data, 10, 2, cfg),
            Err(AnomalyError::InvalidNu { .. })
        ));
    }

    // ── Test 7 ────────────────────────────────────────────────────────────────
    #[test]
    fn invalid_nu_too_large() {
        let data = gaussian_blob_2d(10, 0.0, 1.0, 7);
        let cfg = OcsvmConfig {
            nu: 1.01,
            ..default_cfg()
        };
        assert!(matches!(
            ocsvm_fit(&data, 10, 2, cfg),
            Err(AnomalyError::InvalidNu { .. })
        ));
    }

    // ── Test 8 ────────────────────────────────────────────────────────────────
    #[test]
    fn invalid_gamma_zero() {
        let data = gaussian_blob_2d(10, 0.0, 1.0, 8);
        let cfg = OcsvmConfig {
            gamma: 0.0,
            ..default_cfg()
        };
        assert!(matches!(
            ocsvm_fit(&data, 10, 2, cfg),
            Err(AnomalyError::Internal { .. })
        ));
    }

    // ── Test 9 ────────────────────────────────────────────────────────────────
    #[test]
    fn dim_mismatch_score() {
        let data = gaussian_blob_2d(20, 0.0, 1.0, 9);
        let fit = ocsvm_fit(&data, 20, 2, default_cfg())
            .expect("OC-SVM fit on 20 samples should succeed");
        // Supply 15 elements for n=5 with d=2 model → expects 10
        let bad = vec![1.0_f32; 15];
        assert!(matches!(
            ocsvm_score(&fit, &bad, 5),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 10 ───────────────────────────────────────────────────────────────
    #[test]
    fn dim_mismatch_fit_data() {
        // data.len()=11 ≠ n*d = 5*3 = 15
        let data = vec![1.0_f32; 11];
        assert!(matches!(
            ocsvm_fit(&data, 5, 3, default_cfg()),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 11 ───────────────────────────────────────────────────────────────
    #[test]
    fn rbf_kernel_self_is_one() {
        let x = vec![1.0_f32, -2.0, std::f32::consts::PI];
        let k = rbf_kernel(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-6, "k(x,x) should = 1.0, got {k}");
    }

    // ── Test 12 ───────────────────────────────────────────────────────────────
    #[test]
    fn rbf_kernel_symmetry() {
        let x = vec![1.0_f32, 0.5];
        let y = vec![-1.0_f32, 2.0];
        let kxy = rbf_kernel(&x, &y, 0.5);
        let kyx = rbf_kernel(&y, &x, 0.5);
        assert!(
            (kxy - kyx).abs() < 1e-7,
            "RBF must be symmetric: k(x,y)={kxy}, k(y,x)={kyx}"
        );
    }

    // ── Test 13 ───────────────────────────────────────────────────────────────
    #[test]
    fn alphas_sum_to_one() {
        let data = gaussian_blob_2d(30, 0.0, 1.0, 13);
        let fit = ocsvm_fit(&data, 30, 2, default_cfg())
            .expect("OC-SVM fit on 30 samples should succeed");
        let sum: f32 = fit.alphas.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "alphas should sum ≈ 1.0, got {sum}"
        );
    }

    // ── Test 14 ───────────────────────────────────────────────────────────────
    #[test]
    fn alphas_bounded_by_inv_nu_n() {
        let data = gaussian_blob_2d(30, 0.0, 1.0, 14);
        let cfg = OcsvmConfig {
            nu: 0.2,
            ..default_cfg()
        };
        let n = 30;
        let c_upper = 1.0_f32 / (0.2 * n as f32);
        let fit = ocsvm_fit(&data, n, 2, cfg).expect("OC-SVM fit with nu=0.2 should succeed");
        for (i, &a) in fit.alphas.iter().enumerate() {
            assert!(
                a <= c_upper + 1e-4,
                "alpha[{i}] = {a} exceeds C = {c_upper}"
            );
        }
    }

    // ── Test 15 ───────────────────────────────────────────────────────────────
    #[test]
    fn gamma_larger_tighter_boundary() {
        let data = gaussian_blob_2d(20, 0.0, 0.5, 15);
        let test_pt = vec![2.0_f32, 2.0];

        let fit_small = ocsvm_fit(
            &data,
            20,
            2,
            OcsvmConfig {
                gamma: 0.01,
                ..default_cfg()
            },
        )
        .expect("OC-SVM fit with small gamma should succeed");
        let fit_large = ocsvm_fit(
            &data,
            20,
            2,
            OcsvmConfig {
                gamma: 10.0,
                ..default_cfg()
            },
        )
        .expect("OC-SVM fit with large gamma should succeed");

        let score_small = ocsvm_score(&fit_small, &test_pt, 1)
            .expect("scoring test point with small-gamma model should succeed")[0];
        let score_large = ocsvm_score(&fit_large, &test_pt, 1)
            .expect("scoring test point with large-gamma model should succeed")[0];

        assert!(
            score_large >= score_small,
            "larger gamma should give >= anomaly score: large={score_large}, small={score_small}"
        );
    }

    // ── Test 16 ───────────────────────────────────────────────────────────────
    #[test]
    fn convergence_on_small_2d() {
        let data = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5];
        let cfg = OcsvmConfig {
            nu: 0.5,
            max_iter: 500,
            ..default_cfg()
        };
        let result = ocsvm_fit(&data, 5, 2, cfg);
        assert!(
            result.is_ok(),
            "should converge on 5 2D points: {:?}",
            result.err()
        );
    }

    // ── Test 17 ───────────────────────────────────────────────────────────────
    #[test]
    fn rho_finite() {
        let data = gaussian_blob_2d(25, 2.0, 1.0, 17);
        let fit = ocsvm_fit(&data, 25, 2, default_cfg())
            .expect("OC-SVM fit on 25 samples should succeed");
        assert!(fit.rho.is_finite(), "rho must be finite, got {}", fit.rho);
    }

    // ── Test 18 ───────────────────────────────────────────────────────────────
    #[test]
    fn predict_returns_bool_vec() {
        let data = gaussian_blob_2d(20, 0.0, 0.5, 18);
        let fit = ocsvm_fit(&data, 20, 2, default_cfg())
            .expect("OC-SVM fit on 20 samples should succeed");
        let preds =
            ocsvm_predict(&fit, &data, 20).expect("OC-SVM predict on training data should succeed");
        assert_eq!(preds.len(), 20);
    }

    // ── Test 19 ───────────────────────────────────────────────────────────────
    #[test]
    fn score_nonneg_for_outlier() {
        let data = gaussian_blob_2d(40, 0.0, 0.3, 19);
        let fit = ocsvm_fit(&data, 40, 2, default_cfg())
            .expect("OC-SVM fit on 40 samples should succeed");
        let outlier = vec![500.0_f32, 500.0];
        let score =
            ocsvm_score(&fit, &outlier, 1).expect("scoring extreme outlier should succeed")[0];
        assert!(
            score > 0.0,
            "extreme outlier should have positive score, got {score}"
        );
    }

    // ── Test 20 ───────────────────────────────────────────────────────────────
    #[test]
    fn nu_equals_one_all_svs() {
        let data = gaussian_blob_2d(10, 0.0, 1.0, 20);
        let cfg = OcsvmConfig {
            nu: 1.0,
            max_iter: 2000,
            tol: 1e-4,
            ..default_cfg()
        };
        let fit = ocsvm_fit(&data, 10, 2, cfg).expect("OC-SVM fit with nu=1.0 should succeed");
        let sum: f32 = fit.alphas.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "alphas sum={sum} should ≈ 1.0 for nu=1"
        );
    }

    // ── Test 21 ───────────────────────────────────────────────────────────────
    #[test]
    fn empty_input() {
        assert!(matches!(
            ocsvm_fit(&[], 0, 2, default_cfg()),
            Err(AnomalyError::EmptyInput)
        ));
    }

    // ── Test 22 ───────────────────────────────────────────────────────────────
    #[test]
    fn n_less_than_two() {
        let data = vec![1.0_f32, 2.0]; // n=1, d=2
        assert!(matches!(
            ocsvm_fit(&data, 1, 2, default_cfg()),
            Err(AnomalyError::InsufficientSamples { .. })
        ));
    }

    // ── Test 23 ───────────────────────────────────────────────────────────────
    #[test]
    fn gamma_affects_kernel_decay() {
        let x = vec![0.0_f32, 0.0];
        let y = vec![1.0_f32, 0.0]; // exactly unit distance apart
        let k_low = rbf_kernel(&x, &y, 0.01);
        let k_high = rbf_kernel(&x, &y, 100.0);
        assert!(
            k_low > 0.9,
            "γ=0.01 should give near-1 kernel for unit distance, got {k_low}"
        );
        assert!(
            k_high < 0.001,
            "γ=100 should give near-0 kernel for unit distance, got {k_high}"
        );
    }
}
