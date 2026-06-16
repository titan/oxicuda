//! Fast Minimum Covariance Determinant (FastMCD).
//!
//! Rousseeuw & Van Driessen. "A fast algorithm for the minimum covariance
//! determinant estimator". *JASA* 1999.  Original MCD: Rousseeuw 1984.
//!
//! # Key idea
//!
//! Find the subset H of size h = ⌊support_frac · n⌋ + 1 (default h ≈ n/2) whose
//! sample covariance has the **minimum determinant**.  This subset contains only
//! inliers with high probability, so the resulting location and scatter estimates
//! are robust to up to 50% contamination.
//!
//! ## C-step algorithm
//!
//! Each concentration step:
//! 1. Compute robust (t, S) from current subset H.
//! 2. Compute Mahalanobis² distance to (t, S) for ALL n points.
//! 3. Form new H = indices of h smallest Mahal² distances.
//! 4. Convergence iff det(S) does not decrease.
//!
//! Multiple random starts prevent local-minimum trapping.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`fast_mcd_fit`].
#[derive(Debug, Clone)]
pub struct FastMcdConfig {
    /// Fraction of samples to keep; h = floor(support_frac · n) + 1.  Default 0.5.
    pub support_frac: f32,
    /// Number of random starts.  Default 10.
    pub n_starts: usize,
    /// Maximum C-steps per start.  Default 50.
    pub max_iter: usize,
    /// RNG seed for reproducibility.  Default 42.
    pub seed: u64,
    /// Ridge added to the diagonal of the covariance before inversion.  Default 1e-5.
    pub ridge: f32,
}

impl Default for FastMcdConfig {
    fn default() -> Self {
        Self {
            support_frac: 0.5,
            n_starts: 10,
            max_iter: 50,
            seed: 42,
            ridge: 1e-5,
        }
    }
}

// ─── Fit result ───────────────────────────────────────────────────────────────

/// Output of [`fast_mcd_fit`].
#[derive(Debug, Clone)]
pub struct FastMcdFit {
    /// Robust location estimate (mean of the best h-subset).
    pub location: Vec<f32>,
    /// Inverse of the regularised robust covariance (d × d, row-major).
    pub cov_inv: Vec<f32>,
    /// Indices (into the original data) of the best h-subset.
    pub support: Vec<usize>,
    /// Number of features.
    pub n_features: usize,
    /// Number of samples used in fit.
    pub n_samples_fitted: usize,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit FastMCD on `data` (n × d, row-major f32).
///
/// Returns the robust location, inverse covariance, and support subset.
///
/// # Errors
///
/// - [`AnomalyError::InsufficientSamples`] if h ≤ d (need at least d+1 samples in subset).
/// - [`AnomalyError::InvalidFeatureCount`] if d = 0.
/// - [`AnomalyError::DimensionMismatch`] if `data.len() ≠ n × d`.
/// - [`AnomalyError::SingularCovariance`] if no valid covariance can be computed.
pub fn fast_mcd_fit(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    cfg: &FastMcdConfig,
) -> AnomalyResult<FastMcdFit> {
    if n_features == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }

    let h = (cfg.support_frac * n_samples as f32).floor() as usize + 1;
    let h = h.min(n_samples); // safety cap

    if h <= n_features {
        return Err(AnomalyError::InsufficientSamples {
            need: n_features + 2,
            got: h,
        });
    }

    let mut rng = LcgRng::new(cfg.seed);

    let mut best_log_det = f32::INFINITY;
    let mut best_support: Vec<usize> = (0..h).collect();
    let mut best_location = vec![0.0_f32; n_features];
    let mut best_cov_inv = vec![0.0_f32; n_features * n_features];
    let mut found = false;

    for _ in 0..cfg.n_starts {
        // Random h-subset via partial Fisher-Yates shuffle
        let mut idx: Vec<usize> = (0..n_samples).collect();
        for i in 0..h {
            let j = i + rng.next_usize(n_samples - i);
            idx.swap(i, j);
        }
        let mut support: Vec<usize> = idx[..h].to_vec();

        let mut prev_log_det = f32::INFINITY;

        for _ in 0..cfg.max_iter {
            // Compute (location, cov) from support
            let (loc, cov) = compute_location_cov(data, &support, n_features, cfg.ridge);

            // Log-det via Cholesky for convergence check (INFINITY signals singular)
            let log_det = cholesky_log_det(&cov, n_features).unwrap_or(f32::INFINITY);

            if (log_det - prev_log_det).abs() < 1e-10 * (1.0 + prev_log_det.abs()) {
                // Converged
                if log_det < best_log_det
                    && let Some(cov_inv) = gauss_jordan_inv(&cov, n_features)
                {
                    best_log_det = log_det;
                    best_support = support.clone();
                    best_location = loc;
                    best_cov_inv = cov_inv;
                    found = true;
                }
                break;
            }
            prev_log_det = log_det;

            // C-step: recompute Mahal² distances, keep h smallest
            let cov_inv_opt = gauss_jordan_inv(&cov, n_features);
            let cov_inv = match cov_inv_opt {
                Some(ci) => ci,
                None => break, // singular, skip
            };

            let mut mahal2: Vec<(usize, f32)> = (0..n_samples)
                .map(|i| {
                    let row = &data[i * n_features..(i + 1) * n_features];
                    let d2 = mahal_sq(row, &loc, &cov_inv, n_features);
                    (i, d2)
                })
                .collect();
            mahal2.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            support = mahal2[..h].iter().map(|(i, _)| *i).collect();
        }
    }

    if !found {
        return Err(AnomalyError::SingularCovariance);
    }

    Ok(FastMcdFit {
        location: best_location,
        cov_inv: best_cov_inv,
        support: best_support,
        n_features,
        n_samples_fitted: n_samples,
    })
}

/// Compute robust Mahalanobis² score: `(x − location)ᵀ cov_inv (x − location)`.
pub fn fast_mcd_score(fit: &FastMcdFit, x: &[f32]) -> AnomalyResult<f32> {
    if x.len() != fit.n_features {
        return Err(AnomalyError::FeatureCountMismatch {
            expected: fit.n_features,
            got: x.len(),
        });
    }
    let d2 = mahal_sq(x, &fit.location, &fit.cov_inv, fit.n_features);
    Ok(d2.max(0.0))
}

/// Batch Mahalanobis² scoring.  `x` is `n × d` row-major.
pub fn fast_mcd_score_batch(
    fit: &FastMcdFit,
    x: &[f32],
    n_samples: usize,
) -> AnomalyResult<Vec<f32>> {
    if x.len() != n_samples * fit.n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * fit.n_features,
            got: x.len(),
        });
    }
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row = &x[i * fit.n_features..(i + 1) * fit.n_features];
        out.push(fast_mcd_score(fit, row)?);
    }
    Ok(out)
}

/// Binary outlier prediction: flag sample as outlier if `score > threshold`.
pub fn fast_mcd_predict(
    fit: &FastMcdFit,
    x: &[f32],
    n_samples: usize,
    threshold: f32,
) -> AnomalyResult<Vec<bool>> {
    let scores = fast_mcd_score_batch(fit, x, n_samples)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute mean and sample covariance of a subset of rows, plus ridge regularisation.
fn compute_location_cov(
    data: &[f32],
    support: &[usize],
    d: usize,
    ridge: f32,
) -> (Vec<f32>, Vec<f32>) {
    let h = support.len();
    let mut loc = vec![0.0_f32; d];
    for &i in support {
        let row = &data[i * d..(i + 1) * d];
        for (j, &v) in row.iter().enumerate() {
            loc[j] += v;
        }
    }
    let inv_h = 1.0 / h as f32;
    for v in &mut loc {
        *v *= inv_h;
    }

    let mut cov = vec![0.0_f32; d * d];
    for &i in support {
        let row = &data[i * d..(i + 1) * d];
        for r in 0..d {
            let dr = row[r] - loc[r];
            for c in 0..d {
                let dc = row[c] - loc[c];
                cov[r * d + c] += dr * dc;
            }
        }
    }
    let denom = if h > 1 { (h - 1) as f32 } else { 1.0 };
    for v in &mut cov {
        *v /= denom;
    }
    // Ridge regularisation
    for i in 0..d {
        cov[i * d + i] += ridge;
    }
    (loc, cov)
}

/// Compute squared Mahalanobis distance `(x-mu)ᵀ S⁻¹ (x-mu)`.
fn mahal_sq(x: &[f32], mu: &[f32], s_inv: &[f32], d: usize) -> f32 {
    let mut diff = vec![0.0_f32; d];
    for i in 0..d {
        diff[i] = x[i] - mu[i];
    }
    let mut temp = vec![0.0_f32; d];
    for i in 0..d {
        let row = &s_inv[i * d..(i + 1) * d];
        temp[i] = row.iter().zip(diff.iter()).map(|(a, b)| a * b).sum();
    }
    diff.iter()
        .zip(temp.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
}

/// Cholesky factorisation (lower triangular) returning `log det = 2 Σ ln(L_ii)`.
///
/// Returns `None` if the matrix is not positive-definite.
fn cholesky_log_det(m: &[f32], d: usize) -> Option<f32> {
    let mut l = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = m[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None; // not PD
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    let log_det: f32 = (0..d).map(|i| l[i * d + i].ln()).sum::<f32>() * 2.0;
    Some(log_det)
}

/// Gauss-Jordan inversion with partial pivoting.
///
/// Returns `None` if the matrix is singular (pivot < 1e-10).
fn gauss_jordan_inv(m: &[f32], d: usize) -> Option<Vec<f32>> {
    let w = 2 * d;
    let mut aug = vec![0.0_f32; d * w];
    // Fill [m | I]
    for i in 0..d {
        for j in 0..d {
            aug[i * w + j] = m[i * d + j];
        }
        aug[i * w + d + i] = 1.0;
    }

    for col in 0..d {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col * w + col].abs();
        for row in (col + 1)..d {
            let v = aug[row * w + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for k in 0..w {
                aug.swap(col * w + k, max_row * w + k);
            }
        }
        let pivot = aug[col * w + col];
        if pivot.abs() < 1e-10 {
            return None;
        }
        let inv_p = 1.0 / pivot;
        for k in 0..w {
            aug[col * w + k] *= inv_p;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            if factor.abs() < 1e-30 {
                continue;
            }
            for k in 0..w {
                let sub = factor * aug[col * w + k];
                aug[row * w + k] -= sub;
            }
        }
    }

    let mut inv = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            inv[i * d + j] = aug[i * w + d + j];
        }
    }
    Some(inv)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_normal(state: &mut u64) -> f32 {
        // Box-Muller using two LCG samples
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = ((*state >> 33) as f32 / u32::MAX as f32).max(1e-12);
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (*state >> 33) as f32 / u32::MAX as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn gaussian_2d(n: usize, cx: f32, cy: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .flat_map(|_| {
                let x = lcg_normal(&mut state) * sigma + cx;
                let y = lcg_normal(&mut state) * sigma + cy;
                [x, y]
            })
            .collect()
    }

    #[test]
    fn test_fit_and_score_basic() {
        let data = gaussian_2d(30, 0.0, 0.0, 1.0, 1);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 30, 2, &cfg).expect("fast MCD fit should succeed");
        let s = fast_mcd_score(&fit, &[0.0_f32, 0.0]).expect("MCD score at origin should succeed");
        assert!(s.is_finite(), "score={s}");
    }

    #[test]
    fn test_support_size_correct() {
        let data = gaussian_2d(40, 0.0, 0.0, 1.0, 2);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");
        let expected_h = (0.5 * 40.0_f32).floor() as usize + 1; // 21
        assert_eq!(fit.support.len(), expected_h, "support size mismatch");
    }

    #[test]
    fn test_inlier_vs_outlier_scores() {
        // 40 inliers at origin + 5 outliers far away
        let mut data = gaussian_2d(40, 0.0, 0.0, 0.5, 3);
        for _ in 0..5 {
            data.extend_from_slice(&[50.0_f32, 50.0]);
        }
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 45, 2, &cfg).expect("FastMCD fit should succeed");

        let inlier_scores: Vec<f32> = (0..40)
            .map(|i| {
                fast_mcd_score(&fit, &data[i * 2..(i + 1) * 2])
                    .expect("FastMCD score in iterator should succeed")
            })
            .collect();
        let outlier_scores: Vec<f32> = (40..45)
            .map(|i| {
                fast_mcd_score(&fit, &data[i * 2..(i + 1) * 2])
                    .expect("FastMCD score in iterator should succeed")
            })
            .collect();

        let max_inlier = inlier_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_outlier = outlier_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_outlier > max_inlier,
            "min outlier score {min_outlier} must exceed max inlier score {max_inlier}"
        );
    }

    #[test]
    fn test_robust_location_vs_contaminated_mean() {
        // 30 inliers at (0,0), 10 outliers at (100,100)
        let mut data = gaussian_2d(30, 0.0, 0.0, 0.5, 5);
        for _ in 0..10 {
            data.extend_from_slice(&[100.0_f32, 100.0]);
        }
        let n = 40;
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, n, 2, &cfg).expect("FastMCD fit should succeed");

        // Empirical mean is skewed by outliers
        let emp_mean_x: f32 = (0..n).map(|i| data[i * 2]).sum::<f32>() / n as f32;
        let emp_mean_y: f32 = (0..n).map(|i| data[i * 2 + 1]).sum::<f32>() / n as f32;

        // MCD location should be much closer to (0,0)
        let mcd_dist = (fit.location[0].powi(2) + fit.location[1].powi(2)).sqrt();
        let emp_dist = (emp_mean_x.powi(2) + emp_mean_y.powi(2)).sqrt();
        assert!(
            mcd_dist < emp_dist,
            "MCD location dist {mcd_dist:.2} should be closer to origin than empirical {emp_dist:.2}"
        );
    }

    #[test]
    fn test_location_close_to_true_mean_clean_data() {
        let data = gaussian_2d(60, 5.0, 5.0, 1.0, 7);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 60, 2, &cfg).expect("FastMCD fit should succeed");
        // MCD location should be within ~1.5 of (5,5) with high probability
        let dist = ((fit.location[0] - 5.0).powi(2) + (fit.location[1] - 5.0).powi(2)).sqrt();
        assert!(
            dist < 2.0,
            "location too far from true mean: dist={dist:.3}"
        );
    }

    #[test]
    fn test_score_near_zero_for_center() {
        let data = gaussian_2d(40, 0.0, 0.0, 1.0, 8);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");
        // Score of exact location estimate should be ~0
        let s = fast_mcd_score(&fit, &fit.location.clone())
            .expect("FastMCD score at location should succeed");
        assert!(s < 1e-3, "score at location estimate should be ~0, got {s}");
    }

    #[test]
    fn test_score_grows_with_distance() {
        let data = gaussian_2d(40, 0.0, 0.0, 0.3, 9);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");
        // Score at location < score at (2,0) < score at (10,0)
        let s0 = fast_mcd_score(&fit, &fit.location.clone())
            .expect("FastMCD score at location should succeed");
        let s1 = fast_mcd_score(&fit, &[fit.location[0] + 2.0, fit.location[1]])
            .expect("FastMCD score should succeed");
        let s2 = fast_mcd_score(&fit, &[fit.location[0] + 10.0, fit.location[1]])
            .expect("FastMCD score should succeed");
        assert!(
            s0 < s1,
            "score at center({s0:.3}) should be < score at +2 ({s1:.3})"
        );
        assert!(
            s1 < s2,
            "score at +2 ({s1:.3}) should be < score at +10 ({s2:.3})"
        );
    }

    #[test]
    fn test_predict_flags_outliers() {
        let mut data = gaussian_2d(30, 0.0, 0.0, 0.5, 10);
        let n_outliers = 5;
        for _ in 0..n_outliers {
            data.extend_from_slice(&[30.0_f32, 30.0]);
        }
        let n = 35;
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, n, 2, &cfg).expect("FastMCD fit should succeed");

        // Use a threshold that separates outliers
        let inlier_max = (0..30)
            .map(|i| {
                fast_mcd_score(&fit, &data[i * 2..(i + 1) * 2])
                    .expect("FastMCD score in iterator should succeed")
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = inlier_max * 10.0;

        let predictions =
            fast_mcd_predict(&fit, &data, n, threshold).expect("FastMCD predict should succeed");
        let flagged_outliers = predictions[30..].iter().filter(|&&b| b).count();
        assert!(
            flagged_outliers >= 4,
            "should flag at least 4/5 outliers, got {flagged_outliers}"
        );
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let data = gaussian_2d(40, 0.0, 0.0, 1.0, 11);
        let cfg = FastMcdConfig {
            seed: 123,
            ..Default::default()
        };
        let fit1 = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");
        let fit2 = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");
        assert_eq!(
            fit1.location, fit2.location,
            "same seed must give same location"
        );
    }

    #[test]
    fn test_dimension_mismatch_score() {
        let data = gaussian_2d(20, 0.0, 0.0, 1.0, 12);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 20, 2, &cfg).expect("FastMCD fit should succeed");
        let result = fast_mcd_score(&fit, &[0.0_f32, 0.0, 0.0]); // 3 features, expected 2
        assert!(matches!(
            result,
            Err(AnomalyError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn test_insufficient_samples() {
        // d=5, n=3 → h=2 ≤ d=5 → error
        let data = vec![0.0_f32; 3 * 5];
        let cfg = FastMcdConfig::default();
        let result = fast_mcd_fit(&data, 3, 5, &cfg);
        assert!(matches!(
            result,
            Err(AnomalyError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_high_dimensional() {
        // d=5, n=40 — should work
        let mut state = 77u64;
        let data: Vec<f32> = (0..40 * 5).map(|_| lcg_normal(&mut state)).collect();
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 40, 5, &cfg).expect("FastMCD fit should succeed");
        assert_eq!(fit.n_features, 5);
        assert!(fit.location.len() == 5);
    }

    #[test]
    fn test_single_feature() {
        // d=1, n=20 — 1D case
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 20, 1, &cfg).expect("FastMCD fit should succeed");
        // Score at location estimate ≈ 0
        let s = fast_mcd_score(&fit, &fit.location.clone())
            .expect("FastMCD score at location should succeed");
        assert!(s < 1e-2, "1D score at location={s:.4}");
    }

    #[test]
    fn test_support_has_smaller_mahal_avg() {
        let data = gaussian_2d(40, 0.0, 0.0, 1.0, 14);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 40, 2, &cfg).expect("FastMCD fit should succeed");

        let support_set: std::collections::HashSet<usize> = fit.support.iter().cloned().collect();
        let support_avg: f32 = fit
            .support
            .iter()
            .map(|&i| {
                fast_mcd_score(&fit, &data[i * 2..(i + 1) * 2])
                    .expect("FastMCD score in iterator should succeed")
            })
            .sum::<f32>()
            / fit.support.len() as f32;
        let nonsupport: Vec<usize> = (0..40).filter(|i| !support_set.contains(i)).collect();
        if nonsupport.is_empty() {
            return; // all points in support (edge case)
        }
        let nonsupport_avg: f32 = nonsupport
            .iter()
            .map(|&i| {
                fast_mcd_score(&fit, &data[i * 2..(i + 1) * 2])
                    .expect("FastMCD score in iterator should succeed")
            })
            .sum::<f32>()
            / nonsupport.len() as f32;
        assert!(
            support_avg <= nonsupport_avg,
            "support avg Mahal ({support_avg:.3}) should be ≤ non-support avg ({nonsupport_avg:.3})"
        );
    }

    #[test]
    fn test_n_starts_one() {
        let data = gaussian_2d(30, 0.0, 0.0, 1.0, 15);
        let cfg = FastMcdConfig {
            n_starts: 1,
            ..Default::default()
        };
        let fit = fast_mcd_fit(&data, 30, 2, &cfg).expect("FastMCD fit should succeed");
        assert_eq!(fit.n_features, 2);
    }

    #[test]
    fn test_score_batch_consistent() {
        let data = gaussian_2d(30, 0.0, 0.0, 1.0, 16);
        let cfg = FastMcdConfig::default();
        let fit = fast_mcd_fit(&data, 30, 2, &cfg).expect("FastMCD fit should succeed");
        let queries = vec![0.0_f32, 0.0, 1.0, 1.0, 5.0, 5.0];
        let batch =
            fast_mcd_score_batch(&fit, &queries, 3).expect("FastMCD batch score should succeed");
        for i in 0..3 {
            let single = fast_mcd_score(&fit, &queries[i * 2..(i + 1) * 2])
                .expect("FastMCD single score should match batch score");
            assert!(
                (batch[i] - single).abs() < 1e-6,
                "batch[{i}]={} single={}",
                batch[i],
                single
            );
        }
    }
}
