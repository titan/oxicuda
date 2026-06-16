//! Survey design variance estimators: stratified, clustered, weighted, jackknife.
//!
//! Implements Cochran (1977) stratified design, Taylor-linearization cluster variance,
//! design effects, and jackknife variance for complex surveys.

use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Simple weighted estimators
// ---------------------------------------------------------------------------

/// Compute the weighted mean: μ_w = Σ w_i x_i / Σ w_i.
///
/// Returns `EmptyInput` if slices are empty, `DimensionMismatch` if lengths differ,
/// and `NumericalInstability` if the total weight is zero or non-finite.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> StatsResult<f64> {
    if values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if values.len() != weights.len() {
        return Err(StatsError::DimensionMismatch {
            a: values.len(),
            b: weights.len(),
        });
    }
    let sum_w: f64 = weights.iter().sum();
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "weighted_mean: non-positive or non-finite total weight".into(),
        ));
    }
    let sum_wx: f64 = values.iter().zip(weights).map(|(x, w)| x * w).sum();
    if !sum_wx.is_finite() {
        return Err(StatsError::NumericalInstability(
            "weighted_mean: non-finite weighted sum".into(),
        ));
    }
    Ok(sum_wx / sum_w)
}

/// Bessel-corrected weighted variance.
///
/// Uses the reliability-weights formula:
/// V = (Σ w_i (x_i - μ_w)²) / (Σw_i - Σw_i²/Σw_i)
///
/// The denominator `Σw_i - Σw_i²/Σw_i` is the "effective df" correction for
/// reliability (frequency-scaled) weights; for unit weights it reduces to n-1.
pub fn weighted_variance(values: &[f64], weights: &[f64], mean: f64) -> StatsResult<f64> {
    if values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if values.len() != weights.len() {
        return Err(StatsError::DimensionMismatch {
            a: values.len(),
            b: weights.len(),
        });
    }
    let sum_w: f64 = weights.iter().sum();
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "weighted_variance: non-positive total weight".into(),
        ));
    }
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let denom = sum_w - sum_w2 / sum_w;
    if denom <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "weighted_variance: degenerate weight distribution (denom ≤ 0)".into(),
        ));
    }
    let numer: f64 = values
        .iter()
        .zip(weights)
        .map(|(x, w)| w * (x - mean).powi(2))
        .sum();
    Ok(numer / denom)
}

// ---------------------------------------------------------------------------
// Stratified design
// ---------------------------------------------------------------------------

/// Result of a stratified variance estimation (Cochran 1977).
#[derive(Debug, Clone)]
pub struct StratifiedResult {
    /// Weighted combined mean across strata.
    pub estimate: f64,
    /// Variance of the combined mean under stratified design.
    pub variance: f64,
    /// Standard error of the estimate.
    pub se: f64,
    /// Satterthwaite approximate degrees of freedom.
    pub df: f64,
    /// 95 % confidence interval (t-based with Satterthwaite df).
    pub ci_95: (f64, f64),
}

/// Stratified design variance estimator (Cochran 1977, §5.3).
///
/// # Arguments
/// * `values`   — observed values (length n)
/// * `weights`  — survey weights (length n)
/// * `strata`   — stratum labels in `0..n_strata` (length n)
/// * `n_strata` — number of strata
///
/// # Method
/// Within each stratum h, the weighted mean μ_h and variance V_h are computed.
/// The combined estimate is μ = Σ_h W_h μ_h where W_h = n_h/n (proportion).
/// The variance V(μ) = Σ_h W_h² V_h / n_h uses Satterthwaite df for the CI.
pub fn stratified_variance(
    values: &[f64],
    weights: &[f64],
    strata: &[usize],
    n_strata: usize,
) -> StatsResult<StratifiedResult> {
    let n = values.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if weights.len() != n || strata.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: weights.len().max(strata.len()),
        });
    }
    if n_strata == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_strata".into(),
            reason: "must be at least 1".into(),
        });
    }
    // Validate stratum indices
    for &s in strata {
        if s >= n_strata {
            return Err(StatsError::IndexOutOfBounds {
                index: s,
                len: n_strata,
            });
        }
    }

    // --- collect per-stratum data ---
    let mut stratum_vals: Vec<Vec<f64>> = vec![Vec::new(); n_strata];
    let mut stratum_wts: Vec<Vec<f64>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        stratum_vals[strata[i]].push(values[i]);
        stratum_wts[strata[i]].push(weights[i]);
    }

    // Per-stratum: n_h, μ_h, V_h
    let mut stratum_n = vec![0usize; n_strata];
    let mut stratum_mean = vec![0.0f64; n_strata];
    let mut stratum_var = vec![0.0f64; n_strata];

    for h in 0..n_strata {
        let vals = &stratum_vals[h];
        let wts = &stratum_wts[h];
        let nh = vals.len();
        if nh == 0 {
            // empty stratum contributes nothing
            continue;
        }
        stratum_n[h] = nh;
        let mu_h = weighted_mean(vals, wts)?;
        stratum_mean[h] = mu_h;
        if nh >= 2 {
            stratum_var[h] = weighted_variance(vals, wts, mu_h)?;
        }
        // nh == 1: variance contribution is 0 (single obs)
    }

    // Combined estimate: W_h = n_h / n (proportional size)
    let n_f = n as f64;
    let estimate: f64 = (0..n_strata)
        .map(|h| (stratum_n[h] as f64 / n_f) * stratum_mean[h])
        .sum();

    // Variance contributions per stratum: c_h = W_h^2 * V_h / n_h
    let mut total_var = 0.0f64;
    let mut satterthwaite_denom = 0.0f64; // Σ c_h^2 / (n_h - 1)

    for h in 0..n_strata {
        let nh = stratum_n[h];
        if nh == 0 {
            continue;
        }
        let w_h = nh as f64 / n_f;
        let c_h = w_h * w_h * stratum_var[h] / (nh as f64);
        total_var += c_h;
        if nh >= 2 {
            satterthwaite_denom += c_h * c_h / ((nh - 1) as f64);
        }
    }
    // Satterthwaite df = (Σ c_h)^2 / Σ (c_h^2 / (n_h - 1))
    let satterthwaite_numer_sq = total_var * total_var;

    let df = if satterthwaite_denom > 0.0 {
        satterthwaite_numer_sq / satterthwaite_denom
    } else {
        1.0
    };
    let df = df.max(1.0);

    let se = total_var.sqrt();
    let t_crit = StudentT::new(df)?.ppf(0.975)?;
    let ci_95 = (estimate - t_crit * se, estimate + t_crit * se);

    Ok(StratifiedResult {
        estimate,
        variance: total_var,
        se,
        df,
        ci_95,
    })
}

// ---------------------------------------------------------------------------
// Cluster design (Taylor linearization)
// ---------------------------------------------------------------------------

/// Result of a cluster-based variance estimation.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Overall ratio estimate μ = Σ t_c / Σ m_c.
    pub estimate: f64,
    /// Taylor-linearization variance of the estimate.
    pub variance: f64,
    /// Standard error.
    pub se: f64,
    /// Number of clusters used in the estimation.
    pub n_clusters: usize,
}

/// Cluster-based variance estimator using Taylor (ratio) linearization.
///
/// # Arguments
/// * `values`     — observed values (length n)
/// * `weights`    — survey weights (length n)
/// * `clusters`   — cluster labels in `0..n_clusters` (length n)
/// * `n_clusters` — total number of clusters
///
/// # Method
/// Cluster totals t_c = Σ_{i∈c} w_i x_i and cluster masses m_c = Σ_{i∈c} w_i.
/// Overall estimate: μ = Σ t_c / Σ m_c (ratio estimator).
/// Variance (Taylor linearisation):
/// V = (1/M²) × (n_c/(n_c-1)) × Σ_c (t_c - μ m_c)²
/// where M = Σ m_c.
pub fn cluster_variance(
    values: &[f64],
    weights: &[f64],
    clusters: &[usize],
    n_clusters: usize,
) -> StatsResult<ClusterResult> {
    let n = values.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if weights.len() != n || clusters.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: weights.len().max(clusters.len()),
        });
    }
    if n_clusters < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_clusters,
            need: 2,
        });
    }
    for &c in clusters {
        if c >= n_clusters {
            return Err(StatsError::IndexOutOfBounds {
                index: c,
                len: n_clusters,
            });
        }
    }

    // Cluster totals t_c and masses m_c
    let mut t_c = vec![0.0f64; n_clusters];
    let mut m_c = vec![0.0f64; n_clusters];
    for i in 0..n {
        let c = clusters[i];
        t_c[c] += weights[i] * values[i];
        m_c[c] += weights[i];
    }

    let m_total: f64 = m_c.iter().sum();
    if m_total <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "cluster_variance: total mass is non-positive".into(),
        ));
    }
    let t_total: f64 = t_c.iter().sum();
    let estimate = t_total / m_total;

    // Non-empty clusters only
    let active_clusters: Vec<usize> = (0..n_clusters).filter(|&c| m_c[c] > 0.0).collect();
    let k = active_clusters.len();
    if k < 2 {
        return Err(StatsError::InsufficientSampleSize { got: k, need: 2 });
    }

    // Taylor linearization residuals: r_c = t_c - estimate * m_c
    let ss: f64 = active_clusters
        .iter()
        .map(|&c| {
            let r = t_c[c] - estimate * m_c[c];
            r * r
        })
        .sum();

    // Variance: (1/M^2) * (k/(k-1)) * Σ r_c^2
    let variance = ss * (k as f64) / ((k as f64 - 1.0) * m_total * m_total);
    let se = variance.sqrt();

    Ok(ClusterResult {
        estimate,
        variance,
        se,
        n_clusters: k,
    })
}

// ---------------------------------------------------------------------------
// Design effect
// ---------------------------------------------------------------------------

/// Compute the design effect (DEFF): ratio of complex design variance to SRS variance.
///
/// DEFF = complex_variance / srs_variance.
/// A value > 1 indicates efficiency loss relative to simple random sampling.
///
/// Returns 1.0 if both inputs are zero (degenerate case).
#[must_use]
pub fn design_effect(srs_variance: f64, complex_variance: f64) -> f64 {
    if srs_variance == 0.0 {
        if complex_variance == 0.0 {
            return 1.0;
        }
        return f64::INFINITY;
    }
    complex_variance / srs_variance
}

// ---------------------------------------------------------------------------
// Jackknife variance for complex surveys
// ---------------------------------------------------------------------------

/// Jackknife variance estimator for complex surveys (cluster-drop jackknife).
///
/// Drops one cluster at a time, recomputes the statistic on the remaining data,
/// and accumulates the standard jackknife variance with the (n-1)/n factor.
///
/// V_JK = ((n-1)/n) Σ_c (θ̂_{-c} - θ̂)²
/// where θ̂_{-c} is the statistic with cluster c dropped.
pub fn jackknife_survey_variance(
    values: &[f64],
    weights: &[f64],
    clusters: &[usize],
    n_clusters: usize,
    statistic: impl Fn(&[f64], &[f64]) -> f64,
) -> StatsResult<f64> {
    let n = values.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if weights.len() != n || clusters.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: weights.len().max(clusters.len()),
        });
    }
    if n_clusters < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_clusters,
            need: 2,
        });
    }

    // Identify which observations belong to each cluster
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for i in 0..n {
        if clusters[i] >= n_clusters {
            return Err(StatsError::IndexOutOfBounds {
                index: clusters[i],
                len: n_clusters,
            });
        }
        cluster_indices[clusters[i]].push(i);
    }

    // Overall statistic on the full sample
    let theta_hat = statistic(values, weights);

    // Build once: complement masks (indices not in cluster c)
    let mut jk_vals = Vec::with_capacity(n);
    let mut jk_wts = Vec::with_capacity(n);

    let active_clusters: Vec<usize> = (0..n_clusters)
        .filter(|&c| !cluster_indices[c].is_empty())
        .collect();
    let k = active_clusters.len() as f64;
    if k < 2.0 {
        return Err(StatsError::InsufficientSampleSize {
            got: active_clusters.len(),
            need: 2,
        });
    }

    // Build a fast membership mask
    let mut in_cluster = vec![0usize; n]; // which cluster owns each observation
    in_cluster[..n].copy_from_slice(&clusters[..n]);

    let mut sum_sq = 0.0f64;
    for &dropped in &active_clusters {
        jk_vals.clear();
        jk_wts.clear();
        for i in 0..n {
            if in_cluster[i] != dropped {
                jk_vals.push(values[i]);
                jk_wts.push(weights[i]);
            }
        }
        if jk_vals.is_empty() {
            continue;
        }
        let theta_c = statistic(&jk_vals, &jk_wts);
        let diff = theta_c - theta_hat;
        sum_sq += diff * diff;
    }

    // Jackknife variance: (k-1)/k * Σ (θ̂_{-c} - θ̂)²
    let variance = ((k - 1.0) / k) * sum_sq;
    Ok(variance)
}

// ---------------------------------------------------------------------------
// SurveyDesign descriptor struct
// ---------------------------------------------------------------------------

/// Descriptor for a complex survey design.
///
/// Used for documentation and dispatch; the actual computations use
/// free functions that accept slices directly.
#[derive(Debug, Clone)]
pub struct SurveyDesign {
    /// Sampling weights for each observation.
    pub weights: Vec<f64>,
    /// Stratum label for each observation (values in `0..n_strata`).
    pub strata: Vec<usize>,
    /// Optional cluster label for each observation.
    pub clusters: Option<Vec<usize>>,
    /// Number of strata.
    pub n_strata: usize,
}

impl SurveyDesign {
    /// Create a new survey design descriptor.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `weights` and `strata` differ in length.
    pub fn new(
        weights: Vec<f64>,
        strata: Vec<usize>,
        clusters: Option<Vec<usize>>,
        n_strata: usize,
    ) -> StatsResult<Self> {
        if weights.len() != strata.len() {
            return Err(StatsError::DimensionMismatch {
                a: weights.len(),
                b: strata.len(),
            });
        }
        if let Some(ref cls) = clusters {
            if cls.len() != weights.len() {
                return Err(StatsError::DimensionMismatch {
                    a: weights.len(),
                    b: cls.len(),
                });
            }
        }
        Ok(Self {
            weights,
            strata,
            clusters,
            n_strata,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- weighted_mean ----

    #[test]
    fn weighted_mean_uniform_weights() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let wts = [1.0; 5];
        let mu = weighted_mean(&vals, &wts).expect("weighted_mean should succeed");
        assert!((mu - 3.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_non_uniform() {
        // Heavy weight on value 10.0 — mean should be near 10
        let vals = [1.0, 10.0];
        let wts = [1.0, 9.0];
        let mu = weighted_mean(&vals, &wts).expect("weighted_mean should succeed");
        let expected = (1.0 * 1.0 + 10.0 * 9.0) / 10.0;
        assert!((mu - expected).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_empty_error() {
        assert!(matches!(
            weighted_mean(&[], &[]),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn weighted_mean_dimension_mismatch() {
        assert!(matches!(
            weighted_mean(&[1.0, 2.0], &[1.0]),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    // ---- weighted_variance ----

    #[test]
    fn weighted_variance_unit_weights_matches_unbiased() {
        // With unit weights, the Bessel-corrected formula should give (n-1) denominator
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let wts = [1.0; 8];
        let mu = weighted_mean(&vals, &wts).expect("weighted_mean should succeed");
        let wv = weighted_variance(&vals, &wts, mu).expect("weighted_variance should succeed");
        // Population sample variance of the dataset
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let classic_var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert!((wv - classic_var).abs() < 1e-10);
    }

    #[test]
    fn weighted_variance_degenerate_single_element_ok() {
        // Single element: denom will be zero → error
        let vals = [5.0];
        let wts = [1.0];
        let mu = weighted_mean(&vals, &wts).expect("weighted_mean should succeed");
        // denom = 1 - 1/1 = 0 → NumericalInstability
        let result = weighted_variance(&vals, &wts, mu);
        assert!(result.is_err());
    }

    // ---- stratified_variance ----

    #[test]
    fn stratified_two_strata_basic() {
        // Stratum 0: values 1,2,3 — Stratum 1: values 10,11,12
        let values = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let weights = [1.0; 6];
        let strata = [0, 0, 0, 1, 1, 1];
        let r = stratified_variance(&values, &weights, &strata, 2)
            .expect("stratified_variance should succeed");
        // Overall mean should be around 6.5 (equal strata, equal weights)
        assert!((r.estimate - 6.5).abs() < 1e-6);
        assert!(r.variance >= 0.0);
        assert!(r.se >= 0.0);
        assert!(r.df > 0.0);
        // CI should span the estimate
        assert!(r.ci_95.0 < r.estimate && r.ci_95.1 > r.estimate);
    }

    #[test]
    fn stratified_single_stratum_variance_zero() {
        // With one stratum only, V = W_1^2 * V_1 / n_1 = 1 * V / n
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = [1.0; 5];
        let strata = [0, 0, 0, 0, 0];
        let r = stratified_variance(&values, &weights, &strata, 1)
            .expect("stratified_variance should succeed");
        assert!((r.estimate - 3.0).abs() < 1e-10);
    }

    #[test]
    fn stratified_unequal_strata_size() {
        let values = [1.0, 2.0, 3.0, 4.0, 100.0];
        let weights = [1.0; 5];
        let strata = [0, 0, 0, 0, 1]; // 4 vs 1
        let r = stratified_variance(&values, &weights, &strata, 2)
            .expect("stratified_variance should succeed");
        // Stratum 0 contributes more
        assert!(r.estimate < 100.0 && r.estimate > 2.0);
    }

    #[test]
    fn stratified_empty_input_error() {
        assert!(matches!(
            stratified_variance(&[], &[], &[], 1),
            Err(StatsError::EmptyInput)
        ));
    }

    // ---- cluster_variance ----

    #[test]
    fn cluster_variance_two_clusters() {
        let values = [1.0, 2.0, 10.0, 11.0];
        let weights = [1.0; 4];
        let clusters = [0, 0, 1, 1];
        let r = cluster_variance(&values, &weights, &clusters, 2)
            .expect("cluster_variance should succeed");
        // Overall mean ~ (1+2+10+11)/4 = 6
        assert!((r.estimate - 6.0).abs() < 1e-10);
        assert!(r.variance >= 0.0);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn cluster_variance_identical_clusters() {
        // Identical clusters → variance should be near zero
        let values = [5.0, 5.0, 5.0, 5.0];
        let weights = [1.0; 4];
        let clusters = [0, 0, 1, 1];
        let r = cluster_variance(&values, &weights, &clusters, 2)
            .expect("cluster_variance should succeed");
        assert!((r.estimate - 5.0).abs() < 1e-10);
        assert!(r.variance < 1e-12);
    }

    #[test]
    fn cluster_variance_insufficient_clusters_error() {
        assert!(matches!(
            cluster_variance(&[1.0], &[1.0], &[0], 1),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    // ---- design_effect ----

    #[test]
    fn design_effect_ratio() {
        let deff = design_effect(1.0, 2.5);
        assert!((deff - 2.5).abs() < 1e-12);
    }

    #[test]
    fn design_effect_both_zero() {
        assert!((design_effect(0.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn design_effect_srs_zero_complex_nonzero() {
        assert!(design_effect(0.0, 1.0).is_infinite());
    }

    // ---- jackknife_survey_variance ----

    #[test]
    fn jackknife_survey_variance_basic() {
        // Use weighted mean as statistic
        let values = [1.0, 2.0, 10.0, 11.0];
        let weights = [1.0; 4];
        let clusters = [0, 0, 1, 1];
        let stat = |v: &[f64], w: &[f64]| -> f64 {
            let sw: f64 = w.iter().sum();
            if sw == 0.0 {
                return 0.0;
            }
            v.iter().zip(w).map(|(x, wi)| x * wi).sum::<f64>() / sw
        };
        let jk_var = jackknife_survey_variance(&values, &weights, &clusters, 2, stat)
            .expect("jackknife_survey_variance should succeed");
        assert!(jk_var >= 0.0);
    }

    #[test]
    fn jackknife_survey_variance_insufficient_clusters() {
        let stat = |_v: &[f64], _w: &[f64]| 0.0f64;
        assert!(matches!(
            jackknife_survey_variance(&[1.0], &[1.0], &[0], 1, stat),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    // ---- SurveyDesign constructor ----

    #[test]
    fn survey_design_construction() {
        let design = SurveyDesign::new(vec![1.0, 2.0, 1.0], vec![0, 0, 1], Some(vec![0, 0, 1]), 2)
            .expect("value should be present");
        assert_eq!(design.n_strata, 2);
        assert_eq!(design.weights.len(), 3);
    }

    #[test]
    fn survey_design_dim_mismatch_error() {
        assert!(matches!(
            SurveyDesign::new(vec![1.0, 2.0], vec![0], None, 1),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }
}
