//! Multilevel and cluster bootstrap methods for clustered data.
//!
//! Implements Efron (1979) bootstrap adapted for hierarchical data structures,
//! Rao-Wu (1988) cluster resampling, two-level hierarchical bootstrap, and
//! the delete-a-cluster jackknife variance estimator.
//!
//! # References
//! - Efron, B. (1979). Bootstrap methods: another look at the jackknife.
//!   *Ann. Statist.*, 7(1), 1-26.
//! - Rao, J. N. K. and Wu, C. F. J. (1988). Resampling inference with complex
//!   survey data. *J. Amer. Statist. Assoc.*, 83(401), 231-241.

use crate::descriptive::quantile::quantile;
use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

// ─── public types ─────────────────────────────────────────────────────────────

/// Configuration for cluster bootstrap procedures.
#[derive(Debug, Clone)]
pub struct ClusterBootstrapConfig {
    /// Number of bootstrap replicates.
    pub n_bootstrap: usize,
    /// Seed for the internal LCG RNG.
    pub seed: u64,
    /// If `true`, resample within each selected cluster (Rao-Wu variant);
    /// if `false`, only resample clusters (include all observations in each
    /// selected cluster).
    pub resample_within: bool,
}

impl Default for ClusterBootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 1_000,
            seed: 0,
            resample_within: true,
        }
    }
}

/// Result of a cluster bootstrap or two-level bootstrap procedure.
///
/// Distinct from [`super::bootstrap::BootstrapResult`] which uses different
/// field names (`theta_hat`, `std_error`, `ci_lower`, `ci_upper`).
#[derive(Debug, Clone)]
pub struct ClusterBootstrapResult {
    /// Original estimate of the statistic on the full data.
    pub estimate: f64,
    /// Bootstrap bias estimate: `mean(replicates) - estimate`.
    pub bias: f64,
    /// Bootstrap standard error: std dev of replicates.
    pub std_err: f64,
    /// 95 % percentile bootstrap confidence interval `[2.5th, 97.5th]`.
    pub ci_95: (f64, f64),
    /// All bootstrap replicates (length = `n_bootstrap`).
    pub replicates: Vec<f64>,
}

// ─── internal helpers ─────────────────────────────────────────────────────────

/// Compute a ClusterBootstrapResult from a slice of replicates and the
/// original estimate.
fn make_result(estimate: f64, replicates: Vec<f64>) -> StatsResult<ClusterBootstrapResult> {
    let nb = replicates.len();
    if nb == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_bootstrap".into(),
            reason: "must be > 0".into(),
        });
    }
    let mean_rep: f64 = replicates.iter().sum::<f64>() / nb as f64;
    let var_rep: f64 = replicates
        .iter()
        .map(|&v| (v - mean_rep).powi(2))
        .sum::<f64>()
        / (nb as f64 - 1.0).max(1.0);
    let std_err = var_rep.sqrt();
    let bias = mean_rep - estimate;
    let ci_lo = quantile(&replicates, 0.025)?;
    let ci_hi = quantile(&replicates, 0.975)?;
    Ok(ClusterBootstrapResult {
        estimate,
        bias,
        std_err,
        ci_95: (ci_lo, ci_hi),
        replicates,
    })
}

// ─── public functions ─────────────────────────────────────────────────────────

/// Cluster bootstrap (Efron / Rao-Wu) for clustered data.
///
/// # Arguments
/// * `data` — flattened observation vector (length `n`).
/// * `cluster_ids` — cluster membership for each observation (length `n`);
///   cluster labels must be integers in `0..n_clusters`.
/// * `n` — number of observations (must equal `data.len()` and
///   `cluster_ids.len()`).
/// * `n_clusters` — total number of distinct clusters.
/// * `statistic` — function of `(data, cluster_ids)` → scalar estimate.
/// * `cfg` — bootstrap configuration.
///
/// # Algorithm
/// 1. Compute the original estimate on the full data.
/// 2. For each replicate:
///    a. Draw `n_clusters` cluster labels with replacement.
///    b. For each selected cluster: collect all its observations (or, when
///    `cfg.resample_within = true`, resample within the cluster with
///    replacement to the same cluster size).
///    c. Re-number cluster IDs 0..n_clusters in the resampled data so that
///    the statistic receives a conformant `(data, cluster_ids)` pair.
/// 3. Build [`ClusterBootstrapResult`].
///
/// # Errors
/// Returns an error if `data` is empty, sizes disagree, or `n_clusters = 0`.
pub fn cluster_bootstrap(
    data: &[f64],
    cluster_ids: &[usize],
    n: usize,
    n_clusters: usize,
    statistic: impl Fn(&[f64], &[usize]) -> f64,
    cfg: &ClusterBootstrapConfig,
) -> StatsResult<ClusterBootstrapResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if data.len() != n || cluster_ids.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: data.len(),
            b: cluster_ids.len(),
        });
    }
    if n_clusters == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_clusters".into(),
            reason: "must be > 0".into(),
        });
    }
    if cfg.n_bootstrap == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_bootstrap".into(),
            reason: "must be > 0".into(),
        });
    }

    // Pre-compute cluster membership lists (O(n · n_clusters) but done once).
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &cid) in cluster_ids.iter().enumerate() {
        if cid >= n_clusters {
            return Err(StatsError::IndexOutOfBounds {
                index: cid,
                len: n_clusters,
            });
        }
        cluster_members[cid].push(i);
    }

    let estimate = statistic(data, cluster_ids);
    let mut rng = LcgRng::new(cfg.seed);
    let mut replicates = Vec::with_capacity(cfg.n_bootstrap);

    let mut boot_data: Vec<f64> = Vec::with_capacity(n);
    let mut boot_ids: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..cfg.n_bootstrap {
        boot_data.clear();
        boot_ids.clear();

        for new_cid in 0..n_clusters {
            // Draw a cluster with replacement.
            let selected_cluster = rng.next_usize(n_clusters);
            let members = &cluster_members[selected_cluster];
            if members.is_empty() {
                continue;
            }
            if cfg.resample_within {
                // Rao-Wu: resample within cluster with replacement.
                let cluster_size = members.len();
                for _ in 0..cluster_size {
                    let obs_idx = members[rng.next_usize(cluster_size)];
                    boot_data.push(data[obs_idx]);
                    boot_ids.push(new_cid);
                }
            } else {
                // Include all observations in the selected cluster.
                for &obs_idx in members {
                    boot_data.push(data[obs_idx]);
                    boot_ids.push(new_cid);
                }
            }
        }

        if boot_data.is_empty() {
            continue;
        }
        let rep = statistic(&boot_data, &boot_ids);
        replicates.push(rep);
    }

    make_result(estimate, replicates)
}

/// Two-level hierarchical bootstrap.
///
/// Resamples level-1 units (e.g. schools) with replacement; within each
/// resampled level-1 unit, resamples level-2 observations (e.g. students)
/// with replacement to the same size.
///
/// # Arguments
/// * `level1_data` — `n_l1` slices, each holding level-2 observations for one
///   level-1 unit.
/// * `n_l1` — number of level-1 units (must equal `level1_data.len()`).
/// * `statistic` — function `(&[Vec<f64>]) → f64` over the nested structure.
/// * `n_bootstrap` — number of replicates.
/// * `seed` — RNG seed.
///
/// # Errors
/// Returns an error if `level1_data` is empty or sizes disagree.
pub fn two_level_bootstrap(
    level1_data: &[Vec<f64>],
    n_l1: usize,
    statistic: impl Fn(&[Vec<f64>]) -> f64,
    n_bootstrap: usize,
    seed: u64,
) -> StatsResult<ClusterBootstrapResult> {
    if level1_data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if level1_data.len() != n_l1 {
        return Err(StatsError::DimensionMismatch {
            a: level1_data.len(),
            b: n_l1,
        });
    }
    if n_bootstrap == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_bootstrap".into(),
            reason: "must be > 0".into(),
        });
    }

    let estimate = statistic(level1_data);
    let mut rng = LcgRng::new(seed);
    let mut replicates = Vec::with_capacity(n_bootstrap);

    let mut boot_structure: Vec<Vec<f64>> = Vec::with_capacity(n_l1);

    for _ in 0..n_bootstrap {
        boot_structure.clear();

        for _ in 0..n_l1 {
            // Resample level-1 with replacement.
            let l1_idx = rng.next_usize(n_l1);
            let l2_obs = &level1_data[l1_idx];
            let l2_size = l2_obs.len();
            if l2_size == 0 {
                boot_structure.push(Vec::new());
                continue;
            }
            // Resample level-2 with replacement within the selected l1 unit.
            let mut boot_l2 = Vec::with_capacity(l2_size);
            for _ in 0..l2_size {
                let l2_idx = rng.next_usize(l2_size);
                boot_l2.push(l2_obs[l2_idx]);
            }
            boot_structure.push(boot_l2);
        }

        let rep = statistic(&boot_structure);
        replicates.push(rep);
    }

    make_result(estimate, replicates)
}

/// Delete-a-cluster jackknife variance estimator.
///
/// Removes one cluster at a time and recomputes the statistic on the remaining
/// observations.  Returns the jackknife variance estimate.
///
/// The jackknife variance for cluster-level deletion is:
///
/// ```text
/// V_JK = ((G-1)/G) * Σ_g (θ_{-g} - θ_bar)²
/// ```
///
/// where G = `n_clusters`, θ_{-g} is the estimate with cluster g deleted,
/// and θ_bar = (1/G) Σ_g θ_{-g}.
///
/// # Errors
/// Returns an error if `data` is empty, sizes disagree, or `n_clusters < 2`.
pub fn jackknife_cluster(
    data: &[f64],
    cluster_ids: &[usize],
    n: usize,
    n_clusters: usize,
    statistic: impl Fn(&[f64], &[usize]) -> f64,
) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if data.len() != n || cluster_ids.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: data.len(),
            b: cluster_ids.len(),
        });
    }
    if n_clusters < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_clusters,
            need: 2,
        });
    }

    let mut leave_one_out: Vec<f64> = Vec::with_capacity(n_clusters);
    let mut buf_data: Vec<f64> = Vec::with_capacity(n);
    let mut buf_ids: Vec<usize> = Vec::with_capacity(n);

    for dropped in 0..n_clusters {
        buf_data.clear();
        buf_ids.clear();
        for (i, (&d, &cid)) in data.iter().zip(cluster_ids.iter()).enumerate() {
            let _ = i; // used via zip
            if cid != dropped {
                buf_data.push(d);
                buf_ids.push(cid);
            }
        }
        if buf_data.is_empty() {
            return Err(StatsError::InvalidParameter {
                name: "n_clusters".into(),
                reason: format!("all observations belong to cluster {dropped}; cannot drop it"),
            });
        }
        leave_one_out.push(statistic(&buf_data, &buf_ids));
    }

    let g = n_clusters as f64;
    let theta_bar: f64 = leave_one_out.iter().sum::<f64>() / g;
    let variance: f64 = leave_one_out
        .iter()
        .map(|&t| (t - theta_bar).powi(2))
        .sum::<f64>()
        * ((g - 1.0) / g);

    Ok(variance)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Cluster mean (mean of all observations, ignoring cluster IDs).
    fn global_mean(data: &[f64], _ids: &[usize]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Two-level grand mean.
    fn grand_mean_nested(level1: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for v in level1 {
            sum += v.iter().sum::<f64>();
            cnt += v.len();
        }
        if cnt == 0 { 0.0 } else { sum / cnt as f64 }
    }

    fn make_clustered_data(n_clusters: usize, cluster_size: usize) -> (Vec<f64>, Vec<usize>) {
        let mut data = Vec::new();
        let mut ids = Vec::new();
        for c in 0..n_clusters {
            for j in 0..cluster_size {
                // Cluster effect = c * 10; within-cluster index j
                data.push(c as f64 * 10.0 + j as f64);
                ids.push(c);
            }
        }
        (data, ids)
    }

    // ── 1. cluster_bootstrap returns correct field counts ─────────────────────
    #[test]
    fn cluster_bootstrap_replicate_count() {
        let (data, ids) = make_clustered_data(4, 5);
        let n = data.len();
        let cfg = ClusterBootstrapConfig {
            n_bootstrap: 100,
            seed: 42,
            resample_within: true,
        };
        let result = cluster_bootstrap(&data, &ids, n, 4, global_mean, &cfg).expect("ok");
        assert_eq!(result.replicates.len(), 100);
    }

    // ── 2. cluster_bootstrap estimate equals global_mean ────────────────────
    #[test]
    fn cluster_bootstrap_estimate_correct() {
        let (data, ids) = make_clustered_data(4, 5);
        let n = data.len();
        let expected = global_mean(&data, &ids);
        let cfg = ClusterBootstrapConfig::default();
        let result = cluster_bootstrap(&data, &ids, n, 4, global_mean, &cfg).expect("ok");
        assert!((result.estimate - expected).abs() < 1e-10);
    }

    // ── 3. cluster_bootstrap std_err is finite and positive ─────────────────
    #[test]
    fn cluster_bootstrap_std_err_positive() {
        let (data, ids) = make_clustered_data(5, 8);
        let n = data.len();
        let cfg = ClusterBootstrapConfig {
            n_bootstrap: 200,
            seed: 7,
            resample_within: true,
        };
        let result = cluster_bootstrap(&data, &ids, n, 5, global_mean, &cfg).expect("ok");
        assert!(result.std_err.is_finite() && result.std_err >= 0.0);
    }

    // ── 4. cluster_bootstrap CI contains truth ────────────────────────────────
    #[test]
    fn cluster_bootstrap_ci_contains_mean() {
        // Use many small clusters with moderate variation
        let n_clusters = 10;
        let cluster_size = 10;
        let (data, ids) = make_clustered_data(n_clusters, cluster_size);
        let n = data.len();
        let true_mean = global_mean(&data, &ids);
        let cfg = ClusterBootstrapConfig {
            n_bootstrap: 500,
            seed: 99,
            resample_within: true,
        };
        let result = cluster_bootstrap(&data, &ids, n, n_clusters, global_mean, &cfg).expect("ok");
        let (lo, hi) = result.ci_95;
        assert!(
            lo <= true_mean && true_mean <= hi,
            "CI=[{lo},{hi}] does not contain true_mean={true_mean}"
        );
    }

    // ── 5. resample_within=false also runs cleanly ───────────────────────────
    #[test]
    fn cluster_bootstrap_no_within_resample() {
        let (data, ids) = make_clustered_data(4, 5);
        let n = data.len();
        let cfg = ClusterBootstrapConfig {
            n_bootstrap: 100,
            seed: 11,
            resample_within: false,
        };
        let result = cluster_bootstrap(&data, &ids, n, 4, global_mean, &cfg).expect("ok");
        assert_eq!(result.replicates.len(), 100);
        assert!(result.std_err.is_finite());
    }

    // ── 6. cluster_bootstrap: empty data returns error ────────────────────────
    #[test]
    fn cluster_bootstrap_empty_error() {
        let cfg = ClusterBootstrapConfig::default();
        let r = cluster_bootstrap(&[], &[], 0, 2, global_mean, &cfg);
        assert!(r.is_err());
    }

    // ── 7. two_level_bootstrap replicate count ───────────────────────────────
    #[test]
    fn two_level_bootstrap_replicate_count() {
        let level1: Vec<Vec<f64>> = (0..5)
            .map(|c| (0..8).map(|j| c as f64 * 5.0 + j as f64).collect())
            .collect();
        let result = two_level_bootstrap(&level1, 5, grand_mean_nested, 200, 42).expect("ok");
        assert_eq!(result.replicates.len(), 200);
    }

    // ── 8. two_level_bootstrap estimate is grand mean ─────────────────────────
    #[test]
    fn two_level_bootstrap_estimate_correct() {
        let level1: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let true_mean = grand_mean_nested(&level1);
        let result = two_level_bootstrap(&level1, 3, grand_mean_nested, 300, 1).expect("ok");
        assert!((result.estimate - true_mean).abs() < 1e-10);
    }

    // ── 9. two_level_bootstrap CI width is positive ──────────────────────────
    #[test]
    fn two_level_bootstrap_ci_width_positive() {
        let level1: Vec<Vec<f64>> = (0..6)
            .map(|c| (0..10).map(|j| c as f64 * 3.0 + j as f64 * 0.5).collect())
            .collect();
        let result = two_level_bootstrap(&level1, 6, grand_mean_nested, 400, 33).expect("ok");
        let (lo, hi) = result.ci_95;
        assert!(hi > lo, "CI upper={hi} must exceed lower={lo}");
    }

    // ── 10. two_level_bootstrap empty error ──────────────────────────────────
    #[test]
    fn two_level_bootstrap_empty_error() {
        let r = two_level_bootstrap(&[], 0, grand_mean_nested, 10, 0);
        assert!(r.is_err());
    }

    // ── 11. jackknife_cluster variance is finite and non-negative ────────────
    #[test]
    fn jackknife_cluster_variance_non_negative() {
        let (data, ids) = make_clustered_data(5, 6);
        let n = data.len();
        let var = jackknife_cluster(&data, &ids, n, 5, global_mean).expect("ok");
        assert!(
            var.is_finite() && var >= 0.0,
            "JK variance={var} must be ≥ 0"
        );
    }

    // ── 12. jackknife_cluster: large cluster effect → large variance ──────────
    #[test]
    fn jackknife_cluster_reflects_cluster_effect() {
        // High between-cluster variation
        let (data_high, ids_high) = make_clustered_data(6, 5); // cluster means: 0,10,20,30,40,50
        let n = data_high.len();
        let var_high = jackknife_cluster(&data_high, &ids_high, n, 6, global_mean).expect("ok");

        // Low between-cluster variation: all clusters have similar means
        let data_low: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
        let ids_low: Vec<usize> = (0..30).map(|i| i / 5).collect();
        let var_low = jackknife_cluster(&data_low, &ids_low, 30, 6, global_mean).expect("ok");

        assert!(
            var_high > var_low,
            "high-cluster-effect var={var_high} should exceed low={var_low}"
        );
    }

    // ── 13. jackknife_cluster: n_clusters < 2 returns error ──────────────────
    #[test]
    fn jackknife_cluster_insufficient_clusters_error() {
        let data = vec![1.0, 2.0, 3.0];
        let ids = vec![0usize, 0, 0];
        let r = jackknife_cluster(&data, &ids, 3, 1, global_mean);
        assert!(r.is_err());
    }

    // ── 14. cluster_bootstrap bias is small for unbiased statistic ───────────
    #[test]
    fn cluster_bootstrap_bias_near_zero_for_mean() {
        let (data, ids) = make_clustered_data(8, 10);
        let n = data.len();
        let cfg = ClusterBootstrapConfig {
            n_bootstrap: 800,
            seed: 2024,
            resample_within: true,
        };
        let result = cluster_bootstrap(&data, &ids, n, 8, global_mean, &cfg).expect("ok");
        // Bootstrap bias for the mean should be small (not zero in general, but bounded)
        assert!(
            result.bias.abs() < 5.0,
            "bias={} unexpectedly large",
            result.bias
        );
    }

    // ── 15. two_level_bootstrap std_err positive ─────────────────────────────
    #[test]
    fn two_level_bootstrap_std_err_positive() {
        let level1: Vec<Vec<f64>> = (0..4)
            .map(|c| (0..6).map(|j| c as f64 + j as f64 * 0.2).collect())
            .collect();
        let result = two_level_bootstrap(&level1, 4, grand_mean_nested, 300, 77).expect("ok");
        assert!(
            result.std_err > 0.0,
            "std_err={} should be positive",
            result.std_err
        );
    }
}
