//! CBLOF — Cluster-Based Local Outlier Factor (He, Xu, Deng 2003).
//!
//! Clusters data with k-means (k-means++ initialisation) then splits clusters
//! into **large** (LC) and **small** (SC) subsets.  Anomaly scores reflect
//! how far a point is from large clusters, scaled by cluster size.
//!
//! ## LC/SC Split
//!
//! Sort clusters by size descending.  Extend the LC prefix until:
//! - The cumulative count ≥ `α · n`, **and**
//! - The size ratio between the last LC cluster and the next cluster ≥ `β`.
//!
//! ## Score
//!
//! ```text
//! Score(x ∈ LC_k) = |C_k| · ‖x − μ_k‖₂          (or ‖x − μ_k‖₂ if !use_cluster_weight)
//! Score(x ∈ SC_j) = |C_j| · min_{k∈LC} ‖x − μ_k‖₂  (or min dist if !use_cluster_weight)
//! ```

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the CBLOF anomaly detector.
#[derive(Debug, Clone)]
pub struct CblofConfig {
    /// Number of clusters `k` for k-means (default 8).
    pub n_clusters: usize,
    /// Cumulative-count fraction threshold for LC/SC split (default 0.9).
    pub alpha: f32,
    /// Size-ratio threshold for LC/SC split (default 5.0).
    pub beta: f32,
    /// k-means maximum iterations (default 100).
    pub max_iter: usize,
    /// Fraction of samples labelled outliers by `cblof_predict` (default 0.1).
    pub contamination: f32,
    /// Multiply score by cluster size (default `true`).
    pub use_cluster_weight: bool,
    /// RNG seed for k-means++ initialisation (default 42).
    pub init_seed: u64,
}

impl Default for CblofConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            alpha: 0.9,
            beta: 5.0,
            max_iter: 100,
            contamination: 0.1,
            use_cluster_weight: true,
            init_seed: 42,
        }
    }
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted CBLOF model.
pub struct CblofFit {
    /// Cluster centroids `[k × d]` (row-major).
    pub centroids: Vec<f32>,
    /// Cluster assignment for each training point `[n]`.
    pub labels: Vec<usize>,
    /// Indices (into `centroids`) of large clusters.
    pub large_cluster_ids: Vec<usize>,
    /// Number of training points per cluster `[k]`.
    pub cluster_sizes: Vec<usize>,
    /// Feature dimensionality.
    pub d: usize,
    /// Fraction of outliers for predict.
    pub contamination: f32,
    /// Whether to multiply score by cluster size.
    pub use_cluster_weight: bool,
}

// ─── Euclidean distance helpers ───────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance between two equal-length slices.
#[inline]
fn eucl_dist(a: &[f32], b: &[f32]) -> f32 {
    sq_dist(a, b).sqrt()
}

// ─── k-means++ initialisation ─────────────────────────────────────────────────

/// Initialise `k` centroids from `data [n × d]` using the k-means++ seeding
/// scheme (proportional to squared distance to nearest already-chosen centroid).
fn kmeans_plus_plus_init(data: &[f32], n: usize, d: usize, k: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut centroids: Vec<f32> = Vec::with_capacity(k * d);

    // First centroid: uniform random
    let first = rng.next_usize(n);
    centroids.extend_from_slice(&data[first * d..(first + 1) * d]);

    let mut dists = vec![f32::INFINITY; n];

    for c_idx in 1..k {
        // Update min-squared distances to the most recently added centroid
        let last_c = &centroids[(c_idx - 1) * d..c_idx * d];
        for i in 0..n {
            let xi = &data[i * d..(i + 1) * d];
            let sd = sq_dist(xi, last_c);
            if sd < dists[i] {
                dists[i] = sd;
            }
        }

        // Sample next centroid proportional to D²
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            // All remaining points coincide; pick any
            let idx = rng.next_usize(n);
            centroids.extend_from_slice(&data[idx * d..(idx + 1) * d]);
            continue;
        }

        let threshold = rng.next_f32() * total;
        let mut cumsum = 0.0_f32;
        let mut chosen = n - 1;
        for (i, &d_i) in dists.iter().enumerate() {
            cumsum += d_i;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[chosen * d..(chosen + 1) * d]);
    }

    centroids
}

// ─── k-means ─────────────────────────────────────────────────────────────────

/// Run k-means on `data [n × d]` with `k` clusters (k-means++ init).
///
/// Returns `(centroids [k × d], labels [n])`.
fn kmeans(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    rng: &mut LcgRng,
) -> (Vec<f32>, Vec<usize>) {
    let mut centroids = kmeans_plus_plus_init(data, n, d, k, rng);
    let mut labels = vec![0_usize; n];

    for _ in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let xi = &data[i * d..(i + 1) * d];
            let nearest = (0..k)
                .min_by(|&a, &b| {
                    let da = sq_dist(xi, &centroids[a * d..(a + 1) * d]);
                    let db = sq_dist(xi, &centroids[b * d..(b + 1) * d]);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if labels[i] != nearest {
                labels[i] = nearest;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step: recompute centroids as cluster means
        let mut sums = vec![0.0_f32; k * d];
        let mut counts = vec![0_usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            let xi = &data[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c * d + j] += xi[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            }
            // If a cluster is empty, centroid stays at its previous value.
        }
    }

    (centroids, labels)
}

// ─── LC/SC split ─────────────────────────────────────────────────────────────

/// Determine which cluster indices are **large** clusters (LC).
///
/// Algorithm:
/// 1. Sort clusters by size descending.
/// 2. Extend the LC prefix until:
///    - Cumulative count ≥ `α · n`, **and**
///    - Size of last LC / size of next cluster ≥ `β`.
/// 3. Always include at least one cluster.
fn find_large_clusters(
    cluster_sizes: &[usize],
    k: usize,
    alpha: f32,
    beta: f32,
    n: usize,
) -> Vec<usize> {
    // Sort cluster indices by size descending
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| cluster_sizes[b].cmp(&cluster_sizes[a]));

    let alpha_threshold = (alpha * n as f32).ceil() as usize;
    let mut cumulative = 0_usize;
    let mut lc_prefix_len = 1; // always at least 1

    cumulative += cluster_sizes[order[0]];

    for pos in 1..k {
        if cumulative >= alpha_threshold {
            // Check beta ratio before adding pos-th cluster
            let size_last_lc = cluster_sizes[order[pos - 1]];
            let size_next = cluster_sizes[order[pos]];
            if size_next == 0 || (size_last_lc as f32 / size_next as f32) >= beta {
                // Split here: everything before pos is LC
                lc_prefix_len = pos;
                break;
            }
        }
        cumulative += cluster_sizes[order[pos]];
        lc_prefix_len = pos + 1;
    }

    order[..lc_prefix_len].to_vec()
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a CBLOF model to training data `data` of shape `[n × d]` (row-major).
pub fn cblof_fit(data: &[f32], n: usize, d: usize, cfg: CblofConfig) -> AnomalyResult<CblofFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if data.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: data.len(),
        });
    }
    if cfg.n_clusters == 0 {
        return Err(AnomalyError::InvalidK { k: 0 });
    }
    if cfg.n_clusters > n {
        return Err(AnomalyError::InvalidK { k: cfg.n_clusters });
    }

    let k = cfg.n_clusters;
    let mut rng = LcgRng::new(cfg.init_seed);

    let (centroids, labels) = kmeans(data, n, d, k, cfg.max_iter, &mut rng);

    // Compute cluster sizes
    let mut cluster_sizes = vec![0_usize; k];
    for &l in &labels {
        cluster_sizes[l] += 1;
    }

    let large_cluster_ids = find_large_clusters(&cluster_sizes, k, cfg.alpha, cfg.beta, n);

    if large_cluster_ids.is_empty() {
        return Err(AnomalyError::Internal {
            msg: "LC/SC split produced no large clusters".into(),
        });
    }

    Ok(CblofFit {
        centroids,
        labels,
        large_cluster_ids,
        cluster_sizes,
        d,
        contamination: cfg.contamination,
        use_cluster_weight: cfg.use_cluster_weight,
    })
}

/// Compute per-sample CBLOF anomaly scores for `data` of shape `[n × d]`.
///
/// For each query point:
/// 1. Find the nearest centroid (index `c`).
/// 2. If `c ∈ LC`: score = `|C_c| * dist(x, μ_c)` (or `dist` if `!use_cluster_weight`).
/// 3. If `c ∈ SC`: score = `|C_c| * min_{lc∈LC} dist(x, μ_{lc})` (or `min dist`).
pub fn cblof_score(fit: &CblofFit, data: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
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

    let k = fit.cluster_sizes.len();
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let xi = &data[i * fit.d..(i + 1) * fit.d];

        // Find nearest centroid
        let (nearest_c, nearest_dist) = (0..k)
            .map(|c| {
                let mu_c = &fit.centroids[c * fit.d..(c + 1) * fit.d];
                (c, eucl_dist(xi, mu_c))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        let is_large = fit.large_cluster_ids.contains(&nearest_c);

        let score = if is_large {
            // LC case
            if fit.use_cluster_weight {
                fit.cluster_sizes[nearest_c] as f32 * nearest_dist
            } else {
                nearest_dist
            }
        } else {
            // SC case: distance to nearest LC centroid
            let min_lc_dist = fit
                .large_cluster_ids
                .iter()
                .map(|&lc| {
                    let mu_lc = &fit.centroids[lc * fit.d..(lc + 1) * fit.d];
                    eucl_dist(xi, mu_lc)
                })
                .fold(f32::INFINITY, f32::min);

            if fit.use_cluster_weight {
                fit.cluster_sizes[nearest_c] as f32 * min_lc_dist
            } else {
                min_lc_dist
            }
        };

        scores.push(score);
    }

    Ok(scores)
}

/// Classify each sample as outlier (`true`) or inlier (`false`).
///
/// The top `contamination × n` fraction (by descending score) are labelled `true`.
pub fn cblof_predict(fit: &CblofFit, data: &[f32], n: usize) -> AnomalyResult<Vec<bool>> {
    let scores = cblof_score(fit, data, n)?;

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

    fn default_cfg() -> CblofConfig {
        CblofConfig::default()
    }

    /// Generate `n` 2-D points near `center` with std `sigma`.
    fn cluster_2d(n: usize, cx: f32, cy: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .flat_map(|_| {
                [
                    cx + rng.next_normal() * sigma,
                    cy + rng.next_normal() * sigma,
                ]
            })
            .collect()
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────
    #[test]
    fn two_blobs_outlier_top_ranked() {
        let mut data = cluster_2d(40, 0.0, 0.0, 0.3, 1);
        data.extend(cluster_2d(40, 10.0, 10.0, 0.3, 2));
        data.extend_from_slice(&[100.0_f32, 100.0]); // outlier
        let n = 81;
        let cfg = CblofConfig {
            n_clusters: 3,
            alpha: 0.8,
            beta: 3.0,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        let scores = cblof_score(&fit, &data, n).expect("CBLOF score should succeed");
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("scores vec is non-empty");
        assert_eq!(
            max_idx, 80,
            "outlier at index 80 should have highest CBLOF score"
        );
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────
    #[test]
    fn outlier_score_exceeds_inlier() {
        let mut data = cluster_2d(50, 0.0, 0.0, 0.5, 3);
        let outlier = vec![200.0_f32, 200.0];
        data.extend_from_slice(&outlier);
        let n = 51;
        let cfg = CblofConfig {
            n_clusters: 4,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        let scores = cblof_score(&fit, &data, n).expect("CBLOF score should succeed");
        let out_score = scores[50];
        let max_in = scores[..50]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            out_score > max_in,
            "outlier score {out_score} should exceed max inlier score {max_in}"
        );
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────
    #[test]
    fn cluster_weight_scales_score() {
        let data = cluster_2d(30, 0.0, 0.0, 0.2, 4);
        let test_pt = vec![5.0_f32, 5.0];

        let fit_w = cblof_fit(
            &data,
            30,
            2,
            CblofConfig {
                use_cluster_weight: true,
                n_clusters: 3,
                ..default_cfg()
            },
        )
        .expect("CBLOF fit should succeed");
        let fit_nw = cblof_fit(
            &data,
            30,
            2,
            CblofConfig {
                use_cluster_weight: false,
                n_clusters: 3,
                ..default_cfg()
            },
        )
        .expect("CBLOF fit should succeed");

        let score_w = cblof_score(&fit_w, &test_pt, 1).expect("CBLOF score should succeed")[0];
        let score_nw = cblof_score(&fit_nw, &test_pt, 1).expect("CBLOF score should succeed")[0];

        // Weighted score should be larger (multiplied by cluster size ≥ 1)
        assert!(
            score_w >= score_nw,
            "weighted score {score_w} should ≥ unweighted score {score_nw}"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────
    #[test]
    fn large_cluster_split_alpha() {
        // With alpha=0.5 on 3 balanced clusters, first cluster alone covers ≥50%
        let n = 90;
        let mut data = cluster_2d(30, 0.0, 0.0, 0.1, 5);
        data.extend(cluster_2d(30, 10.0, 0.0, 0.1, 6));
        data.extend(cluster_2d(30, 0.0, 10.0, 0.1, 7));
        let cfg = CblofConfig {
            n_clusters: 3,
            alpha: 0.5,
            beta: 100.0, // very large beta → only alpha triggers split
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        assert!(
            !fit.large_cluster_ids.is_empty(),
            "should have at least one large cluster"
        );
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────
    #[test]
    fn large_cluster_split_beta() {
        // beta=1 (every ratio ≥ 1) with alpha=1.0 → all clusters are LC
        let mut data = cluster_2d(30, 0.0, 0.0, 0.2, 8);
        data.extend(cluster_2d(30, 5.0, 5.0, 0.2, 9));
        let n = 60;
        let cfg = CblofConfig {
            n_clusters: 2,
            alpha: 1.0,
            beta: 1.0,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        // With alpha=1.0 cumulative never exceeds threshold so all clusters stay LC
        assert_eq!(
            fit.large_cluster_ids.len(),
            2,
            "both clusters should be large when alpha=1.0 and beta=1.0"
        );
    }

    // ── Test 6 ────────────────────────────────────────────────────────────────
    #[test]
    fn k_equals_one_all_large() {
        let data = cluster_2d(20, 1.0, 1.0, 0.5, 10);
        let cfg = CblofConfig {
            n_clusters: 1,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 20, 2, cfg).expect("CBLOF fit should succeed");
        assert_eq!(
            fit.large_cluster_ids,
            vec![0],
            "k=1 → single large cluster at index 0"
        );
    }

    // ── Test 7 ────────────────────────────────────────────────────────────────
    #[test]
    fn k_equals_n_each_point_own_cluster() {
        // n=k → each point is its own cluster; all clusters size=1
        let n = 5;
        let data: Vec<f32> = (0..n).flat_map(|i| [i as f32 * 10.0, 0.0_f32]).collect();
        let cfg = CblofConfig {
            n_clusters: n,
            alpha: 0.9,
            beta: 2.0,
            max_iter: 200,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        let total: usize = fit.cluster_sizes.iter().sum();
        assert_eq!(total, n);
    }

    // ── Test 8 ────────────────────────────────────────────────────────────────
    #[test]
    fn invalid_k_zero() {
        let data = cluster_2d(10, 0.0, 0.0, 1.0, 11);
        let cfg = CblofConfig {
            n_clusters: 0,
            ..default_cfg()
        };
        assert!(matches!(
            cblof_fit(&data, 10, 2, cfg),
            Err(AnomalyError::InvalidK { .. })
        ));
    }

    // ── Test 9 ────────────────────────────────────────────────────────────────
    #[test]
    fn invalid_k_exceeds_n() {
        let data = cluster_2d(5, 0.0, 0.0, 1.0, 12);
        let cfg = CblofConfig {
            n_clusters: 10, // k > n
            ..default_cfg()
        };
        assert!(matches!(
            cblof_fit(&data, 5, 2, cfg),
            Err(AnomalyError::InvalidK { .. })
        ));
    }

    // ── Test 10 ───────────────────────────────────────────────────────────────
    #[test]
    fn dim_mismatch_score() {
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 13);
        let cfg = CblofConfig {
            n_clusters: 3,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 20, 2, cfg).expect("CBLOF fit should succeed");
        // Supply 15 elements for n=5 with d=2 model → expects 10
        let bad = vec![1.0_f32; 15];
        assert!(matches!(
            cblof_score(&fit, &bad, 5),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 11 ───────────────────────────────────────────────────────────────
    #[test]
    fn dim_mismatch_fit() {
        // data.len()=11 ≠ n*d = 5*3 = 15
        let data = vec![1.0_f32; 11];
        let cfg = CblofConfig {
            n_clusters: 2,
            ..default_cfg()
        };
        assert!(matches!(
            cblof_fit(&data, 5, 3, cfg),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    // ── Test 12 ───────────────────────────────────────────────────────────────
    #[test]
    fn empty_input() {
        assert!(matches!(
            cblof_fit(&[], 0, 2, default_cfg()),
            Err(AnomalyError::EmptyInput)
        ));
    }

    // ── Test 13 ───────────────────────────────────────────────────────────────
    #[test]
    fn predict_marks_correct_fraction() {
        // n=20, contamination=0.1 → exactly 2 outliers predicted
        let data = cluster_2d(20, 0.0, 0.0, 0.5, 14);
        let cfg = CblofConfig {
            n_clusters: 3,
            contamination: 0.1,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 20, 2, cfg).expect("CBLOF fit should succeed");
        let preds = cblof_predict(&fit, &data, 20).expect("CBLOF predict should succeed");
        let n_out = preds.iter().filter(|&&p| p).count();
        // ceil(0.1 * 20) = 2
        assert_eq!(n_out, 2, "expected 2 predicted outliers, got {n_out}");
    }

    // ── Test 14 ───────────────────────────────────────────────────────────────
    #[test]
    fn centroids_count_equals_k() {
        let data = cluster_2d(30, 0.0, 0.0, 1.0, 15);
        let k = 5;
        let cfg = CblofConfig {
            n_clusters: k,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 30, 2, cfg).expect("CBLOF fit should succeed");
        assert_eq!(
            fit.centroids.len(),
            k * 2,
            "expected {} centroid elements (k={k}, d=2), got {}",
            k * 2,
            fit.centroids.len()
        );
    }

    // ── Test 15 ───────────────────────────────────────────────────────────────
    #[test]
    fn labels_valid_range() {
        let data = cluster_2d(40, 1.0, -1.0, 1.0, 16);
        let k = 4;
        let cfg = CblofConfig {
            n_clusters: k,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 40, 2, cfg).expect("CBLOF fit should succeed");
        for (i, &l) in fit.labels.iter().enumerate() {
            assert!(l < k, "label[{i}] = {l} out of range [0, {k})");
        }
    }

    // ── Test 16 ───────────────────────────────────────────────────────────────
    #[test]
    fn cluster_sizes_sum_to_n() {
        let n = 50;
        let data = cluster_2d(n, 0.0, 0.0, 2.0, 17);
        let cfg = CblofConfig {
            n_clusters: 5,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        let total: usize = fit.cluster_sizes.iter().sum();
        assert_eq!(total, n, "cluster sizes should sum to n={n}, got {total}");
    }

    // ── Test 17 ───────────────────────────────────────────────────────────────
    #[test]
    fn deterministic_with_same_seed() {
        let data = cluster_2d(30, 0.0, 0.0, 1.0, 18);
        let cfg1 = CblofConfig {
            n_clusters: 4,
            init_seed: 77,
            ..default_cfg()
        };
        let cfg2 = cfg1.clone();
        let fit1 = cblof_fit(&data, 30, 2, cfg1).expect("CBLOF fit should succeed");
        let fit2 = cblof_fit(&data, 30, 2, cfg2).expect("CBLOF fit should succeed");
        for (i, (&a, &b)) in fit1.centroids.iter().zip(fit2.centroids.iter()).enumerate() {
            assert_eq!(a, b, "centroid[{i}] differs across runs with same seed");
        }
    }

    // ── Test 18 ───────────────────────────────────────────────────────────────
    #[test]
    fn alpha_1_all_large() {
        // alpha=1.0 → cumulative never satisfies ≥ alpha*n before consuming all clusters
        let data = cluster_2d(30, 0.0, 0.0, 1.0, 19);
        let k = 3;
        let cfg = CblofConfig {
            n_clusters: k,
            alpha: 1.0,
            beta: 1000.0, // extremely tight beta to prevent early split
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 30, 2, cfg).expect("CBLOF fit should succeed");
        // With alpha=1.0 cumulative only meets threshold when all clusters included
        assert_eq!(
            fit.large_cluster_ids.len(),
            k,
            "alpha=1.0 should make all {k} clusters large"
        );
    }

    // ── Test 19 ───────────────────────────────────────────────────────────────
    #[test]
    fn inlier_score_lower_than_outlier() {
        let data = cluster_2d(40, 0.0, 0.0, 0.5, 20);
        let cfg = CblofConfig {
            n_clusters: 4,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 40, 2, cfg).expect("CBLOF fit should succeed");

        let inlier = vec![0.0_f32, 0.0]; // cluster center
        let outlier = vec![100.0_f32, 100.0];

        let s_in = cblof_score(&fit, &inlier, 1).expect("CBLOF score should succeed")[0];
        let s_out = cblof_score(&fit, &outlier, 1).expect("CBLOF score should succeed")[0];
        assert!(
            s_out > s_in,
            "outlier score {s_out} should exceed inlier score {s_in}"
        );
    }

    // ── Test 20 ───────────────────────────────────────────────────────────────
    #[test]
    fn large_cluster_ids_nonempty() {
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 21);
        let fit = cblof_fit(&data, 20, 2, default_cfg()).expect("CBLOF fit should succeed");
        assert!(
            !fit.large_cluster_ids.is_empty(),
            "must have at least 1 large cluster"
        );
    }

    // ── Test 21 ───────────────────────────────────────────────────────────────
    #[test]
    fn small_cluster_score_uses_nearest_lc() {
        // Construct scenario where one cluster is definitively SC.
        // Two large clusters (big) + one small cluster (1 point).
        // The SC point's score should equal |SC| * dist_to_nearest_LC.
        let mut data = cluster_2d(40, 0.0, 0.0, 0.1, 22);
        data.extend(cluster_2d(40, 20.0, 0.0, 0.1, 23));
        // Single isolated point far away
        data.extend_from_slice(&[100.0_f32, 0.0]);
        let n = 81;

        let cfg = CblofConfig {
            n_clusters: 3,
            alpha: 0.95,
            beta: 3.0,
            contamination: 0.05,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        let scores = cblof_score(&fit, &data, n).expect("CBLOF score should succeed");

        // The isolated point at index 80 should have a high score
        let isolated_score = scores[80];
        let mean_cluster_score: f32 = scores[..80].iter().sum::<f32>() / 80.0;
        assert!(
            isolated_score > mean_cluster_score,
            "isolated SC point score {isolated_score} should > cluster mean {mean_cluster_score}"
        );
    }

    // ── Test 22 ───────────────────────────────────────────────────────────────
    #[test]
    fn kmeans_converges_on_obvious_clusters() {
        let n = 90;
        let mut data = cluster_2d(30, 0.0, 0.0, 0.05, 24);
        data.extend(cluster_2d(30, 100.0, 0.0, 0.05, 25));
        data.extend(cluster_2d(30, 0.0, 100.0, 0.05, 26));

        let cfg = CblofConfig {
            n_clusters: 3,
            alpha: 0.9,
            beta: 2.0,
            max_iter: 200,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, n, 2, cfg).expect("CBLOF fit should succeed");
        // Each cluster should have exactly 30 points
        let mut sizes = fit.cluster_sizes.clone();
        sizes.sort_unstable();
        for &sz in &sizes {
            assert_eq!(
                sz, 30,
                "each cluster should have exactly 30 points, got {sz}"
            );
        }
    }

    // ── Test 23 ───────────────────────────────────────────────────────────────
    #[test]
    fn score_is_nonneg() {
        let data = cluster_2d(30, 0.0, 0.0, 1.0, 27);
        let cfg = CblofConfig {
            n_clusters: 4,
            ..default_cfg()
        };
        let fit = cblof_fit(&data, 30, 2, cfg).expect("CBLOF fit should succeed");
        let scores = cblof_score(&fit, &data, 30).expect("CBLOF score should succeed");
        for &s in &scores {
            assert!(s >= 0.0, "CBLOF score should be non-negative, got {s}");
        }
    }
}
