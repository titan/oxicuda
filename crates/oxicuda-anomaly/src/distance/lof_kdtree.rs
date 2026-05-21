//! LOF with k-d Tree acceleration.
//!
//! Implements a full median-split k-d tree for fast approximate nearest-neighbour
//! search, then applies the standard Local Outlier Factor algorithm using the tree
//! instead of a brute-force O(n²) distance matrix.
//!
//! The k-d tree is exact (not approximate): it visits all nodes needed to guarantee
//! the true `k` nearest neighbours up to a configurable priority queue depth.
//!
//! ## Complexity
//! - Build: O(n d log n)
//! - Query: O(d log n) expected, O(d n) worst case
//! - vs brute-force LOF: O(n d) per query instead of O(n² d)

use crate::error::{AnomalyError, AnomalyResult};

// ─── KdNode ──────────────────────────────────────────────────────────────────

/// A node in the k-d tree.
///
/// Internal nodes split on `feature` at `threshold`.
/// Leaf nodes have `point_idx = Some(i)` and `left = right = None`.
pub struct KdNode {
    /// Feature dimension to split on (undefined for leaf; set to 0).
    pub feature: usize,
    /// Split threshold (undefined for leaf).
    pub threshold: f64,
    pub left: Option<Box<KdNode>>,
    pub right: Option<Box<KdNode>>,
    /// Some(i) iff this is a leaf holding the i-th training point.
    pub point_idx: Option<usize>,
}

// ─── KdTree ──────────────────────────────────────────────────────────────────

/// Median-split k-d tree over `f64` feature vectors.
pub struct KdTree {
    pub root: Option<KdNode>,
    /// Number of training points.
    pub n: usize,
    /// Feature dimensionality.
    pub d: usize,
    /// Flat `[n × d]` row-major training data (owned copy).
    pub data: Vec<f64>,
}

// ─── Build ────────────────────────────────────────────────────────────────────

/// Build a k-d tree from flat `[n × d]` row-major data.
pub fn kd_build(data: &[f64], n: usize, d: usize) -> KdTree {
    if n == 0 || d == 0 {
        return KdTree {
            root: None,
            n,
            d,
            data: data.to_vec(),
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let root = kd_build_recursive(data, &mut indices, d, 0);
    KdTree {
        root: Some(root),
        n,
        d,
        data: data.to_vec(),
    }
}

fn kd_build_recursive(data: &[f64], indices: &mut [usize], d: usize, depth: usize) -> KdNode {
    if indices.len() == 1 {
        return KdNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            point_idx: Some(indices[0]),
        };
    }

    let feature = depth % d;

    // Sort indices by the chosen feature dimension
    indices.sort_unstable_by(|&a, &b| {
        let va = data[a * d + feature];
        let vb = data[b * d + feature];
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let median_pos = indices.len() / 2;
    let threshold = data[indices[median_pos] * d + feature];

    let (left_idxs, right_idxs) = indices.split_at_mut(median_pos);

    let left_node = if left_idxs.is_empty() {
        None
    } else {
        Some(Box::new(kd_build_recursive(data, left_idxs, d, depth + 1)))
    };

    let right_node = if right_idxs.is_empty() {
        None
    } else {
        Some(Box::new(kd_build_recursive(data, right_idxs, d, depth + 1)))
    };

    KdNode {
        feature,
        threshold,
        left: left_node,
        right: right_node,
        point_idx: None,
    }
}

// ─── KNN Query ───────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Nearest-neighbour heap entry: (negative squared distance, index).
///
/// We use a max-heap on `-dist²` so the "worst" (largest) found distance is
/// always at the top, letting us prune branches efficiently.
struct NnHeap {
    /// `(sq_dist, idx)` pairs; max-heap by `sq_dist`.
    inner: Vec<(f64, usize)>,
    k: usize,
}

impl NnHeap {
    fn new(k: usize) -> Self {
        Self {
            inner: Vec::with_capacity(k + 1),
            k,
        }
    }

    /// Worst (largest) distance in the heap; or f64::INFINITY if unfull.
    fn worst_sq(&self) -> f64 {
        if self.inner.len() < self.k {
            f64::INFINITY
        } else {
            self.inner
                .iter()
                .map(|(d, _)| *d)
                .fold(f64::NEG_INFINITY, f64::max)
        }
    }

    /// Insert a candidate; if heap is full and candidate is better than worst, replace.
    fn push(&mut self, sq: f64, idx: usize) {
        if self.inner.len() < self.k {
            self.inner.push((sq, idx));
        } else {
            // Find and replace the worst element
            if let Some(pos) = self
                .inner
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                && sq < self.inner[pos].0
            {
                self.inner[pos] = (sq, idx);
            }
        }
    }

    /// Drain as `(idx, dist)` sorted by distance ascending.
    fn into_sorted(mut self) -> Vec<(usize, f64)> {
        self.inner
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.inner
            .into_iter()
            .map(|(sq, i)| (i, sq.sqrt()))
            .collect()
    }
}

/// Recursive k-NN search in the k-d tree.
fn kd_search(
    node: &KdNode,
    query: &[f64],
    data: &[f64],
    d: usize,
    heap: &mut NnHeap,
    exclude_idx: Option<usize>,
) {
    if let Some(idx) = node.point_idx {
        // Leaf
        if exclude_idx != Some(idx) {
            let sq = sq_dist(query, &data[idx * d..(idx + 1) * d]);
            heap.push(sq, idx);
        }
        return;
    }

    // Internal node: choose near/far subtree
    let diff = query[node.feature] - node.threshold;
    let (near, far) = if diff <= 0.0 {
        (node.left.as_deref(), node.right.as_deref())
    } else {
        (node.right.as_deref(), node.left.as_deref())
    };

    if let Some(near_node) = near {
        kd_search(near_node, query, data, d, heap, exclude_idx);
    }

    // Only visit far side if it could contain closer points
    let plane_dist_sq = diff * diff;
    if plane_dist_sq <= heap.worst_sq()
        && let Some(far_node) = far
    {
        kd_search(far_node, query, data, d, heap, exclude_idx);
    }
}

/// Query the `k` nearest neighbours of `query` in the k-d tree.
///
/// Returns `Vec<(index, distance)>` sorted by distance ascending.
/// If `exclude_self` is true, the query itself (if it appears in the training set)
/// is skipped by matching on index equality during leaf visits — callers should
/// pass the known self-index when querying training points.
pub fn kd_knn(tree: &KdTree, query: &[f64], k: usize) -> Vec<(usize, f64)> {
    kd_knn_ex(tree, query, k, None)
}

/// Like `kd_knn` but optionally excludes point `exclude_idx` (for self-query).
pub fn kd_knn_ex(
    tree: &KdTree,
    query: &[f64],
    k: usize,
    exclude_idx: Option<usize>,
) -> Vec<(usize, f64)> {
    if tree.n == 0 || k == 0 {
        return Vec::new();
    }
    let mut heap = NnHeap::new(k);
    if let Some(root) = &tree.root {
        kd_search(root, query, &tree.data, tree.d, &mut heap, exclude_idx);
    }
    heap.into_sorted()
}

// ─── LofKdConfig ─────────────────────────────────────────────────────────────

/// Configuration for k-d tree accelerated LOF.
#[derive(Debug, Clone)]
pub struct LofKdConfig {
    /// Number of nearest neighbours.
    pub k: usize,
}

impl Default for LofKdConfig {
    fn default() -> Self {
        Self { k: 5 }
    }
}

// ─── LofKdFit ─────────────────────────────────────────────────────────────────

/// Fitted LOF-kdTree model.
pub struct LofKdFit {
    pub tree: KdTree,
    /// k-distance for each training point (distance to the k-th nearest neighbour).
    /// `knn_dists[i]` = k-dist of training point i.
    pub knn_dists: Vec<f64>,
    /// Local reachability density for each training point.
    pub lrd: Vec<f64>,
    /// k nearest-neighbour indices per training point: flat `[n × k]`.
    pub knn_indices: Vec<usize>,
    pub n: usize,
    pub d: usize,
    pub k: usize,
}

// ─── Fit ─────────────────────────────────────────────────────────────────────

/// Build a k-d tree accelerated LOF on training data `x` (`[n × d]` row-major).
pub fn lof_kd_fit(x: &[f64], n: usize, d: usize, cfg: &LofKdConfig) -> AnomalyResult<LofKdFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if cfg.k == 0 {
        return Err(AnomalyError::InvalidK { k: 0 });
    }
    if n <= cfg.k {
        return Err(AnomalyError::InsufficientSamples {
            need: cfg.k + 1,
            got: n,
        });
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let k = cfg.k;

    // Build k-d tree
    let tree = kd_build(x, n, d);

    // For each training point, find k nearest neighbours (excluding self)
    let mut knn_indices = vec![0_usize; n * k];
    let mut knn_dists_flat = vec![0.0_f64; n * k];

    for i in 0..n {
        let query = &x[i * d..(i + 1) * d];
        let neighbours = kd_knn_ex(&tree, query, k, Some(i));
        // Pad with last neighbour if tree returned fewer (shouldn't happen)
        for ki in 0..k {
            if ki < neighbours.len() {
                knn_indices[i * k + ki] = neighbours[ki].0;
                knn_dists_flat[i * k + ki] = neighbours[ki].1;
            } else if !neighbours.is_empty() {
                let last = neighbours.last().unwrap();
                knn_indices[i * k + ki] = last.0;
                knn_dists_flat[i * k + ki] = last.1;
            }
        }
    }

    // k-distance of point i = knn_dists_flat[i * k + (k-1)]
    let knn_dists: Vec<f64> = (0..n).map(|i| knn_dists_flat[i * k + k - 1]).collect();

    // Compute lrd for each training point
    // lrd_k(i) = k / Σ_{o ∈ N_k(i)} reach_dist_k(i, o)
    // reach_dist_k(i, o) = max(k-dist(o), dist(i, o))
    let mut lrd = vec![0.0_f64; n];
    for i in 0..n {
        let query = &x[i * d..(i + 1) * d];
        let mut sum_reach = 0.0_f64;
        for ki in 0..k {
            let j = knn_indices[i * k + ki];
            let kd_j = knn_dists[j]; // k-distance of neighbour j
            // Re-compute exact distance i→j
            let pt_j = &x[j * d..(j + 1) * d];
            let dist_ij = query
                .iter()
                .zip(pt_j.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let reach = kd_j.max(dist_ij).max(0.0);
            sum_reach += reach;
        }
        lrd[i] = if sum_reach < 1e-15 {
            f64::INFINITY
        } else {
            k as f64 / sum_reach
        };
    }

    Ok(LofKdFit {
        tree,
        knn_dists,
        lrd,
        knn_indices,
        n,
        d,
        k,
    })
}

// ─── Score ────────────────────────────────────────────────────────────────────

/// Compute LOF scores for `n` test samples (`x` is `[n × d]`).
///
/// LOF >> 1.0 indicates an outlier; LOF ≈ 1.0 indicates normality.
pub fn lof_kd_score(fit: &LofKdFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if x.len() != n * fit.d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * fit.d,
            got: x.len(),
        });
    }

    let k = fit.k;
    let d = fit.d;
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let query = &x[i * d..(i + 1) * d];

        // Find k nearest training neighbours for this query point
        let neighbours = kd_knn(&fit.tree, query, k);
        if neighbours.len() < k {
            // Fallback: use whatever we have
            scores.push(1.0_f64);
            continue;
        }

        // lrd of query point x_q
        let mut sum_reach_q = 0.0_f64;
        for &(j, dist_qj) in &neighbours {
            let kd_j = fit.knn_dists[j]; // k-distance of training point j
            let reach = kd_j.max(dist_qj).max(0.0);
            sum_reach_q += reach;
        }
        let lrd_q = if sum_reach_q < 1e-15 {
            f64::INFINITY
        } else {
            k as f64 / sum_reach_q
        };

        // LOF = 0 if query is inside a dense cluster
        if lrd_q.is_infinite() {
            scores.push(0.0);
            continue;
        }

        // LOF = (1/k) Σ_{o ∈ N_k(q)} lrd(o) / lrd_q
        let lof: f64 = neighbours
            .iter()
            .map(|&(j, _)| fit.lrd[j] / lrd_q)
            .sum::<f64>()
            / k as f64;

        scores.push(lof);
    }

    Ok(scores)
}

/// Predict anomaly labels for `n` test samples.
///
/// Returns `true` if LOF > `threshold` (anomaly).
pub fn lof_kd_predict(
    fit: &LofKdFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = lof_kd_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 0.1).collect()
    }

    #[test]
    fn test_kd_build_nonempty() {
        let data = line_data(20);
        let tree = kd_build(&data, 20, 1);
        assert_eq!(tree.n, 20);
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_kd_knn_returns_k_neighbours() {
        let data = line_data(20);
        let tree = kd_build(&data, 20, 1);
        let neighbours = kd_knn(&tree, &[1.0_f64], 5);
        assert_eq!(neighbours.len(), 5, "should return exactly k=5 neighbours");
    }

    #[test]
    fn test_kd_knn_sorted_ascending() {
        let data = line_data(30);
        let tree = kd_build(&data, 30, 1);
        let neighbours = kd_knn(&tree, &[1.5_f64], 8);
        for i in 1..neighbours.len() {
            assert!(
                neighbours[i].1 >= neighbours[i - 1].1,
                "kNN distances not sorted: {:?}",
                neighbours
            );
        }
    }

    #[test]
    fn test_kd_knn_2d() {
        // 2D grid of points
        let mut data = Vec::new();
        for i in 0..5_usize {
            for j in 0..5_usize {
                data.push(i as f64);
                data.push(j as f64);
            }
        }
        let tree = kd_build(&data, 25, 2);
        let query = [2.0_f64, 2.0_f64]; // centre of grid
        let neighbours = kd_knn(&tree, &query, 4);
        assert_eq!(neighbours.len(), 4);
    }

    #[test]
    fn test_kd_knn_exclude_self() {
        let data = line_data(10);
        let tree = kd_build(&data, 10, 1);
        // Query at the position of point 5 (0.5), excluding index 5
        let neighbours = kd_knn_ex(&tree, &[0.5_f64], 3, Some(5));
        assert!(
            neighbours.iter().all(|(idx, _)| *idx != 5),
            "self excluded but appeared: {:?}",
            neighbours
        );
    }

    #[test]
    fn test_lof_kd_fit_basic() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        assert_eq!(fit.n, 20);
        assert_eq!(fit.d, 1);
        assert_eq!(fit.lrd.len(), 20);
    }

    #[test]
    fn test_lof_kd_score_length() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        let test: Vec<f64> = vec![0.5_f64, 1.0, 1.5];
        let scores = lof_kd_score(&fit, &test, 3).unwrap();
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_lof_kd_scores_finite() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        let test: Vec<f64> = vec![0.5_f64];
        let scores = lof_kd_score(&fit, &test, 1).unwrap();
        assert!(scores[0].is_finite(), "lof score not finite: {}", scores[0]);
    }

    #[test]
    fn test_lof_kd_scores_non_negative() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        let test: Vec<f64> = vec![0.3_f64, 0.7, 1.1];
        let scores = lof_kd_score(&fit, &test, 3).unwrap();
        for &s in &scores {
            assert!(s >= 0.0, "negative score: {s}");
        }
    }

    #[test]
    fn test_lof_kd_outlier_higher_score() {
        // Dense cluster of 30 points near 0, then one outlier far away
        let mut data: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
        // Add a distant "training outlier"
        data.push(100.0_f64);
        let n = 31_usize;
        let cfg = LofKdConfig { k: 5 };
        let fit = lof_kd_fit(&data, n, 1, &cfg).unwrap();

        let test_normal = vec![0.15_f64]; // inside cluster
        let test_outlier = vec![200.0_f64]; // far from cluster

        let s_normal = lof_kd_score(&fit, &test_normal, 1).unwrap()[0];
        let s_outlier = lof_kd_score(&fit, &test_outlier, 1).unwrap()[0];

        assert!(
            s_outlier > s_normal,
            "outlier score {s_outlier} should > normal score {s_normal}"
        );
    }

    #[test]
    fn test_lof_kd_predict_length() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        let test: Vec<f64> = vec![0.5_f64, 1.2, 2.5];
        let preds = lof_kd_predict(&fit, &test, 3, 1.5).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_lof_kd_empty_input_error() {
        let cfg = LofKdConfig { k: 3 };
        let res = lof_kd_fit(&[], 0, 1, &cfg);
        assert!(res.is_err(), "expected EmptyInput error");
    }

    #[test]
    fn test_lof_kd_insufficient_samples_error() {
        let data = vec![0.0_f64, 1.0, 2.0];
        let cfg = LofKdConfig { k: 5 }; // k=5 but n=3
        let res = lof_kd_fit(&data, 3, 1, &cfg);
        assert!(res.is_err(), "expected InsufficientSamples error");
    }

    #[test]
    fn test_lof_kd_dimension_mismatch_error() {
        let data = line_data(20);
        let cfg = LofKdConfig { k: 3 };
        let fit = lof_kd_fit(&data, 20, 1, &cfg).unwrap();
        // Pass 2D test data but fit is 1D
        let bad_test = vec![0.5_f64, 0.5]; // len=2 but expected 1
        let res = lof_kd_score(&fit, &bad_test, 1);
        assert!(res.is_err(), "expected DimensionMismatch error");
    }
}
