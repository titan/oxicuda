//! Extended Isolation Forest (Hariri et al. 2018).
//!
//! Unlike the standard Isolation Forest which uses axis-aligned splits, the
//! Extended Isolation Forest uses random hyperplane splits parameterised by a
//! normal vector **n** ∈ ℝ^d and an intercept *p*.
//!
//! The `extension_level` parameter controls how many feature dimensions
//! contribute to the random normal vector:
//! - `extension_level = 1`  →  axis-aligned splits  (≈ standard iForest)
//! - `extension_level = d`  →  full hyperplane splits (default)
//!
//! # Scoring
//!
//! ```text
//! E[h(x)] = average path length across all trees
//! s(x) = 2^{ -E[h(x)] / c(max_samples) }
//! ```
//!
//! Higher `s(x)` → more anomalous (score 1.0 = perfect isolation).
//!
//! # Reference
//! Hariri, S., Kind, M. C., & Brunner, R. J. (2019). Extended Isolation Forest.
//! *IEEE TNNLS*, 32(10), 4761-4775.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Euler-Mascheroni constant ────────────────────────────────────────────────

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the Extended Isolation Forest.
#[derive(Debug, Clone)]
pub struct ExtIforestConfig {
    /// Number of isolation trees (default: 100).
    pub n_estimators: usize,
    /// Sub-sample size drawn from training data per tree (default: 256).
    pub max_samples: usize,
    /// Expected fraction of outliers in `(0, 0.5)` (default: 0.1).
    pub contamination: f64,
    /// Number of feature dimensions used per hyperplane normal vector.
    /// `1` = axis-aligned (standard iForest); `d` = full hyperplane (default).
    /// Values larger than `d` are silently clamped to `d`.
    pub extension_level: usize,
    /// RNG seed for reproducibility (default: 42).
    pub random_seed: u64,
}

impl Default for ExtIforestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: 0.1,
            extension_level: usize::MAX, // sentinel → d (all dims)
            random_seed: 42,
        }
    }
}

// ─── Node ─────────────────────────────────────────────────────────────────────

/// A single node in an Extended Isolation Tree.
#[derive(Debug, Clone)]
pub struct ExtNode {
    /// Whether this is a leaf node.
    pub is_leaf: bool,
    /// Number of training points that reached this node.
    pub size: usize,
    /// Depth of this node in the tree (root = 0).
    pub depth: usize,
    /// Random hyperplane normal vector (length = `n_features`; zero-padded
    /// for non-sampled dimensions).  Unit-normalised.
    pub normal: Vec<f64>,
    /// Hyperplane intercept.  The split criterion is `x · normal ≤ intercept`.
    pub intercept: f64,
    /// Index into the tree's node array of the left child.  `usize::MAX` if leaf.
    pub left_child: usize,
    /// Index into the tree's node array of the right child.  `usize::MAX` if leaf.
    pub right_child: usize,
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted Extended Isolation Forest model.
pub struct ExtIforestModel {
    /// Collection of trees; each tree is a flat `Vec<ExtNode>` (root at index 0).
    pub trees: Vec<Vec<ExtNode>>,
    /// Number of features the model was fitted on.
    pub n_features: usize,
    /// Configuration used during fitting.
    pub config: ExtIforestConfig,
    /// Anomaly score threshold corresponding to the contamination percentile.
    /// Scores ≥ `threshold` are labelled anomalous (−1).
    pub threshold: f64,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Average path length of an unsuccessful Binary Search Tree search: `c(n)`.
///
/// `c(n) = 2·(ln(n−1) + γ) − 2·(n−1)/n`   for `n ≥ 2`.
/// Special cases: `c(0) = c(1) = 0`, `c(2) = 1`.
#[inline]
fn c_factor(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            let nm1 = (n - 1) as f64;
            2.0 * (nm1.ln() + EULER_MASCHERONI) - 2.0 * nm1 / n as f64
        }
    }
}

/// Sample `k` unique indices from `[0, n)` into `out`, using partial Fisher-Yates.
///
/// `scratch` is a temporary buffer of length `n` (reused to avoid allocations).
fn sample_without_replacement(n: usize, k: usize, rng: &mut LcgRng, scratch: &mut Vec<usize>) {
    scratch.clear();
    scratch.extend(0..n);
    for i in 0..k.min(n) {
        let j = i + rng.next_usize(n - i);
        scratch.swap(i, j);
    }
    // scratch[0..k] holds the sampled indices
}

/// Sample a standard normal value using Box-Muller (f64 precision).
#[inline]
fn next_normal_f64(rng: &mut LcgRng) -> f64 {
    // Reuse LcgRng's f32 Box-Muller and upcast; sufficient for direction sampling.
    rng.next_normal() as f64
}

/// Sample a uniform value in `[lo, hi)`.
#[inline]
fn uniform_f64(lo: f64, hi: f64, rng: &mut LcgRng) -> f64 {
    let u = (rng.next_u32() as f64) / (u32::MAX as f64 + 1.0);
    lo + u * (hi - lo)
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Tree building ─────────────────────────────────────────────────────────────

/// Build one Extended Isolation Tree from a row-major data subsample
/// of shape `[n × d]`.  Returns the flat node array (root at index 0).
fn build_tree(
    data: &[f64],
    n: usize,
    d: usize,
    max_depth: usize,
    ext_level: usize,
    rng: &mut LcgRng,
) -> Vec<ExtNode> {
    // Pre-allocate; typical tree has ≤ 2·n nodes.
    let mut nodes: Vec<ExtNode> = Vec::with_capacity(2 * n + 1);

    // Stack: (node_index, sample_indices, depth)
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::with_capacity(2 * n);

    // Root: all indices
    let root_indices: Vec<usize> = (0..n).collect();
    // Push placeholder root node
    nodes.push(ExtNode {
        is_leaf: true,
        size: n,
        depth: 0,
        normal: vec![0.0; d],
        intercept: 0.0,
        left_child: usize::MAX,
        right_child: usize::MAX,
    });
    stack.push((0, root_indices, 0));

    // Scratch buffer for index sampling (avoids per-node allocation)
    let mut scratch: Vec<usize> = Vec::with_capacity(d);

    while let Some((node_idx, indices, depth)) = stack.pop() {
        let size = indices.len();

        // Base cases: leaf
        if size <= 1 || depth >= max_depth {
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].size = size;
            nodes[node_idx].depth = depth;
            // normal/intercept/children remain at zero/MAX (leaf defaults)
            continue;
        }

        // ── Sample hyperplane normal ─────────────────────────────────────────
        let actual_ext = ext_level.min(d);
        sample_without_replacement(d, actual_ext, rng, &mut scratch);

        let mut normal = vec![0.0_f64; d];
        for &feat_idx in scratch.iter().take(actual_ext) {
            normal[feat_idx] = next_normal_f64(rng);
        }

        // Normalise
        let norm_sq: f64 = normal.iter().map(|v| v * v).sum();
        if norm_sq < 1e-15 {
            // Degenerate: default to first axis
            normal[0] = 1.0;
        } else {
            let inv_norm = norm_sq.sqrt().recip();
            for v in &mut normal {
                *v *= inv_norm;
            }
        }

        // ── Project data onto normal ─────────────────────────────────────────
        let projections: Vec<f64> = indices
            .iter()
            .map(|&i| {
                let row = &data[i * d..(i + 1) * d];
                dot(row, &normal)
            })
            .collect();

        let proj_min = projections.iter().cloned().fold(f64::INFINITY, f64::min);
        let proj_max = projections
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // If all projections are identical → leaf (cannot split)
        if (proj_max - proj_min).abs() < 1e-14 {
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].size = size;
            nodes[node_idx].depth = depth;
            continue;
        }

        // ── Sample intercept uniformly over projected range ──────────────────
        let intercept = uniform_f64(proj_min, proj_max, rng);

        // ── Partition ───────────────────────────────────────────────────────
        let mut left_indices: Vec<usize> = Vec::with_capacity(size / 2 + 1);
        let mut right_indices: Vec<usize> = Vec::with_capacity(size / 2 + 1);
        for (local_idx, &global_idx) in indices.iter().enumerate() {
            if projections[local_idx] <= intercept {
                left_indices.push(global_idx);
            } else {
                right_indices.push(global_idx);
            }
        }

        // Guard: if the split is degenerate (all one side), treat as leaf.
        if left_indices.is_empty() || right_indices.is_empty() {
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].size = size;
            nodes[node_idx].depth = depth;
            continue;
        }

        // ── Allocate child nodes ─────────────────────────────────────────────
        let left_idx = nodes.len();
        nodes.push(ExtNode {
            is_leaf: true,
            size: 0,
            depth: depth + 1,
            normal: vec![0.0; d],
            intercept: 0.0,
            left_child: usize::MAX,
            right_child: usize::MAX,
        });
        let right_idx = nodes.len();
        nodes.push(ExtNode {
            is_leaf: true,
            size: 0,
            depth: depth + 1,
            normal: vec![0.0; d],
            intercept: 0.0,
            left_child: usize::MAX,
            right_child: usize::MAX,
        });

        // Update current node
        nodes[node_idx].is_leaf = false;
        nodes[node_idx].size = size;
        nodes[node_idx].depth = depth;
        nodes[node_idx].normal = normal;
        nodes[node_idx].intercept = intercept;
        nodes[node_idx].left_child = left_idx;
        nodes[node_idx].right_child = right_idx;

        // Push children onto the stack
        stack.push((left_idx, left_indices, depth + 1));
        stack.push((right_idx, right_indices, depth + 1));
    }

    nodes
}

/// Compute the path length of a single query through one tree, including the
/// c-factor leaf correction.
fn path_length_in_tree(tree: &[ExtNode], x: &[f64], d: usize) -> f64 {
    let mut node_idx = 0usize;
    loop {
        let node = &tree[node_idx];
        if node.is_leaf {
            // Path length = depth + c(leaf_size)
            return node.depth as f64 + c_factor(node.size);
        }
        // Traverse
        let proj = dot(x, &node.normal[..d]);
        if proj <= node.intercept {
            node_idx = node.left_child;
        } else {
            node_idx = node.right_child;
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit an Extended Isolation Forest to training data.
///
/// `x` is a row-major `f64` slice of shape `[n_samples × n_features]`.
pub fn ext_iforest_fit(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &ExtIforestConfig,
) -> AnomalyResult<ExtIforestModel> {
    // ── Validation ──────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if n_features == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if x.len() != n_samples * n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * n_features,
            got: x.len(),
        });
    }
    if cfg.n_estimators == 0 {
        return Err(AnomalyError::Internal {
            msg: "n_estimators must be > 0".into(),
        });
    }
    if cfg.max_samples == 0 {
        return Err(AnomalyError::Internal {
            msg: "max_samples must be > 0".into(),
        });
    }
    if !(cfg.contamination > 0.0 && cfg.contamination < 0.5) {
        return Err(AnomalyError::Internal {
            msg: format!(
                "contamination must be in (0, 0.5), got {}",
                cfg.contamination
            ),
        });
    }

    let actual_max_samples = cfg.max_samples.min(n_samples);
    let actual_ext = if cfg.extension_level == usize::MAX {
        n_features
    } else {
        cfg.extension_level.clamp(1, n_features)
    };
    let max_depth = {
        // ceil(log2(max_samples))
        let ms = actual_max_samples;
        if ms <= 1 {
            1
        } else {
            (ms as f64).log2().ceil() as usize
        }
    };

    let mut rng = LcgRng::new(cfg.random_seed);
    let mut trees: Vec<Vec<ExtNode>> = Vec::with_capacity(cfg.n_estimators);
    let mut idx_scratch: Vec<usize> = Vec::with_capacity(n_samples);

    for _ in 0..cfg.n_estimators {
        // Sub-sample without replacement using partial Fisher-Yates
        idx_scratch.clear();
        idx_scratch.extend(0..n_samples);
        for k in 0..actual_max_samples {
            let j = k + rng.next_usize(n_samples - k);
            idx_scratch.swap(k, j);
        }
        let sample_indices: Vec<usize> = idx_scratch[..actual_max_samples].to_vec();

        // Build dense sub-sample array
        let mut sub: Vec<f64> = Vec::with_capacity(actual_max_samples * n_features);
        for &i in &sample_indices {
            sub.extend_from_slice(&x[i * n_features..(i + 1) * n_features]);
        }

        let tree = build_tree(
            &sub,
            actual_max_samples,
            n_features,
            max_depth,
            actual_ext,
            &mut rng,
        );
        trees.push(tree);
    }

    // ── Compute anomaly scores on training data to derive threshold ───────
    let c = c_factor(actual_max_samples);
    let scores = compute_scores_inner(&trees, x, n_samples, n_features, c);

    // Sort ascending; threshold = (1 - contamination) quantile
    let mut sorted_scores = scores.clone();
    sorted_scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold_rank = ((1.0 - cfg.contamination) * n_samples as f64).floor() as usize;
    let threshold_rank = threshold_rank.min(n_samples - 1);
    let threshold = sorted_scores[threshold_rank];

    Ok(ExtIforestModel {
        trees,
        n_features,
        config: ExtIforestConfig {
            extension_level: actual_ext,
            ..cfg.clone()
        },
        threshold,
    })
}

/// Compute anomaly scores (inner, shared by fit and score).
fn compute_scores_inner(
    trees: &[Vec<ExtNode>],
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    c: f64,
) -> Vec<f64> {
    let n_trees = trees.len() as f64;
    (0..n_samples)
        .map(|i| {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let avg_path: f64 = trees
                .iter()
                .map(|tree| path_length_in_tree(tree, xi, n_features))
                .sum::<f64>()
                / n_trees;
            if c < 1e-10 {
                0.5
            } else {
                2.0_f64.powf(-avg_path / c)
            }
        })
        .collect()
}

/// Compute anomaly scores ∈ (0, 1) for each sample in `x`.
///
/// Higher scores indicate stronger anomaly.
///
/// `x` is row-major `[n_samples × n_features]`.
pub fn ext_iforest_score(
    model: &ExtIforestModel,
    x: &[f64],
    n_samples: usize,
) -> AnomalyResult<Vec<f64>> {
    if model.trees.is_empty() {
        return Err(AnomalyError::NotFitted);
    }
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n_samples * model.n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * model.n_features,
            got: x.len(),
        });
    }
    let actual_max = model.config.max_samples.min(
        // We don't store actual_max_samples separately; derive from c_factor
        // by using max_samples from config.
        model.config.max_samples,
    );
    let c = c_factor(actual_max);
    let scores = compute_scores_inner(&model.trees, x, n_samples, model.n_features, c);
    Ok(scores)
}

/// Classify each sample as normal (`+1`) or anomaly (`-1`).
///
/// Samples with anomaly score ≥ `model.threshold` are labelled `−1`.
pub fn ext_iforest_predict(
    model: &ExtIforestModel,
    x: &[f64],
    n_samples: usize,
) -> AnomalyResult<Vec<i32>> {
    let scores = ext_iforest_score(model, x, n_samples)?;
    Ok(scores
        .into_iter()
        .map(|s| if s >= model.threshold { -1 } else { 1 })
        .collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tight inlier cluster (Gaussian noise around zero) plus an outlier.
    fn make_inliers_and_outlier(n_inliers: usize, d: usize, seed: u64) -> (Vec<f64>, usize, usize) {
        let mut rng = LcgRng::new(seed);
        let mut data: Vec<f64> = Vec::with_capacity((n_inliers + 1) * d);
        for _ in 0..n_inliers {
            for _ in 0..d {
                // Cluster at origin with std ≈ 0.1
                data.push(rng.next_normal() as f64 * 0.1);
            }
        }
        // Outlier at (100, 100, …)
        data.extend(std::iter::repeat_n(100.0, d));
        let n = n_inliers + 1;
        (data, n, d)
    }

    // ── Test 1: fit runs without error on 100×4 data ──────────────────────────
    #[test]
    fn ext_iforest_fit_runs() {
        let (data, n, d) = make_inliers_and_outlier(100, 4, 1);
        let cfg = ExtIforestConfig {
            n_estimators: 50,
            max_samples: 64,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 1,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg);
        assert!(model.is_ok(), "fit failed: {:?}", model.err());
    }

    // ── Test 2: scores has length n_samples ──────────────────────────────────
    #[test]
    fn ext_iforest_score_shape() {
        let (data, n, d) = make_inliers_and_outlier(99, 4, 2);
        let cfg = ExtIforestConfig::default();
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();
        assert_eq!(scores.len(), n);
    }

    // ── Test 3: all scores in (0, 1) ─────────────────────────────────────────
    #[test]
    fn ext_iforest_scores_range() {
        let (data, n, d) = make_inliers_and_outlier(99, 4, 3);
        let cfg = ExtIforestConfig {
            n_estimators: 100,
            max_samples: 64,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 3,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();
        for (i, &s) in scores.iter().enumerate() {
            assert!(s > 0.0 && s < 1.0, "score[{i}] = {s} not in (0, 1)");
        }
    }

    // ── Test 4: isolated point scores higher than inlier cluster ─────────────
    #[test]
    fn ext_iforest_outlier_higher_score() {
        let n_inliers = 99usize;
        let d = 4usize;
        let (data, n, _d) = make_inliers_and_outlier(n_inliers, d, 4);
        let cfg = ExtIforestConfig {
            n_estimators: 200,
            max_samples: 64,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 4,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();

        let outlier_score = scores[n_inliers]; // last point = outlier
        let inlier_mean: f64 = scores[..n_inliers].iter().sum::<f64>() / n_inliers as f64;
        assert!(
            outlier_score > inlier_mean,
            "outlier {outlier_score:.4} should > inlier mean {inlier_mean:.4}"
        );
    }

    // ── Test 5: labels ∈ {-1, +1} ────────────────────────────────────────────
    #[test]
    fn ext_iforest_predict_labels() {
        let (data, n, d) = make_inliers_and_outlier(99, 4, 5);
        let cfg = ExtIforestConfig::default();
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let labels = ext_iforest_predict(&model, &data, n).unwrap();
        for &l in &labels {
            assert!(l == 1 || l == -1, "label={l} not in {{-1, +1}}");
        }
    }

    // ── Test 6: fraction of -1 labels ≈ contamination ────────────────────────
    #[test]
    fn ext_iforest_contamination_fraction() {
        let n_inliers = 190usize;
        let d = 4usize;
        let (data, n, _d) = make_inliers_and_outlier(n_inliers, d, 6);
        let contamination = 0.1;
        let cfg = ExtIforestConfig {
            n_estimators: 150,
            max_samples: 128,
            contamination,
            extension_level: usize::MAX,
            random_seed: 6,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let labels = ext_iforest_predict(&model, &data, n).unwrap();
        let n_anomalous = labels.iter().filter(|&&l| l == -1).count();
        let frac = n_anomalous as f64 / n as f64;
        // Allow ±5% tolerance
        assert!(
            (frac - contamination).abs() < 0.05,
            "anomaly fraction {frac:.3} not within 0.05 of contamination {contamination}"
        );
    }

    // ── Test 7: extension_level=1 resembles axis-aligned iForest ─────────────
    #[test]
    fn ext_iforest_extension_level_1() {
        let (data, n, d) = make_inliers_and_outlier(99, 4, 7);
        let cfg = ExtIforestConfig {
            n_estimators: 100,
            max_samples: 64,
            contamination: 0.1,
            extension_level: 1,
            random_seed: 7,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();
        // Must still produce valid scores
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|&s| s > 0.0 && s < 1.0));
        // Outlier should still score highest or near-highest
        let outlier_score = scores[n - 1];
        let max_inlier = scores[..n - 1]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            outlier_score >= max_inlier * 0.8,
            "axis-aligned: outlier {outlier_score:.4}, max inlier {max_inlier:.4}"
        );
    }

    // ── Test 8: max_samples=32 works on 50 points ─────────────────────────────
    #[test]
    fn ext_iforest_small_sample() {
        let (data, n, d) = make_inliers_and_outlier(49, 4, 8);
        let cfg = ExtIforestConfig {
            n_estimators: 50,
            max_samples: 32,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 8,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all scores must be finite"
        );
    }

    // ── Test 9: n_samples=1 returns a valid result ────────────────────────────
    #[test]
    fn ext_iforest_single_point_err() {
        // Single point: cannot split → all leaves → score should be 0.5 (c=0)
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let cfg = ExtIforestConfig {
            n_estimators: 10,
            max_samples: 256,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 9,
        };
        let model = ext_iforest_fit(&data, 1, 4, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, 1).unwrap();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite(), "score must be finite for n=1");
    }

    // ── Test 10: d=20, n=200 runs without panic ───────────────────────────────
    #[test]
    fn ext_iforest_high_dimensional() {
        let d = 20usize;
        let n_inliers = 199usize;
        let (data, n, _d) = make_inliers_and_outlier(n_inliers, d, 10);
        let cfg = ExtIforestConfig {
            n_estimators: 100,
            max_samples: 128,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 10,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let scores = ext_iforest_score(&model, &data, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all scores must be finite in 20-D"
        );
    }

    // ── Test 11: determinism — same seed yields identical scores ──────────────
    #[test]
    fn ext_iforest_deterministic() {
        let (data, n, d) = make_inliers_and_outlier(99, 4, 11);
        let cfg = ExtIforestConfig {
            n_estimators: 50,
            max_samples: 64,
            contamination: 0.1,
            extension_level: usize::MAX,
            random_seed: 11,
        };
        let m1 = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let m2 = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        let s1 = ext_iforest_score(&m1, &data, n).unwrap();
        let s2 = ext_iforest_score(&m2, &data, n).unwrap();
        for (i, (a, b)) in s1.iter().zip(s2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "score[{i}]: {a} vs {b} with same seed"
            );
        }
    }

    // ── Test 12: extension_level > d is clamped safely ────────────────────────
    #[test]
    fn ext_iforest_ext_level_clamped() {
        let (data, n, d) = make_inliers_and_outlier(49, 3, 12);
        let cfg = ExtIforestConfig {
            n_estimators: 30,
            max_samples: 32,
            contamination: 0.1,
            extension_level: 999, // much larger than d=3
            random_seed: 12,
        };
        let model = ext_iforest_fit(&data, n, d, &cfg).unwrap();
        assert_eq!(model.config.extension_level, 3, "should be clamped to d=3");
    }
}
