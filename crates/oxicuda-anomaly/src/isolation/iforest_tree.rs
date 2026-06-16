//! Explicit Isolation Forest with proper binary tree splits (Liu et al. 2008).
//!
//! Unlike the random-projection approximation in `iforest_score`, this module
//! builds real isolation trees by recursively partitioning training sub-samples.
//!
//! **Anomaly score**:
//! ```text
//! s(x) = 2^{ −avg_path_length(x) / c(subsample_size) }
//! ```
//! where `c(n) = 2·(ln(n−1) + γ) − 2·(n−1)/n` (harmonic correction factor).
//!
//! Points that are isolated quickly (short average path) receive scores near 1
//! (anomalous); points buried deep in the tree receive scores near 0 (normal).

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Euler–Mascheroni constant ────────────────────────────────────────────────

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86_f64;

// ─── c_factor (harmonic correction) ──────────────────────────────────────────

/// Average path length of a random unsuccessful BST search: `c(n)`.
///
/// `c(n) = 2·(ln(n−1) + γ) − 2·(n−1)/n`  for `n ≥ 2`.
///
/// * `c(0) = c(1) = 0`
/// * `c(2) = 1`
#[must_use]
pub fn ifor_c_factor(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            let nm1 = (n - 1) as f64;
            2.0 * (nm1.ln() + EULER_MASCHERONI) - 2.0 * nm1 / n as f64
        }
    }
}

// ─── IforNode ─────────────────────────────────────────────────────────────────

/// A node in an isolation tree.
///
/// Internal nodes store `(feature, threshold, left_child, right_child)`.
/// Leaf nodes have `left = right = None` and record the sub-sample `size`
/// for the c-factor adjustment.
#[derive(Debug, Clone)]
pub struct IforNode {
    /// Feature index used for splitting (internal nodes only).
    pub feature: usize,
    /// Split threshold; `x[feature] ≤ threshold` goes left.
    pub threshold: f64,
    /// Index into `IforTree::nodes` of the left child, or `None` for leaves.
    pub left: Option<usize>,
    /// Index into `IforTree::nodes` of the right child, or `None` for leaves.
    pub right: Option<usize>,
    /// Depth of this node in the tree (root = 0).
    pub depth: usize,
    /// Number of training sub-sample points that reached this node.
    pub size: usize,
}

// ─── IforTree ─────────────────────────────────────────────────────────────────

/// A single isolation tree (flat arena-allocated).
#[derive(Debug, Clone)]
pub struct IforTree {
    /// All nodes; `nodes[0]` is the root.
    pub nodes: Vec<IforNode>,
}

impl IforTree {
    /// Compute path length for a single test point `x` traversing this tree.
    ///
    /// Returns depth reached + c-factor correction for the leaf sub-sample size.
    #[must_use]
    pub fn path_length(&self, x: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut node_idx = 0_usize;
        loop {
            let node = &self.nodes[node_idx];
            match (node.left, node.right) {
                (Some(l), Some(r)) => {
                    if x[node.feature] <= node.threshold {
                        node_idx = l;
                    } else {
                        node_idx = r;
                    }
                }
                // Leaf: add c-factor correction for remaining sub-sample
                _ => {
                    return node.depth as f64 + ifor_c_factor(node.size);
                }
            }
        }
    }
}

// ─── IforConfig ───────────────────────────────────────────────────────────────

/// Configuration for the explicit Isolation Forest.
#[derive(Debug, Clone)]
pub struct IforConfig {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// Sub-sample size per tree (typically 256).
    pub subsample_size: usize,
    /// Maximum tree depth (default: `ceil(log2(subsample_size))`).
    pub max_depth: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl IforConfig {
    /// Build with default `max_depth = ceil(log2(subsample_size))`.
    pub fn new(n_trees: usize, subsample_size: usize, seed: u64) -> Self {
        let max_depth = if subsample_size <= 1 {
            1
        } else {
            (subsample_size as f64).log2().ceil() as usize
        };
        Self {
            n_trees,
            subsample_size,
            max_depth,
            seed,
        }
    }
}

// ─── IforFit ──────────────────────────────────────────────────────────────────

/// Fitted Isolation Forest model.
#[derive(Debug, Clone)]
pub struct IforFit {
    /// The ensemble of isolation trees.
    pub trees: Vec<IforTree>,
    /// Number of training samples used (full dataset, not per-tree sub-sample).
    pub n_train: usize,
    /// Feature dimensionality.
    pub d: usize,
    /// Configuration used at fit time.
    pub cfg: IforConfig,
}

// ─── Tree building ────────────────────────────────────────────────────────────

/// Fisher-Yates partial shuffle to sample `k` indices without replacement.
fn sample_without_replacement(n: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

/// Recursively build an isolation tree stored in the flat `nodes` arena.
///
/// Returns the index of the root node in `nodes`.
fn grow_tree(
    data: &[f64],
    d: usize,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    nodes: &mut Vec<IforNode>,
    rng: &mut LcgRng,
) -> usize {
    let size = indices.len();
    // Leaf condition: depth limit reached or only one point
    if depth >= max_depth || size <= 1 {
        let node_idx = nodes.len();
        nodes.push(IforNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            depth,
            size,
        });
        return node_idx;
    }

    // Choose a random feature
    let feat = rng.next_usize(d);

    // Find min/max of chosen feature among current indices
    let mut feat_min = f64::INFINITY;
    let mut feat_max = f64::NEG_INFINITY;
    for &idx in indices {
        let v = data[idx * d + feat];
        if v < feat_min {
            feat_min = v;
        }
        if v > feat_max {
            feat_max = v;
        }
    }

    // If all values identical, force a leaf
    if (feat_max - feat_min).abs() < 1e-15 {
        let node_idx = nodes.len();
        nodes.push(IforNode {
            feature: feat,
            threshold: feat_min,
            left: None,
            right: None,
            depth,
            size,
        });
        return node_idx;
    }

    // Random split threshold in [feat_min, feat_max)
    let t = feat_min + rng.next_f32() as f64 * (feat_max - feat_min);

    // Partition indices
    let left_indices: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| data[i * d + feat] <= t)
        .collect();
    let right_indices: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| data[i * d + feat] > t)
        .collect();

    // Guard against degenerate splits (all go one way)
    if left_indices.is_empty() || right_indices.is_empty() {
        let node_idx = nodes.len();
        nodes.push(IforNode {
            feature: feat,
            threshold: t,
            left: None,
            right: None,
            depth,
            size,
        });
        return node_idx;
    }

    // Reserve slot for this internal node
    let this_idx = nodes.len();
    nodes.push(IforNode {
        feature: feat,
        threshold: t,
        left: None,
        right: None,
        depth,
        size,
    });

    // Recurse left and right
    let left_child = grow_tree(data, d, &left_indices, depth + 1, max_depth, nodes, rng);
    let right_child = grow_tree(data, d, &right_indices, depth + 1, max_depth, nodes, rng);

    // Back-fill children indices
    nodes[this_idx].left = Some(left_child);
    nodes[this_idx].right = Some(right_child);

    this_idx
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit an explicit Isolation Forest.
///
/// # Arguments
/// * `x`   — row-major flat training matrix `[n × d]`.
/// * `n`   — number of training samples.
/// * `d`   — number of features.
/// * `cfg` — forest configuration.
pub fn ifor_fit(x: &[f64], n: usize, d: usize, cfg: &IforConfig) -> AnomalyResult<IforFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    if cfg.n_trees == 0 {
        return Err(AnomalyError::Internal {
            msg: "n_trees must be > 0".into(),
        });
    }

    let mut rng = LcgRng::new(cfg.seed);
    let ss = cfg.subsample_size.min(n); // clamp subsample to available data

    let mut trees = Vec::with_capacity(cfg.n_trees);
    for _ in 0..cfg.n_trees {
        let sub_indices = sample_without_replacement(n, ss, &mut rng);
        let mut nodes = Vec::new();
        grow_tree(x, d, &sub_indices, 0, cfg.max_depth, &mut nodes, &mut rng);
        trees.push(IforTree { nodes });
    }

    Ok(IforFit {
        trees,
        n_train: n,
        d,
        cfg: cfg.clone(),
    })
}

/// Compute the average path length for a single point across all trees.
pub fn ifor_path_length(fit: &IforFit, x: &[f64]) -> f64 {
    if fit.trees.is_empty() {
        return 0.0;
    }
    let total: f64 = fit.trees.iter().map(|t| t.path_length(x)).sum();
    total / fit.trees.len() as f64
}

/// Compute isolation scores for `n` test samples.
///
/// Returns scores in `[0, 1]`:
/// * `> 0.5` → anomaly (isolated quickly)
/// * `< 0.5` → normal (deeply embedded)
pub fn ifor_score(fit: &IforFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * fit.d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * fit.d,
            got: x.len(),
        });
    }

    let ss = fit.cfg.subsample_size.min(fit.n_train);
    let c_n = ifor_c_factor(ss);

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let sample = &x[i * fit.d..(i + 1) * fit.d];
        let avg_path = ifor_path_length(fit, sample);
        let score = if c_n < 1e-12 {
            0.5_f64
        } else {
            2.0_f64.powf(-avg_path / c_n)
        };
        scores.push(score.clamp(0.0, 1.0));
    }
    Ok(scores)
}

/// Predict binary anomaly labels for `n` test samples.
///
/// Returns `true` for samples whose isolation score exceeds `threshold`
/// (typical default: `0.5`).
pub fn ifor_predict(
    fit: &IforFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = ifor_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(n_trees: usize, subsample: usize, seed: u64) -> IforConfig {
        IforConfig::new(n_trees, subsample, seed)
    }

    #[test]
    fn c_factor_special_cases() {
        assert!((ifor_c_factor(0)).abs() < 1e-12);
        assert!((ifor_c_factor(1)).abs() < 1e-12);
        assert!((ifor_c_factor(2) - 1.0).abs() < 1e-12);
        assert!(ifor_c_factor(256) > 0.0);
    }

    #[test]
    fn ifor_fit_basic() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let cfg = make_cfg(10, 64, 42);
        let fit = ifor_fit(&data, 100, 2, &cfg).expect("ifor_fit should succeed with valid input");
        assert_eq!(fit.trees.len(), 10);
        assert_eq!(fit.d, 2);
    }

    #[test]
    fn ifor_score_in_range() {
        let n = 100_usize;
        let d = 2_usize;
        let data: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.01).collect();
        let cfg = make_cfg(20, 64, 7);
        let fit = ifor_fit(&data, n, d, &cfg).expect("ifor_fit should succeed with valid data");
        let scores = ifor_score(&fit, &data, n).expect("ifor_score should succeed on valid fit");
        assert_eq!(scores.len(), n);
        for s in &scores {
            assert!((0.0..=1.0).contains(s), "score={s}");
        }
    }

    #[test]
    fn ifor_outlier_has_higher_score() {
        let n = 200_usize;
        let d = 1_usize;
        // Dense cluster around 0.0
        let mut data: Vec<f64> = (0..n).map(|i| (i as f64 - 100.0) * 0.01).collect();
        let cfg = make_cfg(50, 128, 13);
        let fit = ifor_fit(&data, n, d, &cfg).expect("ifor_fit should succeed with cluster data");

        // Score a typical inlier and a far outlier
        let s_inlier =
            ifor_score(&fit, &[0.0_f64], 1).expect("ifor_score should succeed for inlier")[0];
        let s_outlier =
            ifor_score(&fit, &[1000.0_f64], 1).expect("ifor_score should succeed for outlier")[0];

        // Use data again to silence warning
        let _ = data.pop();

        assert!(
            s_outlier > s_inlier,
            "outlier={s_outlier} inlier={s_inlier}"
        );
    }

    #[test]
    fn ifor_predict_labels() {
        let n = 100_usize;
        let d = 2_usize;
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64).sin()).collect();
        let cfg = make_cfg(30, 64, 99);
        let fit =
            ifor_fit(&data, n, d, &cfg).expect("ifor_fit should succeed with sinusoidal data");
        let preds =
            ifor_predict(&fit, &data, n, 0.5).expect("ifor_predict should succeed on valid fit");
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn ifor_path_length_finite() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg = make_cfg(10, 4, 55);
        let fit = ifor_fit(&data, 4, 2, &cfg).expect("ifor_fit should succeed with small dataset");
        let pl = ifor_path_length(&fit, &[2.5_f64, 3.5]);
        assert!(pl.is_finite() && pl >= 0.0, "path_length={pl}");
    }

    #[test]
    fn ifor_empty_input_error() {
        let cfg = make_cfg(10, 8, 1);
        assert!(ifor_fit(&[], 0, 2, &cfg).is_err());
    }

    #[test]
    fn ifor_subsample_larger_than_n() {
        // subsample_size > n should be clamped, not panic
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = IforConfig {
            n_trees: 5,
            subsample_size: 1024,
            max_depth: 4,
            seed: 3,
        };
        let fit = ifor_fit(&data, 2, 2, &cfg)
            .expect("ifor_fit should clamp oversized subsample gracefully");
        let s = ifor_score(&fit, &[1.5_f64, 2.5], 1)
            .expect("ifor_score should succeed after clamped subsample fit");
        assert_eq!(s.len(), 1);
        assert!((0.0..=1.0).contains(&s[0]));
    }
}
