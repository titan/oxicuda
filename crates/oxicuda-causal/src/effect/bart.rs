//! BART — Bayesian Additive Regression Trees (BART-light, greedy backfitting).
//!
//! Reference: Chipman, H. A., George, E. I., & McCulloch, R. E. (2010).
//! "BART: Bayesian Additive Regression Trees." *Annals of Applied Statistics*,
//! 4(1), 266–298.
//!
//! # Algorithm
//!
//! The original BART represents an outcome regression `ŷ(x) = Σ_m g_m(x)`
//! as a sum of `M` small regression trees `g_m`, with the trees and their
//! leaf parameters drawn from a hierarchical Bayesian posterior via the
//! Chipman-George-McCulloch MCMC backfitting sampler.
//!
//! # Simplification ("BART-light")
//!
//! This module implements a **deterministic greedy backfitting** approximation
//! of BART that captures the sum-of-shallow-trees boosting-like fit but
//! avoids the full Bayesian MCMC.  The procedure is:
//!
//! 1. Initialise all `M` trees to depth-0 leaves with value `0`.
//! 2. For each of `n_iter` backfitting iterations, cycle `m = 0..M`:
//!    a. Compute the partial-residual `r_i = y_i − Σ_{j ≠ m} g_j(x_i)`.
//!    b. Re-fit tree `g_m` to `(X, r)` via greedy variance-reduction
//!    splitting up to `max_depth`, with minimum-leaf-samples constraint.
//!    c. Scale each leaf value of the new tree by `shrinkage ∈ (0, 1]`
//!    (the analogue of BART's prior shrinkage on leaf parameters).
//!
//! The greedy split criterion at each node is the **maximum reduction in
//! sum-of-squared-residuals (SSR)**,
//!
//! ```text
//!   gain(j, s) = SSR_parent − ( SSR_left + SSR_right )
//! ```
//!
//! evaluated over every feature `j` and every candidate threshold `s` =
//! midpoint of consecutive sorted unique values on the rows reaching the
//! node.  Leaves use the sample mean of the residuals as their fitted
//! constant; the SSR of a set of rows with sample mean `r̄` is simply
//! `Σ_i (r_i − r̄)²`.
//!
//! This estimator is fully deterministic (no RNG) and serves as a strong
//! nonparametric outcome-model surrogate for the downstream causal pipeline.
//!
//! # Configuration semantics
//!
//! - `n_trees` ≥ 1            — number of trees `M` in the ensemble.
//! - `max_depth` ≥ 0          — maximum depth per tree (`0` ⇒ root-only leaf).
//! - `min_leaf_samples` ≥ 1   — minimum samples on each side of a split.
//! - `n_iter` ≥ 1             — number of backfitting passes over all trees.
//! - `shrinkage` ∈ (0, 1]     — multiplicative factor applied to every leaf
//!   value of a freshly-fit tree.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`Bart::fit`].
#[derive(Debug, Clone)]
pub struct BartConfig {
    /// Number of regression trees in the additive ensemble (must be ≥ 1).
    pub n_trees: usize,
    /// Maximum depth per tree (a value of `0` keeps the tree as a single leaf).
    pub max_depth: usize,
    /// Minimum number of samples on each side of any candidate split
    /// (must be ≥ 1).
    pub min_leaf_samples: usize,
    /// Number of full backfitting passes over the ensemble (must be ≥ 1).
    pub n_iter: usize,
    /// Per-tree leaf-value shrinkage factor in `(0, 1]`.  Smaller values
    /// emulate stronger Bayesian prior shrinkage on individual leaves.
    pub shrinkage: f32,
}

impl Default for BartConfig {
    fn default() -> Self {
        Self {
            n_trees: 50,
            max_depth: 3,
            min_leaf_samples: 5,
            n_iter: 5,
            shrinkage: 0.1,
        }
    }
}

/// Node of a single BART regression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum BartNode {
    /// Terminal node with a constant fitted value.
    Leaf {
        /// Fitted constant prediction at this leaf (already shrunk).
        value: f32,
    },
    /// Internal split: go `left` if `x[feature] <= threshold`, else `right`.
    Split {
        /// Index of the feature used in the split.
        feature: usize,
        /// Split threshold (midpoint of consecutive sorted unique values).
        threshold: f32,
        /// Subtree for `x[feature] <= threshold`.
        left: Box<BartNode>,
        /// Subtree for `x[feature] > threshold`.
        right: Box<BartNode>,
    },
}

/// A single shallow regression tree within the BART ensemble.
#[derive(Debug, Clone)]
pub struct BartTree {
    /// Root node of the tree.
    pub root: BartNode,
}

/// Fitted BART ensemble.
#[derive(Debug, Clone)]
pub struct Bart {
    /// The `M` constituent regression trees.
    pub trees: Vec<BartTree>,
    /// Configuration used during fitting (retained for diagnostics).
    pub cfg: BartConfig,
}

impl Bart {
    /// Fit a BART-light ensemble on row-major `features` (`n_samples × n_features`)
    /// and target vector `y` (length `n_samples`).
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] when any configuration field
    /// is outside its allowed range, and [`CausalError::DimensionMismatch`]
    /// when any input slice has the wrong length.
    pub fn fit(
        features: &[f32],
        y: &[f32],
        n_samples: usize,
        n_features: usize,
        cfg: BartConfig,
    ) -> CausalResult<Self> {
        validate_fit_inputs(features, y, n_samples, n_features, &cfg)?;

        // All-zero residual contribution at start.
        let mut tree_preds = vec![vec![0.0_f32; n_samples]; cfg.n_trees];
        let mut trees: Vec<BartTree> = (0..cfg.n_trees)
            .map(|_| BartTree {
                root: BartNode::Leaf { value: 0.0 },
            })
            .collect();

        let mut residual = vec![0.0_f32; n_samples];
        let rows_all: Vec<usize> = (0..n_samples).collect();

        for _iter in 0..cfg.n_iter {
            for m in 0..cfg.n_trees {
                // r_i = y_i − Σ_{j ≠ m} g_j(x_i)
                for (i, r_i) in residual.iter_mut().enumerate().take(n_samples) {
                    let mut sum_other = 0.0_f32;
                    for (j, preds_j) in tree_preds.iter().enumerate() {
                        if j == m {
                            continue;
                        }
                        sum_other += preds_j[i];
                    }
                    *r_i = y[i] - sum_other;
                }

                // Fit a shallow tree to (X, residual) via greedy variance reduction.
                let root = fit_tree_node(
                    features,
                    &residual,
                    &rows_all,
                    n_features,
                    cfg.max_depth,
                    cfg.min_leaf_samples,
                    cfg.shrinkage,
                );
                let new_tree = BartTree { root };

                // Cache predictions for the newly-fit tree on the training data.
                let mut new_preds = vec![0.0_f32; n_samples];
                for i in 0..n_samples {
                    let row = &features[i * n_features..(i + 1) * n_features];
                    new_preds[i] = predict_tree(&new_tree.root, row);
                }

                trees[m] = new_tree;
                tree_preds[m] = new_preds;
            }
        }

        Ok(Self { trees, cfg })
    }

    /// Predict `ŷ(x) = Σ_m g_m(x)` for a single feature vector.
    ///
    /// Out-of-range feature indices encountered during traversal yield a
    /// [`CausalError::DimensionMismatch`] error; a depth-0 ensemble of leaves
    /// has no feature access and accepts any (including empty) `x`.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if any split along the
    /// traversed path references a feature index out of range for `x`.
    pub fn predict(&self, x: &[f32]) -> CausalResult<f32> {
        let mut sum = 0.0_f32;
        for tree in &self.trees {
            sum += predict_tree_checked(&tree.root, x)?;
        }
        Ok(sum)
    }

    /// Predict `ŷ(x) = Σ_m g_m(x)` for a row-major `n_samples × n_features`
    /// matrix.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] when `features.len()` does
    /// not match `n_samples * n_features`, or when any split references a
    /// feature index ≥ `n_features`.
    pub fn predict_batch(
        &self,
        features: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> CausalResult<Vec<f32>> {
        if features.len() != n_samples * n_features {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_features,
                got: features.len(),
            });
        }
        let mut out = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let row = &features[s * n_features..(s + 1) * n_features];
            out.push(self.predict(row)?);
        }
        Ok(out)
    }

    /// Number of trees in the fitted ensemble.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

// =====================================================================
// validation
// =====================================================================

fn validate_fit_inputs(
    features: &[f32],
    y: &[f32],
    n_samples: usize,
    n_features: usize,
    cfg: &BartConfig,
) -> CausalResult<()> {
    if cfg.n_trees == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_trees must be >= 1".to_string(),
        });
    }
    if cfg.min_leaf_samples == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "min_leaf_samples must be >= 1".to_string(),
        });
    }
    if cfg.n_iter == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_iter must be >= 1".to_string(),
        });
    }
    if !(cfg.shrinkage > 0.0 && cfg.shrinkage <= 1.0) {
        return Err(CausalError::InvalidParameter {
            reason: format!("shrinkage must be in (0, 1], got {}", cfg.shrinkage),
        });
    }
    if n_samples == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_samples must be >= 1".to_string(),
        });
    }
    if n_features == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_features must be >= 1".to_string(),
        });
    }
    if features.len() != n_samples * n_features {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples * n_features,
            got: features.len(),
        });
    }
    if y.len() != n_samples {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples,
            got: y.len(),
        });
    }
    Ok(())
}

// =====================================================================
// tree fitting (greedy variance reduction)
// =====================================================================

/// Mean of `vals[indices]`. Returns `0.0` for empty `indices`.
fn mean_of(vals: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for &i in indices {
        s += vals[i] as f64;
    }
    (s / indices.len() as f64) as f32
}

/// Sum-of-squared-residuals about the mean for `vals[indices]`.
fn ssr_of(vals: &[f32], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mu = mean_of(vals, indices) as f64;
    let mut s = 0.0_f64;
    for &i in indices {
        let d = vals[i] as f64 - mu;
        s += d * d;
    }
    s
}

/// Best-split record kept during the greedy-split search:
/// `(feature_index, threshold, left_rows, right_rows, ssr_gain)`.
type BestSplit = (usize, f32, Vec<usize>, Vec<usize>, f64);

/// Recursively fit a regression sub-tree on `rows`, residual target `r`,
/// with the given limits.  Leaf values are scaled by `shrinkage`.
fn fit_tree_node(
    features: &[f32],
    r: &[f32],
    rows: &[usize],
    n_features: usize,
    depth_left: usize,
    min_leaf_samples: usize,
    shrinkage: f32,
) -> BartNode {
    let leaf_value = mean_of(r, rows) * shrinkage;
    let leaf = BartNode::Leaf { value: leaf_value };

    if depth_left == 0 || rows.len() < 2 * min_leaf_samples {
        return leaf;
    }

    let parent_ssr = ssr_of(r, rows);
    let mut best: Option<BestSplit> = None;

    for feature in 0..n_features {
        let mut values: Vec<f32> = rows
            .iter()
            .map(|&i| features[i * n_features + feature])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        if values.len() < 2 {
            continue;
        }
        for w in values.windows(2) {
            let threshold = 0.5 * (w[0] + w[1]);
            let mut left_rows = Vec::new();
            let mut right_rows = Vec::new();
            for &i in rows {
                let v = features[i * n_features + feature];
                if v <= threshold {
                    left_rows.push(i);
                } else {
                    right_rows.push(i);
                }
            }
            if left_rows.len() < min_leaf_samples || right_rows.len() < min_leaf_samples {
                continue;
            }
            let left_ssr = ssr_of(r, &left_rows);
            let right_ssr = ssr_of(r, &right_rows);
            let gain = parent_ssr - (left_ssr + right_ssr);
            let improves = match &best {
                Some((_, _, _, _, best_gain)) => gain > *best_gain,
                None => true,
            };
            if improves {
                best = Some((feature, threshold, left_rows, right_rows, gain));
            }
        }
    }

    match best {
        Some((feature, threshold, left_rows, right_rows, gain)) if gain > 0.0 => {
            let left_node = fit_tree_node(
                features,
                r,
                &left_rows,
                n_features,
                depth_left - 1,
                min_leaf_samples,
                shrinkage,
            );
            let right_node = fit_tree_node(
                features,
                r,
                &right_rows,
                n_features,
                depth_left - 1,
                min_leaf_samples,
                shrinkage,
            );
            BartNode::Split {
                feature,
                threshold,
                left: Box::new(left_node),
                right: Box::new(right_node),
            }
        }
        _ => leaf,
    }
}

// =====================================================================
// prediction helpers
// =====================================================================

/// Internal infallible traversal used during fitting where bounds are known.
fn predict_tree(node: &BartNode, x: &[f32]) -> f32 {
    let mut cur = node;
    loop {
        match cur {
            BartNode::Leaf { value } => return *value,
            BartNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let v = x[*feature];
                cur = if v <= *threshold { left } else { right };
            }
        }
    }
}

/// External fallible traversal that returns an error for out-of-range feature
/// indices instead of panicking.
fn predict_tree_checked(node: &BartNode, x: &[f32]) -> CausalResult<f32> {
    let mut cur = node;
    loop {
        match cur {
            BartNode::Leaf { value } => return Ok(*value),
            BartNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let v = *x.get(*feature).ok_or(CausalError::DimensionMismatch {
                    expected: *feature + 1,
                    got: x.len(),
                })?;
                cur = if v <= *threshold { left } else { right };
            }
        }
    }
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut s = 0.0_f64;
        for i in 0..a.len() {
            let d = a[i] as f64 - b[i] as f64;
            s += d * d;
        }
        (s / a.len() as f64) as f32
    }

    /// Deterministic toy dataset where `y_i = x_i[feature_idx]` so a single
    /// shallow tree can perfectly recover the target with enough depth.
    fn make_identity_dataset(
        n_samples: usize,
        n_features: usize,
        feature_idx: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut features = vec![0.0_f32; n_samples * n_features];
        let mut y = vec![0.0_f32; n_samples];
        for i in 0..n_samples {
            for j in 0..n_features {
                let val = ((i * 7 + j * 31) % 100) as f32 / 100.0 - 0.5;
                features[i * n_features + j] = val;
            }
            y[i] = features[i * n_features + feature_idx];
        }
        (features, y)
    }

    /// Smallest leaf size reachable from the given root via recursive
    /// traversal of `rows`.
    fn min_leaf_size(
        node: &BartNode,
        rows: &[usize],
        features: &[f32],
        n_features: usize,
    ) -> usize {
        match node {
            BartNode::Leaf { .. } => rows.len(),
            BartNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let mut lr = Vec::new();
                let mut rr = Vec::new();
                for &i in rows {
                    if features[i * n_features + feature] <= *threshold {
                        lr.push(i);
                    } else {
                        rr.push(i);
                    }
                }
                min_leaf_size(left, &lr, features, n_features)
                    .min(min_leaf_size(right, &rr, features, n_features))
            }
        }
    }

    fn n_features_for_test() -> usize {
        2
    }

    #[test]
    fn single_tree_depth_zero_predicts_mean_times_shrinkage_times_n_iter() {
        // With max_depth=0 every tree is a depth-0 leaf fit to the residual.
        // For n_trees=1 and n_iter=1 the residual at the only tree's fit is
        // exactly y, so the leaf value is `shrinkage · mean(y)` and
        // `predict(x) = leaf · 1 == shrinkage · mean(y)`.
        let n = 6;
        let d = n_features_for_test();
        let (features, y) = make_identity_dataset(n, d, 0);
        let mean_y: f32 = y.iter().sum::<f32>() / n as f32;
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 0,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let pred = model.predict(&features[..d]).unwrap();
        assert!(
            (pred - 0.5 * mean_y).abs() < 1e-5,
            "pred={pred}, expected {}",
            0.5 * mean_y
        );
    }

    #[test]
    fn predict_batch_length_matches_n_samples() {
        let n = 12;
        let d = n_features_for_test();
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 3,
            max_depth: 2,
            min_leaf_samples: 1,
            n_iter: 2,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let preds = model.predict_batch(&features, n, d).unwrap();
        assert_eq!(preds.len(), n);
        for &p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fits_identity_target_with_low_mse() {
        // y = x[:,0] should be recoverable to small MSE by a sum of shallow trees.
        let n = 50;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 20,
            max_depth: 4,
            min_leaf_samples: 1,
            n_iter: 6,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let preds = model.predict_batch(&features, n, d).unwrap();
        let m = mse(&preds, &y);
        assert!(m < 0.02, "MSE = {m} (expected < 0.02 for identity target)");
    }

    #[test]
    fn deeper_trees_lower_train_mse_than_depth_zero() {
        let n = 40;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg_shallow = BartConfig {
            n_trees: 5,
            max_depth: 0,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 1.0,
        };
        let cfg_deep = BartConfig {
            n_trees: 5,
            max_depth: 4,
            min_leaf_samples: 1,
            n_iter: 5,
            shrinkage: 0.5,
        };
        let m_shallow = Bart::fit(&features, &y, n, d, cfg_shallow).unwrap();
        let m_deep = Bart::fit(&features, &y, n, d, cfg_deep).unwrap();
        let s_mse = mse(&m_shallow.predict_batch(&features, n, d).unwrap(), &y);
        let d_mse = mse(&m_deep.predict_batch(&features, n, d).unwrap(), &y);
        assert!(
            d_mse < s_mse,
            "deep mse {d_mse} should beat shallow mse {s_mse}"
        );
    }

    #[test]
    fn more_iters_keep_predictions_bounded() {
        // The greedy backfitting iteration converges to a fixed point of the
        // shrunken-tree update (predictions stay bounded — they do not
        // diverge with `n_iter`).  Verify the predictions remain finite and
        // bounded by a small constant times max|y| even at large n_iter.
        let n = 32;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let y_max = y.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let cfg_many = BartConfig {
            n_trees: 3,
            max_depth: 10,
            min_leaf_samples: 1,
            n_iter: 50,
            shrinkage: 0.5,
        };
        let m_many = Bart::fit(&features, &y, n, d, cfg_many).unwrap();
        let preds = m_many.predict_batch(&features, n, d).unwrap();
        let p_max = preds.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        for &p in &preds {
            assert!(p.is_finite(), "prediction {p} not finite");
        }
        // Sum-of-shallow-trees bounded by n_trees · shrinkage · max|residual|;
        // a generous safety factor `4·y_max` catches genuine divergence.
        assert!(
            p_max <= 4.0 * y_max + 1.0,
            "predictions diverged: |p|_max = {p_max} vs |y|_max = {y_max}"
        );
    }

    #[test]
    fn shrinkage_changes_prediction_magnitude() {
        let n = 16;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg_a = BartConfig {
            n_trees: 1,
            max_depth: 0,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        let cfg_b = BartConfig {
            shrinkage: 1.0,
            ..cfg_a.clone()
        };
        let m_a = Bart::fit(&features, &y, n, d, cfg_a).unwrap();
        let m_b = Bart::fit(&features, &y, n, d, cfg_b).unwrap();
        let pa = m_a.predict(&features[..d]).unwrap();
        let pb = m_b.predict(&features[..d]).unwrap();
        // For depth-0 / n_iter=1 / n_trees=1 the relationship is exactly
        // pa == 0.5 · mean(y), pb == 1.0 · mean(y).
        assert!(
            (pa * 2.0 - pb).abs() < 1e-4,
            "pa*2 = {} vs pb = {}",
            pa * 2.0,
            pb
        );
    }

    #[test]
    fn more_trees_lower_train_mse_at_iter_one() {
        // With shrinkage 0.5 < 1, one tree (iter=1) leaves a 0.5y residual.
        // Adding a second tree (iter=1) lets it absorb a further 0.5·0.5y, so
        // the fit improves as M grows under fixed n_iter=1.
        let n = 32;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg_one = BartConfig {
            n_trees: 1,
            max_depth: 8,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        let cfg_many = BartConfig {
            n_trees: 4,
            max_depth: 8,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        let m_one = Bart::fit(&features, &y, n, d, cfg_one).unwrap();
        let m_many = Bart::fit(&features, &y, n, d, cfg_many).unwrap();
        let one_mse = mse(&m_one.predict_batch(&features, n, d).unwrap(), &y);
        let many_mse = mse(&m_many.predict_batch(&features, n, d).unwrap(), &y);
        assert!(
            many_mse < one_mse,
            "M=4 mse {many_mse} not less than M=1 mse {one_mse}"
        );
    }

    #[test]
    fn n_trees_at_least_two_supported() {
        let n = 20;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 2,
            max_depth: 2,
            min_leaf_samples: 1,
            n_iter: 2,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        assert_eq!(model.n_trees(), 2);
        assert_eq!(model.trees.len(), 2);
    }

    #[test]
    fn min_leaf_samples_respected() {
        let n = 30;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let min_leaf = 4;
        let cfg = BartConfig {
            n_trees: 3,
            max_depth: 5,
            min_leaf_samples: min_leaf,
            n_iter: 2,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let rows: Vec<usize> = (0..n).collect();
        for tree in &model.trees {
            let smallest = min_leaf_size(&tree.root, &rows, &features, d);
            assert!(
                smallest >= min_leaf,
                "leaf below min_leaf {min_leaf}: {smallest}"
            );
        }
    }

    #[test]
    fn deterministic_fit_no_rng() {
        let n = 24;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 1);
        let cfg = BartConfig {
            n_trees: 4,
            max_depth: 3,
            min_leaf_samples: 1,
            n_iter: 3,
            shrinkage: 0.5,
        };
        let a = Bart::fit(&features, &y, n, d, cfg.clone()).unwrap();
        let b = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let pa = a.predict_batch(&features, n, d).unwrap();
        let pb = b.predict_batch(&features, n, d).unwrap();
        assert_eq!(pa, pb);
    }

    #[test]
    fn err_n_trees_zero() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 0,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn max_depth_zero_allowed() {
        // max_depth = 0 must be a valid configuration.
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 0,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 1.0,
        };
        assert!(Bart::fit(&features, &y, n, d, cfg).is_ok());
    }

    #[test]
    fn err_min_leaf_zero() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 0,
            n_iter: 1,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_n_iter_zero() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 0,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_shrinkage_zero() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.0,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_shrinkage_negative() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: -0.1,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_shrinkage_greater_than_one() {
        let n = 5;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 1.5,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_features_wrong_length() {
        let n = 5;
        let d = 2;
        let (mut features, y) = make_identity_dataset(n, d, 0);
        features.push(0.0); // now wrong length
        let cfg = BartConfig::default();
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_y_wrong_length() {
        let n = 5;
        let d = 2;
        let (features, mut y) = make_identity_dataset(n, d, 0);
        y.push(0.0);
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&features, &y, n, d, cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_samples_zero() {
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&[], &[], 0, 2, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_n_features_zero() {
        let cfg = BartConfig {
            n_trees: 1,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        assert!(matches!(
            Bart::fit(&[], &[0.0_f32; 3], 3, 0, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_predict_x_wrong_length() {
        // Train a model whose root is a split (identity target with depth>=1).
        let n = 20;
        let d = 3;
        let (features, y) = make_identity_dataset(n, d, 1);
        let cfg = BartConfig {
            n_trees: 2,
            max_depth: 2,
            min_leaf_samples: 1,
            n_iter: 2,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        // At least one tree must have a split (otherwise the test is trivial).
        let has_split = model
            .trees
            .iter()
            .any(|t| matches!(t.root, BartNode::Split { .. }));
        assert!(has_split, "expected at least one tree to split");
        assert!(matches!(
            model.predict(&[]),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn constant_y_yields_constant_prediction() {
        let n = 12;
        let d = 2;
        let (features, _) = make_identity_dataset(n, d, 0);
        let y = vec![3.5_f32; n];
        let cfg = BartConfig {
            n_trees: 4,
            max_depth: 2,
            min_leaf_samples: 1,
            n_iter: 4,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let preds = model.predict_batch(&features, n, d).unwrap();
        let first = preds[0];
        for &p in &preds {
            assert!(
                (p - first).abs() < 1e-4,
                "constant-y prediction varies: {p} vs {first}"
            );
        }
    }

    #[test]
    fn single_sample_edge_case() {
        let n = 1;
        let d = 2;
        let features = vec![0.5_f32, -0.25];
        let y = vec![1.25_f32];
        let cfg = BartConfig {
            n_trees: 2,
            max_depth: 2,
            min_leaf_samples: 1,
            n_iter: 3,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let p = model.predict(&features).unwrap();
        assert!(p.is_finite());
        // With a single sample no split is feasible (needs 2·min_leaf_samples
        // rows), so every tree is a depth-0 leaf with value shrinkage · y.
        // After convergence the sum should equal y exactly when shrinkage
        // satisfies M · shrinkage · y == y mod backfitting — we only require
        // finiteness here and check the batch interface.
        let pb = model.predict_batch(&features, n, d).unwrap();
        assert_eq!(pb.len(), 1);
        assert!((pb[0] - p).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_matches_per_sample_predict() {
        let n = 18;
        let d = 3;
        let (features, y) = make_identity_dataset(n, d, 2);
        let cfg = BartConfig {
            n_trees: 5,
            max_depth: 3,
            min_leaf_samples: 1,
            n_iter: 3,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let batch = model.predict_batch(&features, n, d).unwrap();
        for i in 0..n {
            let row = &features[i * d..(i + 1) * d];
            let single = model.predict(row).unwrap();
            assert!(
                (single - batch[i]).abs() < 1e-6,
                "row {i}: single {single} vs batch {}",
                batch[i]
            );
        }
    }

    #[test]
    fn err_predict_batch_features_wrong_length() {
        let n = 10;
        let d = 2;
        let (features, y) = make_identity_dataset(n, d, 0);
        let cfg = BartConfig {
            n_trees: 2,
            max_depth: 1,
            min_leaf_samples: 1,
            n_iter: 1,
            shrinkage: 0.5,
        };
        let model = Bart::fit(&features, &y, n, d, cfg).unwrap();
        let bad = vec![0.0_f32; n * d + 1];
        assert!(matches!(
            model.predict_batch(&bad, n, d),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn config_default_is_sane() {
        let cfg = BartConfig::default();
        assert!(cfg.n_trees >= 1);
        assert!(cfg.min_leaf_samples >= 1);
        assert!(cfg.n_iter >= 1);
        assert!(cfg.shrinkage > 0.0 && cfg.shrinkage <= 1.0);
    }
}
