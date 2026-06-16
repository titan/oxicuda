//! Random Forest (Breiman 2001): bootstrap-aggregated CART trees with random
//! feature subsampling at each split.
//!
//! Each tree is grown on a bootstrap resample of the training rows. At every
//! node a random subset of `max_features` candidate columns is considered and
//! the split that maximally reduces the node impurity is chosen
//! (**Gini** for classification, **variance** for regression). The ensemble
//! prediction averages probabilities (classification) or leaf means
//! (regression) across all trees. Trees are grown with an explicit work stack —
//! no recursion — and stored as flat node arrays.
//!
//! The randomness comes from two sources, both driven by the workspace
//! `LcgRng`: the per-tree bootstrap sample and the per-node feature subset.
//! De-correlating the trees this way is what makes the bagged average far more
//! accurate than a single deep tree.
//!
//! ## References
//! - Breiman, L. (2001). "Random Forests." Machine Learning 45(1), 5–32.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Task ──────────────────────────────────────────────────────────────────────

/// Whether the forest performs classification or regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForestTask {
    /// Multi-class classification over `n_classes` labels.
    Classification { n_classes: usize },
    /// Scalar regression.
    Regression,
}

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for a [`RandomForest`].
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    /// Number of trees in the ensemble (≥ 1).
    pub n_trees: usize,
    /// Maximum tree depth (≥ 1).
    pub max_depth: usize,
    /// Minimum samples required to attempt a split (≥ 2).
    pub min_samples_split: usize,
    /// Number of candidate features sampled per split (`0` ⇒ √p heuristic).
    pub max_features: usize,
    /// Bootstrap sample size as a fraction of `n` in `(0, 1]`.
    pub bootstrap_fraction: f64,
    /// Learning task.
    pub task: ForestTask,
    /// RNG seed.
    pub seed: u64,
}

impl RandomForestConfig {
    /// A reasonable classification default (`n_classes` classes).
    #[must_use]
    pub fn classification(n_classes: usize) -> Self {
        Self {
            n_trees: 50,
            max_depth: 8,
            min_samples_split: 2,
            max_features: 0,
            bootstrap_fraction: 1.0,
            task: ForestTask::Classification { n_classes },
            seed: 0,
        }
    }

    /// A reasonable regression default.
    #[must_use]
    pub fn regression() -> Self {
        Self {
            n_trees: 50,
            max_depth: 8,
            min_samples_split: 2,
            max_features: 0,
            bootstrap_fraction: 1.0,
            task: ForestTask::Regression,
            seed: 0,
        }
    }
}

// ─── Flat tree node ────────────────────────────────────────────────────────────

/// A node in a forest tree stored in a flat `Vec`.
#[derive(Debug, Clone)]
pub struct ForestNode {
    /// Whether this node is a leaf.
    pub is_leaf: bool,
    /// Split feature index (internal nodes).
    pub feature_idx: usize,
    /// Split threshold: go left if `x[feature_idx] ≤ threshold`.
    pub threshold: f64,
    /// Left child index.
    pub left_child: usize,
    /// Right child index.
    pub right_child: usize,
    /// Leaf payload: class probabilities (classification) or a single mean
    /// (regression, length 1).
    pub value: Vec<f64>,
}

/// One CART tree within the forest.
#[derive(Debug, Clone)]
pub struct ForestTree {
    /// Flat node array; node 0 is the root.
    pub nodes: Vec<ForestNode>,
}

impl ForestTree {
    /// Predict the leaf payload for a single sample row `x` (length `n_features`).
    fn predict_one(&self, x: &[f64], n_features: usize) -> &[f64] {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return &node.value;
            }
            let v = x[node.feature_idx.min(n_features - 1)];
            idx = if v <= node.threshold {
                node.left_child
            } else {
                node.right_child
            };
        }
    }
}

// ─── Impurity helpers ──────────────────────────────────────────────────────────

/// Gini impurity of a label subset, weighted by subset size: `n · (1 − Σ p_c²)`.
fn weighted_gini(labels: &[usize], indices: &[usize], n_classes: usize) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut counts = vec![0.0_f64; n_classes];
    for &i in indices {
        counts[labels[i]] += 1.0;
    }
    let n = indices.len() as f64;
    let mut sum_sq = 0.0;
    for &c in &counts {
        let p = c / n;
        sum_sq += p * p;
    }
    n * (1.0 - sum_sq)
}

/// Within-group sum of squared deviations for a regression target subset.
fn weighted_sse(targets: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mean = indices.iter().map(|&i| targets[i]).sum::<f64>() / indices.len() as f64;
    indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum()
}

// ─── Sampling helpers ──────────────────────────────────────────────────────────

/// Draw `k` indices with replacement (bootstrap) from `0..n`.
fn bootstrap_indices(rng: &mut LcgRng, n: usize, k: usize) -> Vec<usize> {
    (0..k).map(|_| rng.next_usize(n)).collect()
}

/// Draw `k` distinct column indices from `0..n_features` (partial Fisher–Yates).
fn sample_features(rng: &mut LcgRng, n_features: usize, k: usize) -> Vec<usize> {
    let mut cols: Vec<usize> = (0..n_features).collect();
    let k = k.min(n_features);
    for i in 0..k {
        let j = i + rng.next_usize(n_features - i);
        cols.swap(i, j);
    }
    cols.truncate(k);
    cols
}

// ─── Trained model ─────────────────────────────────────────────────────────────

/// A trained random forest.
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: Vec<ForestTree>,
    config: RandomForestConfig,
    n_features: usize,
    out_dim: usize,
}

struct BuildCtx<'a> {
    x: &'a [f64],
    n_features: usize,
    labels: &'a [usize],
    targets: &'a [f64],
    task: ForestTask,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
}

impl RandomForest {
    /// Fit a random forest classifier.
    ///
    /// `x` is a flat row-major `[n × n_features]` matrix and `y` holds the
    /// integer class labels (`0..n_classes`).
    ///
    /// # Errors
    /// - [`TabularError::InsufficientSamples`] if `n == 0`.
    /// - [`TabularError::InvalidFeatureCount`] if `n_features == 0`.
    /// - [`TabularError::InvalidTreeCount`] / [`TabularError::InvalidTreeDepth`]
    ///   for non-positive ensemble size or depth.
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n * n_features` or
    ///   `y.len() != n`.
    /// - [`TabularError::LabelOutOfRange`] if a label `≥ n_classes`.
    /// - [`TabularError::InvalidParameter`] for an invalid bootstrap fraction.
    pub fn fit_classifier(
        x: &[f64],
        y: &[usize],
        n: usize,
        n_features: usize,
        config: RandomForestConfig,
    ) -> TabularResult<Self> {
        let n_classes = match config.task {
            ForestTask::Classification { n_classes } => n_classes,
            ForestTask::Regression => {
                return Err(TabularError::InvalidParameter {
                    name: "task".into(),
                    msg: "fit_classifier requires a Classification task".into(),
                });
            }
        };
        if n_classes == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_classes".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        validate(x, n, n_features, y.len(), &config)?;
        for &label in y {
            if label >= n_classes {
                return Err(TabularError::LabelOutOfRange { label, n_classes });
            }
        }
        let dummy_targets: Vec<f64> = Vec::new();
        Self::build(x, y, &dummy_targets, n, n_features, n_classes, config)
    }

    /// Fit a random forest regressor.
    ///
    /// `x` is a flat row-major `[n × n_features]` matrix and `y` the scalar
    /// targets.
    ///
    /// # Errors
    /// As [`RandomForest::fit_classifier`] (minus label/class checks); also
    /// [`TabularError::InvalidParameter`] if the task is not `Regression`.
    pub fn fit_regressor(
        x: &[f64],
        y: &[f64],
        n: usize,
        n_features: usize,
        config: RandomForestConfig,
    ) -> TabularResult<Self> {
        if config.task != ForestTask::Regression {
            return Err(TabularError::InvalidParameter {
                name: "task".into(),
                msg: "fit_regressor requires a Regression task".into(),
            });
        }
        validate(x, n, n_features, y.len(), &config)?;
        let dummy_labels: Vec<usize> = Vec::new();
        Self::build(x, &dummy_labels, y, n, n_features, 1, config)
    }

    fn build(
        x: &[f64],
        labels: &[usize],
        targets: &[f64],
        n: usize,
        n_features: usize,
        out_dim: usize,
        config: RandomForestConfig,
    ) -> TabularResult<Self> {
        let max_features = if config.max_features == 0 {
            ((n_features as f64).sqrt().ceil() as usize).max(1)
        } else {
            config.max_features.min(n_features)
        };
        let sample_size = ((config.bootstrap_fraction * n as f64).round() as usize).clamp(1, n);

        let mut rng = LcgRng::new(config.seed);
        let mut trees = Vec::with_capacity(config.n_trees);
        for _ in 0..config.n_trees {
            let rows = bootstrap_indices(&mut rng, n, sample_size);
            let ctx = BuildCtx {
                x,
                n_features,
                labels,
                targets,
                task: config.task,
                max_depth: config.max_depth,
                min_samples_split: config.min_samples_split,
                max_features,
            };
            trees.push(build_tree(&ctx, rows, &mut rng));
        }
        Ok(Self {
            trees,
            config,
            n_features,
            out_dim,
        })
    }

    /// Number of trees in the ensemble.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Predict class probabilities for a single sample (classification only).
    ///
    /// Returns a length-`n_classes` probability vector (averaged over trees).
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    /// - [`TabularError::InvalidParameter`] for a regression forest.
    pub fn predict_proba(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        if !matches!(self.config.task, ForestTask::Classification { .. }) {
            return Err(TabularError::InvalidParameter {
                name: "task".into(),
                msg: "predict_proba is only valid for classification".into(),
            });
        }
        if x.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut acc = vec![0.0_f64; self.out_dim];
        for tree in &self.trees {
            let leaf = tree.predict_one(x, self.n_features);
            for (a, &v) in acc.iter_mut().zip(leaf.iter()) {
                *a += v;
            }
        }
        let n = self.trees.len() as f64;
        for a in &mut acc {
            *a /= n;
        }
        Ok(acc)
    }

    /// Predict the most probable class label for a single sample.
    ///
    /// # Errors
    /// As [`RandomForest::predict_proba`].
    pub fn predict_class(&self, x: &[f64]) -> TabularResult<usize> {
        let proba = self.predict_proba(x)?;
        let mut best = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for (c, &p) in proba.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = c;
            }
        }
        Ok(best)
    }

    /// Predict the regression target for a single sample (regression only).
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
    /// - [`TabularError::InvalidParameter`] for a classification forest.
    pub fn predict(&self, x: &[f64]) -> TabularResult<f64> {
        if self.config.task != ForestTask::Regression {
            return Err(TabularError::InvalidParameter {
                name: "task".into(),
                msg: "predict is only valid for regression".into(),
            });
        }
        if x.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut acc = 0.0_f64;
        for tree in &self.trees {
            acc += tree.predict_one(x, self.n_features)[0];
        }
        Ok(acc / self.trees.len() as f64)
    }
}

// ─── Tree builder (shared with Extra Trees via split callback) ─────────────────

/// Leaf payload for the indices at a node.
fn leaf_payload(ctx: &BuildCtx<'_>, indices: &[usize]) -> Vec<f64> {
    match ctx.task {
        ForestTask::Classification { n_classes } => {
            let mut counts = vec![0.0_f64; n_classes];
            for &i in indices {
                counts[ctx.labels[i]] += 1.0;
            }
            let n = indices.len().max(1) as f64;
            for c in &mut counts {
                *c /= n;
            }
            counts
        }
        ForestTask::Regression => {
            let mean = if indices.is_empty() {
                0.0
            } else {
                indices.iter().map(|&i| ctx.targets[i]).sum::<f64>() / indices.len() as f64
            };
            vec![mean]
        }
    }
}

/// Parent impurity for the node (Gini or SSE depending on task).
fn node_impurity(ctx: &BuildCtx<'_>, indices: &[usize]) -> f64 {
    match ctx.task {
        ForestTask::Classification { n_classes } => weighted_gini(ctx.labels, indices, n_classes),
        ForestTask::Regression => weighted_sse(ctx.targets, indices),
    }
}

type Split = (usize, f64, Vec<usize>, Vec<usize>);

/// Greedy best split over a random feature subset (Random-Forest rule):
/// for each candidate feature, evaluate every midpoint between distinct sorted
/// values and keep the (feature, threshold) with the largest impurity decrease.
fn best_split_rf(ctx: &BuildCtx<'_>, indices: &[usize], rng: &mut LcgRng) -> Option<Split> {
    let parent = node_impurity(ctx, indices);
    let feats = sample_features(rng, ctx.n_features, ctx.max_features);
    let mut best_gain = 1e-12_f64;
    let mut best: Option<Split> = None;

    for feat in feats {
        let mut vals: Vec<(f64, usize)> = indices
            .iter()
            .map(|&i| (ctx.x[i * ctx.n_features + feat], i))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut i = 0usize;
        while i < vals.len() {
            let mut j = i + 1;
            while j < vals.len() && (vals[j].0 - vals[i].0).abs() < 1e-15 {
                j += 1;
            }
            if j >= vals.len() {
                break;
            }
            let threshold = 0.5 * (vals[j - 1].0 + vals[j].0);
            let left: Vec<usize> = vals[..j].iter().map(|&(_, i)| i).collect();
            let right: Vec<usize> = vals[j..].iter().map(|&(_, i)| i).collect();
            let gain = parent - node_impurity(ctx, &left) - node_impurity(ctx, &right);
            if gain > best_gain {
                best_gain = gain;
                best = Some((feat, threshold, left, right));
            }
            i = j;
        }
    }
    best
}

/// Grow a single tree on the given bootstrap rows using the Random-Forest split
/// rule. Iterative (explicit stack).
fn build_tree(ctx: &BuildCtx<'_>, rows: Vec<usize>, rng: &mut LcgRng) -> ForestTree {
    let mut nodes: Vec<ForestNode> = Vec::new();
    nodes.push(ForestNode {
        is_leaf: true,
        feature_idx: 0,
        threshold: 0.0,
        left_child: 0,
        right_child: 0,
        value: leaf_payload(ctx, &rows),
    });
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, rows, 0)];

    while let Some((node_idx, indices, depth)) = stack.pop() {
        let can_split = depth < ctx.max_depth && indices.len() >= ctx.min_samples_split;
        let split = if can_split {
            best_split_rf(ctx, &indices, rng)
        } else {
            None
        };
        match split {
            None => {
                nodes[node_idx].is_leaf = true;
                nodes[node_idx].value = leaf_payload(ctx, &indices);
            }
            Some((feat, threshold, left, right)) => {
                let left_idx = nodes.len();
                nodes.push(ForestNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold: 0.0,
                    left_child: 0,
                    right_child: 0,
                    value: leaf_payload(ctx, &left),
                });
                let right_idx = nodes.len();
                nodes.push(ForestNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold: 0.0,
                    left_child: 0,
                    right_child: 0,
                    value: leaf_payload(ctx, &right),
                });
                nodes[node_idx] = ForestNode {
                    is_leaf: false,
                    feature_idx: feat,
                    threshold,
                    left_child: left_idx,
                    right_child: right_idx,
                    value: Vec::new(),
                };
                stack.push((left_idx, left, depth + 1));
                stack.push((right_idx, right, depth + 1));
            }
        }
    }
    ForestTree { nodes }
}

// ─── Validation ────────────────────────────────────────────────────────────────

fn validate(
    x: &[f64],
    n: usize,
    n_features: usize,
    y_len: usize,
    config: &RandomForestConfig,
) -> TabularResult<()> {
    if n == 0 {
        return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
    }
    if n_features == 0 {
        return Err(TabularError::InvalidFeatureCount { n: 0 });
    }
    if config.n_trees == 0 {
        return Err(TabularError::InvalidTreeCount { n: 0 });
    }
    if config.max_depth == 0 {
        return Err(TabularError::InvalidTreeDepth { depth: 0 });
    }
    if config.min_samples_split < 2 {
        return Err(TabularError::InvalidParameter {
            name: "min_samples_split".into(),
            msg: "must be ≥ 2".into(),
        });
    }
    if !(config.bootstrap_fraction > 0.0 && config.bootstrap_fraction <= 1.0) {
        return Err(TabularError::InvalidParameter {
            name: "bootstrap_fraction".into(),
            msg: "must be in (0, 1]".into(),
        });
    }
    if x.len() != n * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n * n_features,
            got: x.len(),
        });
    }
    if y_len != n {
        return Err(TabularError::DimensionMismatch {
            expected: n,
            got: y_len,
        });
    }
    Ok(())
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Linearly separable 2-class data: class 1 when feature 0 > 0.5.
    fn make_classification(n: usize) -> (Vec<f64>, Vec<usize>) {
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = (i as f64 / n as f64) * 2.0 - 0.5; // spans [-0.5, 1.5]
            let f1 = ((i * 7) % 11) as f64 / 11.0;
            x.push(f0);
            x.push(f1);
            y.push(if f0 > 0.5 { 1 } else { 0 });
        }
        (x, y)
    }

    /// Regression target y ≈ 2·f0 + f1.
    fn make_regression(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = (i as f64 / n as f64) * 4.0;
            let f1 = ((i * 3) % 7) as f64;
            x.push(f0);
            x.push(f1);
            y.push(2.0 * f0 + f1);
        }
        (x, y)
    }

    #[test]
    fn classifier_fits_separable_data() {
        let (x, y) = make_classification(60);
        let mut cfg = RandomForestConfig::classification(2);
        cfg.n_trees = 20;
        cfg.seed = 1;
        let forest = RandomForest::fit_classifier(&x, &y, 60, 2, cfg).expect("ok");
        // Training accuracy should be high on separable data.
        let mut correct = 0;
        for i in 0..60 {
            let row = &x[i * 2..i * 2 + 2];
            if forest.predict_class(row).expect("ok") == y[i] {
                correct += 1;
            }
        }
        assert!(correct >= 55, "accuracy {correct}/60 too low");
    }

    #[test]
    fn classifier_proba_sums_to_one() {
        let (x, y) = make_classification(40);
        let cfg = RandomForestConfig::classification(2);
        let forest = RandomForest::fit_classifier(&x, &y, 40, 2, cfg).expect("ok");
        let p = forest.predict_proba(&[1.0, 0.3]).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum={s}");
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn classifier_predict_class_confident() {
        let (x, y) = make_classification(60);
        let mut cfg = RandomForestConfig::classification(2);
        cfg.n_trees = 25;
        let forest = RandomForest::fit_classifier(&x, &y, 60, 2, cfg).expect("ok");
        // A clearly class-1 point.
        assert_eq!(forest.predict_class(&[1.4, 0.5]).expect("ok"), 1);
        // A clearly class-0 point.
        assert_eq!(forest.predict_class(&[-0.4, 0.5]).expect("ok"), 0);
    }

    #[test]
    fn regressor_fits_linear_trend() {
        let (x, y) = make_regression(80);
        let mut cfg = RandomForestConfig::regression();
        cfg.n_trees = 30;
        cfg.max_depth = 10;
        let forest = RandomForest::fit_regressor(&x, &y, 80, 2, cfg).expect("ok");
        let mut mse = 0.0;
        for i in 0..80 {
            let row = &x[i * 2..i * 2 + 2];
            let pred = forest.predict(row).expect("ok");
            mse += (pred - y[i]).powi(2);
        }
        mse /= 80.0;
        let y_var = {
            let m = y.iter().sum::<f64>() / 80.0;
            y.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / 80.0
        };
        // Forest should explain most of the variance.
        assert!(mse < 0.3 * y_var, "mse={mse} var={y_var}");
    }

    #[test]
    fn regressor_prediction_in_target_range() {
        let (x, y) = make_regression(50);
        let cfg = RandomForestConfig::regression();
        let forest = RandomForest::fit_regressor(&x, &y, 50, 2, cfg).expect("ok");
        let pred = forest.predict(&[2.0, 3.0]).expect("ok");
        let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(pred >= ymin - 1.0 && pred <= ymax + 1.0, "pred={pred}");
    }

    #[test]
    fn n_trees_matches_config() {
        let (x, y) = make_classification(30);
        let mut cfg = RandomForestConfig::classification(2);
        cfg.n_trees = 17;
        let forest = RandomForest::fit_classifier(&x, &y, 30, 2, cfg).expect("ok");
        assert_eq!(forest.n_trees(), 17);
    }

    #[test]
    fn seed_reproducibility() {
        let (x, y) = make_classification(40);
        let cfg = RandomForestConfig {
            seed: 123,
            ..RandomForestConfig::classification(2)
        };
        let f1 = RandomForest::fit_classifier(&x, &y, 40, 2, cfg.clone()).expect("ok");
        let f2 = RandomForest::fit_classifier(&x, &y, 40, 2, cfg).expect("ok");
        let p1 = f1.predict_proba(&[0.8, 0.4]).expect("ok");
        let p2 = f2.predict_proba(&[0.8, 0.4]).expect("ok");
        assert_eq!(p1, p2);
    }

    #[test]
    fn max_features_sqrt_default() {
        // max_features = 0 should not error and should train.
        let (x, y) = make_classification(40);
        let cfg = RandomForestConfig::classification(2);
        let forest = RandomForest::fit_classifier(&x, &y, 40, 2, cfg).expect("ok");
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn zero_trees_error() {
        let (x, y) = make_classification(20);
        let cfg = RandomForestConfig {
            n_trees: 0,
            ..RandomForestConfig::classification(2)
        };
        assert!(matches!(
            RandomForest::fit_classifier(&x, &y, 20, 2, cfg),
            Err(TabularError::InvalidTreeCount { .. })
        ));
    }

    #[test]
    fn zero_depth_error() {
        let (x, y) = make_classification(20);
        let cfg = RandomForestConfig {
            max_depth: 0,
            ..RandomForestConfig::classification(2)
        };
        assert!(matches!(
            RandomForest::fit_classifier(&x, &y, 20, 2, cfg),
            Err(TabularError::InvalidTreeDepth { .. })
        ));
    }

    #[test]
    fn label_out_of_range_error() {
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let y = vec![0usize, 5]; // 5 ≥ n_classes=2
        let cfg = RandomForestConfig::classification(2);
        assert!(matches!(
            RandomForest::fit_classifier(&x, &y, 2, 2, cfg),
            Err(TabularError::LabelOutOfRange { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let x = vec![0.0, 1.0, 1.0]; // 3 ≠ n*n_features = 4
        let y = vec![0usize, 1];
        let cfg = RandomForestConfig::classification(2);
        assert!(matches!(
            RandomForest::fit_classifier(&x, &y, 2, 2, cfg),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_samples_error() {
        let cfg = RandomForestConfig::classification(2);
        assert!(matches!(
            RandomForest::fit_classifier(&[], &[], 0, 2, cfg),
            Err(TabularError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn predict_wrong_dim_error() {
        let (x, y) = make_classification(30);
        let cfg = RandomForestConfig::classification(2);
        let forest = RandomForest::fit_classifier(&x, &y, 30, 2, cfg).expect("ok");
        assert!(matches!(
            forest.predict_proba(&[1.0]),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn predict_on_regressor_rejects_proba() {
        let (x, y) = make_regression(30);
        let cfg = RandomForestConfig::regression();
        let forest = RandomForest::fit_regressor(&x, &y, 30, 2, cfg).expect("ok");
        assert!(matches!(
            forest.predict_proba(&[1.0, 2.0]),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn classifier_rejects_regression_predict() {
        let (x, y) = make_classification(30);
        let cfg = RandomForestConfig::classification(2);
        let forest = RandomForest::fit_classifier(&x, &y, 30, 2, cfg).expect("ok");
        assert!(matches!(
            forest.predict(&[1.0, 0.5]),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn bootstrap_fraction_subsample() {
        let (x, y) = make_classification(60);
        let cfg = RandomForestConfig {
            bootstrap_fraction: 0.6,
            n_trees: 15,
            ..RandomForestConfig::classification(2)
        };
        let forest = RandomForest::fit_classifier(&x, &y, 60, 2, cfg).expect("ok");
        assert_eq!(forest.n_trees(), 15);
    }

    #[test]
    fn three_class_problem() {
        // Three bands along feature 0.
        let n = 90;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = i as f64 / n as f64 * 3.0; // [0,3)
            x.push(f0);
            y.push(f0.floor() as usize); // 0,1,2
        }
        let mut cfg = RandomForestConfig::classification(3);
        cfg.n_trees = 25;
        let forest = RandomForest::fit_classifier(&x, &y, n, 1, cfg).expect("ok");
        let p = forest.predict_proba(&[2.5]).expect("ok");
        assert_eq!(p.len(), 3);
        assert_eq!(forest.predict_class(&[2.5]).expect("ok"), 2);
    }

    #[test]
    fn min_samples_split_too_small_error() {
        let (x, y) = make_classification(20);
        let cfg = RandomForestConfig {
            min_samples_split: 1,
            ..RandomForestConfig::classification(2)
        };
        assert!(matches!(
            RandomForest::fit_classifier(&x, &y, 20, 2, cfg),
            Err(TabularError::InvalidParameter { .. })
        ));
    }
}
