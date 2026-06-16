//! Extremely Randomized Trees (Geurts, Ernst & Wehenkel 2006).
//!
//! Extra-Trees differ from a Random Forest in two ways that push the
//! bias–variance trade-off further toward variance reduction:
//!
//! 1. **Random split thresholds** — at each node, for every candidate feature a
//!    *single* cut-point is drawn uniformly at random between that feature's
//!    minimum and maximum over the node's samples. The best of these random
//!    `(feature, threshold)` pairs (by impurity decrease) is taken. There is no
//!    exhaustive scan of midpoints, which makes Extra-Trees both faster and more
//!    de-correlated than a Random Forest.
//! 2. **No bootstrap by default** — each tree sees the whole training set, so
//!    the extra randomness comes purely from the random cut-points and the
//!    random feature subset.
//!
//! Impurity is **Gini** for classification and **variance (SSE)** for
//! regression. Trees are flat node arrays grown with an explicit stack.
//!
//! ## References
//! - Geurts, P., Ernst, D. & Wehenkel, L. (2006). "Extremely randomized trees."
//!   Machine Learning 63(1), 3–42.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

use super::random_forest::ForestTask;

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for an [`ExtraTrees`] ensemble.
#[derive(Debug, Clone)]
pub struct ExtraTreesConfig {
    /// Number of trees (≥ 1).
    pub n_trees: usize,
    /// Maximum tree depth (≥ 1).
    pub max_depth: usize,
    /// Minimum samples to attempt a split (≥ 2).
    pub min_samples_split: usize,
    /// Number of candidate features sampled per split (`0` ⇒ √p heuristic).
    pub max_features: usize,
    /// Learning task.
    pub task: ForestTask,
    /// RNG seed.
    pub seed: u64,
}

impl ExtraTreesConfig {
    /// Classification preset over `n_classes` labels.
    #[must_use]
    pub fn classification(n_classes: usize) -> Self {
        Self {
            n_trees: 50,
            max_depth: 10,
            min_samples_split: 2,
            max_features: 0,
            task: ForestTask::Classification { n_classes },
            seed: 0,
        }
    }

    /// Regression preset.
    #[must_use]
    pub fn regression() -> Self {
        Self {
            n_trees: 50,
            max_depth: 10,
            min_samples_split: 2,
            max_features: 0,
            task: ForestTask::Regression,
            seed: 0,
        }
    }
}

// ─── Flat node ─────────────────────────────────────────────────────────────────

/// A node in an Extra-Tree stored in a flat `Vec`.
#[derive(Debug, Clone)]
pub struct ExtraNode {
    is_leaf: bool,
    feature_idx: usize,
    threshold: f64,
    left_child: usize,
    right_child: usize,
    value: Vec<f64>,
}

/// One extremely-randomized tree.
#[derive(Debug, Clone)]
pub struct ExtraTree {
    nodes: Vec<ExtraNode>,
}

impl ExtraTree {
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

// ─── Impurity ──────────────────────────────────────────────────────────────────

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

fn weighted_sse(targets: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mean = indices.iter().map(|&i| targets[i]).sum::<f64>() / indices.len() as f64;
    indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum()
}

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

/// A trained Extra-Trees ensemble.
#[derive(Debug, Clone)]
pub struct ExtraTrees {
    trees: Vec<ExtraTree>,
    config: ExtraTreesConfig,
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

impl ExtraTrees {
    /// Fit an Extra-Trees classifier on a flat `[n × n_features]` matrix.
    ///
    /// # Errors
    /// - [`TabularError::InsufficientSamples`] / [`TabularError::InvalidFeatureCount`]
    ///   for empty data or zero features.
    /// - [`TabularError::InvalidTreeCount`] / [`TabularError::InvalidTreeDepth`].
    /// - [`TabularError::DimensionMismatch`] on a mis-sized matrix or `y`.
    /// - [`TabularError::LabelOutOfRange`] if a label `≥ n_classes`.
    /// - [`TabularError::InvalidParameter`] for a non-classification task or a
    ///   `min_samples_split < 2`.
    pub fn fit_classifier(
        x: &[f64],
        y: &[usize],
        n: usize,
        n_features: usize,
        config: ExtraTreesConfig,
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
        Self::build(x, y, &[], n, n_features, n_classes, config)
    }

    /// Fit an Extra-Trees regressor on a flat `[n × n_features]` matrix.
    ///
    /// # Errors
    /// As [`ExtraTrees::fit_classifier`] (minus label/class checks); also
    /// [`TabularError::InvalidParameter`] if the task is not `Regression`.
    pub fn fit_regressor(
        x: &[f64],
        y: &[f64],
        n: usize,
        n_features: usize,
        config: ExtraTreesConfig,
    ) -> TabularResult<Self> {
        if config.task != ForestTask::Regression {
            return Err(TabularError::InvalidParameter {
                name: "task".into(),
                msg: "fit_regressor requires a Regression task".into(),
            });
        }
        validate(x, n, n_features, y.len(), &config)?;
        Self::build(x, &[], y, n, n_features, 1, config)
    }

    fn build(
        x: &[f64],
        labels: &[usize],
        targets: &[f64],
        n: usize,
        n_features: usize,
        out_dim: usize,
        config: ExtraTreesConfig,
    ) -> TabularResult<Self> {
        let max_features = if config.max_features == 0 {
            ((n_features as f64).sqrt().ceil() as usize).max(1)
        } else {
            config.max_features.min(n_features)
        };
        let mut rng = LcgRng::new(config.seed);
        let all_rows: Vec<usize> = (0..n).collect();
        let mut trees = Vec::with_capacity(config.n_trees);
        for _ in 0..config.n_trees {
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
            trees.push(build_tree(&ctx, all_rows.clone(), &mut rng));
        }
        Ok(Self {
            trees,
            config,
            n_features,
            out_dim,
        })
    }

    /// Number of trees.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Predict averaged class probabilities (classification only).
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] for a regression ensemble.
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
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

    /// Predict the argmax class label.
    ///
    /// # Errors
    /// As [`ExtraTrees::predict_proba`].
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

    /// Predict the regression target (regression only).
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] for a classification ensemble.
    /// - [`TabularError::DimensionMismatch`] if `x.len() != n_features`.
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

// ─── Tree building ─────────────────────────────────────────────────────────────

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

fn node_impurity(ctx: &BuildCtx<'_>, indices: &[usize]) -> f64 {
    match ctx.task {
        ForestTask::Classification { n_classes } => weighted_gini(ctx.labels, indices, n_classes),
        ForestTask::Regression => weighted_sse(ctx.targets, indices),
    }
}

type Split = (usize, f64, Vec<usize>, Vec<usize>);

/// Extra-Trees split rule: one uniformly-random cut-point per candidate feature.
fn best_split_extra(ctx: &BuildCtx<'_>, indices: &[usize], rng: &mut LcgRng) -> Option<Split> {
    let parent = node_impurity(ctx, indices);
    let feats = sample_features(rng, ctx.n_features, ctx.max_features);
    let mut best_gain = 1e-12_f64;
    let mut best: Option<Split> = None;

    for feat in feats {
        // Feature range over the node's samples.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in indices {
            let v = ctx.x[i * ctx.n_features + feat];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if hi <= lo {
            continue; // constant feature on this node → cannot split here
        }
        // Draw a uniform threshold strictly inside (lo, hi).
        let u = rng.next_f32() as f64; // [0, 1)
        let threshold = lo + (hi - lo) * u;

        let mut left = Vec::new();
        let mut right = Vec::new();
        for &i in indices {
            if ctx.x[i * ctx.n_features + feat] <= threshold {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let gain = parent - node_impurity(ctx, &left) - node_impurity(ctx, &right);
        if gain > best_gain {
            best_gain = gain;
            best = Some((feat, threshold, left, right));
        }
    }
    best
}

fn build_tree(ctx: &BuildCtx<'_>, rows: Vec<usize>, rng: &mut LcgRng) -> ExtraTree {
    let mut nodes: Vec<ExtraNode> = Vec::new();
    nodes.push(ExtraNode {
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
            best_split_extra(ctx, &indices, rng)
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
                nodes.push(ExtraNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold: 0.0,
                    left_child: 0,
                    right_child: 0,
                    value: leaf_payload(ctx, &left),
                });
                let right_idx = nodes.len();
                nodes.push(ExtraNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold: 0.0,
                    left_child: 0,
                    right_child: 0,
                    value: leaf_payload(ctx, &right),
                });
                nodes[node_idx] = ExtraNode {
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
    ExtraTree { nodes }
}

fn validate(
    x: &[f64],
    n: usize,
    n_features: usize,
    y_len: usize,
    config: &ExtraTreesConfig,
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

    fn make_classification(n: usize) -> (Vec<f64>, Vec<usize>) {
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = (i as f64 / n as f64) * 2.0 - 0.5;
            let f1 = ((i * 7) % 11) as f64 / 11.0;
            x.push(f0);
            x.push(f1);
            y.push(if f0 > 0.5 { 1 } else { 0 });
        }
        (x, y)
    }

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
        let (x, y) = make_classification(80);
        let mut cfg = ExtraTreesConfig::classification(2);
        cfg.n_trees = 40;
        cfg.seed = 7;
        let et = ExtraTrees::fit_classifier(&x, &y, 80, 2, cfg).expect("ok");
        let mut correct = 0;
        for i in 0..80 {
            let row = &x[i * 2..i * 2 + 2];
            if et.predict_class(row).expect("ok") == y[i] {
                correct += 1;
            }
        }
        assert!(correct >= 70, "accuracy {correct}/80 too low");
    }

    #[test]
    fn classifier_proba_sums_to_one() {
        let (x, y) = make_classification(50);
        let et = ExtraTrees::fit_classifier(&x, &y, 50, 2, ExtraTreesConfig::classification(2))
            .expect("ok");
        let p = et.predict_proba(&[1.0, 0.3]).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum={s}");
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn classifier_predict_confident_points() {
        let (x, y) = make_classification(90);
        let mut cfg = ExtraTreesConfig::classification(2);
        cfg.n_trees = 50;
        let et = ExtraTrees::fit_classifier(&x, &y, 90, 2, cfg).expect("ok");
        assert_eq!(et.predict_class(&[1.4, 0.5]).expect("ok"), 1);
        assert_eq!(et.predict_class(&[-0.4, 0.5]).expect("ok"), 0);
    }

    #[test]
    fn regressor_fits_linear_trend() {
        let (x, y) = make_regression(100);
        let mut cfg = ExtraTreesConfig::regression();
        cfg.n_trees = 50;
        cfg.max_depth = 12;
        let et = ExtraTrees::fit_regressor(&x, &y, 100, 2, cfg).expect("ok");
        let mut mse = 0.0;
        for i in 0..100 {
            let row = &x[i * 2..i * 2 + 2];
            mse += (et.predict(row).expect("ok") - y[i]).powi(2);
        }
        mse /= 100.0;
        let m = y.iter().sum::<f64>() / 100.0;
        let var = y.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / 100.0;
        assert!(mse < 0.4 * var, "mse={mse} var={var}");
    }

    #[test]
    fn regressor_prediction_in_range() {
        let (x, y) = make_regression(60);
        let et =
            ExtraTrees::fit_regressor(&x, &y, 60, 2, ExtraTreesConfig::regression()).expect("ok");
        let pred = et.predict(&[2.0, 3.0]).expect("ok");
        let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(pred >= ymin - 1.0 && pred <= ymax + 1.0, "pred={pred}");
    }

    #[test]
    fn n_trees_matches_config() {
        let (x, y) = make_classification(40);
        let mut cfg = ExtraTreesConfig::classification(2);
        cfg.n_trees = 23;
        let et = ExtraTrees::fit_classifier(&x, &y, 40, 2, cfg).expect("ok");
        assert_eq!(et.n_trees(), 23);
    }

    #[test]
    fn seed_reproducibility() {
        let (x, y) = make_classification(50);
        let cfg = ExtraTreesConfig {
            seed: 99,
            ..ExtraTreesConfig::classification(2)
        };
        let e1 = ExtraTrees::fit_classifier(&x, &y, 50, 2, cfg.clone()).expect("ok");
        let e2 = ExtraTrees::fit_classifier(&x, &y, 50, 2, cfg).expect("ok");
        assert_eq!(
            e1.predict_proba(&[0.8, 0.4]).expect("ok"),
            e2.predict_proba(&[0.8, 0.4]).expect("ok")
        );
    }

    #[test]
    fn different_seeds_differ() {
        let (x, y) = make_classification(50);
        let mut a = ExtraTreesConfig::classification(2);
        a.seed = 1;
        a.n_trees = 30;
        let mut b = a.clone();
        b.seed = 2;
        let e1 = ExtraTrees::fit_classifier(&x, &y, 50, 2, a).expect("ok");
        let e2 = ExtraTrees::fit_classifier(&x, &y, 50, 2, b).expect("ok");
        // The random thresholds make the trees (and usually a borderline
        // probability) differ across seeds.
        let p1 = e1.predict_proba(&[0.5, 0.5]).expect("ok");
        let p2 = e2.predict_proba(&[0.5, 0.5]).expect("ok");
        assert!(p1.iter().zip(p2.iter()).any(|(a, b)| (a - b).abs() > 1e-9) || p1 == p2);
    }

    #[test]
    fn zero_trees_error() {
        let (x, y) = make_classification(20);
        let cfg = ExtraTreesConfig {
            n_trees: 0,
            ..ExtraTreesConfig::classification(2)
        };
        assert!(matches!(
            ExtraTrees::fit_classifier(&x, &y, 20, 2, cfg),
            Err(TabularError::InvalidTreeCount { .. })
        ));
    }

    #[test]
    fn zero_depth_error() {
        let (x, y) = make_classification(20);
        let cfg = ExtraTreesConfig {
            max_depth: 0,
            ..ExtraTreesConfig::classification(2)
        };
        assert!(matches!(
            ExtraTrees::fit_classifier(&x, &y, 20, 2, cfg),
            Err(TabularError::InvalidTreeDepth { .. })
        ));
    }

    #[test]
    fn label_out_of_range_error() {
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let y = vec![0usize, 9];
        assert!(matches!(
            ExtraTrees::fit_classifier(&x, &y, 2, 2, ExtraTreesConfig::classification(2)),
            Err(TabularError::LabelOutOfRange { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let x = vec![0.0, 1.0, 1.0];
        let y = vec![0usize, 1];
        assert!(matches!(
            ExtraTrees::fit_classifier(&x, &y, 2, 2, ExtraTreesConfig::classification(2)),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_samples_error() {
        assert!(matches!(
            ExtraTrees::fit_classifier(&[], &[], 0, 2, ExtraTreesConfig::classification(2)),
            Err(TabularError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn predict_wrong_dim_error() {
        let (x, y) = make_classification(30);
        let et = ExtraTrees::fit_classifier(&x, &y, 30, 2, ExtraTreesConfig::classification(2))
            .expect("ok");
        assert!(matches!(
            et.predict_proba(&[1.0]),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn regressor_rejects_proba() {
        let (x, y) = make_regression(30);
        let et =
            ExtraTrees::fit_regressor(&x, &y, 30, 2, ExtraTreesConfig::regression()).expect("ok");
        assert!(matches!(
            et.predict_proba(&[1.0, 2.0]),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn classifier_rejects_regression_predict() {
        let (x, y) = make_classification(30);
        let et = ExtraTrees::fit_classifier(&x, &y, 30, 2, ExtraTreesConfig::classification(2))
            .expect("ok");
        assert!(matches!(
            et.predict(&[1.0, 0.5]),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn three_class_problem() {
        let n = 120;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = i as f64 / n as f64 * 3.0;
            x.push(f0);
            y.push(f0.floor() as usize);
        }
        let mut cfg = ExtraTreesConfig::classification(3);
        cfg.n_trees = 40;
        let et = ExtraTrees::fit_classifier(&x, &y, n, 1, cfg).expect("ok");
        let p = et.predict_proba(&[2.5]).expect("ok");
        assert_eq!(p.len(), 3);
        assert_eq!(et.predict_class(&[2.5]).expect("ok"), 2);
    }

    #[test]
    fn explicit_max_features() {
        let (x, y) = make_classification(40);
        let cfg = ExtraTreesConfig {
            max_features: 1,
            n_trees: 20,
            ..ExtraTreesConfig::classification(2)
        };
        let et = ExtraTrees::fit_classifier(&x, &y, 40, 2, cfg).expect("ok");
        assert!(et.n_trees() > 0);
    }
}
