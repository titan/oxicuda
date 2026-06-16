//! Gradient Boosted Decision Trees (Friedman 2001, Annals of Statistics).
//!
//! Stage-wise additive model:
//!   F_M(x) = F_0 + Σ_{m=1}^{M} ν · h_m(x)
//!
//! where h_m is a CART regression tree fit to pseudo-residuals
//!   rᵢ = -∂L/∂F|_{F=F_{m-1}(xᵢ)}
//!
//! Supports SquaredError and LogLoss (binary cross-entropy) objectives.
//! Tree construction is fully iterative (explicit stack, no recursion).

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Loss ────────────────────────────────────────────────────────────────────

/// Objective / loss function for gradient boosting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbdtLoss {
    /// Mean squared error: pseudo-residuals rᵢ = yᵢ - F(xᵢ).
    SquaredError,
    /// Binary log-loss: pseudo-residuals rᵢ = yᵢ - sigmoid(F(xᵢ)).
    LogLoss,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a `GbdtModel`.
#[derive(Debug, Clone)]
pub struct GbdtConfig {
    /// Number of boosting rounds (trees).
    pub n_estimators: usize,
    /// Learning rate / shrinkage factor ν.
    pub learning_rate: f64,
    /// Maximum tree depth (≥ 1).
    pub max_depth: usize,
    /// Minimum samples required in each leaf.
    pub min_samples_leaf: usize,
    /// Row subsampling ratio in (0, 1].
    pub subsample: f64,
    /// Column subsampling ratio in (0, 1].
    pub col_subsample: f64,
    /// Loss function.
    pub loss: GbdtLoss,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GbdtConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_leaf: 1,
            subsample: 1.0,
            col_subsample: 1.0,
            loss: GbdtLoss::SquaredError,
            seed: 0,
        }
    }
}

// ─── Tree node ───────────────────────────────────────────────────────────────

/// A node in a GBDT regression tree stored as a flat `Vec`.
#[derive(Debug, Clone)]
pub struct GbdtNode {
    /// Whether this node is a leaf.
    pub is_leaf: bool,
    /// Feature index used for the split (internal nodes only).
    pub feature_idx: usize,
    /// Split threshold: go left if `x[feature_idx]` ≤ threshold.
    pub threshold: f64,
    /// Index of the left child in the tree's node vector.
    pub left_child: usize,
    /// Index of the right child in the tree's node vector.
    pub right_child: usize,
    /// Leaf value (used only when `is_leaf = true`).
    pub value: f64,
    /// Number of training samples reaching this node.
    pub n_samples: usize,
}

impl GbdtNode {
    fn new_leaf(value: f64, n_samples: usize) -> Self {
        Self {
            is_leaf: true,
            feature_idx: 0,
            threshold: 0.0,
            left_child: 0,
            right_child: 0,
            value,
            n_samples,
        }
    }

    fn new_internal(feature_idx: usize, threshold: f64, n_samples: usize) -> Self {
        Self {
            is_leaf: false,
            feature_idx,
            threshold,
            left_child: 0,
            right_child: 0,
            value: 0.0,
            n_samples,
        }
    }
}

// ─── Tree ────────────────────────────────────────────────────────────────────

/// A single CART regression tree within the GBDT ensemble.
#[derive(Debug, Clone)]
pub struct GbdtTree {
    /// Flat array of all nodes; node 0 is the root.
    pub nodes: Vec<GbdtNode>,
    /// Cumulative SSR gain attributed to each feature (same length as n_features).
    pub feature_gain: Vec<f64>,
}

impl GbdtTree {
    /// Predict the leaf value for a single sample row `x` (length = n_features).
    pub fn predict_one(&self, x: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return node.value;
            }
            if x[node.feature_idx] <= node.threshold {
                idx = node.left_child;
            } else {
                idx = node.right_child;
            }
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Trained GBDT model.
#[derive(Debug, Clone)]
pub struct GbdtModel {
    /// One tree per boosting iteration.
    pub trees: Vec<GbdtTree>,
    /// Initial prediction F_0.
    pub init_prediction: f64,
    /// Configuration used during training.
    pub config: GbdtConfig,
    /// Number of input features.
    pub n_features: usize,
    /// Mean squared residual at the end of each boosting round.
    pub train_losses: Vec<f64>,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute SSR (sum of squared residuals) for a set of residual values.
#[inline]
fn ssr(residuals: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mean = indices.iter().map(|&i| residuals[i]).sum::<f64>() / indices.len() as f64;
    indices
        .iter()
        .map(|&i| {
            let d = residuals[i] - mean;
            d * d
        })
        .sum()
}

/// Compute the leaf value under SquaredError: mean of residuals.
#[inline]
fn leaf_value_squared_error(residuals: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| residuals[i]).sum::<f64>() / indices.len() as f64
}

/// Compute the leaf value under LogLoss: Σrᵢ / (Σp̂ᵢ(1-p̂ᵢ) + 1e-10).
fn leaf_value_logloss(residuals: &[f64], proba: &[f64], indices: &[usize]) -> f64 {
    let sum_r: f64 = indices.iter().map(|&i| residuals[i]).sum();
    let sum_denom: f64 = indices
        .iter()
        .map(|&i| {
            let p = proba[i];
            p * (1.0 - p)
        })
        .sum::<f64>()
        + 1e-10;
    sum_r / sum_denom
}

/// Partial Fisher-Yates: draw `k` indices uniformly from `0..n`.
fn sample_rows(rng: &mut LcgRng, n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

/// Draw `k` column indices uniformly from `0..n_features`.
fn sample_cols(rng: &mut LcgRng, n_features: usize, k: usize) -> Vec<usize> {
    sample_rows(rng, n_features, k)
}

// ─── Tree builder ─────────────────────────────────────────────────────────────

struct TreeBuildContext<'a> {
    x: &'a [f64],
    n_features: usize,
    residuals: &'a [f64],
    proba: &'a [f64],
    loss: GbdtLoss,
    max_depth: usize,
    min_samples_leaf: usize,
    col_indices: &'a [usize],
}

/// Build one CART regression tree using an explicit stack (no recursion).
///
/// Returns the tree with all nodes and per-feature SSR gains.
fn build_tree(ctx: &TreeBuildContext<'_>) -> GbdtTree {
    let n_features = ctx.n_features;
    let mut nodes: Vec<GbdtNode> = Vec::new();
    let mut feature_gain = vec![0.0_f64; n_features];

    // Bootstrap: create root placeholder node, then process via stack.
    let all_indices: Vec<usize> = (0..ctx.residuals.len()).collect();

    // Allocate root node (placeholder leaf)
    let root_val = compute_leaf_value(ctx, &all_indices);
    nodes.push(GbdtNode::new_leaf(root_val, all_indices.len()));

    // Stack: (node_slot_idx, sample_indices, depth)
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, all_indices, 0)];

    while let Some((node_idx, sample_indices, depth)) = stack.pop() {
        let n = sample_indices.len();

        // Determine whether to split or remain a leaf
        let can_split = depth < ctx.max_depth && n >= ctx.min_samples_leaf * 2;

        if !can_split {
            let val = compute_leaf_value(ctx, &sample_indices);
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].value = val;
            nodes[node_idx].n_samples = n;
            continue;
        }

        // Find best split
        match find_best_split(ctx, &sample_indices) {
            None => {
                let val = compute_leaf_value(ctx, &sample_indices);
                nodes[node_idx].is_leaf = true;
                nodes[node_idx].value = val;
                nodes[node_idx].n_samples = n;
            }
            Some((best_feat, best_thresh, best_gain, left_indices, right_indices)) => {
                feature_gain[best_feat] += best_gain;

                // Allocate left and right child nodes
                let left_idx = nodes.len();
                let left_val = compute_leaf_value(ctx, &left_indices);
                nodes.push(GbdtNode::new_leaf(left_val, left_indices.len()));

                let right_idx = nodes.len();
                let right_val = compute_leaf_value(ctx, &right_indices);
                nodes.push(GbdtNode::new_leaf(right_val, right_indices.len()));

                // Convert current node to internal
                let internal = GbdtNode::new_internal(best_feat, best_thresh, n);
                nodes[node_idx] = GbdtNode {
                    left_child: left_idx,
                    right_child: right_idx,
                    ..internal
                };

                // Push children for further processing
                stack.push((left_idx, left_indices, depth + 1));
                stack.push((right_idx, right_indices, depth + 1));
            }
        }
    }

    GbdtTree {
        nodes,
        feature_gain,
    }
}

/// Compute the leaf value for a given set of sample indices.
fn compute_leaf_value(ctx: &TreeBuildContext<'_>, indices: &[usize]) -> f64 {
    match ctx.loss {
        GbdtLoss::SquaredError => leaf_value_squared_error(ctx.residuals, indices),
        GbdtLoss::LogLoss => leaf_value_logloss(ctx.residuals, ctx.proba, indices),
    }
}

/// Tuple encoding a successful split: `(feat_idx, threshold, gain, left_indices, right_indices)`.
type SplitResult = (usize, f64, f64, Vec<usize>, Vec<usize>);

/// Find the best (feature, threshold) split for the given sample subset.
///
/// Returns `Some((feat_idx, threshold, gain, left_indices, right_indices))`.
fn find_best_split(ctx: &TreeBuildContext<'_>, sample_indices: &[usize]) -> Option<SplitResult> {
    let parent_ssr = ssr(ctx.residuals, sample_indices);
    let min_leaf = ctx.min_samples_leaf;

    let mut best_gain = 0.0_f64;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0_f64;
    let mut best_left: Vec<usize> = Vec::new();
    let mut best_right: Vec<usize> = Vec::new();

    for &feat in ctx.col_indices {
        // Collect (value, sample_idx) pairs and sort by feature value
        let mut vals: Vec<(f64, usize)> = sample_indices
            .iter()
            .map(|&si| (ctx.x[si * ctx.n_features + feat], si))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Try each midpoint threshold between consecutive distinct-value blocks
        let mut i = 0usize;
        while i < vals.len() {
            // Find end of equal-value block starting at i
            let mut j = i + 1;
            while j < vals.len() && (vals[j].0 - vals[i].0).abs() < 1e-15 {
                j += 1;
            }
            // Threshold candidate is midpoint between vals[j-1] and vals[j]
            if j >= vals.len() {
                break;
            }

            let threshold = (vals[j - 1].0 + vals[j].0) * 0.5;

            // Left: indices 0..j, Right: indices j..
            if j >= min_leaf && vals.len() - j >= min_leaf {
                let left_indices: Vec<usize> = vals[..j].iter().map(|&(_, si)| si).collect();
                let right_indices: Vec<usize> = vals[j..].iter().map(|&(_, si)| si).collect();

                let gain = parent_ssr
                    - ssr(ctx.residuals, &left_indices)
                    - ssr(ctx.residuals, &right_indices);

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = threshold;
                    best_left = left_indices;
                    best_right = right_indices;
                }
            }

            i = j;
        }
    }

    if best_gain > 0.0 && !best_left.is_empty() && !best_right.is_empty() {
        Some((best_feat, best_thresh, best_gain, best_left, best_right))
    } else {
        None
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Validate inputs and return an error if anything is out of range.
fn validate_fit_inputs(
    x: &[f64],
    y: &[f64],
    n: usize,
    n_features: usize,
    config: &GbdtConfig,
) -> TabularResult<()> {
    if n == 0 {
        return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
    }
    if n_features == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_features".to_string(),
            msg: "must be >= 1".to_string(),
        });
    }
    if y.len() != n {
        return Err(TabularError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if x.len() != n * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n * n_features,
            got: x.len(),
        });
    }
    if config.subsample <= 0.0 || config.subsample > 1.0 {
        return Err(TabularError::InvalidParameter {
            name: "subsample".to_string(),
            msg: "must be in (0, 1]".to_string(),
        });
    }
    if config.col_subsample <= 0.0 || config.col_subsample > 1.0 {
        return Err(TabularError::InvalidParameter {
            name: "col_subsample".to_string(),
            msg: "must be in (0, 1]".to_string(),
        });
    }
    if config.learning_rate <= 0.0 {
        return Err(TabularError::InvalidParameter {
            name: "learning_rate".to_string(),
            msg: "must be > 0".to_string(),
        });
    }
    if config.max_depth == 0 {
        return Err(TabularError::InvalidParameter {
            name: "max_depth".to_string(),
            msg: "must be >= 1".to_string(),
        });
    }
    Ok(())
}

/// Fit a GBDT model.
///
/// - `x`: row-major feature matrix, shape `[n, n_features]`.
/// - `y`: target labels / values, length `n`.
/// - `n`: number of samples.
/// - `n_features`: number of features.
pub fn gbdt_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    n_features: usize,
    config: &GbdtConfig,
) -> TabularResult<GbdtModel> {
    validate_fit_inputs(x, y, n, n_features, config)?;

    let mut rng = LcgRng::new(config.seed);

    // ── F_0 ──────────────────────────────────────────────────────────────────
    let init_prediction = match config.loss {
        GbdtLoss::SquaredError => y.iter().sum::<f64>() / n as f64,
        GbdtLoss::LogLoss => {
            let p = (y.iter().sum::<f64>() / n as f64).clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln()
        }
    };

    // Current predictions F(x)
    let mut f_pred = vec![init_prediction; n];

    let mut trees: Vec<GbdtTree> = Vec::with_capacity(config.n_estimators);
    let mut train_losses: Vec<f64> = Vec::with_capacity(config.n_estimators);

    // Handle n_estimators == 0
    if config.n_estimators == 0 {
        return Ok(GbdtModel {
            trees,
            init_prediction,
            config: config.clone(),
            n_features,
            train_losses,
        });
    }

    for _round in 0..config.n_estimators {
        // ── Row subsampling ──────────────────────────────────────────────────
        let k_rows = ((n as f64 * config.subsample).floor() as usize).max(1);
        let row_indices = if config.subsample >= 1.0 {
            (0..n).collect::<Vec<_>>()
        } else {
            sample_rows(&mut rng, n, k_rows)
        };

        // ── Column subsampling ────────────────────────────────────────────────
        let k_cols = ((n_features as f64 * config.col_subsample).ceil() as usize).max(1);
        let col_indices = if config.col_subsample >= 1.0 {
            (0..n_features).collect::<Vec<_>>()
        } else {
            sample_cols(&mut rng, n_features, k_cols)
        };

        // ── Pseudo-residuals and probabilities ───────────────────────────────
        let mut residuals = vec![0.0_f64; n];
        let mut proba = vec![0.0_f64; n];
        match config.loss {
            GbdtLoss::SquaredError => {
                for (i, r) in residuals.iter_mut().enumerate() {
                    *r = y[i] - f_pred[i];
                }
            }
            GbdtLoss::LogLoss => {
                for i in 0..n {
                    let p = sigmoid(f_pred[i]);
                    proba[i] = p;
                    residuals[i] = y[i] - p;
                }
            }
        }

        // Build subsampled x and residuals for the tree
        let n_sub = row_indices.len();
        let mut sub_x = vec![0.0_f64; n_sub * n_features];
        let mut sub_residuals = vec![0.0_f64; n_sub];
        let mut sub_proba = vec![0.0_f64; n_sub];
        for (new_i, &orig_i) in row_indices.iter().enumerate() {
            sub_x[new_i * n_features..(new_i + 1) * n_features]
                .copy_from_slice(&x[orig_i * n_features..(orig_i + 1) * n_features]);
            sub_residuals[new_i] = residuals[orig_i];
            sub_proba[new_i] = proba[orig_i];
        }

        let tree_ctx = TreeBuildContext {
            x: &sub_x,
            n_features,
            residuals: &sub_residuals,
            proba: &sub_proba,
            loss: config.loss,
            max_depth: config.max_depth,
            min_samples_leaf: config.min_samples_leaf,
            col_indices: &col_indices,
        };

        let tree = build_tree(&tree_ctx);

        // ── Update F(x) for all samples (using full x) ───────────────────────
        for i in 0..n {
            let row = &x[i * n_features..(i + 1) * n_features];
            let delta = tree.predict_one(row);
            f_pred[i] += config.learning_rate * delta;
        }

        // ── Training loss: recompute residuals after update ───────────────────
        let mse: f64 = match config.loss {
            GbdtLoss::SquaredError => {
                (0..n)
                    .map(|i| {
                        let r = y[i] - f_pred[i];
                        r * r
                    })
                    .sum::<f64>()
                    / n as f64
            }
            GbdtLoss::LogLoss => {
                (0..n)
                    .map(|i| {
                        let p = sigmoid(f_pred[i]);
                        let r = y[i] - p;
                        r * r
                    })
                    .sum::<f64>()
                    / n as f64
            }
        };
        train_losses.push(mse);
        trees.push(tree);
    }

    Ok(GbdtModel {
        trees,
        init_prediction,
        config: config.clone(),
        n_features,
        train_losses,
    })
}

/// Predict raw scores F(x) for each sample.
///
/// For `SquaredError` this is the regression prediction.
/// For `LogLoss` apply `sigmoid` to obtain class probabilities.
pub fn gbdt_predict(model: &GbdtModel, x: &[f64], n: usize) -> TabularResult<Vec<f64>> {
    if x.len() != n * model.n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n * model.n_features,
            got: x.len(),
        });
    }
    let nf = model.n_features;
    let mut preds = vec![model.init_prediction; n];
    for tree in &model.trees {
        for i in 0..n {
            let row = &x[i * nf..(i + 1) * nf];
            preds[i] += model.config.learning_rate * tree.predict_one(row);
        }
    }
    Ok(preds)
}

/// Predict class probabilities (sigmoid of raw scores) for LogLoss models.
///
/// For SquaredError models this returns sigmoid-transformed values (not meaningful
/// for regression but kept for API symmetry).
pub fn gbdt_predict_proba(model: &GbdtModel, x: &[f64], n: usize) -> TabularResult<Vec<f64>> {
    let raw = gbdt_predict(model, x, n)?;
    Ok(raw.into_iter().map(sigmoid).collect())
}

/// Compute normalised feature importances (sum to 1) from cumulative SSR gains.
///
/// If all gains are zero (e.g., no splits occurred), returns a zero vector.
/// Callers should check the total before interpreting the result.
#[must_use]
pub fn gbdt_feature_importances(model: &GbdtModel) -> Vec<f64> {
    let nf = model.n_features;
    let mut gains = vec![0.0_f64; nf];
    for tree in &model.trees {
        for (f, &g) in tree.feature_gain.iter().enumerate() {
            gains[f] += g;
        }
    }
    let total: f64 = gains.iter().sum();
    if total <= 0.0 {
        return gains;
    }
    gains.iter().map(|&g| g / total).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        if n == 1 {
            return vec![start];
        }
        (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    fn lcg_noise(seed: u64, n: usize, scale: f64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| (rng.next_f32() as f64 - 0.5) * 2.0 * scale)
            .collect()
    }

    fn make_regression(n: usize, slope: f64, noise_scale: f64) -> (Vec<f64>, Vec<f64>) {
        let x = linspace(0.0, 1.0, n);
        let noise = lcg_noise(1234, n, noise_scale);
        let y: Vec<f64> = x
            .iter()
            .zip(noise.iter())
            .map(|(&xi, &ni)| slope * xi + ni)
            .collect();
        (x, y)
    }

    // ── Test 1: Regression convergence ───────────────────────────────────────
    #[test]
    fn test_regression_convergence() {
        let n = 50;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: 3,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg);
        assert!(model.is_ok(), "fit should succeed");
        let model = model.expect("model should be present");
        let preds = gbdt_predict(&model, &x, n);
        assert!(preds.is_ok());
        let preds = preds.expect("preds should be present");
        let rmse = (preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &yi)| (p - yi).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 0.3, "RMSE={rmse} should be < 0.3");
    }

    // ── Test 2: Train loss monotone non-increasing ────────────────────────────
    #[test]
    fn test_train_loss_monotone() {
        let n = 50;
        let (x, y) = make_regression(n, 2.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 50,
            learning_rate: 0.2,
            max_depth: 3,
            subsample: 1.0,
            col_subsample: 1.0,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        for i in 1..model.train_losses.len() {
            assert!(
                model.train_losses[i] <= model.train_losses[i - 1] + 1e-9,
                "loss not monotone at round {i}: {} > {}",
                model.train_losses[i],
                model.train_losses[i - 1]
            );
        }
    }

    // ── Test 3: n_estimators=1 ────────────────────────────────────────────────
    #[test]
    fn test_single_estimator() {
        let n = 20;
        let (x, y) = make_regression(n, 1.0, 0.05);
        let cfg = GbdtConfig {
            n_estimators: 1,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        assert_eq!(preds.len(), n);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    // ── Test 4: max_depth=1 (stump ensemble) ─────────────────────────────────
    #[test]
    fn test_stump_ensemble() {
        let n = 30;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 20,
            max_depth: 1,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        assert_eq!(preds.len(), n);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    // ── Test 5: LogLoss on linearly separable data ────────────────────────────
    #[test]
    fn test_logloss_classification() {
        let n = 40;
        let mut x = vec![0.0_f64; n * 2];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let xi = if i < n / 2 {
                -(i as f64 + 1.0) / 5.0
            } else {
                (i as f64 - n as f64 / 2.0 + 1.0) / 5.0
            };
            x[i * 2] = xi;
            x[i * 2 + 1] = 0.0;
            y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
        }
        let cfg = GbdtConfig {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: 3,
            loss: GbdtLoss::LogLoss,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 2, &cfg).expect("gbdt_fit should succeed");
        let proba = gbdt_predict_proba(&model, &x, n).expect("gbdt_predict_proba should succeed");
        for i in 0..n {
            if y[i] > 0.5 {
                assert!(proba[i] > 0.5, "positive sample {i} proba={}", proba[i]);
            } else {
                assert!(proba[i] < 0.5, "negative sample {i} proba={}", proba[i]);
            }
        }
    }

    // ── Test 6: predict output shape ─────────────────────────────────────────
    #[test]
    fn test_predict_shape() {
        let n = 25;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 10,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        assert_eq!(preds.len(), n);
    }

    // ── Test 7: init_prediction == mean(y) for SquaredError ──────────────────
    #[test]
    fn test_init_prediction_squared_error() {
        let n = 20;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let cfg = GbdtConfig {
            n_estimators: 10,
            loss: GbdtLoss::SquaredError,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        assert!(
            (model.init_prediction - mean_y).abs() < 1e-10,
            "init_prediction={} mean_y={}",
            model.init_prediction,
            mean_y
        );
    }

    // ── Test 8: row subsample=0.8 ─────────────────────────────────────────────
    #[test]
    fn test_subsample_row() {
        let n = 50;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 20,
            subsample: 0.8,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg);
        assert!(model.is_ok());
        let preds = gbdt_predict(&model.expect("model should be present"), &x, n)
            .expect("value should be present");
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    // ── Test 9: col_subsample=0.5 ────────────────────────────────────────────
    #[test]
    fn test_subsample_col() {
        let n = 40;
        let n_feat = 4;
        let x: Vec<f64> = (0..n * n_feat)
            .map(|i| (i as f64) / (n * n_feat) as f64)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let cfg = GbdtConfig {
            n_estimators: 10,
            col_subsample: 0.5,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, n_feat, &cfg);
        assert!(model.is_ok());
        let preds = gbdt_predict(&model.expect("model should be present"), &x, n)
            .expect("value should be present");
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    // ── Test 10: n_features=1 ────────────────────────────────────────────────
    #[test]
    fn test_n_features_one() {
        let n = 30;
        let (x, y) = make_regression(n, 1.0, 0.05);
        let cfg = GbdtConfig {
            n_estimators: 10,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        assert_eq!(model.n_features, 1);
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        assert_eq!(preds.len(), n);
    }

    // ── Test 11: feature_importances length ──────────────────────────────────
    #[test]
    fn test_feature_importances_len() {
        let n = 30;
        let n_feat = 3;
        let x: Vec<f64> = (0..n * n_feat).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let cfg = GbdtConfig {
            n_estimators: 10,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, n_feat, &cfg).expect("gbdt_fit should succeed");
        let imp = gbdt_feature_importances(&model);
        assert_eq!(imp.len(), n_feat);
    }

    // ── Test 12: feature_importances sum to 1 ────────────────────────────────
    #[test]
    fn test_feature_importances_sum_to_one() {
        let n = 40;
        let n_feat = 3;
        let x: Vec<f64> = (0..n * n_feat)
            .map(|i| (i as f64) / (n * n_feat) as f64)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let cfg = GbdtConfig {
            n_estimators: 20,
            max_depth: 3,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, n_feat, &cfg).expect("gbdt_fit should succeed");
        let imp = gbdt_feature_importances(&model);
        let total: f64 = imp.iter().sum();
        // Skip check if all zeros (no splits occurred)
        if total > 0.0 {
            assert!((total - 1.0).abs() < 1e-6, "importances sum={total}");
        }
    }

    // ── Test 13: feature_importances non-negative ─────────────────────────────
    #[test]
    fn test_feature_importances_non_negative() {
        let n = 30;
        let n_feat = 2;
        let x: Vec<f64> = (0..n * n_feat).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let cfg = GbdtConfig {
            n_estimators: 10,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, n_feat, &cfg).expect("gbdt_fit should succeed");
        let imp = gbdt_feature_importances(&model);
        assert!(imp.iter().all(|&v| v >= 0.0));
    }

    // ── Test 14: more trees → lower train loss ────────────────────────────────
    #[test]
    fn test_more_trees_lower_loss() {
        let n = 40;
        let (x, y) = make_regression(n, 1.5, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        assert!(
            model.train_losses[99] <= model.train_losses[9] + 1e-9,
            "loss at 100 trees ({}) should be <= loss at 10 trees ({})",
            model.train_losses[99],
            model.train_losses[9]
        );
    }

    // ── Test 15: seed reproducibility ────────────────────────────────────────
    #[test]
    fn test_seed_reproducibility() {
        let n = 30;
        let (x, y) = make_regression(n, 1.0, 0.1);
        let cfg = GbdtConfig {
            n_estimators: 10,
            subsample: 0.8,
            seed: 999,
            ..Default::default()
        };
        let m1 = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let m2 = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let p1 = gbdt_predict(&m1, &x, n).expect("gbdt_predict should succeed");
        let p2 = gbdt_predict(&m2, &x, n).expect("gbdt_predict should succeed");
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a, b, "predictions differ across same seed");
        }
    }

    // ── Test 16: predict on train → MAE < 0.5 ────────────────────────────────
    #[test]
    fn test_predict_on_train_mae() {
        let n = 50;
        let (x, y) = make_regression(n, 1.0, 0.05);
        let cfg = GbdtConfig {
            n_estimators: 50,
            learning_rate: 0.2,
            max_depth: 4,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        let mae = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &yi)| (p - yi).abs())
            .sum::<f64>()
            / n as f64;
        assert!(mae < 0.5, "MAE={mae} should be < 0.5");
    }

    // ── Test 17: predict_proba in [0,1] ──────────────────────────────────────
    #[test]
    fn test_predict_proba_range() {
        let n = 20;
        let mut x = vec![0.0_f64; n * 2];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            x[i * 2] = if i < n / 2 { -1.0 } else { 1.0 };
            x[i * 2 + 1] = 0.0;
            y[i] = if i < n / 2 { 0.0 } else { 1.0 };
        }
        let cfg = GbdtConfig {
            n_estimators: 20,
            loss: GbdtLoss::LogLoss,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 2, &cfg).expect("gbdt_fit should succeed");
        let proba = gbdt_predict_proba(&model, &x, n).expect("gbdt_predict_proba should succeed");
        assert!(proba.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    // ── Test 18: n==0 → InsufficientSamples ─────────────────────────────────
    #[test]
    fn test_n_zero_error() {
        let cfg = GbdtConfig::default();
        let result = gbdt_fit(&[], &[], 0, 1, &cfg);
        assert!(matches!(
            result,
            Err(TabularError::InsufficientSamples { need: 1, got: 0 })
        ));
    }

    // ── Test 19: y.len() != n → DimensionMismatch ────────────────────────────
    #[test]
    fn test_y_len_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0]; // wrong length
        let cfg = GbdtConfig::default();
        let result = gbdt_fit(&x, &y, 3, 1, &cfg);
        assert!(matches!(
            result,
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    // ── Test 20: subsample=0 → InvalidParameter ───────────────────────────────
    #[test]
    fn test_subsample_zero_error() {
        let n = 10;
        let (x, y) = make_regression(n, 1.0, 0.0);
        let cfg = GbdtConfig {
            subsample: 0.0,
            ..Default::default()
        };
        let result = gbdt_fit(&x, &y, n, 1, &cfg);
        assert!(matches!(result, Err(TabularError::InvalidParameter { .. })));
    }

    // ── Test 21: learning_rate=0 → InvalidParameter ───────────────────────────
    #[test]
    fn test_learning_rate_zero_error() {
        let n = 10;
        let (x, y) = make_regression(n, 1.0, 0.0);
        let cfg = GbdtConfig {
            learning_rate: 0.0,
            ..Default::default()
        };
        let result = gbdt_fit(&x, &y, n, 1, &cfg);
        assert!(matches!(result, Err(TabularError::InvalidParameter { .. })));
    }

    // ── Test 22: n_estimators=0 → empty model, predict = init_prediction ──────
    #[test]
    fn test_n_estimators_zero() {
        let n = 10;
        let (x, y) = make_regression(n, 1.0, 0.0);
        let cfg = GbdtConfig {
            n_estimators: 0,
            ..Default::default()
        };
        let model = gbdt_fit(&x, &y, n, 1, &cfg).expect("gbdt_fit should succeed");
        assert!(model.trees.is_empty());
        let preds = gbdt_predict(&model, &x, n).expect("gbdt_predict should succeed");
        assert_eq!(preds.len(), n);
        for &p in &preds {
            assert!(
                (p - model.init_prediction).abs() < 1e-12,
                "pred={p} should equal init={}",
                model.init_prediction
            );
        }
    }
}
