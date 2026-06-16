//! Generalized Random Forests (GRF) — honest causal trees with gradient-based splitting.
//!
//! Reference: Athey, S., Tibshirani, J., & Wager, S. (2019). "Generalized random forests."
//! *The Annals of Statistics*, 47(2), 1148-1178.
//!
//! # Overview
//!
//! GRF generalises CART to moment-condition estimands. Two moment types are supported:
//!
//! - **CausalEffect**: Estimates the heterogeneous treatment effect τ(x) via the
//!   Robinson/partially-linear moment `E[(A − E[A|X])(Y − τ(X) · A) | X = x] = 0`.
//!   The gradient-based split criterion maximises the sum-of-squares of pseudo-residuals
//!   `ρ_i = (A_i − Ā)(Y_i − Ȳ)` across the two children.
//!
//! - **LocalLinear**: Estimates local linear regression coefficients β(x) in a leaf by
//!   ridge regression. Split criterion is variance reduction of the outcome.
//!
//! Honesty (Wager & Athey 2018) separates the sample into a **build set** (used to find
//! the optimal split) and an **estimation set** (used to estimate leaf parameters). This
//! prevents over-fitting of the parameter estimates to the splitting decisions.

use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The moment / estimand that the GRF targets in each leaf.
#[derive(Debug, Clone)]
pub enum GrfMoment {
    /// Heterogeneous causal effect τ(x) = E[Y(1) − Y(0) | X = x].
    CausalEffect,
    /// Local linear regression coefficients β(x) with ridge regularisation.
    LocalLinear { regularization: f64 },
}

/// Configuration for [`GrfForest`].
#[derive(Debug, Clone)]
pub struct GrfConfig {
    /// Number of trees in the ensemble.
    pub n_trees: usize,
    /// Fraction of training samples drawn (without replacement) per tree. Must lie in `(0, 1]`.
    pub subsample_fraction: f64,
    /// Whether to apply honest estimation (build/estimation sample separation).
    pub honesty: bool,
    /// Minimum number of build-set samples required in each child node before a split.
    pub min_node_size: usize,
    /// Number of features to try at each candidate split. `0` → `ceil(√p)`.
    pub mtry: usize,
    /// Seed for the per-forest LCG random number generator.
    pub seed: u64,
}

impl Default for GrfConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            subsample_fraction: 0.5,
            honesty: true,
            min_node_size: 5,
            mtry: 0,
            seed: 42,
        }
    }
}

/// Predicted parameters returned by [`GrfForest::predict`].
#[derive(Debug, Clone)]
pub struct GrfPrediction {
    /// Row-major matrix of shape `n × moment_dim`. Each row is the estimated
    /// parameter vector θ̂(x_i) for the corresponding test point.
    pub theta: Vec<f64>,
    /// Row-major variance estimates of shape `n × moment_dim`. Each entry is
    /// the cross-tree variance of the tree-level predictions divided by `n_trees`.
    pub variance: Vec<f64>,
    /// Number of test points.
    pub n: usize,
    /// Dimension of the moment parameter (1 for CausalEffect, p for LocalLinear).
    pub moment_dim: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private tree structures
// ─────────────────────────────────────────────────────────────────────────────

/// Leaf node data: estimated parameters and the estimation-set sample IDs.
struct GrfLeaf {
    theta: Vec<f64>,
    #[allow(dead_code)]
    sample_ids: Vec<usize>,
}

/// A single node in a GRF tree. Internal nodes have `leaf == None`; terminal
/// nodes have `leaf == Some(_)` and the `feature`/`threshold`/`left`/`right`
/// fields are unused.
struct GrfNode {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    leaf: Option<GrfLeaf>,
}

impl GrfNode {
    fn new_leaf(leaf: GrfLeaf) -> Self {
        Self {
            feature: 0,
            threshold: 0.0,
            left: 0,
            right: 0,
            leaf: Some(leaf),
        }
    }

    fn new_internal(feature: usize, threshold: f64, left: usize, right: usize) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            leaf: None,
        }
    }
}

/// A single honest GRF tree.
struct GrfTree {
    nodes: Vec<GrfNode>,
    root: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Forest
// ─────────────────────────────────────────────────────────────────────────────

/// Generalized Random Forest ensemble.
pub struct GrfForest {
    trees: Vec<GrfTree>,
    n_features: usize,
    moment_dim: usize,
}

impl GrfForest {
    /// Fit a GRF ensemble.
    ///
    /// # Parameters
    /// - `x`: row-major `n × p` covariate matrix.
    /// - `y`: outcome vector of length `n`.
    /// - `a`: treatment vector of length `n`.
    /// - `n`, `p`: number of samples and features.
    /// - `cfg`: forest hyperparameters.
    /// - `moment`: target moment type.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] if inputs are inconsistent or parameters
    /// are out of valid range.
    pub fn fit(
        x: &[f64],
        y: &[f64],
        a: &[f64],
        n: usize,
        p: usize,
        cfg: &GrfConfig,
        moment: GrfMoment,
    ) -> CausalResult<Self> {
        // ── validation ───────────────────────────────────────────────────────
        if n < 2 {
            return Err(CausalError::InvalidParameter {
                reason: format!("n must be ≥ 2, got {n}"),
            });
        }
        if p == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "p must be ≥ 1".into(),
            });
        }
        if cfg.subsample_fraction <= 0.0 || cfg.subsample_fraction > 1.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!(
                    "subsample_fraction must be in (0, 1], got {}",
                    cfg.subsample_fraction
                ),
            });
        }
        if cfg.min_node_size == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "min_node_size must be ≥ 1".into(),
            });
        }
        if x.len() != n * p {
            return Err(CausalError::InvalidParameter {
                reason: format!("x.len()={} != n*p={}", x.len(), n * p),
            });
        }
        if y.len() != n {
            return Err(CausalError::InvalidParameter {
                reason: format!("y.len()={} != n={n}", y.len()),
            });
        }
        if a.len() != n {
            return Err(CausalError::InvalidParameter {
                reason: format!("a.len()={} != n={n}", a.len()),
            });
        }

        let mtry_actual = if cfg.mtry == 0 {
            ((p as f64).sqrt().ceil() as usize).max(1)
        } else {
            cfg.mtry.min(p)
        };

        let moment_dim = match &moment {
            GrfMoment::CausalEffect => 1,
            GrfMoment::LocalLinear { .. } => p,
        };

        let mut rng = LcgRng::new(cfg.seed);
        let mut trees = Vec::with_capacity(cfg.n_trees);

        for _ in 0..cfg.n_trees {
            let tree = build_grf_tree(x, y, a, n, p, cfg, &moment, mtry_actual, &mut rng)?;
            trees.push(tree);
        }

        Ok(Self {
            trees,
            n_features: p,
            moment_dim,
        })
    }

    /// Predict the estimated parameter vector for each row of `x_new`.
    ///
    /// # Errors
    /// Returns an error if `x_new.len() != n_new * n_features`.
    pub fn predict(&self, x_new: &[f64], n_new: usize) -> CausalResult<GrfPrediction> {
        let p = self.n_features;
        let d = self.moment_dim;
        if x_new.len() != n_new * p {
            return Err(CausalError::InvalidParameter {
                reason: format!("x_new.len()={} != n_new*p={}", x_new.len(), n_new * p),
            });
        }

        let n_trees = self.trees.len();
        let mut theta = vec![0.0_f64; n_new * d];
        let mut variance = vec![0.0_f64; n_new * d];

        // Collect per-tree predictions then aggregate.
        let mut per_tree: Vec<Vec<f64>> = Vec::with_capacity(n_trees);
        for tree in &self.trees {
            let mut tree_preds = vec![0.0_f64; n_new * d];
            for i in 0..n_new {
                let xi = &x_new[i * p..(i + 1) * p];
                let leaf_theta = predict_node(tree, xi, tree.root);
                for k in 0..d {
                    tree_preds[i * d + k] = leaf_theta[k];
                }
            }
            per_tree.push(tree_preds);
        }

        // Mean across trees.
        for i in 0..n_new {
            for k in 0..d {
                let idx = i * d + k;
                let sum: f64 = per_tree.iter().map(|t| t[idx]).sum();
                theta[idx] = sum / n_trees as f64;
            }
        }

        // Cross-tree variance / n_trees.
        for i in 0..n_new {
            for k in 0..d {
                let idx = i * d + k;
                let mean = theta[idx];
                let var_sum: f64 = per_tree.iter().map(|t| (t[idx] - mean).powi(2)).sum();
                variance[idx] = if n_trees > 1 {
                    var_sum / (n_trees - 1) as f64 / n_trees as f64
                } else {
                    0.0
                };
            }
        }

        Ok(GrfPrediction {
            theta,
            variance,
            n: n_new,
            moment_dim: d,
        })
    }

    /// Estimate the Average Treatment Effect as `(ate, se)`.
    ///
    /// `ate` is the mean of `theta[:, 0]` (first moment dimension) over the
    /// provided sample, and `se` is `sqrt(mean(variance[:, 0]))`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::predict`].
    pub fn ate(&self, x: &[f64], n: usize) -> CausalResult<(f64, f64)> {
        let pred = self.predict(x, n)?;
        let d = self.moment_dim;
        let ate = pred.theta.iter().step_by(d).sum::<f64>() / n as f64;
        let var_mean = pred.variance.iter().step_by(d).sum::<f64>() / n as f64;
        let se = var_mean.sqrt();
        Ok((ate, se))
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of features the forest was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree building
// ─────────────────────────────────────────────────────────────────────────────

/// Build one honest GRF tree via Fisher-Yates subsampling.
fn build_grf_tree(
    x: &[f64],
    y: &[f64],
    a: &[f64],
    n: usize,
    p: usize,
    cfg: &GrfConfig,
    moment: &GrfMoment,
    mtry_actual: usize,
    rng: &mut LcgRng,
) -> CausalResult<GrfTree> {
    // ── subsample (without replacement via Fisher-Yates) ─────────────────────
    let s = ((n as f64 * cfg.subsample_fraction).floor() as usize).max(2);
    let s = s.min(n);

    let mut perm: Vec<usize> = (0..n).collect();
    for i in (n - s..n).rev() {
        let j = rng.next_usize(i + 1);
        perm.swap(i, j);
    }
    let sampled: Vec<usize> = perm[n - s..].to_vec();

    // ── split into build / estimation sets ───────────────────────────────────
    let (build_idx, est_idx) = if cfg.honesty {
        let half = s / 2;
        (sampled[..half].to_vec(), sampled[half..].to_vec())
    } else {
        (sampled.clone(), sampled)
    };

    // ── recursively build the tree ───────────────────────────────────────────
    let mut nodes: Vec<GrfNode> = Vec::new();
    let root = build_node(
        &build_idx,
        &est_idx,
        x,
        y,
        a,
        n,
        p,
        cfg,
        moment,
        mtry_actual,
        &mut nodes,
        rng,
    );

    Ok(GrfTree { nodes, root })
}

/// Recursively build a node. Returns the index of the newly created node.
#[allow(clippy::too_many_arguments)]
fn build_node(
    build_idx: &[usize],
    est_idx: &[usize],
    x: &[f64],
    y: &[f64],
    a: &[f64],
    n: usize,
    p: usize,
    cfg: &GrfConfig,
    moment: &GrfMoment,
    mtry_actual: usize,
    nodes: &mut Vec<GrfNode>,
    rng: &mut LcgRng,
) -> usize {
    // Leaf condition: too few samples to split.
    if build_idx.len() < cfg.min_node_size * 2 || est_idx.is_empty() {
        let theta = estimate_leaf(est_idx, x, y, a, n, p, moment);
        let node_id = nodes.len();
        nodes.push(GrfNode::new_leaf(GrfLeaf {
            theta,
            sample_ids: est_idx.to_vec(),
        }));
        return node_id;
    }

    // ── select random feature subset ─────────────────────────────────────────
    let mut feat_order: Vec<usize> = (0..p).collect();
    for i in (1..p).rev() {
        let j = rng.next_usize(i + 1);
        feat_order.swap(i, j);
    }
    let feat_subset = &feat_order[..mtry_actual];

    // ── find best split ───────────────────────────────────────────────────────
    let best = find_best_split(build_idx, x, y, a, n, p, cfg, moment, feat_subset);

    let (best_feat, best_thresh) = match best {
        None => {
            let theta = estimate_leaf(est_idx, x, y, a, n, p, moment);
            let node_id = nodes.len();
            nodes.push(GrfNode::new_leaf(GrfLeaf {
                theta,
                sample_ids: est_idx.to_vec(),
            }));
            return node_id;
        }
        Some(v) => v,
    };

    // ── partition build / estimation sets ─────────────────────────────────────
    let left_build: Vec<usize> = build_idx
        .iter()
        .copied()
        .filter(|&i| x[i * p + best_feat] < best_thresh)
        .collect();
    let right_build: Vec<usize> = build_idx
        .iter()
        .copied()
        .filter(|&i| x[i * p + best_feat] >= best_thresh)
        .collect();
    let left_est: Vec<usize> = est_idx
        .iter()
        .copied()
        .filter(|&i| x[i * p + best_feat] < best_thresh)
        .collect();
    let right_est: Vec<usize> = est_idx
        .iter()
        .copied()
        .filter(|&i| x[i * p + best_feat] >= best_thresh)
        .collect();

    // Guard: if the split produced degenerate children, make a leaf.
    if left_build.len() < cfg.min_node_size
        || right_build.len() < cfg.min_node_size
        || left_est.is_empty()
        || right_est.is_empty()
    {
        let theta = estimate_leaf(est_idx, x, y, a, n, p, moment);
        let node_id = nodes.len();
        nodes.push(GrfNode::new_leaf(GrfLeaf {
            theta,
            sample_ids: est_idx.to_vec(),
        }));
        return node_id;
    }

    // Reserve space for the current internal node (index assigned now).
    let node_id = nodes.len();
    nodes.push(GrfNode::new_leaf(GrfLeaf {
        theta: vec![],
        sample_ids: vec![],
    })); // placeholder

    let left_id = build_node(
        &left_build,
        &left_est,
        x,
        y,
        a,
        n,
        p,
        cfg,
        moment,
        mtry_actual,
        nodes,
        rng,
    );
    let right_id = build_node(
        &right_build,
        &right_est,
        x,
        y,
        a,
        n,
        p,
        cfg,
        moment,
        mtry_actual,
        nodes,
        rng,
    );

    // Overwrite placeholder with the real internal node.
    nodes[node_id] = GrfNode::new_internal(best_feat, best_thresh, left_id, right_id);
    node_id
}

// ─────────────────────────────────────────────────────────────────────────────
// Split finding
// ─────────────────────────────────────────────────────────────────────────────

/// Search for the best (feature, threshold) split among `feat_subset` features.
/// Returns `None` if no valid split exists.
fn find_best_split(
    build_idx: &[usize],
    x: &[f64],
    y: &[f64],
    a: &[f64],
    n: usize,
    p: usize,
    cfg: &GrfConfig,
    moment: &GrfMoment,
    feat_subset: &[usize],
) -> Option<(usize, f64)> {
    let min_n = cfg.min_node_size;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_feat = 0_usize;
    let mut best_thresh = 0.0_f64;
    let mut found = false;

    // Pre-compute node-level statistics once.
    let (a_mean, y_mean, y_var) = node_stats(build_idx, y, a, n);

    for &feat in feat_subset {
        // Collect feature values for the build set and sort them.
        let mut vals: Vec<(f64, usize)> = build_idx.iter().map(|&i| (x[i * p + feat], i)).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate thresholds (midpoints between consecutive unique values).
        let unique_vals: Vec<f64> = {
            let mut uv: Vec<f64> = vals.iter().map(|v| v.0).collect();
            uv.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
            uv
        };
        if unique_vals.len() < 2 {
            continue;
        }

        for w in unique_vals.windows(2) {
            let thresh = (w[0] + w[1]) * 0.5;

            let left_idx: Vec<usize> = build_idx
                .iter()
                .copied()
                .filter(|&i| x[i * p + feat] < thresh)
                .collect();
            let right_idx: Vec<usize> = build_idx
                .iter()
                .copied()
                .filter(|&i| x[i * p + feat] >= thresh)
                .collect();

            if left_idx.len() < min_n || right_idx.len() < min_n {
                continue;
            }

            let score = split_score(
                &left_idx, &right_idx, y, a, n, a_mean, y_mean, y_var, moment,
            );
            if score > best_score {
                best_score = score;
                best_feat = feat;
                best_thresh = thresh;
                found = true;
            }
        }
    }

    if found {
        Some((best_feat, best_thresh))
    } else {
        None
    }
}

/// Node-level summary statistics: (a_mean, y_mean, y_var).
fn node_stats(idx: &[usize], y: &[f64], a: &[f64], _n: usize) -> (f64, f64, f64) {
    let m = idx.len() as f64;
    if idx.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let a_mean = idx.iter().map(|&i| a[i]).sum::<f64>() / m;
    let y_mean = idx.iter().map(|&i| y[i]).sum::<f64>() / m;
    let y_var = idx.iter().map(|&i| (y[i] - y_mean).powi(2)).sum::<f64>() / m;
    (a_mean, y_mean, y_var)
}

/// Compute the split score for a candidate (left, right) partition.
fn split_score(
    left: &[usize],
    right: &[usize],
    y: &[f64],
    a: &[f64],
    _n: usize,
    a_mean: f64,
    y_mean: f64,
    y_var_node: f64,
    moment: &GrfMoment,
) -> f64 {
    match moment {
        GrfMoment::CausalEffect => {
            // score = (Σ_L ρ)² / |L| + (Σ_R ρ)² / |R|
            // where ρ_i = (A_i − Ā)(Y_i − Ȳ), computed from NODE-level means.
            let rho_l: f64 = left
                .iter()
                .map(|&i| (a[i] - a_mean) * (y[i] - y_mean))
                .sum();
            let rho_r: f64 = right
                .iter()
                .map(|&i| (a[i] - a_mean) * (y[i] - y_mean))
                .sum();
            let n_l = left.len() as f64;
            let n_r = right.len() as f64;
            rho_l * rho_l / n_l + rho_r * rho_r / n_r
        }
        GrfMoment::LocalLinear { .. } => {
            // score = var_node * n_node - var_L * |L| - var_R * |R|
            let n_node = (left.len() + right.len()) as f64;
            let y_mean_l = left.iter().map(|&i| y[i]).sum::<f64>() / left.len().max(1) as f64;
            let y_mean_r = right.iter().map(|&i| y[i]).sum::<f64>() / right.len().max(1) as f64;
            let var_l = left.iter().map(|&i| (y[i] - y_mean_l).powi(2)).sum::<f64>()
                / left.len().max(1) as f64;
            let var_r = right
                .iter()
                .map(|&i| (y[i] - y_mean_r).powi(2))
                .sum::<f64>()
                / right.len().max(1) as f64;
            y_var_node * n_node - var_l * left.len() as f64 - var_r * right.len() as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Leaf estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the leaf parameter vector from the estimation-set samples.
fn estimate_leaf(
    est_idx: &[usize],
    x: &[f64],
    y: &[f64],
    a: &[f64],
    _n: usize,
    p: usize,
    moment: &GrfMoment,
) -> Vec<f64> {
    match moment {
        GrfMoment::CausalEffect => {
            vec![causal_effect_estimate(est_idx, y, a)]
        }
        GrfMoment::LocalLinear { regularization } => {
            local_linear_estimate(est_idx, x, y, p, *regularization)
        }
    }
}

/// Estimate τ = Cov(A, Y) / Var(A) in a leaf.
fn causal_effect_estimate(idx: &[usize], y: &[f64], a: &[f64]) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    let m = idx.len() as f64;
    let a_mean = idx.iter().map(|&i| a[i]).sum::<f64>() / m;
    let y_mean = idx.iter().map(|&i| y[i]).sum::<f64>() / m;
    let num: f64 = idx.iter().map(|&i| (a[i] - a_mean) * (y[i] - y_mean)).sum();
    let den: f64 = idx.iter().map(|&i| (a[i] - a_mean).powi(2)).sum();
    if den < 1e-8 { 0.0 } else { num / den }
}

/// Estimate local linear coefficients β = (X^T X + reg I)^{-1} X^T y.
/// X here is the p-column raw feature matrix (no intercept column).
fn local_linear_estimate(
    idx: &[usize],
    x: &[f64],
    y: &[f64],
    p: usize,
    regularization: f64,
) -> Vec<f64> {
    let m = idx.len();
    if m < 2 || p == 0 {
        return vec![0.0; p];
    }
    // Build XtX (p×p) and Xty (p).
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for &i in idx {
        let xi = &x[i * p..(i + 1) * p];
        for r in 0..p {
            for c in 0..p {
                xtx[r * p + c] += xi[r] * xi[c];
            }
            xty[r] += xi[r] * y[i];
        }
    }
    // Add ridge to diagonal.
    for k in 0..p {
        xtx[k * p + k] += regularization;
    }
    // Solve via Cholesky (fallback to Gauss-Jordan if not SPD).
    cholesky_solve(&xtx, &xty, p)
        .unwrap_or_else(|_| gauss_jordan_solve(&xtx, &xty, p).unwrap_or_else(|_| vec![0.0; p]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree traversal
// ─────────────────────────────────────────────────────────────────────────────

/// Walk the tree from `node_idx` down and return the leaf theta vector.
fn predict_node(tree: &GrfTree, xi: &[f64], node_idx: usize) -> Vec<f64> {
    let node = &tree.nodes[node_idx];
    match &node.leaf {
        Some(leaf) => leaf.theta.clone(),
        None => {
            let next = if xi[node.feature] < node.threshold {
                node.left
            } else {
                node.right
            };
            predict_node(tree, xi, next)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky L L^T factorisation + forward/backward substitution for SPD systems.
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s: f64 = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s < 1e-18 {
                    return Err(CausalError::MatrixSingular);
                }
                l[i * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    // Forward substitution: L z = b.
    let mut z = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * p + j] * z[j];
        }
        z[i] = s / l[i * p + i];
    }
    // Backward substitution: L^T x = z.
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for j in (i + 1)..p {
            s -= l[j * p + i] * x[j];
        }
        x[i] = s / l[i * p + i];
    }
    Ok(x)
}

/// Gauss-Jordan solver with partial pivoting — fallback when Cholesky fails.
fn gauss_jordan_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = p + 1;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p] = b[i];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        x[i] = m[i * cols + p];
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate (X, Y, A) with a constant treatment effect of `tau`.
    fn make_data(
        n: usize,
        p: usize,
        tau: f64,
        noise_sigma: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f64> = (0..n * p).map(|_| rng.next_normal() as f64).collect();
        let a: Vec<f64> = (0..n).map(|_| rng.next_normal() as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| tau * a[i] + noise_sigma * rng.next_normal() as f64)
            .collect();
        (x, y, a)
    }

    fn default_cfg() -> GrfConfig {
        GrfConfig {
            n_trees: 20,
            subsample_fraction: 0.6,
            honesty: true,
            min_node_size: 3,
            mtry: 0,
            seed: 7,
        }
    }

    #[test]
    fn n_trees_count() {
        let (x, y, a) = make_data(50, 2, 1.0, 0.1, 1);
        let cfg = default_cfg();
        let forest = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        assert_eq!(forest.n_trees(), cfg.n_trees);
    }

    #[test]
    fn n_features_correct() {
        let (x, y, a) = make_data(50, 4, 1.0, 0.1, 2);
        let cfg = default_cfg();
        let forest = GrfForest::fit(&x, &y, &a, 50, 4, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        assert_eq!(forest.n_features(), 4);
    }

    #[test]
    fn predict_shape() {
        let (x, y, a) = make_data(50, 2, 1.0, 0.1, 3);
        let cfg = default_cfg();
        let forest = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let x_new: Vec<f64> = (0..10 * 2).map(|i| i as f64 * 0.01).collect();
        let pred = forest.predict(&x_new, 10).expect("predict should succeed");
        assert_eq!(pred.n, 10);
        assert_eq!(pred.moment_dim, 1);
        assert_eq!(pred.theta.len(), 10);
        assert_eq!(pred.variance.len(), 10);
    }

    #[test]
    fn causal_effect_recovers_ate() {
        let n = 200;
        let p = 2;
        let (x, y, a) = make_data(n, p, 2.0, 0.1, 42);
        let cfg = GrfConfig {
            n_trees: 50,
            subsample_fraction: 0.6,
            honesty: true,
            min_node_size: 5,
            mtry: 0,
            seed: 99,
        };
        let forest = GrfForest::fit(&x, &y, &a, n, p, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let (ate, _se) = forest.ate(&x, n).expect("ate should succeed");
        assert!((ate - 2.0).abs() < 0.5, "ATE={ate} not within 0.5 of 2.0");
    }

    #[test]
    fn variance_positive() {
        let (x, y, a) = make_data(60, 2, 1.0, 0.1, 5);
        let cfg = GrfConfig {
            n_trees: 10,
            ..default_cfg()
        };
        let forest = GrfForest::fit(&x, &y, &a, 60, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let pred = forest.predict(&x, 60).expect("predict should succeed");
        for &v in &pred.variance {
            assert!(v >= 0.0, "variance is negative: {v}");
        }
    }

    #[test]
    fn local_linear_slope() {
        let n = 80;
        let p = 2;
        let mut rng = LcgRng::new(11);
        let x: Vec<f64> = (0..n * p).map(|_| rng.next_normal() as f64).collect();
        // Y = 3 * X[:,0] + noise; A all zero.
        let y: Vec<f64> = (0..n)
            .map(|i| 3.0 * x[i * p] + 0.1 * rng.next_normal() as f64)
            .collect();
        let a: Vec<f64> = vec![0.0; n];
        let cfg = GrfConfig {
            n_trees: 30,
            subsample_fraction: 0.7,
            honesty: false,
            min_node_size: 3,
            mtry: 0,
            seed: 13,
        };
        let forest = GrfForest::fit(
            &x,
            &y,
            &a,
            n,
            p,
            &cfg,
            GrfMoment::LocalLinear {
                regularization: 1e-3,
            },
        )
        .expect("value should be present");
        let pred = forest.predict(&x, n).expect("predict should succeed");
        // Average of first coefficient (slope for X[:,0]) should be ≈ 3.
        let mean_coeff0 = pred.theta.iter().step_by(p).sum::<f64>() / n as f64;
        assert!(
            (mean_coeff0 - 3.0).abs() < 1.0,
            "mean_coeff0={mean_coeff0} not within 1.0 of 3.0"
        );
    }

    #[test]
    fn honesty_flag() {
        let (x, y, a) = make_data(50, 2, 1.0, 0.1, 6);
        let mut cfg = default_cfg();
        cfg.honesty = true;
        let r1 = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect);
        assert!(r1.is_ok());
        cfg.honesty = false;
        let r2 = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect);
        assert!(r2.is_ok());
    }

    #[test]
    fn subsample_fraction_1() {
        let (x, y, a) = make_data(50, 2, 1.0, 0.1, 7);
        let cfg = GrfConfig {
            subsample_fraction: 1.0,
            honesty: false,
            ..default_cfg()
        };
        let r = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect);
        assert!(r.is_ok());
    }

    #[test]
    fn min_node_large() {
        let (x, y, a) = make_data(20, 2, 1.0, 0.1, 8);
        // min_node_size = n forces leaf at root (no valid split can have children each ≥ n).
        let cfg = GrfConfig {
            min_node_size: 20,
            n_trees: 5,
            ..default_cfg()
        };
        let r = GrfForest::fit(&x, &y, &a, 20, 2, &cfg, GrfMoment::CausalEffect);
        assert!(r.is_ok());
    }

    #[test]
    fn mtry_zero_auto() {
        let (x, y, a) = make_data(50, 9, 1.0, 0.1, 9);
        // mtry=0 should be converted to ceil(sqrt(9))=3.
        let cfg = GrfConfig {
            mtry: 0,
            n_trees: 5,
            ..default_cfg()
        };
        let r = GrfForest::fit(&x, &y, &a, 50, 9, &cfg, GrfMoment::CausalEffect);
        assert!(r.is_ok());
        // Verify mtry_actual computation.
        let mtry_actual = (9.0_f64.sqrt().ceil() as usize).max(1);
        assert_eq!(mtry_actual, 3);
    }

    #[test]
    fn single_tree() {
        let (x, y, a) = make_data(50, 2, 1.0, 0.1, 10);
        let cfg = GrfConfig {
            n_trees: 1,
            seed: 42,
            ..default_cfg()
        };
        let f1 = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let f2 = GrfForest::fit(&x, &y, &a, 50, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let p1 = f1.predict(&x, 50).expect("predict should succeed");
        let p2 = f2.predict(&x, 50).expect("predict should succeed");
        assert_eq!(p1.theta, p2.theta, "single tree is not deterministic");
    }

    #[test]
    fn predict_more_than_train() {
        let (x, y, a) = make_data(30, 2, 1.0, 0.1, 11);
        let cfg = GrfConfig {
            n_trees: 5,
            ..default_cfg()
        };
        let forest = GrfForest::fit(&x, &y, &a, 30, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let x_new: Vec<f64> = (0..60 * 2).map(|i| i as f64 * 0.01).collect();
        let pred = forest.predict(&x_new, 60);
        assert!(pred.is_ok());
        assert_eq!(pred.expect("pred should be present").n, 60);
    }

    #[test]
    fn err_n_lt_2() {
        let x = vec![1.0_f64; 2];
        let y = vec![1.0_f64; 1];
        let a = vec![0.0_f64; 1];
        let cfg = GrfConfig::default();
        let r = GrfForest::fit(&x, &y, &a, 1, 2, &cfg, GrfMoment::CausalEffect);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn err_p_zero() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0_f64, 2.0];
        let a = vec![0.0_f64, 1.0];
        let cfg = GrfConfig::default();
        let r = GrfForest::fit(&x, &y, &a, 2, 0, &cfg, GrfMoment::CausalEffect);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn err_subsample_gt_1() {
        let (x, y, a) = make_data(20, 2, 1.0, 0.1, 12);
        let cfg = GrfConfig {
            subsample_fraction: 1.5,
            ..default_cfg()
        };
        let r = GrfForest::fit(&x, &y, &a, 20, 2, &cfg, GrfMoment::CausalEffect);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn err_dim_mismatch() {
        let x: Vec<f64> = vec![1.0; 10]; // n*p = 5*2 = 10 ok
        let y = vec![1.0_f64; 4]; // wrong length
        let a = vec![0.0_f64; 5];
        let cfg = GrfConfig {
            n_trees: 2,
            ..default_cfg()
        };
        let r = GrfForest::fit(&x, &y, &a, 5, 2, &cfg, GrfMoment::CausalEffect);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn ate_returns_pair() {
        let (x, y, a) = make_data(40, 2, 1.5, 0.1, 13);
        let cfg = GrfConfig {
            n_trees: 5,
            ..default_cfg()
        };
        let forest = GrfForest::fit(&x, &y, &a, 40, 2, &cfg, GrfMoment::CausalEffect)
            .expect("fit should succeed");
        let result = forest.ate(&x, 40);
        assert!(result.is_ok());
        let (ate, se) = result.expect("result should be present");
        assert!(ate.is_finite());
        assert!(se.is_finite());
        assert!(se >= 0.0);
    }
}
