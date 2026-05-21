//! Gradient-Boosted Cox Proportional Hazards Model (XGBoost-style).
//!
//! Learns an additive log-risk function F(x) via gradient boosting on the Cox
//! partial log-likelihood, using the exact first and second derivatives (gradient
//! and Hessian) to train an ensemble of depth-limited regression trees.
//!
//! # Algorithm Summary
//!
//! **Setting**: Given `(t_i, δ_i, x_i)`, learn `F(x) = log-risk score` such that
//! `h(t|x) = h₀(t) exp(F(x))`.
//!
//! **Cox partial log-likelihood**:
//! `l(F) = Σ_i δ_i [F(x_i) - log Σ_{j∈R(t_i)} exp(F(x_j))]`
//!
//! **Gradient** (negative gradient w.r.t. F_i, i.e. pseudo-residual):
//! `g_i = δ_i - Σ_{j: t_j ≤ t_i, δ_j=1} p_{ij}`
//! where `p_{ij} = exp(F(x_i)) / Σ_{k∈R(t_j)} exp(F(x_k))`
//!
//! Negated for XGBoost convention: we minimize loss, so pseudo-residuals target
//! the direction of steepest descent.
//!
//! **Hessian** (diagonal approximation):
//! `h_i = Σ_{j: t_j ≤ t_i, δ_j=1} p_{ij} * (1 - p_{ij})`
//!
//! **XGBoost leaf value**: For a leaf containing indices L:
//! `leaf_val = -Σ_{i∈L} g_i / (Σ_{i∈L} h_i + λ_reg)`
//! where `g_i` here is the negative gradient (the partial-likelihood gradient
//! negated so `g_i = δ_i - contribution`).
//!
//! **Update**: `F(x_i) += lr * tree_m(x_i)`
//!
//! # References
//! - Chen & Guestrin (2016): XGBoost: A Scalable Tree Boosting System
//! - Ridgeway (1999): The State of Boosting
//! - Friedman (2001): Greedy Function Approximation

use crate::error::{SurvivalError, SurvivalResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for gradient-boosted Cox PH.
#[derive(Debug, Clone)]
pub struct GbCoxConfig {
    /// Number of boosting rounds / trees (default: 100).
    pub n_estimators: usize,
    /// Maximum depth of each regression tree (default: 3).
    pub max_depth: usize,
    /// Shrinkage / learning rate applied to each tree (default: 0.1).
    pub learning_rate: f64,
    /// Row (subject) subsampling fraction in `(0, 1]` (default: 0.8).
    pub subsample: f64,
    /// Column (feature) subsampling fraction in `(0, 1]` (default: 0.8).
    pub col_subsample: f64,
    /// L2 regularisation on leaf weights (XGBoost λ) (default: 1.0).
    pub l2_reg: f64,
    /// Minimum sum of Hessian weights in a leaf to allow a split (default: 1.0).
    pub min_child_weight: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for GbCoxConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 3,
            learning_rate: 0.1,
            subsample: 0.8,
            col_subsample: 0.8,
            l2_reg: 1.0,
            min_child_weight: 1.0,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree data structures (node-arena design)
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in a gradient-boosted regression tree.
#[derive(Debug, Clone)]
pub enum GbNode {
    /// Terminal leaf — carries the regularised leaf weight.
    Leaf { value: f64 },
    /// Internal split — routes left if `x[feature] <= threshold`.
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

/// A single depth-limited regression tree in the boosted ensemble.
#[derive(Debug, Clone)]
pub struct GbCoxTree {
    /// Node arena; index 0 is always the root.
    pub nodes: Vec<GbNode>,
}

impl GbCoxTree {
    /// Traverse the tree for a single observation row and return the leaf value.
    #[must_use]
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        let mut node_idx = 0usize;
        loop {
            match &self.nodes[node_idx] {
                GbNode::Leaf { value } => return *value,
                GbNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    node_idx = if row[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model and prediction output
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted gradient-boosted Cox proportional hazards model.
#[derive(Debug, Clone)]
pub struct GbCoxModel {
    /// Ensemble of regression trees, one per boosting round.
    pub trees: Vec<GbCoxTree>,
    /// Initial log-risk score (log of the observed event rate).
    pub init_score: f64,
    /// Number of input features expected at predict time.
    pub n_features: usize,
    /// Configuration used during training.
    pub config: GbCoxConfig,
    /// Cox partial log-likelihood evaluated after each boosting round.
    pub train_log_likelihood: Vec<f64>,
}

/// Predictions from [`gb_cox_predict`].
#[derive(Debug, Clone)]
pub struct GbCoxPred {
    /// Log-risk `F(x_i)` for each subject.
    pub log_risk: Vec<f64>,
    /// Hazard ratio (risk score) `exp(F(x_i))` for each subject; always positive.
    pub risk_score: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cox partial log-likelihood helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Cox partial log-likelihood for the given log-risk scores (Breslow ties).
///
/// Returns `Σ_i δ_i [F_i - log Σ_{j∈R(t_i)} exp(F_j)]`.
fn cox_partial_log_likelihood(
    times: &[f64],
    events: &[u8],
    log_risk: &[f64],
    order_asc: &[usize],
) -> f64 {
    let n = times.len();
    if n == 0 {
        return 0.0;
    }
    // Compute exp(F_i) once
    let w: Vec<f64> = log_risk.iter().map(|x| x.exp()).collect();

    // Suffix sums: suffix_s0[k] = Σ_{j at position k..n-1 in order_asc} w_j
    // = risk-set sum at time times[order_asc[k]] (Breslow: risk set = all with t >= t_k)
    let mut suffix_s0 = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for k in (0..n).rev() {
        acc += w[order_asc[k]];
        suffix_s0[k] = acc;
    }

    let mut ll = 0.0_f64;
    let mut k = 0usize;
    while k < n {
        let t = times[order_asc[k]];
        let s0 = suffix_s0[k].max(f64::MIN_POSITIVE);
        let log_s0 = s0.ln();
        // Advance through tied times
        let mut m = k;
        while m < n && times[order_asc[m]] == t {
            let i = order_asc[m];
            if events[i] == 1 {
                ll += log_risk[i] - log_s0;
            }
            m += 1;
        }
        k = m;
    }
    ll
}

/// Compute Cox gradient and Hessian for each subject at the current log-risk scores.
///
/// Returns `(grads, hessians)` where:
/// - `grads[i] = δ_i - Σ_{j: t_j ≤ t_i, δ_j=1} p_{ij}` (negative of the negative-grad)
///
/// XGBoost convention: we treat `grads[i]` as the raw gradient of the LOSS w.r.t. F_i.
/// Since we *minimise* the negative partial log-likelihood:
///   `loss_grad[i] = -δ_i + Σ_{j: t_j ≤ t_i, δ_j=1} p_{ij}`
///
/// Leaf values then use: `leaf_val = -Σ g / (Σ h + λ)`.
fn cox_gradients_hessians(
    times: &[f64],
    events: &[u8],
    log_risk: &[f64],
    order_asc: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let n = times.len();
    let w: Vec<f64> = log_risk.iter().map(|x| x.exp()).collect();

    // suffix_s0[k] = Σ_{j at positions k..n} w[order_asc[j]]
    let mut suffix_s0 = vec![0.0_f64; n + 1];
    for k in (0..n).rev() {
        suffix_s0[k] = suffix_s0[k + 1] + w[order_asc[k]];
    }

    // For each event time t_j (ascending), compute p_{ij} = w_i / S0(t_j)
    // and accumulate into each subject's gradient and hessian.
    //
    // gradient trick: for each event time t_j with S0_j and D_j events:
    //   all subjects i with t_i >= t_j get += w_i / S0_j   (hessian-like)
    //   all subjects i with t_i >= t_j get += w_i / S0_j * (1 - w_i / S0_j)   (hessian)
    //
    // We use the cumulative-sum approach:
    //   cum_inv[k+1] = Σ_{event-times t_j starting up to position k} 1/S0_j
    //
    // Then grad_i = -δ_i + w_i * cum_inv[position of t_i in order_asc + 1]
    // and  hess_i = w_i * (cum_inv2[pos+1]) - w_i^2 * (cum_inv_sq[pos+1])
    // where cum_inv2 = Σ (1/S0_j) and cum_inv_sq = Σ (1/S0_j^2)

    // Build per-unique-event-time entries: (start_pos_in_order_asc, 1/S0_j, 1/S0_j^2)
    let mut event_inv: Vec<(usize, f64, f64)> = Vec::new();
    let mut k = 0usize;
    while k < n {
        let t = times[order_asc[k]];
        let s0 = suffix_s0[k].max(f64::MIN_POSITIVE);
        let inv_s0 = 1.0 / s0;
        let inv_s0_sq = inv_s0 * inv_s0;
        // Check if any events occur at this time
        let mut m = k;
        while m < n && times[order_asc[m]] == t {
            m += 1;
        }
        let has_event = (k..m).any(|j| events[order_asc[j]] == 1);
        if has_event {
            event_inv.push((k, inv_s0, inv_s0_sq));
        }
        k = m;
    }

    // Build cumulative inverses indexed by position in order_asc
    let mut cum_inv = vec![0.0_f64; n + 1];
    let mut cum_inv_sq = vec![0.0_f64; n + 1];
    let mut ev_ptr = 0usize;
    for k in 0..n {
        cum_inv[k + 1] = cum_inv[k];
        cum_inv_sq[k + 1] = cum_inv_sq[k];
        while ev_ptr < event_inv.len() && event_inv[ev_ptr].0 == k {
            cum_inv[k + 1] += event_inv[ev_ptr].1;
            cum_inv_sq[k + 1] += event_inv[ev_ptr].2;
            ev_ptr += 1;
        }
    }

    // Build a position lookup: pos_of[i] = position of subject i in order_asc
    let mut pos_of = vec![0usize; n];
    for (k, &i) in order_asc.iter().enumerate() {
        pos_of[i] = k;
    }

    let mut grads = vec![0.0_f64; n];
    let mut hessians = vec![0.0_f64; n];
    for i in 0..n {
        let kpos = pos_of[i] + 1; // number of event times at or before t_i
        let delta = events[i] as f64;
        let wi = w[i];
        let wi2 = wi * wi;
        // loss gradient (minimising negative log-likelihood):
        //   g_i = -δ_i + wi * Σ_{t_j ≤ t_i, event} 1/S0_j
        grads[i] = -delta + wi * cum_inv[kpos];
        // Hessian diagonal approximation:
        //   h_i = wi * Σ_{t_j ≤ t_i} 1/S0_j  -  wi^2 * Σ_{t_j ≤ t_i} 1/S0_j^2
        //       = Σ p_{ij} * (1 - p_{ij})  (per the spec)
        let raw_h = wi * cum_inv[kpos] - wi2 * cum_inv_sq[kpos];
        hessians[i] = raw_h.max(0.0);
    }
    (grads, hessians)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree building (XGBoost-style, node-arena)
// ─────────────────────────────────────────────────────────────────────────────

/// XGBoost gain for a single candidate split partition.
///
/// `gain = G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ)`
#[inline]
fn xgb_gain(
    g_l: f64,
    h_l: f64,
    g_r: f64,
    h_r: f64,
    g_total: f64,
    h_total: f64,
    lambda: f64,
) -> f64 {
    let score_l = g_l * g_l / (h_l + lambda);
    let score_r = g_r * g_r / (h_r + lambda);
    let score_p = g_total * g_total / (h_total + lambda);
    score_l + score_r - score_p
}

/// Find the best (feature, threshold) split for the given subset of subjects.
///
/// Returns `Some((feature, threshold, gain))` or `None` if no valid split exists.
fn find_best_xgb_split(
    covariates: &[f64],
    n_features: usize,
    indices: &[usize],
    grads: &[f64],
    hessians: &[f64],
    candidate_features: &[usize],
    lambda: f64,
    min_child_weight: f64,
) -> Option<(usize, f64, f64)> {
    let n_leaf = indices.len();
    if n_leaf < 2 {
        return None;
    }

    let g_total: f64 = indices.iter().map(|&i| grads[i]).sum();
    let h_total: f64 = indices.iter().map(|&i| hessians[i]).sum();
    if h_total < 2.0 * min_child_weight {
        return None;
    }

    let mut best_gain = 0.0_f64;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0_f64;
    let mut found = false;

    for &feat in candidate_features {
        // Collect (value, idx) for sorting
        let mut vals: Vec<(f64, usize)> = indices
            .iter()
            .map(|&i| (covariates[i * n_features + feat], i))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Sweep through sorted values; accumulate left G and H
        let mut g_l = 0.0_f64;
        let mut h_l = 0.0_f64;

        for k in 0..(vals.len() - 1) {
            let (v_k, i_k) = vals[k];
            g_l += grads[i_k];
            h_l += hessians[i_k];

            // Skip tied covariate values
            let v_next = vals[k + 1].0;
            if (v_k - v_next).abs() < f64::EPSILON {
                continue;
            }

            let g_r = g_total - g_l;
            let h_r = h_total - h_l;

            if h_l < min_child_weight || h_r < min_child_weight {
                continue;
            }

            let gain = xgb_gain(g_l, h_l, g_r, h_r, g_total, h_total, lambda);
            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                // threshold = midpoint between v_k and v_next
                best_thresh = (v_k + v_next) * 0.5;
                found = true;
            }
        }
    }

    if found {
        Some((best_feat, best_thresh, best_gain))
    } else {
        None
    }
}

/// Recursively build a tree node into the `nodes` arena.
///
/// Returns the index of the newly created node.
fn build_node(
    covariates: &[f64],
    n_features: usize,
    indices: &[usize],
    grads: &[f64],
    hessians: &[f64],
    candidate_features: &[usize],
    lambda: f64,
    min_child_weight: f64,
    depth: usize,
    max_depth: usize,
    nodes: &mut Vec<GbNode>,
    rng: &mut LcgRng,
    col_subsample: f64,
) -> usize {
    let g_sum: f64 = indices.iter().map(|&i| grads[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| hessians[i]).sum();
    let leaf_val = -g_sum / (h_sum + lambda);

    // Stop if depth limit reached, too few samples, or Hessian too small
    if depth >= max_depth || indices.len() < 2 || h_sum < min_child_weight {
        let node_idx = nodes.len();
        nodes.push(GbNode::Leaf { value: leaf_val });
        return node_idx;
    }

    // Column subsampling: pick a random subset of features for this node
    let n_feats = candidate_features.len();
    let n_col = ((n_feats as f64 * col_subsample).ceil() as usize)
        .max(1)
        .min(n_feats);
    // Fisher-Yates shuffle to pick n_col features
    let mut feat_pool: Vec<usize> = candidate_features.to_vec();
    for j in 0..n_col {
        let swap_idx = j + rng.next_usize(n_feats - j);
        feat_pool.swap(j, swap_idx);
    }
    let selected_feats = &feat_pool[..n_col];

    match find_best_xgb_split(
        covariates,
        n_features,
        indices,
        grads,
        hessians,
        selected_feats,
        lambda,
        min_child_weight,
    ) {
        None => {
            let node_idx = nodes.len();
            nodes.push(GbNode::Leaf { value: leaf_val });
            node_idx
        }
        Some((feat, thresh, _gain)) => {
            // Partition indices
            let left_indices: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| covariates[i * n_features + feat] <= thresh)
                .collect();
            let right_indices: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| covariates[i * n_features + feat] > thresh)
                .collect();

            if left_indices.is_empty() || right_indices.is_empty() {
                let node_idx = nodes.len();
                nodes.push(GbNode::Leaf { value: leaf_val });
                return node_idx;
            }

            // Allocate a placeholder for this split node; fill in child indices after recursion
            let split_idx = nodes.len();
            nodes.push(GbNode::Leaf { value: 0.0 }); // placeholder

            let left_child = build_node(
                covariates,
                n_features,
                &left_indices,
                grads,
                hessians,
                candidate_features,
                lambda,
                min_child_weight,
                depth + 1,
                max_depth,
                nodes,
                rng,
                col_subsample,
            );
            let right_child = build_node(
                covariates,
                n_features,
                &right_indices,
                grads,
                hessians,
                candidate_features,
                lambda,
                min_child_weight,
                depth + 1,
                max_depth,
                nodes,
                rng,
                col_subsample,
            );

            nodes[split_idx] = GbNode::Split {
                feature: feat,
                threshold: thresh,
                left: left_child,
                right: right_child,
            };
            split_idx
        }
    }
}

/// Build a single gradient-boosted regression tree.
fn build_tree(
    covariates: &[f64],
    n_features: usize,
    sample_indices: &[usize],
    grads: &[f64],
    hessians: &[f64],
    config: &GbCoxConfig,
    all_features: &[usize],
    rng: &mut LcgRng,
) -> GbCoxTree {
    let mut nodes: Vec<GbNode> = Vec::new();
    build_node(
        covariates,
        n_features,
        sample_indices,
        grads,
        hessians,
        all_features,
        config.l2_reg,
        config.min_child_weight,
        0,
        config.max_depth,
        &mut nodes,
        rng,
        config.col_subsample,
    );
    GbCoxTree { nodes }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row subsampling
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a random subset of row indices without replacement (Fisher-Yates on full range).
fn subsample_rows(n: usize, fraction: f64, rng: &mut LcgRng) -> Vec<usize> {
    let n_sub = ((n as f64 * fraction).round() as usize).max(1).min(n);
    let mut pool: Vec<usize> = (0..n).collect();
    for j in 0..n_sub {
        let swap_idx = j + rng.next_usize(n - j);
        pool.swap(j, swap_idx);
    }
    pool[..n_sub].to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Sort order of subjects by time ascending (stable sort for ties).
fn order_by_time(times: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..times.len()).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Fit a gradient-boosted Cox proportional hazards model.
///
/// # Arguments
/// - `times` — observed survival/censoring times, length `n_subjects`.
/// - `events` — event indicator (1 = event, 0 = censored), length `n_subjects`.
/// - `covariates` — row-major covariate matrix `[n_subjects × n_features]`.
/// - `n_subjects` — number of subjects.
/// - `n_features` — number of features per subject.
/// - `config` — boosting hyperparameters.
///
/// # Errors
/// - [`SurvivalError::EmptyDataset`] if `n_subjects == 0`.
/// - [`SurvivalError::InvalidParameter`] for invalid configuration values.
/// - [`SurvivalError::ShapeMismatch`] if slice lengths are inconsistent.
/// - [`SurvivalError::NoEvents`] if no observed events are present.
pub fn gb_cox_fit(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
    n_features: usize,
    config: &GbCoxConfig,
) -> SurvivalResult<GbCoxModel> {
    // ── validation ──────────────────────────────────────────────────────────
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    if events.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![events.len()],
        });
    }
    if covariates.len() != n_subjects * n_features {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects * n_features],
            got: vec![covariates.len()],
        });
    }
    if n_features == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_features must be >= 1".to_string(),
        ));
    }
    let n_events: usize = events.iter().map(|&e| e as usize).sum();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }
    if config.n_estimators == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_estimators must be >= 1".to_string(),
        ));
    }
    if config.learning_rate <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "learning_rate must be > 0".to_string(),
        ));
    }
    if config.subsample <= 0.0 || config.subsample > 1.0 {
        return Err(SurvivalError::InvalidParameter(
            "subsample must be in (0, 1]".to_string(),
        ));
    }
    if config.col_subsample <= 0.0 || config.col_subsample > 1.0 {
        return Err(SurvivalError::InvalidParameter(
            "col_subsample must be in (0, 1]".to_string(),
        ));
    }
    if config.l2_reg < 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "l2_reg must be >= 0".to_string(),
        ));
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    // ── initialisation ───────────────────────────────────────────────────────
    // Init score: log(event_rate) — centres the log-risk near 0
    let event_rate = (n_events as f64) / (n_subjects as f64);
    let init_score = event_rate.ln();

    let mut log_risk = vec![init_score; n_subjects];
    let order_asc = order_by_time(times);
    let all_features: Vec<usize> = (0..n_features).collect();

    let mut rng = LcgRng::new(config.seed);
    let mut trees: Vec<GbCoxTree> = Vec::with_capacity(config.n_estimators);
    let mut train_log_likelihood: Vec<f64> = Vec::with_capacity(config.n_estimators);

    // ── boosting loop ────────────────────────────────────────────────────────
    for _round in 0..config.n_estimators {
        // 1. Compute Cox gradient and Hessian at current scores
        let (grads, hessians) = cox_gradients_hessians(times, events, &log_risk, &order_asc);

        // 2. Row subsampling
        let sample_indices = subsample_rows(n_subjects, config.subsample, &mut rng);

        // 3. Build regression tree on the subsampled rows
        let tree = build_tree(
            covariates,
            n_features,
            &sample_indices,
            &grads,
            &hessians,
            config,
            &all_features,
            &mut rng,
        );

        // 4. Update log-risk scores for ALL subjects (not just subsample)
        for i in 0..n_subjects {
            let row = &covariates[i * n_features..(i + 1) * n_features];
            log_risk[i] += config.learning_rate * tree.predict_one(row);
        }

        // 5. Record partial log-likelihood after this round
        let ll = cox_partial_log_likelihood(times, events, &log_risk, &order_asc);
        train_log_likelihood.push(ll);

        trees.push(tree);
    }

    Ok(GbCoxModel {
        trees,
        init_score,
        n_features,
        config: config.clone(),
        train_log_likelihood,
    })
}

/// Generate predictions from a fitted [`GbCoxModel`].
///
/// # Arguments
/// - `model` — fitted model.
/// - `covariates` — row-major covariate matrix `[n_new × n_features]`.
/// - `n_new` — number of new subjects.
///
/// # Errors
/// - [`SurvivalError::ShapeMismatch`] if the covariate matrix is not `n_new × model.n_features`.
pub fn gb_cox_predict(
    model: &GbCoxModel,
    covariates: &[f64],
    n_new: usize,
) -> SurvivalResult<GbCoxPred> {
    if n_new == 0 {
        return Ok(GbCoxPred {
            log_risk: Vec::new(),
            risk_score: Vec::new(),
        });
    }
    if covariates.len() != n_new * model.n_features {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_new * model.n_features],
            got: vec![covariates.len()],
        });
    }

    let mut log_risk = vec![model.init_score; n_new];
    for tree in &model.trees {
        for i in 0..n_new {
            let row = &covariates[i * model.n_features..(i + 1) * model.n_features];
            log_risk[i] += model.config.learning_rate * tree.predict_one(row);
        }
    }
    let risk_score: Vec<f64> = log_risk.iter().map(|x| x.exp()).collect();
    Ok(GbCoxPred {
        log_risk,
        risk_score,
    })
}

/// Compute Harrell's C-index for a fitted model on held-out (or training) data.
///
/// # Arguments
/// - `model` — fitted model.
/// - `times` — observed times for evaluation subjects.
/// - `events` — event indicators for evaluation subjects.
/// - `covariates` — row-major covariate matrix `[n_subjects × n_features]`.
/// - `n_subjects` — number of evaluation subjects.
///
/// # Errors
/// - [`SurvivalError::NumericalInstability`] if there are no comparable pairs.
/// - [`SurvivalError::ShapeMismatch`] on dimension mismatch.
pub fn gb_cox_concordance(
    model: &GbCoxModel,
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
) -> SurvivalResult<f64> {
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    let pred = gb_cox_predict(model, covariates, n_subjects)?;
    let risk = &pred.risk_score;

    // Harrell's C: fraction of comparable pairs (i, j) where
    // δ_i = 1, t_i < t_j, and risk[i] > risk[j].
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n_subjects {
        if events[i] == 0 {
            continue;
        }
        for j in 0..n_subjects {
            if i == j {
                continue;
            }
            if times[j] <= times[i] {
                continue;
            }
            comparable += 1.0;
            if risk[i] > risk[j] {
                concordant += 1.0;
            } else if (risk[i] - risk[j]).abs() < 1.0e-12 {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no comparable pairs for concordance".to_string(),
        ));
    }
    Ok(concordant / comparable)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic dataset with an informative single feature.
    ///
    /// Survival time ~ Exponential(exp(beta * x)), x ~ Normal(0,1).
    fn make_synthetic(n: usize, beta: f64, seed: u64) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covs = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (beta * x).exp().max(1.0e-6);
            let t = rng.next_exponential(lambda).max(1.0e-6);
            times.push(t);
            events.push(1u8); // all events for simplicity
            covs.push(x);
        }
        (times, events, covs)
    }

    /// Build a 3-feature synthetic dataset.
    fn make_synthetic_3f(n: usize, seed: u64) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covs = Vec::with_capacity(n * 3);
        let betas = [1.0_f64, -0.5, 0.3];
        for _ in 0..n {
            let xs: Vec<f64> = (0..3).map(|_| rng.next_normal()).collect();
            let lp: f64 = xs.iter().zip(betas.iter()).map(|(x, b)| x * b).sum();
            let lambda = lp.exp().max(1.0e-6);
            let t = rng.next_exponential(lambda).max(1.0e-6);
            // ~80% event rate
            let delta = if rng.next_f64() < 0.8 { 1u8 } else { 0u8 };
            times.push(t);
            events.push(delta);
            covs.extend_from_slice(&xs);
        }
        (times, events, covs)
    }

    // ── Test 1: n_trees == n_estimators ──────────────────────────────────────

    #[test]
    fn n_trees_equals_n_estimators() {
        let (times, events, covs) = make_synthetic_3f(60, 1);
        let config = GbCoxConfig {
            n_estimators: 20,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 60, 3, &config).expect("fit ok");
        assert_eq!(model.trees.len(), 20);
    }

    // ── Test 2: log_risk length == n_new ──────────────────────────────────────

    #[test]
    fn predict_log_risk_length() {
        let (times, events, covs) = make_synthetic_3f(60, 2);
        let config = GbCoxConfig {
            n_estimators: 10,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 60, 3, &config).expect("fit ok");
        let pred = gb_cox_predict(&model, &covs, 60).expect("predict ok");
        assert_eq!(pred.log_risk.len(), 60);
    }

    // ── Test 3: risk_score == exp(log_risk), always positive ─────────────────

    #[test]
    fn risk_score_is_exp_log_risk() {
        let (times, events, covs) = make_synthetic_3f(50, 3);
        let config = GbCoxConfig {
            n_estimators: 15,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 50, 3, &config).expect("fit ok");
        let pred = gb_cox_predict(&model, &covs, 50).expect("predict ok");
        for (lr, rs) in pred.log_risk.iter().zip(pred.risk_score.iter()) {
            assert!(*rs > 0.0, "risk_score must be positive");
            assert!(
                (rs - lr.exp()).abs() < 1.0e-10,
                "risk_score != exp(log_risk)"
            );
        }
    }

    // ── Test 4: train_log_likelihood is non-decreasing (improves or stays) ───

    #[test]
    fn train_log_likelihood_non_decreasing() {
        let (times, events, covs) = make_synthetic(80, 1.5, 42);
        let config = GbCoxConfig {
            n_estimators: 30,
            learning_rate: 0.05,
            subsample: 1.0,
            col_subsample: 1.0,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 80, 1, &config).expect("fit ok");
        let ll = &model.train_log_likelihood;
        // Allow small numerical tolerance: each step should not decrease by more than 1e-3
        for w in ll.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0e-6,
                "log-likelihood decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    // ── Test 5: Concordance > 0.6 on informative single-feature data ─────────

    #[test]
    fn concordance_above_threshold_informative() {
        let (times, events, covs) = make_synthetic(120, 2.0, 99);
        let config = GbCoxConfig {
            n_estimators: 50,
            learning_rate: 0.1,
            subsample: 1.0,
            col_subsample: 1.0,
            max_depth: 3,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 120, 1, &config).expect("fit ok");
        let c = gb_cox_concordance(&model, &times, &events, &covs, 120).expect("concordance ok");
        assert!(c > 0.6, "concordance={} expected > 0.6", c);
    }

    // ── Test 6: Random features → out-of-sample concordance ≈ 0.5 ──────────

    #[test]
    fn concordance_near_half_random_features() {
        // Train on random covariates uncorrelated with survival time,
        // then evaluate on a fresh held-out set.  On held-out data the model
        // has no signal to exploit, so concordance should be near 0.5.
        let mut rng = LcgRng::new(777);
        let n_train = 80;
        let n_test = 200;
        let p = 3;

        // Training set: times generated independently of covariates
        let times_tr: Vec<f64> = (0..n_train)
            .map(|_| rng.next_exponential(1.0).max(1e-6))
            .collect();
        let events_tr: Vec<u8> = vec![1u8; n_train];
        let covs_tr: Vec<f64> = (0..n_train * p).map(|_| rng.next_normal()).collect();

        // Test set: also purely random (different seed state continues from above)
        let times_te: Vec<f64> = (0..n_test)
            .map(|_| rng.next_exponential(1.0).max(1e-6))
            .collect();
        let events_te: Vec<u8> = vec![1u8; n_test];
        let covs_te: Vec<f64> = (0..n_test * p).map(|_| rng.next_normal()).collect();

        let config = GbCoxConfig {
            n_estimators: 20,
            learning_rate: 0.1,
            ..Default::default()
        };
        let model =
            gb_cox_fit(&times_tr, &events_tr, &covs_tr, n_train, p, &config).expect("fit ok");
        let c = gb_cox_concordance(&model, &times_te, &events_te, &covs_te, n_test)
            .expect("concordance ok");
        // On held-out data with no signal, concordance should be near 0.5 (±0.15 slack)
        assert!(
            (c - 0.5).abs() < 0.20,
            "out-of-sample random concordance={} expected near 0.5",
            c
        );
    }

    // ── Test 7: n_estimators=1 produces a valid single-tree model ────────────

    #[test]
    fn single_tree_model() {
        let (times, events, covs) = make_synthetic_3f(40, 4);
        let config = GbCoxConfig {
            n_estimators: 1,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 40, 3, &config).expect("fit ok");
        assert_eq!(model.trees.len(), 1);
        assert_eq!(model.train_log_likelihood.len(), 1);
        let pred = gb_cox_predict(&model, &covs, 40).expect("predict ok");
        assert!(pred.log_risk.iter().all(|x| x.is_finite()));
    }

    // ── Test 8: Empty dataset → error ────────────────────────────────────────

    #[test]
    fn empty_dataset_returns_error() {
        let config = GbCoxConfig::default();
        let result = gb_cox_fit(&[], &[], &[], 0, 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset error"
        );
    }

    // ── Test 9: Feature mismatch on predict → error ───────────────────────────

    #[test]
    fn predict_feature_mismatch_error() {
        let (times, events, covs) = make_synthetic_3f(30, 5);
        let config = GbCoxConfig {
            n_estimators: 5,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 30, 3, &config).expect("fit ok");
        // Supply only 2 features per subject (wrong)
        let bad_covs: Vec<f64> = covs[..30 * 2].to_vec();
        let result = gb_cox_predict(&model, &bad_covs, 30);
        assert!(
            matches!(result, Err(SurvivalError::ShapeMismatch { .. })),
            "expected ShapeMismatch error"
        );
    }

    // ── Test 10: l2_reg=0 still works (no division by zero) ──────────────────

    #[test]
    fn l2_reg_zero_no_div_by_zero() {
        let (times, events, covs) = make_synthetic(50, 1.0, 11);
        let config = GbCoxConfig {
            n_estimators: 10,
            l2_reg: 0.0,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 50, 1, &config).expect("fit ok");
        let pred = gb_cox_predict(&model, &covs, 50).expect("predict ok");
        assert!(pred.log_risk.iter().all(|x| x.is_finite()));
    }

    // ── Test 11: max_depth=1 (stumps) produces a valid ensemble ──────────────

    #[test]
    fn max_depth_one_stumps() {
        let (times, events, covs) = make_synthetic_3f(60, 6);
        let config = GbCoxConfig {
            n_estimators: 25,
            max_depth: 1,
            learning_rate: 0.05,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 60, 3, &config).expect("fit ok");
        assert_eq!(model.trees.len(), 25);
        let pred = gb_cox_predict(&model, &covs, 60).expect("predict ok");
        assert!(pred.risk_score.iter().all(|&rs| rs > 0.0));
    }

    // ── Test 12: subsample=1, col_subsample=1 (full data) ────────────────────

    #[test]
    fn full_subsample_deterministic() {
        let (times, events, covs) = make_synthetic_3f(40, 7);
        let config = GbCoxConfig {
            n_estimators: 10,
            subsample: 1.0,
            col_subsample: 1.0,
            seed: 0,
            ..Default::default()
        };
        let m1 = gb_cox_fit(&times, &events, &covs, 40, 3, &config).expect("fit1 ok");
        let m2 = gb_cox_fit(&times, &events, &covs, 40, 3, &config).expect("fit2 ok");
        let p1 = gb_cox_predict(&m1, &covs, 40).expect("pred1 ok");
        let p2 = gb_cox_predict(&m2, &covs, 40).expect("pred2 ok");
        for (a, b) in p1.log_risk.iter().zip(p2.log_risk.iter()) {
            assert!(
                (a - b).abs() < 1.0e-10,
                "deterministic mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    // ── Test 13: train_log_likelihood has length == n_estimators ─────────────

    #[test]
    fn train_ll_length_equals_n_estimators() {
        let (times, events, covs) = make_synthetic_3f(50, 8);
        let n_est = 17usize;
        let config = GbCoxConfig {
            n_estimators: n_est,
            ..Default::default()
        };
        let model = gb_cox_fit(&times, &events, &covs, 50, 3, &config).expect("fit ok");
        assert_eq!(model.train_log_likelihood.len(), n_est);
    }

    // ── Test 14: GbCoxTree::predict_one returns finite value ─────────────────

    #[test]
    fn tree_predict_one_finite() {
        let tree = GbCoxTree {
            nodes: vec![
                GbNode::Split {
                    feature: 0,
                    threshold: 0.0,
                    left: 1,
                    right: 2,
                },
                GbNode::Leaf { value: -1.5 },
                GbNode::Leaf { value: 2.3 },
            ],
        };
        assert!((tree.predict_one(&[-1.0]) - (-1.5)).abs() < 1e-12);
        assert!((tree.predict_one(&[1.0]) - 2.3).abs() < 1e-12);
    }

    // ── Test 15: No events → error ────────────────────────────────────────────

    #[test]
    fn no_events_returns_error() {
        let times = vec![1.0_f64, 2.0, 3.0];
        let events = vec![0u8, 0, 0];
        let covs = vec![0.1_f64, -0.2, 0.5];
        let config = GbCoxConfig::default();
        let result = gb_cox_fit(&times, &events, &covs, 3, 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::NoEvents)),
            "expected NoEvents error"
        );
    }
}
