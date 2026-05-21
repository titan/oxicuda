//! Survival Random Forests (RSF) — Ishwaran et al. 2008.
//!
//! Adapts the random forest to right-censored survival data by using the
//! log-rank score as the splitting criterion and Kaplan-Meier as the leaf estimator.
//!
//! # Algorithm Overview
//!
//! 1. **Bootstrap**: for each tree, draw n samples with replacement.
//! 2. **Tree building**: recursive binary partitioning.
//!    - At each node: randomly select `mtry` candidate features (default: sqrt(p)).
//!    - For each candidate: find the split value maximising the log-rank statistic.
//!    - Stop when `node_size < min_node_size`, `depth >= max_depth`, or no events remain.
//! 3. **Leaf estimator**: Kaplan-Meier curve from the bootstrap samples in the leaf.
//! 4. **Ensemble prediction**: traverse each tree → average leaf KM curves → CHF + risk score.
//! 5. **OOB concordance**: use out-of-bag samples to compute Harrell's C-index.

use crate::error::{SurvivalError, SurvivalResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a Survival Random Forest.
#[derive(Debug, Clone)]
pub struct SurvivalRfConfig {
    /// Number of trees in the forest (default: 100).
    pub n_trees: usize,
    /// Features considered per node split; `None` → floor(sqrt(p)) (default: None).
    pub mtry: Option<usize>,
    /// Minimum training samples in a leaf (default: 15).
    pub min_node_size: usize,
    /// Maximum tree depth; `usize::MAX` means unlimited (default: usize::MAX).
    pub max_depth: usize,
    /// Minimum number of events in a node to attempt a split (default: 1).
    pub min_events: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SurvivalRfConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            mtry: None,
            min_node_size: 15,
            max_depth: usize::MAX,
            min_events: 1,
            seed: 42,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tree data structures
// ──────────────────────────────────────────────────────────────────────────────

/// A single node in a survival tree.
#[derive(Debug, Clone)]
pub enum SurvivalNode {
    /// Internal split node.
    Internal {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    /// Leaf node, referring into the `leaf_km` array of the parent tree.
    Leaf { leaf_id: usize },
}

/// A single survival tree.
#[derive(Debug, Clone)]
pub struct SurvivalTree {
    /// Node arena (index 0 is the root).
    pub nodes: Vec<SurvivalNode>,
    /// KM curve `(time, S(t))` per leaf; indexed by `leaf_id`.
    pub leaf_km: Vec<Option<Vec<(f64, f64)>>>,
}

impl SurvivalTree {
    /// Traverse the tree for a single observation and return the leaf KM curve.
    #[must_use]
    pub fn predict_leaf(&self, row: &[f64]) -> Option<&Vec<(f64, f64)>> {
        let mut node_idx = 0usize;
        loop {
            match &self.nodes[node_idx] {
                SurvivalNode::Internal {
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
                SurvivalNode::Leaf { leaf_id } => {
                    return self.leaf_km[*leaf_id].as_ref();
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fitted model
// ──────────────────────────────────────────────────────────────────────────────

/// A fitted Survival Random Forest.
#[derive(Debug, Clone)]
pub struct SurvivalRf {
    /// All trees in the ensemble.
    pub trees: Vec<SurvivalTree>,
    /// Global sorted unique event times for aligning predictions.
    pub unique_times: Vec<f64>,
    /// Number of input features.
    pub n_features: usize,
    /// Training configuration.
    pub config: SurvivalRfConfig,
    /// Concordance on OOB samples (Harrell's C-index).
    pub oob_error: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Prediction output
// ──────────────────────────────────────────────────────────────────────────────

/// Prediction output from a Survival Random Forest.
#[derive(Debug, Clone)]
pub struct SurvivalRfPred {
    /// `[n_subjects][n_times]` — survival probability S(t) at each `unique_times` point.
    pub survival: Vec<Vec<f64>>,
    /// `[n_subjects][n_times]` — cumulative hazard H(t) = -log S(t).
    pub ensemble_chf: Vec<Vec<f64>>,
    /// Per-subject integrated CHF: Σ H(t).  Higher → higher risk.
    pub risk_score: Vec<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Kaplan-Meier on raw arrays (no Dataset/RiskSet dependency)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute a simple KM curve from raw time/event arrays.
/// Returns `(sorted_unique_event_times, S(t))` pairs.
fn km_from_raw(times: &[f64], events: &[u8]) -> Vec<(f64, f64)> {
    let n = times.len();
    if n == 0 {
        return Vec::new();
    }
    // Sort indices by time
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = Vec::new();
    let mut s = 1.0_f64;
    let mut at_risk = n as f64;
    let mut i = 0usize;

    while i < n {
        let t = times[order[i]];
        // Gather all observations at time t
        let mut j = i;
        let mut d = 0.0_f64;
        let mut total_at_time = 0.0_f64;
        while j < n && times[order[j]] == t {
            if events[order[j]] == 1 {
                d += 1.0;
            }
            total_at_time += 1.0;
            j += 1;
        }
        if d > 0.0 && at_risk > 0.0 {
            s *= 1.0 - d / at_risk;
            result.push((t, s.max(0.0)));
        }
        at_risk -= total_at_time;
        i = j;
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-rank splitting criterion (Ishwaran 2008 Algorithm 1)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the absolute log-rank statistic for a binary split of a node.
///
/// Returns `|O_L - E_L| / sqrt(V)`, or `0.0` if the split is degenerate.
fn log_rank_score(times: &[f64], events: &[u8], indices: &[usize], left_mask: &[bool]) -> f64 {
    // Collect unique event times in the node
    let mut event_times: Vec<f64> = indices
        .iter()
        .filter(|&&i| events[i] == 1)
        .map(|&i| times[i])
        .collect();
    if event_times.is_empty() {
        return 0.0;
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let mut obs_l = 0.0_f64;
    let mut exp_l = 0.0_f64;
    let mut variance = 0.0_f64;

    for &tk in &event_times {
        // Subjects at risk just before tk: time >= tk
        let mut n_l = 0.0_f64;
        let mut n_r = 0.0_f64;
        let mut d_l = 0.0_f64;
        let mut d_r = 0.0_f64;

        for &idx in indices {
            if times[idx] >= tk {
                // at risk at tk
                if left_mask[idx] {
                    n_l += 1.0;
                    if events[idx] == 1 && (times[idx] - tk).abs() < f64::EPSILON {
                        d_l += 1.0;
                    }
                } else {
                    n_r += 1.0;
                    if events[idx] == 1 && (times[idx] - tk).abs() < f64::EPSILON {
                        d_r += 1.0;
                    }
                }
            }
        }

        let d = d_l + d_r;
        let n = n_l + n_r;

        if n < 2.0 || d == 0.0 {
            continue;
        }

        obs_l += d_l;
        exp_l += d * n_l / n;

        // Hypergeometric variance contribution:
        // V_k = d_k * n_L * n_R * (n_k - d_k) / (n_k^2 * (n_k - 1))
        if n > 1.0 && n > d {
            variance += d * n_l * n_r * (n - d) / (n * n * (n - 1.0));
        }
    }

    if variance <= 0.0 {
        return 0.0;
    }
    (obs_l - exp_l).abs() / variance.sqrt()
}

// ──────────────────────────────────────────────────────────────────────────────
// Find the best (feature, threshold) split in a node
// ──────────────────────────────────────────────────────────────────────────────

/// Try all candidate features and thresholds; return the best `(feature, threshold, score)`.
///
/// Returns `None` if no valid split is found.
fn find_best_split(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_features: usize,
    indices: &[usize],
    candidate_features: &[usize],
    min_node_size: usize,
) -> Option<(usize, f64, f64)> {
    let mut best_score = 0.0_f64;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0_f64;
    let mut found = false;

    for &feat in candidate_features {
        // Collect distinct covariate values in this node, sorted
        let mut vals: Vec<f64> = indices
            .iter()
            .map(|&i| covariates[i * n_features + feat])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

        if vals.len() < 2 {
            continue;
        }

        // Try midpoints between consecutive distinct values as candidate thresholds
        for w in vals.windows(2) {
            let thresh = 0.5 * (w[0] + w[1]);

            // Count left / right sizes
            let n_left = indices
                .iter()
                .filter(|&&i| covariates[i * n_features + feat] <= thresh)
                .count();
            let n_right = indices.len() - n_left;

            if n_left < min_node_size || n_right < min_node_size {
                continue;
            }

            // Build left_mask (length = total n_subjects)
            let n_total = times.len();
            let mut left_mask = vec![false; n_total];
            for &i in indices {
                if covariates[i * n_features + feat] <= thresh {
                    left_mask[i] = true;
                }
            }

            let score = log_rank_score(times, events, indices, &left_mask);
            if score > best_score {
                best_score = score;
                best_feat = feat;
                best_thresh = thresh;
                found = true;
            }
        }
    }

    if found {
        Some((best_feat, best_thresh, best_score))
    } else {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Recursive tree builder
// ──────────────────────────────────────────────────────────────────────────────

struct TreeBuilder<'a> {
    times: &'a [f64],
    events: &'a [u8],
    covariates: &'a [f64],
    n_features: usize,
    config: &'a SurvivalRfConfig,
    mtry: usize,
    rng: &'a mut LcgRng,
    nodes: Vec<SurvivalNode>,
    leaf_km: Vec<Option<Vec<(f64, f64)>>>,
}

impl<'a> TreeBuilder<'a> {
    fn new(
        times: &'a [f64],
        events: &'a [u8],
        covariates: &'a [f64],
        n_features: usize,
        config: &'a SurvivalRfConfig,
        rng: &'a mut LcgRng,
    ) -> Self {
        let mtry = config
            .mtry
            .unwrap_or_else(|| ((n_features as f64).sqrt().floor() as usize).max(1));
        Self {
            times,
            events,
            covariates,
            n_features,
            config,
            mtry,
            rng,
            nodes: Vec::new(),
            leaf_km: Vec::new(),
        }
    }

    /// Recursively build a node for the given set of sample indices.
    /// Returns the node index in `self.nodes`.
    fn build_node(&mut self, indices: &[usize], depth: usize) -> usize {
        let n = indices.len();
        let n_events: usize = indices.iter().filter(|&&i| self.events[i] == 1).count();

        // Stopping criteria
        let should_leaf = n < self.config.min_node_size
            || depth >= self.config.max_depth
            || n_events < self.config.min_events
            || n_events == 0;

        if should_leaf {
            return self.make_leaf(indices);
        }

        // Sample mtry features without replacement (Fisher-Yates partial shuffle)
        let mtry = self.mtry.min(self.n_features);
        let mut feature_pool: Vec<usize> = (0..self.n_features).collect();
        let mut candidates = Vec::with_capacity(mtry);
        for k in 0..mtry {
            let pick = k + self.rng.next_usize(self.n_features - k);
            feature_pool.swap(k, pick);
            candidates.push(feature_pool[k]);
        }

        // Find best split
        let split = find_best_split(
            self.times,
            self.events,
            self.covariates,
            self.n_features,
            indices,
            &candidates,
            self.config.min_node_size,
        );

        let (feat, thresh) = match split {
            Some((f, t, _s)) => (f, t),
            None => {
                return self.make_leaf(indices);
            }
        };

        // Partition indices
        let left_indices: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| self.covariates[i * self.n_features + feat] <= thresh)
            .collect();
        let right_indices: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| self.covariates[i * self.n_features + feat] > thresh)
            .collect();

        // Reserve a slot, build children, then fill the slot
        let node_idx = self.nodes.len();
        // Push a placeholder
        self.nodes.push(SurvivalNode::Leaf { leaf_id: 0 });

        let left_idx = self.build_node(&left_indices, depth + 1);
        let right_idx = self.build_node(&right_indices, depth + 1);

        self.nodes[node_idx] = SurvivalNode::Internal {
            feature: feat,
            threshold: thresh,
            left: left_idx,
            right: right_idx,
        };
        node_idx
    }

    fn make_leaf(&mut self, indices: &[usize]) -> usize {
        let leaf_id = self.leaf_km.len();
        // Build KM from these indices
        let leaf_times: Vec<f64> = indices.iter().map(|&i| self.times[i]).collect();
        let leaf_events: Vec<u8> = indices.iter().map(|&i| self.events[i]).collect();
        let km = km_from_raw(&leaf_times, &leaf_events);
        self.leaf_km.push(Some(km));

        let node_idx = self.nodes.len();
        self.nodes.push(SurvivalNode::Leaf { leaf_id });
        node_idx
    }

    fn finish(self) -> SurvivalTree {
        SurvivalTree {
            nodes: self.nodes,
            leaf_km: self.leaf_km,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bootstrap sampling
// ──────────────────────────────────────────────────────────────────────────────

/// Draw n indices with replacement from [0, n).
fn bootstrap_sample(n: usize, rng: &mut LcgRng) -> Vec<usize> {
    (0..n).map(|_| rng.next_usize(n)).collect()
}

/// Compute the sorted unique event times across the full training set.
fn compute_unique_times(times: &[f64], events: &[u8]) -> Vec<f64> {
    let mut t: Vec<f64> = times
        .iter()
        .copied()
        .zip(events.iter().copied())
        .filter(|&(_, e)| e == 1)
        .map(|(t, _)| t)
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    t.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    t
}

// ──────────────────────────────────────────────────────────────────────────────
// Evaluate an ensemble KM curve at fixed global time points
// ──────────────────────────────────────────────────────────────────────────────

/// Interpolate a step-function KM curve `km_pairs` at a specific time `t`.
/// KM curves are right-continuous; the value before any event is 1.0.
fn km_eval_at(km_pairs: &[(f64, f64)], t: f64) -> f64 {
    if km_pairs.is_empty() {
        return 1.0;
    }
    // find last event time <= t
    let pos = km_pairs.partition_point(|&(tk, _)| tk <= t);
    if pos == 0 { 1.0 } else { km_pairs[pos - 1].1 }
}

/// Average an ensemble of KM curves (from tree leaves) at the global unique_times grid.
fn ensemble_survival(leaf_curves: &[&Vec<(f64, f64)>], unique_times: &[f64]) -> Vec<f64> {
    let m = unique_times.len();
    if leaf_curves.is_empty() || m == 0 {
        return vec![1.0; m];
    }
    let mut avg = vec![0.0_f64; m];
    for curve in leaf_curves {
        for (j, &t) in unique_times.iter().enumerate() {
            avg[j] += km_eval_at(curve, t);
        }
    }
    let k = leaf_curves.len() as f64;
    avg.iter_mut().for_each(|v| *v /= k);
    avg
}

// ──────────────────────────────────────────────────────────────────────────────
// OOB concordance
// ──────────────────────────────────────────────────────────────────────────────

/// Compute Harrell's C-index directly from times, events, and risk scores.
fn harrell_c_raw(times: &[f64], events: &[u8], risk: &[f64]) -> f64 {
    let n = times.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        if events[i] == 0 {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if times[i] >= times[j] {
                continue;
            }
            // Pair (i, j): i had the earlier event
            comparable += 1.0;
            if risk[i] > risk[j] {
                concordant += 1.0;
            } else if (risk[i] - risk[j]).abs() < 1.0e-12 {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        0.5
    } else {
        concordant / comparable
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Primary public API
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a Survival Random Forest to right-censored data.
///
/// # Arguments
/// - `times`        : observed times (length n)
/// - `events`       : event indicators 0/1 (length n)
/// - `covariates`   : row-major feature matrix [n × p]
/// - `n_subjects`   : n
/// - `n_features`   : p
/// - `config`       : RSF hyper-parameters
///
/// # Returns
/// A fitted [`SurvivalRf`] containing all trees and OOB concordance.
pub fn survival_rf_fit(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
    n_features: usize,
    config: &SurvivalRfConfig,
) -> SurvivalResult<SurvivalRf> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects || events.len() != n_subjects {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects],
            got: vec![times.len()],
        });
    }
    if covariates.len() != n_subjects * n_features {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects * n_features],
            got: vec![covariates.len()],
        });
    }
    if n_subjects < 2 * config.min_node_size {
        return Err(SurvivalError::InvalidParameter(format!(
            "n_subjects ({n_subjects}) < 2 * min_node_size ({}): cannot build any tree",
            config.min_node_size
        )));
    }
    if n_features == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_features must be > 0".to_string(),
        ));
    }
    let n_events_total: usize = events.iter().filter(|&&e| e == 1).count();
    if n_events_total == 0 {
        return Err(SurvivalError::NoEvents);
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }

    // ── Global unique event times ─────────────────────────────────────────────
    let unique_times = compute_unique_times(times, events);

    // ── Grow forest ───────────────────────────────────────────────────────────
    let mut rng = LcgRng::new(config.seed);
    let mut trees: Vec<SurvivalTree> = Vec::with_capacity(config.n_trees);
    // oob_leaf_curves[i] = list of KM curves from trees where i was OOB
    let mut oob_curves: Vec<Vec<Vec<(f64, f64)>>> = vec![Vec::new(); n_subjects];

    for _ in 0..config.n_trees {
        // Bootstrap sample
        let boot_idx = bootstrap_sample(n_subjects, &mut rng);

        // OOB mask: subjects not in this bootstrap
        let mut in_bag = vec![false; n_subjects];
        for &i in &boot_idx {
            in_bag[i] = true;
        }

        // Build tree on bootstrap sample (boot_idx may contain duplicates — that is fine)
        let mut builder = TreeBuilder::new(times, events, covariates, n_features, config, &mut rng);
        builder.build_node(&boot_idx, 0);
        let tree = builder.finish();

        // Collect OOB predictions for each out-of-bag subject
        for i in 0..n_subjects {
            if !in_bag[i] {
                let row = &covariates[i * n_features..(i + 1) * n_features];
                if let Some(km) = tree.predict_leaf(row) {
                    oob_curves[i].push(km.clone());
                }
            }
        }

        trees.push(tree);
    }

    // ── OOB concordance ───────────────────────────────────────────────────────
    let oob_error = compute_oob_concordance(times, events, &oob_curves, &unique_times, n_subjects);

    Ok(SurvivalRf {
        trees,
        unique_times,
        n_features,
        config: config.clone(),
        oob_error,
    })
}

/// Compute OOB concordance from accumulated per-subject OOB leaf curves.
fn compute_oob_concordance(
    times: &[f64],
    events: &[u8],
    oob_curves: &[Vec<Vec<(f64, f64)>>],
    unique_times: &[f64],
    n_subjects: usize,
) -> f64 {
    let mut risk_scores = vec![f64::NAN; n_subjects];
    let mut valid_count = 0usize;

    for i in 0..n_subjects {
        if oob_curves[i].is_empty() {
            continue;
        }
        let refs: Vec<&Vec<(f64, f64)>> = oob_curves[i].iter().collect();
        let surv = ensemble_survival(&refs, unique_times);
        // Risk score = Σ H(t) = Σ -log S(t)
        let risk: f64 = surv.iter().map(|&s| -s.max(1.0e-300).ln()).sum();
        risk_scores[i] = risk;
        valid_count += 1;
    }

    if valid_count < 2 {
        return 0.5;
    }

    // Only use subjects with valid OOB predictions
    let valid_times: Vec<f64> = (0..n_subjects)
        .filter(|&i| risk_scores[i].is_finite())
        .map(|i| times[i])
        .collect();
    let valid_events: Vec<u8> = (0..n_subjects)
        .filter(|&i| risk_scores[i].is_finite())
        .map(|i| events[i])
        .collect();
    let valid_risk: Vec<f64> = (0..n_subjects)
        .filter(|&i| risk_scores[i].is_finite())
        .map(|i| risk_scores[i])
        .collect();

    harrell_c_raw(&valid_times, &valid_events, &valid_risk)
}

/// Predict survival probabilities for new observations.
///
/// # Arguments
/// - `rf`              : fitted Survival Random Forest
/// - `new_covariates`  : row-major feature matrix [n_new × p]
/// - `n_new`           : number of subjects to predict
///
/// # Returns
/// [`SurvivalRfPred`] with survival curves, CHF, and risk scores.
pub fn survival_rf_predict(
    rf: &SurvivalRf,
    new_covariates: &[f64],
    n_new: usize,
) -> SurvivalResult<SurvivalRfPred> {
    if n_new == 0 {
        return Ok(SurvivalRfPred {
            survival: Vec::new(),
            ensemble_chf: Vec::new(),
            risk_score: Vec::new(),
        });
    }
    if new_covariates.len() != n_new * rf.n_features {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_new * rf.n_features],
            got: vec![new_covariates.len()],
        });
    }

    let mut survival = Vec::with_capacity(n_new);
    let mut ensemble_chf = Vec::with_capacity(n_new);
    let mut risk_score = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let row = &new_covariates[i * rf.n_features..(i + 1) * rf.n_features];
        // Gather leaf KM curves from all trees
        let leaf_refs: Vec<&Vec<(f64, f64)>> = rf
            .trees
            .iter()
            .filter_map(|tree| tree.predict_leaf(row))
            .collect();

        let surv_i = ensemble_survival(&leaf_refs, &rf.unique_times);
        // Ensure non-increasing (numerical safety)
        let surv_i = enforce_non_increasing(surv_i);

        let chf_i: Vec<f64> = surv_i.iter().map(|&s| -s.max(1.0e-300).ln()).collect();
        let risk_i: f64 = chf_i.iter().sum();

        survival.push(surv_i);
        ensemble_chf.push(chf_i);
        risk_score.push(risk_i);
    }

    Ok(SurvivalRfPred {
        survival,
        ensemble_chf,
        risk_score,
    })
}

/// Force a survival curve to be non-increasing (running minimum).
fn enforce_non_increasing(mut s: Vec<f64>) -> Vec<f64> {
    let mut min_so_far = 1.0_f64;
    for v in &mut s {
        *v = v.min(min_so_far);
        min_so_far = *v;
    }
    s
}

/// Compute variable importance via OOB permutation (Ishwaran et al. 2008).
///
/// For each feature `f`:
/// 1. Start from the OOB set.
/// 2. Permute values of feature `f` across OOB samples.
/// 3. Re-route each OOB observation through its assigned trees using the permuted feature.
/// 4. Compute concordance; importance = original OOB concordance − permuted concordance.
///
/// A large positive value → the feature is important (permuting it hurts predictions).
pub fn survival_rf_importance(
    rf: &SurvivalRf,
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
) -> SurvivalResult<Vec<f64>> {
    if n_subjects == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if covariates.len() != n_subjects * rf.n_features {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_subjects * rf.n_features],
            got: vec![covariates.len()],
        });
    }

    // Re-create the same bootstrap samples using the same seed
    let mut rng_seed = LcgRng::new(rf.config.seed);
    // We need which trees' bootstrap had each subject in-bag:
    // Rebuild bootstrap membership for all trees
    let mut tree_oob: Vec<Vec<bool>> = Vec::with_capacity(rf.config.n_trees);
    for _ in 0..rf.config.n_trees {
        let boot = bootstrap_sample(n_subjects, &mut rng_seed);
        let mut in_bag = vec![false; n_subjects];
        for &i in &boot {
            in_bag[i] = true;
        }
        let oob: Vec<bool> = in_bag.iter().map(|&b| !b).collect();
        tree_oob.push(oob);
    }

    let base_c = rf.oob_error;

    let mut importance = Vec::with_capacity(rf.n_features);
    let mut permute_rng = LcgRng::new(rf.config.seed.wrapping_add(0xDEAD_BEEF));

    for feat in 0..rf.n_features {
        // Build permuted-feature importance by:
        // For each OOB subject i, collect leaf KM curves from trees where i is OOB,
        // but use a permuted covariate for feature `feat`.

        // Collect OOB subjects globally
        let oob_subjects: Vec<usize> = (0..n_subjects)
            .filter(|&i| tree_oob.iter().any(|oob| oob[i]))
            .collect();

        if oob_subjects.is_empty() {
            importance.push(0.0);
            continue;
        }

        // Build a permuted index for feature `feat` among OOB subjects
        let mut perm_order = oob_subjects.clone();
        fisher_yates_shuffle(&mut perm_order, &mut permute_rng);

        // For each OOB subject, collect leaf KM curves using permuted covariate
        let mut perm_oob_curves: Vec<Vec<Vec<(f64, f64)>>> = vec![Vec::new(); n_subjects];

        for (pos, &i) in oob_subjects.iter().enumerate() {
            // Build a modified covariate row for subject i:
            // swap feature `feat` with oob_subjects[perm_order[pos]]
            let perm_src = perm_order[pos];
            let mut row: Vec<f64> = covariates[i * rf.n_features..(i + 1) * rf.n_features].to_vec();
            row[feat] = covariates[perm_src * rf.n_features + feat];

            for (t_idx, tree) in rf.trees.iter().enumerate() {
                if tree_oob[t_idx][i] {
                    if let Some(km) = tree.predict_leaf(&row) {
                        perm_oob_curves[i].push(km.clone());
                    }
                }
            }
        }

        let perm_c = compute_oob_concordance(
            times,
            events,
            &perm_oob_curves,
            &rf.unique_times,
            n_subjects,
        );

        importance.push(base_c - perm_c);
    }

    Ok(importance)
}

/// In-place Fisher-Yates shuffle.
fn fisher_yates_shuffle(v: &mut [usize], rng: &mut LcgRng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        v.swap(i, j);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic survival data.
    ///
    /// If `informative = true`, event time is inversely related to covariate 0
    /// (high covariate 0 → short survival), otherwise all covariates are noise.
    fn synthetic_data(
        n: usize,
        p: usize,
        informative: bool,
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n * p);

        for _ in 0..n {
            let mut row: Vec<f64> = (0..p).map(|_| rng.next_f64()).collect();
            let rate = if informative {
                // Higher covariate 0 → higher hazard → shorter time
                0.5 + 3.0 * row[0]
            } else {
                1.0
            };
            let t_event = -rng.next_f64().max(1.0e-300).ln() / rate;
            let t_censor = 2.0 + rng.next_f64() * 3.0;
            let t_obs = t_event.min(t_censor);
            let ev: u8 = if t_event <= t_censor { 1 } else { 0 };
            times.push(t_obs.max(0.001));
            events.push(ev);
            cov.append(&mut row);
        }
        (times, events, cov)
    }

    // ── Test 1: Basic fit succeeds and n_trees matches ────────────────────────

    #[test]
    fn fit_basic_succeeds() {
        let (times, events, cov) = synthetic_data(60, 3, true, 1);
        let cfg = SurvivalRfConfig {
            n_trees: 20,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 60, 3, &cfg).expect("fit failed");
        assert_eq!(rf.trees.len(), 20);
        assert_eq!(rf.n_features, 3);
    }

    // ── Test 2: Survival curves are non-increasing ────────────────────────────

    #[test]
    fn survival_curves_non_increasing() {
        let (times, events, cov) = synthetic_data(60, 3, true, 2);
        let cfg = SurvivalRfConfig {
            n_trees: 20,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 60, 3, &cfg).expect("fit");
        let pred = survival_rf_predict(&rf, &cov, 60).expect("predict");
        for surv_row in &pred.survival {
            for w in surv_row.windows(2) {
                assert!(w[0] >= w[1] - 1.0e-10, "non-monotone: {} > {}", w[0], w[1]);
            }
        }
    }

    // ── Test 3: risk_score length == n_new ────────────────────────────────────

    #[test]
    fn risk_score_length() {
        let (times, events, cov) = synthetic_data(60, 3, true, 3);
        let cfg = SurvivalRfConfig {
            n_trees: 10,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 60, 3, &cfg).expect("fit");
        // Predict on 15 new subjects
        let (_, _, cov_new) = synthetic_data(15, 3, true, 99);
        let pred = survival_rf_predict(&rf, &cov_new, 15).expect("predict");
        assert_eq!(pred.risk_score.len(), 15);
    }

    // ── Test 4: survival shape [n_new][n_times] ───────────────────────────────

    #[test]
    fn survival_shape_correct() {
        let (times, events, cov) = synthetic_data(50, 3, true, 4);
        let cfg = SurvivalRfConfig {
            n_trees: 10,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 50, 3, &cfg).expect("fit");
        let n_times = rf.unique_times.len();
        let (_, _, cov_new) = synthetic_data(8, 3, true, 77);
        let pred = survival_rf_predict(&rf, &cov_new, 8).expect("predict");
        assert_eq!(pred.survival.len(), 8);
        for row in &pred.survival {
            assert_eq!(row.len(), n_times);
        }
    }

    // ── Test 5: Perfect predictor → concordance > 0.7 ────────────────────────

    #[test]
    fn informative_predictor_concordance() {
        let n = 100;
        let p = 3;
        let (times, events, cov) = synthetic_data(n, p, true, 5);
        let cfg = SurvivalRfConfig {
            n_trees: 50,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, n, p, &cfg).expect("fit");
        let pred = survival_rf_predict(&rf, &cov, n).expect("predict");
        let c = harrell_c_raw(&times, &events, &pred.risk_score);
        assert!(c > 0.65, "concordance too low: {c:.3}");
    }

    // ── Test 6: Random covariate → OOB concordance ≈ 0.5 ────────────────────

    #[test]
    fn random_covariate_oob_near_half() {
        let (times, events, cov) = synthetic_data(80, 3, false, 6);
        let cfg = SurvivalRfConfig {
            n_trees: 30,
            min_node_size: 8,
            seed: 999,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 80, 3, &cfg).expect("fit");
        // OOB error near 0.5 for pure noise covariates (allow wide band)
        assert!(
            rf.oob_error >= 0.25 && rf.oob_error <= 0.75,
            "oob_error out of range: {}",
            rf.oob_error
        );
    }

    // ── Test 7: Variable importance returns length-p vector ───────────────────

    #[test]
    fn importance_length() {
        let n = 60;
        let p = 4;
        let (times, events, cov) = synthetic_data(n, p, true, 7);
        let cfg = SurvivalRfConfig {
            n_trees: 20,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, n, p, &cfg).expect("fit");
        let imp = survival_rf_importance(&rf, &times, &events, &cov, n).expect("importance");
        assert_eq!(imp.len(), p);
    }

    // ── Test 8: Empty dataset → error ─────────────────────────────────────────

    #[test]
    fn empty_dataset_error() {
        let cfg = SurvivalRfConfig::default();
        let err = survival_rf_fit(&[], &[], &[], 0, 3, &cfg);
        assert!(err.is_err(), "expected error for empty dataset");
    }

    // ── Test 9: n_subjects too small → error ──────────────────────────────────

    #[test]
    fn too_small_n_subjects_error() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1u8, 1, 1];
        let cov = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let cfg = SurvivalRfConfig {
            min_node_size: 15,
            n_trees: 5,
            ..Default::default()
        };
        let err = survival_rf_fit(&times, &events, &cov, 3, 3, &cfg);
        assert!(err.is_err(), "expected error: n too small");
    }

    // ── Test 10: Single feature (mtry = 1) works ──────────────────────────────

    #[test]
    fn single_feature_works() {
        let mut rng = LcgRng::new(10);
        let n = 50;
        let times: Vec<f64> = (0..n).map(|_| rng.next_range(0.1, 5.0)).collect();
        let events: Vec<u8> = (0..n)
            .map(|_| if rng.next_bool() { 1 } else { 0 })
            .collect();
        let cov: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let cfg = SurvivalRfConfig {
            n_trees: 10,
            mtry: Some(1),
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, n, 1, &cfg).expect("single feature");
        assert_eq!(rf.n_features, 1);
    }

    // ── Test 11: n_features mismatch on predict → error ──────────────────────

    #[test]
    fn predict_features_mismatch_error() {
        let (times, events, cov) = synthetic_data(50, 3, true, 11);
        let cfg = SurvivalRfConfig {
            n_trees: 5,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 50, 3, &cfg).expect("fit");
        // Wrong number of features (only p=2 instead of p=3)
        let bad_cov: Vec<f64> = vec![0.5, 0.5]; // 1 subject × 2 features
        let err = survival_rf_predict(&rf, &bad_cov, 1);
        assert!(err.is_err(), "expected shape mismatch");
    }

    // ── Test 12: oob_error is in [0, 1] ──────────────────────────────────────

    #[test]
    fn oob_error_in_unit_interval() {
        let (times, events, cov) = synthetic_data(60, 3, true, 12);
        let cfg = SurvivalRfConfig {
            n_trees: 20,
            min_node_size: 5,
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, 60, 3, &cfg).expect("fit");
        assert!(
            rf.oob_error >= 0.0 && rf.oob_error <= 1.0,
            "oob_error={} not in [0,1]",
            rf.oob_error
        );
    }

    // ── Test 13: Log-rank split identifies important feature ──────────────────

    #[test]
    fn logrank_split_identifies_important_feature() {
        // 40 subjects, feature 0 is the only signal, features 1..2 are noise.
        // We test that the log-rank score is higher when splitting by feature 0.
        let n = 40;
        let p = 3;
        let mut rng = LcgRng::new(13);
        let mut times = Vec::new();
        let mut events = Vec::new();
        let mut cov = Vec::new();
        for _ in 0..n {
            let x0: f64 = rng.next_f64(); // signal
            let x1: f64 = rng.next_f64(); // noise
            let x2: f64 = rng.next_f64(); // noise
            // High x0 → fast event
            let rate = 0.5 + 4.0 * x0;
            let t = -rng.next_f64().max(1.0e-300).ln() / rate;
            let c = 1.5 + rng.next_f64();
            let t_obs = t.min(c).max(0.001);
            let ev: u8 = if t <= c { 1 } else { 0 };
            times.push(t_obs);
            events.push(ev);
            cov.push(x0);
            cov.push(x1);
            cov.push(x2);
        }

        // Compute log-rank score for a threshold at 0.5 for each feature
        let all_idx: Vec<usize> = (0..n).collect();
        // Feature 0 split at 0.5
        let mut mask0 = vec![false; n];
        for i in 0..n {
            mask0[i] = cov[i * p] <= 0.5;
        }
        let score0 = log_rank_score(&times, &events, &all_idx, &mask0);

        // Feature 1 split at 0.5
        let mut mask1 = vec![false; n];
        for i in 0..n {
            mask1[i] = cov[i * p + 1] <= 0.5;
        }
        let score1 = log_rank_score(&times, &events, &all_idx, &mask1);

        // Feature 2 split at 0.5
        let mut mask2 = vec![false; n];
        for i in 0..n {
            mask2[i] = cov[i * p + 2] <= 0.5;
        }
        let score2 = log_rank_score(&times, &events, &all_idx, &mask2);

        assert!(
            score0 > score1 && score0 > score2,
            "expected signal feature (0) to have higher log-rank score: s0={score0:.3} s1={score1:.3} s2={score2:.3}"
        );
    }

    // ── Test 14: min_node_size > n → trivially valid (all-leaf trees) ─────────

    #[test]
    fn min_node_size_larger_than_n_trivial() {
        let n = 40;
        let p = 2;
        let (times, events, cov) = synthetic_data(n, p, false, 14);
        let cfg = SurvivalRfConfig {
            n_trees: 5,
            min_node_size: 5, // much smaller than n, but we use large bootstrap
            max_depth: 0,     // depth 0 → immediate leaf
            ..Default::default()
        };
        let rf = survival_rf_fit(&times, &events, &cov, n, p, &cfg).expect("fit");
        // Each tree has at most 1 internal node (root leaf)
        assert_eq!(rf.trees.len(), 5);
    }
}
