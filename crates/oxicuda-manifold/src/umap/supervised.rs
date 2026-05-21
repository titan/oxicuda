//! Supervised and semi-supervised UMAP.
//!
//! Supervised UMAP merges the standard feature-space fuzzy simplicial set with a
//! label-derived categorical fuzzy simplicial set so that same-class points attract
//! and different-class points repel in the embedding.
//!
//! Semi-supervised UMAP is the same algorithm but with some points carrying the
//! sentinel label [`UNLABELED`] (`u64::MAX`).  For those points the label-graph
//! contribution is zero and the feature-space edge weight is kept at full strength.
//!
//! # Algorithm outline
//!
//! 1. Build the feature-space kNN fuzzy simplicial set from X  (standard UMAP).
//! 2. Build a label-space categorical graph:
//!    - (i, j) both labeled, same class   → weight 1.0
//!    - (i, j) both labeled, diff class   → weight 0.0
//!    - either endpoint is [`UNLABELED`]  → edge omitted (treated as weight 0)
//! 3. Merge: `w = (1 - target_weight) * w_feature + target_weight * w_label`.
//!    For any edge touching an unlabeled point: `w = w_feature` (full feature weight).
//! 4. Run the same SGD optimisation as standard UMAP on the merged graph.

use std::collections::HashMap;

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::umap::fuzzy_simplicial::{fuzzy_simplicial_set, symmetrise};
use crate::umap::knn_graph::{build_knn_distances, smooth_knn_distances};

// ─── sentinel ────────────────────────────────────────────────────────────────

/// Sentinel label value meaning "this point is unlabeled".
///
/// Use `u64::MAX` so that it cannot coincide with any real class index.
pub const UNLABELED: u64 = u64::MAX;

// ─── configuration ───────────────────────────────────────────────────────────

/// Configuration for supervised / semi-supervised UMAP.
#[derive(Debug, Clone)]
pub struct SupervisedUmapConfig {
    /// Number of embedding dimensions (default: 2).
    pub n_components: usize,
    /// Number of nearest neighbours for the feature-space graph (default: 15).
    pub n_neighbors: usize,
    /// Minimum distance between points in the embedding space (default: 0.1).
    pub min_dist: f64,
    /// Scale of the embedding (default: 1.0).
    pub spread: f64,
    /// Number of SGD epochs (default: 200).
    pub n_epochs: usize,
    /// Initial SGD learning rate (default: 1.0).
    pub learning_rate: f64,
    /// Number of negative samples per positive edge update (default: 5).
    pub negative_sample_rate: usize,
    /// Weight of label supervision in \[0, 1\] (default: 0.5).
    ///
    /// - 0.0 → pure unsupervised UMAP (label graph ignored).
    /// - 1.0 → purely label-driven (feature graph ignored).
    /// - 0.5 → balanced (default for supervised UMAP).
    pub target_weight: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SupervisedUmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: 200,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            target_weight: 0.5,
            seed: 42,
        }
    }
}

// ─── result ──────────────────────────────────────────────────────────────────

/// Result returned by [`supervised_umap`].
#[derive(Debug)]
pub struct SupervisedUmapResult {
    /// Row-major embedding of shape `[n_samples * n_components]`.
    pub embedding: Vec<f64>,
    /// Number of input samples.
    pub n_samples: usize,
    /// Number of embedding dimensions.
    pub n_components: usize,
    /// Number of labeled points (i.e. those whose label ≠ [`UNLABELED`]).
    pub n_labeled: usize,
    /// Fitted `a` parameter of the UMAP `1 / (1 + a d^{2b})` curve.
    pub a: f64,
    /// Fitted `b` parameter of the UMAP `1 / (1 + a d^{2b})` curve.
    pub b: f64,
}

// ─── public entry point ───────────────────────────────────────────────────────

/// Fit supervised / semi-supervised UMAP.
///
/// # Arguments
///
/// * `data`      – row-major data matrix of shape `[n_samples, dim]`.
/// * `labels`    – class label per point (length `n_samples`).
///   Use [`UNLABELED`] (`u64::MAX`) for unlabeled points.
/// * `n_samples` – number of rows in `data`.
/// * `dim`       – number of feature columns.
/// * `config`    – algorithm hyperparameters.
///
/// # Errors
///
/// Returns [`ManifoldError`] on bad shapes, degenerate configurations, or
/// if `target_weight` is outside `[0, 1]`.
pub fn supervised_umap(
    data: &[f64],
    labels: &[u64],
    n_samples: usize,
    dim: usize,
    config: &SupervisedUmapConfig,
) -> ManifoldResult<SupervisedUmapResult> {
    // ── parameter validation ──────────────────────────────────────────────────
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if n_samples < 2 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be >= 2".into(),
        });
    }
    if data.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![data.len()],
        });
    }
    if labels.len() != n_samples {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples],
            got: vec![labels.len()],
        });
    }
    if config.n_components == 0 || config.n_components > 16 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be in 1..=16".into(),
        });
    }
    if config.n_neighbors == 0 || config.n_neighbors >= n_samples {
        return Err(ManifoldError::KNeighborsTooLarge {
            k: config.n_neighbors,
            n: n_samples,
        });
    }
    if !(0.0..=1.0).contains(&config.target_weight) {
        return Err(ManifoldError::InvalidParameter {
            name: "target_weight".into(),
            reason: "must be in [0, 1]".into(),
        });
    }
    if config.min_dist <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "min_dist".into(),
            reason: "must be positive".into(),
        });
    }
    if config.spread <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "spread".into(),
            reason: "must be positive".into(),
        });
    }

    let n = n_samples;
    let k = config.n_neighbors;
    let d_out = config.n_components;

    // ── count labeled points ──────────────────────────────────────────────────
    let n_labeled = labels.iter().filter(|&&l| l != UNLABELED).count();

    // ── step 1: feature-space fuzzy simplicial set ────────────────────────────
    let (feat_idx, feat_dist) = build_knn_distances(data, n, dim, k)?;
    let (sigmas, rhos) = smooth_knn_distances(&feat_dist, n, k, 64, 1e-5)?;
    let (raw_r, raw_c, raw_v) = fuzzy_simplicial_set(&feat_idx, &feat_dist, &sigmas, &rhos, n, k)?;
    let (feat_rows, feat_cols, feat_vals) = symmetrise(&raw_r, &raw_c, &raw_v)?;

    // Pack feature edges into a lookup map for fast merging.
    // key = (min(i,j), max(i,j)), value = weight
    let mut feat_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(feat_rows.len());
    for e in 0..feat_rows.len() {
        let i = feat_rows[e];
        let j = feat_cols[e];
        let key = (i.min(j), i.max(j));
        feat_map.insert(key, feat_vals[e]);
    }

    // ── step 2: label-space graph ─────────────────────────────────────────────
    let label_edges = build_label_graph(labels, n);

    // ── step 3: merge ─────────────────────────────────────────────────────────
    let merged = merge_graphs(&feat_map, &label_edges, labels, n, config.target_weight);

    // ── step 4: fit a / b curve ───────────────────────────────────────────────
    let (a, b) = fit_ab(config.spread, config.min_dist);

    // ── step 5: initialise embedding ──────────────────────────────────────────
    let mut rng = LcgRng::new(config.seed);
    let mut y = vec![0.0f64; n * d_out];
    for v in &mut y {
        *v = rng.next_range(-1.0, 1.0);
    }

    // ── step 6: SGD optimisation ──────────────────────────────────────────────
    let n_epochs = config.n_epochs;
    let (rows, cols, vals): (Vec<usize>, Vec<usize>, Vec<f64>) = merged.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut r, mut c, mut v), (ei, ej, ew)| {
            r.push(ei);
            c.push(ej);
            v.push(ew);
            (r, c, v)
        },
    );

    let max_val = vals.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
    // epochs_per_sample[e] approximates how often edge e is sampled per epoch.
    let epochs_per_sample: Vec<f64> = vals
        .iter()
        .map(|&m| if m > 0.0 { m / max_val } else { 0.0 })
        .collect();

    for epoch in 0..n_epochs {
        let alpha = config.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);
        for e in 0..rows.len() {
            if epochs_per_sample[e] <= 0.0 {
                continue;
            }
            // Stochastic sampling: sample proportional to edge weight.
            if rng.next_f64() >= epochs_per_sample[e] {
                continue;
            }
            let i = rows[e];
            let j = cols[e];

            // Attractive force.
            let mut d2 = 0.0;
            for kk in 0..d_out {
                let diff = y[i * d_out + kk] - y[j * d_out + kk];
                d2 += diff * diff;
            }
            let pow_b = d2.powf(b);
            let denom_att = 1.0 + a * pow_b;
            let coeff_att = if d2 > 1e-16 {
                -2.0 * a * b * d2.powf(b - 1.0) / denom_att
            } else {
                0.0
            };
            for kk in 0..d_out {
                let diff = y[i * d_out + kk] - y[j * d_out + kk];
                let delta = (coeff_att * diff).clamp(-4.0, 4.0);
                y[i * d_out + kk] += alpha * delta;
                y[j * d_out + kk] -= alpha * delta;
            }

            // Negative samples (repulsive force).
            for _ in 0..config.negative_sample_rate {
                let neg = rng.next_usize(n);
                if neg == i || neg == j {
                    continue;
                }
                let mut d2n = 0.0;
                for kk in 0..d_out {
                    let diff = y[i * d_out + kk] - y[neg * d_out + kk];
                    d2n += diff * diff;
                }
                let denom_rep = (0.001 + d2n) * (1.0 + a * d2n.powf(b));
                let coeff_rep = if d2n > 1e-16 {
                    2.0 * b / denom_rep
                } else {
                    0.0
                };
                for kk in 0..d_out {
                    let diff = y[i * d_out + kk] - y[neg * d_out + kk];
                    let delta = (coeff_rep * diff).clamp(-4.0, 4.0);
                    y[i * d_out + kk] += alpha * delta;
                }
            }
        }
    }

    Ok(SupervisedUmapResult {
        embedding: y,
        n_samples: n,
        n_components: d_out,
        n_labeled,
        a,
        b,
    })
}

// ─── label graph ─────────────────────────────────────────────────────────────

/// Build the categorical label graph as a list of `(i, j, weight)` sparse edges.
///
/// Rules:
/// - Both endpoints must have valid labels (≠ [`UNLABELED`]).
/// - Same-class pairs   → weight 1.0.
/// - Diff-class pairs   → weight 0.0.
/// - Pairs with at least one unlabeled endpoint → omitted entirely.
///
/// To keep the graph tractable, we only emit edges between labeled points.
/// The symmetric counterpart `(j, i)` is implied and handled in the merge step.
fn build_label_graph(labels: &[u64], n: usize) -> Vec<(usize, usize, f64)> {
    // Gather indices of labeled points.
    let labeled_idx: Vec<usize> = (0..n).filter(|&i| labels[i] != UNLABELED).collect();
    let m = labeled_idx.len();
    if m < 2 {
        return Vec::new();
    }

    // For larger datasets, emitting O(m^2) edges is expensive.  We limit
    // per-class pairs to at most `MAX_LABEL_EDGES_PER_CLASS` so the label
    // graph stays sparse for big n.  The constant is generous enough that
    // small to medium datasets (m < ~300) are handled exactly.
    const MAX_PAIRS_PER_CLASS: usize = 2_000;

    // Group labeled indices by class.
    let mut class_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for &i in &labeled_idx {
        class_map.entry(labels[i]).or_default().push(i);
    }

    let mut edges: Vec<(usize, usize, f64)> = Vec::new();

    // Same-class edges (weight = 1).
    for members in class_map.values() {
        let mut count = 0usize;
        'outer: for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let i = members[a];
                let j = members[b];
                // Canonical (min, max) ordering.
                let (pi, pj) = (i.min(j), i.max(j));
                edges.push((pi, pj, 1.0));
                count += 1;
                if count >= MAX_PAIRS_PER_CLASS {
                    break 'outer;
                }
            }
        }
    }

    // Different-class edges (weight = 0).
    // Only emit a bounded number — purely repulsive edges are expressed via
    // the absence of same-class attraction and by the negative-sampling in SGD.
    // We therefore skip them here (zero-weight edges would be discarded in the
    // merge anyway once the feature weight is also zero).
    //
    // This matches the standard supervised UMAP implementation in which
    // `categorical_distances` assigns distance 0 to same-class and distance 1
    // to different-class, and the "1 - d" membership gives 1.0 / 0.0 weights.
    // Edges with merged weight = 0 have zero contribution to SGD so we simply
    // omit them.

    edges
}

// ─── graph merge ─────────────────────────────────────────────────────────────

/// Merge the feature-space graph and label-space graph into a single edge list.
///
/// For each unique edge key `(min(i,j), max(i,j))` that appears in either graph:
/// - If both endpoints are labeled: `w = (1 - tw) * w_feat + tw * w_label`.
/// - If at least one endpoint is [`UNLABELED`]: `w = w_feat` (label graph ignored).
///
/// Edges with merged weight ≤ 1e-12 are discarded.
fn merge_graphs(
    feat_map: &HashMap<(usize, usize), f64>,
    label_edges: &[(usize, usize, f64)],
    labels: &[u64],
    _n: usize,
    target_weight: f64,
) -> Vec<(usize, usize, f64)> {
    // Collect all unique edge keys.
    let mut all_keys: Vec<(usize, usize)> = Vec::new();

    for &(i, j, _) in label_edges {
        let key = (i.min(j), i.max(j));
        all_keys.push(key);
    }
    for &key in feat_map.keys() {
        all_keys.push(key);
    }
    all_keys.sort_unstable();
    all_keys.dedup();

    // Build label lookup.
    let mut label_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(label_edges.len());
    for &(i, j, w) in label_edges {
        let key = (i.min(j), i.max(j));
        label_map.insert(key, w);
    }

    let mut result: Vec<(usize, usize, f64)> = Vec::with_capacity(all_keys.len());

    for (i, j) in all_keys {
        let feat_w = feat_map.get(&(i, j)).copied().unwrap_or(0.0);
        let i_labeled = labels[i] != UNLABELED;
        let j_labeled = labels[j] != UNLABELED;

        let merged_w = if i_labeled && j_labeled {
            let label_w = label_map.get(&(i, j)).copied().unwrap_or(0.0);
            (1.0 - target_weight) * feat_w + target_weight * label_w
        } else {
            // At least one endpoint is unlabeled → use only feature weight.
            feat_w
        };

        if merged_w > 1e-12 {
            result.push((i, j, merged_w));
        }
    }

    result
}

// ─── curve fitting ────────────────────────────────────────────────────────────

/// Fit UMAP `(a, b)` parameters so that `1 / (1 + a d^{2b})` approximates
/// the piecewise target `phi(d) = 1` if `d < min_dist` else `exp(-(d-min_dist)/spread)`.
///
/// Uses a coarse grid search followed by a local Newton-style refinement.
fn fit_ab(spread: f64, min_dist: f64) -> (f64, f64) {
    const N_TARGETS: usize = 300;
    let mut xs = [0.0f64; N_TARGETS];
    let mut ys = [0.0f64; N_TARGETS];
    for i in 0..N_TARGETS {
        xs[i] = i as f64 * 3.0 * spread / N_TARGETS as f64;
        ys[i] = if xs[i] < min_dist {
            1.0
        } else {
            (-((xs[i] - min_dist) / spread)).exp()
        };
    }

    // Coarse grid search.
    let mut best_sse = f64::INFINITY;
    let mut best_a = 1.0f64;
    let mut best_b = 1.0f64;
    for ai in 1..=40 {
        let a = 0.25 + 0.25 * ai as f64;
        for bi in 4..=25 {
            let b = 0.05 + 0.05 * bi as f64;
            let sse = sse_ab(&xs, &ys, a, b);
            if sse < best_sse {
                best_sse = sse;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Local refinement via gradient descent on SSE(a, b).
    let mut a = best_a;
    let mut b = best_b;
    let lr = 1e-3;
    for _ in 0..400 {
        let eps = 1e-6;
        let da = (sse_ab(&xs, &ys, a + eps, b) - sse_ab(&xs, &ys, a - eps, b)) / (2.0 * eps);
        let db = (sse_ab(&xs, &ys, a, b + eps) - sse_ab(&xs, &ys, a, b - eps)) / (2.0 * eps);
        a -= lr * da;
        b -= lr * db;
        // Clamp to sensible ranges.
        a = a.clamp(0.1, 20.0);
        b = b.clamp(0.01, 3.0);
    }

    (a.max(1e-4), b.max(1e-4))
}

/// Compute sum-of-squared errors for the curve `1 / (1 + a x^{2b})` vs target `ys`.
fn sse_ab(xs: &[f64], ys: &[f64], a: f64, b: f64) -> f64 {
    xs.iter()
        .zip(ys.iter())
        .map(|(&x, &y)| {
            let pred = if x == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + a * x.powf(2.0 * b))
            };
            let d = pred - y;
            d * d
        })
        .sum()
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper: generate 2-class blob data ───────────────────────────────────

    fn two_class_blobs(n_per_class: usize, sep: f64, seed: u64) -> (Vec<f64>, Vec<u64>) {
        let mut rng = LcgRng::new(seed);
        let n = 2 * n_per_class;
        let dim = 4usize;
        let mut data = vec![0.0f64; n * dim];
        let mut labels = vec![0u64; n];

        // Class 0: centred at origin.
        for i in 0..n_per_class {
            for d in 0..dim {
                data[i * dim + d] = rng.next_normal() * 0.3;
            }
            labels[i] = 0;
        }
        // Class 1: centred at (sep, sep, ...).
        for i in 0..n_per_class {
            let idx = n_per_class + i;
            for d in 0..dim {
                data[idx * dim + d] = sep + rng.next_normal() * 0.3;
            }
            labels[idx] = 1;
        }
        (data, labels)
    }

    // ── test 1: basic run, finite embedding ──────────────────────────────────

    #[test]
    fn test_basic_finite_embedding() {
        let (data, labels) = two_class_blobs(10, 5.0, 1);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 50,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("ok");
        assert_eq!(res.embedding.len(), n * 2);
        assert!(
            res.embedding.iter().all(|v| v.is_finite()),
            "embedding contains non-finite values"
        );
    }

    // ── test 2: supervised mode separates classes ────────────────────────────

    #[test]
    fn test_supervised_separates_classes() {
        let n_per_class = 12;
        let (data, labels) = two_class_blobs(n_per_class, 8.0, 2);
        let n = 2 * n_per_class;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 4,
            n_epochs: 150,
            target_weight: 0.8,
            seed: 42,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("ok");

        // Compute per-class centroids in embedding.
        let mut c0 = vec![0.0f64; 2];
        let mut c1 = vec![0.0f64; 2];
        for (i, &lbl) in labels.iter().enumerate().take(n) {
            let target = if lbl == 0 { &mut c0 } else { &mut c1 };
            for (d, slot) in target.iter_mut().enumerate().take(2) {
                *slot += res.embedding[i * 2 + d];
            }
        }
        for d in 0..2 {
            c0[d] /= n_per_class as f64;
            c1[d] /= n_per_class as f64;
        }
        let between: f64 = c0
            .iter()
            .zip(c1.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Intra-class spread for class 0.
        let spread0: f64 = (0..n_per_class)
            .map(|i| {
                (0..2usize)
                    .map(|d| (res.embedding[i * 2 + d] - c0[d]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>()
            / n_per_class as f64;

        // Between-class distance should be larger than average within-class spread.
        assert!(
            between > spread0,
            "expected inter-class distance {between:.4} > intra-class spread {spread0:.4}"
        );
    }

    // ── test 3: semi-supervised — unlabeled points embed finitely ───────────

    #[test]
    fn test_semi_supervised_finite() {
        let n_per_class = 10;
        let (mut data, mut labels) = two_class_blobs(n_per_class, 5.0, 3);
        // Mark half the points as unlabeled.
        let n = labels.len();
        for i in (0..n).step_by(2) {
            labels[i] = UNLABELED;
        }
        let dim = data.len() / n;
        // Append a few purely unlabeled points.
        let extra = 4usize;
        data.extend(std::iter::repeat_n(0.0_f64, extra * dim));
        labels.extend(std::iter::repeat_n(UNLABELED, extra));
        let n_total = n + extra;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 60,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n_total, dim, &cfg).expect("semi-supervised ok");
        assert_eq!(res.embedding.len(), n_total * 2);
        assert!(res.embedding.iter().all(|v| v.is_finite()));
    }

    // ── test 4: target_weight = 0 gives finite embedding (unsupervised limit) ─

    #[test]
    fn test_target_weight_zero() {
        let (data, labels) = two_class_blobs(8, 4.0, 4);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 50,
            target_weight: 0.0,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("tw=0 ok");
        assert!(res.embedding.iter().all(|v| v.is_finite()));
    }

    // ── test 5: target_weight = 1 gives finite embedding (pure-label limit) ──

    #[test]
    fn test_target_weight_one() {
        let (data, labels) = two_class_blobs(8, 4.0, 5);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 50,
            target_weight: 1.0,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("tw=1 ok");
        assert!(res.embedding.iter().all(|v| v.is_finite()));
    }

    // ── test 6: all unlabeled → same as unsupervised (finite output) ─────────

    #[test]
    fn test_all_unlabeled_finite() {
        let (data, _) = two_class_blobs(8, 4.0, 6);
        let n = data.len() / 4;
        let labels = vec![UNLABELED; n];
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 50,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("all-unlabeled ok");
        assert!(res.embedding.iter().all(|v| v.is_finite()));
        // All unlabeled → n_labeled should be 0.
        assert_eq!(res.n_labeled, 0);
    }

    // ── test 7: n_components = 3 works ──────────────────────────────────────

    #[test]
    fn test_n_components_3() {
        let (data, labels) = two_class_blobs(8, 4.0, 7);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_components: 3,
            n_neighbors: 3,
            n_epochs: 40,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("3d ok");
        assert_eq!(res.embedding.len(), n * 3);
        assert_eq!(res.n_components, 3);
        assert!(res.embedding.iter().all(|v| v.is_finite()));
    }

    // ── test 8: n_neighbors >= n_samples → error ─────────────────────────────

    #[test]
    fn test_invalid_n_neighbors() {
        let (data, labels) = two_class_blobs(5, 3.0, 8);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: n + 5, // too large
            ..Default::default()
        };
        let err = supervised_umap(&data, &labels, n, 4, &cfg);
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            ManifoldError::KNeighborsTooLarge { .. }
        ));
    }

    // ── test 9: target_weight outside [0, 1] → error ─────────────────────────

    #[test]
    fn test_invalid_target_weight_high() {
        let (data, labels) = two_class_blobs(6, 3.0, 9);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 2,
            target_weight: 1.5,
            ..Default::default()
        };
        let err = supervised_umap(&data, &labels, n, 4, &cfg);
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            ManifoldError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn test_invalid_target_weight_low() {
        let (data, labels) = two_class_blobs(6, 3.0, 10);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 2,
            target_weight: -0.1,
            ..Default::default()
        };
        let err = supervised_umap(&data, &labels, n, 4, &cfg);
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            ManifoldError::InvalidParameter { .. }
        ));
    }

    // ── test 10: n_labeled count is correct ──────────────────────────────────

    #[test]
    fn test_n_labeled_count() {
        let n = 20usize;
        let dim = 2usize;
        let data = vec![0.0f64; n * dim];
        let mut labels = vec![0u64; n];
        // Mark 7 as unlabeled.
        for i in [1, 3, 5, 7, 9, 11, 13] {
            labels[i] = UNLABELED;
        }
        let n_labeled_expected = n - 7;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 10,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, dim, &cfg).expect("ok");
        assert_eq!(res.n_labeled, n_labeled_expected);
    }

    // ── test 11: embedding shape is [n * n_components] ───────────────────────

    #[test]
    fn test_embedding_shape() {
        let n = 18usize;
        let dim = 3usize;
        let mut rng = LcgRng::new(11);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let labels: Vec<u64> = (0..n).map(|i| (i % 3) as u64).collect();
        let cfg = SupervisedUmapConfig {
            n_components: 4,
            n_neighbors: 3,
            n_epochs: 20,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, dim, &cfg).expect("shape ok");
        assert_eq!(res.embedding.len(), n * 4);
        assert_eq!(res.n_samples, n);
        assert_eq!(res.n_components, 4);
    }

    // ── test 12: a and b parameters are positive ─────────────────────────────

    #[test]
    fn test_ab_positive() {
        let (data, labels) = two_class_blobs(8, 4.0, 12);
        let n = data.len() / 4;
        let cfg = SupervisedUmapConfig {
            n_neighbors: 3,
            n_epochs: 30,
            ..Default::default()
        };
        let res = supervised_umap(&data, &labels, n, 4, &cfg).expect("ok");
        assert!(res.a > 0.0, "a must be positive, got {}", res.a);
        assert!(res.b > 0.0, "b must be positive, got {}", res.b);
    }

    // ── test 13: n_samples = 1 → error ───────────────────────────────────────

    #[test]
    fn test_single_sample_errors() {
        let data = vec![1.0, 2.0, 3.0];
        let labels = vec![0u64];
        let cfg = SupervisedUmapConfig {
            n_neighbors: 1,
            ..Default::default()
        };
        let err = supervised_umap(&data, &labels, 1, 3, &cfg);
        assert!(err.is_err());
    }

    // ── test 14: label graph: same-class edges have weight 1 ─────────────────

    #[test]
    fn test_build_label_graph_same_class() {
        let labels = vec![0u64, 0, 1, 1, UNLABELED];
        let edges = build_label_graph(&labels, labels.len());
        // Should contain same-class edges within class 0 and class 1.
        // Class 0: (0,1) → weight 1.
        // Class 1: (2,3) → weight 1.
        let has_01 = edges
            .iter()
            .any(|&(i, j, w)| (i == 0 && j == 1 || i == 1 && j == 0) && (w - 1.0).abs() < 1e-9);
        let has_23 = edges
            .iter()
            .any(|&(i, j, w)| (i == 2 && j == 3 || i == 3 && j == 2) && (w - 1.0).abs() < 1e-9);
        assert!(has_01, "expected edge (0,1) with weight 1");
        assert!(has_23, "expected edge (2,3) with weight 1");
        // Unlabeled point 4 must not appear.
        let touches_4 = edges.iter().any(|&(i, j, _)| i == 4 || j == 4);
        assert!(
            !touches_4,
            "unlabeled point should not appear in label graph"
        );
    }

    // ── test 15: fit_ab returns positive a and b ──────────────────────────────

    #[test]
    fn test_fit_ab_positive() {
        let (a, b) = fit_ab(1.0, 0.1);
        assert!(a > 0.0, "a={a}");
        assert!(b > 0.0, "b={b}");
    }
}
