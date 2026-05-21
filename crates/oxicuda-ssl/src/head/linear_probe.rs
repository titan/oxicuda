//! Linear probing evaluation helper for SSL representations.
//!
//! Implements a full one-vs-all (OVA) multiclass logistic regression via
//! Iteratively Reweighted Least Squares (IRLS), with k-fold cross-validation.
//!
//! # Protocol
//! Freeze the backbone, extract features once, then fit a linear classifier
//! (logistic regression) on the features.  This is the standard SSL evaluation
//! protocol: a higher accuracy indicates a richer representation.
//!
//! # Algorithm
//! * **OVA-IRLS** — one binary IRLS logistic regression per class.
//! * **Augmented features** — a 1 is appended to each feature vector so that
//!   the bias term is absorbed into the weight vector.
//! * **Regularisation** — isotropic L2 (λ·I) added to the Gram matrix.
//! * **Cholesky WLS** — each IRLS step solves a (D+1)×(D+1) system via
//!   Cholesky decomposition instead of a full matrix inversion.
//! * **k-Fold CV** — Fisher-Yates shuffle then split into contiguous folds.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the linear probing evaluator.
#[derive(Debug, Clone)]
pub struct LinearProbeConfig {
    /// Number of target classes.
    pub n_classes: usize,
    /// Number of CV folds.
    pub n_folds: usize,
    /// Maximum IRLS iterations per binary sub-problem.
    pub max_iter: usize,
    /// Convergence tolerance on the relative weight update.
    pub tol: f64,
    /// L2 regularisation strength λ.
    pub l2_reg: f64,
    /// Seed for the fold-shuffling RNG.
    pub seed: u64,
}

impl Default for LinearProbeConfig {
    fn default() -> Self {
        Self {
            n_classes: 2,
            n_folds: 5,
            max_iter: 200,
            tol: 1e-5,
            l2_reg: 1e-3,
            seed: 42,
        }
    }
}

// ─── Result / fitted struct ───────────────────────────────────────────────────

/// Summary of a k-fold cross-validation linear probing run.
#[derive(Debug, Clone)]
pub struct LinearProbeResult {
    /// Mean accuracy across all folds.
    pub mean_accuracy: f64,
    /// Standard deviation of per-fold accuracies.
    pub std_accuracy: f64,
    /// Per-fold accuracy values (length = n_folds).
    pub per_fold_accuracy: Vec<f64>,
    /// Macro-averaged F1 score across all folds (mean per-class harmonic mean).
    pub macro_f1: f64,
    /// Per-class F1 scores (length = n_classes).
    pub per_class_f1: Vec<f64>,
}

/// A fitted one-vs-all logistic regression model.
#[derive(Debug, Clone)]
pub struct FittedLinearProbe {
    /// Weight matrix stored row-major `[n_classes × (in_dim + 1)]`.
    /// The last column of each row is the absorbed bias term.
    pub weights: Vec<f64>,
    /// Feature dimensionality (before bias augmentation).
    pub in_dim: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// IRLS iterations taken per binary sub-problem.
    pub n_iter: Vec<usize>,
    /// Whether each binary sub-problem converged within `max_iter`.
    pub converged: Vec<bool>,
}

// ─── Private numerics ─────────────────────────────────────────────────────────

/// Numerically stable sigmoid: avoids overflow for large |x|.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Cholesky decomposition + forward/back substitution to solve `A·x = b`.
///
/// `a` is a row-major n×n **symmetric positive-definite** matrix.
/// Returns `Err(Internal)` if the matrix is not positive-definite.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> SslResult<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);

    // ── Cholesky factorisation A = L·Lᵀ (lower triangular L in-place) ──────
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(SslError::Internal(
                        "cholesky_solve: matrix not positive-definite".into(),
                    ));
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    // ── Forward substitution: solve L·y = b ─────────────────────────────────
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }

    // ── Back substitution: solve Lᵀ·x = y ───────────────────────────────────
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }

    Ok(x)
}

/// Accuracy: fraction of elements where `predicted[i] == truth[i]`.
fn accuracy(predicted: &[usize], truth: &[usize]) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }
    let correct = predicted
        .iter()
        .zip(truth.iter())
        .filter(|&(&p, &t)| p == t)
        .count();
    correct as f64 / predicted.len() as f64
}

/// Per-class F1 = TP / (TP + 0.5·(FP + FN)).
fn f1_per_class(predicted: &[usize], truth: &[usize], n_classes: usize) -> Vec<f64> {
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_ = vec![0usize; n_classes];

    for (&p, &t) in predicted.iter().zip(truth.iter()) {
        if p < n_classes && t < n_classes {
            if p == t {
                tp[p] += 1;
            } else {
                fp[p] += 1;
                fn_[t] += 1;
            }
        }
    }

    (0..n_classes)
        .map(|k| {
            let denom = tp[k] as f64 + 0.5 * (fp[k] + fn_[k]) as f64;
            if denom < 1e-12 {
                0.0
            } else {
                tp[k] as f64 / denom
            }
        })
        .collect()
}

/// Fisher-Yates in-place shuffle using `LcgRng`.
fn fisher_yates_shuffle(indices: &mut [usize], rng: &mut LcgRng) {
    rng.shuffle(indices);
}

// ─── Core IRLS binary logistic regression ─────────────────────────────────────

/// Fit a single binary (OVA) logistic regression for class `k` against the
/// rest using IRLS.
///
/// `x_aug` — augmented feature matrix [n × (d+1)] (row-major).
/// `y_bin` — binary labels for this class (0 or 1), length n.
///
/// Returns `(weights, iterations_taken, converged)`.
fn irls_binary(
    x_aug: &[f64],
    y_bin: &[f64],
    n: usize,
    d_aug: usize,
    config: &LinearProbeConfig,
) -> SslResult<(Vec<f64>, usize, bool)> {
    const EPS: f64 = 1e-7;

    let mut w = vec![0.0_f64; d_aug];
    let mut iters_done = 0usize;
    let mut converged = false;

    for iter in 0..config.max_iter {
        // ── Step 1-3: compute η_i, p_i, W_i ─────────────────────────────────
        let mut p_vec = vec![0.0_f64; n];
        for i in 0..n {
            let row = &x_aug[i * d_aug..(i + 1) * d_aug];
            let eta_i: f64 = row.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum();
            p_vec[i] = sigmoid(eta_i).clamp(EPS, 1.0 - EPS);
        }

        // ── Step 4: working response z_i ─────────────────────────────────────
        // z_i = η_i + (y_i - p_i) / (p_i · (1 - p_i))
        // but we recompute η_i from w to keep things numerically fresh.
        let mut eta_vec = vec![0.0_f64; n];
        for i in 0..n {
            let row = &x_aug[i * d_aug..(i + 1) * d_aug];
            eta_vec[i] = row.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum();
        }

        // ── Step 5: WLS normal equations ──────────────────────────────────────
        // Accumulate Xᵀ W X and Xᵀ W z rank-1.
        let mut xtwx = vec![0.0_f64; d_aug * d_aug];
        let mut xtwz = vec![0.0_f64; d_aug];

        for i in 0..n {
            let p_i = p_vec[i];
            let w_i = p_i * (1.0 - p_i); // IRLS weight
            let z_i = eta_vec[i] + (y_bin[i] - p_i) / w_i;
            let row = &x_aug[i * d_aug..(i + 1) * d_aug];

            // rank-1 update of Xᵀ W X
            for r in 0..d_aug {
                let val_r = w_i * row[r];
                for c in 0..d_aug {
                    xtwx[r * d_aug + c] += val_r * row[c];
                }
                xtwz[r] += val_r * z_i;
            }
        }

        // Add λ·I (L2 regularisation).
        for j in 0..d_aug {
            xtwx[j * d_aug + j] += config.l2_reg;
        }

        // Solve (Xᵀ W X + λI) · w_new = Xᵀ W z.
        let w_new = cholesky_solve(&xtwx, &xtwz, d_aug)?;

        // ── Step 6: convergence check ─────────────────────────────────────────
        let delta_norm: f64 = w_new
            .iter()
            .zip(w.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let w_norm: f64 = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let rel = delta_norm / w_norm.max(1.0);

        w = w_new;
        iters_done = iter + 1;

        if rel < config.tol {
            converged = true;
            break;
        }
    }

    // Validate no NaN leaked through.
    for &v in &w {
        if v.is_nan() {
            return Err(SslError::NanEncountered {
                location: "irls_binary weight",
            });
        }
    }

    Ok((w, iters_done, converged))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a one-vs-all multiclass logistic regression on `(features, labels)`.
///
/// * `features` — row-major `[n_samples × in_dim]` slice.
/// * `labels`   — class indices in `0..config.n_classes`, length `n_samples`.
///
/// # Errors
/// Returns [`SslError::EmptyInput`] if `n_samples == 0`,
/// [`SslError::InvalidParameter`] for degenerate configuration, or
/// [`SslError::DimensionMismatch`] on shape mismatches.
pub fn linear_probe_fit(
    features: &[f64],
    labels: &[usize],
    n_samples: usize,
    in_dim: usize,
    config: &LinearProbeConfig,
) -> SslResult<FittedLinearProbe> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(SslError::EmptyInput);
    }
    if in_dim == 0 {
        return Err(SslError::InvalidParameter {
            name: "in_dim".into(),
            reason: "feature dimension must be > 0".into(),
        });
    }
    if config.n_classes < 2 {
        return Err(SslError::InvalidParameter {
            name: "n_classes".into(),
            reason: "must be >= 2".into(),
        });
    }
    if config.l2_reg < 0.0 || !config.l2_reg.is_finite() {
        return Err(SslError::InvalidParameter {
            name: "l2_reg".into(),
            reason: "must be non-negative and finite".into(),
        });
    }
    if features.len() != n_samples * in_dim {
        return Err(SslError::DimensionMismatch {
            expected: n_samples * in_dim,
            got: features.len(),
        });
    }
    if labels.len() != n_samples {
        return Err(SslError::DimensionMismatch {
            expected: n_samples,
            got: labels.len(),
        });
    }
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl >= config.n_classes {
            return Err(SslError::InvalidParameter {
                name: "labels".into(),
                reason: format!(
                    "label {} at index {} is out of range [0, {})",
                    lbl, i, config.n_classes
                ),
            });
        }
    }

    // ── Build augmented feature matrix [n × (D+1)] ───────────────────────────
    let d_aug = in_dim + 1;
    let mut x_aug = vec![0.0_f64; n_samples * d_aug];
    for i in 0..n_samples {
        let src = &features[i * in_dim..(i + 1) * in_dim];
        let dst = &mut x_aug[i * d_aug..(i + 1) * d_aug];
        dst[..in_dim].copy_from_slice(src);
        dst[in_dim] = 1.0; // bias
    }

    // Check for NaN/Inf in features.
    for (j, &v) in x_aug.iter().enumerate() {
        if !v.is_finite() {
            let sample = j / d_aug;
            let _ = sample; // suppress unused var warning if debug assert only
            return Err(SslError::NanEncountered {
                location: "features (augmented)",
            });
        }
    }

    // ── Fit one binary classifier per class ───────────────────────────────────
    let mut all_weights = vec![0.0_f64; config.n_classes * d_aug];
    let mut n_iter_per_class = vec![0usize; config.n_classes];
    let mut converged_per_class = vec![false; config.n_classes];

    for k in 0..config.n_classes {
        let y_bin: Vec<f64> = labels
            .iter()
            .map(|&lbl| if lbl == k { 1.0 } else { 0.0 })
            .collect();

        let (w_k, iters, conv) = irls_binary(&x_aug, &y_bin, n_samples, d_aug, config)?;

        all_weights[k * d_aug..(k + 1) * d_aug].copy_from_slice(&w_k);
        n_iter_per_class[k] = iters;
        converged_per_class[k] = conv;
    }

    Ok(FittedLinearProbe {
        weights: all_weights,
        in_dim,
        n_classes: config.n_classes,
        n_iter: n_iter_per_class,
        converged: converged_per_class,
    })
}

/// Predict class labels for `n_samples` feature vectors.
///
/// Uses argmax over OVA sigmoid scores.
///
/// # Errors
/// Returns [`SslError::DimensionMismatch`] if `features.len() != n_samples * probe.in_dim`.
pub fn linear_probe_predict(
    probe: &FittedLinearProbe,
    features: &[f64],
    n_samples: usize,
) -> SslResult<Vec<usize>> {
    let d_aug = probe.in_dim + 1;

    if features.len() != n_samples * probe.in_dim {
        return Err(SslError::DimensionMismatch {
            expected: n_samples * probe.in_dim,
            got: features.len(),
        });
    }

    let mut predictions = vec![0usize; n_samples];
    for i in 0..n_samples {
        let src = &features[i * probe.in_dim..(i + 1) * probe.in_dim];

        // Build augmented row.
        let mut x_aug = vec![0.0_f64; d_aug];
        x_aug[..probe.in_dim].copy_from_slice(src);
        x_aug[probe.in_dim] = 1.0;

        // Compute sigmoid score for each class and pick argmax.
        let mut best_class = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for k in 0..probe.n_classes {
            let w_k = &probe.weights[k * d_aug..(k + 1) * d_aug];
            let eta: f64 = w_k.iter().zip(x_aug.iter()).map(|(&w, &x)| w * x).sum();
            let score = sigmoid(eta);
            if score > best_score {
                best_score = score;
                best_class = k;
            }
        }
        predictions[i] = best_class;
    }

    Ok(predictions)
}

/// k-Fold cross-validation linear probing evaluation.
///
/// Shuffles `n_samples` indices with `LcgRng::new(config.seed)`, splits into
/// `config.n_folds` contiguous folds, trains on the remaining folds, evaluates
/// on the held-out fold, and aggregates accuracy + macro-F1.
///
/// # Errors
/// Propagates any errors from [`linear_probe_fit`] or [`linear_probe_predict`].
pub fn linear_probe_eval(
    features: &[f64],
    labels: &[usize],
    n_samples: usize,
    in_dim: usize,
    config: &LinearProbeConfig,
) -> SslResult<LinearProbeResult> {
    if n_samples == 0 {
        return Err(SslError::EmptyInput);
    }
    if config.n_folds < 2 {
        return Err(SslError::InvalidParameter {
            name: "n_folds".into(),
            reason: "must be >= 2".into(),
        });
    }
    if n_samples < config.n_folds {
        return Err(SslError::BatchTooSmall);
    }

    // ── Shuffle indices ───────────────────────────────────────────────────────
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = LcgRng::new(config.seed);
    fisher_yates_shuffle(&mut indices, &mut rng);

    // ── Build fold boundaries ─────────────────────────────────────────────────
    // Each fold gets floor(n/k) elements; the last fold absorbs the remainder.
    let fold_size = n_samples / config.n_folds;
    let mut fold_starts = Vec::with_capacity(config.n_folds + 1);
    for f in 0..config.n_folds {
        fold_starts.push(f * fold_size);
    }
    fold_starts.push(n_samples); // sentinel end

    // ── Per-fold evaluation ───────────────────────────────────────────────────
    let mut per_fold_accuracy = Vec::with_capacity(config.n_folds);
    // Accumulate per-class F1 across folds (we'll average at the end).
    let mut per_class_f1_sum = vec![0.0_f64; config.n_classes];

    for fold_idx in 0..config.n_folds {
        let val_start = fold_starts[fold_idx];
        let val_end = fold_starts[fold_idx + 1];

        // Collect validation indices and training indices.
        let val_indices: Vec<usize> = indices[val_start..val_end].to_vec();
        let train_indices: Vec<usize> = indices[..val_start]
            .iter()
            .chain(&indices[val_end..])
            .copied()
            .collect();

        let n_train = train_indices.len();
        let n_val = val_indices.len();

        if n_train == 0 || n_val == 0 {
            return Err(SslError::BatchTooSmall);
        }

        // Build train/val feature arrays.
        let mut train_feat = vec![0.0_f64; n_train * in_dim];
        let mut train_lbl = vec![0usize; n_train];
        for (out_i, &src_i) in train_indices.iter().enumerate() {
            train_feat[out_i * in_dim..(out_i + 1) * in_dim]
                .copy_from_slice(&features[src_i * in_dim..(src_i + 1) * in_dim]);
            train_lbl[out_i] = labels[src_i];
        }

        let mut val_feat = vec![0.0_f64; n_val * in_dim];
        let mut val_lbl = vec![0usize; n_val];
        for (out_i, &src_i) in val_indices.iter().enumerate() {
            val_feat[out_i * in_dim..(out_i + 1) * in_dim]
                .copy_from_slice(&features[src_i * in_dim..(src_i + 1) * in_dim]);
            val_lbl[out_i] = labels[src_i];
        }

        // Fit and predict.
        let probe = linear_probe_fit(&train_feat, &train_lbl, n_train, in_dim, config)?;
        let preds = linear_probe_predict(&probe, &val_feat, n_val)?;

        // Metrics.
        let fold_acc = accuracy(&preds, &val_lbl);
        per_fold_accuracy.push(fold_acc);

        let f1s = f1_per_class(&preds, &val_lbl, config.n_classes);
        for (k, &f1_k) in f1s.iter().enumerate() {
            per_class_f1_sum[k] += f1_k;
        }
    }

    // ── Aggregate ─────────────────────────────────────────────────────────────
    let mean_accuracy = per_fold_accuracy.iter().sum::<f64>() / config.n_folds as f64;

    let variance = per_fold_accuracy
        .iter()
        .map(|&a| {
            let d = a - mean_accuracy;
            d * d
        })
        .sum::<f64>()
        / config.n_folds as f64;
    let std_accuracy = variance.sqrt();

    let per_class_f1: Vec<f64> = per_class_f1_sum
        .iter()
        .map(|&s| s / config.n_folds as f64)
        .collect();

    let macro_f1 = per_class_f1.iter().sum::<f64>() / config.n_classes as f64;

    Ok(LinearProbeResult {
        mean_accuracy,
        std_accuracy,
        per_fold_accuracy,
        macro_f1,
        per_class_f1,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a linearly separable binary dataset:
    /// first `n/2` samples at origin (label 0), last `n/2` samples at [c, 0, …] (label 1).
    fn make_binary_separable(n: usize, dim: usize, offset: f64) -> (Vec<f64>, Vec<usize>) {
        let half = n / 2;
        let mut feats = vec![0.0_f64; n * dim];
        let mut lbls = vec![0usize; n];
        for i in half..n {
            feats[i * dim] = offset;
            lbls[i] = 1;
        }
        (feats, lbls)
    }

    /// Build a 3-class perfectly separated dataset (each class in a corner).
    fn make_multiclass_separable(n_per_class: usize, dim: usize) -> (Vec<f64>, Vec<usize>) {
        let n = n_per_class * 3;
        let mut feats = vec![0.0_f64; n * dim];
        let mut lbls = vec![0usize; n];
        for k in 0..3usize {
            for i in 0..n_per_class {
                let row = k * n_per_class + i;
                // Place class k far from the others along dimension k.
                feats[row * dim + k.min(dim - 1)] = (k + 1) as f64 * 20.0;
                lbls[row] = k;
            }
        }
        (feats, lbls)
    }

    // ── test 1: config defaults ───────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = LinearProbeConfig::default();
        assert_eq!(cfg.n_folds, 5);
        assert_eq!(cfg.max_iter, 200);
        assert!((cfg.l2_reg - 1e-3).abs() < 1e-15);
        assert_eq!(cfg.n_classes, 2);
        assert!((cfg.tol - 1e-5).abs() < 1e-18);
        assert_eq!(cfg.seed, 42);
    }

    // ── test 2: sigmoid numerical stability ──────────────────────────────────

    #[test]
    fn sigmoid_stable() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-100.0) < 1e-6);
        // Check it doesn't produce NaN for extreme values.
        assert!(sigmoid(f64::MAX / 2.0).is_finite());
        assert!(sigmoid(f64::MIN / 2.0).is_finite());
    }

    // ── test 3: empty input error ─────────────────────────────────────────────

    #[test]
    fn fit_empty_error() {
        let cfg = LinearProbeConfig::default();
        let result = linear_probe_fit(&[], &[], 0, 4, &cfg);
        assert!(matches!(result, Err(SslError::EmptyInput)));
    }

    // ── test 4: single-class error ────────────────────────────────────────────

    #[test]
    fn fit_single_class_error() {
        let cfg = LinearProbeConfig {
            n_classes: 1,
            ..Default::default()
        };
        let feats = vec![0.0_f64; 10 * 4];
        let lbls = vec![0usize; 10];
        let result = linear_probe_fit(&feats, &lbls, 10, 4, &cfg);
        assert!(matches!(
            result,
            Err(SslError::InvalidParameter { name: _, reason: _ })
        ));
    }

    // ── test 5: binary linearly separable ────────────────────────────────────

    #[test]
    fn fit_binary_linearly_separable() {
        let cfg = LinearProbeConfig {
            n_classes: 2,
            max_iter: 200,
            l2_reg: 1e-4,
            ..Default::default()
        };
        let (feats, lbls) = make_binary_separable(20, 2, 10.0);
        let probe = linear_probe_fit(&feats, &lbls, 20, 2, &cfg).unwrap();
        let preds = linear_probe_predict(&probe, &feats, 20).unwrap();
        let acc = accuracy(&preds, &lbls);
        assert!(
            acc >= 0.9,
            "expected accuracy >= 0.9 on separable data, got {acc:.4}"
        );
    }

    // ── test 6: predict shape ─────────────────────────────────────────────────

    #[test]
    fn predict_shape() {
        let cfg = LinearProbeConfig::default();
        let (feats, lbls) = make_binary_separable(20, 4, 5.0);
        let probe = linear_probe_fit(&feats, &lbls, 20, 4, &cfg).unwrap();
        let preds = linear_probe_predict(&probe, &feats, 20).unwrap();
        assert_eq!(preds.len(), 20);
    }

    // ── test 7: multiclass perfectly separated → accuracy = 1.0 ──────────────

    #[test]
    fn fit_multiclass() {
        let cfg = LinearProbeConfig {
            n_classes: 3,
            max_iter: 300,
            l2_reg: 1e-4,
            ..Default::default()
        };
        let (feats, lbls) = make_multiclass_separable(10, 4);
        let probe = linear_probe_fit(&feats, &lbls, 30, 4, &cfg).unwrap();
        let preds = linear_probe_predict(&probe, &feats, 30).unwrap();
        let acc = accuracy(&preds, &lbls);
        assert!(
            (acc - 1.0).abs() < 1e-9,
            "expected perfect accuracy, got {acc:.4}"
        );
    }

    // ── test 8: weights layout ────────────────────────────────────────────────

    #[test]
    fn fit_returns_n_class_rows() {
        let cfg = LinearProbeConfig {
            n_classes: 3,
            ..Default::default()
        };
        let in_dim = 5;
        let (feats, lbls) = make_multiclass_separable(5, in_dim);
        let probe = linear_probe_fit(&feats, &lbls, 15, in_dim, &cfg).unwrap();
        assert_eq!(probe.weights.len(), cfg.n_classes * (in_dim + 1));
        assert_eq!(probe.in_dim, in_dim);
        assert_eq!(probe.n_classes, cfg.n_classes);
    }

    // ── test 9: CV mean accuracy on separable data ────────────────────────────

    #[test]
    fn eval_cv_mean_accuracy_positive() {
        let cfg = LinearProbeConfig {
            n_classes: 2,
            n_folds: 5,
            max_iter: 200,
            l2_reg: 1e-4,
            ..Default::default()
        };
        // Build a larger separable dataset so each fold has enough training data.
        let (feats, lbls) = make_binary_separable(50, 4, 10.0);
        let result = linear_probe_eval(&feats, &lbls, 50, 4, &cfg).unwrap();
        assert!(
            result.mean_accuracy > 0.8,
            "expected mean_accuracy > 0.8, got {:.4}",
            result.mean_accuracy
        );
    }

    // ── test 10: std accuracy finite and non-negative ─────────────────────────

    #[test]
    fn eval_std_accuracy_finite() {
        let cfg = LinearProbeConfig {
            n_classes: 2,
            n_folds: 5,
            l2_reg: 1e-3,
            ..Default::default()
        };
        let (feats, lbls) = make_binary_separable(50, 4, 10.0);
        let result = linear_probe_eval(&feats, &lbls, 50, 4, &cfg).unwrap();
        assert!(result.std_accuracy.is_finite());
        assert!(result.std_accuracy >= 0.0);
    }

    // ── test 11: macro F1 in [0, 1] ───────────────────────────────────────────

    #[test]
    fn eval_macro_f1_range() {
        let cfg = LinearProbeConfig {
            n_classes: 2,
            n_folds: 5,
            l2_reg: 1e-3,
            ..Default::default()
        };
        let (feats, lbls) = make_binary_separable(50, 4, 10.0);
        let result = linear_probe_eval(&feats, &lbls, 50, 4, &cfg).unwrap();
        assert!(
            result.macro_f1 >= 0.0 && result.macro_f1 <= 1.0,
            "macro_f1 = {:.4} out of [0, 1]",
            result.macro_f1
        );
    }

    // ── test 12: per_class_f1 length ──────────────────────────────────────────

    #[test]
    fn per_class_f1_length() {
        let cfg = LinearProbeConfig {
            n_classes: 3,
            n_folds: 3,
            l2_reg: 1e-3,
            ..Default::default()
        };
        let (feats, lbls) = make_multiclass_separable(15, 4);
        let result = linear_probe_eval(&feats, &lbls, 45, 4, &cfg).unwrap();
        assert_eq!(result.per_class_f1.len(), 3);
    }

    // ── test 13: cholesky_solve on identity ───────────────────────────────────

    #[test]
    fn cholesky_solve_identity() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![1.0, -2.0, std::f64::consts::PI, 0.0];
        let x = cholesky_solve(&a, &b, n).unwrap();
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-12, "expected x={bi}, got {xi}");
        }
    }

    // ── test 14 (bonus): cholesky_solve on a 3×3 SPD matrix ─────────────────

    #[test]
    fn cholesky_solve_spd_3x3() {
        // A = [[4,2,1],[2,5,3],[1,3,6]] — positive definite.
        let a = vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &b, 3).unwrap();
        // Verify A·x ≈ b.
        let ax0 = 4.0 * x[0] + 2.0 * x[1] + 1.0 * x[2];
        let ax1 = 2.0 * x[0] + 5.0 * x[1] + 3.0 * x[2];
        let ax2 = 1.0 * x[0] + 3.0 * x[1] + 6.0 * x[2];
        assert!((ax0 - 1.0).abs() < 1e-10);
        assert!((ax1 - 2.0).abs() < 1e-10);
        assert!((ax2 - 3.0).abs() < 1e-10);
    }
}
