//! PR-AUC, log-loss, Brier score, ECE and related binary/multiclass metrics.

use crate::error::{TabularError, TabularResult};

// ─── Validation helpers ───────────────────────────────────────────────────────

fn check_binary_inputs(probs: &[f32], labels: &[u32]) -> TabularResult<()> {
    if probs.len() != labels.len() {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len(),
            got: probs.len(),
        });
    }
    if probs.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    Ok(())
}

// ─── Binary metrics ───────────────────────────────────────────────────────────

/// Binary cross-entropy (log-loss): −(1/n)Σ[y·log(p) + (1−y)·log(1−p)].
///
/// Probabilities are clipped to `[eps, 1−eps]` before taking logarithms.
pub fn log_loss(probs: &[f32], labels: &[u32], eps: f32) -> TabularResult<f32> {
    check_binary_inputs(probs, labels)?;
    let n = probs.len() as f32;
    let total: f32 = probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let p_clip = p.clamp(eps, 1.0 - eps);
            let yf = y as f32;
            -(yf * p_clip.ln() + (1.0 - yf) * (1.0 - p_clip).ln())
        })
        .sum();
    Ok(total / n)
}

/// Binary Brier score: (1/n)Σ(p − y)².
pub fn brier_score(probs: &[f32], labels: &[u32]) -> TabularResult<f32> {
    check_binary_inputs(probs, labels)?;
    let n = probs.len() as f32;
    let total: f32 = probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let diff = p - y as f32;
            diff * diff
        })
        .sum();
    Ok(total / n)
}

/// Precision at threshold t: TP / (TP + FP).
///
/// Returns `1.0` when there are no positive predictions (convention: perfect precision).
pub fn precision_at_threshold(probs: &[f32], labels: &[u32], threshold: f32) -> TabularResult<f32> {
    check_binary_inputs(probs, labels)?;
    let mut tp = 0usize;
    let mut fp = 0usize;
    for (&p, &y) in probs.iter().zip(labels.iter()) {
        if p >= threshold {
            if y == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    let denom = tp + fp;
    Ok(if denom == 0 {
        1.0
    } else {
        tp as f32 / denom as f32
    })
}

/// Recall at threshold t: TP / (TP + FN).
///
/// Returns `0.0` when there are no actual positives.
pub fn recall_at_threshold(probs: &[f32], labels: &[u32], threshold: f32) -> TabularResult<f32> {
    check_binary_inputs(probs, labels)?;
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    for (&p, &y) in probs.iter().zip(labels.iter()) {
        if y == 1 {
            if p >= threshold {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    let denom = tp + fn_;
    Ok(if denom == 0 {
        0.0
    } else {
        tp as f32 / denom as f32
    })
}

/// F1 score at threshold: 2·P·R / (P + R).
///
/// Returns `0.0` when P + R = 0.
pub fn f1_at_threshold(probs: &[f32], labels: &[u32], threshold: f32) -> TabularResult<f32> {
    let p = precision_at_threshold(probs, labels, threshold)?;
    let r = recall_at_threshold(probs, labels, threshold)?;
    let denom = p + r;
    Ok(if denom < 1e-12 {
        0.0
    } else {
        2.0 * p * r / denom
    })
}

// ─── Precision-Recall curve ───────────────────────────────────────────────────

/// Full precision-recall curve, sorted by decreasing threshold.
///
/// Returns `(thresholds, precisions, recalls)`.
/// The curve appends a sentinel point `(threshold=0, precision=1, recall=recall_at_0)`.
pub fn precision_recall_curve(
    probs: &[f32],
    labels: &[u32],
) -> TabularResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    check_binary_inputs(probs, labels)?;

    let n_pos: usize = labels.iter().filter(|&&y| y == 1).count();

    // Sort by decreasing predicted probability
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut thresholds = Vec::new();
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_threshold = f32::INFINITY;

    for &idx in &indices {
        let threshold = probs[idx];
        let label = labels[idx];

        // Emit a point whenever the threshold changes
        if (threshold - prev_threshold).abs() > 1e-9 {
            let denom = tp + fp;
            let prec = if denom == 0 {
                1.0
            } else {
                tp as f32 / denom as f32
            };
            let rec = if n_pos == 0 {
                0.0
            } else {
                tp as f32 / n_pos as f32
            };
            thresholds.push(prev_threshold);
            precisions.push(prec);
            recalls.push(rec);
            prev_threshold = threshold;
        }

        if label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    // Final threshold point
    {
        let denom = tp + fp;
        let prec = if denom == 0 {
            1.0
        } else {
            tp as f32 / denom as f32
        };
        let rec = if n_pos == 0 {
            0.0
        } else {
            tp as f32 / n_pos as f32
        };
        thresholds.push(prev_threshold);
        precisions.push(prec);
        recalls.push(rec);
    }

    // Sentinel: threshold=0, recall=1 (all samples predicted positive)
    thresholds.push(0.0);
    precisions.push(if n_pos == 0 {
        0.0
    } else {
        n_pos as f32 / probs.len() as f32
    });
    recalls.push(if n_pos == 0 { 0.0 } else { 1.0 });

    Ok((thresholds, precisions, recalls))
}

/// PR-AUC via trapezoidal integration over the precision-recall curve.
pub fn pr_auc(probs: &[f32], labels: &[u32]) -> TabularResult<f32> {
    let (_, precisions, recalls) = precision_recall_curve(probs, labels)?;
    let mut area = 0.0_f32;
    for i in 1..recalls.len() {
        let dr = (recalls[i] - recalls[i - 1]).abs();
        let avg_p = 0.5 * (precisions[i] + precisions[i - 1]);
        area += avg_p * dr;
    }
    Ok(area.clamp(0.0, 1.0))
}

/// Average Precision: `Σ_k precision[k] · (recall[k] − recall[k-1])`.
///
/// Points are sorted by ascending recall for the interpolation.
pub fn average_precision(probs: &[f32], labels: &[u32]) -> TabularResult<f32> {
    let (_, mut precisions, mut recalls) = precision_recall_curve(probs, labels)?;

    // Sort by ascending recall
    let mut idx: Vec<usize> = (0..recalls.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        recalls[a]
            .partial_cmp(&recalls[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_prec: Vec<f32> = idx.iter().map(|&i| precisions[i]).collect();
    let sorted_rec: Vec<f32> = idx.iter().map(|&i| recalls[i]).collect();
    precisions = sorted_prec;
    recalls = sorted_rec;

    let mut ap = 0.0_f32;
    for k in 1..recalls.len() {
        let dr = recalls[k] - recalls[k - 1];
        if dr > 0.0 {
            ap += precisions[k] * dr;
        }
    }
    Ok(ap.clamp(0.0, 1.0))
}

// ─── Multiclass metrics ───────────────────────────────────────────────────────

/// Multiclass log-loss: −(1/n)Σ_i log(p_{i, y_i}).
///
/// `probs` is `n_samples × n_classes`, row-major. Probabilities clipped to `[eps, 1−eps]`.
pub fn multiclass_log_loss(
    probs: &[f32],
    labels: &[u32],
    n_classes: usize,
    eps: f32,
) -> TabularResult<f32> {
    if n_classes < 2 {
        return Err(TabularError::Internal {
            msg: "n_classes must be >= 2".into(),
        });
    }
    let n_samples = labels.len();
    if n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if probs.len() != n_samples * n_classes {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_classes,
            got: probs.len(),
        });
    }
    let mut total = 0.0_f32;
    for (i, &y) in labels.iter().enumerate() {
        if y as usize >= n_classes {
            return Err(TabularError::LabelOutOfRange {
                label: y as usize,
                n_classes,
            });
        }
        let p = probs[i * n_classes + y as usize].clamp(eps, 1.0 - eps);
        total -= p.ln();
    }
    Ok(total / n_samples as f32)
}

/// Macro-averaged precision, recall, F1 for multiclass (one-vs-rest per class).
///
/// `probs` is `n_samples × n_classes`, row-major; argmax gives the predicted class.
pub fn multiclass_prf(
    probs: &[f32],
    labels: &[u32],
    n_classes: usize,
    threshold: f32,
) -> TabularResult<(f32, f32, f32)> {
    if n_classes < 2 {
        return Err(TabularError::Internal {
            msg: "n_classes must be >= 2".into(),
        });
    }
    let n_samples = labels.len();
    if n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if probs.len() != n_samples * n_classes {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_classes,
            got: probs.len(),
        });
    }

    let mut macro_p = 0.0_f32;
    let mut macro_r = 0.0_f32;
    let mut macro_f = 0.0_f32;

    for c in 0..n_classes {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (i, &y) in labels.iter().enumerate() {
            let row = &probs[i * n_classes..(i + 1) * n_classes];
            let pred_positive = row[c] >= threshold;
            let actual_positive = y as usize == c;

            match (pred_positive, actual_positive) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let p = if tp + fp == 0 {
            0.0
        } else {
            tp as f32 / (tp + fp) as f32
        };
        let r = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f32 / (tp + fn_) as f32
        };
        let f = if p + r < 1e-12 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };

        macro_p += p;
        macro_r += r;
        macro_f += f;
    }

    let nc = n_classes as f32;
    Ok((macro_p / nc, macro_r / nc, macro_f / nc))
}

// ─── Calibration ─────────────────────────────────────────────────────────────

/// Expected Calibration Error (ECE) for binary classifier.
///
/// Bins predicted probabilities into `n_bins` equally-spaced intervals on [0, 1].
/// ECE = Σ_b (|b| / n) · |avg_confidence_b − accuracy_b|.
pub fn binary_ece(probs: &[f32], labels: &[u32], n_bins: usize) -> TabularResult<f32> {
    check_binary_inputs(probs, labels)?;
    if n_bins == 0 {
        return Err(TabularError::Internal {
            msg: "n_bins must be > 0".into(),
        });
    }

    let n = probs.len();
    // Per-bin accumulators: (sum_confidence, sum_correct, count)
    let mut bin_conf = vec![0.0_f32; n_bins];
    let mut bin_acc = vec![0.0_f32; n_bins];
    let mut bin_cnt = vec![0usize; n_bins];

    for (&p, &y) in probs.iter().zip(labels.iter()) {
        // Map p in [0,1) to a bin index; p==1.0 goes into last bin
        let bin = ((p * n_bins as f32) as usize).min(n_bins - 1);
        bin_conf[bin] += p;
        bin_acc[bin] += y as f32;
        bin_cnt[bin] += 1;
    }

    let mut ece = 0.0_f32;
    for b in 0..n_bins {
        let cnt = bin_cnt[b];
        if cnt == 0 {
            continue;
        }
        let avg_conf = bin_conf[b] / cnt as f32;
        let avg_acc = bin_acc[b] / cnt as f32;
        ece += (cnt as f32 / n as f32) * (avg_conf - avg_acc).abs();
    }

    Ok(ece)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-7;

    #[test]
    fn log_loss_perfect_predictions() {
        let probs = vec![1.0_f32, 1.0, 0.0, 0.0];
        let labels = vec![1u32, 1, 0, 0];
        let loss = log_loss(&probs, &labels, 1e-7).unwrap();
        // With clipping p→1-eps, loss ≈ -log(1-eps) ≈ eps (very small)
        assert!(loss < 1e-5, "loss={loss}");
    }

    #[test]
    fn log_loss_random_noise() {
        let probs = vec![0.5_f32; 10];
        let labels = vec![1u32, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let loss = log_loss(&probs, &labels, 1e-7).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn brier_score_perfect() {
        let probs = vec![1.0_f32, 0.0, 1.0, 0.0];
        let labels = vec![1u32, 0, 1, 0];
        let bs = brier_score(&probs, &labels).unwrap();
        assert!(bs.abs() < 1e-6, "bs={bs}");
    }

    #[test]
    fn brier_score_worst_case() {
        let probs = vec![1.0_f32, 1.0];
        let labels = vec![0u32, 0];
        let bs = brier_score(&probs, &labels).unwrap();
        assert!((bs - 1.0).abs() < 1e-6, "bs={bs}");
    }

    #[test]
    fn brier_score_random() {
        let probs = vec![0.3_f32, 0.7, 0.9, 0.1];
        let labels = vec![0u32, 1, 1, 0];
        let bs = brier_score(&probs, &labels).unwrap();
        assert!((0.0..=1.0).contains(&bs), "bs={bs}");
    }

    #[test]
    fn precision_at_threshold_all_positive() {
        let probs = vec![0.9_f32, 0.8, 0.7, 0.6];
        let labels = vec![1u32, 0, 1, 0];
        // threshold = 0 → all predicted positive
        let prec = precision_at_threshold(&probs, &labels, 0.0).unwrap();
        // TP=2, FP=2 → precision = 0.5 = mean(labels)
        assert!((prec - 0.5).abs() < 1e-6, "prec={prec}");
    }

    #[test]
    fn recall_perfect_threshold() {
        let probs = vec![0.9_f32, 0.8, 0.7, 0.6];
        let labels = vec![1u32, 0, 1, 0];
        // threshold = 0 → all predicted positive → all positives recalled
        let rec = recall_at_threshold(&probs, &labels, 0.0).unwrap();
        assert!((rec - 1.0).abs() < 1e-6, "rec={rec}");
    }

    #[test]
    fn f1_harmonic_mean() {
        let probs = vec![0.9_f32, 0.8, 0.2, 0.1];
        let labels = vec![1u32, 1, 0, 0];
        let threshold = 0.5;
        let p = precision_at_threshold(&probs, &labels, threshold).unwrap();
        let r = recall_at_threshold(&probs, &labels, threshold).unwrap();
        let f1_expected = if p + r < 1e-12 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };
        let f1 = f1_at_threshold(&probs, &labels, threshold).unwrap();
        assert!(
            (f1 - f1_expected).abs() < 1e-6,
            "f1={f1} expected={f1_expected}"
        );
    }

    #[test]
    fn pr_curve_monotone() {
        let probs = vec![0.9_f32, 0.8, 0.7, 0.4, 0.3, 0.2];
        let labels = vec![1u32, 0, 1, 0, 1, 0];
        let (_, _, recalls) = precision_recall_curve(&probs, &labels).unwrap();
        // Recalls should be non-decreasing as we move along the curve (lower thresholds)
        // The curve is sorted by decreasing threshold, so recalls are non-decreasing
        for i in 1..recalls.len() {
            assert!(
                recalls[i] >= recalls[i - 1] - 1e-6,
                "recall not monotone at i={i}: {} < {}",
                recalls[i],
                recalls[i - 1]
            );
        }
    }

    #[test]
    fn pr_auc_in_range() {
        let probs = vec![0.9_f32, 0.4, 0.35, 0.8, 0.65, 0.7];
        let labels = vec![1u32, 1, 0, 1, 0, 1];
        let auc = pr_auc(&probs, &labels).unwrap();
        assert!((0.0..=1.0).contains(&auc), "auc={auc}");
    }

    #[test]
    fn pr_auc_perfect_classifier() {
        // Perfect classifier: all positives have higher score than all negatives
        let probs = vec![0.9_f32, 0.8, 0.7, 0.2, 0.1, 0.05];
        let labels = vec![1u32, 1, 1, 0, 0, 0];
        let auc = pr_auc(&probs, &labels).unwrap();
        assert!(auc > 0.95, "auc={auc}");
    }

    #[test]
    fn average_precision_in_range() {
        let probs = vec![0.9_f32, 0.4, 0.35, 0.8, 0.65, 0.7];
        let labels = vec![1u32, 1, 0, 1, 0, 1];
        let ap = average_precision(&probs, &labels).unwrap();
        assert!((0.0..=1.0).contains(&ap), "ap={ap}");
    }

    #[test]
    fn multiclass_log_loss_one_class() {
        let probs = vec![0.5_f32, 0.5];
        let labels = vec![0u32, 0];
        let result = multiclass_log_loss(&probs, &labels, 1, EPS);
        assert!(result.is_err());
    }

    #[test]
    fn multiclass_prf_shape() {
        let probs = vec![
            0.8_f32, 0.1, 0.1, // sample 0: class 0
            0.2, 0.6, 0.2, // sample 1: class 1
            0.1, 0.2, 0.7, // sample 2: class 2
        ];
        let labels = vec![0u32, 1, 2];
        let (p, r, f) = multiclass_prf(&probs, &labels, 3, 0.5).unwrap();
        assert!(p.is_finite());
        assert!(r.is_finite());
        assert!(f.is_finite());
    }

    #[test]
    fn binary_ece_perfect_calibration() {
        // Bins where avg_confidence == accuracy → ECE = 0
        // Use 2 bins: [0, 0.5) and [0.5, 1]
        // All predictions in bin 0 (p=0.25) have label 0 → acc=0, conf=0.25 → not perfect
        // Instead use predictions that equal labels exactly
        let probs = vec![0.0_f32, 0.0, 1.0, 1.0];
        let labels = vec![0u32, 0, 1, 1];
        let ece = binary_ece(&probs, &labels, 10).unwrap();
        // conf=0 for negatives, acc=0 → |0-0|=0; conf=1 for positives, acc=1 → |1-1|=0
        assert!(ece.abs() < 1e-5, "ece={ece}");
    }

    #[test]
    fn binary_ece_in_range() {
        let probs = vec![0.9_f32, 0.4, 0.35, 0.8, 0.65, 0.7];
        let labels = vec![1u32, 1, 0, 1, 0, 1];
        let ece = binary_ece(&probs, &labels, 5).unwrap();
        assert!((0.0..=1.0).contains(&ece), "ece={ece}");
    }

    #[test]
    fn log_loss_err_length_mismatch() {
        let probs = vec![0.5_f32, 0.5];
        let labels = vec![1u32];
        let result = log_loss(&probs, &labels, 1e-7);
        assert!(result.is_err());
    }

    #[test]
    fn brier_score_err_empty() {
        let result = brier_score(&[], &[]);
        assert!(result.is_err());
    }
}
