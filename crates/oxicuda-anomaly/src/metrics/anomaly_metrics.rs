//! Evaluation metrics for anomaly detection.
//!
//! * AUC-ROC: trapezoidal area under the ROC curve
//! * AUC-PR:  area under the precision-recall curve
//! * F1 @ threshold: binary classification F1
//! * `compute_detection_metrics`: all-in-one

use crate::error::{AnomalyError, AnomalyResult};

// ─── auc_roc_anomaly ──────────────────────────────────────────────────────────

/// AUC-ROC for anomaly detection.
///
/// `scores`: higher = more anomalous.
/// `is_anomaly`: `true` = positive class (anomaly).
pub fn auc_roc_anomaly(scores: &[f32], is_anomaly: &[bool]) -> AnomalyResult<f32> {
    if scores.len() != is_anomaly.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: scores.len(),
            got: is_anomaly.len(),
        });
    }
    if scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }

    let n_pos = is_anomaly.iter().filter(|&&b| b).count();
    let n_neg = is_anomaly.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5); // Undefined; return chance level
    }

    // Sort by score descending
    let mut pairs: Vec<(f32, bool)> = scores
        .iter()
        .cloned()
        .zip(is_anomaly.iter().cloned())
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Trapezoidal AUC
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut auc = 0.0_f32;
    let mut prev_tp = 0_usize;
    let mut prev_fp = 0_usize;

    for &(_, label) in &pairs {
        if label {
            tp += 1;
        } else {
            fp += 1;
        }
        // Trapezoid contribution
        let tpr = tp as f32 / n_pos as f32;
        let fpr = fp as f32 / n_neg as f32;
        let prev_tpr = prev_tp as f32 / n_pos as f32;
        let prev_fpr = prev_fp as f32 / n_neg as f32;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tp = tp;
        prev_fp = fp;
    }

    Ok(auc.clamp(0.0, 1.0))
}

// ─── f1_at_threshold ─────────────────────────────────────────────────────────

/// F1 score at a given score threshold.
///
/// Predictions: `score > threshold` → anomaly.
pub fn f1_at_threshold(scores: &[f32], is_anomaly: &[bool], threshold: f32) -> AnomalyResult<f32> {
    if scores.len() != is_anomaly.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: scores.len(),
            got: is_anomaly.len(),
        });
    }
    if scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut fn_ = 0_usize;

    for (&s, &label) in scores.iter().zip(is_anomaly.iter()) {
        let pred = s > threshold;
        match (pred, label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }

    let precision = if tp + fp == 0 {
        0.0_f32
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_ == 0 {
        0.0_f32
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let f1 = if precision + recall < 1e-8 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok(f1)
}

// ─── auc_pr ───────────────────────────────────────────────────────────────────

/// AUC-PR (area under the precision-recall curve).
///
/// Thresholds are swept by sorting scores descending.
pub fn auc_pr(scores: &[f32], is_anomaly: &[bool]) -> AnomalyResult<f32> {
    if scores.len() != is_anomaly.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: scores.len(),
            got: is_anomaly.len(),
        });
    }
    if scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }

    let n_pos = is_anomaly.iter().filter(|&&b| b).count();
    if n_pos == 0 {
        return Ok(0.0);
    }

    let mut pairs: Vec<(f32, bool)> = scores
        .iter()
        .cloned()
        .zip(is_anomaly.iter().cloned())
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut auc = 0.0_f32;
    let mut prev_recall = 0.0_f32;
    let mut prev_precision = 1.0_f32;

    for &(_, label) in &pairs {
        if label {
            tp += 1;
        } else {
            fp += 1;
        }
        let recall = tp as f32 / n_pos as f32;
        let precision = tp as f32 / (tp + fp) as f32;
        // Trapezoidal interpolation
        auc += (recall - prev_recall) * (precision + prev_precision) * 0.5;
        prev_recall = recall;
        prev_precision = precision;
    }

    Ok(auc.clamp(0.0, 1.0))
}

// ─── AnomalyDetectionMetrics ──────────────────────────────────────────────────

/// Summary metrics for anomaly detection performance.
pub struct AnomalyDetectionMetrics {
    /// AUC-ROC.
    pub auc_roc: f32,
    /// AUC-PR.
    pub auc_pr: f32,
    /// Best F1 over all thresholds.
    pub best_f1: f32,
    /// Threshold achieving the best F1.
    pub best_threshold: f32,
}

/// Compute all detection metrics in one pass.
pub fn compute_detection_metrics(
    scores: &[f32],
    is_anomaly: &[bool],
) -> AnomalyResult<AnomalyDetectionMetrics> {
    if scores.len() != is_anomaly.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: scores.len(),
            got: is_anomaly.len(),
        });
    }
    if scores.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }

    let auc_roc_val = auc_roc_anomaly(scores, is_anomaly)?;
    let auc_pr_val = auc_pr(scores, is_anomaly)?;

    // Best F1: sweep over all unique score thresholds
    let mut unique_scores: Vec<f32> = scores.to_vec();
    unique_scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_scores.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut best_f1 = 0.0_f32;
    let mut best_threshold = f32::NEG_INFINITY;

    for &thresh in &unique_scores {
        let f1 = f1_at_threshold(scores, is_anomaly, thresh)?;
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thresh;
        }
    }

    Ok(AnomalyDetectionMetrics {
        auc_roc: auc_roc_val,
        auc_pr: auc_pr_val,
        best_f1,
        best_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_roc_perfect() {
        let scores = vec![0.9_f32, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let auc = auc_roc_anomaly(&scores, &labels).unwrap();
        assert!((auc - 1.0).abs() < 0.01, "auc={auc}");
    }

    #[test]
    fn auc_roc_random() {
        let scores = vec![0.5_f32; 10];
        let labels = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let auc = auc_roc_anomaly(&scores, &labels).unwrap();
        assert!((0.0..=1.0).contains(&auc), "auc={auc}");
    }

    #[test]
    fn f1_threshold_basic() {
        let scores = vec![0.9_f32, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let f1 = f1_at_threshold(&scores, &labels, 0.5).unwrap();
        assert!((f1 - 1.0).abs() < 0.01, "f1={f1}");
    }

    #[test]
    fn compute_all_metrics_finite() {
        let scores = vec![0.9_f32, 0.7, 0.3, 0.1, 0.8, 0.2];
        let labels = vec![true, true, false, false, true, false];
        let m = compute_detection_metrics(&scores, &labels).unwrap();
        assert!(m.auc_roc.is_finite());
        assert!(m.auc_pr.is_finite());
        assert!(m.best_f1.is_finite());
    }
}
