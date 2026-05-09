//! Evaluation metrics for tabular learning: accuracy, RMSE, MAE, AUC-ROC.

use crate::error::{TabularError, TabularResult};

/// Binary classification accuracy at a given threshold.
pub fn binary_accuracy(preds: &[f32], labels: &[u32], threshold: f32) -> TabularResult<f32> {
    if preds.len() != labels.len() {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len(),
            got: preds.len(),
        });
    }
    if preds.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|&(&p, &l)| {
            let pred_label = u32::from(p >= threshold);
            pred_label == l
        })
        .count();
    Ok(correct as f32 / preds.len() as f32)
}

/// Multi-class accuracy: each row of `logits` has `n_classes` values; argmax is the predicted class.
pub fn multiclass_accuracy(logits: &[f32], labels: &[u32], n_classes: usize) -> TabularResult<f32> {
    if n_classes == 0 {
        return Err(TabularError::Internal {
            msg: "n_classes must be > 0".into(),
        });
    }
    if logits.len() != labels.len() * n_classes {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: logits.len(),
        });
    }
    if labels.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    let correct = labels
        .iter()
        .enumerate()
        .filter(|&(i, &l)| {
            let row = &logits[i * n_classes..(i + 1) * n_classes];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            pred == l as usize
        })
        .count();
    Ok(correct as f32 / labels.len() as f32)
}

/// Root mean squared error.
pub fn rmse(preds: &[f32], targets: &[f32]) -> TabularResult<f32> {
    if preds.len() != targets.len() {
        return Err(TabularError::DimensionMismatch {
            expected: targets.len(),
            got: preds.len(),
        });
    }
    if preds.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    let mse: f32 = preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f32>()
        / preds.len() as f32;
    Ok(mse.sqrt())
}

/// Mean absolute error.
pub fn mae(preds: &[f32], targets: &[f32]) -> TabularResult<f32> {
    if preds.len() != targets.len() {
        return Err(TabularError::DimensionMismatch {
            expected: targets.len(),
            got: preds.len(),
        });
    }
    if preds.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    let total: f32 = preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum();
    Ok(total / preds.len() as f32)
}

/// AUC-ROC via trapezoidal rule.
///
/// Sorts by descending score and computes FPR/TPR at each threshold.
pub fn auc_roc(scores: &[f32], labels: &[u32]) -> TabularResult<f32> {
    if scores.len() != labels.len() {
        return Err(TabularError::DimensionMismatch {
            expected: labels.len(),
            got: scores.len(),
        });
    }
    if scores.is_empty() {
        return Err(TabularError::EmptyInput);
    }

    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = labels.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err(TabularError::NormalizationFailed {
            msg: "AUC requires both positive and negative labels".into(),
        });
    }

    // Sort by descending score
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut auc = 0.0_f32;
    let mut prev_tpr = 0.0_f32;
    let mut prev_fpr = 0.0_f32;

    for &idx in &order {
        if labels[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let tpr = tp as f32 / n_pos as f32;
        let fpr = fp as f32 / n_neg as f32;
        // Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    Ok(auc)
}

/// Summary of binary classification metrics.
pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub auc: f32,
    pub n_correct: usize,
    pub n_total: usize,
}

/// Compute binary accuracy and AUC together.
pub fn compute_binary_metrics(
    scores: &[f32],
    labels: &[u32],
    threshold: f32,
) -> TabularResult<ClassificationMetrics> {
    let acc = binary_accuracy(scores, labels, threshold)?;
    let auc = auc_roc(scores, labels)?;
    let n_correct = scores
        .iter()
        .zip(labels.iter())
        .filter(|&(&p, &l)| u32::from(p >= threshold) == l)
        .count();
    Ok(ClassificationMetrics {
        accuracy: acc,
        auc,
        n_correct,
        n_total: labels.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_accuracy_perfect() {
        let preds = vec![0.9_f32, 0.1, 0.8, 0.2];
        let labels = vec![1u32, 0, 1, 0];
        let acc = binary_accuracy(&preds, &labels, 0.5).unwrap();
        assert!((acc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn auc_roc_perfect() {
        let scores = vec![0.9_f32, 0.8, 0.3, 0.2];
        let labels = vec![1u32, 1, 0, 0];
        let auc = auc_roc(&scores, &labels).unwrap();
        assert!((auc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rmse_zero_for_perfect() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.0_f32, 2.0, 3.0];
        let r = rmse(&preds, &targets).unwrap();
        assert!(r < 1e-6);
    }

    #[test]
    fn mae_known_value() {
        let preds = vec![1.0_f32, 3.0];
        let targets = vec![0.0_f32, 0.0];
        let m = mae(&preds, &targets).unwrap();
        assert!((m - 2.0).abs() < 1e-5);
    }
}
