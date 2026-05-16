//! DIST loss (Huang et al. 2022) — Pearson correlation for non-target distribution alignment.

use crate::error::{DistillError, DistillResult};

/// Pearson correlation coefficient between two equal-length slices.
///
/// Returns (Σ(xᵢ−x̄)(yᵢ−ȳ)) / (σ_x · σ_y + ε), ε = 1e-8.
#[must_use]
pub fn pearson_corr(x: &[f32], y: &[f32]) -> f32 {
    const EPS: f32 = 1e-8;
    let n = x.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;
    let cov: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let var_x: f32 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f32>();
    let var_y: f32 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f32>();
    cov / (var_x.sqrt() * var_y.sqrt() + EPS)
}

/// Per-sample inter-class loss: 1 − Pearson(student_logits, teacher_logits).
#[must_use]
pub fn inter_class_loss(student_logits: &[f32], teacher_logits: &[f32]) -> f32 {
    1.0 - pearson_corr(student_logits, teacher_logits)
}

/// Intra-class loss for class `class_idx` across a batch: 1 − Pearson over the batch axis.
///
/// Extracts column `class_idx` from each sample to form a per-sample scalar, then computes
/// Pearson correlation between the student and teacher column vectors.
pub fn intra_class_loss(
    s_batch: &[Vec<f32>],
    t_batch: &[Vec<f32>],
    class_idx: usize,
) -> DistillResult<f32> {
    if s_batch.is_empty() || t_batch.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_batch.len() != t_batch.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_batch.len(),
            got: t_batch.len(),
        });
    }
    let s_col: Vec<f32> = s_batch
        .iter()
        .map(|row| {
            if class_idx < row.len() {
                row[class_idx]
            } else {
                0.0
            }
        })
        .collect();
    let t_col: Vec<f32> = t_batch
        .iter()
        .map(|row| {
            if class_idx < row.len() {
                row[class_idx]
            } else {
                0.0
            }
        })
        .collect();
    Ok(1.0 - pearson_corr(&s_col, &t_col))
}

/// Compute the DIST loss over a batch.
///
/// `dist_loss = mean(inter_class_loss) + beta * mean(intra_class_loss over classes)`
/// `gamma` is accepted for API consistency but not used.
pub fn dist_loss(
    s_batch: &[Vec<f32>],
    t_batch: &[Vec<f32>],
    beta: f32,
    _gamma: f32,
) -> DistillResult<f32> {
    if s_batch.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_batch.len() != t_batch.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_batch.len(),
            got: t_batch.len(),
        });
    }
    // Mean inter-class loss across samples.
    let inter_mean: f32 = s_batch
        .iter()
        .zip(t_batch.iter())
        .map(|(s, t)| inter_class_loss(s, t))
        .sum::<f32>()
        / s_batch.len() as f32;

    // Mean intra-class loss across classes.
    let num_classes = s_batch.first().map(|r| r.len()).unwrap_or(0);
    let intra_mean = if num_classes == 0 {
        0.0
    } else {
        let mut sum = 0.0_f32;
        for c in 0..num_classes {
            sum += intra_class_loss(s_batch, t_batch, c)?;
        }
        sum / num_classes as f32
    };

    Ok(inter_mean + beta * intra_mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_corr_identical() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let r = pearson_corr(&v, &v);
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pearson_corr_anticorrelated() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let y = vec![3.0_f32, 2.0, 1.0];
        let r = pearson_corr(&x, &y);
        assert!((r + 1.0).abs() < 1e-5);
    }

    #[test]
    fn dist_loss_finite() {
        let s = vec![vec![1.0_f32, 2.0, 3.0], vec![1.5, 2.5, 3.5]];
        let t = vec![vec![1.1_f32, 1.9, 3.1], vec![1.4, 2.6, 3.4]];
        let loss = dist_loss(&s, &t, 1.0, 0.5).unwrap();
        assert!(loss.is_finite());
    }
}
