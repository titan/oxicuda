//! Agreement metrics between student and teacher prediction distributions.

use crate::error::{DistillError, DistillResult};

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

/// Top-k agreement: mean fraction of top-k predictions shared between student and teacher.
pub fn top_k_agreement(
    s_logits: &[Vec<f32>],
    t_logits: &[Vec<f32>],
    k: usize,
) -> DistillResult<f32> {
    if s_logits.is_empty() || t_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_logits.len() != t_logits.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_logits.len(),
            got: t_logits.len(),
        });
    }
    if k == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "k must be > 0".into(),
        });
    }
    let mut total = 0.0_f32;
    for (s, t) in s_logits.iter().zip(t_logits.iter()) {
        let s_top: std::collections::HashSet<usize> = top_k_indices(s, k).into_iter().collect();
        let t_top: std::collections::HashSet<usize> = top_k_indices(t, k).into_iter().collect();
        let intersection = s_top.intersection(&t_top).count();
        total += intersection as f32 / k as f32;
    }
    Ok(total / s_logits.len() as f32)
}

/// Cohen's kappa agreement statistic between two sets of discrete predictions.
pub fn cohen_kappa(s_preds: &[usize], t_preds: &[usize], num_classes: usize) -> DistillResult<f32> {
    if s_preds.is_empty() || t_preds.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_preds.len() != t_preds.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_preds.len(),
            got: t_preds.len(),
        });
    }
    let n = s_preds.len() as f32;
    let p_o: f32 = s_preds
        .iter()
        .zip(t_preds.iter())
        .filter(|&(s, t)| s == t)
        .count() as f32
        / n;

    let mut count_s = vec![0usize; num_classes];
    let mut count_t = vec![0usize; num_classes];
    for (&s, &t) in s_preds.iter().zip(t_preds.iter()) {
        if s < num_classes {
            count_s[s] += 1;
        }
        if t < num_classes {
            count_t[t] += 1;
        }
    }
    let p_e: f32 = (0..num_classes)
        .map(|c| (count_s[c] as f32 / n) * (count_t[c] as f32 / n))
        .sum();

    let denom = 1.0 - p_e;
    if denom.abs() < 1e-10 {
        return Ok(1.0);
    }
    Ok((p_o - p_e) / denom)
}

/// Simple prediction overlap: fraction of samples where s == t.
pub fn prediction_overlap(s_preds: &[usize], t_preds: &[usize]) -> DistillResult<f32> {
    if s_preds.is_empty() || t_preds.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_preds.len() != t_preds.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_preds.len(),
            got: t_preds.len(),
        });
    }
    let overlap = s_preds
        .iter()
        .zip(t_preds.iter())
        .filter(|&(a, b)| a == b)
        .count();
    Ok(overlap as f32 / s_preds.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_agreement_perfect() {
        let logits = vec![vec![3.0_f32, 1.0, 2.0], vec![1.0, 3.0, 2.0]];
        let agree = top_k_agreement(&logits, &logits, 1).expect("top_k_agreement should succeed");
        assert!((agree - 1.0).abs() < 1e-5);
    }

    #[test]
    fn prediction_overlap_all_same() {
        let preds = vec![0_usize, 1, 2, 1, 0];
        assert!(
            (prediction_overlap(&preds, &preds).expect("prediction_overlap should succeed") - 1.0)
                .abs()
                < 1e-5
        );
    }
}
