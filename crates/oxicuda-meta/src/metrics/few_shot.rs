use crate::error::{MetaError, MetaResult};

pub fn episode_accuracy(preds: &[u32], labels: &[u32]) -> MetaResult<f32> {
    if preds.len() != labels.len() {
        return Err(MetaError::DimensionMismatch {
            expected: labels.len(),
            got: preds.len(),
        });
    }
    if preds.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|&(&p, &l)| p == l)
        .count();
    Ok(correct as f32 / preds.len() as f32)
}

pub fn mean_and_ci95(accuracies: &[f32]) -> MetaResult<(f32, f32)> {
    if accuracies.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    let n = accuracies.len() as f32;
    let mean = accuracies.iter().sum::<f32>() / n;
    let variance = accuracies
        .iter()
        .map(|&a| (a - mean) * (a - mean))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();
    let ci95 = 1.96 * std_dev / n.sqrt();
    Ok((mean, ci95))
}

pub fn accuracy_at_k(
    logits: &[f32],
    labels: &[u32],
    n_classes: usize,
    k: usize,
) -> MetaResult<f32> {
    if n_classes == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "n_classes must be > 0".into(),
        });
    }
    if k == 0 {
        return Err(MetaError::InvalidQuerySize { size: k });
    }
    let n = labels.len();
    if n == 0 {
        return Err(MetaError::EmptySupport);
    }
    if logits.len() != n * n_classes {
        return Err(MetaError::DimensionMismatch {
            expected: n * n_classes,
            got: logits.len(),
        });
    }

    let actual_k = k.min(n_classes);
    let mut correct = 0_usize;

    for (i, &lbl) in labels.iter().enumerate() {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        // Find top-k indices by sorting descending
        let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<usize> = indexed[..actual_k].iter().map(|&(idx, _)| idx).collect();
        if top_k.contains(&(lbl as usize)) {
            correct += 1;
        }
    }

    Ok(correct as f32 / n as f32)
}
