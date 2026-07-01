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

#[cfg(test)]
mod tests {
    use super::*;

    // ── episode_accuracy ─────────────────────────────────────────────────────

    #[test]
    fn episode_accuracy_exact_fraction() {
        // preds=[1,0,2,1] vs labels=[1,0,2,0]: indices 0,1,2 match, index 3 does not → 3/4
        let preds = [1_u32, 0, 2, 1];
        let labels = [1_u32, 0, 2, 0];
        let acc = episode_accuracy(&preds, &labels).expect("episode_accuracy should succeed");
        assert!((acc - 0.75_f32).abs() < 1e-6, "expected 0.75, got {acc}");
    }

    #[test]
    fn episode_accuracy_all_correct() {
        let v: Vec<u32> = vec![0, 1, 2, 3, 4];
        let acc = episode_accuracy(&v, &v).expect("episode_accuracy should succeed");
        assert!((acc - 1.0_f32).abs() < 1e-6, "expected 1.0, got {acc}");
    }

    #[test]
    fn episode_accuracy_all_wrong() {
        let preds = [1_u32, 2, 3];
        let labels = [0_u32, 0, 0];
        let acc = episode_accuracy(&preds, &labels).expect("episode_accuracy should succeed");
        assert!((acc - 0.0_f32).abs() < 1e-6, "expected 0.0, got {acc}");
    }

    #[test]
    fn episode_accuracy_length_mismatch_errors() {
        assert!(matches!(
            episode_accuracy(&[0_u32, 1], &[0_u32]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn episode_accuracy_empty_errors() {
        assert!(matches!(
            episode_accuracy(&[], &[]),
            Err(MetaError::EmptySupport)
        ));
    }

    // ── mean_and_ci95 ────────────────────────────────────────────────────────

    #[test]
    fn mean_and_ci95_exact_mean_and_half_width() {
        // accuracies = [0.0, 1.0]
        // mean           = 0.5
        // pop variance   = ((0.0−0.5)² + (1.0−0.5)²) / 2 = 0.25
        // std_dev        = 0.5
        // ci95           = 1.96 × 0.5 / √2
        let accs = [0.0_f32, 1.0];
        let (mean, hw) = mean_and_ci95(&accs).expect("mean_and_ci95 should succeed");
        assert!(
            (mean - 0.5_f32).abs() < 1e-6,
            "expected mean=0.5, got {mean}"
        );
        let expected_hw = 1.96_f32 * 0.5_f32 / 2.0_f32.sqrt();
        assert!(
            (hw - expected_hw).abs() < 1e-5,
            "expected ci95 half-width={expected_hw}, got {hw}"
        );
    }

    #[test]
    fn mean_and_ci95_constant_zero_half_width() {
        // All elements equal → population std = 0 → half-width = 0
        let accs = [0.5_f32, 0.5, 0.5, 0.5];
        let (mean, hw) = mean_and_ci95(&accs).expect("mean_and_ci95 should succeed");
        assert!(
            (mean - 0.5_f32).abs() < 1e-6,
            "expected mean=0.5, got {mean}"
        );
        assert!(hw.abs() < 1e-6, "expected zero half-width, got {hw}");
    }

    #[test]
    fn mean_and_ci95_single_element() {
        // Single element: variance=0, std=0, ci95=0
        let accs = [0.75_f32];
        let (mean, hw) = mean_and_ci95(&accs).expect("mean_and_ci95 should succeed");
        assert!(
            (mean - 0.75_f32).abs() < 1e-6,
            "expected mean=0.75, got {mean}"
        );
        assert!(
            hw.abs() < 1e-6,
            "expected zero half-width for single element, got {hw}"
        );
    }

    #[test]
    fn mean_and_ci95_empty_errors() {
        assert!(matches!(mean_and_ci95(&[]), Err(MetaError::EmptySupport)));
    }

    // ── accuracy_at_k ────────────────────────────────────────────────────────

    #[test]
    fn accuracy_at_k_label_in_top1_all_correct() {
        // 3 samples, 3 classes, k=1
        // sample 0 logits [3,1,2] → top-1 = class 0, label=0 ✓
        // sample 1 logits [1,3,2] → top-1 = class 1, label=1 ✓
        // sample 2 logits [1,2,3] → top-1 = class 2, label=2 ✓  →  1.0
        let logits = [3.0_f32, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let labels = [0_u32, 1, 2];
        let acc = accuracy_at_k(&logits, &labels, 3, 1).expect("accuracy_at_k should succeed");
        assert!((acc - 1.0_f32).abs() < 1e-6, "expected 1.0, got {acc}");
    }

    #[test]
    fn accuracy_at_k_label_not_in_top1() {
        // 2 samples, 2 classes, k=1
        // sample 0 logits [0.1, 0.9] → top-1 = class 1, label=0 ✗
        // sample 1 logits [0.9, 0.1] → top-1 = class 0, label=1 ✗  →  0.0
        let logits = [0.1_f32, 0.9, 0.9, 0.1];
        let labels = [0_u32, 1];
        let acc = accuracy_at_k(&logits, &labels, 2, 1).expect("accuracy_at_k should succeed");
        assert!((acc - 0.0_f32).abs() < 1e-6, "expected 0.0, got {acc}");
    }

    #[test]
    fn accuracy_at_k_top2_partial_hit() {
        // 2 samples, 3 classes, k=2
        // sample 0 logits [3,2,1] → top-2 = {0,1}, label=1 ✓
        // sample 1 logits [1,2,3] → top-2 = {2,1}, label=0 ✗  →  0.5
        let logits = [3.0_f32, 2.0, 1.0, 1.0, 2.0, 3.0];
        let labels = [1_u32, 0];
        let acc = accuracy_at_k(&logits, &labels, 3, 2).expect("accuracy_at_k should succeed");
        assert!((acc - 0.5_f32).abs() < 1e-6, "expected 0.5, got {acc}");
    }

    #[test]
    fn accuracy_at_k_k_larger_than_n_classes_clamps() {
        // k=100 with n_classes=2 → actual_k clamped to 2 → all classes covered → 1.0
        let logits = [0.9_f32, 0.1, 0.1, 0.9];
        let labels = [0_u32, 1];
        let acc = accuracy_at_k(&logits, &labels, 2, 100).expect("accuracy_at_k should succeed");
        assert!(
            (acc - 1.0_f32).abs() < 1e-6,
            "expected 1.0 (all classes in top-k), got {acc}"
        );
    }

    #[test]
    fn accuracy_at_k_zero_k_errors() {
        assert!(matches!(
            accuracy_at_k(&[0.5_f32, 0.5], &[0_u32], 2, 0),
            Err(MetaError::InvalidQuerySize { .. })
        ));
    }

    #[test]
    fn accuracy_at_k_zero_n_classes_errors() {
        assert!(matches!(
            accuracy_at_k(&[], &[], 0, 1),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }
}
