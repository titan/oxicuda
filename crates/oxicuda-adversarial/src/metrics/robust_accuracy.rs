//! Robust accuracy and certified accuracy.
//!
//! * **Robust accuracy** — fraction of samples that the model classifies
//!   correctly under an attack.
//! * **Certified accuracy** — fraction of samples that are *both* correctly
//!   classified by the smoothed/base classifier *and* whose certified radius
//!   meets or exceeds a target radius.

use crate::error::{AdvError, AdvResult};

/// Robust accuracy on a batch of `(adv_pred, label)` pairs.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty inputs.
/// * [`AdvError::DimensionMismatch`] — length mismatch.
pub fn robust_accuracy(adv_pred: &[usize], labels: &[usize]) -> AdvResult<f32> {
    if adv_pred.is_empty() || labels.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if adv_pred.len() != labels.len() {
        return Err(AdvError::DimensionMismatch {
            expected: labels.len(),
            got: adv_pred.len(),
        });
    }
    let n = adv_pred.len();
    let correct = adv_pred
        .iter()
        .zip(labels.iter())
        .filter(|(p, y)| p == y)
        .count();
    Ok(correct as f32 / n as f32)
}

/// Certified accuracy at a target radius `r_target`.
///
/// `pred[i]` is the (smoothed) classifier's prediction, `labels[i]` is the
/// ground truth, and `radii[i]` is the certified L2 radius (or any other
/// pre-chosen Lp radius). A sample is "certified" iff the prediction is
/// correct and `radii[i] >= r_target`.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty inputs.
/// * [`AdvError::DimensionMismatch`] — any length mismatch.
/// * [`AdvError::InvalidEpsilon`]    — `r_target` is negative or non-finite.
/// * [`AdvError::NanEncountered`]    — non-finite radius.
pub fn certified_accuracy(
    pred: &[usize],
    labels: &[usize],
    radii: &[f32],
    r_target: f32,
) -> AdvResult<f32> {
    if pred.is_empty() || labels.is_empty() || radii.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if pred.len() != labels.len() {
        return Err(AdvError::DimensionMismatch {
            expected: labels.len(),
            got: pred.len(),
        });
    }
    if radii.len() != pred.len() {
        return Err(AdvError::DimensionMismatch {
            expected: pred.len(),
            got: radii.len(),
        });
    }
    if !(r_target.is_finite() && r_target >= 0.0) {
        return Err(AdvError::InvalidEpsilon { eps: r_target });
    }
    if radii.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "certified_accuracy:radii",
        });
    }
    let n = pred.len();
    let cert = pred
        .iter()
        .zip(labels.iter())
        .zip(radii.iter())
        .filter(|((p, y), r)| p == y && **r >= r_target)
        .count();
    Ok(cert as f32 / n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_acc_perfect() {
        let p = vec![0_usize, 1, 2];
        let y = p.clone();
        assert!((robust_accuracy(&p, &y).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn robust_acc_zero() {
        let p = vec![0_usize, 1, 2];
        let y = vec![1_usize, 2, 0];
        assert!((robust_accuracy(&p, &y).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn robust_acc_dim_mismatch() {
        assert!(matches!(
            robust_accuracy(&[0_usize, 1], &[0_usize]).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn robust_acc_empty() {
        assert!(matches!(
            robust_accuracy(&[], &[]).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn certified_acc_basic() {
        // 3 samples; predictions all correct; radii [0.1, 0.5, 1.0].
        // r_target = 0.4 ⇒ samples 1 and 2 qualify ⇒ 2/3.
        let p = vec![0_usize, 1, 2];
        let y = p.clone();
        let r = vec![0.1_f32, 0.5, 1.0];
        let acc = certified_accuracy(&p, &y, &r, 0.4).unwrap();
        assert!((acc - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn certified_acc_wrong_pred_excluded() {
        // Sample 0 wrong → never certified regardless of radius.
        let p = vec![1_usize, 1];
        let y = vec![0_usize, 1];
        let r = vec![10.0_f32, 0.5];
        let acc = certified_accuracy(&p, &y, &r, 0.1).unwrap();
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    fn certified_acc_invalid_target() {
        let p = vec![0_usize];
        let y = vec![0_usize];
        let r = vec![0.5_f32];
        assert!(matches!(
            certified_accuracy(&p, &y, &r, -0.1).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
        assert!(matches!(
            certified_accuracy(&p, &y, &r, f32::NAN).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    #[test]
    fn certified_acc_nan_radius() {
        let p = vec![0_usize];
        let y = vec![0_usize];
        let r = vec![f32::NAN];
        assert!(matches!(
            certified_accuracy(&p, &y, &r, 0.1).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }
}
