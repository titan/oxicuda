//! Attack Success Rate (ASR).
//!
//! Fraction of inputs for which an attack flips the model's prediction away
//! from the true label.

use crate::error::{AdvError, AdvResult};

/// Compute the attack success rate (fraction of mis-predictions on `adv_pred`).
///
/// `adv_pred[i]` is the model's predicted class on the adversarial example
/// for sample `i`; `labels[i]` is the corresponding ground-truth label.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty inputs.
/// * [`AdvError::DimensionMismatch`] — `adv_pred.len() != labels.len()`.
pub fn attack_success_rate(adv_pred: &[usize], labels: &[usize]) -> AdvResult<f32> {
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
    let mis = adv_pred
        .iter()
        .zip(labels.iter())
        .filter(|(p, y)| p != y)
        .count();
    Ok(mis as f32 / n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_attack_is_one() {
        let pred = vec![1_usize, 2, 0];
        let truth = vec![0_usize, 0, 1];
        assert!((attack_success_rate(&pred, &truth).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_success_is_zero() {
        let pred = vec![0_usize, 1, 2];
        let truth = pred.clone();
        assert!((attack_success_rate(&pred, &truth).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn half_correct() {
        let pred = vec![0_usize, 1, 0, 1];
        let truth = vec![0_usize, 1, 1, 0];
        assert!((attack_success_rate(&pred, &truth).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dim_mismatch() {
        let pred = vec![0_usize; 3];
        let truth = vec![0_usize; 4];
        assert!(matches!(
            attack_success_rate(&pred, &truth).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn empty_rejected() {
        assert!(matches!(
            attack_success_rate(&[], &[]).unwrap_err(),
            AdvError::EmptyInput
        ));
    }
}
