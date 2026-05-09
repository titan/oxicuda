//! TRADES — Theoretically principled trade-off between robustness and accuracy.
//!
//! Reference: Zhang, Yu, Jiao, Xing, El Ghaoui & Jordan (2019),
//! *"Theoretically Principled Trade-off between Robustness and Accuracy"*,
//! ICML.
//!
//! The TRADES surrogate loss for an `(x, x_adv, y)` triplet on a classifier
//! `f` producing logits is
//!
//! ```text
//! L_TRADES = CE(f(x), y) + β · KL(softmax(f(x)) ‖ softmax(f(x_adv)))
//! ```
//!
//! The first term keeps clean accuracy high; the KL term pulls the
//! distribution under the adversarial input toward the distribution under the
//! clean input, encouraging local smoothness of the predictive distribution.
//! Setting `β = 0` recovers plain cross-entropy on the clean batch.
//!
//! All math here is implemented in numerically stable form (log-sum-exp on the
//! logit vectors, exp(log-prob) for the softmax probabilities) so it never
//! over- or underflows for practical f32 logit ranges.
//!
//! # Conventions
//!
//! * Logits are laid out row-major: `clean_logits[i*k .. i*k + k]` is the
//!   logit vector for sample `i`.
//! * Labels are integer class indices in `[0, k)`.
//! * The loss is reduced by **mean over the batch**.
//!
//! # Errors
//!
//! All entry points return `AdvError` on shape mismatches, invalid
//! hyper-parameters, invalid labels, or non-finite intermediate values.

use crate::error::{AdvError, AdvResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for the TRADES surrogate loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradesConfig {
    /// Weight of the KL regularizer. Must be finite and `>= 0`.
    /// Typical value: `6.0` (Zhang et al. 2019).
    pub beta: f32,
}

impl TradesConfig {
    /// Build a new `TradesConfig`.
    ///
    /// # Errors
    /// [`AdvError::InvalidLossWeight`] if `beta` is non-finite or negative.
    pub fn new(beta: f32) -> AdvResult<Self> {
        if !(beta.is_finite() && beta >= 0.0) {
            return Err(AdvError::InvalidLossWeight { weight: beta });
        }
        Ok(Self { beta })
    }
}

impl Default for TradesConfig {
    fn default() -> Self {
        Self { beta: 6.0 }
    }
}

// ─── Numerical helpers ───────────────────────────────────────────────────────

/// Stable log-sum-exp over a single logit vector.
fn log_sum_exp(logits: &[f32]) -> f32 {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !m.is_finite() {
        return m;
    }
    let s: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    m + s.ln()
}

/// Stable per-class log-softmax: returns `Vec<f32>` of length `logits.len()`.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let lse = log_sum_exp(logits);
    logits.iter().map(|&v| v - lse).collect()
}

// ─── TRADES loss ─────────────────────────────────────────────────────────────

/// Compute the TRADES loss on a batch of `[N x K]` clean and adversarial
/// logits.
///
/// Returns the mean loss across the batch.
///
/// # Parameters
/// * `clean_logits` — flat `N*K` clean logits.
/// * `adv_logits`   — flat `N*K` adversarial logits.
/// * `labels`       — class indices in `[0, K)`, length `N`.
/// * `n`, `k`       — batch size and number of classes.
/// * `cfg`          — TRADES hyper-parameters.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty batch.
/// * [`AdvError::DimensionMismatch`] — any of `clean_logits`, `adv_logits`,
///   `labels` has the wrong length.
/// * [`AdvError::InvalidLossWeight`] — invalid label index.
/// * [`AdvError::NanEncountered`]    — non-finite logits / loss components.
pub fn trades_loss(
    clean_logits: &[f32],
    adv_logits: &[f32],
    labels: &[usize],
    n: usize,
    k: usize,
    cfg: &TradesConfig,
) -> AdvResult<f32> {
    if n == 0 || k == 0 {
        return Err(AdvError::EmptyInput);
    }
    let expected = n * k;
    if clean_logits.len() != expected {
        return Err(AdvError::DimensionMismatch {
            expected,
            got: clean_logits.len(),
        });
    }
    if adv_logits.len() != expected {
        return Err(AdvError::DimensionMismatch {
            expected,
            got: adv_logits.len(),
        });
    }
    if labels.len() != n {
        return Err(AdvError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }
    if clean_logits.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "trades_loss:clean_logits",
        });
    }
    if adv_logits.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "trades_loss:adv_logits",
        });
    }

    let mut total = 0.0_f32;
    for i in 0..n {
        let y = labels[i];
        if y >= k {
            return Err(AdvError::InvalidLossWeight { weight: y as f32 });
        }
        let clean_row = &clean_logits[i * k..(i + 1) * k];
        let adv_row = &adv_logits[i * k..(i + 1) * k];

        let clean_log_p = log_softmax(clean_row);
        let adv_log_p = log_softmax(adv_row);

        // Cross-entropy on clean logits.
        let ce = -clean_log_p[y];

        // KL(softmax(clean) ‖ softmax(adv))
        //   = Σ_j p_clean_j · (log p_clean_j − log p_adv_j)
        let mut kl = 0.0_f32;
        for j in 0..k {
            let p_clean = clean_log_p[j].exp();
            kl += p_clean * (clean_log_p[j] - adv_log_p[j]);
        }
        // Numerical guard: KL is non-negative analytically; floor minor
        // round-off slack so the test for clean==adv is exact at zero.
        if kl < 0.0 && kl > -1e-5 {
            kl = 0.0;
        }

        let sample_loss = ce + cfg.beta * kl;
        if !sample_loss.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "trades_loss:sample_loss",
            });
        }
        total += sample_loss;
    }
    Ok(total / (n as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain cross-entropy reference for comparison.
    fn ce_reference(logits: &[f32], labels: &[usize], n: usize, k: usize) -> f32 {
        let mut sum = 0.0;
        for i in 0..n {
            let row = &logits[i * k..(i + 1) * k];
            let lp = log_softmax(row);
            sum += -lp[labels[i]];
        }
        sum / n as f32
    }

    #[test]
    fn config_rejects_negative_beta() {
        assert!(TradesConfig::new(-0.1).is_err());
        assert!(TradesConfig::new(f32::NAN).is_err());
        assert!(TradesConfig::new(f32::INFINITY).is_err());
    }

    #[test]
    fn config_accepts_zero_and_positive_beta() {
        assert!(TradesConfig::new(0.0).is_ok());
        assert!(TradesConfig::new(6.0).is_ok());
        assert_eq!(TradesConfig::default().beta, 6.0);
    }

    #[test]
    fn clean_equals_adv_reduces_to_ce() {
        // When clean == adv, KL = 0 → TRADES = CE regardless of β.
        let logits = vec![2.0_f32, 1.0, 0.5, -1.0, 0.0, 3.0];
        let labels = vec![0_usize, 2];
        let cfg = TradesConfig::new(6.0).expect("cfg");
        let trades = trades_loss(&logits, &logits, &labels, 2, 3, &cfg).expect("loss");
        let ce = ce_reference(&logits, &labels, 2, 3);
        assert!((trades - ce).abs() < 1e-5, "trades={trades} ce={ce}");
    }

    #[test]
    fn beta_zero_reduces_to_clean_ce() {
        let clean = vec![1.0_f32, 0.5, -1.0, 2.0, 0.0, -0.5];
        let adv = vec![0.0_f32, 5.0, 1.0, -3.0, 4.0, 2.0]; // Very different.
        let labels = vec![0_usize, 1];
        let cfg = TradesConfig::new(0.0).expect("cfg");
        let trades = trades_loss(&clean, &adv, &labels, 2, 3, &cfg).expect("loss");
        let ce = ce_reference(&clean, &labels, 2, 3);
        assert!((trades - ce).abs() < 1e-5);
    }

    #[test]
    fn larger_beta_increases_loss_when_kl_positive() {
        let clean = vec![3.0_f32, 0.0, 0.0];
        let adv = vec![0.0_f32, 3.0, 0.0]; // Different distribution.
        let labels = vec![0_usize];
        let cfg_low = TradesConfig::new(1.0).expect("cfg");
        let cfg_high = TradesConfig::new(10.0).expect("cfg");
        let l_low = trades_loss(&clean, &adv, &labels, 1, 3, &cfg_low).expect("l");
        let l_high = trades_loss(&clean, &adv, &labels, 1, 3, &cfg_high).expect("l");
        assert!(l_high > l_low + 1e-3);
    }

    #[test]
    fn dim_mismatch_clean_logits() {
        let clean = vec![1.0_f32; 5]; // Should be 6.
        let adv = vec![0.0_f32; 6];
        let labels = vec![0_usize, 1];
        let cfg = TradesConfig::new(1.0).expect("cfg");
        let err = trades_loss(&clean, &adv, &labels, 2, 3, &cfg).unwrap_err();
        assert!(matches!(
            err,
            AdvError::DimensionMismatch {
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn dim_mismatch_labels() {
        let clean = vec![1.0_f32; 6];
        let adv = vec![0.0_f32; 6];
        let labels = vec![0_usize]; // Should be length 2.
        let cfg = TradesConfig::new(1.0).expect("cfg");
        let err = trades_loss(&clean, &adv, &labels, 2, 3, &cfg).unwrap_err();
        assert!(matches!(
            err,
            AdvError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn invalid_label_errors() {
        let clean = vec![1.0_f32; 3];
        let adv = vec![0.0_f32; 3];
        let labels = vec![5_usize]; // 5 >= 3 = k.
        let cfg = TradesConfig::new(1.0).expect("cfg");
        let err = trades_loss(&clean, &adv, &labels, 1, 3, &cfg).unwrap_err();
        assert!(matches!(err, AdvError::InvalidLossWeight { .. }));
    }

    #[test]
    fn nan_logits_rejected() {
        let clean = vec![f32::NAN, 0.0, 0.0];
        let adv = vec![0.0_f32; 3];
        let labels = vec![0_usize];
        let cfg = TradesConfig::new(1.0).expect("cfg");
        let err = trades_loss(&clean, &adv, &labels, 1, 3, &cfg).unwrap_err();
        assert!(matches!(err, AdvError::NanEncountered { .. }));
    }

    #[test]
    fn empty_input_rejected() {
        let cfg = TradesConfig::new(1.0).expect("cfg");
        assert!(matches!(
            trades_loss(&[], &[], &[], 0, 3, &cfg).unwrap_err(),
            AdvError::EmptyInput
        ));
        assert!(matches!(
            trades_loss(&[], &[], &[], 1, 0, &cfg).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn loss_is_non_negative() {
        // CE >= 0 and KL >= 0, so TRADES loss is always non-negative.
        let clean = vec![1.0_f32, 2.0, 0.0, -1.0, 0.5, 3.0];
        let adv = vec![0.0_f32, 0.5, 2.0, 4.0, -2.0, 1.0];
        let labels = vec![1_usize, 2];
        let cfg = TradesConfig::new(6.0).expect("cfg");
        let l = trades_loss(&clean, &adv, &labels, 2, 3, &cfg).expect("loss");
        assert!(l >= 0.0);
    }
}
