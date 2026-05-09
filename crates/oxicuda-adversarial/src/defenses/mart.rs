//! MART — Misclassification Aware adveRsarial Training.
//!
//! Reference: Wang, Zou, Yi, Bailey, Ma & Gu (2020),
//! *"Improving Adversarial Robustness Requires Revisiting Misclassified
//! Examples"*, ICLR.
//!
//! MART augments adversarial training with two ingredients:
//!
//! 1. A boosted-CE first term that pushes the runner-up class probability of
//!    the adversarial prediction down:
//!    ```text
//!    L1 = − log(1 − max_{j ≠ y} p_j(x_adv))
//!    ```
//! 2. A KL regularizer weighted per-sample by `(1 − p_y(x))` — emphasising
//!    samples whose **clean** predictions are uncertain or wrong:
//!    ```text
//!    L2 = (1 − p_y(x)) · KL(softmax(f(x_adv)) ‖ softmax(f(x)))
//!    ```
//!
//! The final batch loss is `mean_i (L1_i + λ · L2_i)`.
//!
//! Setting `λ = 0` keeps only the boosted-CE term; together with `β = 0` for
//! TRADES this is the simplest robust-only objective.
//!
//! All math here is in numerically stable form (log-sum-exp on logits,
//! `log1p` / clamping for the boosted CE term).
//!
//! # Conventions
//!
//! * Logits are laid out row-major: `clean_logits[i*k .. i*k + k]` is sample
//!   `i`'s logit vector.
//! * Labels are integer class indices in `[0, k)`.
//! * The loss is reduced by **mean over the batch**.

use crate::error::{AdvError, AdvResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for the MART surrogate loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MartConfig {
    /// Weight of the per-sample weighted KL regularizer. Must be finite and
    /// `>= 0`. Typical value: `5.0` (Wang et al. 2020).
    pub lambda: f32,
}

impl MartConfig {
    /// Build a new `MartConfig`.
    ///
    /// # Errors
    /// [`AdvError::InvalidLossWeight`] if `lambda` is non-finite or negative.
    pub fn new(lambda: f32) -> AdvResult<Self> {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(AdvError::InvalidLossWeight { weight: lambda });
        }
        Ok(Self { lambda })
    }
}

impl Default for MartConfig {
    fn default() -> Self {
        Self { lambda: 5.0 }
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

/// Largest non-target log-probability (i.e. log of `max_{j != y} p_j`).
fn max_nontarget_log_prob(log_probs: &[f32], y: usize) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for (j, &lp) in log_probs.iter().enumerate() {
        if j != y && lp > best {
            best = lp;
        }
    }
    best
}

// ─── MART loss ───────────────────────────────────────────────────────────────

/// Compute the MART loss on a batch of `[N x K]` clean and adversarial
/// logits.
///
/// Returns the mean loss across the batch.
///
/// # Parameters
/// * `clean_logits` — flat `N*K` clean logits.
/// * `adv_logits`   — flat `N*K` adversarial logits.
/// * `labels`       — class indices in `[0, K)`, length `N`.
/// * `n`, `k`       — batch size and number of classes.
/// * `cfg`          — MART hyper-parameters.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty batch.
/// * [`AdvError::DimensionMismatch`] — any of `clean_logits`, `adv_logits`,
///   `labels` has the wrong length.
/// * [`AdvError::InvalidLossWeight`] — invalid label index, or `k < 2` (the
///   "max over non-target" term is undefined for a single class).
/// * [`AdvError::NanEncountered`]    — non-finite logits / loss components.
pub fn mart_loss(
    clean_logits: &[f32],
    adv_logits: &[f32],
    labels: &[usize],
    n: usize,
    k: usize,
    cfg: &MartConfig,
) -> AdvResult<f32> {
    if n == 0 || k == 0 {
        return Err(AdvError::EmptyInput);
    }
    if k < 2 {
        return Err(AdvError::InvalidLossWeight { weight: k as f32 });
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
            location: "mart_loss:clean_logits",
        });
    }
    if adv_logits.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "mart_loss:adv_logits",
        });
    }

    // Numerical clamp to keep `log(1 - p_runner_up)` finite even when the
    // adversarial classifier is fully confident on the runner-up.
    const P_EPS: f32 = 1e-7;

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

        // ── L1: boosted-CE  −log(1 − max_{j≠y} p_j(x_adv))
        // Use log-prob and clamp probability into [P_EPS, 1 − P_EPS].
        let max_other_lp = max_nontarget_log_prob(&adv_log_p, y);
        let p_runner_up = max_other_lp.exp().clamp(P_EPS, 1.0 - P_EPS);
        // log(1 - p) is well-conditioned since p ∈ [eps, 1-eps].
        let l1 = -((1.0 - p_runner_up).ln());

        // ── L2: weighted KL  (1 − p_y(x_clean)) · KL(softmax(adv) ‖ softmax(clean))
        let p_y_clean = clean_log_p[y].exp().clamp(0.0, 1.0);
        let weight = (1.0 - p_y_clean).max(0.0);

        let mut kl = 0.0_f32;
        for j in 0..k {
            let p_adv = adv_log_p[j].exp();
            kl += p_adv * (adv_log_p[j] - clean_log_p[j]);
        }
        if kl < 0.0 && kl > -1e-5 {
            kl = 0.0;
        }

        let sample_loss = l1 + cfg.lambda * weight * kl;
        if !sample_loss.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "mart_loss:sample_loss",
            });
        }
        total += sample_loss;
    }
    Ok(total / (n as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L1 reference: −log(1 − max_{j ≠ y} p_j(x_adv)) averaged over the batch.
    fn l1_only_reference(adv_logits: &[f32], labels: &[usize], n: usize, k: usize) -> f32 {
        let mut sum = 0.0_f32;
        for i in 0..n {
            let row = &adv_logits[i * k..(i + 1) * k];
            let lp = log_softmax(row);
            let m = max_nontarget_log_prob(&lp, labels[i]);
            let p = m.exp().clamp(1e-7, 1.0 - 1e-7);
            sum += -((1.0 - p).ln());
        }
        sum / n as f32
    }

    #[test]
    fn config_rejects_negative_lambda() {
        assert!(MartConfig::new(-0.1).is_err());
        assert!(MartConfig::new(f32::NAN).is_err());
        assert!(MartConfig::new(f32::INFINITY).is_err());
    }

    #[test]
    fn config_default_is_five() {
        assert_eq!(MartConfig::default().lambda, 5.0);
        assert!(MartConfig::new(0.0).is_ok());
    }

    #[test]
    fn lambda_zero_reduces_to_l1_only() {
        let clean = vec![3.0_f32, 0.0, 0.0, 0.0, 5.0, 1.0];
        let adv = vec![0.0_f32, 4.0, 1.0, 2.0, 0.0, 0.0]; // Different.
        let labels = vec![0_usize, 1];
        let cfg = MartConfig::new(0.0).expect("cfg");
        let mart = mart_loss(&clean, &adv, &labels, 2, 3, &cfg).expect("loss");
        let l1 = l1_only_reference(&adv, &labels, 2, 3);
        assert!((mart - l1).abs() < 1e-5, "mart={mart} l1={l1}");
    }

    #[test]
    fn confident_correct_adv_yields_low_loss() {
        // adv == clean and clean is fully confident on the true label.
        // Then runner-up p ≈ 0 → L1 ≈ −log(1) = 0; KL = 0 → L2 = 0.
        let logits = vec![20.0_f32, 0.0, 0.0]; // clean is very confident on class 0.
        let labels = vec![0_usize];
        let cfg = MartConfig::new(5.0).expect("cfg");
        let l = mart_loss(&logits, &logits, &labels, 1, 3, &cfg).expect("loss");
        assert!(l < 1e-5, "loss should be near zero, got {l}");
    }

    #[test]
    fn misclassified_clean_amplifies_kl_term() {
        // When clean is wrong (low p_y_clean), the KL weight (1 - p_y) is high.
        // Compare to a sample where clean is confident-correct but adv differs.
        let clean_wrong = vec![0.0_f32, 5.0, 0.0]; // Clean predicts class 1, true=0.
        let clean_right = vec![5.0_f32, 0.0, 0.0]; // Clean confidently right.
        let adv_diff = vec![0.0_f32, 0.0, 5.0]; // Adv far from both.
        let labels = vec![0_usize];
        let cfg = MartConfig::new(5.0).expect("cfg");
        let l_wrong = mart_loss(&clean_wrong, &adv_diff, &labels, 1, 3, &cfg).expect("l");
        let l_right = mart_loss(&clean_right, &adv_diff, &labels, 1, 3, &cfg).expect("l");
        assert!(
            l_wrong > l_right,
            "wrong-clean loss should exceed right-clean loss: {l_wrong} vs {l_right}"
        );
    }

    #[test]
    fn dim_mismatch_clean_logits() {
        let clean = vec![1.0_f32; 5]; // Should be 6.
        let adv = vec![0.0_f32; 6];
        let labels = vec![0_usize, 1];
        let cfg = MartConfig::new(1.0).expect("cfg");
        let err = mart_loss(&clean, &adv, &labels, 2, 3, &cfg).unwrap_err();
        assert!(matches!(
            err,
            AdvError::DimensionMismatch {
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn dim_mismatch_adv_logits_and_labels() {
        let clean = vec![1.0_f32; 6];
        let adv = vec![0.0_f32; 5]; // Should be 6.
        let labels = vec![0_usize, 1];
        let cfg = MartConfig::new(1.0).expect("cfg");
        let err = mart_loss(&clean, &adv, &labels, 2, 3, &cfg).unwrap_err();
        assert!(matches!(
            err,
            AdvError::DimensionMismatch {
                expected: 6,
                got: 5
            }
        ));

        let adv2 = vec![0.0_f32; 6];
        let labels_bad = vec![0_usize]; // Should be length 2.
        let err2 = mart_loss(&clean, &adv2, &labels_bad, 2, 3, &cfg).unwrap_err();
        assert!(matches!(
            err2,
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
        let cfg = MartConfig::new(1.0).expect("cfg");
        assert!(matches!(
            mart_loss(&clean, &adv, &labels, 1, 3, &cfg).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn k_below_two_rejected() {
        let cfg = MartConfig::new(1.0).expect("cfg");
        assert!(matches!(
            mart_loss(&[1.0_f32], &[1.0_f32], &[0_usize], 1, 1, &cfg).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn nan_logits_rejected() {
        let clean = vec![1.0_f32, 0.0, f32::NAN];
        let adv = vec![0.0_f32; 3];
        let labels = vec![0_usize];
        let cfg = MartConfig::new(1.0).expect("cfg");
        assert!(matches!(
            mart_loss(&clean, &adv, &labels, 1, 3, &cfg).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn loss_is_non_negative() {
        let clean = vec![1.0_f32, 2.0, 0.0, -1.0, 0.5, 3.0];
        let adv = vec![0.0_f32, 0.5, 2.0, 4.0, -2.0, 1.0];
        let labels = vec![1_usize, 2];
        let cfg = MartConfig::new(5.0).expect("cfg");
        let l = mart_loss(&clean, &adv, &labels, 2, 3, &cfg).expect("loss");
        assert!(l >= 0.0);
    }
}
