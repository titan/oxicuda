//! SLiC-HF — Sequence Likelihood Calibration with Human Feedback (Zhao et al. 2023).
//!
//! Reference: Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., & Liu, P. J. (2023).
//! *SLiC-HF: Sequence Likelihood Calibration with Human Feedback*.
//! <https://arxiv.org/abs/2305.10425>
//!
//! SLiC-HF aligns a policy to pairwise human preferences with a **max-margin rank-calibration**
//! loss plus a **cross-entropy regularisation** term that keeps the policy close to a fixed SFT
//! reference, all *without* an explicit reward model or RL loop.
//!
//! For a preferred response `y⁺` and a dispreferred `y⁻` (given prompt `x`), let `s⁺` and `s⁻`
//! be the policy's sequence log-likelihoods. The **calibration (ranking) loss** is a hinge with
//! margin `δ`:
//!
//! ```text
//!   L_cal = max( 0,  δ − ( s⁺ − s⁻ ) )
//! ```
//!
//! i.e. the preferred sequence must out-score the dispreferred one by at least `δ`. SLiC-HF adds
//! a **regularisation** term — the negative log-likelihood of a reference (SFT) target `y_ref`,
//! scaled by `λ` — to prevent the calibration objective from degrading fluency:
//!
//! ```text
//!   L_reg = − s_ref                     (s_ref = log-likelihood of the reference target)
//!   L     = L_cal + λ · L_reg
//! ```
//!
//! This module operates directly on sequence-level log-likelihoods (sums of token log-probs),
//! matching the SLiC-HF formulation.

use crate::error::{RlhfError, RlhfResult};

/// Configuration for SLiC-HF.
#[derive(Debug, Clone)]
pub struct SlicConfig {
    /// Margin `δ` of the rank-calibration hinge (`≥ 0`, finite).
    pub delta: f32,
    /// Weight `λ` of the cross-entropy regularisation toward the SFT reference (`≥ 0`, finite).
    pub reg_weight: f32,
}

impl Default for SlicConfig {
    fn default() -> Self {
        // Defaults from the paper's calibration setup.
        Self {
            delta: 1.0,
            reg_weight: 0.1,
        }
    }
}

impl SlicConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.delta.is_finite() || self.delta < 0.0 {
            return Err(RlhfError::InvalidMargin { margin: self.delta });
        }
        if !self.reg_weight.is_finite() || self.reg_weight < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.reg_weight,
            });
        }
        Ok(())
    }
}

/// A SLiC-HF training pair: sequence log-likelihoods of the preferred / dispreferred responses
/// and of the (optional) SFT reference target.
#[derive(Debug, Clone)]
pub struct SlicPair {
    /// `s⁺` — policy sequence log-likelihood of the preferred response.
    pub pos_logp: f32,
    /// `s⁻` — policy sequence log-likelihood of the dispreferred response.
    pub neg_logp: f32,
    /// `s_ref` — policy sequence log-likelihood of the SFT reference target.
    ///
    /// Used only by the regularisation term; set equal to `pos_logp` to regularise toward the
    /// preferred response (a common SLiC-HF choice).
    pub ref_logp: f32,
}

/// Rank-calibration hinge loss `max(0, δ − (s⁺ − s⁻))` for a single pair.
///
/// # Errors
/// [`RlhfError::NanEncountered`] if either log-prob is NaN.
pub fn calibration_loss(pos_logp: f32, neg_logp: f32, delta: f32) -> RlhfResult<f32> {
    if pos_logp.is_nan() || neg_logp.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    let margin = pos_logp - neg_logp;
    Ok((delta - margin).max(0.0))
}

/// Cross-entropy regularisation loss `− s_ref` for a single pair.
///
/// # Errors
/// [`RlhfError::NanEncountered`] if `ref_logp` is NaN.
pub fn regularization_loss(ref_logp: f32) -> RlhfResult<f32> {
    if ref_logp.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(-ref_logp)
}

/// Full SLiC-HF loss for a single pair: `L_cal + λ · L_reg`.
///
/// # Errors
/// - [`RlhfError::InvalidMargin`] / [`RlhfError::InvalidLambda`] for an invalid config.
/// - [`RlhfError::NanEncountered`] if any log-prob is NaN or the result is NaN.
pub fn slic_loss(pair: &SlicPair, cfg: &SlicConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    let cal = calibration_loss(pair.pos_logp, pair.neg_logp, cfg.delta)?;
    let reg = regularization_loss(pair.ref_logp)?;
    let loss = cal + cfg.reg_weight * reg;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean SLiC-HF loss over a batch of pairs.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if the batch is empty.
/// - Propagates per-pair errors from [`slic_loss`].
pub fn slic_loss_batch(pairs: &[SlicPair], cfg: &SlicConfig) -> RlhfResult<f32> {
    if pairs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let mut total = 0.0_f32;
    for p in pairs {
        total += slic_loss(p, cfg)?;
    }
    Ok(total / pairs.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_zero_when_margin_exceeds_delta() {
        // s⁺ − s⁻ = 3 ≥ δ=1 → hinge is zero.
        let l = calibration_loss(-1.0, -4.0, 1.0).expect("calibration_loss should succeed");
        assert!(l.abs() < 1e-6, "satisfied margin → 0, got {l}");
    }

    #[test]
    fn calibration_positive_when_margin_below_delta() {
        // s⁺ − s⁻ = 0.5 < δ=2 → hinge = 2 − 0.5 = 1.5.
        let l = calibration_loss(-1.0, -1.5, 2.0).expect("calibration_loss should succeed");
        assert!((l - 1.5).abs() < 1e-6, "expected 1.5, got {l}");
    }

    #[test]
    fn calibration_at_exactly_delta_is_zero() {
        // s⁺ − s⁻ = δ exactly → hinge = 0.
        let l = calibration_loss(0.0, -1.0, 1.0).expect("calibration_loss should succeed");
        assert!(l.abs() < 1e-6, "at margin → 0, got {l}");
    }

    #[test]
    fn calibration_max_when_pos_below_neg() {
        // s⁺ < s⁻ → margin negative → hinge = δ − margin > δ.
        let l = calibration_loss(-3.0, -1.0, 1.0).expect("calibration_loss should succeed");
        // δ − (−3 − −1) = 1 − (−2) = 3
        assert!((l - 3.0).abs() < 1e-6, "expected 3.0, got {l}");
    }

    #[test]
    fn calibration_nan_errors() {
        assert!(matches!(
            calibration_loss(f32::NAN, -1.0, 1.0),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn regularization_is_negative_logp() {
        let l = regularization_loss(-2.5).expect("regularization_loss should succeed");
        assert!((l - 2.5).abs() < 1e-6, "reg = -s_ref = 2.5, got {l}");
    }

    #[test]
    fn regularization_nan_errors() {
        assert!(matches!(
            regularization_loss(f32::NAN),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn slic_loss_combines_calibration_and_reg() {
        let pair = SlicPair {
            pos_logp: -1.0,
            neg_logp: -1.5,
            ref_logp: -2.0,
        };
        let cfg = SlicConfig {
            delta: 2.0,
            reg_weight: 0.1,
        };
        let loss = slic_loss(&pair, &cfg).expect("slic_loss should succeed");
        // cal = 2 − 0.5 = 1.5 ; reg = 2.0 ; total = 1.5 + 0.1·2.0 = 1.7
        assert!((loss - 1.7).abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn slic_loss_reg_weight_zero_is_pure_calibration() {
        let pair = SlicPair {
            pos_logp: -1.0,
            neg_logp: -1.5,
            ref_logp: -2.0,
        };
        let cfg = SlicConfig {
            delta: 2.0,
            reg_weight: 0.0,
        };
        let loss = slic_loss(&pair, &cfg).expect("slic_loss should succeed");
        let cal = calibration_loss(pair.pos_logp, pair.neg_logp, cfg.delta)
            .expect("calibration_loss should succeed");
        assert!(
            (loss - cal).abs() < 1e-6,
            "reg_weight=0 → loss == calibration"
        );
    }

    #[test]
    fn slic_loss_lower_for_well_separated_pairs() {
        let cfg = SlicConfig {
            delta: 1.0,
            reg_weight: 0.0,
        };
        // Well-separated (s⁺ ≫ s⁻).
        let good = SlicPair {
            pos_logp: -0.5,
            neg_logp: -5.0,
            ref_logp: -0.5,
        };
        // Poorly separated.
        let bad = SlicPair {
            pos_logp: -2.0,
            neg_logp: -2.1,
            ref_logp: -2.0,
        };
        let l_good = slic_loss(&good, &cfg).expect("slic_loss should succeed");
        let l_bad = slic_loss(&bad, &cfg).expect("slic_loss should succeed");
        assert!(
            l_good < l_bad,
            "well-separated should have lower loss: good={l_good}, bad={l_bad}"
        );
    }

    #[test]
    fn slic_loss_invalid_delta_errors() {
        let pair = SlicPair {
            pos_logp: -1.0,
            neg_logp: -2.0,
            ref_logp: -1.0,
        };
        let cfg = SlicConfig {
            delta: -1.0,
            reg_weight: 0.1,
        };
        assert!(matches!(
            slic_loss(&pair, &cfg),
            Err(RlhfError::InvalidMargin { .. })
        ));
    }

    #[test]
    fn slic_loss_invalid_reg_weight_errors() {
        let pair = SlicPair {
            pos_logp: -1.0,
            neg_logp: -2.0,
            ref_logp: -1.0,
        };
        let cfg = SlicConfig {
            delta: 1.0,
            reg_weight: -0.1,
        };
        assert!(matches!(
            slic_loss(&pair, &cfg),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    #[test]
    fn slic_loss_nan_errors() {
        let pair = SlicPair {
            pos_logp: f32::NAN,
            neg_logp: -2.0,
            ref_logp: -1.0,
        };
        let cfg = SlicConfig::default();
        assert!(matches!(
            slic_loss(&pair, &cfg),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn slic_loss_batch_is_mean() {
        let p1 = SlicPair {
            pos_logp: -1.0,
            neg_logp: -1.5,
            ref_logp: -2.0,
        };
        let p2 = SlicPair {
            pos_logp: -0.5,
            neg_logp: -5.0,
            ref_logp: -0.5,
        };
        let cfg = SlicConfig::default();
        let l1 = slic_loss(&p1, &cfg).expect("slic_loss should succeed");
        let l2 = slic_loss(&p2, &cfg).expect("slic_loss should succeed");
        let mean = slic_loss_batch(&[p1, p2], &cfg).expect("slic_loss_batch should succeed");
        assert!(
            (mean - (l1 + l2) / 2.0).abs() < 1e-5,
            "batch mean mismatch: {mean}"
        );
    }

    #[test]
    fn slic_loss_batch_empty_errors() {
        let cfg = SlicConfig::default();
        assert!(matches!(
            slic_loss_batch(&[], &cfg),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn default_config_values() {
        let cfg = SlicConfig::default();
        assert!((cfg.delta - 1.0).abs() < 1e-6, "default delta=1.0");
        assert!(
            (cfg.reg_weight - 0.1).abs() < 1e-6,
            "default reg_weight=0.1"
        );
    }
}
