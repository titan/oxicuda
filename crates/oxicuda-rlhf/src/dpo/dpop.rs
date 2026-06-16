//! DPOP — DPO-Positive (Pal et al. 2024).
//!
//! Reference: Pal, A., Karkhanis, D., Dooley, S., Roberts, M., Naidu, S., & White, C. (2024).
//! *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*.
//! <https://arxiv.org/abs/2402.13228>
//!
//! Pal et al. identify a failure mode of standard DPO: when the edit distance between the
//! chosen `y_w` and rejected `y_l` responses is small, minimising the DPO loss can *reduce* the
//! model's log-probability of the **chosen** response (it only needs to reduce `log π(y_l)`
//! faster than `log π(y_w)`). DPO-Positive adds a one-sided penalty that explicitly discourages
//! the chosen log-prob from falling below its reference value:
//!
//! ```text
//!   h        = β · ( (logπθ(y_w) − logπref(y_w)) − (logπθ(y_l) − logπref(y_l)) )
//!   penalty  = λ · max( 0,  logπref(y_w) − logπθ(y_w) )
//!   loss     = − log σ( h − penalty )
//! ```
//!
//! The `penalty` is positive whenever the policy assigns *less* probability to the chosen
//! response than the reference does, shifting the logit down and increasing the loss — so the
//! optimiser is pushed to keep `logπθ(y_w) ≥ logπref(y_w)`. With `λ = 0` the loss reduces
//! exactly to standard DPO.

use crate::dpo::step_dpo::log_sigmoid;
use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::{PairBatch, PreferencePair};

/// Configuration for DPO-Positive.
#[derive(Debug, Clone)]
pub struct DpopConfig {
    /// DPO temperature `β` (> 0, finite).
    pub beta: f32,
    /// Weight `λ ≥ 0` of the positive (chosen-log-prob) penalty.
    ///
    /// `λ = 0` recovers standard DPO; the Smaug paper uses values such as `λ = 50`.
    pub lambda: f32,
}

impl Default for DpopConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            lambda: 50.0,
        }
    }
}

impl DpopConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.lambda,
            });
        }
        Ok(())
    }
}

/// The DPO log-ratio `h = β · ((logπθ_w − logπref_w) − (logπθ_l − logπref_l))` for one pair.
#[must_use]
#[inline]
pub fn dpop_log_ratio(pair: &PreferencePair, beta: f32) -> f32 {
    let log_ratio_chosen = pair.chosen_logp - pair.ref_chosen_logp;
    let log_ratio_rejected = pair.rejected_logp - pair.ref_rejected_logp;
    beta * (log_ratio_chosen - log_ratio_rejected)
}

/// The one-sided positive penalty `λ · max(0, logπref(y_w) − logπθ(y_w))` for one pair.
///
/// Positive exactly when the policy's chosen log-prob has dropped below the reference's.
#[must_use]
#[inline]
pub fn dpop_penalty(pair: &PreferencePair, lambda: f32) -> f32 {
    let deficit = pair.ref_chosen_logp - pair.chosen_logp;
    lambda * deficit.max(0.0)
}

/// DPO-Positive loss for a single pair: `−log σ(h − penalty)`.
///
/// # Errors
/// - [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidLambda`] for an invalid config.
/// - [`RlhfError::NanEncountered`] if any log-prob is NaN or the result is NaN.
pub fn dpop_loss_per_pair(pair: &PreferencePair, cfg: &DpopConfig) -> RlhfResult<f32> {
    cfg.validate()?;
    if pair.chosen_logp.is_nan()
        || pair.rejected_logp.is_nan()
        || pair.ref_chosen_logp.is_nan()
        || pair.ref_rejected_logp.is_nan()
    {
        return Err(RlhfError::NanEncountered);
    }
    let h = dpop_log_ratio(pair, cfg.beta);
    let penalty = dpop_penalty(pair, cfg.lambda);
    let loss = -log_sigmoid(h - penalty);
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Mean DPO-Positive loss over a [`PairBatch`].
///
/// # Errors
/// - [`RlhfError::EmptyInput`] if the batch is empty.
/// - [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidLambda`] for an invalid config.
/// - [`RlhfError::NanEncountered`] if any element is NaN.
pub fn dpop_loss(batch: &PairBatch, cfg: &DpopConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let mut total = 0.0_f32;
    for i in 0..batch.len() {
        let pair = PreferencePair {
            chosen_logp: batch.chosen_logps[i],
            rejected_logp: batch.rejected_logps[i],
            ref_chosen_logp: batch.ref_chosen_logps[i],
            ref_rejected_logp: batch.ref_rejected_logps[i],
        };
        total += dpop_loss_per_pair(&pair, cfg)?;
    }
    let mean = total / batch.len() as f32;
    if mean.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(c: f32, rc: f32, r: f32, rr: f32) -> PreferencePair {
        PreferencePair {
            chosen_logp: c,
            rejected_logp: r,
            ref_chosen_logp: rc,
            ref_rejected_logp: rr,
        }
    }

    #[test]
    fn log_ratio_matches_formula() {
        let p = pair(-1.0, -1.5, -2.0, -1.8);
        let h = dpop_log_ratio(&p, 0.5);
        // 0.5 · ((-1 - -1.5) - (-2 - -1.8)) = 0.5 · (0.5 - (-0.2)) = 0.5·0.7 = 0.35
        assert!((h - 0.35).abs() < 1e-6, "h={h}");
    }

    #[test]
    fn penalty_zero_when_chosen_above_ref() {
        // logπθ(y_w) = -1.0 > logπref(y_w) = -2.0 → deficit negative → penalty 0.
        let p = pair(-1.0, -2.0, -3.0, -3.0);
        let pen = dpop_penalty(&p, 50.0);
        assert!(pen.abs() < 1e-6, "no penalty expected, got {pen}");
    }

    #[test]
    fn penalty_positive_when_chosen_below_ref() {
        // logπθ(y_w) = -3.0 < logπref(y_w) = -1.0 → deficit = 2.0 → penalty = λ·2.
        let p = pair(-3.0, -1.0, -4.0, -4.0);
        let pen = dpop_penalty(&p, 10.0);
        assert!((pen - 20.0).abs() < 1e-5, "expected 20.0, got {pen}");
    }

    #[test]
    fn lambda_zero_equals_standard_dpo() {
        // With λ=0 the loss must equal -log σ(h).
        let p = pair(-1.0, -1.5, -2.0, -1.8);
        let cfg = DpopConfig {
            beta: 0.1,
            lambda: 0.0,
        };
        let loss = dpop_loss_per_pair(&p, &cfg).expect("dpop_loss_per_pair should succeed");
        let h = dpop_log_ratio(&p, 0.1);
        let expected = -log_sigmoid(h);
        assert!(
            (loss - expected).abs() < 1e-6,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn penalty_increases_loss_when_chosen_dropped() {
        // Same h, but one pair has chosen below ref → penalty raises its loss.
        let no_deficit = pair(-1.0, -1.0, -2.0, -1.0);
        let deficit = pair(-2.0, -1.0, -3.0, -1.0);
        // Both have the same h: β·((c-rc)-(r-rr))
        let cfg = DpopConfig {
            beta: 0.1,
            lambda: 50.0,
        };
        let h_no = dpop_log_ratio(&no_deficit, cfg.beta);
        let h_def = dpop_log_ratio(&deficit, cfg.beta);
        assert!(
            (h_no - h_def).abs() < 1e-6,
            "construct equal h: {h_no} vs {h_def}"
        );
        let l_no =
            dpop_loss_per_pair(&no_deficit, &cfg).expect("dpop_loss_per_pair should succeed");
        let l_def = dpop_loss_per_pair(&deficit, &cfg).expect("dpop_loss_per_pair should succeed");
        assert!(
            l_def > l_no,
            "deficit must increase loss: no={l_no}, deficit={l_def}"
        );
    }

    #[test]
    fn loss_lower_for_aligned_pair() {
        // Aligned: chosen ≫ rejected and chosen ≥ ref.
        let aligned = pair(-0.5, -1.0, -3.0, -1.0);
        let unaligned = pair(-3.0, -1.0, -0.5, -1.0);
        let cfg = DpopConfig {
            beta: 0.5,
            lambda: 50.0,
        };
        let l_aligned =
            dpop_loss_per_pair(&aligned, &cfg).expect("dpop_loss_per_pair should succeed");
        let l_unaligned =
            dpop_loss_per_pair(&unaligned, &cfg).expect("dpop_loss_per_pair should succeed");
        assert!(
            l_aligned < l_unaligned,
            "aligned must have lower loss: aligned={l_aligned}, unaligned={l_unaligned}"
        );
    }

    #[test]
    fn loss_finite() {
        let p = pair(-1.0, -1.1, -2.0, -1.9);
        let cfg = DpopConfig::default();
        let loss = dpop_loss_per_pair(&p, &cfg).expect("dpop_loss_per_pair should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn loss_invalid_beta_errors() {
        let p = pair(-1.0, -1.1, -2.0, -1.9);
        let cfg = DpopConfig {
            beta: 0.0,
            lambda: 50.0,
        };
        assert!(matches!(
            dpop_loss_per_pair(&p, &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    #[test]
    fn loss_invalid_lambda_errors() {
        let p = pair(-1.0, -1.1, -2.0, -1.9);
        let cfg = DpopConfig {
            beta: 0.1,
            lambda: -1.0,
        };
        assert!(matches!(
            dpop_loss_per_pair(&p, &cfg),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    #[test]
    fn loss_nan_errors() {
        let p = pair(f32::NAN, -1.1, -2.0, -1.9);
        let cfg = DpopConfig::default();
        assert!(matches!(
            dpop_loss_per_pair(&p, &cfg),
            Err(RlhfError::NanEncountered)
        ));
    }

    #[test]
    fn batch_loss_lambda_zero_matches_dpo() {
        let batch = PairBatch::new(
            vec![-1.0_f32, -0.5],
            vec![-2.0_f32, -3.0],
            vec![-1.1_f32, -0.6],
            vec![-2.1_f32, -3.1],
        )
        .expect("value should be present");
        let cfg = DpopConfig {
            beta: 0.1,
            lambda: 0.0,
        };
        let loss = dpop_loss(&batch, &cfg).expect("dpop_loss should succeed");
        // Manual standard DPO mean.
        let mut expected = 0.0_f32;
        for i in 0..batch.len() {
            let h = 0.1_f32
                * ((batch.chosen_logps[i] - batch.ref_chosen_logps[i])
                    - (batch.rejected_logps[i] - batch.ref_rejected_logps[i]));
            expected += -log_sigmoid(h);
        }
        expected /= batch.len() as f32;
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn batch_loss_is_mean() {
        let batch = PairBatch::new(
            vec![-1.0_f32, -2.0],
            vec![-3.0_f32, -1.0],
            vec![-1.0_f32, -1.0],
            vec![-1.0_f32, -1.0],
        )
        .expect("value should be present");
        let cfg = DpopConfig::default();
        let p0 = pair(
            batch.chosen_logps[0],
            batch.ref_chosen_logps[0],
            batch.rejected_logps[0],
            batch.ref_rejected_logps[0],
        );
        let p1 = pair(
            batch.chosen_logps[1],
            batch.ref_chosen_logps[1],
            batch.rejected_logps[1],
            batch.ref_rejected_logps[1],
        );
        let l0 = dpop_loss_per_pair(&p0, &cfg).expect("dpop_loss_per_pair should succeed");
        let l1 = dpop_loss_per_pair(&p1, &cfg).expect("dpop_loss_per_pair should succeed");
        let mean = dpop_loss(&batch, &cfg).expect("dpop_loss should succeed");
        assert!(
            (mean - (l0 + l1) / 2.0).abs() < 1e-5,
            "mean mismatch: {mean}"
        );
    }

    #[test]
    fn batch_loss_empty_errors() {
        let batch = PairBatch::new(vec![], vec![], vec![], vec![]).expect("new should succeed");
        let cfg = DpopConfig::default();
        assert!(matches!(
            dpop_loss(&batch, &cfg),
            Err(RlhfError::EmptyInput)
        ));
    }

    #[test]
    fn default_config_values() {
        let cfg = DpopConfig::default();
        assert!((cfg.beta - 0.1).abs() < 1e-6, "default beta=0.1");
        assert!((cfg.lambda - 50.0).abs() < 1e-6, "default lambda=50.0");
    }
}
