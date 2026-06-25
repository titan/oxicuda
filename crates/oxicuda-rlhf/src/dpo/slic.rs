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

/// Gradient of the per-pair SLiC-HF loss w.r.t. the three sequence log-likelihoods.
///
/// Finite-difference verified against [`slic_loss`].
#[derive(Debug, Clone, Copy)]
pub struct SlicGrad {
    /// `∂L/∂s⁺` (preferred sequence log-likelihood).
    pub d_pos_logp: f32,
    /// `∂L/∂s⁻` (dispreferred sequence log-likelihood).
    pub d_neg_logp: f32,
    /// `∂L/∂s_ref` (reference-target log-likelihood) — always `−reg_weight`.
    pub d_ref_logp: f32,
}

#[inline]
fn slic_pair_grad_inner(pair: &SlicPair, cfg: &SlicConfig) -> SlicGrad {
    let margin = pair.pos_logp - pair.neg_logp;
    // Hinge active (binding) iff δ − margin > 0; in the flat region the
    // calibration gradient is exactly 0.
    let binding = (cfg.delta - margin) > 0.0;
    let (d_pos, d_neg) = if binding { (-1.0, 1.0) } else { (0.0, 0.0) };
    SlicGrad {
        d_pos_logp: d_pos,
        d_neg_logp: d_neg,
        // L_reg = −s_ref enters linearly with weight λ.
        d_ref_logp: -cfg.reg_weight,
    }
}

/// Analytic gradient of [`slic_loss`] for a single pair.
///
/// `L = max(0, δ − (s⁺ − s⁻)) + λ·(−s_ref)`. Where the hinge binds
/// (`s⁺ − s⁻ < δ`) the calibration term contributes `∂L/∂s⁺ = −1`,
/// `∂L/∂s⁻ = +1`; in the satisfied-margin region both are `0` (sub-gradient).
/// The regularisation term is linear, giving `∂L/∂s_ref = −λ` always.
///
/// # Errors
/// - [`RlhfError::InvalidMargin`] / [`RlhfError::InvalidLambda`] for an invalid config.
/// - [`RlhfError::NanEncountered`] if any log-prob is NaN.
pub fn slic_grad(pair: &SlicPair, cfg: &SlicConfig) -> RlhfResult<SlicGrad> {
    cfg.validate()?;
    if pair.pos_logp.is_nan() || pair.neg_logp.is_nan() || pair.ref_logp.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(slic_pair_grad_inner(pair, cfg))
}

/// Analytic gradient of the mean-reduced [`slic_loss_batch`].
///
/// Returns one [`SlicGrad`] per pair, each scaled by `1 / pairs.len()` for the
/// mean reduction.
///
/// # Errors
/// - [`RlhfError::EmptyInput`] for an empty batch.
/// - Propagates config / NaN errors from [`slic_grad`].
pub fn slic_grad_batch(pairs: &[SlicPair], cfg: &SlicConfig) -> RlhfResult<Vec<SlicGrad>> {
    if pairs.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let inv_n = 1.0 / pairs.len() as f32;
    let mut grads = Vec::with_capacity(pairs.len());
    for p in pairs {
        if p.pos_logp.is_nan() || p.neg_logp.is_nan() || p.ref_logp.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        let g = slic_pair_grad_inner(p, cfg);
        grads.push(SlicGrad {
            d_pos_logp: g.d_pos_logp * inv_n,
            d_neg_logp: g.d_neg_logp * inv_n,
            d_ref_logp: g.d_ref_logp * inv_n,
        });
    }
    Ok(grads)
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

#[cfg(test)]
mod grad_tests {
    use super::*;

    fn central_diff(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        ((f(x + h) as f64 - f(x - h) as f64) / (2.0 * h as f64)) as f32
    }

    fn assert_close(analytic: f32, fd: f32, label: &str) {
        let denom = analytic.abs().max(1e-3);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel <= 1e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    fn mk(pos: f32, neg: f32, refl: f32) -> SlicPair {
        SlicPair {
            pos_logp: pos,
            neg_logp: neg,
            ref_logp: refl,
        }
    }

    #[test]
    fn slic_grad_matches_fd_binding_region() {
        // margin = pos - neg = 0.5 < δ = 2 → hinge binds (stable under ±h).
        let cfg = SlicConfig {
            delta: 2.0,
            reg_weight: 0.3,
        };
        let p = mk(-1.0, -1.5, -2.0);
        let g = slic_grad(&p, &cfg).expect("grad");
        let h = 1e-2;
        let fd_pos = central_diff(|v| slic_loss(&mk(v, -1.5, -2.0), &cfg).expect("l"), -1.0, h);
        let fd_neg = central_diff(|v| slic_loss(&mk(-1.0, v, -2.0), &cfg).expect("l"), -1.5, h);
        let fd_ref = central_diff(|v| slic_loss(&mk(-1.0, -1.5, v), &cfg).expect("l"), -2.0, h);
        assert_close(g.d_pos_logp, fd_pos, "d_pos");
        assert_close(g.d_neg_logp, fd_neg, "d_neg");
        assert_close(g.d_ref_logp, fd_ref, "d_ref");
        // Binding: ∂/∂s⁺ = −1, ∂/∂s⁻ = +1.
        assert!((g.d_pos_logp + 1.0).abs() < 1e-7);
        assert!((g.d_neg_logp - 1.0).abs() < 1e-7);
    }

    #[test]
    fn slic_grad_zero_calibration_in_satisfied_region() {
        // margin = 4.5 ≫ δ = 1 → hinge not binding → calibration gradient 0.
        let cfg = SlicConfig {
            delta: 1.0,
            reg_weight: 0.2,
        };
        let p = mk(-0.5, -5.0, -0.5);
        let g = slic_grad(&p, &cfg).expect("grad");
        let h = 1e-2;
        let fd_pos = central_diff(|v| slic_loss(&mk(v, -5.0, -0.5), &cfg).expect("l"), -0.5, h);
        assert!(fd_pos.abs() < 1e-6, "fd in flat region = {fd_pos}");
        assert_eq!(g.d_pos_logp, 0.0);
        assert_eq!(g.d_neg_logp, 0.0);
        // Regulariser still active.
        assert!((g.d_ref_logp + cfg.reg_weight).abs() < 1e-7);
    }

    #[test]
    fn slic_grad_batch_matches_fd() {
        let cfg = SlicConfig {
            delta: 2.0,
            reg_weight: 0.1,
        };
        let pairs = vec![mk(-1.0, -1.5, -2.0), mk(-0.5, -1.0, -0.5)];
        let grads = slic_grad_batch(&pairs, &cfg).expect("grads");
        let h = 1e-2;
        let fd = central_diff(
            |v| {
                let mut ps = pairs.clone();
                ps[0].pos_logp = v;
                slic_loss_batch(&ps, &cfg).expect("loss")
            },
            pairs[0].pos_logp,
            h,
        );
        assert_close(grads[0].d_pos_logp, fd, "batch d_pos[0]");
    }

    #[test]
    fn slic_grad_invalid_config_errors() {
        let cfg = SlicConfig {
            delta: -1.0,
            reg_weight: 0.1,
        };
        assert!(matches!(
            slic_grad(&mk(-1.0, -2.0, -1.0), &cfg),
            Err(RlhfError::InvalidMargin { .. })
        ));
    }
}
