//! DPO with Identity-Preference-Optimisation regularisation (a DPO + IPO blend).
//!
//! Reference: Rafailov et al. 2023, "Direct Preference Optimization",
//! arXiv:2305.18290 (DPO); Azar et al. 2024, "A General Theoretical Paradigm to
//! Understand Learning from Human Preferences", arXiv:2310.12036 (IPO / ΨPO).
//!
//! Standard DPO optimises the sigmoid log-likelihood of the preference,
//!
//! ```text
//! L_DPO = -log σ(h),    h = β · ((π_w − ref_w) − (π_l − ref_l))
//! ```
//!
//! while IPO replaces the sigmoid log-loss with a squared loss that anchors the
//! implicit-reward margin to the constant `1 / (2β)`,
//!
//! ```text
//! L_IPO = (h/β − 1/(2β))² = ((π_w − ref_w) − (π_l − ref_l) − 1/(2β))²
//! ```
//!
//! DPO has an unbounded objective: as the margin `h → +∞` the loss → 0, so the
//! policy can over-fit on easy / noisy pairs and drift arbitrarily far from the
//! reference. IPO is bounded-margin (it *targets* a finite margin) but discards
//! the calibrated probabilistic interpretation of DPO. A practical compromise is
//! to optimise a convex blend
//!
//! ```text
//! L = (1 − α) · L_DPO + α · λ · L_IPO,   α ∈ [0, 1]
//! ```
//!
//! where `α` trades the DPO log-loss against the IPO margin-anchor and `λ ≥ 0`
//! scales the (typically larger) squared term so the two contributions are
//! comparable. `α = 0` reduces to plain DPO and `α = 1` (with `λ = 1`) reduces to
//! plain IPO, both reproduced bit-for-bit by reusing the same primitives the
//! standalone [`crate::dpo::dpo`] / [`crate::dpo::ipo`] modules use.

use crate::dpo::dpo::dpo_log_ratio;
use crate::dpo::step_dpo::log_sigmoid;
use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PairBatch;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the DPO + IPO blended loss.
#[derive(Debug, Clone)]
pub struct DpoIpoBlendConfig {
    /// KL-regularisation temperature β. Must be positive and finite; identical
    /// in meaning to [`crate::dpo::dpo::DpoConfig::beta`].
    pub beta: f32,
    /// Blend coefficient α ∈ [0, 1]. `0.0` → pure DPO, `1.0` → pure IPO (scaled
    /// by `ipo_weight`). Intermediate values mix the two objectives.
    pub alpha: f32,
    /// Non-negative scale λ applied to the IPO squared term so it is comparable
    /// in magnitude to the DPO log-loss. `1.0` leaves IPO un-scaled.
    pub ipo_weight: f32,
}

impl DpoIpoBlendConfig {
    /// Validate β (> 0, finite), α ∈ [0, 1], and λ (≥ 0, finite).
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(RlhfError::InvalidLambda { lambda: self.alpha });
        }
        if !self.ipo_weight.is_finite() || self.ipo_weight < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.ipo_weight,
            });
        }
        Ok(())
    }
}

// ── Per-pair decomposition ──────────────────────────────────────────────────

/// Per-pair breakdown of the blended loss into its DPO and IPO components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendComponents {
    /// The DPO log-loss `-log σ(h)` for this pair.
    pub dpo: f32,
    /// The (un-weighted) IPO squared loss `(Δ − 1/(2β))²` for this pair, where
    /// `Δ = (π_w − ref_w) − (π_l − ref_l)` is the raw log-ratio margin.
    pub ipo: f32,
    /// The combined `(1 − α)·dpo + α·λ·ipo`.
    pub blended: f32,
}

/// Compute the per-pair blended-loss components.
///
/// `Δ = (chosen_lp − ref_chosen_lp) − (rejected_lp − ref_rejected_lp)` is the
/// raw log-ratio margin; `h = β·Δ` is the DPO logit.
///
/// # Errors
///
/// Returns [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidLambda`] for an
/// invalid config and [`RlhfError::NanEncountered`] if any component is NaN.
pub fn blend_components_per_pair(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    cfg: &DpoIpoBlendConfig,
) -> RlhfResult<BlendComponents> {
    cfg.validate()?;
    let logit = dpo_log_ratio(
        chosen_lp,
        ref_chosen_lp,
        rejected_lp,
        ref_rejected_lp,
        cfg.beta,
    );
    let dpo = -log_sigmoid(logit);

    // Raw (un-scaled-by-β) margin Δ = logit / β.
    let margin = logit / cfg.beta;
    let target = 1.0 / (2.0 * cfg.beta);
    let residual = margin - target;
    let ipo = residual * residual;

    let blended = (1.0 - cfg.alpha) * dpo + cfg.alpha * cfg.ipo_weight * ipo;
    if dpo.is_nan() || ipo.is_nan() || blended.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(BlendComponents { dpo, ipo, blended })
}

// ── Batch loss ──────────────────────────────────────────────────────────────

/// Mean blended DPO + IPO loss over a preference batch.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for an empty batch, config errors from
/// `DpoIpoBlendConfig::validate`, and [`RlhfError::NanEncountered`] on NaN.
pub fn dpo_ipo_blend_loss(batch: &PairBatch, cfg: &DpoIpoBlendConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let mut total = 0.0_f32;
    for (((&clp, &rlp), &rclp), &rrlp) in batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
    {
        let c = blend_components_per_pair(clp, rclp, rlp, rrlp, cfg)?;
        total += c.blended;
    }
    let loss = total / batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

// ── Gradient ────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid `σ(x)`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Gradient of the per-pair blended DPO + IPO loss w.r.t. the four log-probs.
///
/// Finite-difference verified against [`blend_components_per_pair`]`.blended`.
#[derive(Debug, Clone, Copy)]
pub struct BlendGrad {
    /// `∂L/∂(policy chosen log-prob)`.
    pub d_chosen_logp: f32,
    /// `∂L/∂(policy rejected log-prob)`.
    pub d_rejected_logp: f32,
    /// `∂L/∂(reference chosen log-prob)`.
    pub d_ref_chosen_logp: f32,
    /// `∂L/∂(reference rejected log-prob)`.
    pub d_ref_rejected_logp: f32,
}

#[inline]
fn blend_grad_inner(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    cfg: &DpoIpoBlendConfig,
) -> BlendGrad {
    let logit = dpo_log_ratio(
        chosen_lp,
        ref_chosen_lp,
        rejected_lp,
        ref_rejected_lp,
        cfg.beta,
    );
    let margin = logit / cfg.beta;
    let target = 1.0 / (2.0 * cfg.beta);
    // d(dpo)/dΔ = −β·σ(−logit) ; d(ipo)/dΔ = 2(Δ − τ).
    let d_dpo = -cfg.beta * sigmoid(-logit);
    let d_ipo = 2.0 * (margin - target);
    let d_delta = (1.0 - cfg.alpha) * d_dpo + cfg.alpha * cfg.ipo_weight * d_ipo;
    BlendGrad {
        d_chosen_logp: d_delta,
        d_rejected_logp: -d_delta,
        d_ref_chosen_logp: -d_delta,
        d_ref_rejected_logp: d_delta,
    }
}

/// Analytic gradient of the per-pair blended loss.
///
/// `L = (1−α)·(−log σ(β·Δ)) + α·λ·(Δ − τ)²` with `Δ = (c−rc) − (l−rr)` and
/// `τ = 1/(2β)`. Differentiating, `dL/dΔ = (1−α)·(−β·σ(−β·Δ)) + α·λ·2(Δ − τ)`,
/// then the four partials follow the `+1, −1, −1, +1` pattern of `Δ`.
///
/// # Errors
/// Returns [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidLambda`] for an
/// invalid config and [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn blend_grad_per_pair(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    cfg: &DpoIpoBlendConfig,
) -> RlhfResult<BlendGrad> {
    cfg.validate()?;
    let grad = blend_grad_inner(chosen_lp, ref_chosen_lp, rejected_lp, ref_rejected_lp, cfg);
    if !grad.d_chosen_logp.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(grad)
}

/// Analytic gradient of the mean-reduced [`dpo_ipo_blend_loss`].
///
/// Returns one [`BlendGrad`] per pair, each scaled by `1 / batch.len()`.
///
/// # Errors
/// Returns [`RlhfError::EmptyInput`] for an empty batch, config errors from
/// `DpoIpoBlendConfig::validate`, and [`RlhfError::NanEncountered`] on NaN.
pub fn dpo_ipo_blend_grad(
    batch: &PairBatch,
    cfg: &DpoIpoBlendConfig,
) -> RlhfResult<Vec<BlendGrad>> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let inv_n = 1.0 / batch.len() as f32;
    let mut grads = Vec::with_capacity(batch.len());
    for (((&clp, &rlp), &rclp), &rrlp) in batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
    {
        let g = blend_grad_inner(clp, rclp, rlp, rrlp, cfg);
        let scaled = BlendGrad {
            d_chosen_logp: g.d_chosen_logp * inv_n,
            d_rejected_logp: g.d_rejected_logp * inv_n,
            d_ref_chosen_logp: g.d_ref_chosen_logp * inv_n,
            d_ref_rejected_logp: g.d_ref_rejected_logp * inv_n,
        };
        if !scaled.d_chosen_logp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(scaled);
    }
    Ok(grads)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpo::dpo::{DpoConfig, dpo_loss};
    use crate::dpo::ipo::{IpoConfig, ipo_loss};

    fn batch() -> PairBatch {
        PairBatch::new(
            vec![-0.5_f32, -1.0, -1.5],
            vec![-2.0_f32, -2.5, -3.0],
            vec![-1.0_f32, -1.1, -1.2],
            vec![-1.0_f32, -1.1, -1.2],
        )
        .expect("valid batch fixture")
    }

    // 1. alpha = 0 reproduces plain DPO exactly.
    #[test]
    fn alpha_zero_equals_dpo() {
        let b = batch();
        let beta = 0.3_f32;
        let blend = dpo_ipo_blend_loss(
            &b,
            &DpoIpoBlendConfig {
                beta,
                alpha: 0.0,
                ipo_weight: 1.0,
            },
        )
        .expect("blend loss");
        let dpo = dpo_loss(&b, &DpoConfig { beta }).expect("dpo loss");
        assert!((blend - dpo).abs() < 1e-6, "blend {blend} vs dpo {dpo}");
    }

    // 2. alpha = 1, ipo_weight = 1 reproduces plain IPO exactly.
    #[test]
    fn alpha_one_equals_ipo() {
        let b = batch();
        let beta = 0.3_f32;
        let blend = dpo_ipo_blend_loss(
            &b,
            &DpoIpoBlendConfig {
                beta,
                alpha: 1.0,
                ipo_weight: 1.0,
            },
        )
        .expect("blend loss");
        let ipo = ipo_loss(&b, &IpoConfig { beta }).expect("ipo loss");
        assert!((blend - ipo).abs() < 1e-6, "blend {blend} vs ipo {ipo}");
    }

    // 3. Intermediate alpha is the exact convex combination of the two.
    #[test]
    fn intermediate_is_convex_combination() {
        let b = batch();
        let beta = 0.25_f32;
        let alpha = 0.4_f32;
        let lambda = 0.5_f32;
        let dpo = dpo_loss(&b, &DpoConfig { beta }).expect("dpo");
        let ipo = ipo_loss(&b, &IpoConfig { beta }).expect("ipo");
        let expected = (1.0 - alpha) * dpo + alpha * lambda * ipo;
        let blend = dpo_ipo_blend_loss(
            &b,
            &DpoIpoBlendConfig {
                beta,
                alpha,
                ipo_weight: lambda,
            },
        )
        .expect("blend");
        assert!(
            (blend - expected).abs() < 1e-6,
            "blend {blend} vs expected {expected}"
        );
    }

    // 4. Per-pair components: dpo and ipo are individually correct.
    #[test]
    fn per_pair_components_match_definitions() {
        let beta = 0.5_f32;
        let cfg = DpoIpoBlendConfig {
            beta,
            alpha: 0.5,
            ipo_weight: 1.0,
        };
        // Equal log-probs everywhere → logit 0 → dpo = ln 2.
        let c = blend_components_per_pair(-1.0, -1.0, -1.0, -1.0, &cfg).expect("components");
        assert!((c.dpo - std::f32::consts::LN_2).abs() < 1e-6);
        // margin Δ = 0, target = 1/(2β) = 1.0 → ipo = 1.0.
        assert!((c.ipo - 1.0).abs() < 1e-6, "ipo {}", c.ipo);
        let expected = 0.5 * c.dpo + 0.5 * 1.0 * c.ipo;
        assert!((c.blended - expected).abs() < 1e-6);
    }

    // 5. Aligned pair gives a lower DPO component than a misaligned one.
    #[test]
    fn aligned_lower_dpo_component() {
        let cfg = DpoIpoBlendConfig {
            beta: 0.5,
            alpha: 0.0,
            ipo_weight: 1.0,
        };
        let aligned = blend_components_per_pair(-0.2, -1.0, -3.0, -1.0, &cfg).expect("aligned");
        let misaligned =
            blend_components_per_pair(-3.0, -1.0, -0.2, -1.0, &cfg).expect("misaligned");
        assert!(
            aligned.dpo < misaligned.dpo,
            "aligned dpo {} should be < misaligned {}",
            aligned.dpo,
            misaligned.dpo
        );
    }

    // 6. IPO component is minimised when Δ hits the target 1/(2β).
    #[test]
    fn ipo_component_minimised_at_target() {
        let beta = 1.0_f32;
        let cfg = DpoIpoBlendConfig {
            beta,
            alpha: 1.0,
            ipo_weight: 1.0,
        };
        let target = 1.0 / (2.0 * beta); // 0.5
        // Construct Δ = chosen_lp - rejected_lp (refs equal) exactly = target.
        let at_target = blend_components_per_pair(-0.5, -1.0, -1.0, -1.0, &cfg).expect("at");
        assert!(
            at_target.ipo < 1e-6,
            "ipo at target should be ~0, got {}",
            at_target.ipo
        );
        let _ = target;
        let off = blend_components_per_pair(-3.0, -1.0, -1.0, -1.0, &cfg).expect("off");
        assert!(off.ipo > at_target.ipo, "off-target ipo must be larger");
    }

    // 7. Invalid beta rejected.
    #[test]
    fn invalid_beta_errors() {
        let b = batch();
        assert!(matches!(
            dpo_ipo_blend_loss(
                &b,
                &DpoIpoBlendConfig {
                    beta: 0.0,
                    alpha: 0.5,
                    ipo_weight: 1.0
                }
            ),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 8. alpha out of [0,1] rejected.
    #[test]
    fn alpha_out_of_range_errors() {
        let b = batch();
        assert!(matches!(
            dpo_ipo_blend_loss(
                &b,
                &DpoIpoBlendConfig {
                    beta: 0.1,
                    alpha: 1.5,
                    ipo_weight: 1.0
                }
            ),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            dpo_ipo_blend_loss(
                &b,
                &DpoIpoBlendConfig {
                    beta: 0.1,
                    alpha: -0.1,
                    ipo_weight: 1.0
                }
            ),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 9. Negative ipo_weight rejected.
    #[test]
    fn negative_ipo_weight_errors() {
        let b = batch();
        assert!(matches!(
            dpo_ipo_blend_loss(
                &b,
                &DpoIpoBlendConfig {
                    beta: 0.1,
                    alpha: 0.5,
                    ipo_weight: -1.0
                }
            ),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 10. Empty batch rejected.
    #[test]
    fn empty_batch_errors() {
        let empty = PairBatch::new(vec![], vec![], vec![], vec![]).expect("empty batch");
        assert!(matches!(
            dpo_ipo_blend_loss(
                &empty,
                &DpoIpoBlendConfig {
                    beta: 0.1,
                    alpha: 0.5,
                    ipo_weight: 1.0
                }
            ),
            Err(RlhfError::EmptyInput)
        ));
    }
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use crate::dpo::dpo::{DpoConfig, dpo_grad_per_pair};
    use crate::dpo::ipo::{IpoConfig, ipo_grad};

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

    fn blended(c: f32, rc: f32, r: f32, rr: f32, cfg: &DpoIpoBlendConfig) -> f32 {
        blend_components_per_pair(c, rc, r, rr, cfg)
            .expect("components")
            .blended
    }

    #[test]
    fn blend_grad_matches_fd() {
        let cfg = DpoIpoBlendConfig {
            beta: 0.4,
            alpha: 0.5,
            ipo_weight: 0.7,
        };
        let (c, rc, r, rr) = (-0.6_f32, -1.0, -1.4, -1.1);
        let g = blend_grad_per_pair(c, rc, r, rr, &cfg).expect("grad");
        let h = 1e-2;
        let fd_c = central_diff(|v| blended(v, rc, r, rr, &cfg), c, h);
        let fd_rc = central_diff(|v| blended(c, v, r, rr, &cfg), rc, h);
        let fd_r = central_diff(|v| blended(c, rc, v, rr, &cfg), r, h);
        let fd_rr = central_diff(|v| blended(c, rc, r, v, &cfg), rr, h);
        assert_close(g.d_chosen_logp, fd_c, "d_chosen");
        assert_close(g.d_ref_chosen_logp, fd_rc, "d_ref_chosen");
        assert_close(g.d_rejected_logp, fd_r, "d_rejected");
        assert_close(g.d_ref_rejected_logp, fd_rr, "d_ref_rejected");
    }

    #[test]
    fn blend_grad_alpha_zero_equals_dpo_grad() {
        let cfg = DpoIpoBlendConfig {
            beta: 0.3,
            alpha: 0.0,
            ipo_weight: 1.0,
        };
        let p = crate::preference::pair::PreferencePair {
            chosen_logp: -0.7,
            rejected_logp: -1.4,
            ref_chosen_logp: -0.9,
            ref_rejected_logp: -1.1,
        };
        let g = blend_grad_per_pair(-0.7, -0.9, -1.4, -1.1, &cfg).expect("grad");
        let dg = dpo_grad_per_pair(&p, &DpoConfig { beta: 0.3 }).expect("dpo grad");
        assert!((g.d_chosen_logp - dg.d_chosen_logp).abs() < 1e-6);
        assert!((g.d_rejected_logp - dg.d_rejected_logp).abs() < 1e-6);
    }

    #[test]
    fn blend_grad_alpha_one_equals_ipo_grad() {
        let beta = 0.3_f32;
        let cfg = DpoIpoBlendConfig {
            beta,
            alpha: 1.0,
            ipo_weight: 1.0,
        };
        let batch = PairBatch::new(vec![-0.7], vec![-1.4], vec![-0.9], vec![-1.1]).expect("batch");
        let bg = dpo_ipo_blend_grad(&batch, &cfg).expect("grad");
        let ig = ipo_grad(&batch, &IpoConfig { beta }).expect("ipo grad");
        assert!((bg[0].d_chosen_logp - ig[0].d_chosen_logp).abs() < 1e-6);
        assert!((bg[0].d_ref_rejected_logp - ig[0].d_ref_rejected_logp).abs() < 1e-6);
    }

    #[test]
    fn blend_grad_batch_matches_fd() {
        let cfg = DpoIpoBlendConfig {
            beta: 0.25,
            alpha: 0.4,
            ipo_weight: 0.5,
        };
        let batch = PairBatch::new(
            vec![-0.5_f32, -1.0],
            vec![-2.0_f32, -2.5],
            vec![-1.0_f32, -1.1],
            vec![-1.0_f32, -1.1],
        )
        .expect("batch");
        let grads = dpo_ipo_blend_grad(&batch, &cfg).expect("grads");
        let h = 1e-2;
        let fd = central_diff(
            |v| {
                let mut c = batch.chosen_logps.clone();
                c[1] = v;
                let b = PairBatch::new(
                    c,
                    batch.rejected_logps.clone(),
                    batch.ref_chosen_logps.clone(),
                    batch.ref_rejected_logps.clone(),
                )
                .expect("b");
                dpo_ipo_blend_loss(&b, &cfg).expect("loss")
            },
            batch.chosen_logps[1],
            h,
        );
        assert_close(grads[1].d_chosen_logp, fd, "batch d_chosen[1]");
    }

    #[test]
    fn blend_grad_invalid_beta_errors() {
        let cfg = DpoIpoBlendConfig {
            beta: 0.0,
            alpha: 0.5,
            ipo_weight: 1.0,
        };
        assert!(matches!(
            blend_grad_per_pair(-1.0, -1.0, -1.0, -1.0, &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}
