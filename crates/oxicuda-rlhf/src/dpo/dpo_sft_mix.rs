//! Mixed DPO + SFT auxiliary loss combiner, with optional offline-policy
//! (behaviour-cloning) regularisation.
//!
//! References:
//! * Rafailov et al. 2023, "Direct Preference Optimization", arXiv:2305.18290.
//! * Pang et al. 2024, "Iterative Reasoning Preference Optimization",
//!   arXiv:2404.19733 — adds an NLL term on the chosen response to the DPO loss
//!   so the policy keeps high likelihood on the preferred completions.
//! * Hong et al. 2024, "ORPO", arXiv:2403.07691 — motivates anchoring the DPO
//!   policy with a supervised term to avoid degenerate likelihood collapse.
//!
//! Pure DPO only constrains the *relative* log-probability of chosen vs rejected
//! responses; nothing pins the *absolute* likelihood, so the model can lower the
//! likelihood of the chosen response as long as it lowers the rejected one more.
//! Adding a supervised (negative-log-likelihood) auxiliary term on the chosen
//! response — and, optionally, an *offline-policy regularisation* term that
//! keeps the policy close to the behaviour policy that generated the offline
//! data — counteracts that drift:
//!
//! ```text
//! L = L_DPO
//!   + λ_sft · NLL(chosen)                       (SFT anchor)
//!   + λ_reg · mean[(π_chosen − μ_chosen)²]       (offline-policy regulariser)
//! ```
//!
//! `NLL(chosen) = mean_i (−chosen_logp_i)` is the per-pair negative
//! log-likelihood of the preferred response (log-probs are summed over tokens
//! upstream). The offline regulariser penalises the squared deviation of the
//! current policy's chosen log-prob from the behaviour-policy log-prob `μ` that
//! produced the offline dataset, i.e. a quadratic trust region around the data
//! distribution. With `λ_sft = λ_reg = 0` the combiner reduces exactly to
//! [`crate::dpo::dpo::dpo_loss`].

use crate::dpo::dpo::dpo_log_ratio;
use crate::dpo::step_dpo::log_sigmoid;
use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PairBatch;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the mixed DPO + SFT (+ offline-regularised) loss.
#[derive(Debug, Clone)]
pub struct DpoSftMixConfig {
    /// KL-regularisation temperature β for the DPO term. Must be positive and
    /// finite.
    pub beta: f32,
    /// Weight λ_sft (≥ 0) of the supervised NLL anchor on the chosen response.
    pub sft_weight: f32,
    /// Weight λ_reg (≥ 0) of the offline-policy regulariser. When `0.0` the
    /// regulariser and its `behaviour_logps` argument are ignored.
    pub offline_reg_weight: f32,
}

impl DpoSftMixConfig {
    fn validate(&self) -> RlhfResult<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(RlhfError::InvalidBeta { beta: self.beta });
        }
        if !self.sft_weight.is_finite() || self.sft_weight < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.sft_weight,
            });
        }
        if !self.offline_reg_weight.is_finite() || self.offline_reg_weight < 0.0 {
            return Err(RlhfError::InvalidLambda {
                lambda: self.offline_reg_weight,
            });
        }
        Ok(())
    }
}

// ── Loss breakdown ──────────────────────────────────────────────────────────

/// Decomposition of the combined loss into its constituent terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixLoss {
    /// Mean DPO log-loss over the batch.
    pub dpo: f32,
    /// Mean SFT negative-log-likelihood of the chosen responses.
    pub sft_nll: f32,
    /// Mean offline-policy regularisation penalty (`0.0` when disabled).
    pub offline_reg: f32,
    /// The weighted total `dpo + λ_sft·sft_nll + λ_reg·offline_reg`.
    pub total: f32,
}

// ── Core ────────────────────────────────────────────────────────────────────

fn dpo_term(batch: &PairBatch, beta: f32) -> f32 {
    let total: f32 = batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, beta);
            -log_sigmoid(logit)
        })
        .sum();
    total / batch.len() as f32
}

/// Compute the full DPO + SFT (+ offline-reg) loss breakdown.
///
/// `behaviour_logps` are the log-probs of the chosen responses under the
/// behaviour policy that generated the offline data. They are only consulted
/// when `cfg.offline_reg_weight > 0.0`; in that case they must have the same
/// length as the batch. Pass an empty slice (or any slice) when the regulariser
/// is disabled.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for an empty batch, config errors from
/// `DpoSftMixConfig::validate`, [`RlhfError::DimensionMismatch`] if the
/// regulariser is enabled and `behaviour_logps` has the wrong length, and
/// [`RlhfError::NanEncountered`] on a NaN result.
pub fn dpo_sft_mix_loss(
    batch: &PairBatch,
    behaviour_logps: &[f32],
    cfg: &DpoSftMixConfig,
) -> RlhfResult<MixLoss> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let n = batch.len();

    let dpo = dpo_term(batch, cfg.beta);

    // SFT anchor: mean NLL of the chosen response = mean(-chosen_logp).
    let sft_nll = -batch.chosen_logps.iter().sum::<f32>() / n as f32;

    // Offline-policy regulariser: mean squared deviation from behaviour logps.
    let offline_reg = if cfg.offline_reg_weight > 0.0 {
        if behaviour_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: behaviour_logps.len(),
            });
        }
        let sq: f32 = batch
            .chosen_logps
            .iter()
            .zip(behaviour_logps.iter())
            .map(|(&p, &mu)| {
                let d = p - mu;
                d * d
            })
            .sum();
        sq / n as f32
    } else {
        0.0
    };

    let total = dpo + cfg.sft_weight * sft_nll + cfg.offline_reg_weight * offline_reg;
    if !total.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(MixLoss {
        dpo,
        sft_nll,
        offline_reg,
        total,
    })
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

/// Gradient of the combined DPO + SFT (+ offline-reg) loss w.r.t. the four
/// per-pair log-probability vectors.
///
/// Finite-difference verified against [`dpo_sft_mix_loss`]`.total`.
#[derive(Debug, Clone)]
pub struct MixGrad {
    /// `∂L/∂(chosen log-prob)` — carries the DPO, SFT-anchor and offline-reg terms.
    pub d_chosen_logp: Vec<f32>,
    /// `∂L/∂(rejected log-prob)` — DPO term only.
    pub d_rejected_logp: Vec<f32>,
    /// `∂L/∂(reference chosen log-prob)` — DPO term only.
    pub d_ref_chosen_logp: Vec<f32>,
    /// `∂L/∂(reference rejected log-prob)` — DPO term only.
    pub d_ref_rejected_logp: Vec<f32>,
}

/// Analytic gradient of [`dpo_sft_mix_loss`].
///
/// `total = L_DPO + λ_sft·mean(−c_i) + λ_reg·mean((c_i − μ_i)²)`. The DPO term
/// gives the usual `±β·σ(−logit_i)/N` partials (pattern `+1, −1, −1, +1` over the
/// log-ratio `Δ_i`). The SFT anchor adds `−λ_sft/N` to each chosen partial, and
/// the offline regulariser (when enabled) adds `λ_reg·2(c_i − μ_i)/N`. The
/// behaviour log-probs `μ` are held constant.
///
/// # Errors
///
/// Mirrors [`dpo_sft_mix_loss`]: [`RlhfError::EmptyInput`], config validation,
/// [`RlhfError::DimensionMismatch`] when the regulariser is enabled with a
/// wrong-length `behaviour_logps`, and [`RlhfError::NanEncountered`] on a
/// non-finite gradient.
pub fn dpo_sft_mix_grad(
    batch: &PairBatch,
    behaviour_logps: &[f32],
    cfg: &DpoSftMixConfig,
) -> RlhfResult<MixGrad> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    cfg.validate()?;
    let n = batch.len();
    let inv_n = 1.0 / n as f32;

    let reg_enabled = cfg.offline_reg_weight > 0.0;
    if reg_enabled && behaviour_logps.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: behaviour_logps.len(),
        });
    }

    let mut d_chosen_logp = Vec::with_capacity(n);
    let mut d_rejected_logp = Vec::with_capacity(n);
    let mut d_ref_chosen_logp = Vec::with_capacity(n);
    let mut d_ref_rejected_logp = Vec::with_capacity(n);

    for (i, (((&clp, &rclp), &rlp), &rrlp)) in batch
        .chosen_logps
        .iter()
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .enumerate()
    {
        let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta);
        // DPO: dL/dΔ = −β·σ(−logit), mean-scaled.
        let d_delta = -cfg.beta * sigmoid(-logit) * inv_n;
        let mut dc = d_delta; // chosen DPO part
        // SFT anchor: ∂(λ_sft·mean(−c_i))/∂c_i = −λ_sft/N.
        dc -= cfg.sft_weight * inv_n;
        // Offline regulariser: ∂(λ_reg·mean((c−μ)²))/∂c_i = λ_reg·2(c−μ)/N.
        if reg_enabled {
            dc += cfg.offline_reg_weight * 2.0 * (clp - behaviour_logps[i]) * inv_n;
        }
        if !dc.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_chosen_logp.push(dc);
        d_rejected_logp.push(-d_delta);
        d_ref_chosen_logp.push(-d_delta);
        d_ref_rejected_logp.push(d_delta);
    }

    Ok(MixGrad {
        d_chosen_logp,
        d_rejected_logp,
        d_ref_chosen_logp,
        d_ref_rejected_logp,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpo::dpo::{DpoConfig, dpo_loss};

    fn batch() -> PairBatch {
        PairBatch::new(
            vec![-0.5_f32, -1.0, -1.5],
            vec![-2.0_f32, -2.5, -3.0],
            vec![-1.0_f32, -1.1, -1.2],
            vec![-1.0_f32, -1.1, -1.2],
        )
        .expect("valid batch fixture")
    }

    // 1. Both weights zero → total equals plain DPO loss.
    #[test]
    fn zero_weights_equal_dpo() {
        let b = batch();
        let beta = 0.3_f32;
        let m = dpo_sft_mix_loss(
            &b,
            &[],
            &DpoSftMixConfig {
                beta,
                sft_weight: 0.0,
                offline_reg_weight: 0.0,
            },
        )
        .expect("mix loss");
        let dpo = dpo_loss(&b, &DpoConfig { beta }).expect("dpo loss");
        assert!(
            (m.total - dpo).abs() < 1e-6,
            "total {} vs dpo {dpo}",
            m.total
        );
        assert!((m.dpo - dpo).abs() < 1e-6);
        assert!(m.offline_reg.abs() < 1e-12);
    }

    // 2. SFT NLL component is mean(-chosen_logp).
    #[test]
    fn sft_nll_is_mean_neg_chosen_logp() {
        let b = batch();
        let m = dpo_sft_mix_loss(
            &b,
            &[],
            &DpoSftMixConfig {
                beta: 0.1,
                sft_weight: 1.0,
                offline_reg_weight: 0.0,
            },
        )
        .expect("mix loss");
        let expected = -(-0.5_f32 - 1.0 - 1.5) / 3.0; // = 1.0
        assert!(
            (m.sft_nll - expected).abs() < 1e-6,
            "nll {} vs {expected}",
            m.sft_nll
        );
    }

    // 3. Adding SFT weight increases the total when chosen logps are negative.
    #[test]
    fn sft_weight_increases_total() {
        let b = batch();
        let beta = 0.2_f32;
        let without = dpo_sft_mix_loss(
            &b,
            &[],
            &DpoSftMixConfig {
                beta,
                sft_weight: 0.0,
                offline_reg_weight: 0.0,
            },
        )
        .expect("without");
        let with = dpo_sft_mix_loss(
            &b,
            &[],
            &DpoSftMixConfig {
                beta,
                sft_weight: 0.5,
                offline_reg_weight: 0.0,
            },
        )
        .expect("with");
        assert!(
            with.total > without.total,
            "sft anchor should raise total: {} vs {}",
            with.total,
            without.total
        );
        // Exactly dpo + 0.5 * nll.
        assert!((with.total - (without.dpo + 0.5 * with.sft_nll)).abs() < 1e-6);
    }

    // 4. Offline regulariser is zero when policy equals behaviour logps.
    #[test]
    fn offline_reg_zero_at_behaviour() {
        let b = batch();
        let behaviour = b.chosen_logps.clone();
        let m = dpo_sft_mix_loss(
            &b,
            &behaviour,
            &DpoSftMixConfig {
                beta: 0.2,
                sft_weight: 0.0,
                offline_reg_weight: 1.0,
            },
        )
        .expect("mix loss");
        assert!(
            m.offline_reg.abs() < 1e-6,
            "reg should be ~0, got {}",
            m.offline_reg
        );
    }

    // 5. Offline regulariser grows with deviation from behaviour logps.
    #[test]
    fn offline_reg_grows_with_deviation() {
        let b = batch();
        // Behaviour logps shifted by a constant from policy chosen logps.
        let behaviour: Vec<f32> = b.chosen_logps.iter().map(|&p| p - 1.0).collect();
        let m = dpo_sft_mix_loss(
            &b,
            &behaviour,
            &DpoSftMixConfig {
                beta: 0.2,
                sft_weight: 0.0,
                offline_reg_weight: 1.0,
            },
        )
        .expect("mix loss");
        // Each deviation is exactly +1.0 → squared = 1.0 → mean = 1.0.
        assert!(
            (m.offline_reg - 1.0).abs() < 1e-6,
            "reg {} should be 1.0",
            m.offline_reg
        );
    }

    // 6. Offline reg disabled → behaviour_logps ignored even if wrong length.
    #[test]
    fn offline_reg_disabled_ignores_behaviour() {
        let b = batch();
        let m = dpo_sft_mix_loss(
            &b,
            &[1.0, 2.0], // wrong length, but reg disabled
            &DpoSftMixConfig {
                beta: 0.2,
                sft_weight: 1.0,
                offline_reg_weight: 0.0,
            },
        );
        assert!(
            m.is_ok(),
            "disabled regulariser must ignore behaviour slice"
        );
    }

    // 7. Offline reg enabled with wrong-length behaviour → DimensionMismatch.
    #[test]
    fn offline_reg_length_mismatch_errors() {
        let b = batch();
        assert!(matches!(
            dpo_sft_mix_loss(
                &b,
                &[1.0, 2.0],
                &DpoSftMixConfig {
                    beta: 0.2,
                    sft_weight: 0.0,
                    offline_reg_weight: 1.0
                }
            ),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 8. Full combiner total is the exact weighted sum of its three parts.
    #[test]
    fn total_is_weighted_sum_of_parts() {
        let b = batch();
        let behaviour: Vec<f32> = b.chosen_logps.iter().map(|&p| p - 0.5).collect();
        let cfg = DpoSftMixConfig {
            beta: 0.25,
            sft_weight: 0.3,
            offline_reg_weight: 0.7,
        };
        let m = dpo_sft_mix_loss(&b, &behaviour, &cfg).expect("mix loss");
        let recomputed =
            m.dpo + cfg.sft_weight * m.sft_nll + cfg.offline_reg_weight * m.offline_reg;
        assert!((m.total - recomputed).abs() < 1e-6);
    }

    // 9. Invalid beta rejected.
    #[test]
    fn invalid_beta_errors() {
        let b = batch();
        assert!(matches!(
            dpo_sft_mix_loss(
                &b,
                &[],
                &DpoSftMixConfig {
                    beta: -1.0,
                    sft_weight: 0.0,
                    offline_reg_weight: 0.0
                }
            ),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }

    // 10. Negative weights rejected.
    #[test]
    fn negative_weights_error() {
        let b = batch();
        assert!(matches!(
            dpo_sft_mix_loss(
                &b,
                &[],
                &DpoSftMixConfig {
                    beta: 0.1,
                    sft_weight: -0.1,
                    offline_reg_weight: 0.0
                }
            ),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            dpo_sft_mix_loss(
                &b,
                &[],
                &DpoSftMixConfig {
                    beta: 0.1,
                    sft_weight: 0.0,
                    offline_reg_weight: -2.0
                }
            ),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 11. Empty batch rejected.
    #[test]
    fn empty_batch_errors() {
        let empty = PairBatch::new(vec![], vec![], vec![], vec![]).expect("empty");
        assert!(matches!(
            dpo_sft_mix_loss(
                &empty,
                &[],
                &DpoSftMixConfig {
                    beta: 0.1,
                    sft_weight: 0.0,
                    offline_reg_weight: 0.0
                }
            ),
            Err(RlhfError::EmptyInput)
        ));
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

    fn batch() -> PairBatch {
        PairBatch::new(
            vec![-0.5_f32, -1.0, -1.5],
            vec![-2.0_f32, -2.5, -3.0],
            vec![-1.0_f32, -1.1, -1.2],
            vec![-1.0_f32, -1.1, -1.2],
        )
        .expect("batch")
    }

    fn total(b: &PairBatch, beh: &[f32], cfg: &DpoSftMixConfig) -> f32 {
        dpo_sft_mix_loss(b, beh, cfg).expect("loss").total
    }

    #[test]
    fn mix_grad_matches_fd_all_terms() {
        let b = batch();
        let beh: Vec<f32> = b.chosen_logps.iter().map(|&p| p - 0.5).collect();
        let cfg = DpoSftMixConfig {
            beta: 0.3,
            sft_weight: 0.4,
            offline_reg_weight: 0.6,
        };
        let g = dpo_sft_mix_grad(&b, &beh, &cfg).expect("grad");
        let h = 1e-2;
        let idx = 1usize;
        let fd_c = central_diff(
            |v| {
                let mut c = b.chosen_logps.clone();
                c[idx] = v;
                let nb = PairBatch::new(
                    c,
                    b.rejected_logps.clone(),
                    b.ref_chosen_logps.clone(),
                    b.ref_rejected_logps.clone(),
                )
                .expect("b");
                total(&nb, &beh, &cfg)
            },
            b.chosen_logps[idx],
            h,
        );
        let fd_r = central_diff(
            |v| {
                let mut r = b.rejected_logps.clone();
                r[idx] = v;
                let nb = PairBatch::new(
                    b.chosen_logps.clone(),
                    r,
                    b.ref_chosen_logps.clone(),
                    b.ref_rejected_logps.clone(),
                )
                .expect("b");
                total(&nb, &beh, &cfg)
            },
            b.rejected_logps[idx],
            h,
        );
        let fd_rc = central_diff(
            |v| {
                let mut rc = b.ref_chosen_logps.clone();
                rc[idx] = v;
                let nb = PairBatch::new(
                    b.chosen_logps.clone(),
                    b.rejected_logps.clone(),
                    rc,
                    b.ref_rejected_logps.clone(),
                )
                .expect("b");
                total(&nb, &beh, &cfg)
            },
            b.ref_chosen_logps[idx],
            h,
        );
        assert_close(g.d_chosen_logp[idx], fd_c, "d_chosen");
        assert_close(g.d_rejected_logp[idx], fd_r, "d_rejected");
        assert_close(g.d_ref_chosen_logp[idx], fd_rc, "d_ref_chosen");
    }

    #[test]
    fn mix_grad_zero_weights_equals_dpo() {
        let b = batch();
        let cfg = DpoSftMixConfig {
            beta: 0.3,
            sft_weight: 0.0,
            offline_reg_weight: 0.0,
        };
        let g = dpo_sft_mix_grad(&b, &[], &cfg).expect("grad");
        let dg = crate::dpo::dpo::dpo_grad(&b, &crate::dpo::dpo::DpoConfig { beta: 0.3 })
            .expect("dpo grad");
        for (i, dgi) in dg.iter().enumerate() {
            assert!((g.d_chosen_logp[i] - dgi.d_chosen_logp).abs() < 1e-6);
            assert!((g.d_rejected_logp[i] - dgi.d_rejected_logp).abs() < 1e-6);
        }
    }

    #[test]
    fn mix_grad_reg_disabled_ignores_behaviour_length() {
        let b = batch();
        // Wrong-length behaviour slice is fine when reg disabled.
        let g = dpo_sft_mix_grad(
            &b,
            &[1.0, 2.0],
            &DpoSftMixConfig {
                beta: 0.2,
                sft_weight: 1.0,
                offline_reg_weight: 0.0,
            },
        );
        assert!(g.is_ok());
    }

    #[test]
    fn mix_grad_reg_length_mismatch_errors() {
        let b = batch();
        assert!(matches!(
            dpo_sft_mix_grad(
                &b,
                &[1.0, 2.0],
                &DpoSftMixConfig {
                    beta: 0.2,
                    sft_weight: 0.0,
                    offline_reg_weight: 1.0
                }
            ),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }
}
