use crate::error::{RlhfError, RlhfResult};

pub struct OrpoConfig {
    pub lambda: f32,
}

pub fn log_odds(lp: f32) -> f32 {
    let clamped = lp.clamp(-30.0, -1e-6);
    let p = clamped.exp();
    let odds = p / (1.0 - p + 1e-7);
    odds.max(1e-7).ln()
}

pub fn orpo_loss(
    chosen_logps: &[f32],
    rejected_logps: &[f32],
    sft_loss: f32,
    cfg: &OrpoConfig,
) -> RlhfResult<f32> {
    if chosen_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_logps.len() != rejected_logps.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_logps.len(),
            rejected: rejected_logps.len(),
        });
    }
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(RlhfError::InvalidLambda { lambda: cfg.lambda });
    }
    if !sft_loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }

    let odds_ratio_sum: f32 = chosen_logps
        .iter()
        .zip(rejected_logps.iter())
        .map(|(&clp, &rlp)| {
            let lo_c = log_odds(clp);
            let lo_r = log_odds(rlp);
            let log_ratio = lo_c - lo_r;
            -log_sigmoid_stable(log_ratio)
        })
        .sum();
    let odds_penalty = odds_ratio_sum / chosen_logps.len() as f32;
    let loss = sft_loss + cfg.lambda * odds_penalty;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

fn log_sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of [`log_odds`] w.r.t. its log-probability argument.
///
/// In the smooth interior, `log_odds(lp) = lp − ln(1 − p + ε)` with
/// `p = exp(lp)`, so `d/dlp = 1 + p / (1 − p + ε)`.
///
/// The forward applies `clamp(lp, −30, −1e-6)` and `max(odds, 1e-7)`; in either
/// saturated region the derivative is `0` (the output no longer depends on
/// `lp`). This is exactly the gradient the chain rule must respect.
#[must_use]
pub fn log_odds_grad(lp: f32) -> f32 {
    if lp <= -30.0 || lp >= -1e-6 {
        return 0.0; // clamp saturated
    }
    let p = lp.exp();
    let denom = 1.0 - p + 1e-7;
    let odds = p / denom;
    if odds <= 1e-7 {
        return 0.0; // max(odds, 1e-7) floor saturated
    }
    1.0 + p / denom
}

/// Gradient of the ORPO loss w.r.t. its inputs.
///
/// Finite-difference verified against [`orpo_loss`].
#[derive(Debug, Clone)]
pub struct OrpoGrad {
    /// `∂L/∂(chosen log-prob)` for each pair.
    pub d_chosen_logp: Vec<f32>,
    /// `∂L/∂(rejected log-prob)` for each pair.
    pub d_rejected_logp: Vec<f32>,
    /// `∂L/∂(SFT loss)` — the SFT term enters linearly, so this is always `1`.
    pub d_sft_loss: f32,
}

/// Analytic gradient of [`orpo_loss`].
///
/// `L = L_SFT + λ·mean_i(−log σ(δ_i))` with `δ_i = log_odds(c_i) − log_odds(r_i)`.
/// Per pair, `d(−log σ(δ))/dδ = −σ(−δ)`, and chaining through `log_odds`
/// (see [`log_odds_grad`]) gives
/// `∂L/∂c_i = (λ/N)·(−σ(−δ_i))·log_odds_grad(c_i)` and
/// `∂L/∂r_i = (λ/N)·(+σ(−δ_i))·log_odds_grad(r_i)`. The SFT term contributes
/// `∂L/∂L_SFT = 1`.
///
/// # Errors
/// Mirrors [`orpo_loss`]: shape / config validation plus
/// [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn orpo_grad(
    chosen_logps: &[f32],
    rejected_logps: &[f32],
    sft_loss: f32,
    cfg: &OrpoConfig,
) -> RlhfResult<OrpoGrad> {
    if chosen_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if chosen_logps.len() != rejected_logps.len() {
        return Err(RlhfError::MismatchedPairLength {
            chosen: chosen_logps.len(),
            rejected: rejected_logps.len(),
        });
    }
    if !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(RlhfError::InvalidLambda { lambda: cfg.lambda });
    }
    if !sft_loss.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    let n = chosen_logps.len();
    let scale = cfg.lambda / n as f32;
    let mut d_chosen_logp = Vec::with_capacity(n);
    let mut d_rejected_logp = Vec::with_capacity(n);
    for (&clp, &rlp) in chosen_logps.iter().zip(rejected_logps.iter()) {
        let lo_c = log_odds(clp);
        let lo_r = log_odds(rlp);
        let delta = lo_c - lo_r;
        // d(−log σ(δ))/dδ = −σ(−δ)
        let d_delta = -sigmoid(-delta);
        let dc = scale * d_delta * log_odds_grad(clp);
        let dr = scale * (-d_delta) * log_odds_grad(rlp);
        if !dc.is_finite() || !dr.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_chosen_logp.push(dc);
        d_rejected_logp.push(dr);
    }
    Ok(OrpoGrad {
        d_chosen_logp,
        d_rejected_logp,
        d_sft_loss: 1.0,
    })
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
            rel <= 2e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    #[test]
    fn log_odds_grad_matches_finite_difference() {
        let h = 1e-3;
        for &lp in &[-0.5_f32, -1.0, -2.0, -3.5] {
            let analytic = log_odds_grad(lp);
            let fd = central_diff(log_odds, lp, h);
            assert_close(analytic, fd, "log_odds_grad");
        }
    }

    #[test]
    fn log_odds_grad_zero_in_clamp_region() {
        // lp >= -1e-6 and lp <= -30 are clamp-saturated -> derivative 0.
        assert_eq!(log_odds_grad(0.0), 0.0);
        assert_eq!(log_odds_grad(-40.0), 0.0);
    }

    #[test]
    fn orpo_grad_matches_finite_difference() {
        let cfg = OrpoConfig { lambda: 0.5 };
        let c = [-0.8_f32, -1.5];
        let r = [-1.6_f32, -0.9];
        let sft = 2.0_f32;
        let g = orpo_grad(&c, &r, sft, &cfg).expect("grad");
        let h = 1e-2;
        for i in 0..c.len() {
            let fd_c = central_diff(
                |v| {
                    let mut cc = c.to_vec();
                    cc[i] = v;
                    orpo_loss(&cc, &r, sft, &cfg).expect("loss")
                },
                c[i],
                h,
            );
            let fd_r = central_diff(
                |v| {
                    let mut rr = r.to_vec();
                    rr[i] = v;
                    orpo_loss(&c, &rr, sft, &cfg).expect("loss")
                },
                r[i],
                h,
            );
            assert_close(g.d_chosen_logp[i], fd_c, "d_chosen_logp");
            assert_close(g.d_rejected_logp[i], fd_r, "d_rejected_logp");
        }
        // SFT term enters linearly.
        let fd_sft = central_diff(|v| orpo_loss(&c, &r, v, &cfg).expect("loss"), sft, h);
        assert_close(g.d_sft_loss, fd_sft, "d_sft_loss");
    }

    #[test]
    fn orpo_grad_prefers_chosen() {
        // Chosen has higher log-prob than rejected, so raising chosen / lowering
        // rejected reduces the odds-ratio penalty.
        let cfg = OrpoConfig { lambda: 1.0 };
        let g = orpo_grad(&[-0.5], &[-2.0], 1.0, &cfg).expect("grad");
        assert!(g.d_chosen_logp[0] < 0.0, "{}", g.d_chosen_logp[0]);
        assert!(g.d_rejected_logp[0] > 0.0, "{}", g.d_rejected_logp[0]);
    }

    #[test]
    fn orpo_grad_mismatch_errors() {
        let cfg = OrpoConfig { lambda: 0.5 };
        assert!(matches!(
            orpo_grad(&[-1.0, -2.0], &[-1.0], 1.0, &cfg),
            Err(RlhfError::MismatchedPairLength { .. })
        ));
    }
}
