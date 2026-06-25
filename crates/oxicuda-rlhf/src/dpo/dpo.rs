use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::{PairBatch, PreferencePair};

pub struct DpoConfig {
    pub beta: f32,
}

pub fn dpo_log_ratio(
    chosen_lp: f32,
    ref_chosen_lp: f32,
    rejected_lp: f32,
    ref_rejected_lp: f32,
    beta: f32,
) -> f32 {
    let log_ratio_chosen = chosen_lp - ref_chosen_lp;
    let log_ratio_rejected = rejected_lp - ref_rejected_lp;
    beta * (log_ratio_chosen - log_ratio_rejected)
}

pub fn dpo_loss_per_pair(pair: &PreferencePair, cfg: &DpoConfig) -> RlhfResult<f32> {
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let logit = dpo_log_ratio(
        pair.chosen_logp,
        pair.ref_chosen_logp,
        pair.rejected_logp,
        pair.ref_rejected_logp,
        cfg.beta,
    );
    let loss = -log_sigmoid(logit);
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

pub fn dpo_loss(batch: &PairBatch, cfg: &DpoConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let total: f32 = batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let logit = dpo_log_ratio(clp, rclp, rlp, rrlp, cfg.beta);
            -log_sigmoid(logit)
        })
        .sum();
    let loss = total / batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Gradient of the per-pair DPO loss w.r.t. the four log-probability inputs.
///
/// All four partials are finite-difference verified against
/// [`dpo_loss_per_pair`].
#[derive(Debug, Clone, Copy)]
pub struct DpoGrad {
    /// `∂L/∂(policy chosen log-prob)`.
    pub d_chosen_logp: f32,
    /// `∂L/∂(policy rejected log-prob)`.
    pub d_rejected_logp: f32,
    /// `∂L/∂(reference chosen log-prob)`.
    pub d_ref_chosen_logp: f32,
    /// `∂L/∂(reference rejected log-prob)`.
    pub d_ref_rejected_logp: f32,
}

/// Analytic gradient of [`dpo_loss_per_pair`].
///
/// With `L = -log σ(β·Δ)` and `Δ = (lp_w − rlp_w) − (lp_l − rlp_l)`,
/// `dL/dΔ = −β·σ(−β·Δ)`. Chaining through `Δ` (whose partials are
/// `+1, −1, −1, +1` for `lp_w, rlp_w, lp_l, rlp_l`) yields the four gradients.
///
/// # Errors
/// Returns [`RlhfError::InvalidBeta`] for a non-positive / non-finite `beta`,
/// or [`RlhfError::NanEncountered`] if the gradient is not finite.
pub fn dpo_grad_per_pair(pair: &PreferencePair, cfg: &DpoConfig) -> RlhfResult<DpoGrad> {
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let grad = dpo_pair_grad_inner(
        pair.chosen_logp,
        pair.ref_chosen_logp,
        pair.rejected_logp,
        pair.ref_rejected_logp,
        cfg.beta,
    );
    if !grad.d_chosen_logp.is_finite() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(grad)
}

/// Analytic gradient of the mean-reduced [`dpo_loss`] over a batch.
///
/// Returns one [`DpoGrad`] per pair; each holds the gradient of the *mean*
/// batch loss, i.e. the per-pair gradient scaled by `1 / batch.len()`.
///
/// # Errors
/// Returns [`RlhfError::EmptyInput`] for an empty batch,
/// [`RlhfError::InvalidBeta`] for an invalid `beta`, or
/// [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn dpo_grad(batch: &PairBatch, cfg: &DpoConfig) -> RlhfResult<Vec<DpoGrad>> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let inv_n = 1.0 / batch.len() as f32;
    let mut grads = Vec::with_capacity(batch.len());
    for (((&clp, &rlp), &rclp), &rrlp) in batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
    {
        let g = dpo_pair_grad_inner(clp, rclp, rlp, rrlp, cfg.beta);
        let scaled = DpoGrad {
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

#[inline]
fn dpo_pair_grad_inner(clp: f32, rclp: f32, rlp: f32, rrlp: f32, beta: f32) -> DpoGrad {
    let delta = (clp - rclp) - (rlp - rrlp);
    let logit = beta * delta;
    // dL/dΔ = −β·σ(−β·Δ)
    let d_delta = -beta * sigmoid(-logit);
    DpoGrad {
        d_chosen_logp: d_delta,
        d_rejected_logp: -d_delta,
        d_ref_chosen_logp: -d_delta,
        d_ref_rejected_logp: d_delta,
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

fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod grad_tests {
    use super::*;

    /// Central finite difference of a scalar→scalar loss, evaluated in f64 so
    /// the divided difference itself adds no extra f32 rounding.
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

    fn make(c: f32, r: f32, rc: f32, rr: f32) -> PreferencePair {
        PreferencePair {
            chosen_logp: c,
            rejected_logp: r,
            ref_chosen_logp: rc,
            ref_rejected_logp: rr,
        }
    }

    #[test]
    fn dpo_grad_per_pair_matches_finite_difference() {
        let cfg = DpoConfig { beta: 0.3 };
        let (c0, r0, rc0, rr0) = (-0.7_f32, -1.4, -0.9, -1.1);
        let g = dpo_grad_per_pair(&make(c0, r0, rc0, rr0), &cfg).expect("grad");
        let h = 1e-2;

        let fd_c = central_diff(
            |v| dpo_loss_per_pair(&make(v, r0, rc0, rr0), &cfg).expect("loss"),
            c0,
            h,
        );
        let fd_r = central_diff(
            |v| dpo_loss_per_pair(&make(c0, v, rc0, rr0), &cfg).expect("loss"),
            r0,
            h,
        );
        let fd_rc = central_diff(
            |v| dpo_loss_per_pair(&make(c0, r0, v, rr0), &cfg).expect("loss"),
            rc0,
            h,
        );
        let fd_rr = central_diff(
            |v| dpo_loss_per_pair(&make(c0, r0, rc0, v), &cfg).expect("loss"),
            rr0,
            h,
        );
        assert_close(g.d_chosen_logp, fd_c, "d_chosen_logp");
        assert_close(g.d_rejected_logp, fd_r, "d_rejected_logp");
        assert_close(g.d_ref_chosen_logp, fd_rc, "d_ref_chosen_logp");
        assert_close(g.d_ref_rejected_logp, fd_rr, "d_ref_rejected_logp");
    }

    #[test]
    fn dpo_grad_pushes_margin_up() {
        // Increasing the chosen-vs-rejected margin must lower the loss, so the
        // gradient on the chosen log-prob is negative and on the rejected
        // log-prob is positive (gradient descent raises chosen, lowers rejected).
        let cfg = DpoConfig { beta: 0.5 };
        let g = dpo_grad_per_pair(&make(-1.0, -1.0, -1.0, -1.0), &cfg).expect("grad");
        assert!(g.d_chosen_logp < 0.0, "d_chosen={}", g.d_chosen_logp);
        assert!(g.d_rejected_logp > 0.0, "d_rejected={}", g.d_rejected_logp);
        // Reference partials mirror the policy partials.
        assert!((g.d_ref_chosen_logp + g.d_chosen_logp).abs() < 1e-7);
        assert!((g.d_ref_rejected_logp + g.d_rejected_logp).abs() < 1e-7);
    }

    #[test]
    fn dpo_grad_batch_matches_finite_difference() {
        let cfg = DpoConfig { beta: 0.5 };
        let batch = PairBatch::new(
            vec![-0.5_f32, -1.2],
            vec![-1.5_f32, -0.8],
            vec![-0.6_f32, -1.0],
            vec![-1.1_f32, -0.9],
        )
        .expect("batch");
        let grads = dpo_grad(&batch, &cfg).expect("grads");
        assert_eq!(grads.len(), 2);
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
                dpo_loss(&b, &cfg).expect("loss")
            },
            batch.chosen_logps[1],
            h,
        );
        // Batch gradient is the per-pair gradient scaled by 1/N.
        assert_close(grads[1].d_chosen_logp, fd, "batch d_chosen[1]");
        for g in &grads {
            assert!(g.d_chosen_logp.is_finite());
        }
    }

    #[test]
    fn dpo_grad_is_deterministic() {
        let cfg = DpoConfig { beta: 0.2 };
        let p = make(-0.3, -0.9, -0.4, -0.7);
        let a = dpo_grad_per_pair(&p, &cfg).expect("a");
        let b = dpo_grad_per_pair(&p, &cfg).expect("b");
        assert_eq!(a.d_chosen_logp, b.d_chosen_logp);
        assert_eq!(a.d_rejected_logp, b.d_rejected_logp);
    }

    #[test]
    fn dpo_grad_invalid_beta_errors() {
        let cfg = DpoConfig { beta: 0.0 };
        assert!(matches!(
            dpo_grad_per_pair(&make(-1.0, -1.0, -1.0, -1.0), &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}
