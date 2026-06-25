use crate::error::{RlhfError, RlhfResult};
use crate::preference::pair::PairBatch;

pub struct IpoConfig {
    pub beta: f32,
}

pub fn ipo_loss(batch: &PairBatch, cfg: &IpoConfig) -> RlhfResult<f32> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let target = 1.0 / (2.0 * cfg.beta);
    let total: f32 = batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
        .map(|(((&clp, &rlp), &rclp), &rrlp)| {
            let h = (clp - rclp) - (rlp - rrlp);
            let diff = h - target;
            diff * diff
        })
        .sum();
    let loss = total / batch.len() as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

/// Gradient of the per-pair IPO loss w.r.t. the four log-probability inputs.
///
/// Finite-difference verified against [`ipo_loss`].
#[derive(Debug, Clone, Copy)]
pub struct IpoGrad {
    /// `∂L/∂(policy chosen log-prob)`.
    pub d_chosen_logp: f32,
    /// `∂L/∂(policy rejected log-prob)`.
    pub d_rejected_logp: f32,
    /// `∂L/∂(reference chosen log-prob)`.
    pub d_ref_chosen_logp: f32,
    /// `∂L/∂(reference rejected log-prob)`.
    pub d_ref_rejected_logp: f32,
}

/// Analytic gradient of the mean-reduced [`ipo_loss`] over a batch.
///
/// Per pair `L = (h − τ)²` with `h = (lp_w − rlp_w) − (lp_l − rlp_l)` and
/// `τ = 1 / (2β)`. So `dL/dh = 2(h − τ)`, and chaining through `h` (partials
/// `+1, −1, −1, +1` for `lp_w, rlp_w, lp_l, rlp_l`) gives the four gradients,
/// each scaled by `1 / batch.len()` for the mean reduction.
///
/// # Errors
/// Returns [`RlhfError::EmptyInput`] for an empty batch,
/// [`RlhfError::InvalidBeta`] for an invalid `beta`, or
/// [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn ipo_grad(batch: &PairBatch, cfg: &IpoConfig) -> RlhfResult<Vec<IpoGrad>> {
    if batch.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    let target = 1.0 / (2.0 * cfg.beta);
    let inv_n = 1.0 / batch.len() as f32;
    let mut grads = Vec::with_capacity(batch.len());
    for (((&clp, &rlp), &rclp), &rrlp) in batch
        .chosen_logps
        .iter()
        .zip(batch.rejected_logps.iter())
        .zip(batch.ref_chosen_logps.iter())
        .zip(batch.ref_rejected_logps.iter())
    {
        let h = (clp - rclp) - (rlp - rrlp);
        // dL/dh = 2(h − τ), already mean-scaled.
        let g = 2.0 * (h - target) * inv_n;
        let grad = IpoGrad {
            d_chosen_logp: g,
            d_rejected_logp: -g,
            d_ref_chosen_logp: -g,
            d_ref_rejected_logp: g,
        };
        if !grad.d_chosen_logp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        grads.push(grad);
    }
    Ok(grads)
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

    fn make_batch(c: &[f32], r: &[f32], rc: &[f32], rr: &[f32]) -> PairBatch {
        PairBatch::new(c.to_vec(), r.to_vec(), rc.to_vec(), rr.to_vec()).expect("batch")
    }

    #[test]
    fn ipo_grad_matches_finite_difference() {
        let cfg = IpoConfig { beta: 0.4 };
        let c = [-0.6_f32, -1.3];
        let r = [-1.4_f32, -0.7];
        let rc = [-0.8_f32, -1.0];
        let rr = [-1.2_f32, -0.9];
        let grads = ipo_grad(&make_batch(&c, &r, &rc, &rr), &cfg).expect("grads");
        let h = 1e-2;
        let idx = 1usize;

        let fd_c = central_diff(
            |v| {
                let mut cc = c.to_vec();
                cc[idx] = v;
                ipo_loss(&make_batch(&cc, &r, &rc, &rr), &cfg).expect("loss")
            },
            c[idx],
            h,
        );
        let fd_r = central_diff(
            |v| {
                let mut rr2 = r.to_vec();
                rr2[idx] = v;
                ipo_loss(&make_batch(&c, &rr2, &rc, &rr), &cfg).expect("loss")
            },
            r[idx],
            h,
        );
        let fd_rc = central_diff(
            |v| {
                let mut rc2 = rc.to_vec();
                rc2[idx] = v;
                ipo_loss(&make_batch(&c, &r, &rc2, &rr), &cfg).expect("loss")
            },
            rc[idx],
            h,
        );
        let fd_rr = central_diff(
            |v| {
                let mut rr2 = rr.to_vec();
                rr2[idx] = v;
                ipo_loss(&make_batch(&c, &r, &rc, &rr2), &cfg).expect("loss")
            },
            rr[idx],
            h,
        );
        assert_close(grads[idx].d_chosen_logp, fd_c, "d_chosen_logp");
        assert_close(grads[idx].d_rejected_logp, fd_r, "d_rejected_logp");
        assert_close(grads[idx].d_ref_chosen_logp, fd_rc, "d_ref_chosen_logp");
        assert_close(grads[idx].d_ref_rejected_logp, fd_rr, "d_ref_rejected_logp");
    }

    #[test]
    fn ipo_grad_pushes_h_toward_target() {
        // With h below the target τ = 1/(2β), dL/dh < 0, so the chosen-log-prob
        // gradient is negative (descent raises h toward τ).
        let cfg = IpoConfig { beta: 0.4 }; // τ = 1.25
        let grads = ipo_grad(
            &make_batch(&[-1.0], &[-1.0], &[-1.0], &[-1.0]),
            &cfg, // h = 0 < 1.25
        )
        .expect("grads");
        assert!(grads[0].d_chosen_logp < 0.0, "{}", grads[0].d_chosen_logp);
        assert!(
            grads[0].d_rejected_logp > 0.0,
            "{}",
            grads[0].d_rejected_logp
        );
    }

    #[test]
    fn ipo_grad_invalid_beta_errors() {
        let cfg = IpoConfig { beta: -1.0 };
        assert!(matches!(
            ipo_grad(&make_batch(&[-1.0], &[-1.0], &[-1.0], &[-1.0]), &cfg),
            Err(RlhfError::InvalidBeta { .. })
        ));
    }
}
