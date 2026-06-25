use crate::error::{RlhfError, RlhfResult};

pub struct SimpoConfig {
    pub beta: f32,
    pub gamma: f32,
}

pub fn simpo_loss(
    chosen_sum_logps: &[f32],
    rejected_sum_logps: &[f32],
    chosen_lengths: &[usize],
    rejected_lengths: &[usize],
    cfg: &SimpoConfig,
) -> RlhfResult<f32> {
    if chosen_sum_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let n = chosen_sum_logps.len();
    if rejected_sum_logps.len() != n {
        return Err(RlhfError::MismatchedPairLength {
            chosen: n,
            rejected: rejected_sum_logps.len(),
        });
    }
    if chosen_lengths.len() != n || rejected_lengths.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: chosen_lengths.len(),
        });
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.gamma.is_finite() {
        return Err(RlhfError::InvalidMargin { margin: cfg.gamma });
    }
    let total: f32 = chosen_sum_logps
        .iter()
        .zip(rejected_sum_logps.iter())
        .zip(chosen_lengths.iter())
        .zip(rejected_lengths.iter())
        .map(|(((&cslp, &rslp), &cl), &rl)| {
            let cl_f = cl.max(1) as f32;
            let rl_f = rl.max(1) as f32;
            let norm_c = cslp / cl_f;
            let norm_r = rslp / rl_f;
            let logit = cfg.beta * (norm_c - norm_r) - cfg.gamma;
            -log_sigmoid_stable(logit)
        })
        .sum();
    let loss = total / n as f32;
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

/// Gradient of the SimPO loss w.r.t. the summed log-probability inputs.
///
/// Finite-difference verified against [`simpo_loss`]. The sequence lengths are
/// integer constants, so only the chosen / rejected summed log-probs carry a
/// gradient.
#[derive(Debug, Clone)]
pub struct SimpoGrad {
    /// `∂L/∂(chosen summed log-prob)` for each pair.
    pub d_chosen_sum_logp: Vec<f32>,
    /// `∂L/∂(rejected summed log-prob)` for each pair.
    pub d_rejected_sum_logp: Vec<f32>,
}

/// Analytic gradient of the mean-reduced [`simpo_loss`].
///
/// Per pair `L = −log σ(z)` with
/// `z = β·(s_w/|y_w| − s_l/|y_l|) − γ`, so `dL/dz = −σ(−z)`. Chaining through
/// the length-normalised log-probs gives
/// `∂L/∂s_w = −σ(−z)·β/|y_w|` and `∂L/∂s_l = +σ(−z)·β/|y_l|`, each scaled by
/// `1 / N` for the mean reduction.
///
/// # Errors
/// Mirrors [`simpo_loss`]: shape / config validation, plus
/// [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn simpo_grad(
    chosen_sum_logps: &[f32],
    rejected_sum_logps: &[f32],
    chosen_lengths: &[usize],
    rejected_lengths: &[usize],
    cfg: &SimpoConfig,
) -> RlhfResult<SimpoGrad> {
    if chosen_sum_logps.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let n = chosen_sum_logps.len();
    if rejected_sum_logps.len() != n {
        return Err(RlhfError::MismatchedPairLength {
            chosen: n,
            rejected: rejected_sum_logps.len(),
        });
    }
    if chosen_lengths.len() != n || rejected_lengths.len() != n {
        return Err(RlhfError::DimensionMismatch {
            expected: n,
            got: chosen_lengths.len(),
        });
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.gamma.is_finite() {
        return Err(RlhfError::InvalidMargin { margin: cfg.gamma });
    }
    let inv_n = 1.0 / n as f32;
    let mut d_chosen_sum_logp = Vec::with_capacity(n);
    let mut d_rejected_sum_logp = Vec::with_capacity(n);
    for (((&cslp, &rslp), &cl), &rl) in chosen_sum_logps
        .iter()
        .zip(rejected_sum_logps.iter())
        .zip(chosen_lengths.iter())
        .zip(rejected_lengths.iter())
    {
        let cl_f = cl.max(1) as f32;
        let rl_f = rl.max(1) as f32;
        let logit = cfg.beta * (cslp / cl_f - rslp / rl_f) - cfg.gamma;
        // dL/dz = −σ(−z)
        let d_logit = -sigmoid(-logit);
        let dc = d_logit * cfg.beta / cl_f * inv_n;
        let dr = -d_logit * cfg.beta / rl_f * inv_n;
        if !dc.is_finite() || !dr.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        d_chosen_sum_logp.push(dc);
        d_rejected_sum_logp.push(dr);
    }
    Ok(SimpoGrad {
        d_chosen_sum_logp,
        d_rejected_sum_logp,
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
            rel <= 1e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    #[test]
    fn simpo_grad_matches_finite_difference() {
        let cfg = SimpoConfig {
            beta: 2.0,
            gamma: 0.5,
        };
        let cs = [-8.0_f32, -4.0];
        let rs = [-6.0_f32, -3.0];
        let cl = [8_usize, 5];
        let rl = [6_usize, 4];
        let g = simpo_grad(&cs, &rs, &cl, &rl, &cfg).expect("grad");
        let h = 1e-2;
        for i in 0..cs.len() {
            let fd_c = central_diff(
                |v| {
                    let mut c = cs.to_vec();
                    c[i] = v;
                    simpo_loss(&c, &rs, &cl, &rl, &cfg).expect("loss")
                },
                cs[i],
                h,
            );
            let fd_r = central_diff(
                |v| {
                    let mut r = rs.to_vec();
                    r[i] = v;
                    simpo_loss(&cs, &r, &cl, &rl, &cfg).expect("loss")
                },
                rs[i],
                h,
            );
            assert_close(g.d_chosen_sum_logp[i], fd_c, "d_chosen_sum_logp");
            assert_close(g.d_rejected_sum_logp[i], fd_r, "d_rejected_sum_logp");
        }
    }

    #[test]
    fn simpo_grad_raises_chosen_lowers_rejected() {
        let cfg = SimpoConfig {
            beta: 2.0,
            gamma: 0.5,
        };
        let g = simpo_grad(&[-8.0], &[-6.0], &[8], &[6], &cfg).expect("grad");
        assert!(g.d_chosen_sum_logp[0] < 0.0, "{}", g.d_chosen_sum_logp[0]);
        assert!(
            g.d_rejected_sum_logp[0] > 0.0,
            "{}",
            g.d_rejected_sum_logp[0]
        );
    }

    #[test]
    fn simpo_grad_mismatch_errors() {
        let cfg = SimpoConfig {
            beta: 2.0,
            gamma: 0.5,
        };
        assert!(simpo_grad(&[-8.0, -4.0], &[-6.0], &[8, 5], &[6, 4], &cfg).is_err());
    }
}
