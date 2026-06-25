use crate::error::{RlhfError, RlhfResult};

pub struct KtoConfig {
    pub beta: f32,
    pub lambda_d: f32,
    pub lambda_u: f32,
}

pub fn kto_loss(
    desirable_rewards: &[f32],
    undesirable_rewards: &[f32],
    cfg: &KtoConfig,
) -> RlhfResult<f32> {
    if desirable_rewards.is_empty() && undesirable_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.lambda_d.is_finite() || cfg.lambda_d < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_d,
        });
    }
    if !cfg.lambda_u.is_finite() || cfg.lambda_u < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_u,
        });
    }
    let z0 = std::f32::consts::LN_2;

    let desirable_loss = if desirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = desirable_rewards
            .iter()
            .map(|&r| {
                let arg = cfg.beta * (r - z0);
                1.0 - sigmoid(arg)
            })
            .sum();
        cfg.lambda_d * sum / desirable_rewards.len() as f32
    };

    let undesirable_loss = if undesirable_rewards.is_empty() {
        0.0_f32
    } else {
        let sum: f32 = undesirable_rewards
            .iter()
            .map(|&r| {
                let arg = cfg.beta * (z0 - r);
                1.0 - sigmoid(arg)
            })
            .sum();
        cfg.lambda_u * sum / undesirable_rewards.len() as f32
    };

    let loss = desirable_loss + undesirable_loss;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Gradient of the KTO loss w.r.t. the desirable / undesirable reward inputs.
///
/// Finite-difference verified against [`kto_loss`].
#[derive(Debug, Clone)]
pub struct KtoGrad {
    /// `∂L/∂r` for each desirable reward (same length / order as the input).
    pub d_desirable: Vec<f32>,
    /// `∂L/∂r` for each undesirable reward (same length / order as the input).
    pub d_undesirable: Vec<f32>,
}

/// Analytic gradient of [`kto_loss`] w.r.t. the per-example rewards.
///
/// For a desirable reward `r`, the contribution is
/// `λ_d/N_d · (1 − σ(a))` with `a = β(r − z₀)`, whose derivative is
/// `−λ_d·β·σ(a)(1 − σ(a)) / N_d` (negative: raising a desirable reward lowers
/// the loss).
///
/// For an undesirable reward `r`, the contribution is
/// `λ_u/N_u · (1 − σ(b))` with `b = β(z₀ − r)`, whose derivative is
/// `+λ_u·β·σ(b)(1 − σ(b)) / N_u` (positive: raising an undesirable reward
/// raises the loss).
///
/// # Errors
/// Mirrors [`kto_loss`]: [`RlhfError::EmptyInput`] when both sides are empty,
/// [`RlhfError::InvalidBeta`] / [`RlhfError::InvalidLambda`] for bad config,
/// and [`RlhfError::NanEncountered`] on a non-finite gradient.
pub fn kto_grad(
    desirable_rewards: &[f32],
    undesirable_rewards: &[f32],
    cfg: &KtoConfig,
) -> RlhfResult<KtoGrad> {
    if desirable_rewards.is_empty() && undesirable_rewards.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(RlhfError::InvalidBeta { beta: cfg.beta });
    }
    if !cfg.lambda_d.is_finite() || cfg.lambda_d < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_d,
        });
    }
    if !cfg.lambda_u.is_finite() || cfg.lambda_u < 0.0 {
        return Err(RlhfError::InvalidLambda {
            lambda: cfg.lambda_u,
        });
    }
    let z0 = std::f32::consts::LN_2;

    let d_desirable = if desirable_rewards.is_empty() {
        Vec::new()
    } else {
        let inv_n = 1.0 / desirable_rewards.len() as f32;
        desirable_rewards
            .iter()
            .map(|&r| {
                let s = sigmoid(cfg.beta * (r - z0));
                -cfg.lambda_d * cfg.beta * s * (1.0 - s) * inv_n
            })
            .collect()
    };

    let d_undesirable = if undesirable_rewards.is_empty() {
        Vec::new()
    } else {
        let inv_n = 1.0 / undesirable_rewards.len() as f32;
        undesirable_rewards
            .iter()
            .map(|&r| {
                let s = sigmoid(cfg.beta * (z0 - r));
                cfg.lambda_u * cfg.beta * s * (1.0 - s) * inv_n
            })
            .collect()
    };

    let grad = KtoGrad {
        d_desirable,
        d_undesirable,
    };
    if grad
        .d_desirable
        .iter()
        .chain(grad.d_undesirable.iter())
        .any(|g| !g.is_finite())
    {
        return Err(RlhfError::NanEncountered);
    }
    Ok(grad)
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
    fn kto_grad_matches_finite_difference() {
        let cfg = KtoConfig {
            beta: 0.5,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        let des = [0.9_f32, 0.2];
        let und = [-0.4_f32, 0.1];
        let g = kto_grad(&des, &und, &cfg).expect("grad");
        let h = 1e-2;

        for i in 0..des.len() {
            let fd = central_diff(
                |v| {
                    let mut d = des.to_vec();
                    d[i] = v;
                    kto_loss(&d, &und, &cfg).expect("loss")
                },
                des[i],
                h,
            );
            assert_close(g.d_desirable[i], fd, "d_desirable");
        }
        for i in 0..und.len() {
            let fd = central_diff(
                |v| {
                    let mut u = und.to_vec();
                    u[i] = v;
                    kto_loss(&des, &u, &cfg).expect("loss")
                },
                und[i],
                h,
            );
            assert_close(g.d_undesirable[i], fd, "d_undesirable");
        }
    }

    #[test]
    fn kto_grad_signs_are_aligned() {
        // Desirable rewards should be pushed up (negative gradient), undesirable
        // pushed down (positive gradient).
        let cfg = KtoConfig {
            beta: 0.3,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        let g = kto_grad(&[0.5], &[0.5], &cfg).expect("grad");
        assert!(g.d_desirable[0] < 0.0, "{}", g.d_desirable[0]);
        assert!(g.d_undesirable[0] > 0.0, "{}", g.d_undesirable[0]);
    }

    #[test]
    fn kto_grad_handles_empty_side() {
        let cfg = KtoConfig {
            beta: 0.3,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        let g = kto_grad(&[0.5, 0.2], &[], &cfg).expect("grad");
        assert_eq!(g.d_desirable.len(), 2);
        assert!(g.d_undesirable.is_empty());
    }

    #[test]
    fn kto_grad_empty_both_errors() {
        let cfg = KtoConfig {
            beta: 0.3,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        assert!(matches!(
            kto_grad(&[], &[], &cfg),
            Err(RlhfError::EmptyInput)
        ));
    }
}
