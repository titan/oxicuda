use crate::error::{RlhfError, RlhfResult};

pub struct PpoConfig {
    pub clip_epsilon: f32,
    pub value_loss_coeff: f32,
    pub entropy_coeff: f32,
}

pub struct PpoLoss {
    config: PpoConfig,
}

/// Gradients of [`PpoLoss::total_loss`] w.r.t. its trainable inputs.
///
/// Finite-difference verified against [`PpoLoss::total_loss`].
#[derive(Debug, Clone)]
pub struct PpoGrad {
    /// `∂(total)/∂ratio_i` — clip-aware (zero in the clipped-and-binding region).
    pub d_ratios: Vec<f32>,
    /// `∂(total)/∂value_i` = `value_loss_coeff · (v_i − ret_i) / N`.
    pub d_values: Vec<f32>,
    /// `∂(total)/∂entropy` = `−entropy_coeff`.
    pub d_entropy: f32,
}

impl PpoLoss {
    pub fn new(config: PpoConfig) -> RlhfResult<Self> {
        if !config.clip_epsilon.is_finite()
            || config.clip_epsilon <= 0.0
            || config.clip_epsilon >= 1.0
        {
            return Err(RlhfError::Internal {
                msg: format!(
                    "clip_epsilon must be in (0, 1), got {}",
                    config.clip_epsilon
                ),
            });
        }
        if !config.value_loss_coeff.is_finite() || config.value_loss_coeff < 0.0 {
            return Err(RlhfError::Internal {
                msg: format!(
                    "value_loss_coeff must be finite and >= 0, got {}",
                    config.value_loss_coeff
                ),
            });
        }
        if !config.entropy_coeff.is_finite() || config.entropy_coeff < 0.0 {
            return Err(RlhfError::Internal {
                msg: format!(
                    "entropy_coeff must be finite and >= 0, got {}",
                    config.entropy_coeff
                ),
            });
        }
        Ok(Self { config })
    }

    pub fn policy_loss(&self, ratios: &[f32], advantages: &[f32]) -> RlhfResult<f32> {
        if ratios.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if ratios.len() != advantages.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: ratios.len(),
                got: advantages.len(),
            });
        }
        let eps = self.config.clip_epsilon;
        let lo = 1.0 - eps;
        let hi = 1.0 + eps;

        let mut surrogate_sum = 0.0f32;
        for (&r, &a) in ratios.iter().zip(advantages.iter()) {
            if !r.is_finite() || !a.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            let clipped = r.clamp(lo, hi);
            let s = (r * a).min(clipped * a);
            surrogate_sum += s;
        }
        let loss = -(surrogate_sum / ratios.len() as f32);
        if !loss.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(loss)
    }

    pub fn value_loss(&self, values: &[f32], returns: &[f32]) -> RlhfResult<f32> {
        if values.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if values.len() != returns.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: values.len(),
                got: returns.len(),
            });
        }
        let mut mse_sum = 0.0f32;
        for (&v, &ret) in values.iter().zip(returns.iter()) {
            if !v.is_finite() || !ret.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            let diff = v - ret;
            mse_sum += diff * diff;
        }
        let loss = 0.5 * mse_sum / values.len() as f32;
        Ok(loss)
    }

    pub fn total_loss(
        &self,
        ratios: &[f32],
        advantages: &[f32],
        values: &[f32],
        returns: &[f32],
        entropy: f32,
    ) -> RlhfResult<f32> {
        let pl = self.policy_loss(ratios, advantages)?;
        let vl = self.value_loss(values, returns)?;
        Ok(pl - self.config.entropy_coeff * entropy + self.config.value_loss_coeff * vl)
    }

    pub fn clip_epsilon(&self) -> f32 {
        self.config.clip_epsilon
    }

    /// Analytic gradient of [`PpoLoss::policy_loss`] w.r.t. each ratio.
    ///
    /// The per-sample surrogate is `min(r·A, clamp(r, 1−ε, 1+ε)·A)` and the
    /// loss is `−mean(surrogate)`. The surrogate's derivative w.r.t. `r` is `A`
    /// when the unclipped branch is selected (or in the clamp interior) and `0`
    /// when the clipped branch binds in a saturated region — i.e. zero gradient
    /// exactly where the clip is active and binding. The loss gradient is then
    /// `−(1/N)·∂surrogate/∂r`.
    ///
    /// To obtain the gradient w.r.t. the new log-prob, multiply element-wise by
    /// `r` (since `r = exp(logπ − logπ_old)` gives `∂r/∂logπ = r`).
    ///
    /// # Errors
    /// Mirrors [`PpoLoss::policy_loss`].
    pub fn policy_grad(&self, ratios: &[f32], advantages: &[f32]) -> RlhfResult<Vec<f32>> {
        if ratios.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if ratios.len() != advantages.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: ratios.len(),
                got: advantages.len(),
            });
        }
        let eps = self.config.clip_epsilon;
        let lo = 1.0 - eps;
        let hi = 1.0 + eps;
        let inv_n = 1.0 / ratios.len() as f32;

        let mut grads = Vec::with_capacity(ratios.len());
        for (&r, &a) in ratios.iter().zip(advantages.iter()) {
            if !r.is_finite() || !a.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            let unclipped = r * a;
            let clipped = r.clamp(lo, hi) * a;
            // min picks unclipped (or interior tie) -> dsurrogate/dr = a;
            // otherwise the clamp is saturated -> 0.
            let d_surrogate = if unclipped <= clipped { a } else { 0.0 };
            grads.push(-d_surrogate * inv_n);
        }
        Ok(grads)
    }

    /// Analytic gradient of [`PpoLoss::value_loss`] w.r.t. each predicted value.
    ///
    /// `value_loss = 0.5·mean((v − ret)²)`, so `∂/∂v_i = (v_i − ret_i) / N`.
    ///
    /// # Errors
    /// Mirrors [`PpoLoss::value_loss`].
    pub fn value_grad(&self, values: &[f32], returns: &[f32]) -> RlhfResult<Vec<f32>> {
        if values.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if values.len() != returns.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: values.len(),
                got: returns.len(),
            });
        }
        let inv_n = 1.0 / values.len() as f32;
        let mut grads = Vec::with_capacity(values.len());
        for (&v, &ret) in values.iter().zip(returns.iter()) {
            if !v.is_finite() || !ret.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            grads.push((v - ret) * inv_n);
        }
        Ok(grads)
    }

    /// Gradient of [`PpoLoss::total_loss`] w.r.t. the (scalar) entropy input.
    ///
    /// The entropy bonus appears as `−entropy_coeff·entropy`, so the gradient is
    /// simply `−entropy_coeff`.
    #[must_use]
    pub fn entropy_grad(&self) -> f32 {
        -self.config.entropy_coeff
    }

    /// Combined analytic gradient of [`PpoLoss::total_loss`] w.r.t. ratios,
    /// values, and the entropy scalar.
    ///
    /// `total = policy_loss − entropy_coeff·entropy + value_loss_coeff·value_loss`,
    /// so the ratio gradient equals [`PpoLoss::policy_grad`], the value gradient
    /// is `value_loss_coeff ·` [`PpoLoss::value_grad`], and the entropy gradient
    /// is [`PpoLoss::entropy_grad`].
    ///
    /// # Errors
    /// Mirrors [`PpoLoss::total_loss`].
    pub fn total_grad(
        &self,
        ratios: &[f32],
        advantages: &[f32],
        values: &[f32],
        returns: &[f32],
    ) -> RlhfResult<PpoGrad> {
        let d_ratios = self.policy_grad(ratios, advantages)?;
        let d_values_raw = self.value_grad(values, returns)?;
        let vc = self.config.value_loss_coeff;
        let d_values = d_values_raw.iter().map(|&g| g * vc).collect();
        Ok(PpoGrad {
            d_ratios,
            d_values,
            d_entropy: self.entropy_grad(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ppo(eps: f32) -> PpoLoss {
        PpoLoss::new(PpoConfig {
            clip_epsilon: eps,
            value_loss_coeff: 0.5,
            entropy_coeff: 0.01,
        })
        .expect("valid config")
    }

    #[test]
    fn policy_loss_finite() {
        let ppo = make_ppo(0.2);
        let ratios = vec![1.0f32, 1.1, 0.9, 1.05];
        let advantages = vec![0.5f32, 1.0, -0.3, 0.8];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        assert!(loss.is_finite());
    }

    #[test]
    fn value_loss_nonneg() {
        let ppo = make_ppo(0.2);
        let values = vec![1.0f32, 2.0, 3.0];
        let returns = vec![1.5f32, 2.5, 3.5];
        let loss = ppo.value_loss(&values, &returns).expect("ok");
        assert!(loss >= 0.0, "value loss must be >= 0, got {loss}");
    }

    #[test]
    fn ratio_1_unchanged_loss() {
        // ratio=1 everywhere → policy_loss = -mean(advantages)
        let ppo = make_ppo(0.2);
        let advantages = vec![0.5f32, 1.0, -0.5, 0.0];
        let ratios = vec![1.0f32; 4];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        let expected = -advantages.iter().sum::<f32>() / advantages.len() as f32;
        assert!(
            (loss - expected).abs() < 1e-5,
            "expected {expected}, got {loss}"
        );
    }

    #[test]
    fn ratio_clipped_above() {
        // ratio=2.0, eps=0.2, A=1.0: clipped=1.2, surrogate=min(2.0, 1.2)=1.2, loss=-1.2
        let ppo = make_ppo(0.2);
        let ratios = vec![2.0f32];
        let advantages = vec![1.0f32];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        assert!((loss - (-1.2f32)).abs() < 1e-5, "expected -1.2, got {loss}");
    }

    #[test]
    fn ratio_clipped_below() {
        // ratio=0.5, eps=0.2, A=-1.0:
        // unclipped: 0.5 * -1.0 = -0.5
        // clipped ratio = clamp(0.5, 0.8, 1.2) = 0.8; clipped * A = 0.8 * -1.0 = -0.8
        // surrogate = min(-0.5, -0.8) = -0.8
        // loss = -(-0.8) = 0.8
        let ppo = make_ppo(0.2);
        let ratios = vec![0.5f32];
        let advantages = vec![-1.0f32];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        assert!((loss - 0.8f32).abs() < 1e-5, "expected 0.8, got {loss}");
    }

    #[test]
    fn negative_advantage_inverts() {
        // ratio > 1 with negative advantage: clipping reduces magnitude (makes it less negative)
        let ppo = make_ppo(0.2);
        // ratio=1.5, A=-1.0, clipped=1.2
        // unclipped: 1.5 * -1.0 = -1.5
        // clipped: 1.2 * -1.0 = -1.2
        // surrogate = min(-1.5, -1.2) = -1.5
        // Actually min(-1.5, -1.2) = -1.5 since -1.5 < -1.2
        // loss = -(-1.5) = 1.5 (unclipped wins here)
        // The point is clipping prevents exploiting large ratio updates
        let ratios = vec![1.5f32];
        let advantages = vec![-1.0f32];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        // min(1.5*-1, 1.2*-1) = min(-1.5, -1.2) = -1.5, loss = 1.5
        assert!((loss - 1.5f32).abs() < 1e-5, "expected 1.5, got {loss}");
    }

    #[test]
    fn total_loss_components() {
        let ppo = PpoLoss::new(PpoConfig {
            clip_epsilon: 0.2,
            value_loss_coeff: 0.5,
            entropy_coeff: 0.01,
        })
        .expect("valid");
        let ratios = vec![1.0f32; 4];
        let advantages = vec![1.0f32; 4];
        let values = vec![1.0f32; 4];
        let returns = vec![1.5f32; 4];
        let entropy = 0.5f32;

        let pl = ppo.policy_loss(&ratios, &advantages).expect("ok");
        let vl = ppo.value_loss(&values, &returns).expect("ok");
        let total = ppo
            .total_loss(&ratios, &advantages, &values, &returns, entropy)
            .expect("ok");
        let expected = pl - 0.01 * entropy + 0.5 * vl;
        assert!(
            (total - expected).abs() < 1e-5,
            "total={total}, expected={expected}"
        );
    }

    #[test]
    fn clip_0_equals_vanilla_pg() {
        // With very small epsilon ≈ 0 (but > 0): clipped_ratio ≈ ratio → surrogate = ratio*A
        // Use eps=1e-6 to approximate clip_epsilon→0
        let ppo = PpoLoss::new(PpoConfig {
            clip_epsilon: 1e-6,
            value_loss_coeff: 0.0,
            entropy_coeff: 0.0,
        })
        .expect("valid");
        let ratios = vec![1.0f32]; // ratio=1: no clipping either way
        let advantages = vec![2.0f32];
        let loss = ppo.policy_loss(&ratios, &advantages).expect("ok");
        // surrogate = min(1.0*2.0, clamp(1.0, 1-eps, 1+eps)*2.0) = min(2.0, 2.0) = 2.0
        // loss = -2.0
        assert!((loss - (-2.0f32)).abs() < 1e-4, "expected -2.0, got {loss}");
    }

    #[test]
    fn batch_size_mismatch_error() {
        let ppo = make_ppo(0.2);
        let result = ppo.policy_loss(&[1.0, 1.0], &[0.5]);
        assert!(result.is_err(), "should return Err on size mismatch");
    }

    #[test]
    fn entropy_coeff_changes_total() {
        let ppo1 = PpoLoss::new(PpoConfig {
            clip_epsilon: 0.2,
            value_loss_coeff: 0.5,
            entropy_coeff: 0.01,
        })
        .expect("valid");
        let ppo2 = PpoLoss::new(PpoConfig {
            clip_epsilon: 0.2,
            value_loss_coeff: 0.5,
            entropy_coeff: 0.1,
        })
        .expect("valid");
        let ratios = vec![1.0f32; 4];
        let advantages = vec![1.0f32; 4];
        let values = vec![1.0f32; 4];
        let returns = vec![1.5f32; 4];
        let entropy = 1.0f32;
        let t1 = ppo1
            .total_loss(&ratios, &advantages, &values, &returns, entropy)
            .expect("ok");
        let t2 = ppo2
            .total_loss(&ratios, &advantages, &values, &returns, entropy)
            .expect("ok");
        assert!(
            (t1 - t2).abs() > 1e-5,
            "different entropy_coeff should give different total: {t1} vs {t2}"
        );
    }

    // ── Gradient tests ────────────────────────────────────────────────────────

    fn central_diff(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        ((f(x + h) as f64 - f(x - h) as f64) / (2.0 * h as f64)) as f32
    }

    fn assert_grad_close(analytic: f32, fd: f32, label: &str) {
        let denom = analytic.abs().max(1e-3);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel <= 1e-3,
            "{label}: analytic={analytic}, fd={fd}, rel_err={rel}"
        );
    }

    #[test]
    fn policy_grad_matches_finite_difference_interior() {
        // All ratios in the clip interior -> gradient is -A/N everywhere.
        let ppo = make_ppo(0.2);
        let ratios = [1.05_f32, 0.95, 1.1, 0.9];
        let adv = [0.5_f32, 1.0, -0.3, 0.8];
        let g = ppo.policy_grad(&ratios, &adv).expect("grad");
        let h = 1e-3;
        for i in 0..ratios.len() {
            let fd = central_diff(
                |v| {
                    let mut x = ratios.to_vec();
                    x[i] = v;
                    ppo.policy_loss(&x, &adv).expect("loss")
                },
                ratios[i],
                h,
            );
            assert_grad_close(g[i], fd, "policy_grad interior");
            // interior gradient equals -A/N
            assert_grad_close(
                g[i],
                -adv[i] / ratios.len() as f32,
                "policy_grad closed form",
            );
        }
    }

    #[test]
    fn policy_grad_zero_in_clipped_binding_region() {
        // r=2.0, A>0, eps=0.2: clipped (1.2) binds and is the min -> grad 0.
        // r=0.4, A<0, eps=0.2: clipped (0.8) binds and is the min -> grad 0.
        let ppo = make_ppo(0.2);
        let ratios = [2.0_f32, 0.4];
        let adv = [1.0_f32, -1.0];
        let g = ppo.policy_grad(&ratios, &adv).expect("grad");
        assert_eq!(g[0], 0.0, "clipped-above binding gradient must be 0");
        assert_eq!(g[1], 0.0, "clipped-below binding gradient must be 0");
        // Finite difference is also exactly 0 in the flat clipped region.
        let h = 1e-3;
        for i in 0..ratios.len() {
            let fd = central_diff(
                |v| {
                    let mut x = ratios.to_vec();
                    x[i] = v;
                    ppo.policy_loss(&x, &adv).expect("loss")
                },
                ratios[i],
                h,
            );
            assert!(fd.abs() < 1e-6, "fd in clipped region = {fd}");
        }
    }

    #[test]
    fn policy_grad_nonzero_when_clip_not_binding() {
        // r=1.5, A=-1.0, eps=0.2: min(-1.5, -1.2) = -1.5 unclipped -> grad = -A/N.
        let ppo = make_ppo(0.2);
        let ratios = [1.5_f32];
        let adv = [-1.0_f32];
        let g = ppo.policy_grad(&ratios, &adv).expect("grad");
        let h = 1e-3;
        let fd = central_diff(|v| ppo.policy_loss(&[v], &adv).expect("loss"), ratios[0], h);
        assert_grad_close(g[0], fd, "policy_grad non-binding");
        assert!((g[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", g[0]);
    }

    #[test]
    fn value_grad_matches_finite_difference() {
        let ppo = make_ppo(0.2);
        let values = [1.0_f32, 2.5, -0.5];
        let returns = [1.5_f32, 2.0, 0.3];
        let g = ppo.value_grad(&values, &returns).expect("grad");
        let h = 1e-3;
        for i in 0..values.len() {
            let fd = central_diff(
                |v| {
                    let mut x = values.to_vec();
                    x[i] = v;
                    ppo.value_loss(&x, &returns).expect("loss")
                },
                values[i],
                h,
            );
            assert_grad_close(g[i], fd, "value_grad");
            assert_grad_close(
                g[i],
                (values[i] - returns[i]) / values.len() as f32,
                "value_grad closed form",
            );
        }
    }

    #[test]
    fn entropy_grad_matches_finite_difference() {
        let ppo = make_ppo(0.2); // entropy_coeff = 0.01
        let ratios = [1.0_f32; 3];
        let adv = [0.5_f32; 3];
        let values = [1.0_f32; 3];
        let returns = [1.2_f32; 3];
        let h = 1e-2;
        let fd = central_diff(
            |e| {
                ppo.total_loss(&ratios, &adv, &values, &returns, e)
                    .expect("loss")
            },
            0.5,
            h,
        );
        assert_grad_close(ppo.entropy_grad(), fd, "entropy_grad");
        assert!((ppo.entropy_grad() - (-0.01)).abs() < 1e-7);
    }

    #[test]
    fn total_grad_matches_finite_difference() {
        let ppo = make_ppo(0.2); // value_coeff=0.5, entropy_coeff=0.01
        let ratios = [1.05_f32, 0.9];
        let adv = [0.7_f32, -0.4];
        let values = [1.0_f32, 2.0];
        let returns = [1.5_f32, 1.0];
        let entropy = 0.5_f32;
        let g = ppo
            .total_grad(&ratios, &adv, &values, &returns)
            .expect("grad");
        let h = 1e-3;
        // d total / d ratio_i
        for i in 0..ratios.len() {
            let fd = central_diff(
                |v| {
                    let mut x = ratios.to_vec();
                    x[i] = v;
                    ppo.total_loss(&x, &adv, &values, &returns, entropy)
                        .expect("loss")
                },
                ratios[i],
                h,
            );
            assert_grad_close(g.d_ratios[i], fd, "total d_ratio");
        }
        // d total / d value_i
        for i in 0..values.len() {
            let fd = central_diff(
                |v| {
                    let mut x = values.to_vec();
                    x[i] = v;
                    ppo.total_loss(&ratios, &adv, &x, &returns, entropy)
                        .expect("loss")
                },
                values[i],
                h,
            );
            assert_grad_close(g.d_values[i], fd, "total d_value");
        }
        // d total / d entropy
        let fd_e = central_diff(
            |e| {
                ppo.total_loss(&ratios, &adv, &values, &returns, e)
                    .expect("loss")
            },
            entropy,
            1e-2,
        );
        assert_grad_close(g.d_entropy, fd_e, "total d_entropy");
    }

    #[test]
    fn grad_size_mismatch_errors() {
        let ppo = make_ppo(0.2);
        assert!(ppo.policy_grad(&[1.0, 1.0], &[0.5]).is_err());
        assert!(ppo.value_grad(&[1.0], &[0.5, 0.5]).is_err());
    }
}
