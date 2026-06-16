use crate::error::{RlhfError, RlhfResult};

pub struct PpoConfig {
    pub clip_epsilon: f32,
    pub value_loss_coeff: f32,
    pub entropy_coeff: f32,
}

pub struct PpoLoss {
    config: PpoConfig,
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
}
