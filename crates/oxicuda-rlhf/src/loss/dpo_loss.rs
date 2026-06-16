use crate::error::{RlhfError, RlhfResult};

pub struct DpoConfig {
    pub beta: f32,
    pub label_smoothing: f32,
}

pub struct DpoLoss {
    config: DpoConfig,
}

fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

impl DpoLoss {
    pub fn new(config: DpoConfig) -> RlhfResult<Self> {
        if !config.beta.is_finite() || config.beta < 0.0 {
            return Err(RlhfError::InvalidBeta { beta: config.beta });
        }
        if !config.label_smoothing.is_finite()
            || config.label_smoothing < 0.0
            || config.label_smoothing >= 0.5
        {
            return Err(RlhfError::Internal {
                msg: format!(
                    "label_smoothing must be in [0.0, 0.5), got {}",
                    config.label_smoothing
                ),
            });
        }
        Ok(Self { config })
    }

    pub fn compute(
        &self,
        log_prob_w: &[f32],
        log_prob_l: &[f32],
        ref_log_prob_w: &[f32],
        ref_log_prob_l: &[f32],
        batch_size: usize,
    ) -> RlhfResult<f32> {
        if log_prob_w.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        let n = log_prob_w.len();
        if log_prob_l.len() != n || ref_log_prob_w.len() != n || ref_log_prob_l.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: log_prob_l
                    .len()
                    .min(ref_log_prob_w.len())
                    .min(ref_log_prob_l.len()),
            });
        }
        if batch_size == 0 {
            return Err(RlhfError::Internal {
                msg: "batch_size must be > 0".to_string(),
            });
        }

        let s = self.config.label_smoothing;
        let beta = self.config.beta;

        let mut loss_sum = 0.0f32;
        for i in 0..n {
            let logit =
                beta * ((log_prob_w[i] - ref_log_prob_w[i]) - (log_prob_l[i] - ref_log_prob_l[i]));
            if !logit.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            let term = if s == 0.0 {
                log_sigmoid(logit)
            } else {
                (1.0 - s) * log_sigmoid(logit) + s * log_sigmoid(-logit)
            };
            loss_sum += -term;
        }

        let loss = loss_sum / n as f32;
        if !loss.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(loss)
    }

    pub fn implicit_reward(&self, log_prob: &[f32], ref_log_prob: &[f32]) -> RlhfResult<Vec<f32>> {
        if log_prob.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if log_prob.len() != ref_log_prob.len() {
            return Err(RlhfError::DimensionMismatch {
                expected: log_prob.len(),
                got: ref_log_prob.len(),
            });
        }
        let rewards: Vec<f32> = log_prob
            .iter()
            .zip(ref_log_prob.iter())
            .map(|(&lp, &rlp)| self.config.beta * (lp - rlp))
            .collect();
        for &r in &rewards {
            if !r.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
        }
        Ok(rewards)
    }

    pub fn reward_margin(
        &self,
        log_prob_w: &[f32],
        ref_log_prob_w: &[f32],
        log_prob_l: &[f32],
        ref_log_prob_l: &[f32],
        batch_size: usize,
    ) -> RlhfResult<f32> {
        let rewards_w = self.implicit_reward(log_prob_w, ref_log_prob_w)?;
        let rewards_l = self.implicit_reward(log_prob_l, ref_log_prob_l)?;
        if batch_size == 0 {
            return Err(RlhfError::Internal {
                msg: "batch_size must be > 0".to_string(),
            });
        }
        let n = rewards_w.len();
        let sum: f32 = rewards_w
            .iter()
            .zip(rewards_l.iter())
            .map(|(&rw, &rl)| rw - rl)
            .sum();
        Ok(sum / n as f32)
    }

    pub fn beta(&self) -> f32 {
        self.config.beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dpo(beta: f32) -> DpoLoss {
        DpoLoss::new(DpoConfig {
            beta,
            label_smoothing: 0.0,
        })
        .expect("valid config")
    }

    #[test]
    fn loss_finite() {
        let dpo = make_dpo(0.1);
        let lp_w = vec![0.0f32; 4];
        let lp_l = vec![-1.0f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let rlp_l = vec![-1.0f32; 4];
        let loss = dpo.compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4).expect("ok");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    #[test]
    fn loss_nonneg() {
        let dpo = make_dpo(0.5);
        let lp_w = vec![-0.5f32; 4];
        let lp_l = vec![-0.5f32; 4];
        let rlp_w = vec![-0.5f32; 4];
        let rlp_l = vec![-0.5f32; 4];
        let loss = dpo.compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4).expect("ok");
        assert!(loss >= 0.0, "loss must be >= 0, got {loss}");
    }

    #[test]
    fn perfect_separation_low_loss() {
        // Policy strongly prefers w: log_prob_w >> ref_log_prob_w relative to l
        let dpo = make_dpo(1.0);
        let lp_w = vec![0.0f32; 4];
        let lp_l = vec![-10.0f32; 4];
        let rlp_w = vec![-5.0f32; 4];
        let rlp_l = vec![-5.0f32; 4];
        // logit = 1.0 * ((0 - (-5)) - (-10 - (-5))) = (5 - (-5)) = 10
        // loss = -log_sigmoid(10) ≈ 0.0000454
        let loss = dpo.compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4).expect("ok");
        assert!(
            loss < 0.01,
            "loss should be near 0 for perfect separation, got {loss}"
        );
    }

    #[test]
    fn reversed_preference_high_loss() {
        // Policy prefers l over w → negative logit → high loss
        let dpo = make_dpo(1.0);
        let lp_w = vec![-10.0f32; 4];
        let lp_l = vec![0.0f32; 4];
        let rlp_w = vec![-5.0f32; 4];
        let rlp_l = vec![-5.0f32; 4];
        // logit = 1.0 * ((-10 - (-5)) - (0 - (-5))) = (-5) - (5) = -10
        // loss = -log_sigmoid(-10) ≈ 10.0000454
        let loss = dpo.compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4).expect("ok");
        assert!(
            loss > 5.0,
            "loss should be high for reversed preference, got {loss}"
        );
    }

    #[test]
    fn implicit_reward_shape() {
        let dpo = make_dpo(0.5);
        let lp = vec![0.0f32, -1.0, -2.0];
        let rlp = vec![0.0f32, 0.0, 0.0];
        let r = dpo.implicit_reward(&lp, &rlp).expect("ok");
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn implicit_reward_proportional_to_beta() {
        let dpo1 = make_dpo(1.0);
        let dpo2 = make_dpo(2.0);
        let lp = vec![0.5f32, -0.3, 1.2];
        let rlp = vec![0.0f32, 0.0, 0.0];
        let r1 = dpo1.implicit_reward(&lp, &rlp).expect("ok");
        let r2 = dpo2.implicit_reward(&lp, &rlp).expect("ok");
        for (a, b) in r1.iter().zip(r2.iter()) {
            let ratio = b / a;
            assert!(
                (ratio - 2.0).abs() < 1e-5,
                "doubling beta should double reward: {a} vs {b}"
            );
        }
    }

    #[test]
    fn reward_margin_positive_for_good_policy() {
        let dpo = make_dpo(1.0);
        // Policy assigns higher prob to w
        let lp_w = vec![0.0f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let lp_l = vec![-1.0f32; 4];
        let rlp_l = vec![-1.0f32; 4];
        // r_w = 1.0*(0 - 0) = 0, r_l = 1.0*(-1 - (-1)) = 0, margin = 0
        // Let's use different ref probs to get positive margin
        let lp_w2 = vec![1.0f32; 4];
        let rlp_w2 = vec![0.0f32; 4];
        let lp_l2 = vec![0.0f32; 4];
        let rlp_l2 = vec![0.0f32; 4];
        // r_w = 1.0*(1-0)=1, r_l = 1.0*(0-0)=0, margin=1
        let margin = dpo
            .reward_margin(&lp_w2, &rlp_w2, &lp_l2, &rlp_l2, 4)
            .expect("ok");
        assert!(margin > 0.0, "margin should be positive, got {margin}");
    }

    #[test]
    fn reward_margin_negative_for_bad_policy() {
        let dpo = make_dpo(1.0);
        // Policy assigns higher prob to l than w
        let lp_w = vec![0.0f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let lp_l = vec![1.0f32; 4];
        let rlp_l = vec![0.0f32; 4];
        // r_w = 0, r_l = 1, margin = -1
        let margin = dpo
            .reward_margin(&lp_w, &rlp_w, &lp_l, &rlp_l, 4)
            .expect("ok");
        assert!(margin < 0.0, "margin should be negative, got {margin}");
    }

    #[test]
    fn batch_size_1_works() {
        let dpo = make_dpo(0.1);
        let loss = dpo
            .compute(&[0.0], &[-1.0], &[0.0], &[-1.0], 1)
            .expect("ok");
        assert!(loss.is_finite());
    }

    #[test]
    fn beta_0_loss_is_log2() {
        // When beta=0, logit=0, log_sigmoid(0) = -ln(2), so loss = ln(2)
        let dpo = make_dpo(0.0);
        let lp_w = vec![0.5f32; 4];
        let lp_l = vec![-0.5f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let rlp_l = vec![0.0f32; 4];
        let loss = dpo.compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4).expect("ok");
        let expected = 2.0f32.ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "beta=0 loss should be ln(2)={expected}, got {loss}"
        );
    }

    #[test]
    fn len_mismatch_error() {
        let dpo = make_dpo(0.1);
        let result = dpo.compute(&[0.0, 1.0], &[0.0], &[0.0, 1.0], &[0.0, 1.0], 2);
        assert!(result.is_err(), "should return Err on length mismatch");
    }

    #[test]
    fn label_smoothing_changes_loss() {
        let dpo_no_smooth = DpoLoss::new(DpoConfig {
            beta: 0.5,
            label_smoothing: 0.0,
        })
        .expect("valid");
        let dpo_smooth = DpoLoss::new(DpoConfig {
            beta: 0.5,
            label_smoothing: 0.1,
        })
        .expect("valid");
        let lp_w = vec![0.5f32; 4];
        let lp_l = vec![-0.5f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let rlp_l = vec![0.0f32; 4];
        let loss1 = dpo_no_smooth
            .compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4)
            .expect("ok");
        let loss2 = dpo_smooth
            .compute(&lp_w, &lp_l, &rlp_w, &rlp_l, 4)
            .expect("ok");
        assert!(
            (loss1 - loss2).abs() > 1e-6,
            "smoothing should change loss: {loss1} vs {loss2}"
        );
    }
}
