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

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Per-element gradients of [`DpoLoss::compute`] w.r.t. its four log-prob slices.
///
/// Each vector has the same length / order as the corresponding input slice and
/// holds the gradient of the *mean* loss (`compute` divides the sum by `n`).
/// Finite-difference verified against [`DpoLoss::compute`].
#[derive(Debug, Clone)]
pub struct DpoGradients {
    /// `∂L/∂(policy chosen log-prob)`.
    pub d_log_prob_w: Vec<f32>,
    /// `∂L/∂(policy rejected log-prob)`.
    pub d_log_prob_l: Vec<f32>,
    /// `∂L/∂(reference chosen log-prob)`.
    pub d_ref_log_prob_w: Vec<f32>,
    /// `∂L/∂(reference rejected log-prob)`.
    pub d_ref_log_prob_l: Vec<f32>,
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

    /// Analytic gradient of [`DpoLoss::compute`] w.r.t. its four log-prob slices.
    ///
    /// With `term_i = (1−s)·log σ(z_i) + s·log σ(−z_i)`,
    /// `z_i = β·((lp_w − rlp_w) − (lp_l − rlp_l))`, and `L = mean_i(−term_i)`:
    ///
    /// `dL/dz_i = (1/n)·[−(1−s)·σ(−z_i) + s·σ(z_i)]`,
    ///
    /// then chaining through `z_i` (partials `+β, −β, −β, +β` for
    /// `lp_w, rlp_w, lp_l, rlp_l`). For `s = 0` this reduces to
    /// `dL/d lp_w = −(β/n)·σ(−z_i)`.
    ///
    /// # Errors
    /// Mirrors [`DpoLoss::compute`]: [`RlhfError::EmptyInput`],
    /// [`RlhfError::DimensionMismatch`], an `Internal` error for `batch_size == 0`,
    /// and [`RlhfError::NanEncountered`] for a non-finite logit / gradient.
    pub fn grad(
        &self,
        log_prob_w: &[f32],
        log_prob_l: &[f32],
        ref_log_prob_w: &[f32],
        ref_log_prob_l: &[f32],
        batch_size: usize,
    ) -> RlhfResult<DpoGradients> {
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
        let inv_n = 1.0 / n as f32;

        let mut d_log_prob_w = Vec::with_capacity(n);
        let mut d_log_prob_l = Vec::with_capacity(n);
        let mut d_ref_log_prob_w = Vec::with_capacity(n);
        let mut d_ref_log_prob_l = Vec::with_capacity(n);

        for i in 0..n {
            let logit =
                beta * ((log_prob_w[i] - ref_log_prob_w[i]) - (log_prob_l[i] - ref_log_prob_l[i]));
            if !logit.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            // dL/dlogit_i = (1/n)·[−(1−s)·σ(−z) + s·σ(z)]
            let d_logit = inv_n * (-(1.0 - s) * sigmoid(-logit) + s * sigmoid(logit));
            let g_w = d_logit * beta;
            if !g_w.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
            d_log_prob_w.push(g_w);
            d_log_prob_l.push(-g_w);
            d_ref_log_prob_w.push(-g_w);
            d_ref_log_prob_l.push(g_w);
        }

        Ok(DpoGradients {
            d_log_prob_w,
            d_log_prob_l,
            d_ref_log_prob_w,
            d_ref_log_prob_l,
        })
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
        // Policy raises prob of w above reference while leaving l at reference:
        // r_w = 1.0*(1-0)=1, r_l = 1.0*(0-0)=0, margin = 1 > 0
        let lp_w = vec![1.0f32; 4];
        let rlp_w = vec![0.0f32; 4];
        let lp_l = vec![0.0f32; 4];
        let rlp_l = vec![0.0f32; 4];
        let margin = dpo
            .reward_margin(&lp_w, &rlp_w, &lp_l, &rlp_l, 4)
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
    fn grad_matches_finite_difference_no_smoothing() {
        let dpo = make_dpo(0.4);
        let w = [-0.5_f32, -1.2];
        let l = [-1.4_f32, -0.7];
        let rw = [-0.7_f32, -1.0];
        let rl = [-1.1_f32, -0.9];
        let g = dpo.grad(&w, &l, &rw, &rl, w.len()).expect("grad");
        let h = 1e-2;
        for i in 0..w.len() {
            let fd_w = central_diff(
                |v| {
                    let mut x = w.to_vec();
                    x[i] = v;
                    dpo.compute(&x, &l, &rw, &rl, w.len()).expect("loss")
                },
                w[i],
                h,
            );
            let fd_l = central_diff(
                |v| {
                    let mut x = l.to_vec();
                    x[i] = v;
                    dpo.compute(&w, &x, &rw, &rl, w.len()).expect("loss")
                },
                l[i],
                h,
            );
            let fd_rw = central_diff(
                |v| {
                    let mut x = rw.to_vec();
                    x[i] = v;
                    dpo.compute(&w, &l, &x, &rl, w.len()).expect("loss")
                },
                rw[i],
                h,
            );
            let fd_rl = central_diff(
                |v| {
                    let mut x = rl.to_vec();
                    x[i] = v;
                    dpo.compute(&w, &l, &rw, &x, w.len()).expect("loss")
                },
                rl[i],
                h,
            );
            assert_grad_close(g.d_log_prob_w[i], fd_w, "d_log_prob_w");
            assert_grad_close(g.d_log_prob_l[i], fd_l, "d_log_prob_l");
            assert_grad_close(g.d_ref_log_prob_w[i], fd_rw, "d_ref_log_prob_w");
            assert_grad_close(g.d_ref_log_prob_l[i], fd_rl, "d_ref_log_prob_l");
        }
    }

    #[test]
    fn grad_matches_finite_difference_with_smoothing() {
        let dpo = DpoLoss::new(DpoConfig {
            beta: 0.6,
            label_smoothing: 0.1,
        })
        .expect("valid");
        let w = [0.2_f32];
        let l = [-0.8_f32];
        let rw = [0.0_f32];
        let rl = [0.0_f32];
        let g = dpo.grad(&w, &l, &rw, &rl, 1).expect("grad");
        let h = 1e-2;
        let fd_w = central_diff(
            |v| dpo.compute(&[v], &l, &rw, &rl, 1).expect("loss"),
            w[0],
            h,
        );
        assert_grad_close(g.d_log_prob_w[0], fd_w, "d_log_prob_w smoothed");
    }

    #[test]
    fn grad_chosen_negative_when_preferred() {
        let dpo = make_dpo(0.5);
        let g = dpo.grad(&[0.0], &[0.0], &[0.0], &[0.0], 1).expect("grad");
        assert!(g.d_log_prob_w[0] < 0.0, "{}", g.d_log_prob_w[0]);
        assert!(g.d_log_prob_l[0] > 0.0, "{}", g.d_log_prob_l[0]);
    }

    #[test]
    fn grad_len_mismatch_error() {
        let dpo = make_dpo(0.1);
        assert!(
            dpo.grad(&[0.0, 1.0], &[0.0], &[0.0, 1.0], &[0.0, 1.0], 2)
                .is_err()
        );
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
