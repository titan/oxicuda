//! Variational Continual Learning (Nguyen, Li, Bui, Turner 2018, ICLR).
//!
//! Online variational inference for a sequence of tasks `T₁, T₂, ...`.
//! After each task is trained the resulting variational posterior
//! `q_t(θ)` replaces the prior for the next task:
//!
//! ```text
//! p_{t+1}(θ) = q_t(θ),     q_{t+1}(θ) ← argmax_q  ELBO(D_{t+1}; q, p_{t+1}).
//! ```
//!
//! The implementation here uses the standard mean-field diagonal Gaussian
//! parameterisation `q(θ) = Π_i N(μ_i, σ_i²)` where σ_i² = exp(`log_var_i`).
//! For a diagonal Gaussian prior `p = Π_i N(μ_pᵢ, σ_pᵢ²)` the KL divergence
//! has the well-known closed form
//!
//! ```text
//! KL(q ‖ p) = ½ Σ_i [ log(σ_pᵢ² / σ_qᵢ²)
//!                   + (σ_qᵢ² + (μ_qᵢ − μ_pᵢ)²) / σ_pᵢ²
//!                   − 1 ].
//! ```
//!
//! The ELBO `ℒ = E_q[log p(D|θ)] − KL(q ‖ p)` is maximised by passing in
//! caller-provided gradients `∂ E_q[log p(D|θ)] / ∂μ_i` and
//! `∂ E_q[log p(D|θ)] / ∂(log σ_i²)` (typically obtained from the
//! reparameterisation trick) and subtracting the analytic KL gradient.
//! Posterior consolidation (`consolidate`) returns `(mean, log_var)` so the
//! caller can feed them as the next-task prior.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

/// f32 log guard. The `LcgRng::next_f32()` precision and `(1−ε)/ε` ratios in
/// Box-Muller make `1e-12` smaller than `f32::EPSILON`; use `1e-7` so the
/// guard remains representable in f32.
const LOG_EPS: f32 = 1e-7;

/// Configuration of [`VclState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VclConfig {
    /// Number of parameters of the variational posterior (≥ 1).
    pub n_params: usize,
    /// Initial prior variance σ_p² of every parameter (`> 0`). The first
    /// task's prior is `N(0, σ_p²)`.
    pub init_prior_var: f32,
}

/// Mean-field Gaussian variational posterior maintained across tasks.
#[derive(Debug, Clone)]
pub struct VclState {
    /// Configuration.
    cfg: VclConfig,
    /// Variational mean `μ` (length `n_params`).
    mean: Vec<f32>,
    /// Variational log-variance `log σ²` (length `n_params`).
    log_var: Vec<f32>,
}

impl VclState {
    /// Construct a posterior initialised to the *first task's prior*:
    /// `μ = 0`, `log σ² = ln(init_prior_var)`.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] if `n_params == 0`.
    /// - [`BayesError::InvalidPriorVariance`] if `init_prior_var` is not
    ///   strictly positive and finite.
    pub fn new(cfg: VclConfig) -> BayesResult<Self> {
        if cfg.n_params == 0 {
            return Err(BayesError::EmptyInputs);
        }
        if !(cfg.init_prior_var.is_finite() && cfg.init_prior_var > 0.0) {
            return Err(BayesError::InvalidPriorVariance);
        }
        let lv = cfg.init_prior_var.ln();
        Ok(Self {
            cfg,
            mean: vec![0.0_f32; cfg.n_params],
            log_var: vec![lv; cfg.n_params],
        })
    }

    /// Number of parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.cfg.n_params
    }

    /// Variational mean `μ` (length `n_params`).
    #[must_use]
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Variational log-variance `log σ²` (length `n_params`).
    #[must_use]
    pub fn log_var(&self) -> &[f32] {
        &self.log_var
    }

    /// Closed-form `KL(q ‖ prior)` for two mean-field diagonal Gaussians.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if either prior slice has length
    ///   different from `n_params`.
    /// - [`BayesError::NanEncountered`] if a non-finite log-variance is
    ///   encountered.
    pub fn kl_to_prior(&self, prior_mean: &[f32], prior_log_var: &[f32]) -> BayesResult<f32> {
        let p = self.cfg.n_params;
        if prior_mean.len() != p {
            return Err(BayesError::DimensionMismatch {
                expected: p,
                got: prior_mean.len(),
            });
        }
        if prior_log_var.len() != p {
            return Err(BayesError::DimensionMismatch {
                expected: p,
                got: prior_log_var.len(),
            });
        }
        let mut kl = 0.0_f32;
        for i in 0..p {
            let mu_q = self.mean[i];
            let lv_q = self.log_var[i];
            let mu_p = prior_mean[i];
            let lv_p = prior_log_var[i];
            if !lv_q.is_finite() || !lv_p.is_finite() {
                return Err(BayesError::NanEncountered {
                    location: "VclState::kl_to_prior: non-finite log_var",
                });
            }
            let var_q = lv_q.exp().max(LOG_EPS);
            let var_p = lv_p.exp().max(LOG_EPS);
            let diff = mu_q - mu_p;
            // KL_i = ½ [ log(var_p/var_q) + (var_q + diff²)/var_p − 1 ].
            let term = (lv_p - lv_q) + (var_q + diff * diff) / var_p - 1.0;
            kl += 0.5 * term;
        }
        Ok(kl)
    }

    /// Analytic gradient of `KL(q ‖ prior)` w.r.t. `(μ_q, log σ_q²)`.
    /// Both returned vectors have length `n_params`.
    ///
    /// ```text
    /// ∂KL/∂μ_q   = (μ_q − μ_p) / σ_p²
    /// ∂KL/∂log σ_q² = ½ (σ_q² / σ_p² − 1)
    /// ```
    fn kl_grad(&self, prior_mean: &[f32], prior_log_var: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let p = self.cfg.n_params;
        let mut g_mu = vec![0.0_f32; p];
        let mut g_lv = vec![0.0_f32; p];
        for i in 0..p {
            let var_p = prior_log_var[i].exp().max(LOG_EPS);
            g_mu[i] = (self.mean[i] - prior_mean[i]) / var_p;
            let var_q = self.log_var[i].exp().max(LOG_EPS);
            g_lv[i] = 0.5 * (var_q / var_p - 1.0);
        }
        (g_mu, g_lv)
    }

    /// One ELBO gradient-ascent step.
    ///
    /// `ll_grad_mean` and `ll_grad_logvar` are caller-supplied gradients of
    /// the expected log-likelihood with respect to the variational mean and
    /// log-variance respectively (usually obtained via the reparameterisation
    /// trick on a Monte-Carlo estimate). `ll_value` is the value of the
    /// expected log-likelihood at the *current* `(μ, log σ²)` -- it is used
    /// only to assemble the returned ELBO.
    ///
    /// The update rule is
    ///
    /// ```text
    /// μ      ← μ      + lr · ( ∂E[log p]/∂μ      − ∂KL/∂μ )
    /// log σ² ← log σ² + lr · ( ∂E[log p]/∂log σ² − ∂KL/∂log σ² )
    /// ```
    ///
    /// Returns the ELBO `ll_value − KL(q ‖ prior)` evaluated *before* the
    /// step (this is the standard quantity reported when training VI).
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if any input slice has length
    ///   different from `n_params`.
    /// - [`BayesError::InvalidTemperature`] if `lr ≤ 0` or non-finite.
    /// - [`BayesError::NanEncountered`] if any input grad / value is
    ///   non-finite, or if the step would produce a non-finite log-variance.
    #[allow(clippy::too_many_arguments)]
    pub fn elbo_step(
        &mut self,
        ll_grad_mean: &[f32],
        ll_grad_logvar: &[f32],
        prior_mean: &[f32],
        prior_log_var: &[f32],
        ll_value: f32,
        lr: f32,
    ) -> BayesResult<f32> {
        let p = self.cfg.n_params;
        for (name, s) in [
            ("ll_grad_mean", ll_grad_mean),
            ("ll_grad_logvar", ll_grad_logvar),
            ("prior_mean", prior_mean),
            ("prior_log_var", prior_log_var),
        ] {
            if s.len() != p {
                let _ = name;
                return Err(BayesError::DimensionMismatch {
                    expected: p,
                    got: s.len(),
                });
            }
        }
        if !(lr.is_finite() && lr > 0.0) {
            return Err(BayesError::InvalidTemperature { temp: lr });
        }
        if !ll_value.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "VclState::elbo_step: non-finite ll_value",
            });
        }
        for g in ll_grad_mean.iter().chain(ll_grad_logvar.iter()) {
            if !g.is_finite() {
                return Err(BayesError::NanEncountered {
                    location: "VclState::elbo_step: non-finite ll grad",
                });
            }
        }
        // Evaluate KL and ELBO at the current parameters.
        let kl = self.kl_to_prior(prior_mean, prior_log_var)?;
        let elbo_value = ll_value - kl;
        // Subtract analytic KL gradient to obtain ELBO gradient (since
        // ELBO = ll − KL).
        let (kl_g_mu, kl_g_lv) = self.kl_grad(prior_mean, prior_log_var);
        for i in 0..p {
            let dm = ll_grad_mean[i] - kl_g_mu[i];
            let dlv = ll_grad_logvar[i] - kl_g_lv[i];
            let new_mean = self.mean[i] + lr * dm;
            let new_lv = self.log_var[i] + lr * dlv;
            if !new_mean.is_finite() || !new_lv.is_finite() {
                return Err(BayesError::NanEncountered {
                    location: "VclState::elbo_step: non-finite parameter update",
                });
            }
            self.mean[i] = new_mean;
            self.log_var[i] = new_lv;
        }
        Ok(elbo_value)
    }

    /// Consolidate the current posterior: returns owned copies of
    /// `(mean, log_var)` to be used as the prior for the next task.
    #[must_use]
    pub fn consolidate(&self) -> (Vec<f32>, Vec<f32>) {
        (self.mean.clone(), self.log_var.clone())
    }

    /// Reparameterised weight sample: `θ_i = μ_i + exp(log σ_i² / 2) · ε_i`,
    /// with `ε_i ~ N(0, 1)` drawn from `rng.next_normal_pair`.
    pub fn sample(&self, rng: &mut LcgRng) -> Vec<f32> {
        let p = self.cfg.n_params;
        let mut out = Vec::with_capacity(p);
        let mut i = 0;
        while i + 1 < p {
            let (z0, z1) = rng.next_normal_pair();
            let sigma0 = (0.5 * self.log_var[i]).exp();
            let sigma1 = (0.5 * self.log_var[i + 1]).exp();
            out.push(self.mean[i] + sigma0 * z0);
            out.push(self.mean[i + 1] + sigma1 * z1);
            i += 2;
        }
        if i < p {
            let (z, _) = rng.next_normal_pair();
            let sigma = (0.5 * self.log_var[i]).exp();
            out.push(self.mean[i] + sigma * z);
        }
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n_params: usize, init_prior_var: f32) -> VclState {
        VclState::new(VclConfig {
            n_params,
            init_prior_var,
        })
        .expect("test invariant: VclState::new must succeed")
    }

    #[test]
    fn new_initialises_mean_and_log_var() {
        let s = make(5, 4.0);
        assert_eq!(s.n_params(), 5);
        for &m in s.mean() {
            assert_eq!(m, 0.0);
        }
        let expected_lv = 4.0_f32.ln();
        for &lv in s.log_var() {
            assert!((lv - expected_lv).abs() < 1e-6);
        }
    }

    #[test]
    fn kl_zero_when_q_equals_prior() {
        let s = make(4, 2.0);
        let prior_mean = vec![0.0_f32; 4];
        let prior_lv = vec![2.0_f32.ln(); 4];
        let kl = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(
            kl.abs() < 1e-5,
            "KL should be 0 when q equals prior, got {kl}"
        );
    }

    #[test]
    fn kl_non_negative() {
        let mut s = make(3, 1.0);
        // Move mean away from the prior; KL must stay ≥ 0.
        s.mean[0] = 1.5;
        s.mean[2] = -0.7;
        s.log_var[1] = 0.4;
        let prior_mean = vec![0.0_f32; 3];
        let prior_lv = vec![0.0_f32; 3];
        let kl = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(kl >= -1e-6, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn kl_increases_with_mean_drift() {
        let mut s = make(2, 1.0);
        let prior_mean = vec![0.0_f32; 2];
        let prior_lv = vec![0.0_f32; 2];
        let kl_close = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        s.mean[0] = 2.0;
        s.mean[1] = -3.0;
        let kl_far = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(kl_far > kl_close, "KL should grow with mean drift");
    }

    #[test]
    fn elbo_step_moves_mean_toward_ll_gradient() {
        let mut s = make(3, 1.0);
        let prior_mean = vec![0.0_f32; 3];
        let prior_lv = vec![0.0_f32; 3];
        let ll_grad_mean = vec![1.0_f32, -1.0, 0.5];
        let ll_grad_logvar = vec![0.0_f32; 3];
        let m_before = s.mean().to_vec();
        let _ = s
            .elbo_step(
                &ll_grad_mean,
                &ll_grad_logvar,
                &prior_mean,
                &prior_lv,
                0.0,
                0.01,
            )
            .expect("test invariant: elbo_step must succeed");
        // At μ = 0 = prior, KL-grad for μ is 0, so the mean moves by lr·ll_grad_mean.
        assert!(s.mean()[0] > m_before[0], "mean[0] should increase");
        assert!(s.mean()[1] < m_before[1], "mean[1] should decrease");
        assert!(s.mean()[2] > m_before[2], "mean[2] should increase");
    }

    #[test]
    fn elbo_step_returns_finite_elbo() {
        let mut s = make(3, 1.0);
        let v = s
            .elbo_step(
                &[0.1_f32, -0.1, 0.05],
                &[0.0_f32; 3],
                &[0.0_f32; 3],
                &[0.0_f32; 3],
                -2.5,
                0.05,
            )
            .expect("test invariant: elbo_step must succeed");
        assert!(v.is_finite(), "ELBO must be finite, got {v}");
    }

    #[test]
    fn consolidate_returns_current_state() {
        let mut s = make(2, 1.0);
        s.mean[0] = 0.3;
        s.mean[1] = -0.7;
        s.log_var[0] = 0.5;
        s.log_var[1] = -0.2;
        let (m, lv) = s.consolidate();
        assert_eq!(m, vec![0.3_f32, -0.7]);
        assert_eq!(lv, vec![0.5_f32, -0.2]);
    }

    #[test]
    fn prior_replacement_yields_zero_kl() {
        let mut s = make(4, 1.0);
        s.mean = vec![0.5_f32, -0.3, 0.1, 0.8];
        s.log_var = vec![0.2_f32, -0.4, 0.3, -0.1];
        let (new_prior_mean, new_prior_lv) = s.consolidate();
        let kl = s
            .kl_to_prior(&new_prior_mean, &new_prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(
            kl.abs() < 1e-5,
            "after consolidate, KL(q‖q) should be 0, got {kl}"
        );
    }

    #[test]
    fn sample_length_matches_n_params() {
        let s = make(7, 1.0);
        let mut rng = LcgRng::new(42);
        let theta = s.sample(&mut rng);
        assert_eq!(theta.len(), 7);
    }

    #[test]
    fn sample_collapses_to_mean_when_variance_is_tiny() {
        let mut s = make(4, 1.0);
        s.mean = vec![1.0_f32, -2.0, 3.0, 0.5];
        // Very small variance.
        s.log_var = vec![-30.0_f32; 4];
        let mut rng = LcgRng::new(7);
        let theta = s.sample(&mut rng);
        for (i, &v) in theta.iter().enumerate() {
            assert!(
                (v - s.mean[i]).abs() < 1e-3,
                "sample[{i}]={v} should be ≈ mean[{i}]={}",
                s.mean[i]
            );
        }
    }

    #[test]
    fn sample_deterministic_given_seed() {
        let s = make(5, 1.0);
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        assert_eq!(s.sample(&mut r1), s.sample(&mut r2));
    }

    #[test]
    fn kl_closed_form_one_param_hand_check() {
        // Single parameter, q = N(0.5, 1) and p = N(0, 4).
        // KL = ½ [ log(4/1) + (1 + 0.25)/4 − 1 ]
        //    = ½ [ ln 4 + 0.3125 − 1 ]
        //    = ½ [ ln 4 − 0.6875 ]
        let mut s = make(1, 1.0);
        s.mean[0] = 0.5;
        s.log_var[0] = 0.0; // var = 1.
        let prior_mean = vec![0.0_f32];
        let prior_lv = vec![4.0_f32.ln()]; // var = 4.
        let kl = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        let expected = 0.5_f32 * (4.0_f32.ln() - 0.6875);
        assert!(
            (kl - expected).abs() < 1e-4,
            "kl={kl} vs expected={expected}"
        );
    }

    #[test]
    fn err_kl_prior_mean_length_mismatch() {
        let s = make(3, 1.0);
        let r = s.kl_to_prior(&[0.0_f32, 0.0], &[0.0_f32, 0.0, 0.0]);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_kl_prior_log_var_length_mismatch() {
        let s = make(3, 1.0);
        let r = s.kl_to_prior(&[0.0_f32; 3], &[0.0_f32, 0.0]);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_elbo_step_grad_length_mismatch() {
        let mut s = make(3, 1.0);
        let r = s.elbo_step(
            &[0.0_f32, 0.0], // wrong length
            &[0.0_f32; 3],
            &[0.0_f32; 3],
            &[0.0_f32; 3],
            0.0,
            0.01,
        );
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_n_params_zero() {
        let r = VclState::new(VclConfig {
            n_params: 0,
            init_prior_var: 1.0,
        });
        assert!(matches!(r, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn err_init_prior_var_non_positive() {
        for bad in [0.0_f32, -0.1, -1.0, f32::NAN, f32::INFINITY] {
            let r = VclState::new(VclConfig {
                n_params: 3,
                init_prior_var: bad,
            });
            assert!(
                matches!(r, Err(BayesError::InvalidPriorVariance)),
                "expected InvalidPriorVariance for init_prior_var={bad}"
            );
        }
    }

    #[test]
    fn err_elbo_step_lr_non_positive() {
        let mut s = make(2, 1.0);
        for bad in [0.0_f32, -0.1, f32::NAN, f32::INFINITY] {
            let r = s.elbo_step(
                &[0.0_f32, 0.0],
                &[0.0_f32, 0.0],
                &[0.0_f32, 0.0],
                &[0.0_f32, 0.0],
                0.0,
                bad,
            );
            assert!(
                matches!(r, Err(BayesError::InvalidTemperature { .. })),
                "expected InvalidTemperature for lr={bad}"
            );
        }
    }

    #[test]
    fn two_task_sequence_kl_anchors_to_previous_posterior() {
        // Train on task 1: nudge mean. Consolidate. Then for task 2, KL is
        // measured against the new prior (= old posterior), and an immediate
        // KL just after consolidation is 0. After perturbing q for task 2,
        // KL grows -- demonstrating the anti-forgetting anchor.
        let mut s = make(2, 1.0);
        // Task 1 update.
        let _ = s
            .elbo_step(
                &[0.4_f32, -0.2],
                &[0.0_f32; 2],
                &[0.0_f32; 2],
                &[0.0_f32; 2],
                0.0,
                0.1,
            )
            .expect("test invariant: elbo_step must succeed");
        // Consolidate.
        let (prior2_mean, prior2_lv) = s.consolidate();
        // Immediately after consolidate: KL = 0.
        let kl0 = s
            .kl_to_prior(&prior2_mean, &prior2_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(kl0.abs() < 1e-5);
        // Take a task-2 step that moves the mean further away.
        let _ = s
            .elbo_step(
                &[0.5_f32, 0.5],
                &[0.0_f32; 2],
                &prior2_mean,
                &prior2_lv,
                0.0,
                0.1,
            )
            .expect("test invariant: elbo_step must succeed");
        let kl1 = s
            .kl_to_prior(&prior2_mean, &prior2_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(
            kl1 > kl0,
            "anchor KL should grow after task-2 step: kl0={kl0}, kl1={kl1}"
        );
    }

    #[test]
    fn log_var_guard_no_nan() {
        // Even with a tiny prior variance, KL must remain finite thanks to
        // the LOG_EPS guard.
        let mut s = make(2, 1.0);
        s.log_var[0] = -50.0;
        s.log_var[1] = -50.0;
        let prior_mean = vec![0.0_f32; 2];
        let prior_lv = vec![-50.0_f32; 2];
        let kl = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        assert!(kl.is_finite(), "KL must be finite, got {kl}");
    }

    #[test]
    fn elbo_step_subtracts_kl_gradient() {
        // With ll grad = 0, the step performs pure KL minimisation; the mean
        // should be pulled toward the prior mean.
        let mut s = make(2, 1.0);
        s.mean = vec![1.0_f32, -1.0];
        let prior_mean = vec![0.0_f32, 0.0];
        let prior_lv = vec![0.0_f32, 0.0];
        let m_before = s.mean().to_vec();
        let _ = s
            .elbo_step(
                &[0.0_f32; 2],
                &[0.0_f32; 2],
                &prior_mean,
                &prior_lv,
                0.0,
                0.5,
            )
            .expect("test invariant: elbo_step must succeed");
        assert!(
            s.mean()[0] < m_before[0],
            "mean[0] should be pulled down toward 0"
        );
        assert!(
            s.mean()[1] > m_before[1],
            "mean[1] should be pulled up toward 0"
        );
    }

    #[test]
    fn elbo_value_matches_ll_minus_kl() {
        let mut s = make(2, 1.0);
        s.mean = vec![0.5_f32, -0.5];
        let prior_mean = vec![0.0_f32, 0.0];
        let prior_lv = vec![0.0_f32, 0.0];
        let ll_value = -3.5_f32;
        // Stash KL ahead of the step (the step updates parameters).
        let kl_before = s
            .kl_to_prior(&prior_mean, &prior_lv)
            .expect("test invariant: kl_to_prior must succeed");
        let elbo = s
            .elbo_step(
                &[0.0_f32; 2],
                &[0.0_f32; 2],
                &prior_mean,
                &prior_lv,
                ll_value,
                0.0001,
            )
            .expect("test invariant: elbo_step must succeed");
        let expected = ll_value - kl_before;
        assert!(
            (elbo - expected).abs() < 1e-4,
            "elbo={elbo} vs expected={expected}"
        );
    }
}
