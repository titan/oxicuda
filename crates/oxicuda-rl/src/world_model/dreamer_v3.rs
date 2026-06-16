//! DreamerV3 Recurrent State Space Model (RSSM).
//!
//! Hafner et al. (2023). Simplified CPU version: linear+tanh GRU approximation,
//! diagonal Gaussian stochastic state, imagination rollouts.
//!
//! The RSSM factorises the world model into:
//! - A **deterministic** recurrent state `h_t` (GRU cell).
//! - A **stochastic** latent state `z_t` (diagonal Gaussian).
//!
//! Two inference modes:
//! - [`Rssm::imagine_step`]   — prior rollout (no observation).
//! - [`Rssm::observe_step`]   — posterior update with encoded observation.

use crate::error::{RlError, RlResult};

/// Convenience alias so callers need not import `handle` directly.
pub type RlRng = crate::handle::LcgRng;

// ─── Box-Muller helper ───────────────────────────────────────────────────────

/// Sample one standard normal variate via the Box-Muller transform.
///
/// Uses two uniform samples from `rng` to produce `N(0,1)`.
fn sample_normal(rng: &mut RlRng) -> f32 {
    // Clamp u1 away from zero to keep ln() finite
    let u1 = (rng.next_f32() + 1e-10_f32).min(1.0 - 1e-10_f32);
    let u2 = rng.next_f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

// ─── Matrix-vector multiply ──────────────────────────────────────────────────

/// Row-major matrix-vector multiply: `y = W * x` where `W` has shape
/// `[out_dim × in_dim]` stored in row-major order.
///
/// `x.len()` must equal `in_dim` = `w.len() / out_dim`.
fn matmul(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    (0..out_dim)
        .map(|i| {
            let row_start = i * in_dim;
            w[row_start..row_start + in_dim]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

// ─── RssmConfig ──────────────────────────────────────────────────────────────

/// Configuration for the Recurrent State Space Model.
#[derive(Debug)]
pub struct RssmConfig {
    /// Dimension of the deterministic recurrent state `h`.
    pub d_deter: usize,
    /// Dimension of the stochastic latent state `z`.
    pub d_stoch: usize,
    /// Dimension of the encoded observation embedding fed to the posterior.
    pub d_obs: usize,
    /// Dimension of the action vector.
    pub d_action: usize,
    /// Number of discrete classes (reserved for categorical future extension).
    pub n_classes: usize,
}

// ─── RssmState ───────────────────────────────────────────────────────────────

/// One time-step of RSSM state: `(h_t, z_t)`.
pub struct RssmState {
    /// Deterministic recurrent state `h_t` of length `d_deter`.
    pub h: Vec<f32>,
    /// Stochastic latent state `z_t` of length `d_stoch`.
    pub z: Vec<f32>,
}

// ─── Rssm ────────────────────────────────────────────────────────────────────

/// Recurrent State Space Model (RSSM) from DreamerV3.
///
/// Implements a simplified single-gate GRU with tanh activation and diagonal
/// Gaussian latent distributions.
#[derive(Debug)]
pub struct Rssm {
    /// GRU weight matrix: `[d_deter × (d_deter + d_stoch + d_action)]`, row-major.
    gru_w: Vec<f32>,
    /// GRU bias: `[d_deter]`.
    gru_b: Vec<f32>,
    /// Prior network weight: `[2*d_stoch × d_deter]`, row-major.
    prior_w: Vec<f32>,
    /// Prior network bias: `[2*d_stoch]`.
    prior_b: Vec<f32>,
    /// Posterior network weight: `[2*d_stoch × (d_deter + d_obs)]`, row-major.
    post_w: Vec<f32>,
    /// Posterior network bias: `[2*d_stoch]`.
    post_b: Vec<f32>,
    /// Model configuration.
    config: RssmConfig,
}

impl Rssm {
    /// Create a new RSSM with He-like random weight initialisation (scale 0.1).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if any of `d_deter`, `d_stoch`,
    /// or `d_action` is zero.
    pub fn new(config: RssmConfig, rng: &mut RlRng) -> RlResult<Self> {
        if config.d_deter == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "d_deter".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.d_stoch == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "d_stoch".into(),
                msg: "must be > 0".into(),
            });
        }
        if config.d_action == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "d_action".into(),
                msg: "must be > 0".into(),
            });
        }

        let gru_in = config.d_deter + config.d_stoch + config.d_action;
        let gru_out = config.d_deter;
        let prior_out = 2 * config.d_stoch;
        let prior_in = config.d_deter;
        let post_out = 2 * config.d_stoch;
        let post_in = config.d_deter + config.d_obs;

        let scale = 0.1_f32;

        let mut init_weights =
            |n: usize| -> Vec<f32> { (0..n).map(|_| sample_normal(rng) * scale).collect() };

        let gru_w = init_weights(gru_out * gru_in);
        let gru_b = init_weights(gru_out);
        let prior_w = init_weights(prior_out * prior_in);
        let prior_b = init_weights(prior_out);
        let post_w = init_weights(post_out * post_in);
        let post_b = init_weights(post_out);

        Ok(Self {
            gru_w,
            gru_b,
            prior_w,
            prior_b,
            post_w,
            post_b,
            config,
        })
    }

    /// Dimension of the deterministic recurrent state.
    #[must_use]
    #[inline]
    pub fn d_deter(&self) -> usize {
        self.config.d_deter
    }

    /// Dimension of the stochastic latent state.
    #[must_use]
    #[inline]
    pub fn d_stoch(&self) -> usize {
        self.config.d_stoch
    }

    /// Construct a zero-filled initial RSSM state.
    #[must_use]
    pub fn zero_state(&self) -> RssmState {
        RssmState {
            h: vec![0.0_f32; self.config.d_deter],
            z: vec![0.0_f32; self.config.d_stoch],
        }
    }

    // ── Internal GRU step ────────────────────────────────────────────────────

    /// Advance the GRU cell: `h_new = tanh(W_gru * [h, z, a] + b_gru)`.
    fn gru_step(&self, prev: &RssmState, action: &[f32]) -> Vec<f32> {
        let mut concat =
            Vec::with_capacity(self.config.d_deter + self.config.d_stoch + self.config.d_action);
        concat.extend_from_slice(&prev.h);
        concat.extend_from_slice(&prev.z);
        concat.extend_from_slice(action);

        let pre_act = matmul(&self.gru_w, &concat, self.config.d_deter);
        pre_act
            .into_iter()
            .zip(self.gru_b.iter())
            .map(|(x, &b)| (x + b).tanh())
            .collect()
    }

    // ── Gaussian sample from params ──────────────────────────────────────────

    /// Sample `z ~ N(mu, exp(log_sigma)^2)` from a raw parameter vector of
    /// length `2 * d_stoch`: `[mu | log_sigma]`.
    ///
    /// `log_sigma` is clamped to `[-4, 4]` for numerical stability.
    fn sample_z(&self, params: &[f32], rng: &mut RlRng) -> Vec<f32> {
        let d = self.config.d_stoch;
        (0..d)
            .map(|i| {
                let mu = params[i];
                let log_sigma = params[d + i].clamp(-4.0, 4.0);
                let sigma = log_sigma.exp();
                mu + sigma * sample_normal(rng)
            })
            .collect()
    }

    /// Perform one imagination (prior) step without an observation.
    ///
    /// Advances the recurrent state and samples `z_new` from the prior.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `action.len() != d_action`.
    pub fn imagine_step(
        &self,
        prev_state: &RssmState,
        action: &[f32],
        rng: &mut RlRng,
    ) -> RlResult<RssmState> {
        if action.len() != self.config.d_action {
            return Err(RlError::DimensionMismatch {
                expected: self.config.d_action,
                got: action.len(),
            });
        }

        let h_new = self.gru_step(prev_state, action);

        // Prior: params = W_prior * h_new + b_prior
        let prior_params_pre = matmul(&self.prior_w, &h_new, 2 * self.config.d_stoch);
        let prior_params: Vec<f32> = prior_params_pre
            .into_iter()
            .zip(self.prior_b.iter())
            .map(|(x, &b)| x + b)
            .collect();

        let z_new = self.sample_z(&prior_params, rng);

        Ok(RssmState { h: h_new, z: z_new })
    }

    /// Perform one observation (posterior) step given an encoded observation.
    ///
    /// Advances the recurrent state using the same GRU as `imagine_step`, then
    /// samples `z_new` from the posterior conditioned on `obs_embed`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `action.len() != d_action` or
    /// `obs_embed.len() != d_obs`.
    pub fn observe_step(
        &self,
        prev_state: &RssmState,
        action: &[f32],
        obs_embed: &[f32],
        rng: &mut RlRng,
    ) -> RlResult<RssmState> {
        if action.len() != self.config.d_action {
            return Err(RlError::DimensionMismatch {
                expected: self.config.d_action,
                got: action.len(),
            });
        }
        if obs_embed.len() != self.config.d_obs {
            return Err(RlError::DimensionMismatch {
                expected: self.config.d_obs,
                got: obs_embed.len(),
            });
        }

        let h_new = self.gru_step(prev_state, action);

        // Posterior: input = [h_new, obs_embed]
        let mut post_input = Vec::with_capacity(self.config.d_deter + self.config.d_obs);
        post_input.extend_from_slice(&h_new);
        post_input.extend_from_slice(obs_embed);

        let post_params_pre = matmul(&self.post_w, &post_input, 2 * self.config.d_stoch);
        let post_params: Vec<f32> = post_params_pre
            .into_iter()
            .zip(self.post_b.iter())
            .map(|(x, &b)| x + b)
            .collect();

        let z_new = self.sample_z(&post_params, rng);

        Ok(RssmState { h: h_new, z: z_new })
    }

    /// Roll out `horizon` imagination steps starting from `start`.
    ///
    /// `actions` is a flat buffer of shape `[horizon × d_action]` in row-major
    /// order. An empty `Vec` is returned when `horizon == 0`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `actions.len() != horizon * d_action`
    /// (for `horizon > 0`).
    pub fn imagine_rollout(
        &self,
        start: &RssmState,
        actions: &[f32],
        horizon: usize,
        rng: &mut RlRng,
    ) -> RlResult<Vec<RssmState>> {
        if horizon == 0 {
            return Ok(Vec::new());
        }

        let expected_len = horizon * self.config.d_action;
        if actions.len() != expected_len {
            return Err(RlError::DimensionMismatch {
                expected: expected_len,
                got: actions.len(),
            });
        }

        let mut states = Vec::with_capacity(horizon);
        let mut prev = RssmState {
            h: start.h.clone(),
            z: start.z.clone(),
        };

        for t in 0..horizon {
            let a_start = t * self.config.d_action;
            let a_end = a_start + self.config.d_action;
            let next = self.imagine_step(&prev, &actions[a_start..a_end], rng)?;
            prev = RssmState {
                h: next.h.clone(),
                z: next.z.clone(),
            };
            states.push(next);
        }

        Ok(states)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rssm(seed: u64) -> (Rssm, RlRng) {
        let mut rng = RlRng::new(seed);
        let config = RssmConfig {
            d_deter: 8,
            d_stoch: 4,
            d_obs: 6,
            d_action: 3,
            n_classes: 32,
        };
        let rssm = Rssm::new(config, &mut rng).expect("valid config should create RSSM");
        (rssm, rng)
    }

    #[test]
    fn imagine_step_shape() {
        let (rssm, mut rng) = make_rssm(1);
        let state = rssm.zero_state();
        let action = vec![0.1_f32; 3];
        let next = rssm
            .imagine_step(&state, &action, &mut rng)
            .expect("imagine_step should succeed with valid action");
        assert_eq!(next.h.len(), 8, "h should have d_deter=8 elements");
        assert_eq!(next.z.len(), 4, "z should have d_stoch=4 elements");
    }

    #[test]
    fn observe_step_shape() {
        let (rssm, mut rng) = make_rssm(2);
        let state = rssm.zero_state();
        let action = vec![0.1_f32; 3];
        let obs = vec![0.5_f32; 6];
        let next = rssm
            .observe_step(&state, &action, &obs, &mut rng)
            .expect("observe_step should succeed with valid inputs");
        assert_eq!(next.h.len(), 8, "h should have d_deter=8 elements");
        assert_eq!(next.z.len(), 4, "z should have d_stoch=4 elements");
    }

    #[test]
    fn imagine_rollout_len() {
        let (rssm, mut rng) = make_rssm(3);
        let state = rssm.zero_state();
        let horizon = 5;
        let actions = vec![0.0_f32; horizon * 3];
        let states = rssm
            .imagine_rollout(&state, &actions, horizon, &mut rng)
            .expect("imagine_rollout should succeed with valid inputs");
        assert_eq!(
            states.len(),
            horizon,
            "rollout should return horizon states"
        );
    }

    #[test]
    fn h_finite() {
        let (rssm, mut rng) = make_rssm(4);
        let state = rssm.zero_state();
        let action = vec![1.0_f32; 3];
        let next = rssm
            .imagine_step(&state, &action, &mut rng)
            .expect("imagine_step should succeed");
        for (i, &v) in next.h.iter().enumerate() {
            assert!(v.is_finite(), "h[{i}]={v} is not finite");
        }
    }

    #[test]
    fn z_finite() {
        let (rssm, mut rng) = make_rssm(5);
        let state = rssm.zero_state();
        let action = vec![1.0_f32; 3];
        let next = rssm
            .imagine_step(&state, &action, &mut rng)
            .expect("imagine_step should succeed");
        for (i, &v) in next.z.iter().enumerate() {
            assert!(v.is_finite(), "z[{i}]={v} is not finite");
        }
    }

    #[test]
    fn zero_action_works() {
        let (rssm, mut rng) = make_rssm(6);
        let state = rssm.zero_state();
        let action = vec![0.0_f32; 3];
        rssm.imagine_step(&state, &action, &mut rng)
            .expect("all-zero action should not error");
    }

    #[test]
    fn rollout_horizon_0_empty() {
        let (rssm, mut rng) = make_rssm(7);
        let state = rssm.zero_state();
        let states = rssm
            .imagine_rollout(&state, &[], 0, &mut rng)
            .expect("horizon=0 should succeed and return empty vec");
        assert!(states.is_empty(), "horizon=0 should give empty rollout");
    }

    #[test]
    fn state_changes_each_step() {
        let (rssm, mut rng) = make_rssm(8);
        let state = rssm.zero_state();
        let action = vec![0.5_f32; 3];
        let s1 = rssm
            .imagine_step(&state, &action, &mut rng)
            .expect("first imagine_step should succeed");
        let s2 = rssm
            .imagine_step(&s1, &action, &mut rng)
            .expect("second imagine_step should succeed");
        // h is deterministic given the same weights; s2.h should differ from s1.h
        let same_h =
            s1.h.iter()
                .zip(s2.h.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(
            !same_h,
            "consecutive imagine steps should produce different h"
        );
    }

    #[test]
    fn prior_posterior_differ() {
        // Build two identical RSSMs (same seed) and compare imagine vs observe z
        let mut rng1 = RlRng::new(42);
        let mut rng2 = RlRng::new(42);
        let config1 = RssmConfig {
            d_deter: 8,
            d_stoch: 4,
            d_obs: 6,
            d_action: 3,
            n_classes: 32,
        };
        let config2 = RssmConfig {
            d_deter: 8,
            d_stoch: 4,
            d_obs: 6,
            d_action: 3,
            n_classes: 32,
        };
        let rssm1 = Rssm::new(config1, &mut rng1).expect("rssm1 init");
        let rssm2 = Rssm::new(config2, &mut rng2).expect("rssm2 init");

        let state1 = rssm1.zero_state();
        let state2 = rssm2.zero_state();
        let action = vec![0.3_f32; 3];
        let obs = vec![2.0_f32; 6]; // non-trivial observation shifts posterior

        let s_prior = rssm1
            .imagine_step(&state1, &action, &mut rng1)
            .expect("prior step should succeed");
        // Use different obs to ensure posterior differs from prior
        let s_post = rssm2
            .observe_step(&state2, &action, &obs, &mut rng2)
            .expect("posterior step should succeed");

        // h is deterministic and should be equal (same weights, same input)
        let h_eq = s_prior
            .h
            .iter()
            .zip(s_post.h.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            h_eq,
            "h from prior and posterior should be identical (same GRU)"
        );

        // z differs because prior params come from h alone while posterior uses obs
        let z_eq = s_prior
            .z
            .iter()
            .zip(s_post.z.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !z_eq,
            "z from prior and posterior should differ (different params + RNG samples)"
        );
    }

    #[test]
    fn d_deter_0_error() {
        let mut rng = RlRng::new(10);
        let config = RssmConfig {
            d_deter: 0,
            d_stoch: 4,
            d_obs: 6,
            d_action: 3,
            n_classes: 32,
        };
        let result = Rssm::new(config, &mut rng);
        assert!(result.is_err(), "d_deter=0 should return an error");
        match result.unwrap_err() {
            RlError::InvalidHyperparameter { name, .. } => {
                assert_eq!(name, "d_deter");
            }
            other => panic!("expected InvalidHyperparameter, got {:?}", other),
        }
    }

    #[test]
    fn action_dim_mismatch_error() {
        let (rssm, mut rng) = make_rssm(11);
        let state = rssm.zero_state();
        let wrong_action = vec![0.0_f32; 5]; // should be 3
        let result = rssm.imagine_step(&state, &wrong_action, &mut rng);
        assert!(result.is_err(), "wrong action dim should error");
    }

    #[test]
    fn rollout_all_states_finite() {
        let (rssm, mut rng) = make_rssm(12);
        let state = rssm.zero_state();
        let horizon = 10;
        let actions = vec![0.1_f32; horizon * 3];
        let states = rssm
            .imagine_rollout(&state, &actions, horizon, &mut rng)
            .expect("rollout should succeed");
        for (t, s) in states.iter().enumerate() {
            for (i, &v) in s.h.iter().enumerate() {
                assert!(v.is_finite(), "rollout t={t} h[{i}]={v} not finite");
            }
            for (i, &v) in s.z.iter().enumerate() {
                assert!(v.is_finite(), "rollout t={t} z[{i}]={v} not finite");
            }
        }
    }
}
