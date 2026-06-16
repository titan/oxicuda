//! # DDPG (Deep Deterministic Policy Gradient) Loss Functions
//!
//! Lillicrap et al. (2016), "Continuous Control with Deep Reinforcement
//! Learning". ICLR 2016. <https://arxiv.org/abs/1509.02971>
//!
//! DDPG is an off-policy actor-critic for continuous action spaces. Unlike its
//! successor TD3 (which uses twin critics, delayed updates, and target-policy
//! smoothing), DDPG uses a **single** critic `Q(s, a)` and a deterministic actor
//! `μ(s)`, trained against slowly-tracking target networks.
//!
//! ## Critic loss (single-Q Bellman MSE/Huber)
//!
//! ```text
//! y_t = r_t + γ·(1 − done_t)·Q'(s_{t+1}, μ'(s_{t+1}))
//! L_Q = mean_t  ℓ( Q(s_t, a_t) − y_t )
//! ```
//!
//! where `ℓ` is squared error (default) or the Huber loss (`κ = 1`).
//!
//! ## Actor loss (deterministic policy gradient)
//!
//! ```text
//! L_μ = − mean_t  Q(s_t, μ(s_t))
//! ```
//!
//! Maximising `Q` along the actor's actions; the supplied `q_pi` values are
//! `Q(s_t, μ(s_t))` evaluated with the actor's *current* actions.
//!
//! ## Polyak (soft) target update
//!
//! ```text
//! θ' ← τ·θ + (1 − τ)·θ'
//! ```
//!
//! provided as [`polyak_update`] for slowly tracking online parameters.

use crate::error::{RlError, RlResult};

// ─── Configuration ──────────────────────────────────────────────────────────────

/// DDPG hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct DdpgConfig {
    /// Discount factor γ (must be in `(0, 1]`).
    pub gamma: f32,
    /// Use the Huber (smooth-L1) critic loss instead of plain MSE.
    pub huber: bool,
    /// Huber threshold κ (must be `> 0`; only used when `huber == true`).
    pub kappa: f32,
}

impl Default for DdpgConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            huber: false,
            kappa: 1.0,
        }
    }
}

// ─── Output ─────────────────────────────────────────────────────────────────────

/// DDPG critic-loss output.
#[derive(Debug, Clone)]
pub struct DdpgCriticLoss {
    /// Mean critic loss (scalar to minimise).
    pub loss: f32,
    /// Per-sample TD errors `Q(s,a) − y` (for PER priority updates). Length `B`.
    pub td_errors: Vec<f32>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

/// Scalar Huber (smooth-L1) loss `ℓ_κ(u)`.
#[inline]
fn huber(u: f32, kappa: f32) -> f32 {
    if u.abs() <= kappa {
        0.5 * u * u
    } else {
        kappa * (u.abs() - 0.5 * kappa)
    }
}

/// Validate the DDPG configuration.
fn validate_cfg(cfg: &DdpgConfig) -> RlResult<()> {
    if cfg.gamma <= 0.0 || cfg.gamma > 1.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "gamma".into(),
            msg: "must be in (0, 1]".into(),
        });
    }
    if cfg.huber && cfg.kappa <= 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "kappa".into(),
            msg: "must be > 0".into(),
        });
    }
    Ok(())
}

// ─── Public API ─────────────────────────────────────────────────────────────────

/// Compute the DDPG single-critic Bellman loss.
///
/// # Arguments
///
/// * `q_sa`        — `[B]` online critic values `Q(s_t, a_t)`.
/// * `rewards`     — `[B]` rewards `r_t`.
/// * `q_next`      — `[B]` target-critic values `Q'(s_{t+1}, μ'(s_{t+1}))`.
/// * `dones`       — `[B]` terminal flags (1.0 = terminal).
/// * `is_weights`  — `[B]` importance-sampling weights (all 1.0 without PER).
/// * `cfg`         — DDPG configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config.
/// * [`RlError::DimensionMismatch`]     — slice lengths inconsistent or empty.
/// * [`RlError::Internal`]              — NaN loss produced.
pub fn ddpg_critic_loss(
    q_sa: &[f32],
    rewards: &[f32],
    q_next: &[f32],
    dones: &[f32],
    is_weights: &[f32],
    cfg: DdpgConfig,
) -> RlResult<DdpgCriticLoss> {
    validate_cfg(&cfg)?;
    let b = q_sa.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    for (name, len) in [
        ("rewards", rewards.len()),
        ("q_next", q_next.len()),
        ("dones", dones.len()),
        ("is_weights", is_weights.len()),
    ] {
        if len != b {
            let _ = name;
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: len,
            });
        }
    }

    let mut td_errors = Vec::with_capacity(b);
    let mut weighted_loss = 0.0_f32;
    for i in 0..b {
        let y = rewards[i] + cfg.gamma * (1.0 - dones[i]) * q_next[i];
        let td = q_sa[i] - y;
        let l = if cfg.huber {
            huber(td, cfg.kappa)
        } else {
            0.5 * td * td
        };
        weighted_loss += is_weights[i] * l;
        td_errors.push(td);
    }
    let loss = weighted_loss / b as f32;
    if loss.is_nan() {
        return Err(RlError::Internal(
            "NaN loss encountered in ddpg_critic_loss".into(),
        ));
    }
    Ok(DdpgCriticLoss { loss, td_errors })
}

/// Compute the DDPG deterministic-policy-gradient actor loss `−mean Q(s, μ(s))`.
///
/// # Arguments
///
/// * `q_pi` — `[B]` critic values evaluated at the actor's *current* actions.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] — `q_pi` is empty.
/// * [`RlError::Internal`]          — NaN loss produced.
pub fn ddpg_actor_loss(q_pi: &[f32]) -> RlResult<f32> {
    if q_pi.is_empty() {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mean_q: f32 = q_pi.iter().copied().sum::<f32>() / q_pi.len() as f32;
    let loss = -mean_q;
    if loss.is_nan() {
        return Err(RlError::Internal(
            "NaN loss encountered in ddpg_actor_loss".into(),
        ));
    }
    Ok(loss)
}

/// Polyak (soft) update of a target parameter vector in place.
///
/// `θ' ← τ·θ + (1 − τ)·θ'` element-wise.
///
/// # Arguments
///
/// * `target` — `[P]` target parameters, updated in place.
/// * `online` — `[P]` online parameters.
/// * `tau`    — interpolation coefficient in `[0, 1]` (small, e.g. 0.005).
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`]     — length mismatch.
/// * [`RlError::InvalidHyperparameter`] — `tau` outside `[0, 1]`.
pub fn polyak_update(target: &mut [f32], online: &[f32], tau: f32) -> RlResult<()> {
    if target.len() != online.len() {
        return Err(RlError::DimensionMismatch {
            expected: target.len(),
            got: online.len(),
        });
    }
    if !(0.0..=1.0).contains(&tau) {
        return Err(RlError::InvalidHyperparameter {
            name: "tau".into(),
            msg: "must be in [0, 1]".into(),
        });
    }
    for (t, &o) in target.iter_mut().zip(online) {
        *t = tau * o + (1.0 - tau) * *t;
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let c = DdpgConfig::default();
        assert!((c.gamma - 0.99).abs() < 1e-6);
        assert!(!c.huber);
        assert!((c.kappa - 1.0).abs() < 1e-6);
    }

    #[test]
    fn critic_loss_zero_when_q_equals_target() {
        // Q(s,a) == y for all samples ⇒ loss 0, td_errors 0.
        let cfg = DdpgConfig::default();
        let rewards = vec![1.0_f32, 0.5, -0.2];
        let q_next = vec![2.0_f32, 1.0, 0.0];
        let dones = vec![0.0_f32; 3];
        // y = r + 0.99 * q_next
        let q_sa: Vec<f32> = rewards
            .iter()
            .zip(&q_next)
            .map(|(&r, &qn)| r + 0.99 * qn)
            .collect();
        let is_w = vec![1.0_f32; 3];
        let out = ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &is_w, cfg).expect("ok");
        assert!(out.loss < 1e-6, "loss should be ~0, got {}", out.loss);
        for &td in &out.td_errors {
            assert!(td.abs() < 1e-5, "td should be ~0, got {td}");
        }
    }

    #[test]
    fn critic_loss_terminal_ignores_bootstrap() {
        // done=1 ⇒ y = reward (no γ·q_next term).
        let cfg = DdpgConfig::default();
        let q_sa = vec![5.0_f32];
        let rewards = vec![3.0_f32];
        let q_next = vec![100.0_f32]; // should be ignored
        let dones = vec![1.0_f32];
        let is_w = vec![1.0_f32];
        let out = ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &is_w, cfg).expect("ok");
        // td = 5 - 3 = 2 ⇒ loss = 0.5 * 4 = 2.0
        assert!((out.td_errors[0] - 2.0).abs() < 1e-5);
        assert!((out.loss - 2.0).abs() < 1e-5, "loss={}", out.loss);
    }

    #[test]
    fn critic_loss_mse_value() {
        let cfg = DdpgConfig {
            gamma: 1.0,
            huber: false,
            kappa: 1.0,
        };
        let q_sa = vec![0.0_f32, 0.0];
        let rewards = vec![2.0_f32, 4.0];
        let q_next = vec![0.0_f32, 0.0];
        let dones = vec![0.0_f32, 0.0];
        let is_w = vec![1.0_f32; 2];
        let out = ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &is_w, cfg).expect("ok");
        // td = [-2, -4]; mse = mean(0.5*4, 0.5*16) = mean(2, 8) = 5
        assert!((out.loss - 5.0).abs() < 1e-5, "loss={}", out.loss);
    }

    #[test]
    fn critic_loss_huber_caps_large_error() {
        let cfg = DdpgConfig {
            gamma: 1.0,
            huber: true,
            kappa: 1.0,
        };
        let q_sa = vec![10.0_f32];
        let rewards = vec![0.0_f32];
        let q_next = vec![0.0_f32];
        let dones = vec![0.0_f32];
        let is_w = vec![1.0_f32];
        let out = ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &is_w, cfg).expect("ok");
        // td=10, |td|>kappa ⇒ huber = 1*(10 - 0.5) = 9.5
        assert!((out.loss - 9.5).abs() < 1e-5, "loss={}", out.loss);
    }

    #[test]
    fn critic_loss_is_weights_scale() {
        let cfg = DdpgConfig {
            gamma: 1.0,
            ..Default::default()
        };
        let q_sa = vec![0.0_f32, 0.0];
        let rewards = vec![2.0_f32, 2.0];
        let q_next = vec![0.0_f32, 0.0];
        let dones = vec![0.0_f32, 0.0];
        // Doubling all IS weights doubles the mean loss.
        let out1 =
            ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &[1.0, 1.0], cfg).expect("ok");
        let out2 =
            ddpg_critic_loss(&q_sa, &rewards, &q_next, &dones, &[2.0, 2.0], cfg).expect("ok");
        assert!((out2.loss - 2.0 * out1.loss).abs() < 1e-5);
    }

    #[test]
    fn critic_loss_td_errors_length() {
        let cfg = DdpgConfig::default();
        let b = 7;
        let out = ddpg_critic_loss(
            &vec![0.0; b],
            &vec![1.0; b],
            &vec![0.0; b],
            &vec![0.0; b],
            &vec![1.0; b],
            cfg,
        )
        .expect("ok");
        assert_eq!(out.td_errors.len(), b);
    }

    #[test]
    fn actor_loss_is_negative_mean_q() {
        let q_pi = vec![1.0_f32, 2.0, 3.0];
        let l = ddpg_actor_loss(&q_pi).expect("ok");
        assert!((l - (-2.0)).abs() < 1e-6, "expected -2, got {l}");
    }

    #[test]
    fn actor_loss_higher_q_means_lower_loss() {
        let low = ddpg_actor_loss(&[1.0, 1.0]).expect("ok");
        let high = ddpg_actor_loss(&[5.0, 5.0]).expect("ok");
        assert!(
            high < low,
            "higher Q should give lower (more negative) loss"
        );
    }

    #[test]
    fn polyak_tau_zero_keeps_target() {
        let mut target = vec![1.0_f32, 2.0, 3.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        polyak_update(&mut target, &online, 0.0).expect("ok");
        assert_eq!(target, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn polyak_tau_one_copies_online() {
        let mut target = vec![1.0_f32, 2.0, 3.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        polyak_update(&mut target, &online, 1.0).expect("ok");
        assert_eq!(target, online);
    }

    #[test]
    fn polyak_interpolates() {
        let mut target = vec![0.0_f32];
        let online = vec![10.0_f32];
        polyak_update(&mut target, &online, 0.1).expect("ok");
        // 0.1*10 + 0.9*0 = 1.0
        assert!((target[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn err_empty_critic_input() {
        let cfg = DdpgConfig::default();
        assert!(matches!(
            ddpg_critic_loss(&[], &[], &[], &[], &[], cfg),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_critic_dim_mismatch() {
        let cfg = DdpgConfig::default();
        assert!(matches!(
            ddpg_critic_loss(
                &[1.0, 2.0],
                &[1.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[1.0, 1.0],
                cfg
            ),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_empty_actor_input() {
        assert!(matches!(
            ddpg_actor_loss(&[]),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_invalid_gamma() {
        let cfg = DdpgConfig {
            gamma: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            ddpg_critic_loss(&[0.0], &[0.0], &[0.0], &[0.0], &[1.0], cfg),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_polyak_dim_mismatch() {
        let mut target = vec![1.0_f32, 2.0];
        assert!(matches!(
            polyak_update(&mut target, &[1.0], 0.5),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_polyak_invalid_tau() {
        let mut target = vec![1.0_f32];
        assert!(matches!(
            polyak_update(&mut target, &[1.0], 1.5),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }
}
