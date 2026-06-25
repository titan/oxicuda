//! # Offline (Batch) Reinforcement Learning Losses
//!
//! Offline RL learns a policy purely from a fixed dataset `D` of transitions
//! without further environment interaction. The central difficulty is
//! *distributional shift*: the learned policy may query the value function on
//! state–action pairs absent from `D`, where the Q-estimates are unreliably
//! extrapolated. The algorithms below counteract this with three distinct
//! mechanisms — value conservatism, expectile/in-sample bootstrapping, and
//! advantage-weighted policy extraction.
//!
//! ## CQL — Conservative Q-Learning (Kumar et al. 2020)
//!
//! Augments the standard Bellman error with a *conservative* regulariser that
//! pushes **down** Q-values on out-of-distribution actions and pulls **up**
//! Q-values on the dataset actions:
//! ```text
//! L_CQL = α · ( log Σ_a exp(Q(s,a))  −  E_{a~D}[Q(s,a)] )  +  L_Bellman
//! ```
//! The `logsumexp` is a soft-max over *all* candidate actions (a uniform / OOD
//! set); subtracting the data-action value yields a lower bound on the true
//! value, so the learned policy cannot exploit erroneously-high OOD estimates.
//!
//! ## IQL — Implicit Q-Learning (Kostrikov et al. 2021)
//!
//! Avoids ever evaluating Q on actions outside the dataset. A state-value
//! `V(s)` is fit to an **expectile** `τ ∈ (0.5, 1)` of `Q(s,a)`:
//! ```text
//! L_V = E_{(s,a)~D} [ |τ − 1(u < 0)| · u² ],   u = Q(s,a) − V(s)
//! ```
//! The asymmetric weight makes `V(s)` approximate `max_a Q` from in-sample
//! actions only. The critic then bootstraps off `V`:
//! ```text
//! L_Q = E [ (Q(s,a) − (r + γ (1-done) V(s')))² ]
//! ```
//!
//! ## AWAC / AWR — Advantage-Weighted Actor-Critic (Nair et al. 2020)
//!
//! Extracts a policy by *weighted* maximum-likelihood on the dataset actions,
//! up-weighting actions whose advantage is positive:
//! ```text
//! L_π = −E_{(s,a)~D} [ log π(a|s) · exp( (Q(s,a) − V(s)) / λ ) ]
//! ```
//! with the exponential weights optionally clamped to a maximum for stability.
//!
//! ## BCQ — Batch-Constrained Q-Learning (Fujimoto et al. 2019)
//!
//! Constrains the policy to actions close to the dataset by sampling
//! candidates from a generative model and selecting the highest-valued
//! perturbed action. The trainable pieces exposed here are the **soft
//! clipped-double-Q** target,
//! ```text
//! Q_tgt = λ · min(Q1', Q2') + (1 − λ) · max(Q1', Q2')
//! ```
//! and the generative-model (cVAE) evidence lower bound
//! ```text
//! L_VAE = ‖a − â‖²  +  β · D_KL( N(μ, σ²) ‖ N(0, I) ).
//! ```

use crate::error::{RlError, RlResult};

// ─── shared helpers ──────────────────────────────────────────────────────────

/// Numerically-stable `log Σ_i exp(x_i)` over a slice.
#[inline]
fn logsumexp(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    let s: f32 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// Validate that every length in `lens` equals `b`, else `DimensionMismatch`.
#[inline]
fn check_lengths(b: usize, lens: &[usize]) -> RlResult<()> {
    for &l in lens {
        if l != b {
            return Err(RlError::DimensionMismatch {
                expected: b,
                got: l,
            });
        }
    }
    Ok(())
}

// ─── CQL ──────────────────────────────────────────────────────────────────────

/// Conservative Q-Learning hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct CqlConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Conservative penalty weight α (the trade-off vs. the Bellman term).
    pub cql_alpha: f32,
    /// Huber κ for the Bellman term (`None` ⇒ squared error).
    pub huber_kappa: Option<f32>,
}

impl Default for CqlConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            cql_alpha: 1.0,
            huber_kappa: None,
        }
    }
}

impl CqlConfig {
    /// Validate ranges.
    ///
    /// # Errors
    /// [`RlError::InvalidHyperparameter`] if `gamma ∉ [0,1]` or `cql_alpha < 0`.
    pub fn validate(&self) -> RlResult<()> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if self.cql_alpha < 0.0 || !self.cql_alpha.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "cql_alpha".into(),
                msg: "must be finite and >= 0".into(),
            });
        }
        Ok(())
    }
}

/// Decomposed CQL loss terms.
#[derive(Debug, Clone)]
pub struct CqlLoss {
    /// Total loss: `cql_alpha * conservative_gap + bellman_loss`.
    pub total: f32,
    /// Mean Bellman (TD) loss component.
    pub bellman_loss: f32,
    /// Mean conservative gap `logsumexp_a Q(s,a) − E_{a~D}[Q(s,a)]` (≥ 0).
    pub conservative_gap: f32,
    /// Per-sample absolute TD errors (for PER priority updates).
    pub td_errors: Vec<f32>,
}

#[inline]
fn huber(delta: f32, kappa: f32) -> f32 {
    if delta.abs() <= kappa {
        0.5 * delta * delta
    } else {
        kappa * (delta.abs() - 0.5 * kappa)
    }
}

/// Conservative Q-Learning loss with the `H(Q)`-style `logsumexp` regulariser.
///
/// # Arguments
/// * `q_sa`         — `[B]` Q(s_t, a_t) for the dataset actions.
/// * `q_all`        — `[B × A]` Q(s_t, ·) over all candidate actions (for the
///   `logsumexp`); flattened row-major.
/// * `n_actions`    — number of candidate actions `A`.
/// * `rewards`      — `[B]`.
/// * `max_q_next`   — `[B]` `max_a Q_target(s_{t+1}, a)`.
/// * `dones`        — `[B]`.
/// * `cfg`          — CQL configuration.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch; config validation errors.
#[allow(clippy::too_many_arguments)]
pub fn cql_loss(
    q_sa: &[f32],
    q_all: &[f32],
    n_actions: usize,
    rewards: &[f32],
    max_q_next: &[f32],
    dones: &[f32],
    cfg: CqlConfig,
) -> RlResult<CqlLoss> {
    cfg.validate()?;
    let b = q_sa.len();
    if n_actions == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_actions".into(),
            msg: "must be > 0".into(),
        });
    }
    check_lengths(b, &[rewards.len(), max_q_next.len(), dones.len()])?;
    if q_all.len() != b * n_actions {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: q_all.len(),
        });
    }

    let mut bellman = 0.0_f32;
    let mut gap = 0.0_f32;
    let mut td_errors = Vec::with_capacity(b);
    for i in 0..b {
        // Bellman term.
        let target = rewards[i] + cfg.gamma * max_q_next[i] * (1.0 - dones[i]);
        let delta = target - q_sa[i];
        td_errors.push(delta.abs());
        bellman += match cfg.huber_kappa {
            Some(k) => huber(delta, k),
            None => 0.5 * delta * delta,
        };
        // Conservative term: logsumexp_a Q(s,a) − Q(s, a_data).
        let row = &q_all[i * n_actions..(i + 1) * n_actions];
        gap += logsumexp(row) - q_sa[i];
    }
    let bn = b as f32;
    let bellman_loss = bellman / bn;
    let conservative_gap = gap / bn;
    Ok(CqlLoss {
        total: cfg.cql_alpha * conservative_gap + bellman_loss,
        bellman_loss,
        conservative_gap,
        td_errors,
    })
}

// ─── IQL ──────────────────────────────────────────────────────────────────────

/// Implicit Q-Learning hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct IqlConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Expectile `τ ∈ (0.5, 1)` for the value loss (closer to 1 ⇒ more
    /// optimistic in-sample max).
    pub expectile: f32,
    /// Inverse-temperature `β` for advantage-weighted policy extraction.
    pub beta: f32,
    /// Clamp on the exponential advantage weight `exp(β·A)` (numerical safety).
    pub max_weight: f32,
}

impl Default for IqlConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            expectile: 0.7,
            beta: 3.0,
            max_weight: 100.0,
        }
    }
}

impl IqlConfig {
    /// Validate ranges.
    ///
    /// # Errors
    /// [`RlError::InvalidHyperparameter`] if `expectile ∉ (0,1)`, `gamma ∉ [0,1]`,
    /// `beta <= 0`, or `max_weight <= 0`.
    pub fn validate(&self) -> RlResult<()> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if !(0.0..1.0).contains(&self.expectile) || self.expectile == 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "expectile".into(),
                msg: "must be in (0, 1)".into(),
            });
        }
        if self.beta <= 0.0 || !self.beta.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "beta".into(),
                msg: "must be finite and > 0".into(),
            });
        }
        if self.max_weight <= 0.0 || !self.max_weight.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "max_weight".into(),
                msg: "must be finite and > 0".into(),
            });
        }
        Ok(())
    }
}

/// Asymmetric (expectile) least-squares weight for residual `u` at level `τ`.
///
/// `w = |τ − 1(u < 0)|`. For `u ≥ 0` the weight is `τ`; for `u < 0` it is
/// `1 − τ`. With `τ > 0.5` positive residuals (under-estimates of `V`) are
/// penalised more, pulling `V` toward the in-sample maximum of `Q`.
#[inline]
pub fn expectile_weight(u: f32, tau: f32) -> f32 {
    if u >= 0.0 { tau } else { 1.0 - tau }
}

/// IQL **value** loss — expectile regression of `V(s)` toward `Q(s,a)`.
///
/// ```text
/// L_V = mean_i  |τ − 1(u_i < 0)| · u_i²,   u_i = Q(s_i,a_i) − V(s_i)
/// ```
///
/// # Arguments
/// * `q_sa` — `[B]` target-Q values for dataset actions (treated as constants).
/// * `v_s`  — `[B]` current value estimates.
/// * `cfg`  — IQL configuration.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch; config errors.
pub fn iql_value_loss(q_sa: &[f32], v_s: &[f32], cfg: IqlConfig) -> RlResult<f32> {
    cfg.validate()?;
    let b = q_sa.len();
    check_lengths(b, &[v_s.len()])?;
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mut loss = 0.0_f32;
    for i in 0..b {
        let u = q_sa[i] - v_s[i];
        loss += expectile_weight(u, cfg.expectile) * u * u;
    }
    Ok(loss / b as f32)
}

/// IQL **critic** loss — standard Bellman MSE bootstrapping off `V(s')`.
///
/// ```text
/// target_i = r_i + γ (1−done_i) V(s'_i)
/// L_Q      = mean_i (Q(s_i,a_i) − target_i)²
/// ```
///
/// # Arguments
/// * `q_sa`    — `[B]` online Q(s,a).
/// * `rewards` — `[B]`.
/// * `v_next`  — `[B]` value of next state `V(s')` (target value network).
/// * `dones`   — `[B]`.
/// * `cfg`     — IQL configuration.
///
/// Returns `(mean_loss, per_sample_td_errors)`.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch; config errors.
pub fn iql_critic_loss(
    q_sa: &[f32],
    rewards: &[f32],
    v_next: &[f32],
    dones: &[f32],
    cfg: IqlConfig,
) -> RlResult<(f32, Vec<f32>)> {
    cfg.validate()?;
    let b = q_sa.len();
    check_lengths(b, &[rewards.len(), v_next.len(), dones.len()])?;
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mut loss = 0.0_f32;
    let mut td = Vec::with_capacity(b);
    for i in 0..b {
        let target = rewards[i] + cfg.gamma * (1.0 - dones[i]) * v_next[i];
        let delta = q_sa[i] - target;
        td.push(delta.abs());
        loss += delta * delta;
    }
    Ok((loss / b as f32, td))
}

/// Advantage-weighted **policy-extraction** loss shared by IQL and AWAC/AWR.
///
/// ```text
/// w_i = min( exp(β · (Q_i − V_i)), max_weight )
/// L_π = −mean_i  w_i · log π(a_i | s_i)
/// ```
///
/// With `normalize_weights = true` the weights are divided by their batch mean
/// (the AWR/AWAC convention), which keeps the effective learning-rate scale
/// independent of the advantage magnitude.
///
/// # Arguments
/// * `log_pi`  — `[B]` log π(a|s) for dataset actions.
/// * `q_sa`    — `[B]` Q(s,a).
/// * `v_s`     — `[B]` V(s) baseline.
/// * `beta`    — inverse temperature β (> 0).
/// * `max_weight` — clamp on `exp(β·A)`.
/// * `normalize_weights` — divide weights by their mean.
///
/// Returns `(loss, mean_weight)`.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch;
/// [`RlError::InvalidHyperparameter`] for non-positive `beta` / `max_weight`.
pub fn advantage_weighted_policy_loss(
    log_pi: &[f32],
    q_sa: &[f32],
    v_s: &[f32],
    beta: f32,
    max_weight: f32,
    normalize_weights: bool,
) -> RlResult<(f32, f32)> {
    let b = log_pi.len();
    check_lengths(b, &[q_sa.len(), v_s.len()])?;
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if beta <= 0.0 || !beta.is_finite() {
        return Err(RlError::InvalidHyperparameter {
            name: "beta".into(),
            msg: "must be finite and > 0".into(),
        });
    }
    if max_weight <= 0.0 || !max_weight.is_finite() {
        return Err(RlError::InvalidHyperparameter {
            name: "max_weight".into(),
            msg: "must be finite and > 0".into(),
        });
    }
    let mut weights = Vec::with_capacity(b);
    for i in 0..b {
        let adv = q_sa[i] - v_s[i];
        let w = (beta * adv).exp().min(max_weight);
        weights.push(w);
    }
    let mean_w = weights.iter().sum::<f32>() / b as f32;
    let denom = if normalize_weights && mean_w > 1e-8 {
        mean_w
    } else {
        1.0
    };
    let mut loss = 0.0_f32;
    for i in 0..b {
        loss += -(weights[i] / denom) * log_pi[i];
    }
    Ok((loss / b as f32, mean_w))
}

// ─── AWAC ──────────────────────────────────────────────────────────────────────

/// AWAC hyperparameters (Lagrangian temperature λ and weight clamp).
#[derive(Debug, Clone, Copy)]
pub struct AwacConfig {
    /// Lagrangian temperature `λ` (the advantage is divided by it). Larger λ ⇒
    /// softer, more uniform weighting.
    pub lambda: f32,
    /// Clamp on the exponential advantage weight.
    pub max_weight: f32,
    /// Whether to normalise weights by their batch mean (AWR convention).
    pub normalize_weights: bool,
}

impl Default for AwacConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            max_weight: 20.0,
            normalize_weights: false,
        }
    }
}

impl AwacConfig {
    /// Validate ranges.
    ///
    /// # Errors
    /// [`RlError::InvalidHyperparameter`] for non-positive `lambda`/`max_weight`.
    pub fn validate(&self) -> RlResult<()> {
        if self.lambda <= 0.0 || !self.lambda.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "lambda".into(),
                msg: "must be finite and > 0".into(),
            });
        }
        if self.max_weight <= 0.0 || !self.max_weight.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "max_weight".into(),
                msg: "must be finite and > 0".into(),
            });
        }
        Ok(())
    }
}

/// AWAC actor loss — advantage-weighted maximum-likelihood with `A = Q − V`.
///
/// Internally this is [`advantage_weighted_policy_loss`] with `β = 1/λ`.
///
/// # Arguments
/// * `log_pi` — `[B]` log π(a|s).
/// * `q_sa`   — `[B]` Q(s,a).
/// * `v_s`    — `[B]` V(s) baseline.
/// * `cfg`    — AWAC configuration.
///
/// Returns `(loss, mean_weight)`.
///
/// # Errors
/// As [`advantage_weighted_policy_loss`]; plus config validation.
pub fn awac_actor_loss(
    log_pi: &[f32],
    q_sa: &[f32],
    v_s: &[f32],
    cfg: AwacConfig,
) -> RlResult<(f32, f32)> {
    cfg.validate()?;
    advantage_weighted_policy_loss(
        log_pi,
        q_sa,
        v_s,
        1.0 / cfg.lambda,
        cfg.max_weight,
        cfg.normalize_weights,
    )
}

// ─── BCQ ──────────────────────────────────────────────────────────────────────

/// Batch-Constrained Q-Learning hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct BcqConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Soft clipped-double-Q mixing weight `λ ∈ [0,1]` (`1` ⇒ pure `min`, the
    /// most conservative; BCQ default `0.75`).
    pub lambda: f32,
    /// β coefficient on the cVAE KL term.
    pub vae_beta: f32,
}

impl Default for BcqConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.75,
            vae_beta: 0.5,
        }
    }
}

impl BcqConfig {
    /// Validate ranges.
    ///
    /// # Errors
    /// [`RlError::InvalidHyperparameter`] if `gamma ∉ [0,1]`, `lambda ∉ [0,1]`,
    /// or `vae_beta < 0`.
    pub fn validate(&self) -> RlResult<()> {
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(RlError::InvalidHyperparameter {
                name: "gamma".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.lambda) {
            return Err(RlError::InvalidHyperparameter {
                name: "lambda".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if self.vae_beta < 0.0 || !self.vae_beta.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "vae_beta".into(),
                msg: "must be finite and >= 0".into(),
            });
        }
        Ok(())
    }
}

/// BCQ soft clipped-double-Q **bootstrap target**.
///
/// ```text
/// Q_tgt_i = λ · min(Q1'_i, Q2'_i) + (1 − λ) · max(Q1'_i, Q2'_i)
/// y_i     = r_i + γ (1 − done_i) Q_tgt_i
/// ```
/// The candidate next-action values `q1_next` / `q2_next` are assumed to have
/// already been *max-reduced* over the perturbation-model action candidates by
/// the caller.
///
/// # Arguments
/// * `q1_next`, `q2_next` — `[B]` twin target-Q values at the chosen next action.
/// * `rewards`, `dones`   — `[B]`.
/// * `cfg`                — BCQ configuration.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch; config errors.
pub fn bcq_target(
    q1_next: &[f32],
    q2_next: &[f32],
    rewards: &[f32],
    dones: &[f32],
    cfg: BcqConfig,
) -> RlResult<Vec<f32>> {
    cfg.validate()?;
    let b = q1_next.len();
    check_lengths(b, &[q2_next.len(), rewards.len(), dones.len()])?;
    let mut targets = Vec::with_capacity(b);
    for i in 0..b {
        let mn = q1_next[i].min(q2_next[i]);
        let mx = q1_next[i].max(q2_next[i]);
        let soft = cfg.lambda * mn + (1.0 - cfg.lambda) * mx;
        targets.push(rewards[i] + cfg.gamma * (1.0 - dones[i]) * soft);
    }
    Ok(targets)
}

/// BCQ critic loss given a precomputed bootstrap `target`.
///
/// `L_Q = mean_i [ (Q1_i − y_i)² + (Q2_i − y_i)² ]`.
///
/// # Arguments
/// * `q1_sa`, `q2_sa` — `[B]` online twin-Q at the dataset action.
/// * `target`         — `[B]` bootstrap target (e.g. from [`bcq_target`]).
///
/// Returns `(mean_loss, per_sample_td_errors)` where the TD error uses `Q1`.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch.
pub fn bcq_critic_loss(q1_sa: &[f32], q2_sa: &[f32], target: &[f32]) -> RlResult<(f32, Vec<f32>)> {
    let b = q1_sa.len();
    check_lengths(b, &[q2_sa.len(), target.len()])?;
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mut loss = 0.0_f32;
    let mut td = Vec::with_capacity(b);
    for i in 0..b {
        let d1 = q1_sa[i] - target[i];
        let d2 = q2_sa[i] - target[i];
        td.push(d1.abs());
        loss += d1 * d1 + d2 * d2;
    }
    Ok((loss / b as f32, td))
}

/// BCQ generative-model (conditional VAE) evidence-lower-bound loss.
///
/// ```text
/// L_VAE = mean( ‖a − â‖² )  +  β · D_KL( N(μ, σ²) ‖ N(0, I) )
/// D_KL  = ½ Σ_j ( μ_j² + σ_j² − ln σ_j² − 1 )
/// ```
///
/// The cVAE clones the behaviour policy so the perturbation actor stays within
/// the data manifold.
///
/// # Arguments
/// * `recon`     — `[B × A]` reconstructed actions `â`.
/// * `actions`   — `[B × A]` true dataset actions `a`.
/// * `mean`      — `[B × Z]` latent mean `μ`.
/// * `log_var`   — `[B × Z]` latent log-variance `ln σ²`.
/// * `act_dim`   — action dimensionality `A`.
/// * `latent_dim`— latent dimensionality `Z`.
/// * `beta`      — KL weight.
///
/// Returns `(total, recon_loss, kl_loss)`.
///
/// # Errors
/// [`RlError::DimensionMismatch`] on length mismatch;
/// [`RlError::InvalidHyperparameter`] for `beta < 0` or zero dims.
#[allow(clippy::too_many_arguments)]
pub fn bcq_vae_loss(
    recon: &[f32],
    actions: &[f32],
    mean: &[f32],
    log_var: &[f32],
    act_dim: usize,
    latent_dim: usize,
    beta: f32,
) -> RlResult<(f32, f32, f32)> {
    if act_dim == 0 || latent_dim == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "act_dim/latent_dim".into(),
            msg: "must be > 0".into(),
        });
    }
    if beta < 0.0 || !beta.is_finite() {
        return Err(RlError::InvalidHyperparameter {
            name: "beta".into(),
            msg: "must be finite and >= 0".into(),
        });
    }
    if recon.len() != actions.len() {
        return Err(RlError::DimensionMismatch {
            expected: actions.len(),
            got: recon.len(),
        });
    }
    if recon.len() % act_dim != 0 {
        return Err(RlError::DimensionMismatch {
            expected: act_dim,
            got: recon.len(),
        });
    }
    let b = recon.len() / act_dim;
    if mean.len() != b * latent_dim || log_var.len() != b * latent_dim {
        return Err(RlError::DimensionMismatch {
            expected: b * latent_dim,
            got: mean.len(),
        });
    }
    // Reconstruction: mean over batch of the per-sample summed squared error.
    let mut recon_sum = 0.0_f32;
    for i in 0..recon.len() {
        let d = recon[i] - actions[i];
        recon_sum += d * d;
    }
    let recon_loss = recon_sum / b as f32;
    // KL: ½ Σ (μ² + σ² − ln σ² − 1), averaged over batch.
    let mut kl_sum = 0.0_f32;
    for i in 0..mean.len() {
        let lv = log_var[i];
        let var = lv.exp();
        kl_sum += 0.5 * (mean[i] * mean[i] + var - lv - 1.0);
    }
    let kl_loss = kl_sum / b as f32;
    Ok((recon_loss + beta * kl_loss, recon_loss, kl_loss))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn logsumexp_matches_manual() {
        // logsumexp([0, ln2]) = ln(1 + 2) = ln 3.
        let v = logsumexp(&[0.0, 2.0_f32.ln()]);
        assert!((v - 3.0_f32.ln()).abs() < 1e-6, "logsumexp={v}");
    }

    #[test]
    fn logsumexp_ge_max() {
        let xs = [-3.0_f32, 1.5, 0.2, -0.7];
        let lse = logsumexp(&xs);
        let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(lse >= mx - 1e-6, "lse {lse} < max {mx}");
    }

    // ── CQL ──
    #[test]
    fn cql_conservative_gap_nonnegative() {
        // logsumexp_a Q ≥ Q(s, a_data) ⇒ gap ≥ 0 always.
        let q_sa = vec![0.5_f32, -1.0, 2.0];
        // 2 actions per state; include the data value somewhere in the row.
        let q_all = vec![
            0.5, 0.3, // state 0
            -1.0, -2.0, // state 1
            2.0, 1.0, // state 2
        ];
        let r = vec![0.0_f32; 3];
        let qn = vec![0.0_f32; 3];
        let d = vec![0.0_f32; 3];
        let l = cql_loss(&q_sa, &q_all, 2, &r, &qn, &d, CqlConfig::default())
            .expect("valid CQL inputs should compute");
        assert!(
            l.conservative_gap >= -1e-6,
            "gap should be >= 0, got {}",
            l.conservative_gap
        );
    }

    #[test]
    fn cql_alpha_zero_equals_bellman() {
        // With α = 0 the total must equal the plain Bellman loss.
        let q_sa = vec![0.0_f32, 0.0];
        let q_all = vec![0.0, 5.0, 0.0, 5.0]; // big OOD values, but α=0 ignores them
        let r = vec![1.0_f32, 1.0];
        let qn = vec![1.0_f32, 1.0];
        let d = vec![0.0_f32, 0.0];
        let cfg = CqlConfig {
            gamma: 1.0,
            cql_alpha: 0.0,
            huber_kappa: None,
        };
        let l = cql_loss(&q_sa, &q_all, 2, &r, &qn, &d, cfg).expect("valid");
        // target = 1 + 1 = 2; delta = 2; 0.5*4 = 2.0 mean.
        assert!((l.total - l.bellman_loss).abs() < 1e-6);
        assert!((l.bellman_loss - 2.0).abs() < 1e-5, "{}", l.bellman_loss);
    }

    #[test]
    fn cql_penalty_raises_loss() {
        // Larger OOD Q-values ⇒ larger conservative gap ⇒ larger total.
        let q_sa = vec![0.0_f32];
        let r = vec![0.0_f32];
        let qn = vec![0.0_f32];
        let d = vec![0.0_f32];
        let cfg = CqlConfig {
            cql_alpha: 1.0,
            ..CqlConfig::default()
        };
        let low = cql_loss(&q_sa, &[0.0, 0.0], 2, &r, &qn, &d, cfg).expect("ok");
        let high = cql_loss(&q_sa, &[0.0, 5.0], 2, &r, &qn, &d, cfg).expect("ok");
        assert!(
            high.total > low.total,
            "OOD penalty should raise loss: {} vs {}",
            high.total,
            low.total
        );
    }

    #[test]
    fn cql_dim_mismatch_errs() {
        let q_sa = vec![0.0_f32; 2];
        let q_all = vec![0.0_f32; 3]; // not 2*A
        let r = vec![0.0_f32; 2];
        let qn = vec![0.0_f32; 2];
        let d = vec![0.0_f32; 2];
        assert!(cql_loss(&q_sa, &q_all, 2, &r, &qn, &d, CqlConfig::default()).is_err());
    }

    // ── IQL ──
    #[test]
    fn expectile_weight_asymmetric() {
        let tau = 0.7_f32;
        assert!((expectile_weight(1.0, tau) - 0.7).abs() < 1e-6);
        assert!((expectile_weight(-1.0, tau) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn iql_value_loss_symmetric_at_half() {
        // At τ = 0.5 the expectile loss is exactly ½·MSE.
        let q = vec![1.0_f32, -1.0, 2.0];
        let v = vec![0.0_f32, 0.0, 0.0];
        let cfg = IqlConfig {
            expectile: 0.5,
            ..IqlConfig::default()
        };
        let l = iql_value_loss(&q, &v, cfg).expect("ok");
        // 0.5 * mean(1 + 1 + 4) = 0.5 * 2 = 1.0
        assert!((l - 1.0).abs() < 1e-5, "{l}");
    }

    #[test]
    fn iql_value_pulls_up_with_high_expectile() {
        // With τ → 1, V is pushed toward the in-sample max of Q. Optimising V by
        // a few gradient-free steps should monotonically reduce the expectile
        // loss while raising V above the mean of Q.
        let q = vec![0.0_f32, 1.0, 2.0, 3.0]; // mean 1.5, max 3.0
        let cfg = IqlConfig {
            expectile: 0.9,
            ..IqlConfig::default()
        };
        // Subgradient descent on a scalar V shared across the batch.
        let mut v = 0.0_f32;
        let lr = 0.1_f32;
        let v_vec = |val: f32| vec![val; q.len()];
        let mut last = iql_value_loss(&q, &v_vec(v), cfg).expect("ok");
        for _ in 0..200 {
            // d/dV of mean w(u)·u²  where u = q − V ⇒ grad = mean(-2 w u).
            let mut g = 0.0_f32;
            for &qi in &q {
                let u = qi - v;
                g += -2.0 * expectile_weight(u, cfg.expectile) * u;
            }
            g /= q.len() as f32;
            v -= lr * g;
            let now = iql_value_loss(&q, &v_vec(v), cfg).expect("ok");
            assert!(now <= last + 1e-4, "loss must not increase: {now} > {last}");
            last = now;
        }
        // τ=0.9 expectile of {0,1,2,3} sits well above the mean (1.5).
        assert!(v > 1.5, "V should exceed the mean toward the max, got {v}");
        assert!(v < 3.0 + 1e-3, "V should not exceed the max, got {v}");
    }

    #[test]
    fn iql_critic_target_uses_v_next() {
        let q = vec![0.0_f32];
        let r = vec![1.0_f32];
        let vn = vec![2.0_f32];
        let d = vec![0.0_f32];
        let cfg = IqlConfig {
            gamma: 1.0,
            ..IqlConfig::default()
        };
        let (loss, td) = iql_critic_loss(&q, &r, &vn, &d, cfg).expect("ok");
        // target = 1 + 2 = 3; delta = -3; loss = 9.
        assert!((loss - 9.0).abs() < 1e-4, "{loss}");
        assert!((td[0] - 3.0).abs() < 1e-5);
    }

    // ── advantage-weighted policy ──
    #[test]
    fn adv_weighted_prefers_positive_advantage() {
        // Two samples: one with high advantage, one negative. The high-advantage
        // log-prob should dominate the (negated) loss gradient.
        let log_pi = vec![-1.0_f32, -1.0];
        let q = vec![2.0_f32, 0.0];
        let v = vec![0.0_f32, 1.0]; // advantages +2 and −1
        let (loss, mean_w) =
            advantage_weighted_policy_loss(&log_pi, &q, &v, 1.0, 100.0, false).expect("ok");
        // w0 = e^2 ≈ 7.389, w1 = e^{-1} ≈ 0.368.
        // loss = −mean(w·log_pi) = −mean(7.389·(−1) + 0.368·(−1)) = mean(7.757)/2 ≈ 3.878
        assert!((mean_w - ((2.0_f32).exp() + (-1.0_f32).exp()) / 2.0).abs() < 1e-3);
        assert!(loss > 0.0, "loss {loss}");
    }

    #[test]
    fn adv_weighted_clamp_caps_weight() {
        let log_pi = vec![-1.0_f32];
        let q = vec![100.0_f32];
        let v = vec![0.0_f32];
        // exp(100) would overflow; clamp to 5.
        let (loss, mean_w) =
            advantage_weighted_policy_loss(&log_pi, &q, &v, 1.0, 5.0, false).expect("ok");
        assert!((mean_w - 5.0).abs() < 1e-4, "mean_w={mean_w}");
        assert!(loss.is_finite() && (loss - 5.0).abs() < 1e-4, "loss={loss}");
    }

    #[test]
    fn adv_weighted_normalize_unit_mean() {
        // After normalisation by mean weight, the effective weights average 1.
        let log_pi = vec![-2.0_f32, -2.0, -2.0];
        let q = vec![1.0_f32, 2.0, 3.0];
        let v = vec![0.0_f32, 0.0, 0.0];
        let (loss, _) =
            advantage_weighted_policy_loss(&log_pi, &q, &v, 1.0, 1e9, true).expect("ok");
        // normalised weights average 1 ⇒ loss = −mean(1·log_pi)·(scaling) but per
        // sample weights differ; total = −mean(w_i/mean_w · log_pi). With all
        // log_pi equal to −2: loss = 2 · mean(w_i/mean_w) = 2 · 1 = 2.
        assert!((loss - 2.0).abs() < 1e-4, "loss={loss}");
    }

    // ── AWAC ──
    #[test]
    fn awac_is_adv_weighted_with_inv_lambda() {
        let log_pi = vec![-1.0_f32, -0.5];
        let q = vec![1.0_f32, 0.5];
        let v = vec![0.0_f32, 1.0];
        let cfg = AwacConfig {
            lambda: 2.0,
            max_weight: 100.0,
            normalize_weights: false,
        };
        let (la, _) = awac_actor_loss(&log_pi, &q, &v, cfg).expect("ok");
        let (lb, _) =
            advantage_weighted_policy_loss(&log_pi, &q, &v, 0.5, 100.0, false).expect("ok");
        assert!((la - lb).abs() < 1e-6, "{la} vs {lb}");
    }

    #[test]
    fn awac_invalid_lambda_errs() {
        let cfg = AwacConfig {
            lambda: 0.0,
            ..AwacConfig::default()
        };
        assert!(awac_actor_loss(&[-1.0], &[1.0], &[0.0], cfg).is_err());
    }

    // ── BCQ ──
    #[test]
    fn bcq_target_soft_min_interpolation() {
        // λ=1 ⇒ min; λ=0 ⇒ max; λ=0.5 ⇒ average.
        let q1 = vec![1.0_f32];
        let q2 = vec![3.0_f32];
        let r = vec![0.0_f32];
        let d = vec![0.0_f32];
        let base = BcqConfig {
            gamma: 1.0,
            vae_beta: 0.5,
            lambda: 1.0,
        };
        let t_min = bcq_target(&q1, &q2, &r, &d, base).expect("ok");
        let t_max = bcq_target(
            &q1,
            &q2,
            &r,
            &d,
            BcqConfig {
                lambda: 0.0,
                ..base
            },
        )
        .expect("ok");
        let t_mid = bcq_target(
            &q1,
            &q2,
            &r,
            &d,
            BcqConfig {
                lambda: 0.5,
                ..base
            },
        )
        .expect("ok");
        assert!((t_min[0] - 1.0).abs() < 1e-6);
        assert!((t_max[0] - 3.0).abs() < 1e-6);
        assert!((t_mid[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn bcq_target_done_masks_bootstrap() {
        let q1 = vec![5.0_f32];
        let q2 = vec![5.0_f32];
        let r = vec![1.0_f32];
        let d = vec![1.0_f32]; // terminal ⇒ no bootstrap
        let t = bcq_target(&q1, &q2, &r, &d, BcqConfig::default()).expect("ok");
        assert!((t[0] - 1.0).abs() < 1e-6, "{}", t[0]);
    }

    #[test]
    fn bcq_critic_loss_both_heads() {
        let q1 = vec![0.0_f32, 0.0];
        let q2 = vec![2.0_f32, 2.0];
        let tgt = vec![1.0_f32, 1.0];
        let (loss, td) = bcq_critic_loss(&q1, &q2, &tgt).expect("ok");
        // per sample: (0-1)^2 + (2-1)^2 = 2; mean = 2.
        assert!((loss - 2.0).abs() < 1e-5, "{loss}");
        assert!((td[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bcq_vae_kl_zero_at_standard_normal() {
        // μ=0, ln σ²=0 (σ²=1) ⇒ KL = 0; perfect recon ⇒ recon = 0.
        // batch = 2, act_dim = 1, latent_dim = 1.
        let recon = vec![0.5_f32, -0.5];
        let actions = vec![0.5_f32, -0.5];
        let mean = vec![0.0_f32, 0.0];
        let log_var = vec![0.0_f32, 0.0];
        let (total, rl, kl) =
            bcq_vae_loss(&recon, &actions, &mean, &log_var, 1, 1, 0.5).expect("ok");
        assert!(rl.abs() < 1e-6, "recon={rl}");
        assert!(kl.abs() < 1e-6, "kl={kl}");
        assert!(total.abs() < 1e-6, "total={total}");
    }

    #[test]
    fn bcq_vae_kl_positive_off_prior() {
        // Non-zero mean ⇒ positive KL.
        let recon = vec![0.0_f32];
        let actions = vec![0.0_f32];
        let mean = vec![2.0_f32];
        let log_var = vec![0.0_f32];
        let (_, _, kl) = bcq_vae_loss(&recon, &actions, &mean, &log_var, 1, 1, 1.0).expect("ok");
        // KL = 0.5*(4 + 1 − 0 − 1) = 2.
        assert!((kl - 2.0).abs() < 1e-5, "kl={kl}");
    }

    /// End-to-end smoke test: a tiny offline batch produces finite, ordered
    /// losses with seeded random Q-values.
    #[test]
    fn offline_pipeline_finite() {
        let mut rng = LcgRng::new(2024);
        let b = 16;
        let q_sa: Vec<f32> = (0..b).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let v_s: Vec<f32> = (0..b).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let log_pi: Vec<f32> = (0..b).map(|_| -(rng.next_f32() + 0.1)).collect();
        let rewards: Vec<f32> = (0..b).map(|_| rng.next_f32()).collect();
        let dones: Vec<f32> = (0..b).map(|i| if i % 4 == 3 { 1.0 } else { 0.0 }).collect();

        let lv = iql_value_loss(&q_sa, &v_s, IqlConfig::default()).expect("ok");
        let (lq, _) =
            iql_critic_loss(&q_sa, &rewards, &v_s, &dones, IqlConfig::default()).expect("ok");
        let (la, mw) = awac_actor_loss(&log_pi, &q_sa, &v_s, AwacConfig::default()).expect("ok");
        assert!(lv.is_finite() && lv >= 0.0);
        assert!(lq.is_finite() && lq >= 0.0);
        assert!(la.is_finite());
        assert!(mw > 0.0);
    }
}
