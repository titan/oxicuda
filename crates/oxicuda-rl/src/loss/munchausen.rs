//! # Munchausen-RL — Munchausen DQN Loss
//!
//! Vieillard, Pietquin, Geist (2020)
//! "Munchausen Reinforcement Learning". NeurIPS 2020.
//!
//! Munchausen-RL augments the Bellman target with a scaled log-policy term —
//! the agent "bootstraps from its own behaviour". This encourages implicit
//! KL-regularisation toward a soft-optimal policy and often improves stability.
//!
//! ## Munchausen bonus
//!
//! ```text
//! log π(a|s) = Q(s,a)/τ − log Σ_{a'} exp(Q(s,a')/τ)   [numerically stable log-softmax]
//!
//! m(s,a) = α · clip(τ · log π(a|s), l₀, 0)            [clip to ≤ 0; l₀ = clip_min]
//! ```
//!
//! ## Soft Bellman target (equation 9 in the paper)
//!
//! ```text
//! π'(a'|s') = softmax(Q'(s',a')/τ)
//!
//! V_soft(s') = Σ_{a'} π'(a') · (Q'(s',a') − τ · log π'(a'))
//!
//! y = r + m(s,a) + γ · (1 − done) · V_soft(s')
//! ```
//!
//! ## Loss
//!
//! Standard MSE between predicted Q(s,a) and target y, optionally weighted
//! by PER importance-sampling weights.

use crate::error::{RlError, RlResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Munchausen-DQN configuration.
#[derive(Debug, Clone, Copy)]
pub struct MunchausenConfig {
    /// Scale α applied to the log-policy Munchausen bonus (default 0.9).
    /// When α = 0 this reduces to soft DQN.
    pub alpha: f32,
    /// Softmax temperature τ (must be > 0; default 0.03).
    pub tau: f32,
    /// Lower bound for clipping the log-policy before scaling (default −1.0).
    /// The bonus is further clipped to ≤ 0 after scaling, ensuring it never
    /// inflates the target.
    pub clip_min: f32,
    /// Discount factor γ (must be in (0, 1]; default 0.99).
    pub gamma: f32,
}

impl Default for MunchausenConfig {
    fn default() -> Self {
        Self {
            alpha: 0.9,
            tau: 0.03,
            clip_min: -1.0,
            gamma: 0.99,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Munchausen-DQN loss output.
#[derive(Debug, Clone)]
pub struct MunchausenLoss {
    /// Mean MSE loss over the batch (scalar to minimise).
    pub loss: f32,
    /// Per-sample absolute TD error |Q_pred − y|, length B.
    /// Used for PER priority updates.
    pub td_errors: Vec<f32>,
}

// ─── Validation helpers ───────────────────────────────────────────────────────

/// Validate Munchausen config.
fn validate_cfg(cfg: &MunchausenConfig) -> RlResult<()> {
    if cfg.alpha < 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "alpha".into(),
            msg: "must be >= 0".into(),
        });
    }
    if cfg.tau <= 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "tau".into(),
            msg: "must be > 0".into(),
        });
    }
    if cfg.gamma <= 0.0 || cfg.gamma > 1.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "gamma".into(),
            msg: "must be in (0, 1]".into(),
        });
    }
    Ok(())
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Numerically stable log-softmax over a slice of Q-values divided by τ.
///
/// Returns a vector of the same length where each element is:
/// ```text
/// log_softmax[a] = q[a]/τ − log Σ_{a'} exp(q[a']/τ)
/// ```
///
/// Subtracts the maximum before exponentiating to prevent overflow.
fn log_softmax_over_actions(q: &[f32], tau: f32) -> Vec<f32> {
    // Scale by 1/τ first.
    let scaled: Vec<f32> = q.iter().map(|&qi| qi / tau).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = scaled.iter().map(|&s| (s - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    scaled.iter().map(|&s| s - log_sum_exp).collect()
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Build Munchausen Bellman targets for a batch.
///
/// For each sample b:
/// 1. Compute log π(a_b|s_b) from current Q-values `q_cur_all`.
///    Since we know Q(s_b, a_b) = `q_sa[b]` we can obtain the log-probability
///    without knowing the action index:
///    `log π(a_b) = q_sa[b]/τ − log Σ_a' exp(q_cur[a']/τ)`.
/// 2. Compute the Munchausen bonus: `m(b) = α · clip(τ · log π(a_b), clip_min, 0)`.
/// 3. Compute next-state soft value target:
///    `V_soft(s') = Σ_a' π'(a') · (Q'(a') − τ · log π'(a'))`.
/// 4. Target: `y[b] = rewards[b] + m(b) + γ · (1 − dones[b]) · V_soft(s')`.
///
/// # Arguments
///
/// * `q_sa`       — `[B]` Q(s_b, a_b) for each chosen action.
/// * `q_cur_all`  — `[B × A]` current Q-values over all actions (for log π(a|s)).
/// * `q_next`     — `[B × A]` target-network Q-values at next states.
/// * `rewards`    — `[B]`.
/// * `dones`      — `[B]` done flags (1.0 = terminal).
/// * `n_actions`  — A, number of discrete actions.
/// * `cfg`        — Munchausen configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config fields.
/// * [`RlError::DimensionMismatch`]     — inconsistent slice lengths.
/// * [`RlError::Internal`]              — NaN encountered in targets.
pub fn munchausen_target(
    q_sa: &[f32],
    q_cur_all: &[f32],
    q_next: &[f32],
    rewards: &[f32],
    dones: &[f32],
    n_actions: usize,
    cfg: MunchausenConfig,
) -> RlResult<Vec<f32>> {
    validate_cfg(&cfg)?;

    let b = rewards.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if n_actions == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if q_sa.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: q_sa.len(),
        });
    }
    if dones.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: dones.len(),
        });
    }
    if q_cur_all.len() != b * n_actions {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: q_cur_all.len(),
        });
    }
    if q_next.len() != b * n_actions {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: q_next.len(),
        });
    }

    let mut targets = Vec::with_capacity(b);

    for b_idx in 0..b {
        let cur_slice = &q_cur_all[b_idx * n_actions..(b_idx + 1) * n_actions];
        let next_slice = &q_next[b_idx * n_actions..(b_idx + 1) * n_actions];

        // ── Step 1: log π(a_b|s_b) via log-softmax trick ─────────────────────
        // log π(a_b) = Q(s,a_b)/τ − log Σ_a' exp(Q(s,a')/τ)
        // We compute the log-partition directly (numerically stable).
        let scaled_cur: Vec<f32> = cur_slice.iter().map(|&qi| qi / cfg.tau).collect();
        let max_cur = scaled_cur.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_cur: f32 = scaled_cur.iter().map(|&s| (s - max_cur).exp()).sum();
        let log_partition_cur = max_cur + sum_exp_cur.ln();

        let log_pi_sa = q_sa[b_idx] / cfg.tau - log_partition_cur;

        // ── Step 2: Munchausen bonus ──────────────────────────────────────────
        // m(s,a) = α · clip(τ · log π(a|s), clip_min, 0)
        let tau_log_pi = cfg.tau * log_pi_sa;
        let munchausen_bonus = cfg.alpha * tau_log_pi.max(cfg.clip_min).min(0.0);

        // ── Step 3: next-state soft value V_soft(s') ─────────────────────────
        // π'(a'|s') = softmax(Q'(a')/τ)
        // V_soft(s') = Σ_a' π'(a') · (Q'(a') − τ · log π'(a'))
        //            = Σ_a' π'(a') · Q'(a') − τ · Σ_a' π'(a') · log π'(a')
        //            = E_{π'}[Q'] + τ · H(π')   [where H is the entropy]
        let log_pi_next = log_softmax_over_actions(next_slice, cfg.tau);
        let mut soft_target_next = 0.0_f32;
        for a in 0..n_actions {
            let pi_next_a = log_pi_next[a].exp();
            // V_soft contribution: π'(a') · (Q'(a') − τ · log π'(a'))
            soft_target_next += pi_next_a * (next_slice[a] - cfg.tau * log_pi_next[a]);
        }

        // ── Step 4: Munchausen target ─────────────────────────────────────────
        let y =
            rewards[b_idx] + munchausen_bonus + cfg.gamma * (1.0 - dones[b_idx]) * soft_target_next;

        targets.push(y);
    }

    // NaN guard on targets
    for &t in &targets {
        if t.is_nan() {
            return Err(RlError::Internal(
                "NaN encountered in munchausen_target".into(),
            ));
        }
    }

    Ok(targets)
}

/// Compute the Munchausen-DQN loss (MSE between predicted Q and Munchausen target).
///
/// ```text
/// loss_b   = (q_pred_sa[b] − target[b])²
/// batch    = (1/B) Σ_b is_weight_b · loss_b
/// td_error = |q_pred_sa[b] − target[b]|
/// ```
///
/// # Arguments
///
/// * `q_pred_sa`  — `[B]` predicted Q(s,a) at the chosen actions.
/// * `target`     — `[B]` Munchausen targets from [`munchausen_target`].
/// * `is_weights` — `[B]` PER importance-sampling weights.
/// * `cfg`        — Munchausen configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config fields.
/// * [`RlError::DimensionMismatch`]     — inconsistent lengths.
/// * [`RlError::Internal`]              — NaN loss encountered.
pub fn munchausen_dqn_loss(
    q_pred_sa: &[f32],
    target: &[f32],
    is_weights: &[f32],
    cfg: MunchausenConfig,
) -> RlResult<MunchausenLoss> {
    validate_cfg(&cfg)?;

    let b = is_weights.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if q_pred_sa.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: q_pred_sa.len(),
        });
    }
    if target.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: target.len(),
        });
    }

    let mut td_errors = Vec::with_capacity(b);
    let mut weighted_loss = 0.0_f32;

    for b_idx in 0..b {
        let diff = q_pred_sa[b_idx] - target[b_idx];
        let loss_b = diff * diff; // MSE
        td_errors.push(diff.abs());
        weighted_loss += is_weights[b_idx] * loss_b;
    }

    let loss = weighted_loss / b as f32;

    if loss.is_nan() {
        return Err(RlError::Internal(
            "NaN loss encountered in munchausen_dqn_loss".into(),
        ));
    }

    Ok(MunchausenLoss { loss, td_errors })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a uniform Q-value slice [B × A]
    fn uniform_q(b: usize, n_actions: usize, val: f32) -> Vec<f32> {
        vec![val; b * n_actions]
    }

    // ── log_softmax_over_actions ──────────────────────────────────────────────

    #[test]
    fn log_softmax_sums_to_zero_in_logspace() {
        // Σ exp(log_softmax) must be ≈ 1.0
        let q = vec![1.0_f32, 2.0, 3.0, 0.5];
        let tau = 1.0;
        let lp = log_softmax_over_actions(&q, tau);
        let sum_prob: f32 = lp.iter().map(|&l| l.exp()).sum();
        assert!(
            (sum_prob - 1.0).abs() < 1e-5,
            "softmax must sum to 1: {sum_prob}"
        );
    }

    #[test]
    fn log_softmax_uniform_q_is_uniform() {
        // All Q equal → all log π equal → softmax is uniform.
        let n = 5_usize;
        let q = vec![2.0_f32; n];
        let lp = log_softmax_over_actions(&q, 1.0);
        let expected_log = -(n as f32).ln();
        for (a, &l) in lp.iter().enumerate() {
            assert!(
                (l - expected_log).abs() < 1e-5,
                "action {a}: log_pi={l}, expected={expected_log}"
            );
        }
    }

    #[test]
    fn log_softmax_max_action_dominant() {
        // One action with much larger Q → near-1 probability for that action.
        let q = vec![-100.0_f32, 100.0, -100.0];
        let lp = log_softmax_over_actions(&q, 1.0);
        let pi: Vec<f32> = lp.iter().map(|&l| l.exp()).collect();
        assert!(
            pi[1] > 0.99,
            "dominant action prob should be near 1: {}",
            pi[1]
        );
    }

    // ── munchausen_target ─────────────────────────────────────────────────────

    #[test]
    fn alpha_zero_reduces_to_soft_dqn() {
        // α=0 → Munchausen bonus=0; target = r + γ*(1−done)*V_soft(s')
        let cfg = MunchausenConfig {
            alpha: 0.0,
            tau: 0.1,
            clip_min: -1.0,
            gamma: 0.9,
        };
        let n_actions = 4_usize;
        let b = 2_usize;
        let q_sa = vec![0.5_f32; b];
        let q_cur = uniform_q(b, n_actions, 0.5);
        let q_next = uniform_q(b, n_actions, 1.0);
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("alpha=0 target should succeed");

        // V_soft for uniform Q=1.0: E[Q']=1 and H=−Σ(1/n)log(1/n)=log(n)
        // so V_soft = 1 + τ*log(n_actions) = 1 + 0.1*ln(4) ≈ 1.1386
        // target = 0 + 0 + 0.9 * 1.1386 ≈ 1.0247
        for (b_idx, &t) in targets.iter().enumerate() {
            let expected_v_soft = 1.0 + cfg.tau * (n_actions as f32).ln();
            let expected = cfg.gamma * expected_v_soft;
            assert!(
                (t - expected).abs() < 1e-4,
                "sample {b_idx}: target={t}, expected≈{expected}"
            );
        }
    }

    #[test]
    fn tau_zero_limit_is_hard_max() {
        // Very small τ → log π of the greedy action ≈ 0, bonus ≈ 0,
        // and soft value ≈ max Q'.
        let cfg = MunchausenConfig {
            alpha: 0.9,
            tau: 1e-6,
            clip_min: -1.0,
            gamma: 0.99,
        };
        let n_actions = 3_usize;
        let q_sa = vec![2.0_f32];
        // Current Q: action 0 is maximal
        let q_cur = vec![2.0_f32, 0.0, 0.0];
        let q_next = vec![1.0_f32, 0.5, 0.0]; // max Q_next = 1.0
        let rewards = vec![0.0_f32];
        let dones = vec![0.0_f32];
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("small-tau target should succeed");
        // At τ→0, V_soft → max Q' = 1.0, bonus → 0
        // target ≈ 0 + 0 + 0.99 * 1.0 = 0.99
        assert!(
            (targets[0] - 0.99).abs() < 1e-2,
            "small-tau target={}, expected≈0.99",
            targets[0]
        );
    }

    #[test]
    fn clip_min_bounds_bonus() {
        // The unclipped log-policy contribution τ·log π must be clamped to [clip_min, 0].
        // With a very negative Q for the chosen action, log π → −∞ → clip kicks in.
        let clip_min = -0.5_f32;
        let cfg = MunchausenConfig {
            alpha: 1.0,
            tau: 1.0,
            clip_min,
            gamma: 0.99,
        };
        let n_actions = 3_usize;
        // Chosen action Q is very negative → dominates softmax denom → log π very negative
        let q_sa = vec![-100.0_f32];
        let q_cur = vec![-100.0_f32, 5.0, 5.0];
        let q_next = vec![0.0_f32; n_actions];
        let rewards = vec![0.0_f32];
        let dones = vec![1.0_f32]; // done=1 → next ignored
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("clip_min target should succeed");
        // bonus = α * clip(τ·logπ, clip_min, 0) ≥ α * clip_min = 1.0 * -0.5 = -0.5
        // target = 0 + bonus + 0 (done) ≥ -0.5
        assert!(
            targets[0] >= cfg.alpha * clip_min - 1e-4,
            "target={} should be >= alpha*clip_min={}",
            targets[0],
            cfg.alpha * clip_min
        );
    }

    #[test]
    fn bonus_is_nonpositive() {
        // log π ≤ 0 everywhere (log of a probability) → clip to 0 → bonus ≤ 0.
        let cfg = MunchausenConfig::default();
        let n_actions = 4_usize;
        let b = 5_usize;
        // Generate various Q values where the chosen action is middling.
        let q_sa: Vec<f32> = (0..b).map(|i| i as f32 * 0.3).collect();
        let q_cur: Vec<f32> = (0..b * n_actions).map(|k| (k as f32) * 0.1).collect();
        let q_next = uniform_q(b, n_actions, 0.5);
        let rewards = vec![0.0_f32; b];
        let dones = vec![1.0_f32; b]; // done=1 → next contribution=0
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("bonus_is_nonpositive target should succeed");
        // target = 0 + bonus + 0; bonus ≤ 0 → target ≤ 0
        for (b_idx, &t) in targets.iter().enumerate() {
            assert!(
                t <= 1e-5,
                "bonus should be nonpositive: sample {b_idx} target={t}"
            );
        }
    }

    #[test]
    fn terminal_state_ignores_next_q() {
        // done=1 → next Q dropped; target = r + bonus (only).
        let cfg = MunchausenConfig {
            alpha: 0.0, // bonus=0, so target = r exactly
            tau: 0.1,
            clip_min: -1.0,
            gamma: 0.99,
        };
        let n_actions = 2_usize;
        let q_sa = vec![1.0_f32];
        let q_cur = vec![1.0_f32, 0.0];
        let q_next = vec![999.0_f32; n_actions]; // should be ignored
        let reward = 5.0_f32;
        let rewards = vec![reward];
        let dones = vec![1.0_f32];
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("terminal target should succeed");
        assert!(
            (targets[0] - reward).abs() < 1e-4,
            "terminal target={}, expected reward={reward}",
            targets[0]
        );
    }

    #[test]
    fn target_shape_equals_batch() {
        let cfg = MunchausenConfig::default();
        let n_actions = 3_usize;
        let b = 7_usize;
        let q_sa = vec![0.5_f32; b];
        let q_cur = uniform_q(b, n_actions, 0.5);
        let q_next = uniform_q(b, n_actions, 0.5);
        let rewards = vec![1.0_f32; b];
        let dones = vec![0.0_f32; b];
        let targets = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg)
            .expect("target shape test should succeed");
        assert_eq!(targets.len(), b);
    }

    // ── munchausen_dqn_loss ───────────────────────────────────────────────────

    #[test]
    fn munchausen_loss_zero_pred_eq_target() {
        let cfg = MunchausenConfig::default();
        let b = 4_usize;
        let val = 2.5_f32;
        let pred = vec![val; b];
        let target = vec![val; b];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&pred, &target, &is_weights, cfg)
            .expect("zero-error loss should succeed");
        assert!(
            result.loss.abs() < 1e-10,
            "loss should be 0 when pred==target: {}",
            result.loss
        );
    }

    #[test]
    fn munchausen_loss_positive_mismatch() {
        let cfg = MunchausenConfig::default();
        let b = 3_usize;
        let pred = vec![0.0_f32; b];
        let target = vec![1.0_f32; b];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&pred, &target, &is_weights, cfg)
            .expect("positive mismatch loss should succeed");
        assert!(result.loss > 0.0, "loss must be > 0 when pred != target");
    }

    #[test]
    fn munchausen_loss_td_errors_len() {
        let cfg = MunchausenConfig::default();
        let b = 6_usize;
        let pred = vec![1.0_f32; b];
        let target = vec![2.0_f32; b];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&pred, &target, &is_weights, cfg)
            .expect("td_errors len test should succeed");
        assert_eq!(result.td_errors.len(), b);
    }

    // ── DimensionMismatch guards ──────────────────────────────────────────────

    #[test]
    fn dim_mismatch_q_cur_all() {
        let cfg = MunchausenConfig::default();
        let n_actions = 4_usize;
        let b = 2_usize;
        let q_sa = vec![0.5_f32; b];
        // q_cur should be b*n_actions = 8, provide 7 instead
        let q_cur = vec![0.5_f32; b * n_actions - 1];
        let q_next = uniform_q(b, n_actions, 0.5);
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let result = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg);
        assert!(
            result.is_err(),
            "wrong q_cur_all len should return DimensionMismatch"
        );
    }

    #[test]
    fn dim_mismatch_q_next() {
        let cfg = MunchausenConfig::default();
        let n_actions = 4_usize;
        let b = 2_usize;
        let q_sa = vec![0.5_f32; b];
        let q_cur = uniform_q(b, n_actions, 0.5);
        // q_next should be b*n_actions = 8, provide 5 instead
        let q_next = vec![0.5_f32; 5];
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let result = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg);
        assert!(
            result.is_err(),
            "wrong q_next len should return DimensionMismatch"
        );
    }

    #[test]
    fn dim_mismatch_q_pred() {
        let cfg = MunchausenConfig::default();
        let b = 3_usize;
        // q_pred should be b=3, provide 2 instead
        let q_pred = vec![0.5_f32; 2];
        let target = vec![1.0_f32; b];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&q_pred, &target, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong q_pred len should return DimensionMismatch"
        );
    }

    #[test]
    fn dim_mismatch_target() {
        let cfg = MunchausenConfig::default();
        let b = 3_usize;
        let q_pred = vec![0.5_f32; b];
        // target should be b=3, provide 5 instead → mismatch since is_weights says b=3
        let target = vec![1.0_f32; 5];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&q_pred, &target, &is_weights, cfg);
        assert!(
            result.is_err(),
            "wrong target len should return DimensionMismatch"
        );
    }

    // ── validate_cfg guards ───────────────────────────────────────────────────

    #[test]
    fn validate_cfg_bad_alpha() {
        let cfg = MunchausenConfig {
            alpha: -0.1,
            ..MunchausenConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "alpha<0 should return InvalidHyperparameter"
        );
    }

    #[test]
    fn validate_cfg_bad_tau() {
        let cfg = MunchausenConfig {
            tau: 0.0,
            ..MunchausenConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "tau=0 should return InvalidHyperparameter"
        );
    }

    #[test]
    fn validate_cfg_bad_gamma() {
        let cfg = MunchausenConfig {
            gamma: 1.5,
            ..MunchausenConfig::default()
        };
        assert!(
            validate_cfg(&cfg).is_err(),
            "gamma>1 should return InvalidHyperparameter"
        );
    }

    // ── NaN guards ────────────────────────────────────────────────────────────

    #[test]
    fn nan_guard_target() {
        let cfg = MunchausenConfig::default();
        let n_actions = 3_usize;
        let b = 2_usize;
        let q_sa = vec![0.5_f32; b];
        let q_cur = uniform_q(b, n_actions, 0.5);
        // Inject NaN into q_next → soft value will be NaN
        let mut q_next = uniform_q(b, n_actions, 1.0);
        q_next[0] = f32::NAN;
        let rewards = vec![0.0_f32; b];
        let dones = vec![0.0_f32; b];
        let result = munchausen_target(&q_sa, &q_cur, &q_next, &rewards, &dones, n_actions, cfg);
        assert!(
            result.is_err(),
            "NaN in q_next should return Internal error"
        );
    }

    #[test]
    fn nan_guard_loss() {
        let cfg = MunchausenConfig::default();
        let b = 2_usize;
        // Inject NaN into pred → diff will be NaN → loss NaN
        let q_pred = vec![f32::NAN, 1.0];
        let target = vec![0.5_f32; b];
        let is_weights = vec![1.0_f32; b];
        let result = munchausen_dqn_loss(&q_pred, &target, &is_weights, cfg);
        // Should either err with Internal or td_errors contain non-finite values
        match result {
            Err(_) => {} // expected path
            Ok(m) => {
                // If it somehow didn't error, loss or a td_error should be non-finite
                let has_non_finite = m.loss.is_nan() || m.td_errors.iter().any(|&e| !e.is_finite());
                assert!(
                    has_non_finite,
                    "NaN pred should produce non-finite output; loss={}",
                    m.loss
                );
            }
        }
    }

    // ── IS weight upweighting ─────────────────────────────────────────────────

    #[test]
    fn is_weight_upweighting() {
        // Doubling the IS weight of every sample should double the batch loss
        // (since loss = Σ w_b * loss_b / B and all loss_b are equal).
        let cfg = MunchausenConfig::default();
        let b = 4_usize;
        let pred = vec![0.0_f32; b];
        let target = vec![1.0_f32; b];
        let is_uniform = vec![1.0_f32; b];
        let is_doubled = vec![2.0_f32; b];

        let loss_uniform = munchausen_dqn_loss(&pred, &target, &is_uniform, cfg)
            .expect("uniform IS loss should succeed")
            .loss;
        let loss_doubled = munchausen_dqn_loss(&pred, &target, &is_doubled, cfg)
            .expect("doubled IS loss should succeed")
            .loss;

        assert!(
            (loss_doubled - 2.0 * loss_uniform).abs() < 1e-5,
            "doubled IS weight should double loss: uniform={loss_uniform}, doubled={loss_doubled}"
        );
    }
}
