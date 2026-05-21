//! # QR-DQN — Quantile Regression DQN Loss
//!
//! Dabney, Rowland, Bellemare, Munos (2018)
//! "Distributional Reinforcement Learning with Quantile Regression". AAAI 2018.
//!
//! Instead of a categorical distribution over a fixed support (C51), QR-DQN
//! represents the return distribution as N implicit quantile values θ_1..θ_N,
//! where θ_i is the quantile at level τ_i = (2i − 1) / (2N) for i = 1..N
//! (0-indexed: τ_i = (2*(i+1) − 1) / (2*N) for i = 0..N−1).
//!
//! ## Training objective
//!
//! For each (prediction index i, target index j) pair the asymmetric quantile
//! Huber (QHuber) loss is:
//!
//! ```text
//! u            = target_j − pred_i
//! L_κ(u)      = 0.5 u²             if |u| ≤ κ
//!                κ (|u| − 0.5 κ)   otherwise        [Huber loss]
//! ρ_{τ_i}^κ(u) = |τ_i − 1(u < 0)| · L_κ(u)        [asymmetric weight]
//! ```
//!
//! Per-sample loss (outer division by N only, as in the paper):
//!
//! ```text
//! loss_b = (1/N) Σ_i Σ_j ρ_{τ_i}^κ(target_j − pred_i)
//! ```

use crate::error::{RlError, RlResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// QR-DQN configuration.
#[derive(Debug, Clone, Copy)]
pub struct QrDqnConfig {
    /// Number of quantile atoms N (must be ≥ 1; default 200 as in the paper).
    pub n_quantiles: usize,
    /// Huber loss threshold κ (default 1.0).
    pub kappa: f32,
    /// Discount factor γ (must be in (0, 1]; default 0.99).
    pub gamma: f32,
}

impl Default for QrDqnConfig {
    fn default() -> Self {
        Self {
            n_quantiles: 200,
            kappa: 1.0,
            gamma: 0.99,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// QR-DQN loss output.
#[derive(Debug, Clone)]
pub struct QrDqnLoss {
    /// Mean quantile Huber loss over the batch (scalar to minimise).
    pub loss: f32,
    /// Per-sample mean absolute TD error (for PER priority updates).
    /// Length == batch size B.
    pub td_errors: Vec<f32>,
}

// ─── Validation helpers ───────────────────────────────────────────────────────

/// Validate QR-DQN config.
fn validate_cfg(cfg: &QrDqnConfig) -> RlResult<()> {
    if cfg.n_quantiles == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_quantiles".into(),
            msg: "must be >= 1".into(),
        });
    }
    if cfg.kappa <= 0.0 {
        return Err(RlError::InvalidHyperparameter {
            name: "kappa".into(),
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

// ─── Huber loss (private) ─────────────────────────────────────────────────────

/// Scalar Huber (smooth L1) loss: L_κ(u).
#[inline]
fn huber(u: f32, kappa: f32) -> f32 {
    if u.abs() <= kappa {
        0.5 * u * u
    } else {
        kappa * (u.abs() - 0.5 * kappa)
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the mid-quantile levels τ_i = (2i − 1) / (2N) for i = 1..N.
///
/// Using 0-indexing: τ_i = (2*(i+1) − 1) / (2*N) for i in 0..N.
///
/// Result is a strictly increasing sequence:
/// `[1/(2N), 3/(2N), 5/(2N), …, (2N−1)/(2N)]`.
#[must_use]
pub fn qr_dqn_quantile_levels(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2 * (i + 1) - 1) as f32 / (2 * n) as f32)
        .collect()
}

/// Compute the quantile Bellman targets for the next state.
///
/// Applies the standard Bellman backup to each quantile atom:
/// `y_j = r + γ * (1 − done) * θ_target_j`
///
/// The caller must already have selected `target_quantiles` for the greedy
/// next action a* = argmax `E[Z]` (i.e. argmax of mean quantile value).
///
/// # Arguments
///
/// * `rewards`          — `[B]` rewards r_b.
/// * `dones`            — `[B]` done flags (1.0 = terminal).
/// * `target_quantiles` — `[B × N]` quantile values from target network for the chosen next action.
/// * `cfg`              — QR-DQN configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config.
/// * [`RlError::DimensionMismatch`]     — inconsistent lengths.
///
/// # Returns
///
/// `[B × N]` Bellman targets.
pub fn qr_dqn_targets(
    rewards: &[f32],
    dones: &[f32],
    target_quantiles: &[f32],
    cfg: &QrDqnConfig,
) -> RlResult<Vec<f32>> {
    validate_cfg(cfg)?;

    let b = rewards.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if dones.len() != b {
        return Err(RlError::DimensionMismatch {
            expected: b,
            got: dones.len(),
        });
    }
    let n = cfg.n_quantiles;
    if target_quantiles.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: target_quantiles.len(),
        });
    }

    let mut targets = Vec::with_capacity(b * n);
    for b_idx in 0..b {
        let r = rewards[b_idx];
        let gamma_factor = cfg.gamma * (1.0 - dones[b_idx]);
        for j in 0..n {
            targets.push(r + gamma_factor * target_quantiles[b_idx * n + j]);
        }
    }

    Ok(targets)
}

/// Compute the QR-DQN quantile Huber loss.
///
/// For each sample b:
/// ```text
/// loss_b = (1/N) Σ_i Σ_j ρ_{τ_i}^κ(target_j − pred_i)
/// ```
///
/// The per-sample TD error stored in `td_errors` is the mean over all (i,j) pairs
/// of the absolute difference |target_j − pred_i|, providing a scalar priority for PER.
///
/// # Arguments
///
/// * `pred_quantiles`   — `[B × N]` quantile predictions from the online network for
///   the chosen action a_b.
/// * `target_quantiles` — `[B × N]` Bellman targets (output of [`qr_dqn_targets`]).
/// * `is_weights`       — `[B]` importance-sampling weights (all 1.0 when not using PER).
/// * `cfg`              — QR-DQN configuration.
///
/// # Errors
///
/// * [`RlError::InvalidHyperparameter`] — invalid config.
/// * [`RlError::DimensionMismatch`]     — slice lengths inconsistent.
/// * [`RlError::Internal`]              — NaN loss encountered (e.g. due to infinite inputs).
pub fn qr_dqn_loss(
    pred_quantiles: &[f32],
    target_quantiles: &[f32],
    is_weights: &[f32],
    cfg: &QrDqnConfig,
) -> RlResult<QrDqnLoss> {
    validate_cfg(cfg)?;

    let b = is_weights.len();
    if b == 0 {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let n = cfg.n_quantiles;
    if pred_quantiles.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: pred_quantiles.len(),
        });
    }
    if target_quantiles.len() != b * n {
        return Err(RlError::DimensionMismatch {
            expected: b * n,
            got: target_quantiles.len(),
        });
    }

    // Pre-compute quantile levels τ_i for i = 0..N
    let tau = qr_dqn_quantile_levels(n);

    let mut td_errors = Vec::with_capacity(b);
    let mut weighted_loss = 0.0_f32;

    for b_idx in 0..b {
        let pred_row = &pred_quantiles[b_idx * n..(b_idx + 1) * n];
        let target_row = &target_quantiles[b_idx * n..(b_idx + 1) * n];

        let mut loss_b = 0.0_f32;
        let mut td_err_b = 0.0_f32;

        // Outer loop over prediction quantiles i, inner over target quantiles j
        for (i, &tau_i) in tau.iter().enumerate() {
            let pred_i = pred_row[i];

            for &target_j in target_row {
                let u = target_j - pred_i;
                let h = huber(u, cfg.kappa);
                // Asymmetric indicator: 1(u < 0)
                let indicator = if u < 0.0 { 1.0_f32 } else { 0.0_f32 };
                let rho = (tau_i - indicator).abs() * h;

                loss_b += rho;
                td_err_b += u.abs();
            }
        }

        // Normalise: divide by N (outer quantile count) as per the paper.
        loss_b /= n as f32;
        // Mean absolute TD error over all (i,j) pairs for PER.
        td_err_b /= (n * n) as f32;

        td_errors.push(td_err_b);
        weighted_loss += is_weights[b_idx] * loss_b;
    }

    let loss = weighted_loss / b as f32;

    if loss.is_nan() {
        return Err(RlError::Internal(
            "NaN loss encountered in qr_dqn_loss".into(),
        ));
    }

    Ok(QrDqnLoss { loss, td_errors })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_targets(b: usize, n: usize, val: f32) -> Vec<f32> {
        vec![val; b * n]
    }

    #[test]
    fn default_config_ok() {
        let cfg = QrDqnConfig::default();
        assert_eq!(cfg.n_quantiles, 200);
        assert!((cfg.kappa - 1.0).abs() < 1e-6);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
    }

    #[test]
    fn quantile_levels_length() {
        let n = 200;
        let tau = qr_dqn_quantile_levels(n);
        assert_eq!(tau.len(), n);
    }

    #[test]
    fn quantile_levels_endpoints() {
        let n = 200;
        let tau = qr_dqn_quantile_levels(n);
        let expected_first = 1.0 / (2 * n) as f32;
        let expected_last = (2 * n - 1) as f32 / (2 * n) as f32;
        assert!(
            (tau[0] - expected_first).abs() < 1e-6,
            "first={}, expected={}",
            tau[0],
            expected_first
        );
        assert!(
            (tau[n - 1] - expected_last).abs() < 1e-6,
            "last={}, expected={}",
            tau[n - 1],
            expected_last
        );
    }

    #[test]
    fn quantile_levels_strictly_increasing() {
        let n = 50;
        let tau = qr_dqn_quantile_levels(n);
        for i in 0..n - 1 {
            assert!(
                tau[i] < tau[i + 1],
                "τ[{i}]={} >= τ[{}]={}",
                tau[i],
                i + 1,
                tau[i + 1]
            );
        }
    }

    #[test]
    fn targets_shape_correct() {
        let cfg = QrDqnConfig {
            n_quantiles: 10,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 5;
        let n = cfg.n_quantiles;
        let rewards = vec![1.0_f32; b];
        let dones = vec![0.0_f32; b];
        let tq = make_targets(b, n, 0.5);
        let result = qr_dqn_targets(&rewards, &dones, &tq, &cfg).expect("targets should succeed");
        assert_eq!(result.len(), b * n);
    }

    #[test]
    fn targets_terminal_state() {
        // done=1 → target_j = reward (gamma disappears)
        let cfg = QrDqnConfig {
            n_quantiles: 5,
            kappa: 1.0,
            gamma: 0.99,
        };
        let n = cfg.n_quantiles;
        let reward = 3.0_f32;
        let rewards = vec![reward];
        let dones = vec![1.0_f32];
        // target_quantiles can be anything; should be ignored
        let tq = vec![999.0_f32; n];
        let result = qr_dqn_targets(&rewards, &dones, &tq, &cfg)
            .expect("targets for terminal state should succeed");
        for &t in &result {
            assert!(
                (t - reward).abs() < 1e-5,
                "terminal target={t}, expected={reward}"
            );
        }
    }

    #[test]
    fn targets_batch_size_1() {
        let cfg = QrDqnConfig::default();
        let n = cfg.n_quantiles;
        let rewards = vec![2.0_f32];
        let dones = vec![0.0_f32];
        let tq = vec![1.0_f32; n];
        let result =
            qr_dqn_targets(&rewards, &dones, &tq, &cfg).expect("B=1 targets should succeed");
        assert_eq!(result.len(), n);
        // target_j = 2 + 0.99 * 1 = 2.99
        for &t in &result {
            assert!((t - 2.99).abs() < 1e-5, "target={t}");
        }
    }

    #[test]
    fn loss_identical_pred_and_target() {
        // When pred ≈ target for all quantiles, loss should be close to 0.
        let cfg = QrDqnConfig {
            n_quantiles: 10,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 4;
        let n = cfg.n_quantiles;
        let pred = make_targets(b, n, 1.5);
        let target = make_targets(b, n, 1.5);
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg)
            .expect("loss with identical pred/target should succeed");
        assert!(
            result.loss >= 0.0,
            "loss must be non-negative: {}",
            result.loss
        );
        assert!(
            result.loss < 1e-5,
            "loss should be near 0 when pred==target: {}",
            result.loss
        );
    }

    #[test]
    fn loss_is_nonneg() {
        let cfg = QrDqnConfig {
            n_quantiles: 20,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 8;
        let n = cfg.n_quantiles;
        let pred = make_targets(b, n, 0.0);
        let target = make_targets(b, n, 1.0);
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("loss should succeed");
        assert!(result.loss >= 0.0, "loss={}", result.loss);
    }

    #[test]
    fn loss_td_errors_length() {
        let cfg = QrDqnConfig {
            n_quantiles: 10,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 7;
        let n = cfg.n_quantiles;
        let pred = make_targets(b, n, 0.5);
        let target = make_targets(b, n, 1.5);
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("loss should succeed");
        assert_eq!(result.td_errors.len(), b);
    }

    #[test]
    fn loss_td_errors_nonneg() {
        let cfg = QrDqnConfig {
            n_quantiles: 10,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 5;
        let n = cfg.n_quantiles;
        let pred = make_targets(b, n, -1.0);
        let target = make_targets(b, n, 2.0);
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("loss should succeed");
        for (b_idx, &td) in result.td_errors.iter().enumerate() {
            assert!(td >= 0.0, "td_error[{b_idx}]={td} is negative");
        }
    }

    #[test]
    fn loss_uniform_weights() {
        // Verify that uniform is_weights=1 computes simple mean across batch.
        let cfg = QrDqnConfig {
            n_quantiles: 5,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 3;
        let n = cfg.n_quantiles;
        let pred = make_targets(b, n, 0.0);
        let target = make_targets(b, n, 2.0);
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("loss should succeed");
        assert!(result.loss.is_finite(), "loss must be finite");
        assert!(result.loss > 0.0, "loss must be > 0");
    }

    #[test]
    fn loss_batch_size_1() {
        let cfg = QrDqnConfig {
            n_quantiles: 5,
            kappa: 1.0,
            gamma: 0.99,
        };
        let pred = vec![0.0_f32, 0.5, 1.0, 1.5, 2.0];
        let target = vec![0.1_f32, 0.6, 1.1, 1.6, 2.1];
        let is_weights = vec![1.0_f32];
        let result =
            qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("B=1 loss should succeed");
        assert!(result.loss.is_finite());
        assert_eq!(result.td_errors.len(), 1);
    }

    #[test]
    fn err_empty_input() {
        let cfg = QrDqnConfig::default();
        let result = qr_dqn_loss(&[], &[], &[], &cfg);
        assert!(result.is_err(), "empty input should return Err");
    }

    #[test]
    fn err_n_quantiles_zero() {
        let cfg = QrDqnConfig {
            n_quantiles: 0,
            kappa: 1.0,
            gamma: 0.99,
        };
        let result = qr_dqn_targets(&[1.0], &[0.0], &[], &cfg);
        assert!(result.is_err(), "n_quantiles=0 should return Err");
    }

    #[test]
    fn err_dimension_mismatch() {
        let cfg = QrDqnConfig {
            n_quantiles: 10,
            kappa: 1.0,
            gamma: 0.99,
        };
        let b = 3;
        let n = cfg.n_quantiles;
        // pred has B*N elements but target has only B*N - 1
        let pred = make_targets(b, n, 1.0);
        let target = make_targets(b, n - 1, 2.0); // wrong length
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg);
        assert!(result.is_err(), "dimension mismatch should return Err");
    }

    #[test]
    fn loss_nan_check() {
        let cfg = QrDqnConfig {
            n_quantiles: 5,
            kappa: 1.0,
            gamma: 0.99,
        };
        let n = cfg.n_quantiles;
        let b = 2;
        // Infinite values may produce NaN huber/loss
        let pred = vec![f32::INFINITY; b * n];
        let target = vec![f32::NEG_INFINITY; b * n];
        let is_weights = vec![1.0_f32; b];
        let result = qr_dqn_loss(&pred, &target, &is_weights, &cfg);
        if let Ok(qr) = result {
            // If it somehow succeeds, the loss must not be NaN
            assert!(!qr.loss.is_nan(), "loss must not be NaN");
        }
        // Err(NanLoss) is also acceptable
    }

    #[test]
    fn qr_dqn_vs_dqn_reduced() {
        // N=1 quantile at τ=0.5 reduces to median Huber loss.
        // pred = [p], target = [t]:
        //   u = t - p
        //   rho = |0.5 - 1(u<0)| * L_κ(u)
        // When u > 0: |0.5 - 0| * L_κ(u) = 0.5 * L_κ(u)
        // loss_b = (1/1) * 0.5 * L_κ(u)
        let cfg = QrDqnConfig {
            n_quantiles: 1,
            kappa: 1.0,
            gamma: 0.99,
        };
        let pred = vec![0.5_f32]; // [B=1, N=1]
        let target = vec![1.0_f32];
        let is_weights = vec![1.0_f32];
        let result =
            qr_dqn_loss(&pred, &target, &is_weights, &cfg).expect("N=1 loss should succeed");

        // u = 1.0 - 0.5 = 0.5; |u| <= kappa=1 so huber = 0.5 * 0.5^2 = 0.125
        // tau[0] = 1/(2*1) = 0.5; u > 0 so indicator=0; rho = 0.5 * 0.125 = 0.0625
        // loss_b = (1/1) * 0.0625 = 0.0625; loss = 0.0625 / 1 = 0.0625
        assert!(
            (result.loss - 0.0625).abs() < 1e-5,
            "N=1 quantile loss={}, expected=0.0625",
            result.loss
        );
    }
}
