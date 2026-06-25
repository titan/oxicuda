//! # Retrace(λ) Safe Off-Policy Returns
//!
//! Munos et al. (2016), "Safe and Efficient Off-Policy Reinforcement Learning",
//! NeurIPS 2016.
//!
//! ## Algorithm
//!
//! Retrace uses a product of truncated IS ratios to form a multi-step return
//! that is safe (finite variance) for any behaviour policy:
//!
//! ```text
//! c_t = λ * min(1, π(a_t|s_t) / μ(a_t|s_t))
//!
//! Q^ret(s_t, a_t) = r_t + γ E_{a~π}[Q(s_{t+1}, a)]
//!   + γ Σ_{s=t+1}^{T-1} γ^{s-t} (Π_{i=t+1}^{s} c_i)
//!       [r_s + γ E_{a~π}[Q(s_{s+1}, a)] - Q(s_s, a_s)]
//! ```
//!
//! In the common approximate form (using V instead of E\[Q\]):
//! ```text
//! δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - Q(s_t, a_t)
//! Q^ret_t = Q(s_t, a_t) + δ_t + Σ_{s=t+1}^{T-1} (Π_{i=t+1}^s c_i) δ_s
//! ```

use crate::error::{RlError, RlResult};

/// Retrace(λ) configuration.
#[derive(Debug, Clone, Copy)]
pub struct RetraceConfig {
    /// Discount factor γ.
    pub gamma: f32,
    /// Retrace λ ∈ [0, 1].
    pub lambda: f32,
}

impl Default for RetraceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 1.0,
        }
    }
}

/// Retrace(λ) return output.
#[derive(Debug, Clone)]
pub struct RetraceOutput {
    /// Retrace Q-value targets `Q^ret_t`.
    pub q_targets: Vec<f32>,
    /// Temporal difference errors `δ_t`.
    pub td_errors: Vec<f32>,
}

/// Compute Retrace(λ) Q-value targets.
///
/// # Arguments
///
/// * `rewards`       — `[T]` rewards.
/// * `q_values`      — `[T]` Q(s_t, a_t) estimates.
/// * `values`        — `[T+1]` V(s_t) = E_{a~π}[Q(s_t, a)] estimates.
/// * `dones`         — `[T]` done flags.
/// * `log_probs_new` — `[T]` log-probs under current policy π.
/// * `log_probs_old` — `[T]` log-probs under behaviour policy μ.
/// * `cfg`           — Retrace configuration.
///
/// # Errors
///
/// * [`RlError::DimensionMismatch`] for inconsistent slice lengths.
pub fn compute_retrace(
    rewards: &[f32],
    q_values: &[f32],
    values: &[f32],
    dones: &[f32],
    log_probs_new: &[f32],
    log_probs_old: &[f32],
    cfg: RetraceConfig,
) -> RlResult<RetraceOutput> {
    let t = rewards.len();
    if q_values.len() != t
        || values.len() != t + 1
        || dones.len() != t
        || log_probs_new.len() != t
        || log_probs_old.len() != t
    {
        return Err(RlError::DimensionMismatch {
            expected: t,
            got: q_values.len(),
        });
    }

    // c_t = λ * min(1, ρ_t)
    let c: Vec<f32> = log_probs_new
        .iter()
        .zip(log_probs_old.iter())
        .map(|(&lp_new, &lp_old)| {
            let rho = (lp_new - lp_old).exp().clamp(0.0, 1e6);
            cfg.lambda * rho.min(1.0)
        })
        .collect();

    // δ_t = r_t + γ * V(s_{t+1}) * mask - Q(s_t, a_t)
    let td_errors: Vec<f32> = (0..t)
        .map(|i| {
            let mask = 1.0 - dones[i];
            rewards[i] + cfg.gamma * values[i + 1] * mask - q_values[i]
        })
        .collect();

    // Q^ret via backward accumulation
    // Q^ret_t = Q_t + δ_t + γ * c_{t+1} * (Q^ret_{t+1} - Q_{t+1})
    let mut q_targets = vec![0.0_f32; t];
    let mut ret_next = values[t]; // bootstrap: V(s_T)

    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        // c_{t+1} is at index i+1 (0 if i = T-1)
        let c_next = if i + 1 < t { c[i + 1] } else { 0.0 };
        let q_next = if i + 1 < t {
            q_values[i + 1]
        } else {
            values[t]
        };
        q_targets[i] = q_values[i] + td_errors[i] + cfg.gamma * mask * c_next * (ret_next - q_next);
        ret_next = q_targets[i];
    }

    Ok(RetraceOutput {
        q_targets,
        td_errors,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_retrace(t: usize, same_policy: bool) -> RetraceOutput {
        let r = vec![1.0_f32; t];
        let q = vec![0.5_f32; t];
        let v = vec![0.5_f32; t + 1];
        let d = vec![0.0_f32; t];
        let lp_new = vec![0.0_f32; t];
        let lp_old = if same_policy {
            vec![0.0_f32; t]
        } else {
            vec![-0.5_f32; t]
        };
        compute_retrace(&r, &q, &v, &d, &lp_new, &lp_old, RetraceConfig::default())
            .expect("valid equal-length slices should not fail")
    }

    #[test]
    fn retrace_output_length() {
        let out = simple_retrace(5, true);
        assert_eq!(out.q_targets.len(), 5);
        assert_eq!(out.td_errors.len(), 5);
    }

    #[test]
    fn retrace_td_errors_finite() {
        let out = simple_retrace(4, false);
        for (i, &d) in out.td_errors.iter().enumerate() {
            assert!(d.is_finite(), "td_error[{i}]={d}");
        }
    }

    #[test]
    fn retrace_on_policy_q_targets_finite() {
        let out = simple_retrace(4, true);
        for (i, &q) in out.q_targets.iter().enumerate() {
            assert!(q.is_finite(), "q_target[{i}]={q}");
        }
    }

    #[test]
    fn retrace_dimension_mismatch() {
        let r = vec![1.0_f32; 3];
        let q = vec![0.5_f32; 3];
        let v = vec![0.5_f32; 3]; // should be 4
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        assert!(compute_retrace(&r, &q, &v, &d, &lp, &lp, RetraceConfig::default()).is_err());
    }

    #[test]
    fn retrace_done_stops_accumulation() {
        let cfg = RetraceConfig::default();
        let r = vec![1.0, 1.0, 1.0];
        let q = vec![0.0_f32; 3];
        let v = vec![0.0_f32; 4];
        let d = vec![0.0, 1.0, 0.0];
        let lp = vec![0.0_f32; 3];
        let out = compute_retrace(&r, &q, &v, &d, &lp, &lp, cfg)
            .expect("valid equal-length slices should not fail");
        assert!(out.q_targets.iter().all(|&q| q.is_finite()));
    }

    #[test]
    fn retrace_multi_episode_boundary_severs_trace() {
        // A batch containing two episodes glued together: the done flag at the
        // boundary must stop the backward Retrace accumulation from leaking
        // future-episode returns into the pre-boundary step. We verify the step
        // at the terminal index equals its one-step Bellman target exactly
        // (mask=0 zeroes the bootstrap), and a step strictly inside the second
        // episode is unaffected by the first.
        let cfg = RetraceConfig {
            gamma: 0.9,
            lambda: 1.0,
        };
        // Episode A: steps 0,1 (step 1 terminal). Episode B: steps 2,3.
        let r = vec![1.0_f32, 2.0, 3.0, 4.0];
        let q = vec![0.0_f32; 4];
        let v = vec![0.0_f32; 5];
        let d = vec![0.0_f32, 1.0, 0.0, 0.0]; // boundary after step 1
        let lp = vec![0.0_f32; 4]; // on-policy ⇒ c = λ·min(1,1) = 1
        let out = compute_retrace(&r, &q, &v, &d, &lp, &lp, cfg)
            .expect("valid equal-length slices should not fail");

        // Step 1 is terminal: mask=0 ⇒ Q^ret_1 = Q_1 + δ_1 = r_1 = 2.0 (no
        // bootstrap, no future trace).
        assert!(
            (out.q_targets[1] - 2.0).abs() < 1e-5,
            "terminal step target should equal its reward, got {}",
            out.q_targets[1]
        );

        // Step 0 must NOT see episode B. With Q=V=0 and on-policy c=1:
        //   δ_0 = r_0 + γ·V(s_1)·mask − Q_0 = 1.0
        //   Q^ret_0 = Q_0 + δ_0 + γ·mask_0·c_1·(Q^ret_1 − Q_1)
        //           = 1.0 + 0.9·1·1·(2.0 − 0.0) = 1.0 + 1.8 = 2.8
        // The +1.8 comes only from episode A's terminal step, never from B.
        assert!(
            (out.q_targets[0] - 2.8).abs() < 1e-5,
            "pre-boundary target must include only its own episode: {}",
            out.q_targets[0]
        );

        // Compare step-0 against an A-only batch (episodes truly independent):
        // the same two A-steps in isolation must yield the identical Q^ret_0.
        let out_a_only =
            compute_retrace(&r[..2], &q[..2], &v[..3], &d[..2], &lp[..2], &lp[..2], cfg)
                .expect("valid A-only slices should not fail");
        assert!(
            (out.q_targets[0] - out_a_only.q_targets[0]).abs() < 1e-5,
            "boundary must isolate episodes: {} vs {}",
            out.q_targets[0],
            out_a_only.q_targets[0]
        );

        // Step 2 (start of episode B, off-policy ratios) must equal a B-only
        // computation as well.
        let cfg_op = RetraceConfig {
            gamma: 0.9,
            lambda: 0.8,
        };
        let lp_new_b = vec![0.0_f32, 0.0];
        let lp_old_b = vec![0.3_f32, -0.4]; // arbitrary off-policy ratios
        let out_full = compute_retrace(
            &r,
            &q,
            &v,
            &d,
            &[0.0, 0.0, lp_new_b[0], lp_new_b[1]],
            &[0.0, 0.0, lp_old_b[0], lp_old_b[1]],
            cfg_op,
        )
        .expect("valid full slices should not fail");
        let out_b_only = compute_retrace(
            &r[2..],
            &q[2..],
            &v[2..],
            &d[2..],
            &lp_new_b,
            &lp_old_b,
            cfg_op,
        )
        .expect("valid B-only slices should not fail");
        // B-only batch indexes the second episode from 0, so its step 0 maps to
        // the full batch's step 2.
        assert!(
            (out_full.q_targets[2] - out_b_only.q_targets[0]).abs() < 1e-5,
            "second-episode start must match B-only trace: {} vs {}",
            out_full.q_targets[2],
            out_b_only.q_targets[0]
        );
    }

    #[test]
    fn retrace_lambda_zero_is_bellman() {
        // λ=0 → c=0 → Q^ret_t = Q_t + δ_t = r_t + γ*V(s_{t+1})*(1-done)
        let cfg = RetraceConfig {
            gamma: 0.99,
            lambda: 0.0,
        };
        let r = vec![1.0_f32; 3];
        let q = vec![0.5_f32; 3];
        let v = vec![0.5_f32; 4];
        let d = vec![0.0_f32; 3];
        let lp = vec![0.0_f32; 3];
        let out = compute_retrace(&r, &q, &v, &d, &lp, &lp, cfg)
            .expect("valid equal-length slices should not fail");
        // Q^ret = Q + δ = Q + (r + γ*V_next - Q) = r + γ*V_next
        let expected = 1.0 + 0.99 * 0.5;
        for (i, &qt) in out.q_targets.iter().enumerate() {
            assert!(
                (qt - expected).abs() < 1e-4,
                "Q_target[{i}]={qt} vs {expected}"
            );
        }
    }
}
