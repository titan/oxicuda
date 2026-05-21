//! Off-policy evaluation (OPE) for contextual bandits.
//!
//! Estimates V(π) = E_x[Σ_a π(a|x) Q(x,a)] for a target policy π given
//! logged data collected under a behavior policy π_0. Implements three
//! estimators: IPS, SNIPS, and Doubly-Robust, with jackknife standard errors.

use crate::error::{CausalError, CausalResult};

/// Off-policy evaluation for contextual bandit data.
///
/// Input: n logged samples (context x_i, action a_i, reward r_i, behavior
/// probability π_0(a_i|x_i)).
/// Goal: estimate V(π) = E_{x}[Σ_a π(a|x) Q(x,a)] for target policy π.
#[derive(Debug, Clone)]
pub struct OpeInput {
    /// π_0(a_i | x_i): behavior policy probability for the taken action. Length n.
    pub logging_probs: Vec<f32>,
    /// π(a_i | x_i): target policy probability for the same action. Length n.
    pub target_probs: Vec<f32>,
    /// r_i: observed reward. Length n.
    pub rewards: Vec<f32>,
    /// Q̂(x_i, a_i): estimated Q-value of the taken action. Length n. Used by DR.
    /// If None: DR is not computed.
    pub q_estimates: Option<Vec<f32>>,
    /// Σ_a π(a|x_i) * Q̂(x_i, a): direct method value estimate for x_i. Length n.
    /// If None: DR is not computed.
    pub direct_values: Option<Vec<f32>>,
}

/// Off-policy evaluation result.
#[derive(Debug, Clone)]
pub struct OpeResult {
    /// IPS estimate: (1/n) Σ_i w_i r_i, w_i = π/π_0.
    pub ips: f32,
    /// SNIPS (self-normalized IPS): Σ_i w_i r_i / Σ_i w_i.
    pub snips: f32,
    /// DR estimate (if q_estimates and direct_values were provided).
    pub dr: Option<f32>,
    /// Jackknife standard error of IPS.
    pub ips_se: f32,
    /// Jackknife standard error of SNIPS.
    pub snips_se: f32,
    /// Maximum clipping value applied to importance weights.
    pub clip_max: f32,
}

/// Evaluate a target policy given logged contextual-bandit data.
///
/// # Arguments
/// * `input`    — logged data and estimates.
/// * `clip_max` — maximum importance weight w = π/π_0 (clipped to [0, clip_max]).
///   Use f32::MAX for no clipping.
///
/// # Errors
/// Returns `CausalError::EmptyInput` if all vectors are empty.
/// Returns `CausalError::IncompatibleData` if vector lengths differ or
/// q_estimates/direct_values are inconsistent.
/// Returns `CausalError::PropensityOutOfBounds` if any logging_prob is ≤ 0.
/// Returns `CausalError::InvalidParameter` if clip_max < 1.0.
/// Returns `CausalError::Internal` if SNIPS denominator is ≈ 0.
pub fn ope_evaluate(input: &OpeInput, clip_max: f32) -> CausalResult<OpeResult> {
    let n = input.logging_probs.len();

    // Validate non-empty
    if n == 0 {
        return Err(CausalError::EmptyInput);
    }

    // Validate all primary vectors same length
    if input.target_probs.len() != n || input.rewards.len() != n {
        return Err(CausalError::IncompatibleData);
    }

    // Validate optional vectors
    match (&input.q_estimates, &input.direct_values) {
        (Some(q), Some(dm)) => {
            if q.len() != n || dm.len() != n {
                return Err(CausalError::IncompatibleData);
            }
        }
        (None, None) => {}
        _ => {
            // One is Some and the other is None — inconsistent
            return Err(CausalError::IncompatibleData);
        }
    }

    // Validate clip_max ≥ 1.0
    if clip_max < 1.0 {
        return Err(CausalError::InvalidParameter {
            reason: format!("clip_max must be ≥ 1.0, got {clip_max}"),
        });
    }

    // Validate all logging probs > 0
    for &p in &input.logging_probs {
        if p <= 0.0 {
            return Err(CausalError::PropensityOutOfBounds { value: p });
        }
    }

    // Compute importance weights w_i = clamp(π(a|x) / π_0(a|x), 0, clip_max)
    let weights: Vec<f32> = input
        .target_probs
        .iter()
        .zip(input.logging_probs.iter())
        .map(|(&pi, &pi0)| (pi / pi0).clamp(0.0, clip_max))
        .collect();

    // Compute w_i * r_i products
    let wr: Vec<f32> = weights
        .iter()
        .zip(input.rewards.iter())
        .map(|(&w, &r)| w * r)
        .collect();

    // IPS: (1/n) Σ w_i r_i
    let sum_wr: f32 = wr.iter().sum();
    let ips = sum_wr / n as f32;

    // SNIPS: Σ w_i r_i / Σ w_i
    let sum_w: f32 = weights.iter().sum();
    if sum_w.abs() < 1e-10 {
        return Err(CausalError::Internal {
            msg: "SNIPS denominator (sum of importance weights) is approximately zero".to_string(),
        });
    }
    let snips = sum_wr / sum_w;

    // Doubly-Robust estimate (optional)
    let dr = match (&input.q_estimates, &input.direct_values) {
        (Some(q), Some(dm)) => {
            let dr_val: f32 = (0..n)
                .map(|i| dm[i] + weights[i] * (input.rewards[i] - q[i]))
                .sum::<f32>()
                / n as f32;
            Some(dr_val)
        }
        _ => None,
    };

    // Jackknife standard errors
    let (ips_se, snips_se) = if n == 1 {
        (0.0_f32, 0.0_f32)
    } else {
        let n_f = n as f32;

        // IPS jackknife: IPS_{-i} = (sum_wr - w_i r_i) / (n - 1)
        let ips_loos: Vec<f32> = (0..n).map(|i| (sum_wr - wr[i]) / (n_f - 1.0)).collect();
        let mean_ips_loo = ips_loos.iter().sum::<f32>() / n_f;
        let ips_var = ips_loos
            .iter()
            .map(|&v| (v - mean_ips_loo) * (v - mean_ips_loo))
            .sum::<f32>();
        let ips_se = ((n_f - 1.0) / n_f * ips_var).sqrt();

        // SNIPS jackknife: SNIPS_{-i} = (sum_wr - w_i r_i) / (sum_w - w_i)
        let snips_loos: Vec<f32> = (0..n)
            .map(|i| {
                let denom = sum_w - weights[i];
                if denom.abs() < 1e-10 {
                    snips // fallback to full-sample estimate
                } else {
                    (sum_wr - wr[i]) / denom
                }
            })
            .collect();
        let mean_snips_loo = snips_loos.iter().sum::<f32>() / n_f;
        let snips_var = snips_loos
            .iter()
            .map(|&v| (v - mean_snips_loo) * (v - mean_snips_loo))
            .sum::<f32>();
        let snips_se = ((n_f - 1.0) / n_f * snips_var).sqrt();

        (ips_se, snips_se)
    };

    Ok(OpeResult {
        ips,
        snips,
        dr,
        ips_se,
        snips_se,
        clip_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_input(n: usize, reward_val: f32) -> OpeInput {
        OpeInput {
            logging_probs: vec![0.5; n],
            target_probs: vec![0.5; n],
            rewards: vec![reward_val; n],
            q_estimates: None,
            direct_values: None,
        }
    }

    #[test]
    fn ips_uniform_policy_equals_mean_reward() {
        // When π = π_0, w_i = 1.0, IPS = mean(rewards)
        let input = OpeInput {
            logging_probs: vec![0.3, 0.3, 0.3, 0.3],
            target_probs: vec![0.3, 0.3, 0.3, 0.3],
            rewards: vec![1.0, 2.0, 3.0, 4.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        let expected_mean = (1.0_f32 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!(
            (result.ips - expected_mean).abs() < 1e-5,
            "ips={} expected={}",
            result.ips,
            expected_mean
        );
    }

    #[test]
    fn snips_uniform_policy_equals_mean_reward() {
        // When π = π_0, w_i = 1.0, SNIPS = Σ r_i / Σ 1 = mean(rewards)
        let input = OpeInput {
            logging_probs: vec![0.4, 0.4, 0.4],
            target_probs: vec![0.4, 0.4, 0.4],
            rewards: vec![2.0, 4.0, 6.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        let expected_mean = (2.0_f32 + 4.0 + 6.0) / 3.0;
        assert!(
            (result.snips - expected_mean).abs() < 1e-5,
            "snips={} expected={}",
            result.snips,
            expected_mean
        );
    }

    #[test]
    fn dr_none_when_no_q_estimates() {
        let input = uniform_input(5, 1.0);
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.dr.is_none());
    }

    #[test]
    fn dr_reduces_to_dm_when_rewards_match_q() {
        // When r_i == q(x_i, a_i), the correction term vanishes:
        // dr = (1/n) Σ [dm_i + w_i * (r_i - q_i)] = (1/n) Σ dm_i
        let n = 5;
        let rewards = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let dm = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5];
        let input = OpeInput {
            logging_probs: vec![0.5; n],
            target_probs: vec![0.5; n],
            rewards: rewards.clone(),
            q_estimates: Some(rewards), // q_i == r_i → correction = 0
            direct_values: Some(dm.clone()),
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        let expected_dm_mean = dm.iter().sum::<f32>() / n as f32;
        let dr = result.dr.unwrap();
        assert!(
            (dr - expected_dm_mean).abs() < 1e-5,
            "dr={} expected={}",
            dr,
            expected_dm_mean
        );
    }

    #[test]
    fn clipping_reduces_large_weights() {
        // With very small logging probs, weights blow up without clipping
        let n = 4;
        let input_no_clip = OpeInput {
            logging_probs: vec![0.001; n],
            target_probs: vec![0.5; n],
            rewards: vec![1.0; n],
            q_estimates: None,
            direct_values: None,
        };
        let result_clipped = ope_evaluate(&input_no_clip, 10.0).unwrap();
        // With clip_max=10.0, ips ≤ 10 * mean(rewards) = 10
        assert!(
            result_clipped.ips <= 10.0 + 1e-5,
            "ips with clipping={}",
            result_clipped.ips
        );

        // Without clipping the weight would be 0.5/0.001 = 500
        let result_no_clip = ope_evaluate(&input_no_clip, f32::MAX).unwrap();
        assert!(
            result_no_clip.ips > 10.0,
            "unclipped ips={}",
            result_no_clip.ips
        );
    }

    #[test]
    fn ips_nonneg_for_nonneg_rewards() {
        let input = OpeInput {
            logging_probs: vec![0.3, 0.5, 0.2, 0.4],
            target_probs: vec![0.6, 0.3, 0.5, 0.1],
            rewards: vec![0.0, 1.0, 2.0, 3.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.ips >= 0.0, "ips={}", result.ips);
    }

    #[test]
    fn snips_bounded_by_max_reward() {
        // For non-negative weights and rewards in [-R, R], |SNIPS| ≤ R
        let r_max = 5.0_f32;
        let input = OpeInput {
            logging_probs: vec![0.2, 0.3, 0.5, 0.4, 0.1],
            target_probs: vec![0.4, 0.6, 0.2, 0.8, 0.3],
            rewards: vec![-r_max, r_max, -r_max, r_max, -r_max],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.snips.abs() <= r_max + 1e-4, "snips={}", result.snips);
    }

    #[test]
    fn ips_single_sample() {
        let input = OpeInput {
            logging_probs: vec![0.5],
            target_probs: vec![0.3],
            rewards: vec![2.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        // w = 0.3/0.5 = 0.6, ips = 0.6 * 2.0 = 1.2
        assert!((result.ips - 1.2).abs() < 1e-5, "ips={}", result.ips);
        assert_eq!(result.ips_se, 0.0);
    }

    #[test]
    fn err_empty_input() {
        let input = OpeInput {
            logging_probs: vec![],
            target_probs: vec![],
            rewards: vec![],
            q_estimates: None,
            direct_values: None,
        };
        assert!(matches!(
            ope_evaluate(&input, f32::MAX),
            Err(CausalError::EmptyInput)
        ));
    }

    #[test]
    fn err_mismatched_lengths() {
        let input = OpeInput {
            logging_probs: vec![0.5, 0.5, 0.5],
            target_probs: vec![0.3, 0.3],
            rewards: vec![1.0, 2.0, 3.0],
            q_estimates: None,
            direct_values: None,
        };
        assert!(matches!(
            ope_evaluate(&input, f32::MAX),
            Err(CausalError::IncompatibleData)
        ));
    }

    #[test]
    fn err_zero_logging_prob() {
        let input = OpeInput {
            logging_probs: vec![0.0, 0.5],
            target_probs: vec![0.3, 0.3],
            rewards: vec![1.0, 2.0],
            q_estimates: None,
            direct_values: None,
        };
        assert!(matches!(
            ope_evaluate(&input, f32::MAX),
            Err(CausalError::PropensityOutOfBounds { .. })
        ));
    }

    #[test]
    fn err_clip_max_lt_1() {
        let input = uniform_input(3, 1.0);
        assert!(matches!(
            ope_evaluate(&input, 0.5),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn ips_se_nonneg() {
        let input = OpeInput {
            logging_probs: vec![0.2, 0.4, 0.3, 0.5, 0.1],
            target_probs: vec![0.5, 0.2, 0.6, 0.1, 0.7],
            rewards: vec![1.0, -1.0, 2.0, -2.0, 3.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.ips_se >= 0.0, "ips_se={}", result.ips_se);
    }

    #[test]
    fn snips_se_nonneg() {
        let input = OpeInput {
            logging_probs: vec![0.2, 0.4, 0.3, 0.5, 0.1],
            target_probs: vec![0.5, 0.2, 0.6, 0.1, 0.7],
            rewards: vec![1.0, -1.0, 2.0, -2.0, 3.0],
            q_estimates: None,
            direct_values: None,
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.snips_se >= 0.0, "snips_se={}", result.snips_se);
    }

    #[test]
    fn clip_max_stored_in_result() {
        let input = uniform_input(4, 1.0);
        let clip = 5.0_f32;
        let result = ope_evaluate(&input, clip).unwrap();
        assert!((result.clip_max - clip).abs() < 1e-10);
    }

    #[test]
    fn dr_computed_when_provided() {
        let n = 4;
        let input = OpeInput {
            logging_probs: vec![0.5; n],
            target_probs: vec![0.3; n],
            rewards: vec![1.0, 2.0, 3.0, 4.0],
            q_estimates: Some(vec![1.0, 2.0, 3.0, 4.0]),
            direct_values: Some(vec![1.5, 2.5, 3.5, 4.5]),
        };
        let result = ope_evaluate(&input, f32::MAX).unwrap();
        assert!(result.dr.is_some());
    }

    #[test]
    fn deterministic_computation() {
        let input = OpeInput {
            logging_probs: vec![0.3, 0.5, 0.4, 0.2],
            target_probs: vec![0.6, 0.2, 0.8, 0.4],
            rewards: vec![1.0, 2.0, -1.0, 3.0],
            q_estimates: Some(vec![0.9, 1.8, -0.9, 2.7]),
            direct_values: Some(vec![1.1, 1.9, -0.8, 3.1]),
        };
        let r1 = ope_evaluate(&input, 10.0).unwrap();
        let r2 = ope_evaluate(&input, 10.0).unwrap();
        assert!((r1.ips - r2.ips).abs() < 1e-10);
        assert!((r1.snips - r2.snips).abs() < 1e-10);
        assert!((r1.dr.unwrap() - r2.dr.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn large_n_works() {
        let n = 1000;
        // Generate pseudo-random data using a simple LCG
        let mut state: u64 = 42;
        let mut next_f32 = || -> f32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };

        let logging_probs: Vec<f32> = (0..n).map(|_| 0.05 + next_f32() * 0.9).collect();
        let target_probs: Vec<f32> = (0..n).map(|_| 0.05 + next_f32() * 0.9).collect();
        let rewards: Vec<f32> = (0..n).map(|_| next_f32() * 2.0 - 1.0).collect();

        let input = OpeInput {
            logging_probs,
            target_probs,
            rewards,
            q_estimates: None,
            direct_values: None,
        };

        let result = ope_evaluate(&input, 20.0).unwrap();
        assert!(result.ips.is_finite());
        assert!(result.snips.is_finite());
        assert!(result.ips_se.is_finite());
        assert!(result.snips_se.is_finite());
    }

    #[test]
    fn err_inconsistent_q_and_dm() {
        // q_estimates Some, direct_values None → IncompatibleData
        let input = OpeInput {
            logging_probs: vec![0.5, 0.5],
            target_probs: vec![0.3, 0.3],
            rewards: vec![1.0, 2.0],
            q_estimates: Some(vec![1.0, 2.0]),
            direct_values: None,
        };
        assert!(matches!(
            ope_evaluate(&input, f32::MAX),
            Err(CausalError::IncompatibleData)
        ));
    }
}
