//! Switch Transformer load balance auxiliary loss.
//!
//! `L_aux = n_experts * Σ_{i=1}^{E} f_i * P_i`
//!
//! where:
//! - `f_i` = fraction of tokens dispatched to expert `i` (hard assignment)
//! - `P_i` = mean gate probability for expert `i` (soft routing logits)
//!
//! The loss is minimized when load is perfectly uniform across experts.

use crate::error::{MoeError, MoeResult};
use crate::routing::top_k::stable_softmax;

/// Per-expert load statistics.
#[derive(Debug, Clone)]
pub struct LoadStats {
    /// Fraction of tokens routed to each expert: `f_i = count_i / T`.
    pub f: Vec<f32>,
    /// Mean gate probability for each expert: `P_i = (1/T) Σ_t p_{t,i}`.
    pub p: Vec<f32>,
    /// Load balance loss value.
    pub balance_loss: f32,
    /// Maximum load fraction across experts (imbalance indicator).
    pub max_load_fraction: f32,
}

/// Compute the Switch Transformer load balance loss.
///
/// # Arguments
/// * `router_logits` — raw logits before softmax, shape `[n_tokens * n_experts]`
/// * `expert_assignments` — hard assignments per token (usize::MAX = dropped), shape `[n_tokens]`
/// * `n_tokens` — number of tokens
/// * `n_experts` — number of experts
pub fn load_balance_loss(
    router_logits: &[f32],
    expert_assignments: &[usize],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    let stats = compute_load_stats(router_logits, expert_assignments, n_tokens, n_experts)?;
    Ok(stats.balance_loss)
}

/// Compute detailed load statistics including per-expert fractions.
pub fn compute_load_stats(
    router_logits: &[f32],
    expert_assignments: &[usize],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<LoadStats> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected_logits = n_tokens * n_experts;
    if router_logits.len() != expected_logits {
        return Err(MoeError::DimensionMismatch {
            expected: expected_logits,
            got: router_logits.len(),
        });
    }
    if expert_assignments.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: expert_assignments.len(),
        });
    }

    // f_i: count tokens assigned to expert i
    let mut expert_counts = vec![0_usize; n_experts];
    for &assignment in expert_assignments.iter() {
        if assignment != usize::MAX {
            if assignment >= n_experts {
                return Err(MoeError::ExpertIndexOutOfRange {
                    idx: assignment,
                    n_experts,
                });
            }
            expert_counts[assignment] += 1;
        }
    }
    let token_count_f32 = n_tokens as f32;
    let fraction_per_expert: Vec<f32> = expert_counts
        .iter()
        .map(|&cnt| cnt as f32 / token_count_f32)
        .collect();

    // P_i: mean gate probability for expert i
    let mut prob_sum_per_expert = vec![0.0_f32; n_experts];
    for tok in 0..n_tokens {
        let logit_row = &router_logits[tok * n_experts..(tok + 1) * n_experts];
        let probs = stable_softmax(logit_row);
        for (exp_idx, &prob) in probs.iter().enumerate() {
            prob_sum_per_expert[exp_idx] += prob;
        }
    }
    let mean_prob_per_expert: Vec<f32> = prob_sum_per_expert
        .iter()
        .map(|&s| s / token_count_f32)
        .collect();

    // L_aux = n_experts * Σ f_i * P_i
    let balance_loss: f32 = n_experts as f32
        * fraction_per_expert
            .iter()
            .zip(mean_prob_per_expert.iter())
            .map(|(&fi, &pi)| fi * pi)
            .sum::<f32>();

    let max_load_fraction = fraction_per_expert.iter().cloned().fold(0.0_f32, f32::max);

    Ok(LoadStats {
        f: fraction_per_expert,
        p: mean_prob_per_expert,
        balance_loss,
        max_load_fraction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_balance_uniform_is_low() {
        // Perfectly uniform: 2 tokens per expert, 4 experts
        let n_tokens = 8;
        let n_experts = 4;
        // Equal logits → equal probs
        let router_logits = vec![1.0_f32; n_tokens * n_experts];
        // Assign tokens round-robin
        let assignments: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let loss = load_balance_loss(&router_logits, &assignments, n_tokens, n_experts).unwrap();
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn load_stats_sum_to_one() {
        let n_tokens = 4;
        let n_experts = 2;
        let router_logits = vec![0.5_f32; n_tokens * n_experts];
        let assignments = [0_usize, 0, 1, 1];
        let stats = compute_load_stats(&router_logits, &assignments, n_tokens, n_experts).unwrap();
        let f_sum: f32 = stats.f.iter().sum();
        assert!((f_sum - 1.0).abs() < 1e-5, "f_sum={f_sum}");
        let p_sum: f32 = stats.p.iter().sum();
        assert!((p_sum - 1.0).abs() < 1e-4, "p_sum={p_sum}");
    }
}
