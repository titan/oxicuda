//! Expert utilization metrics.

use crate::error::{MoeError, MoeResult};

/// Expert utilization statistics.
#[derive(Debug, Clone)]
pub struct ExpertUtilization {
    /// Number of tokens each expert received.
    pub tokens_per_expert: Vec<usize>,
    /// Number of tokens that were dropped (overflow).
    pub overflow_count: usize,
    /// `max(tokens_per_expert) / mean(tokens_per_expert)` — load imbalance ratio.
    /// 1.0 = perfectly balanced; higher = more imbalanced.
    pub load_imbalance_ratio: f32,
    /// `tokens_per_expert[i] / capacity` for each expert.
    pub utilization_fraction: Vec<f32>,
}

/// Compute expert utilization metrics from token assignments.
///
/// # Arguments
/// * `expert_assignments` — expert index per token (`usize::MAX` = dropped), shape `[n_tokens]`
/// * `n_tokens` — number of tokens
/// * `n_experts` — number of experts
/// * `capacity` — token capacity per expert
pub fn compute_utilization(
    expert_assignments: &[usize],
    n_tokens: usize,
    n_experts: usize,
    capacity: usize,
) -> MoeResult<ExpertUtilization> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    if expert_assignments.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: expert_assignments.len(),
        });
    }

    let mut tokens_per_expert = vec![0_usize; n_experts];
    let mut overflow_count = 0_usize;

    for &assignment in expert_assignments.iter() {
        if assignment == usize::MAX {
            overflow_count += 1;
        } else {
            if assignment >= n_experts {
                return Err(MoeError::ExpertIndexOutOfRange {
                    idx: assignment,
                    n_experts,
                });
            }
            tokens_per_expert[assignment] += 1;
        }
    }

    // Load imbalance ratio = max / mean
    let max_tokens = tokens_per_expert.iter().cloned().max().unwrap_or(0);
    let mean_tokens = tokens_per_expert.iter().sum::<usize>() as f32 / n_experts as f32;
    let load_imbalance_ratio = if mean_tokens > 1e-12 {
        max_tokens as f32 / mean_tokens
    } else {
        1.0
    };

    // Utilization fraction per expert
    let cap_f32 = capacity.max(1) as f32;
    let utilization_fraction: Vec<f32> = tokens_per_expert
        .iter()
        .map(|&cnt| cnt as f32 / cap_f32)
        .collect();

    Ok(ExpertUtilization {
        tokens_per_expert,
        overflow_count,
        load_imbalance_ratio,
        utilization_fraction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utilization_balanced() {
        let assignments = [0_usize, 1, 0, 1, 0, 1, 0, 1];
        let util = compute_utilization(&assignments, 8, 2, 4).unwrap();
        assert_eq!(util.tokens_per_expert, [4, 4]);
        assert_eq!(util.overflow_count, 0);
        assert!((util.load_imbalance_ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn utilization_with_overflow() {
        let assignments = [0_usize, usize::MAX, 1, usize::MAX];
        let util = compute_utilization(&assignments, 4, 2, 2).unwrap();
        assert_eq!(util.overflow_count, 2);
        assert_eq!(util.tokens_per_expert[0], 1);
        assert_eq!(util.tokens_per_expert[1], 1);
    }
}
