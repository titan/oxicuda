//! Expert Choice routing: each expert selects its top-c preferred tokens.
//!
//! Implements the routing from:
//! Zhou et al. "Mixture-of-Experts with Expert Choice Routing." NeurIPS 2022.
//!
//! Guarantees perfect load balance: each expert processes exactly `capacity` tokens.

use crate::error::{MoeError, MoeResult};
use crate::routing::top_k::stable_softmax;

/// Configuration for Expert Choice routing.
#[derive(Debug, Clone)]
pub struct ExpertChoiceConfig {
    /// Total number of experts.
    pub n_experts: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Capacity factor for token selection (capacity = floor(T / E * factor)).
    pub capacity_factor: f32,
}

/// Output of Expert Choice routing.
#[derive(Debug, Clone)]
pub struct ExpertChoiceResult {
    /// Token indices selected by each expert.
    /// Shape: `[n_experts * capacity]` (expert-major).
    pub token_indices: Vec<usize>,
    /// Gate scores for each selected (expert, slot) pair.
    /// Shape: `[n_experts * capacity]`.
    pub scores: Vec<f32>,
    /// Actual capacity used (tokens per expert).
    pub capacity: usize,
}

/// Run expert-choice routing.
///
/// # Arguments
/// * `x` — token inputs, shape `[n_tokens * d_input]`
/// * `gate_weights` — expert gate matrix, shape `[n_experts * d_input]`
/// * `n_tokens` — number of tokens T
/// * `cfg` — expert choice configuration
pub fn expert_choice_route(
    x: &[f32],
    gate_weights: &[f32],
    n_tokens: usize,
    cfg: &ExpertChoiceConfig,
) -> MoeResult<ExpertChoiceResult> {
    if cfg.n_experts == 0 {
        return Err(MoeError::InvalidExpertCount {
            n_experts: cfg.n_experts,
        });
    }
    if cfg.input_dim == 0 {
        return Err(MoeError::InvalidInputDim { dim: cfg.input_dim });
    }
    if !cfg.capacity_factor.is_finite() || cfg.capacity_factor <= 0.0 {
        return Err(MoeError::InvalidCapacityFactor {
            factor: cfg.capacity_factor,
        });
    }
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    let expected_x = n_tokens * cfg.input_dim;
    if x.len() != expected_x {
        return Err(MoeError::DimensionMismatch {
            expected: expected_x,
            got: x.len(),
        });
    }
    let expected_w = cfg.n_experts * cfg.input_dim;
    if gate_weights.len() != expected_w {
        return Err(MoeError::DimensionMismatch {
            expected: expected_w,
            got: gate_weights.len(),
        });
    }

    // capacity = floor(T / E * cap_factor), minimum 1
    let capacity =
        ((n_tokens as f32 / cfg.n_experts as f32 * cfg.capacity_factor).floor() as usize).max(1);

    // Compute scores S = softmax(X · W_g^T) of shape [T, E]
    // First compute raw logits
    let mut logit_matrix = vec![0.0_f32; n_tokens * cfg.n_experts];
    for tok in 0..n_tokens {
        let x_row = &x[tok * cfg.input_dim..(tok + 1) * cfg.input_dim];
        for exp_idx in 0..cfg.n_experts {
            let w_row = &gate_weights[exp_idx * cfg.input_dim..(exp_idx + 1) * cfg.input_dim];
            let dot: f32 = x_row
                .iter()
                .zip(w_row.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum();
            logit_matrix[tok * cfg.n_experts + exp_idx] = dot;
        }
    }

    // Apply softmax over the expert dimension for each token
    let mut prob_matrix = vec![0.0_f32; n_tokens * cfg.n_experts];
    for tok in 0..n_tokens {
        let logit_row = &logit_matrix[tok * cfg.n_experts..(tok + 1) * cfg.n_experts];
        let probs = stable_softmax(logit_row);
        prob_matrix[tok * cfg.n_experts..(tok + 1) * cfg.n_experts].copy_from_slice(&probs);
    }

    // For each expert e: select top-capacity tokens by column e of prob_matrix
    let mut token_indices = vec![0_usize; cfg.n_experts * capacity];
    let mut output_scores = vec![0.0_f32; cfg.n_experts * capacity];

    for exp_idx in 0..cfg.n_experts {
        // Collect (prob, token_idx) for this expert column
        let mut expert_scores: Vec<(f32, usize)> = (0..n_tokens)
            .map(|tok| (prob_matrix[tok * cfg.n_experts + exp_idx], tok))
            .collect();

        // Partial sort to find top-capacity tokens
        if capacity < n_tokens {
            expert_scores.select_nth_unstable_by(capacity - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        expert_scores[..capacity]
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for slot in 0..capacity {
            let (score, tok) = expert_scores[slot];
            token_indices[exp_idx * capacity + slot] = tok;
            output_scores[exp_idx * capacity + slot] = score;
        }
    }

    Ok(ExpertChoiceResult {
        token_indices,
        scores: output_scores,
        capacity,
    })
}

/// Combine expert outputs using expert-choice assignments.
///
/// Scatters expert outputs back to token space:
/// `output[token_idx] += score * expert_output[slot]`
///
/// # Arguments
/// * `expert_out` — expert FFN outputs, shape `[n_experts * capacity * d_model]`
/// * `result` — expert choice routing result
/// * `scores` — gate scores, shape `[n_experts * capacity]` (from `ExpertChoiceResult`)
/// * `n_tokens` — number of tokens
/// * `n_experts` — number of experts
/// * `capacity` — tokens per expert
/// * `d_model` — output dimension
pub fn expert_choice_combine(
    expert_out: &[f32],
    result: &ExpertChoiceResult,
    scores: &[f32],
    n_tokens: usize,
    n_experts: usize,
    capacity: usize,
    d_model: usize,
) -> MoeResult<Vec<f32>> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    let expected_expert_out = n_experts * capacity * d_model;
    if expert_out.len() != expected_expert_out {
        return Err(MoeError::DimensionMismatch {
            expected: expected_expert_out,
            got: expert_out.len(),
        });
    }
    let expected_scores = n_experts * capacity;
    if scores.len() != expected_scores {
        return Err(MoeError::DimensionMismatch {
            expected: expected_scores,
            got: scores.len(),
        });
    }

    let mut output = vec![0.0_f32; n_tokens * d_model];

    for exp_idx in 0..n_experts {
        for slot in 0..capacity {
            let linear_idx = exp_idx * capacity + slot;
            let tok = result.token_indices[linear_idx];
            let score = scores[linear_idx];
            let expert_offset = linear_idx * d_model;
            let expert_slice = &expert_out[expert_offset..expert_offset + d_model];
            let out_slice = &mut output[tok * d_model..(tok + 1) * d_model];
            for (out_val, &exp_val) in out_slice.iter_mut().zip(expert_slice.iter()) {
                *out_val += score * exp_val;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_choice_basic() {
        let n_tokens = 8_usize;
        let n_experts = 2_usize;
        let input_dim = 4_usize;
        let cfg = ExpertChoiceConfig {
            n_experts,
            input_dim,
            capacity_factor: 1.0,
        };
        let x = vec![1.0_f32; n_tokens * input_dim];
        let gate_weights = vec![0.5_f32; n_experts * input_dim];
        let result = expert_choice_route(&x, &gate_weights, n_tokens, &cfg)
            .expect("expert_choice_route should succeed");
        // capacity = floor(8/2 * 1.0) = 4
        assert_eq!(result.capacity, 4);
        assert_eq!(result.token_indices.len(), n_experts * result.capacity);
        // Each index must be < n_tokens
        for &idx in &result.token_indices {
            assert!(idx < n_tokens);
        }
    }

    #[test]
    fn combine_produces_correct_shape() {
        let n_tokens = 4_usize;
        let n_experts = 2_usize;
        let capacity = 2_usize;
        let d_model = 3_usize;
        let result = ExpertChoiceResult {
            token_indices: vec![0, 1, 2, 3],
            scores: vec![0.5_f32; n_experts * capacity],
            capacity,
        };
        let expert_out = vec![1.0_f32; n_experts * capacity * d_model];
        let output = expert_choice_combine(
            &expert_out,
            &result,
            &result.scores.clone(),
            n_tokens,
            n_experts,
            capacity,
            d_model,
        )
        .expect("value should be present");
        assert_eq!(output.len(), n_tokens * d_model);
    }
}
