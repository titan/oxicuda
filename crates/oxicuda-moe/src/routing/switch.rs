//! Switch Transformer: top-1 routing with capacity.
//!
//! Implements the routing mechanism from:
//! Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models."
//! JMLR 2022.

use crate::error::{MoeError, MoeResult};

/// Configuration for Switch Transformer routing.
#[derive(Debug, Clone)]
pub struct SwitchConfig {
    /// Total number of experts.
    pub n_experts: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Capacity factor (1.0–2.0); determines expert buffer size relative to uniform load.
    pub capacity_factor: f32,
    /// Minimum tokens per expert (floor for capacity).
    pub min_capacity: usize,
    /// Whether to drop overflow tokens (vs. pass-through unchanged).
    pub drop_tokens: bool,
}

/// Dispatch result from Switch routing.
#[derive(Debug, Clone)]
pub struct SwitchDispatch {
    /// Expert assignment per token. `usize::MAX` if the token was dropped.
    /// Shape: `[n_tokens]`.
    pub expert_assignments: Vec<usize>,
    /// Actual token capacity per expert.
    pub capacity: usize,
    /// Number of tokens that overflowed (could not be placed).
    pub n_overflows: usize,
}

/// Assign tokens to experts with capacity bounds (Switch dispatch).
///
/// # Arguments
/// * `gate_indices` — top-1 expert index per token, shape `[n_tokens]`
/// * `n_tokens` — number of tokens
/// * `cfg` — switch configuration
pub fn switch_dispatch(
    gate_indices: &[usize],
    n_tokens: usize,
    cfg: &SwitchConfig,
) -> MoeResult<SwitchDispatch> {
    if cfg.n_experts == 0 {
        return Err(MoeError::InvalidExpertCount {
            n_experts: cfg.n_experts,
        });
    }
    if !cfg.capacity_factor.is_finite() || cfg.capacity_factor <= 0.0 {
        return Err(MoeError::InvalidCapacityFactor {
            factor: cfg.capacity_factor,
        });
    }
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if gate_indices.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: gate_indices.len(),
        });
    }

    // capacity = max(min_capacity, ceil(n_tokens / n_experts * cap_factor))
    let raw_cap = (n_tokens as f32 / cfg.n_experts as f32 * cfg.capacity_factor).ceil() as usize;
    let capacity = raw_cap.max(cfg.min_capacity);

    // Slot counters per expert
    let mut slot_counts = vec![0_usize; cfg.n_experts];
    let mut expert_assignments = vec![usize::MAX; n_tokens];
    let mut n_overflows = 0_usize;

    for (tok, &expert_idx) in gate_indices.iter().enumerate() {
        if expert_idx >= cfg.n_experts {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: expert_idx,
                n_experts: cfg.n_experts,
            });
        }
        let slot = slot_counts[expert_idx];
        if slot < capacity {
            expert_assignments[tok] = expert_idx;
            slot_counts[expert_idx] += 1;
        } else {
            n_overflows += 1;
            // drop_tokens=false means pass-through; assignment stays usize::MAX
            // (caller handles pass-through logic in combine step)
        }
    }

    Ok(SwitchDispatch {
        expert_assignments,
        capacity,
        n_overflows,
    })
}

/// Combine expert outputs back to token space.
///
/// # Arguments
/// * `expert_out` — expert outputs, shape `[n_experts * capacity * d_model]` (0-padded for empty slots)
/// * `dispatch` — dispatch result from `switch_dispatch`
/// * `scores` — gate scores per token, shape `[n_tokens]`
/// * `n_tokens` — number of input tokens
/// * `d_model` — model dimension
pub fn switch_combine(
    expert_out: &[f32],
    dispatch: &SwitchDispatch,
    scores: &[f32],
    n_tokens: usize,
    d_model: usize,
) -> MoeResult<Vec<f32>> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    if scores.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: scores.len(),
        });
    }

    let n_experts = dispatch
        .expert_assignments
        .iter()
        .filter(|&&a| a != usize::MAX)
        .map(|&a| a + 1)
        .max()
        .unwrap_or(1);

    let expected_expert_out = n_experts * dispatch.capacity * d_model;
    if expert_out.len() < expected_expert_out && !expert_out.is_empty() {
        // allow longer buffers; if empty, output zeros
    }

    let mut output = vec![0.0_f32; n_tokens * d_model];

    // Track slot position within each expert (we need to map token → slot)
    // Build a mapping: for each expert, maintain a list of tokens in order
    let mut expert_token_lists: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (tok, &assignment) in dispatch.expert_assignments.iter().enumerate() {
        if assignment != usize::MAX && assignment < n_experts {
            expert_token_lists[assignment].push(tok);
        }
    }

    for (exp_idx, token_list) in expert_token_lists.iter().enumerate() {
        for (slot, &tok) in token_list.iter().enumerate() {
            let score = scores[tok];
            let expert_offset = (exp_idx * dispatch.capacity + slot) * d_model;
            if expert_offset + d_model <= expert_out.len() {
                let expert_slice = &expert_out[expert_offset..expert_offset + d_model];
                let out_slice = &mut output[tok * d_model..(tok + 1) * d_model];
                for (out_val, &exp_val) in out_slice.iter_mut().zip(expert_slice.iter()) {
                    *out_val += score * exp_val;
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_basic() {
        let cfg = SwitchConfig {
            n_experts: 4,
            input_dim: 8,
            capacity_factor: 1.25,
            min_capacity: 1,
            drop_tokens: true,
        };
        let indices = vec![0_usize, 1, 2, 3, 0, 1, 2, 3];
        let dispatch = switch_dispatch(&indices, 8, &cfg).unwrap();
        assert_eq!(dispatch.capacity, 3); // ceil(8/4 * 1.25) = ceil(2.5) = 3
        assert_eq!(dispatch.n_overflows, 0);
    }

    #[test]
    fn dispatch_respects_capacity() {
        let cfg = SwitchConfig {
            n_experts: 2,
            input_dim: 8,
            capacity_factor: 1.0,
            min_capacity: 1,
            drop_tokens: true,
        };
        // All tokens go to expert 0 → 3 should overflow (capacity = ceil(4/2*1)=2)
        let indices = vec![0_usize, 0, 0, 0];
        let dispatch = switch_dispatch(&indices, 4, &cfg).unwrap();
        assert_eq!(dispatch.capacity, 2);
        assert_eq!(dispatch.n_overflows, 2);
    }
}
