//! DeepSeekMoE shared-expert isolation routing (Dai et al. 2024).
//!
//! Implements the routing-side bookkeeping of:
//! Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in
//! Mixture-of-Experts Language Models." ACL 2024.
//!
//! DeepSeekMoE splits the experts into two disjoint pools:
//!
//! * **`n_shared` shared experts** — *always* activated for every token. They
//!   capture common knowledge so the routed experts do not have to redundantly
//!   re-learn it, freeing them to specialise.
//! * **`n_routed` routed experts** — fine-grained experts of which each token
//!   activates its top-`k` by gate score (softmax over the routed pool only).
//!
//! The final mixture weight for a token is
//!
//! ```text
//! out = Σ_{s∈shared} 1·E_s(x)  +  Σ_{e∈top-k routed} g_e·E_e(x)
//! ```
//!
//! where the routed gates `g_e` are the renormalised top-k softmax scores. This
//! module computes the activation plan (which experts fire for each token and
//! with what weight); the actual expert FFNs are applied by the caller.

use crate::error::{MoeError, MoeResult};
use crate::routing::top_k::{stable_softmax, topk};

/// Configuration for DeepSeekMoE shared-expert isolation.
#[derive(Debug, Clone)]
pub struct SharedExpertConfig {
    /// Number of always-on shared experts `m_s ≥ 0`.
    pub n_shared: usize,
    /// Number of fine-grained routed experts `m_r ≥ 1`.
    pub n_routed: usize,
    /// Number of routed experts activated per token `k` (`1 ≤ k ≤ n_routed`).
    pub top_k_routed: usize,
    /// Input feature dimension `d`.
    pub input_dim: usize,
}

/// Activation plan produced by shared-expert routing.
#[derive(Debug, Clone)]
pub struct SharedExpertResult {
    /// Shared-expert *global* indices (`0 .. n_shared`), always activated.
    pub shared_indices: Vec<usize>,
    /// Routed-expert *global* indices selected per token `[T×k]`.
    ///
    /// Global index of routed expert `r` is `n_shared + r`.
    pub routed_indices: Vec<usize>,
    /// Routed gate weights aligned with `routed_indices` `[T×k]`
    /// (top-k softmax over the routed pool, renormalised to sum to `1`).
    pub routed_gates: Vec<f32>,
    /// Combine weight applied to every shared expert (constant `1.0`).
    pub shared_weight: f32,
}

impl SharedExpertResult {
    /// Number of always-on shared experts in this plan.
    #[must_use]
    pub fn n_shared(&self) -> usize {
        self.shared_indices.len()
    }

    /// Number of routed experts activated per token (`top_k_routed`).
    ///
    /// Returns `0` when `n_tokens == 0`.
    #[must_use]
    pub fn top_k_routed(&self, n_tokens: usize) -> usize {
        self.routed_gates.len().checked_div(n_tokens).unwrap_or(0)
    }

    /// Total number of expert activations per token (`n_shared + top_k_routed`).
    #[must_use]
    pub fn activations_per_token(&self, n_tokens: usize) -> usize {
        self.n_shared() + self.top_k_routed(n_tokens)
    }
}

/// Total number of experts in the layer (`n_shared + n_routed`).
#[must_use]
pub fn total_experts(cfg: &SharedExpertConfig) -> usize {
    cfg.n_shared + cfg.n_routed
}

/// Compute the DeepSeekMoE activation plan for a batch of tokens.
///
/// # Arguments
/// * `routed_logits` — router logits over the **routed** pool only,
///   shape `[T × n_routed]`, row-major (typically `x @ W_router^T`).
/// * `n_tokens` — `T`.
/// * `cfg` — shared-expert configuration.
///
/// # Errors
/// Returns [`MoeError`] for an empty input, `n_routed == 0`, an invalid
/// `top_k_routed`, or a `routed_logits` / `T·n_routed` length mismatch.
pub fn shared_expert_route(
    routed_logits: &[f32],
    n_tokens: usize,
    cfg: &SharedExpertConfig,
) -> MoeResult<SharedExpertResult> {
    if cfg.n_routed == 0 {
        return Err(MoeError::InvalidExpertCount {
            n_experts: cfg.n_routed,
        });
    }
    if cfg.top_k_routed == 0 || cfg.top_k_routed > cfg.n_routed {
        return Err(MoeError::InvalidTopK {
            k: cfg.top_k_routed,
            n_experts: cfg.n_routed,
        });
    }
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    let expected = n_tokens * cfg.n_routed;
    if routed_logits.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: routed_logits.len(),
        });
    }

    let k = cfg.top_k_routed;
    let n_r = cfg.n_routed;

    // Shared experts always fire (global indices 0 .. n_shared).
    let shared_indices: Vec<usize> = (0..cfg.n_shared).collect();

    let mut routed_indices = vec![0_usize; n_tokens * k];
    let mut routed_gates = vec![0.0_f32; n_tokens * k];

    for t in 0..n_tokens {
        let logit_row = &routed_logits[t * n_r..(t + 1) * n_r];
        // Softmax over the full routed pool (DeepSeek normalises over all routed
        // experts, then keeps the top-k and renormalises those).
        let probs = stable_softmax(logit_row);
        let (_top_p, top_idx) = topk(&probs, k)?;

        // Renormalise the selected k probabilities to sum to 1.
        let mut sel_sum = 0.0_f32;
        for &ri in &top_idx {
            sel_sum += probs[ri];
        }
        let denom = sel_sum + 1e-12;
        for j in 0..k {
            let ri = top_idx[j];
            routed_indices[t * k + j] = cfg.n_shared + ri; // global index
            routed_gates[t * k + j] = probs[ri] / denom;
        }
    }

    if routed_gates.iter().any(|v| !v.is_finite()) {
        return Err(MoeError::NanEncountered {
            context: "shared_expert_route".to_string(),
        });
    }

    Ok(SharedExpertResult {
        shared_indices,
        routed_indices,
        routed_gates,
        shared_weight: 1.0,
    })
}

/// Combine shared and routed expert outputs into the final token representation.
///
/// `out[t] = Σ_shared shared_out[s,t] + Σ_j routed_gates[t,j]·routed_out[t,j]`.
///
/// # Arguments
/// * `shared_out` — shared-expert outputs, shape `[n_shared × T × d_model]`
///   (shared-expert-major, then token, then feature).
/// * `routed_out` — routed-expert outputs in selection order,
///   shape `[T × k × d_model]`.
/// * `result` — the activation plan from [`shared_expert_route`].
/// * `n_tokens` — `T`.
/// * `d_model` — output feature dimension.
///
/// # Errors
/// Returns [`MoeError`] on any shape mismatch or `d_model == 0`.
pub fn shared_expert_combine(
    shared_out: &[f32],
    routed_out: &[f32],
    result: &SharedExpertResult,
    n_tokens: usize,
    d_model: usize,
) -> MoeResult<Vec<f32>> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    let n_shared = result.shared_indices.len();
    let k = result.routed_gates.len() / n_tokens;

    let expected_shared = n_shared * n_tokens * d_model;
    if shared_out.len() != expected_shared {
        return Err(MoeError::DimensionMismatch {
            expected: expected_shared,
            got: shared_out.len(),
        });
    }
    let expected_routed = n_tokens * k * d_model;
    if routed_out.len() != expected_routed {
        return Err(MoeError::DimensionMismatch {
            expected: expected_routed,
            got: routed_out.len(),
        });
    }

    let mut out = vec![0.0_f32; n_tokens * d_model];

    // Shared contribution (weight 1.0 each).
    for s in 0..n_shared {
        for t in 0..n_tokens {
            let src = (s * n_tokens + t) * d_model;
            let dst = t * d_model;
            for f in 0..d_model {
                out[dst + f] += result.shared_weight * shared_out[src + f];
            }
        }
    }

    // Routed contribution (gate-weighted).
    for t in 0..n_tokens {
        for j in 0..k {
            let g = result.routed_gates[t * k + j];
            let src = (t * k + j) * d_model;
            let dst = t * d_model;
            for f in 0..d_model {
                out[dst + f] += g * routed_out[src + f];
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_shared: usize, n_routed: usize, k: usize, d: usize) -> SharedExpertConfig {
        SharedExpertConfig {
            n_shared,
            n_routed,
            top_k_routed: k,
            input_dim: d,
        }
    }

    fn logits(n_tokens: usize, n_routed: usize) -> Vec<f32> {
        (0..n_tokens * n_routed)
            .map(|i| ((i as f32) * 0.21).sin() * 2.0)
            .collect()
    }

    #[test]
    fn total_experts_sums_pools() {
        assert_eq!(total_experts(&cfg(2, 6, 2, 8)), 8);
    }

    #[test]
    fn route_zero_routed_errors() {
        let l: Vec<f32> = vec![];
        assert!(matches!(
            shared_expert_route(&l, 4, &cfg(2, 0, 1, 8)),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn route_invalid_k_errors() {
        let l = logits(4, 6);
        assert!(shared_expert_route(&l, 4, &cfg(2, 6, 0, 8)).is_err());
        assert!(shared_expert_route(&l, 4, &cfg(2, 6, 7, 8)).is_err());
    }

    #[test]
    fn route_zero_tokens_errors() {
        let l: Vec<f32> = vec![];
        assert!(matches!(
            shared_expert_route(&l, 0, &cfg(2, 6, 2, 8)),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn route_logit_mismatch_errors() {
        let l = vec![0.0_f32; 10]; // should be 4*6 = 24
        assert!(matches!(
            shared_expert_route(&l, 4, &cfg(2, 6, 2, 8)),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn shared_always_activated() {
        let l = logits(5, 6);
        let res = shared_expert_route(&l, 5, &cfg(3, 6, 2, 8)).expect("value should be present");
        assert_eq!(res.shared_indices, vec![0, 1, 2]);
        assert!((res.shared_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_shared_is_valid() {
        let l = logits(4, 6);
        let res = shared_expert_route(&l, 4, &cfg(0, 6, 2, 8)).expect("value should be present");
        assert!(res.shared_indices.is_empty());
    }

    #[test]
    fn routed_gates_sum_to_one_per_token() {
        let n_tokens = 8;
        let k = 3;
        let l = logits(n_tokens, 8);
        let res =
            shared_expert_route(&l, n_tokens, &cfg(2, 8, k, 16)).expect("value should be present");
        for t in 0..n_tokens {
            let s: f32 = res.routed_gates[t * k..(t + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "token {t} routed gate sum {s}");
        }
    }

    #[test]
    fn routed_indices_offset_by_n_shared() {
        let n_tokens = 6;
        let n_shared = 2;
        let res = shared_expert_route(&logits(n_tokens, 6), n_tokens, &cfg(n_shared, 6, 2, 8))
            .expect("value should be present");
        // global routed indices must be >= n_shared and < total.
        let total = total_experts(&cfg(n_shared, 6, 2, 8));
        for &gi in &res.routed_indices {
            assert!(gi >= n_shared && gi < total, "global idx {gi} out of range");
        }
    }

    #[test]
    fn routed_picks_highest_logit() {
        // Token 0 strongly prefers routed expert 3 (global 3+n_shared).
        let n_shared = 1;
        let n_routed = 4;
        let mut l = vec![0.0_f32; n_routed];
        l[3] = 10.0;
        let res = shared_expert_route(&l, 1, &cfg(n_shared, n_routed, 1, 8))
            .expect("value should be present");
        assert_eq!(res.routed_indices[0], n_shared + 3);
    }

    #[test]
    fn combine_shape_and_values() {
        let n_tokens = 2;
        let n_shared = 1;
        let n_routed = 4;
        let k = 2;
        let d = 3;
        let res = shared_expert_route(
            &logits(n_tokens, n_routed),
            n_tokens,
            &cfg(n_shared, n_routed, k, d),
        )
        .expect("value should be present");
        // shared_out: [n_shared × T × d]; all ones.
        let shared_out = vec![1.0_f32; n_shared * n_tokens * d];
        // routed_out: [T × k × d]; all twos.
        let routed_out = vec![2.0_f32; n_tokens * k * d];
        let out = shared_expert_combine(&shared_out, &routed_out, &res, n_tokens, d)
            .expect("shared_expert_combine should succeed");
        assert_eq!(out.len(), n_tokens * d);
        // Each output = 1 (shared) + sum_j g_j * 2 = 1 + 2*(sum g_j=1) = 3.
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-4, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn combine_zero_dmodel_errors() {
        let res = shared_expert_route(&logits(2, 4), 2, &cfg(1, 4, 2, 8))
            .expect("value should be present");
        assert!(matches!(
            shared_expert_combine(&[], &[], &res, 2, 0),
            Err(MoeError::InvalidHiddenDim { .. })
        ));
    }

    #[test]
    fn combine_shared_shape_mismatch_errors() {
        let res = shared_expert_route(&logits(2, 4), 2, &cfg(1, 4, 2, 8))
            .expect("value should be present");
        let bad_shared = vec![1.0_f32; 5]; // wrong
        let routed_out = vec![2.0_f32; 2 * 2 * 3];
        assert!(matches!(
            shared_expert_combine(&bad_shared, &routed_out, &res, 2, 3),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn deterministic_route() {
        let l = logits(6, 6);
        let a = shared_expert_route(&l, 6, &cfg(2, 6, 2, 8)).expect("value should be present");
        let b = shared_expert_route(&l, 6, &cfg(2, 6, 2, 8)).expect("value should be present");
        assert_eq!(a.routed_indices, b.routed_indices);
        for (x, y) in a.routed_gates.iter().zip(b.routed_gates.iter()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn activation_count_helpers() {
        let n_tokens = 4;
        let res = shared_expert_route(&logits(n_tokens, 6), n_tokens, &cfg(2, 6, 3, 8))
            .expect("value should be present");
        assert_eq!(res.n_shared(), 2);
        assert_eq!(res.top_k_routed(n_tokens), 3);
        assert_eq!(res.activations_per_token(n_tokens), 5);
        assert_eq!(res.top_k_routed(0), 0);
    }
}
