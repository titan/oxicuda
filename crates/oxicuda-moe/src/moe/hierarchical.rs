//! Hierarchical Mixture-of-Experts: two-level group → expert routing.
//!
//! Implements tree-structured conditional routing in the spirit of:
//! Jordan & Jacobs. "Hierarchical Mixtures of Experts and the EM Algorithm."
//! Neural Computation, 1994 — modernised to sparse top-k gating at each level.
//!
//! The `n_groups · experts_per_group` experts are partitioned into `n_groups`
//! disjoint groups. Routing factorises across two levels:
//!
//! 1. A **group gate** scores the `n_groups` groups; the token keeps its
//!    `top_k_group` groups (softmax → top-k → renormalised to sum to `1`).
//! 2. Within each selected group a **per-group expert gate** scores that group's
//!    `experts_per_group` experts; the token keeps `top_k_expert` of them
//!    (softmax → top-k → renormalised to sum to `1`).
//!
//! The combine weight for a `(group, expert)` pair is the product of the two
//! gate weights, and the global expert index is `group · experts_per_group +
//! local`. Because each level renormalises to `1`, the combine weights over all
//! selected pairs sum to `1` per token, and summing the pairs of a single group
//! recovers that group's weight:
//!
//! ```text
//! Σ_e combine[g, e] = group_weight[g] · Σ_e expert_weight[g, e] = group_weight[g]
//! Σ_g group_weight[g] = 1   ⇒   Σ_{g,e} combine[g, e] = 1
//! ```
//!
//! This structure shrinks the active router fan-out and is the basis of grouped
//! / hierarchical routing used in large expert pools.

use crate::error::{MoeError, MoeResult};
use crate::expert::bank::ExpertBank;
use crate::expert::ffn::ExpertActivation;
use crate::handle::LcgRng;
use crate::moe::{matvec, topk_balance_loss};
use crate::routing::top_k::{stable_softmax, topk};

/// Configuration for a [`HierarchicalMoeLayer`].
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Model (input/output) dimension `d_model`.
    pub d_model: usize,
    /// Hidden dimension of every expert FFN.
    pub ffn_dim: usize,
    /// Number of expert groups.
    pub n_groups: usize,
    /// Experts within each group (groups are equal-sized).
    pub experts_per_group: usize,
    /// Groups activated per token (`1 ≤ top_k_group ≤ n_groups`).
    pub top_k_group: usize,
    /// Experts activated per selected group (`1 ≤ top_k_expert ≤ experts_per_group`).
    pub top_k_expert: usize,
    /// Expert activation function.
    pub activation: ExpertActivation,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            ffn_dim: 1024,
            n_groups: 4,
            experts_per_group: 4,
            top_k_group: 1,
            top_k_expert: 1,
            activation: ExpertActivation::Gelu,
        }
    }
}

/// Routing decisions produced by a hierarchical forward pass.
#[derive(Debug, Clone)]
pub struct HierarchicalRouteResult {
    /// Selected group indices per token, shape `[n_tokens · top_k_group]`.
    pub group_indices: Vec<usize>,
    /// Renormalised group weights aligned with `group_indices`,
    /// shape `[n_tokens · top_k_group]`; each token's weights sum to `1`.
    pub group_weights: Vec<f32>,
    /// Selected **global** expert indices, shape
    /// `[n_tokens · top_k_group · top_k_expert]`; the inner axis is the expert
    /// slot, the middle axis the group slot.
    pub expert_indices: Vec<usize>,
    /// Combine weights aligned with `expert_indices` (`group_weight · expert_weight`),
    /// shape `[n_tokens · top_k_group · top_k_expert]`; each token's weights sum to `1`.
    pub combine_weights: Vec<f32>,
    /// Raw group-gate logits, shape `[n_tokens · n_groups]`.
    pub group_logits: Vec<f32>,
}

/// Output of a [`HierarchicalMoeLayer`] forward pass.
#[derive(Debug)]
pub struct HierarchicalOutput {
    /// Output hidden states, shape `[n_tokens · d_model]`.
    pub hidden: Vec<f32>,
    /// Group-level load-balancing auxiliary loss (non-negative).
    pub aux_loss: f32,
    /// Routing decisions for inspection / downstream losses.
    pub routing: HierarchicalRouteResult,
}

/// Two-level (group → expert) hierarchical sparse MoE layer.
pub struct HierarchicalMoeLayer {
    /// Group gate, row-major `[n_groups · d_model]`.
    group_gate: Vec<f32>,
    /// Per-group expert gates, row-major `[n_groups · experts_per_group · d_model]`.
    expert_gates: Vec<f32>,
    /// Flat bank of `n_groups · experts_per_group` experts.
    experts: ExpertBank,
    /// Model dimension.
    pub d_model: usize,
    /// Number of groups.
    pub n_groups: usize,
    /// Experts per group.
    pub experts_per_group: usize,
    /// Groups activated per token.
    pub top_k_group: usize,
    /// Experts activated per selected group.
    pub top_k_expert: usize,
}

impl HierarchicalMoeLayer {
    /// Build a new hierarchical MoE layer with random gates and experts.
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero `d_model` / `ffn_dim` / `n_groups` /
    /// `experts_per_group`, a `top_k_group` outside `1 ..= n_groups`, or a
    /// `top_k_expert` outside `1 ..= experts_per_group`.
    pub fn new(cfg: HierarchicalConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.d_model == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.d_model });
        }
        if cfg.ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: cfg.ffn_dim });
        }
        if cfg.n_groups == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_groups,
            });
        }
        if cfg.experts_per_group == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.experts_per_group,
            });
        }
        if cfg.top_k_group == 0 || cfg.top_k_group > cfg.n_groups {
            return Err(MoeError::InvalidTopK {
                k: cfg.top_k_group,
                n_experts: cfg.n_groups,
            });
        }
        if cfg.top_k_expert == 0 || cfg.top_k_expert > cfg.experts_per_group {
            return Err(MoeError::InvalidTopK {
                k: cfg.top_k_expert,
                n_experts: cfg.experts_per_group,
            });
        }

        let mut group_gate = vec![0.0_f32; cfg.n_groups * cfg.d_model];
        rng.fill_normal_scaled(&mut group_gate, 0.01);
        let mut expert_gates = vec![0.0_f32; cfg.n_groups * cfg.experts_per_group * cfg.d_model];
        rng.fill_normal_scaled(&mut expert_gates, 0.01);

        let total_experts = cfg.n_groups * cfg.experts_per_group;
        let experts =
            ExpertBank::new(total_experts, cfg.d_model, cfg.ffn_dim, cfg.activation, rng)?;

        Ok(Self {
            group_gate,
            expert_gates,
            experts,
            d_model: cfg.d_model,
            n_groups: cfg.n_groups,
            experts_per_group: cfg.experts_per_group,
            top_k_group: cfg.top_k_group,
            top_k_expert: cfg.top_k_expert,
        })
    }

    /// Total number of experts (`n_groups · experts_per_group`).
    #[must_use]
    pub fn total_experts(&self) -> usize {
        self.n_groups * self.experts_per_group
    }

    /// Run the hierarchical forward pass.
    ///
    /// # Arguments
    /// * `tokens` — input activations, row-major `[n_tokens · d_model]`.
    /// * `n_tokens` — number of tokens.
    /// * `d_model` — feature dimension, validated against the layer's `d_model`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input, a `d_model` mismatch, or a token
    /// buffer that is not `n_tokens · d_model` long.
    pub fn forward(
        &self,
        tokens: &[f32],
        n_tokens: usize,
        d_model: usize,
    ) -> MoeResult<HierarchicalOutput> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        if d_model != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: d_model,
            });
        }
        let expected = n_tokens * self.d_model;
        if tokens.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: tokens.len(),
            });
        }

        let kg = self.top_k_group;
        let ke = self.top_k_expert;
        let epg = self.experts_per_group;
        let pairs = kg * ke;

        let mut group_indices = vec![0_usize; n_tokens * kg];
        let mut group_weights = vec![0.0_f32; n_tokens * kg];
        let mut expert_indices = vec![0_usize; n_tokens * pairs];
        let mut combine_weights = vec![0.0_f32; n_tokens * pairs];
        let mut group_logits = vec![0.0_f32; n_tokens * self.n_groups];
        let mut hidden = vec![0.0_f32; n_tokens * self.d_model];

        for tok in 0..n_tokens {
            let x_tok = &tokens[tok * self.d_model..(tok + 1) * self.d_model];

            // Level 1: group gate.
            let g_logits = matvec(&self.group_gate, x_tok, self.d_model)?;
            group_logits[tok * self.n_groups..(tok + 1) * self.n_groups].copy_from_slice(&g_logits);
            let g_probs = stable_softmax(&g_logits);
            let (g_vals, g_idx) = topk(&g_probs, kg)?;
            let g_denom: f32 = g_vals.iter().sum::<f32>() + 1e-12;

            for (a, (&gi, &gp)) in g_idx.iter().zip(g_vals.iter()).enumerate() {
                let gw = gp / g_denom;
                group_indices[tok * kg + a] = gi;
                group_weights[tok * kg + a] = gw;

                // Level 2: per-group expert gate.
                let gate_slice =
                    &self.expert_gates[gi * epg * self.d_model..(gi + 1) * epg * self.d_model];
                let e_logits = matvec(gate_slice, x_tok, self.d_model)?;
                let e_probs = stable_softmax(&e_logits);
                let (e_vals, e_idx) = topk(&e_probs, ke)?;
                let e_denom: f32 = e_vals.iter().sum::<f32>() + 1e-12;

                for (b, (&le, &ep)) in e_idx.iter().zip(e_vals.iter()).enumerate() {
                    let ew = ep / e_denom;
                    let global = gi * epg + le;
                    let combine = gw * ew;
                    let slot = tok * pairs + a * ke + b;
                    expert_indices[slot] = global;
                    combine_weights[slot] = combine;

                    let e_out = self.experts.forward_expert(global, x_tok, 1)?;
                    let out_slice = &mut hidden[tok * self.d_model..(tok + 1) * self.d_model];
                    for (acc, &v) in out_slice.iter_mut().zip(e_out.iter()) {
                        *acc += combine * v;
                    }
                }
            }
        }

        // Group-level load balance (encourages even use of the groups).
        let aux_loss = topk_balance_loss(&group_logits, &group_indices, n_tokens, self.n_groups)?;

        Ok(HierarchicalOutput {
            hidden,
            aux_loss,
            routing: HierarchicalRouteResult {
                group_indices,
                group_weights,
                expert_indices,
                combine_weights,
                group_logits,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn layer(n_groups: usize, epg: usize, kg: usize, ke: usize) -> HierarchicalMoeLayer {
        let mut rng = LcgRng::new(2024);
        let cfg = HierarchicalConfig {
            d_model: 8,
            ffn_dim: 16,
            n_groups,
            experts_per_group: epg,
            top_k_group: kg,
            top_k_expert: ke,
            activation: ExpertActivation::Gelu,
        };
        HierarchicalMoeLayer::new(cfg, &mut rng).expect("new should succeed")
    }

    fn inputs(n_tokens: usize, d: usize) -> Vec<f32> {
        (0..n_tokens * d)
            .map(|i| ((i as f32) * 0.11).sin() * 1.5)
            .collect()
    }

    /// (a) Group weights and combine weights each sum to 1 per token.
    #[test]
    fn weights_sum_to_one() {
        let lyr = layer(3, 3, 2, 2);
        let n_tokens = 5;
        let x = inputs(n_tokens, 8);
        let out = lyr
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        let kg = 2;
        let pairs = 2 * 2;
        for tok in 0..n_tokens {
            let gsum: f32 = out.routing.group_weights[tok * kg..(tok + 1) * kg]
                .iter()
                .sum();
            assert!((gsum - 1.0).abs() < 1e-4, "token {tok} group sum {gsum}");
            let csum: f32 = out.routing.combine_weights[tok * pairs..(tok + 1) * pairs]
                .iter()
                .sum();
            assert!((csum - 1.0).abs() < 1e-4, "token {tok} combine sum {csum}");
        }
    }

    /// (b) Global expert index equals group · experts_per_group + local and is in range.
    #[test]
    fn global_index_mapping() {
        let n_groups = 3;
        let epg = 4;
        let kg = 2;
        let ke = 2;
        let lyr = layer(n_groups, epg, kg, ke);
        let n_tokens = 4;
        let x = inputs(n_tokens, 8);
        let out = lyr
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        let total = n_groups * epg;
        for tok in 0..n_tokens {
            for a in 0..kg {
                let group = out.routing.group_indices[tok * kg + a];
                for b in 0..ke {
                    let global = out.routing.expert_indices[tok * kg * ke + a * ke + b];
                    assert!(global < total, "global {global} out of range");
                    // The expert must live in the selected group.
                    assert_eq!(global / epg, group, "expert {global} not in group {group}");
                }
            }
        }
    }

    /// (c) Factorisation: combine weights of one group slot sum to that group's weight.
    #[test]
    fn combine_factorises_into_group_times_expert() {
        let kg = 2;
        let ke = 2;
        let lyr = layer(3, 3, kg, ke);
        let n_tokens = 4;
        let x = inputs(n_tokens, 8);
        let out = lyr
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        for tok in 0..n_tokens {
            for a in 0..kg {
                let group_w = out.routing.group_weights[tok * kg + a];
                let pair_sum: f32 = (0..ke)
                    .map(|b| out.routing.combine_weights[tok * kg * ke + a * ke + b])
                    .sum();
                assert!(
                    (pair_sum - group_w).abs() < 1e-5,
                    "token {tok} group {a}: Σ combine {pair_sum} != group_w {group_w}"
                );
            }
        }
    }

    /// (d) Output shape is correct, values finite, aux loss non-negative.
    #[test]
    fn output_shape_finite_and_aux_nonneg() {
        let lyr = layer(4, 2, 2, 1);
        let n_tokens = 6;
        let x = inputs(n_tokens, 8);
        let out = lyr
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        assert_eq!(out.hidden.len(), n_tokens * 8);
        assert!(out.hidden.iter().all(|v| v.is_finite()));
        assert!(out.aux_loss.is_finite() && out.aux_loss >= 0.0);
        assert_eq!(lyr.total_experts(), 8);
    }

    /// (e) A single group degenerates to a flat MoE over its experts.
    #[test]
    fn single_group_degenerates() {
        let lyr = layer(1, 4, 1, 2);
        let n_tokens = 4;
        let x = inputs(n_tokens, 8);
        let out = lyr
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        assert_eq!(out.hidden.len(), n_tokens * 8);
        assert!(out.hidden.iter().all(|v| v.is_finite()));
        // The only group always has weight 1.
        for &gw in &out.routing.group_weights {
            assert!((gw - 1.0).abs() < 1e-6, "single group weight {gw} != 1");
        }
        // Combine weights collapse to the within-group expert weights (sum to 1).
        for tok in 0..n_tokens {
            let csum: f32 = out.routing.combine_weights[tok * 2..tok * 2 + 2]
                .iter()
                .sum();
            assert!((csum - 1.0).abs() < 1e-4, "token {tok} combine sum {csum}");
        }
    }

    /// (f) Shape / dimension mismatches and invalid configs error.
    #[test]
    fn shape_and_config_errors() {
        let lyr = layer(3, 3, 2, 2);
        assert!(matches!(
            lyr.forward(&[0.0_f32; 4 * 8], 4, 7),
            Err(MoeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            lyr.forward(&[0.0_f32; 4 * 8 + 2], 4, 8),
            Err(MoeError::DimensionMismatch { .. })
        ));
        assert!(matches!(lyr.forward(&[], 0, 8), Err(MoeError::EmptyInput)));

        let mut rng = LcgRng::new(9);
        let bad_group_k = HierarchicalConfig {
            d_model: 8,
            ffn_dim: 16,
            n_groups: 3,
            experts_per_group: 3,
            top_k_group: 4,
            top_k_expert: 1,
            activation: ExpertActivation::Relu,
        };
        assert!(matches!(
            HierarchicalMoeLayer::new(bad_group_k, &mut rng),
            Err(MoeError::InvalidTopK { .. })
        ));
        let bad_expert_k = HierarchicalConfig {
            d_model: 8,
            ffn_dim: 16,
            n_groups: 3,
            experts_per_group: 3,
            top_k_group: 1,
            top_k_expert: 5,
            activation: ExpertActivation::Relu,
        };
        assert!(matches!(
            HierarchicalMoeLayer::new(bad_expert_k, &mut rng),
            Err(MoeError::InvalidTopK { .. })
        ));
    }
}
