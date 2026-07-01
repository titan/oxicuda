//! # oxicuda-moe
//!
//! Mixture of Experts (MoE) primitives for OxiCUDA.
//!
//! Implements Switch Transformer, Top-K routing, Expert Choice, Soft MoE,
//! Gumbel-softmax stochastic routing, expert-parallel all-to-all dispatch,
//! MegaBlocks block-sparse dispatch, load balancing, and associated PTX kernels
//! for GPU execution.

pub mod error;
pub mod expert;
pub mod handle;
pub mod layer;
pub mod loss;
pub mod metrics;
pub mod moe;
pub mod ptx_kernels;
pub mod routing;

/// Convenience re-exports for common MoE types and functions.
pub mod prelude {
    pub use crate::error::{MoeError, MoeResult};
    pub use crate::expert::bank::{ExpertBank, SwiGluBank};
    pub use crate::expert::block_sparse::{
        BlockSparseDispatcher, BlockSparseLayout, PAD_ROW, build_block_sparse_layout,
        gather_tokens, scatter_tokens,
    };
    pub use crate::expert::ffn::{ExpertActivation, ExpertFfn, SwiGluExpert};
    pub use crate::expert::prune_merge::{
        CompressionResult, merge_experts, prune_experts, remap_indices,
    };
    pub use crate::handle::{LcgRng, MoeHandle, SmVersion};
    pub use crate::layer::moe_layer::{MoeLayer, MoeLayerConfig, MoeLayerOutput};
    pub use crate::loss::entropy::routing_entropy;
    pub use crate::loss::load_balance::{LoadStats, compute_load_stats, load_balance_loss};
    pub use crate::loss::router_z::router_z_loss;
    pub use crate::metrics::utilization::{ExpertUtilization, compute_utilization};
    pub use crate::moe::hierarchical::{
        HierarchicalConfig, HierarchicalMoeLayer, HierarchicalOutput, HierarchicalRouteResult,
    };
    pub use crate::moe::lora_moe::{LoraExpert, LoraMoe, LoraMoeConfig, LoraMoeOutput};
    pub use crate::moe::mixtral::{
        MixtralConfig, MixtralMoeLayer, MixtralOutput, MixtralRoutingInfo,
        mixtral_load_balance_loss,
    };
    pub use crate::moe::upcycle::{
        DenseFfnCheckpoint, UpcycleConfig, upcycle_expert_bank, upcycle_moe_layer,
    };
    pub use crate::ptx_kernels::{
        expert_combine_ptx, expert_dispatch_ptx, expert_ffn_ptx, f32_hex, load_balance_loss_ptx,
        router_z_loss_ptx, soft_moe_dispatch_ptx, top_k_gate_ptx,
    };
    pub use crate::routing::conditional::{
        ConditionalConfig, ConditionalRouter, ConditionalRouting,
    };
    pub use crate::routing::diff_capacity::{DiffCapacityConfig, DifferentiableCapacity};
    pub use crate::routing::expert_choice::{
        ExpertChoiceConfig, ExpertChoiceResult, expert_choice_combine, expert_choice_route,
    };
    pub use crate::routing::expert_dropout::{ExpertDropout, ExpertDropoutConfig};
    pub use crate::routing::expert_parallel::{
        ExpertParallelConfig, ExpertParallelPlan, TokenPlacement, build_dispatch_plan,
        combine_all_to_all, dispatch_all_to_all,
    };
    pub use crate::routing::gumbel::{
        GumbelConfig, GumbelRouteResult, GumbelRouter, gumbel_softmax,
    };
    pub use crate::routing::hash::{HashRouter, HashRoutingConfig};
    pub use crate::routing::layer_conditional::{
        LayerConditionalConfig, LayerConditionalRouter, LayerRouteResult, RouterSharing,
    };
    pub use crate::routing::mamba_route::{MambaRouteConfig, MambaRouteResult, MambaRouter};
    pub use crate::routing::multi_gate::{MultiGateConfig, MultiGateRouter};
    pub use crate::routing::noisy_top_k::{
        NoisyTopKConfig, NoisyTopKResult, NoisyTopKRouter, softplus,
    };
    pub use crate::routing::shared_expert::{
        SharedExpertConfig, SharedExpertResult, shared_expert_combine, shared_expert_route,
        total_experts,
    };
    pub use crate::routing::sinkhorn_route::{
        SinkhornRouteConfig, SinkhornRouteResult, marginal_deviation, sinkhorn_route,
    };
    pub use crate::routing::soft_moe::{SoftMoeConfig, SoftMoeRouter};
    pub use crate::routing::st_moe::{
        StMoeConfig, StMoeLayer, StMoeOutput, StMoeRouting, st_load_balance_loss, st_router_z_loss,
    };
    pub use crate::routing::switch::{
        SwitchConfig, SwitchDispatch, switch_combine, switch_dispatch,
    };
    pub use crate::routing::top_k::{TopKConfig, TopKResult, TopKRouter, topk};
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

#[cfg(test)]
mod e2e_tests {
    use super::prelude::*;

    /// Test 1: Top-k scores sum to 1.0 for each token.
    #[test]
    fn e2e_top_k_gate_sum_to_one() {
        let mut rng = LcgRng::new(42);
        let cfg = TopKConfig {
            k: 2,
            n_experts: 8,
            input_dim: 32,
            noise_std: 0.0,
        };
        let router = TopKRouter::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 16;
        let x = vec![0.5_f32; n_tokens * 32];
        let result = router.route(&x, n_tokens).expect("route should succeed");
        for tok in 0..n_tokens {
            let score_sum: f32 = result.scores[tok * 2..tok * 2 + 2].iter().sum();
            assert!(
                (score_sum - 1.0).abs() < 1e-4,
                "token {tok} scores sum to {score_sum}, expected 1.0"
            );
        }
    }

    /// Test 2: All returned indices are < n_experts.
    #[test]
    fn e2e_top_k_indices_valid() {
        let mut rng = LcgRng::new(7);
        let n_experts = 8_usize;
        let cfg = TopKConfig {
            k: 2,
            n_experts,
            input_dim: 16,
            noise_std: 0.0,
        };
        let router = TopKRouter::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 32;
        let x = vec![0.3_f32; n_tokens * 16];
        let result = router.route(&x, n_tokens).expect("route should succeed");
        for &idx in &result.indices {
            assert!(idx < n_experts, "index {idx} >= n_experts {n_experts}");
        }
    }

    /// Test 3: No expert exceeds capacity tokens in Switch dispatch.
    #[test]
    fn e2e_switch_dispatch_capacity_respected() {
        let n_tokens = 64_usize;
        let n_experts = 8_usize;
        let cfg = SwitchConfig {
            n_experts,
            input_dim: 32,
            capacity_factor: 1.25,
            min_capacity: 1,
            drop_tokens: true,
        };
        // Round-robin assignment
        let gate_indices: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let dispatch =
            switch_dispatch(&gate_indices, n_tokens, &cfg).expect("switch_dispatch should succeed");
        // Count tokens per expert
        let mut counts = vec![0_usize; n_experts];
        for &assignment in &dispatch.expert_assignments {
            if assignment != usize::MAX {
                counts[assignment] += 1;
            }
        }
        for (exp_idx, &count) in counts.iter().enumerate() {
            assert!(
                count <= dispatch.capacity,
                "expert {exp_idx} received {count} tokens, capacity={cap}",
                cap = dispatch.capacity
            );
        }
    }

    /// Test 4: Overflow count is non-zero when capacity < n_tokens.
    #[test]
    fn e2e_switch_dispatch_overflow_counted() {
        // Send all tokens to a single expert with tight capacity
        let n_tokens = 16_usize;
        let n_experts = 4_usize;
        let cfg = SwitchConfig {
            n_experts,
            input_dim: 16,
            capacity_factor: 0.5, // capacity = ceil(16/4 * 0.5) = 2
            min_capacity: 1,
            drop_tokens: true,
        };
        // All tokens to expert 0 → many overflows
        let gate_indices = vec![0_usize; n_tokens];
        let dispatch =
            switch_dispatch(&gate_indices, n_tokens, &cfg).expect("switch_dispatch should succeed");
        assert!(
            dispatch.n_overflows > 0,
            "expected overflows with tight capacity, got 0"
        );
    }

    /// Test 5: ExpertFfn forward returns all-finite output.
    #[test]
    fn e2e_expert_ffn_output_finite() {
        let mut rng = LcgRng::new(0);
        let ffn = ExpertFfn::new(32, 128, ExpertActivation::Gelu, &mut rng);
        let x = vec![0.5_f32; 32];
        let output = ffn.forward(&x).expect("forward should succeed");
        assert!(
            output.iter().all(|v| v.is_finite()),
            "ExpertFfn output contains non-finite values"
        );
    }

    /// Test 6: ExpertFfn output has the same shape as input.
    #[test]
    fn e2e_expert_ffn_output_shape() {
        let input_dim = 64_usize;
        let mut rng = LcgRng::new(1);
        let ffn = ExpertFfn::new(input_dim, 256, ExpertActivation::Relu, &mut rng);
        let x = vec![1.0_f32; input_dim];
        let output = ffn.forward(&x).expect("forward should succeed");
        assert_eq!(
            output.len(),
            input_dim,
            "output shape {out} != input shape {input_dim}",
            out = output.len()
        );
    }

    /// Test 7: SwiGluExpert forward returns finite output.
    #[test]
    fn e2e_swiglu_expert_finite() {
        let mut rng = LcgRng::new(2);
        let expert = SwiGluExpert::new(32, 128, &mut rng);
        let x = vec![0.7_f32; 32];
        let output = expert.forward(&x).expect("forward should succeed");
        assert!(
            output.iter().all(|v| v.is_finite()),
            "SwiGluExpert output contains non-finite values"
        );
    }

    /// Test 8: Load balance loss is non-negative.
    #[test]
    fn e2e_load_balance_loss_range() {
        let n_tokens = 32_usize;
        let n_experts = 4_usize;
        let mut rng = LcgRng::new(3);
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        rng.fill_normal_scaled(&mut logits, 1.0);
        // Assignments: round-robin
        let assignments: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let loss = load_balance_loss(&logits, &assignments, n_tokens, n_experts)
            .expect("load_balance_loss should succeed");
        assert!(
            loss >= 0.0,
            "load balance loss must be non-negative, got {loss}"
        );
        assert!(
            loss.is_finite(),
            "load balance loss must be finite, got {loss}"
        );
    }

    /// Test 9: Router z-loss is >= 0 (it's a squared log).
    #[test]
    fn e2e_router_z_loss_nonneg() {
        let n_tokens = 16_usize;
        let n_experts = 8_usize;
        let mut rng = LcgRng::new(4);
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        rng.fill_normal_scaled(&mut logits, 2.0);
        let loss =
            router_z_loss(&logits, n_tokens, n_experts).expect("router_z_loss should succeed");
        assert!(loss >= 0.0, "z-loss must be >= 0, got {loss}");
        assert!(loss.is_finite(), "z-loss must be finite, got {loss}");
    }

    /// Test 10: Soft MoE dispatch weights sum to 1 per token.
    #[test]
    fn e2e_soft_moe_dispatch_weights_sum_to_one() {
        let mut rng = LcgRng::new(5);
        let cfg = SoftMoeConfig {
            n_experts: 4,
            n_slots_per_expert: 2,
            input_dim: 16,
        };
        let n_tokens = 8;
        let router = SoftMoeRouter::new(cfg.clone(), &mut rng).expect("value should be present");
        let x = vec![0.5_f32; n_tokens * cfg.input_dim];
        let dispatch = router
            .dispatch_weights(&x, n_tokens)
            .expect("dispatch_weights should succeed");
        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        for tok in 0..n_tokens {
            let row_sum: f32 = dispatch[tok * n_slots..(tok + 1) * n_slots].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "token {tok} dispatch weights sum to {row_sum}"
            );
        }
    }

    /// Test 11: MoeLayer forward returns [T * d_model] output.
    #[test]
    fn e2e_moe_layer_forward_shape() {
        let mut rng = LcgRng::new(6);
        let input_dim = 32_usize;
        let n_tokens = 16_usize;
        let cfg = MoeLayerConfig {
            input_dim,
            ffn_dim: 128,
            n_experts: 4,
            top_k: 1,
            capacity_factor: 1.5,
            load_balance_coef: 0.01,
            router_z_loss_coef: 0.001,
            activation: ExpertActivation::Gelu,
        };
        let layer = MoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.3_f32; n_tokens * input_dim];
        let output = layer.forward(&x, n_tokens).expect("forward should succeed");
        assert_eq!(
            output.hidden.len(),
            n_tokens * input_dim,
            "output shape mismatch"
        );
        assert!(output.aux_loss.is_finite(), "aux_loss must be finite");
    }

    /// Test 12: All 7 kernels × 6 SM versions produce valid PTX.
    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("top_k_gate_kernel", top_k_gate_ptx),
            ("expert_dispatch_kernel", expert_dispatch_ptx),
            ("expert_ffn_kernel", expert_ffn_ptx),
            ("expert_combine_kernel", expert_combine_ptx),
            ("load_balance_loss_kernel", load_balance_loss_ptx),
            ("router_z_loss_kernel", router_z_loss_ptx),
            ("soft_moe_dispatch_kernel", soft_moe_dispatch_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing sm target"
                );
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} sm={sm} missing .version"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "PTX for {kernel_name} sm={sm} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} sm={sm} missing kernel name"
                );
            }
        }
        // Extra check on f32_hex
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
