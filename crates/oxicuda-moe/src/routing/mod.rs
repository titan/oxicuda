//! Routing strategies for Mixture of Experts.

pub mod base;
pub mod conditional;
pub mod diff_capacity;
pub mod expert_choice;
pub mod expert_dropout;
pub mod expert_parallel;
pub mod gumbel;
pub mod hash;
pub mod layer_conditional;
pub mod mamba_route;
pub mod multi_gate;
pub mod noisy_top_k;
pub mod shared_expert;
pub mod sinkhorn_route;
pub mod soft_moe;
pub mod st_moe;
pub mod stable_moe;
pub mod switch;
pub mod top_k;

pub use base::{BaseConfig, BaseResult, BaseRouter, row_softmax, sinkhorn_convergence};
pub use conditional::{ConditionalConfig, ConditionalRouter, ConditionalRouting};
pub use diff_capacity::{DiffCapacityConfig, DifferentiableCapacity};
pub use expert_dropout::{ExpertDropout, ExpertDropoutConfig};
pub use expert_parallel::{
    ExpertParallelConfig, ExpertParallelPlan, TokenPlacement, build_dispatch_plan,
    combine_all_to_all, dispatch_all_to_all,
};
pub use gumbel::{GumbelConfig, GumbelRouteResult, GumbelRouter, gumbel_softmax};
pub use hash::{HashRouter, HashRoutingConfig};
pub use layer_conditional::{
    LayerConditionalConfig, LayerConditionalRouter, LayerRouteResult, RouterSharing,
};
pub use mamba_route::{MambaRouteConfig, MambaRouteResult, MambaRouter};
pub use multi_gate::{MultiGateConfig, MultiGateRouter};
pub use noisy_top_k::{NoisyTopKConfig, NoisyTopKResult, NoisyTopKRouter, softplus};
pub use shared_expert::{
    SharedExpertConfig, SharedExpertResult, shared_expert_combine, shared_expert_route,
    total_experts,
};
pub use sinkhorn_route::{
    SinkhornRouteConfig, SinkhornRouteResult, marginal_deviation, sinkhorn_route,
};
pub use st_moe::{
    StMoeConfig, StMoeLayer, StMoeOutput, StMoeRouting, st_load_balance_loss, st_router_z_loss,
};
pub use stable_moe::{
    StableMoeConfig, StableMoeGating, StableMoeResult, StableMoeRouter, load_balance_loss, sigmoid,
    z_loss,
};
