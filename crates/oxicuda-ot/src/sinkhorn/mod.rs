//! Entropic Optimal Transport — Sinkhorn-Knopp algorithm and friends.

/// Anchor-based partial OT (Chapel et al. 2020).
pub mod anchor_partial;
/// Conditional gradient (Frank-Wolfe) method for non-entropic (LP) OT.
pub mod cg_wasserstein;
/// Debiased Sinkhorn divergence with shared dual structure and weight gradients (Feydy 2020).
pub mod debiased_divergence;
/// Debiased Sinkhorn divergence.
pub mod divergence;
/// Sinkhorn with Exponential Moving Average on dual potentials.
pub mod ema_sinkhorn;
/// Greenkhorn (greedy Sinkhorn) algorithm.
pub mod greenkhorn;
/// Low-level log-stabilised single Sinkhorn iteration.
pub mod log_sinkhorn;
/// Low-rank Sinkhorn factorisation (Scetbon & Cuturi 2020).
pub mod low_rank;
/// Sinkhorn-Knopp with momentum / Nesterov / Anderson acceleration.
pub mod momentum_sinkhorn;
/// Screened Sinkhorn — sparsity-inducing optimal transport.
pub mod screened;
/// Log-domain Sinkhorn-Knopp algorithm.
pub mod sinkhorn;

pub use anchor_partial::{
    AnchorPartialConfig, AnchorPartialFit, anchor_partial_ot, anchor_partial_plan,
    anchor_partial_transport_cost,
};
pub use cg_wasserstein::{
    CgWassConfig, CgWassFit, cg_dual_gap, cg_marginal_violation, cg_transport_cost, cg_wasserstein,
};
pub use debiased_divergence::{
    DebiasedDivergenceConfig, DebiasedDivergenceResult, debiased_sinkhorn_divergence,
};
pub use ema_sinkhorn::{
    EmaSinkhornConfig, EmaSinkhornFit, ema_marginal_violation, ema_sinkhorn, ema_transport_cost,
    ema_transport_plan,
};
pub use low_rank::{
    LowRankConfig, LowRankFit, low_rank_dense, low_rank_marginals, low_rank_sinkhorn,
    low_rank_transport_cost,
};
pub use momentum_sinkhorn::{
    MomentumScheme, MomentumSinkhornConfig, MomentumSinkhornResult, momentum_sinkhorn,
};
pub use screened::{
    ScreenedConfig, ScreenedFit, screened_marginal_violation, screened_sinkhorn, screened_sparsity,
    screened_transport_cost, screened_transport_cost_with_reg,
};

/// Numerically stabilised Sinkhorn with periodic potential absorption (Schmitzer 2019).
pub mod stabilised_sinkhorn;
pub use stabilised_sinkhorn::{
    StabilisedSinkhornConfig, StabilisedSinkhornResult, marginal_violation_row, sq_euclidean_cost,
    stabilised_sinkhorn,
};

/// Epsilon-scaling (deterministic-annealing) Sinkhorn for the `ε → 0` regime (Schmitzer 2019).
pub mod epsilon_scaling;
pub use epsilon_scaling::{
    EpsilonScalingConfig, EpsilonScalingResult, StabilityRecord, epsilon_scaling_sinkhorn,
    stability_sweep,
};
