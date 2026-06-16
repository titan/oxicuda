//! Gromov-Wasserstein and Fused-Gromov-Wasserstein OT for distributions on
//! possibly different metric spaces.
//!
//! Where ordinary OT requires a single ground cost between source and target
//! supports, Gromov-Wasserstein lifts that constraint by comparing
//! intra-domain distance matrices `C^1` and `C^2`. Fused-GW interpolates
//! between intra-domain GW and an inter-domain Wasserstein term using a mixing
//! parameter `α ∈ [0, 1]`.

/// Batched entropic Gromov-Wasserstein for k-way hyperparameter sweeps and ensembling.
pub mod batched_gw;
/// Entropic GW with linear-memory column-sketching approximation (Peyré et al. 2016).
pub mod entropic_gw_fast;
/// Fused Gromov-Wasserstein combining intra-domain GW and inter-domain Wasserstein.
pub mod fused;
/// Entropic Gromov-Wasserstein for distributions on possibly different metric spaces.
pub mod gromov_wasserstein;
/// GW-Wasserstein hybrid for graph matching (Titouan et al. 2019).
pub mod gw_graph_matching;

pub use batched_gw::{BatchedGwConfig, BatchedGwResult, batched_gromov_wasserstein};
pub use entropic_gw_fast::{
    EntropicGwFastConfig, EntropicGwFastFit, entropic_gw_fast, gw_cost_matrix, gw_distance,
};
pub use gw_graph_matching::{GwGraphConfig, GwGraphResult, gw_frobenius_cost, gw_graph_matching};

/// Bregman-projected Gromov-Wasserstein via mirror descent (Xu et al. 2019).
pub mod bregman_gw;
pub use bregman_gw::{
    BregmanGwConfig, BregmanGwResult, bregman_gw, bregman_gw_distance, gw_linear_cost, gw_objective,
};
