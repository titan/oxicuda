//! Neural network modules for NeRF.
//!
//! - `human_nerf`: HumanNeRF / InstantAvatar skeleton-driven canonical mapping
//! - `nerf_mlp`: Full 8-layer NeRF MLP with skip connection
//! - `nerf_w`: NeRF in the Wild — per-image appearance embeddings + β-uncertainty NLL
//! - `tiny_nerf`: Compact 4-layer NeRF for tests

pub mod human_nerf;
pub mod nerf_mlp;
pub mod nerf_w;
pub mod tiny_nerf;

pub use human_nerf::{HumanNerf, HumanNerfConfig, NO_PARENT, Rigid, Skeleton};
pub use nerf_w::{BetaHead, NerfWConfig, NerfWEmbeddings, concat_features, nerf_w_nll};
