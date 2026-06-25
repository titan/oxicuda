//! Encoding modules for NeRF input features.
//!
//! - `positional`: NeRF positional encoding γ(p)
//! - `hash_grid`: Instant-NGP multi-resolution hash grid
//! - `hash_grid_grad`: Trainable hash grid (forward cache + analytic backward)
//! - `integrated_pe`: Mip-NeRF integrated positional encoding (IPE)
//! - `spherical_harmonics`: Real SH basis up to degree L = 4

pub mod hash_grid;
pub mod hash_grid_grad;
pub mod integrated_pe;
pub mod positional;
pub mod spherical_harmonics;

pub use hash_grid_grad::{GridCache, TrainableHashGrid};
pub use spherical_harmonics::{ShConfig, ShEncoder, evaluate_sh};
