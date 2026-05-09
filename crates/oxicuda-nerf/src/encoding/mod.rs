//! Encoding modules for NeRF input features.
//!
//! - `positional`: NeRF positional encoding γ(p)
//! - `hash_grid`: Instant-NGP multi-resolution hash grid
//! - `integrated_pe`: Mip-NeRF integrated positional encoding (IPE)

pub mod hash_grid;
pub mod integrated_pe;
pub mod positional;
