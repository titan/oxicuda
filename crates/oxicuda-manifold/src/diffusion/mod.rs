//! Diffusion Maps (Coifman & Lafon, 2006) and PHATE (Moon et al. 2019).

pub mod diffusion_map;
pub mod phate;

pub use diffusion_map::{DiffusionMapResult, diffusion_map_fit};
pub use phate::{PhateConfig, PhateResult, phate_fit};
