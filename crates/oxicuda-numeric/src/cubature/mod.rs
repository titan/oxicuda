//! Multi-dimensional cubature.

pub mod adaptive_cubature;
pub mod genz_malik;
pub mod monte_carlo;
pub mod quasi_monte_carlo_sobol;
pub mod tensor_product_gauss;

pub use adaptive_cubature::{AdaptiveCubature, CubatureResult};
