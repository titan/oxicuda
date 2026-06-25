//! Automatic differentiation: dual numbers (forward mode) and tape (reverse mode).

pub mod dual;
pub mod multidim;
pub mod pde_residual;
pub mod tape;

pub use pde_residual::{
    HyperDual, burgers_residual_ad, heat_residual_ad, linear_2nd_order_residual_ad,
    poisson_residual_ad,
};
