//! Time-stepping schemes for ODE / spatially-discretised PDE systems.

pub mod backward_euler;
pub mod bdf2;
pub mod crank_nicolson;
pub mod dirk_imex;
pub mod exponential;
pub mod forward_euler;
pub mod imex;
pub mod rk4;
pub mod rk_implicit;
pub mod sdirk;
pub mod symplectic;

pub use backward_euler::backward_euler_solve_linear;
pub use bdf2::bdf2_step_linear;
pub use crank_nicolson::crank_nicolson_step_linear;
pub use dirk_imex::ImexArk;
pub use exponential::{
    etd_rk4_integrate, etd_rk4_step, exp_diag, lawson_euler_integrate, lawson_euler_step,
    lawson_rk4_integrate, lawson_rk4_step,
};
pub use forward_euler::forward_euler_step;
pub use imex::imex_step;
pub use rk_implicit::{ImplicitRk, ImplicitRkMethod};
pub use rk4::rk4_step;
pub use sdirk::{SdirkConfig, sdirk2, sdirk2_step, sdirk3, sdirk3_step};
pub use symplectic::{forest_ruth, forest_ruth_step, velocity_verlet, velocity_verlet_step};
