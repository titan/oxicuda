//! Finite-difference methods (FDM) for Poisson, heat, wave, advection PDEs.

pub mod advection_1d;
pub mod burgers_1d;
pub mod crank_nicolson;
pub mod crank_nicolson_2d;
pub mod heat_1d;
pub mod heat_2d;
pub mod mol;
pub mod navier_stokes_1d;
pub mod poisson_1d;
pub mod poisson_2d;
pub mod poisson_3d;
pub mod wave_1d;
pub mod wave_2d;

pub use crank_nicolson::CrankNicolson;
pub use crank_nicolson_2d::CrankNicolson2d;
