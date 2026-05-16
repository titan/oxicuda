//! Local-neighbourhood manifold-learning methods.
//!
//! - [`lle`] Locally Linear Embedding (Roweis & Saul, 2000).
//! - [`mlle`] Modified LLE (Zhang & Wang, 2007).
//! - [`isomap`] Isomap geodesic embedding (Tenenbaum, 2000).
//! - [`laplacian_eigenmaps`] Laplacian Eigenmaps (Belkin & Niyogi, 2003).

pub mod isomap;
pub mod laplacian_eigenmaps;
pub mod lle;
pub mod mlle;

pub use isomap::{IsomapResult, isomap_fit};
pub use laplacian_eigenmaps::{LapEigResult, laplacian_eigenmaps_fit};
pub use lle::{LleResult, lle_fit};
pub use mlle::{MlleResult, mlle_fit};
