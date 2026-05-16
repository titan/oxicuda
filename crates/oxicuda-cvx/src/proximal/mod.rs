//! Proximal-gradient methods.

pub mod accelerated;
pub mod douglas_rachford;
pub mod fista;
pub mod prox_gradient;

pub use accelerated::accelerated_prox_gradient;
pub use douglas_rachford::douglas_rachford;
pub use fista::fista;
pub use prox_gradient::proximal_gradient;
