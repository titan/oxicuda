//! Proximal-gradient methods.

pub mod accelerated;
pub mod bundle;
pub mod douglas_rachford;
pub mod fista;
pub mod inexact_prox;
pub mod peaceman_rachford;
pub mod prox_gradient;
pub mod proximal_newton;

pub use accelerated::accelerated_prox_gradient;
pub use bundle::{BundleConfig, BundleResult, ProximalBundle};
pub use douglas_rachford::douglas_rachford;
pub use fista::fista;
pub use inexact_prox::{InexactProx, InexactProxConfig};
pub use peaceman_rachford::peaceman_rachford;
pub use prox_gradient::proximal_gradient;
pub use proximal_newton::{ProximalNewtonConfig, ProximalNewtonResult, proximal_newton};
