//! PINN loss functions: residual, boundary, initial, adaptive weighting,
//! causal training, and self-adaptive per-point weighting.

pub mod boundary;
pub mod causal;
pub mod conservative;
pub mod deep_ritz;
pub mod initial;
pub mod residual;
pub mod sa_pinn;
pub mod weighting;

pub use causal::{CausalPinnConfig, CausalPinnLoss};
pub use conservative::{ConservativeConfig, ConservativeLoss, SubdomainBox};
pub use deep_ritz::{DeepRitz, DeepRitzBlock, DeepRitzConfig, DeepRitzEnergy, DeepRitzNet};
pub use sa_pinn::{SaPinn, SaPinnConfig};
