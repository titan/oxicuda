//! PINN training variants: alternative loss formulations and residual
//! augmentations layered on top of the core PDE-residual machinery.

pub mod gpinn;

pub use gpinn::{GPinnConfig, GPinnLoss, GPinnLossTerms};
