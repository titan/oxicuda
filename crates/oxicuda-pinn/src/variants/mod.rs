//! PINN training variants: alternative loss formulations and residual
//! augmentations layered on top of the core PDE-residual machinery.

pub mod gpinn;
pub mod pde_discovery;

pub use gpinn::{GPinnConfig, GPinnLoss, GPinnLossTerms};
pub use pde_discovery::{
    LibraryConfig, PdeNetCell, SindyConfig, SindyModel, build_library, fit_sindy,
};
