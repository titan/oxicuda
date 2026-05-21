//! Variational inference primitives.
//!
//! Provides ELBO computation, reparameterization tricks, mean-field distributions,
//! normalizing flows for flexible variational posteriors, Real NVP affine coupling
//! flows, and Stein Variational Gradient Descent.

pub mod elbo;
pub mod flows;
pub mod hmc;
pub mod mean_field;
pub mod real_nvp;
pub mod reparam;
pub mod svgd;
pub mod vcl;

pub use hmc::{Hmc, HmcConfig, HmcResult, Nuts, NutsConfig, NutsResult};
pub use real_nvp::{CouplingLayer, RealNvp};
pub use svgd::{Svgd, SvgdConfig, SvgdResult};
pub use vcl::{VclConfig, VclState};
