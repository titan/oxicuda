//! Variational inference primitives.
//!
//! Provides ELBO computation, reparameterization tricks, mean-field distributions,
//! normalizing flows for flexible variational posteriors, Real NVP affine coupling
//! flows, and Stein Variational Gradient Descent.

pub mod elbo;
pub mod flows;
pub mod hmc;
pub mod iaf_flow;
pub mod mean_field;
pub mod nvae;
pub mod real_nvp;
pub mod reparam;
pub mod svgd;
pub mod vcl;

pub use hmc::{Hmc, HmcConfig, HmcResult, Nuts, NutsConfig, NutsResult};
pub use iaf_flow::{IafFlow, IafStep, MadeNet, standard_normal_log_prob};
pub use nvae::{NVae, NVaeConfig, NVaeOutput, apply_free_bits, kl_gaussian_diag};
pub use real_nvp::{CouplingLayer, RealNvp};
pub use svgd::{Svgd, SvgdConfig, SvgdResult};
pub use vcl::{VclConfig, VclState};
