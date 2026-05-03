//! Variational inference primitives.
//!
//! Provides ELBO computation, reparameterization tricks, mean-field distributions,
//! and normalizing flows for flexible variational posteriors.

pub mod elbo;
pub mod flows;
pub mod mean_field;
pub mod reparam;
