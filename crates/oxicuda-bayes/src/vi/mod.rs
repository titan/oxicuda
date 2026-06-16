//! Automatic-differentiation variational inference (ADVI).
//!
//! Provides a mean-field Gaussian variational family optimised in an
//! unconstrained latent space by stochastic gradient ascent on the
//! reparameterised evidence lower bound (Kucukelbir et al., 2017).

pub mod advi;

pub use advi::{Advi, AdviConfig, AdviModel, AdviResult, Transform};
