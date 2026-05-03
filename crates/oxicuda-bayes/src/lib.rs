//! `oxicuda-bayes` — Bayesian deep learning primitives for OxiCUDA.
//!
//! Pure-Rust implementation of variational inference and Bayesian neural network
//! building blocks suitable for CPU simulation and PTX kernel generation for GPU
//! execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-bayes
//! ├── layers/         — BayesLinear, BayesConv2d, Flipout layers
//! ├── variational/    — ELBO, normalizing flows, mean-field, reparameterization
//! ├── error           — BayesError / BayesResult
//! ├── handle          — BayesHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod error;
pub mod handle;
pub mod layers;
pub mod ptx_kernels;
pub mod variational;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common Bayesian deep learning types.
pub mod prelude {
    pub use crate::error::{BayesError, BayesResult};
    pub use crate::handle::{BayesHandle, LcgRng, SmVersion};
    pub use crate::layers::bayes_conv::BayesConv2d;
    pub use crate::layers::bayes_linear::{BayesLinear, softplus};
    pub use crate::layers::flipout::{FlipoutConv2d, FlipoutLinear};
    pub use crate::ptx_kernels::{
        ece_bucket_ptx, ensemble_aggregate_ptx, f32_hex, flipout_perturb_ptx, kl_gaussian_ptx,
        local_reparam_ptx, mc_dropout_mask_ptx, temp_scale_logits_ptx,
    };
    pub use crate::variational::elbo::{ElboConfig, elbo, iwae, kl_gaussian, kl_gaussian_vec};
    pub use crate::variational::flows::{PlanarFlow, RadialFlow};
    pub use crate::variational::mean_field::MeanFieldDist;
    pub use crate::variational::reparam::{
        gaussian_log_prob, gaussian_sample, laplacian_log_prob, laplacian_sample,
        log_prob_gaussian_vec, sample_gaussian_vec, straight_through,
    };
}
