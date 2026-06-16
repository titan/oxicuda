//! Normalizing flows organised as standalone, invertible transforms.
//!
//! This module hosts the **Masked Autoregressive Flow** (MAF) of Papamakarios,
//! Pavlakou & Murray (2017). It complements the encoder-conditioned inverse
//! autoregressive flow in [`crate::variational::iaf_flow`] and the affine
//! coupling flow in [`crate::variational::real_nvp`]: MAF is fast in the
//! density-evaluation (data → noise) direction and sequential in the sampling
//! (noise → data) direction — the dual trade-off to IAF.

pub mod maf;

pub use maf::{MafFlow, MafLayer, standard_normal_log_prob_vec};
