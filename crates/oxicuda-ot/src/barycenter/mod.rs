//! Wasserstein barycenters.
//!
//! A barycenter `B = arg min_μ Σ_k λ_k W_2²(μ, μ_k)` of weighted measures
//! `(μ_k, λ_k)`. Two regimes are supported:
//!
//! * **Free-support**: both the support locations `Y` and the weights `b` are
//!   optimised. Implements the alternating Sinkhorn / centroid-update
//!   scheme of Cuturi-Doucet (2014).
//! * **Fixed-support**: support is fixed and only the weights `b` are
//!   optimised, via an entropic geometric mean of input measures.

/// Fixed-support Wasserstein barycenter (Cuturi-Doucet).
pub mod fixed_support;
/// Free-support Wasserstein barycenter via alternating Sinkhorn + support update.
pub mod free_support;
