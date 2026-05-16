//! Entropic Optimal Transport — Sinkhorn-Knopp algorithm and friends.

/// Debiased Sinkhorn divergence.
pub mod divergence;
/// Low-level log-stabilised single Sinkhorn iteration.
pub mod log_sinkhorn;
/// Log-domain Sinkhorn-Knopp algorithm.
pub mod sinkhorn;
