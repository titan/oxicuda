//! Gromov-Wasserstein and Fused-Gromov-Wasserstein OT for distributions on
//! possibly different metric spaces.
//!
//! Where ordinary OT requires a single ground cost between source and target
//! supports, Gromov-Wasserstein lifts that constraint by comparing
//! intra-domain distance matrices `C^1` and `C^2`. Fused-GW interpolates
//! between intra-domain GW and an inter-domain Wasserstein term using a mixing
//! parameter `α ∈ [0, 1]`.

/// Fused Gromov-Wasserstein combining intra-domain GW and inter-domain Wasserstein.
pub mod fused;
/// Entropic Gromov-Wasserstein for distributions on possibly different metric spaces.
pub mod gromov_wasserstein;
