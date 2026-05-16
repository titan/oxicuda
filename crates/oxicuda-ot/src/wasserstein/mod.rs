//! Wasserstein distances — closed-form 1D, exact higher-dim, sliced approximations.

/// Max-sliced Wasserstein with gradient-ascent optimisation of the projection direction.
pub mod max_sliced;
/// Sliced Wasserstein with random projections.
pub mod sliced;
/// Wasserstein-1 distances (1D closed-form and exact via simplex).
pub mod w1;
/// Wasserstein-2 distances (1D closed-form and exact via simplex).
pub mod w2;
