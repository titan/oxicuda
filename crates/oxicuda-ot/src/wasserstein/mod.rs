//! Wasserstein distances — closed-form 1D, exact higher-dim, sliced approximations.

/// Knothe-Rosenblatt triangular transport map and 1D quantile coupling.
pub mod knothe_rosenblatt;
/// Max-sliced Wasserstein with gradient-ascent optimisation of the projection direction.
pub mod max_sliced;
/// Mini-batch Wasserstein / Sinkhorn-GAN-style OT loss.
pub mod minibatch_ot;
/// Sliced Wasserstein with random projections.
pub mod sliced;
/// Spherical Sliced-Wasserstein for distributions on the unit sphere.
pub mod spherical_sliced;
/// Stochastic OT via mini-batch dual potential EMA (Genevay et al. 2016, Seguy et al. 2018).
pub mod stochastic_ot;
/// Sliced-Wasserstein gradient flow for generative modelling (Liutkus et al. 2019).
pub mod sw_gradient_flow;
/// Wasserstein-1 distances (1D closed-form and exact via simplex).
pub mod w1;
/// Wasserstein-2 distances (1D closed-form and exact via simplex).
pub mod w2;

pub use knothe_rosenblatt::{
    KrConfig, KrFit, kr_fit_1d, kr_fit_nd, kr_transform_1d, kr_transform_nd, kr_transport_cost_1d,
    kr_wasserstein_1d,
};
pub use minibatch_ot::{
    MinibatchOtConfig, MinibatchOtFit, minibatch_sinkhorn_divergence, minibatch_wasserstein,
};
pub use spherical_sliced::{
    MaxSSWConfig, SphericalSlicedConfig, max_spherical_sliced_wasserstein, normalise_to_sphere,
    sample_uniform_sphere, spherical_sliced_wasserstein, w_p_1d,
};
pub use stochastic_ot::{
    StochasticOtConfig, StochasticOtFit, stochastic_marginal_violation, stochastic_ot,
    stochastic_transport_cost, stochastic_transport_plan,
};
pub use sw_gradient_flow::{SwgfConfig, SwgfFit, sw_distance, sw_gradient_flow, sw_gradient_step};
