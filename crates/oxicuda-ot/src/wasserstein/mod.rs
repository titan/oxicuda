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

/// Neural OT map via Input-Convex Neural Networks (Makkuva 2020, Korotin 2021).
pub mod neural_ot;
pub use neural_ot::{
    IcnnGrad, IcnnWeights, NeuralOtConfig, NeuralOtFit, neural_ot, neural_ot_dual_bound,
};

/// Sliced Wasserstein distance (f64 API, Rabin 2012).
pub mod sliced_wasserstein;
/// Unbalanced OT in Wasserstein space (f64, KL marginal relaxation, Chizat 2018).
pub mod unbalanced_ot;

pub use sliced_wasserstein::{SlicedWassersteinConfig, random_unit_vector, sliced_wasserstein};
pub use unbalanced_ot::{UnbalancedOtConfig, unbalanced_sinkhorn};

/// McCann displacement interpolation along the Wasserstein-2 geodesic (McCann 1997).
pub mod w2_interpolation;
pub use w2_interpolation::{
    InterpolatedMeasure, displacement_interpolate_1d, displacement_interpolate_plan,
    displacement_path_plan,
};

/// Wasserstein-1 dual (Kantorovich-Rubinstein potentials, f64 API).
pub mod w1_dual;
pub use w1_dual::{W1DualConfig, W1DualResult, w1_1d_exact, w1_dual, w1_dual_from_cost};

/// Tree-Wasserstein distance with closed-form linear-time evaluation (Le 2019).
pub mod tree_wasserstein;
pub use tree_wasserstein::{WeightedTree, balanced_binary_tree, tree_wasserstein};
