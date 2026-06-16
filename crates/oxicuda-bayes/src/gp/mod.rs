//! Gaussian Process models.
pub mod deep_gp;
pub mod gpr;
pub mod sparse_gp;
pub use deep_gp::{DeepGp, DeepGpConfig, DeepGpLayer, DeepGpLayerConfig};
pub use gpr::{
    GprConfig, GprFit, GprKernel, gpr_fit, gpr_kernel_matrix, gpr_log_marginal_likelihood,
    gpr_predict,
};
pub use sparse_gp::{
    InducingInit, SparseGpConfig, SparseGpFit, sparse_gp_elbo, sparse_gp_fit, sparse_gp_predict,
};
