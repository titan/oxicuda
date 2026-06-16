//! Pairwise distance matrix, k-NN graph, and persistence-diagram kernels.

pub mod fisher;
pub mod kernel;
pub mod pairwise;

pub use fisher::{PersistenceFisherConfig, persistence_fisher_distance, persistence_fisher_kernel};
pub use kernel::{
    KernelConfig, PwgkConfig, persistence_scale_space_distance, persistence_scale_space_kernel,
    persistence_weighted_gaussian_distance, persistence_weighted_gaussian_kernel,
    sliced_wasserstein_kernel,
};
pub use pairwise::{
    knn_graph, pairwise_euclidean, pairwise_euclidean_sq, pairwise_manhattan,
    points_to_distance_matrix,
};
