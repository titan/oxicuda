//! Pairwise distance matrix and k-NN graph.

pub mod pairwise;

pub use pairwise::{
    knn_graph, pairwise_euclidean, pairwise_euclidean_sq, pairwise_manhattan,
    points_to_distance_matrix,
};
