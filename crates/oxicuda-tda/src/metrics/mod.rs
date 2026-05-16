//! TDA summary statistics and topological descriptors.

pub mod metrics;

pub use metrics::{
    betti_numbers, count_components, landscape_distance, persistence_landscape, persistent_entropy,
    total_persistence,
};
