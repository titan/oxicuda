//! TDA summary statistics and topological descriptors.

pub mod bottleneck;
pub mod diagram_wasserstein;
pub mod metrics;

pub use bottleneck::{
    bottleneck_distance as bottleneck_distance_raw, landscape_mean,
    persistence_landscape as persistence_landscape_raw,
};
pub use diagram_wasserstein::{diagram_wasserstein_2, diagram_wasserstein_p};
pub use metrics::{
    betti_numbers, count_components, landscape_distance, persistence_landscape, persistent_entropy,
    total_persistence,
};
