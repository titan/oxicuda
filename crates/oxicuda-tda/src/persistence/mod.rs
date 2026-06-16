//! Persistence diagram, barcode, and diagram distance metrics.

pub mod barcode;
pub mod diagram;
pub mod distance;
pub mod extended;
pub mod landscape_p;
pub mod persistence_image;
pub mod tropical;
pub mod wasserstein_p;

pub use barcode::{Bar, Barcode};
pub use diagram::PersistenceDiagram;
pub use distance::{bottleneck_distance, wasserstein_1};
pub use landscape_p::PersistenceLandscape;
pub use persistence_image::{
    PersistenceImageConfig, PersistenceImageResult, WeightFunction, persistence_image,
    persistence_image_batch, persistence_image_distance, persistence_silhouette,
};
pub use tropical::{
    TropicalConfig, tropical_coordinates, tropical_max_plus, tropical_total_persistence,
};
pub use wasserstein_p::{
    build_cost_matrix, diagonal_dist, hungarian, matching_cost, point_dist_inf, sliced_wasserstein,
    wasserstein_p,
};
