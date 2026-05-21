//! Clustering algorithms for manifold-embedded data.
//!
//! Currently provides:
//! - [`spectral`]    — Spectral Clustering (Laplacian eigenmaps + k-means++).
//! - [`kohonen_som`] — Kohonen Self-Organizing Map (SOM).

pub mod kohonen_som;
pub mod spectral;

pub use kohonen_som::{
    KohonenSomConfig, KohonenSomResult, SomInit, kohonen_som_fit, som_grid_pos, som_predict,
    som_weight_at,
};
pub use spectral::{SpectralClusteringConfig, SpectralClusteringResult, spectral_clustering};
