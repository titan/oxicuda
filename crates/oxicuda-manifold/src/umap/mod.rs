//! UMAP: Uniform Manifold Approximation and Projection.

pub mod embedding;
pub mod fuzzy_simplicial;
pub mod knn_graph;
pub mod multiscale;
pub mod supervised;

pub use embedding::{UmapConfig, UmapResult, umap_fit};
pub use fuzzy_simplicial::{fuzzy_simplicial_set, symmetrise};
pub use knn_graph::{build_knn_distances, smooth_knn_distances};
pub use multiscale::{
    MultiScaleUmapConfig, MultiScaleUmapResult, combine_fuzzy_sets, multiscale_umap_fit,
};
pub use supervised::{SupervisedUmapConfig, SupervisedUmapResult, UNLABELED, supervised_umap};
