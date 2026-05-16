//! Wasserstein k-means clustering.
//!
//! Cluster a set of probability measures using OT-induced distances and
//! barycenter-based centroids. This is the natural lift of Lloyd's algorithm
//! to Wasserstein space: distances are computed with `wasserstein::w2`, and
//! centroids are recomputed by `barycenter::free_support_barycenter`.

/// Wasserstein k-means clustering using OT-barycenter centroids.
pub mod wasserstein_kmeans;
