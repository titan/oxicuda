//! Graph-based ANN indices that build on the shared distance, k-NN-graph and
//! top-K primitives.
//!
//! * [`nsg`] — Navigating Spreading-out Graph (Fu et al., VLDB 2019): an MRNG-
//!   pruned, single-entry navigable graph derived from an approximate k-NN
//!   graph.
//! * [`filtered_search`] — Filtered-DiskANN style label-constrained search
//!   (Gollapudi et al., WWW 2023): per-point label sets, per-label entry points
//!   and filter-aware graph traversal.
//! * [`spann`] — SPANN posting-list index (Chen et al., NeurIPS 2021): ~√n
//!   centroids with boundary-point duplication and a coarse centroid head index.

pub mod filtered_search;
pub mod nsg;
pub mod spann;
