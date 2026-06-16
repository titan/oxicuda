//! `oxicuda-tda` — Topological Data Analysis for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-tda
//! ├── complex/        — Simplex, SimplicialComplex, Filtration (Vietoris-Rips, Čech, sublevel)
//! ├── distance/       — Pairwise distance matrix, k-NN graph
//! ├── homology/       — BoundaryMatrix, column reduction (Z₂), persistence pairs
//! ├── persistence/    — PersistenceDiagram, Barcode, bottleneck/Wasserstein distances
//! ├── mapper/         — Mapper algorithm: cover, single-linkage clustering, MapperGraph
//! ├── witness/        — Maxmin landmark selection, lazy witness complex
//! ├── vector/         — Vectorised summaries: Betti curves
//! └── metrics/        — Betti numbers, persistent entropy, landscape, total persistence
//! ```

pub mod complex;
pub mod distance;
pub mod error;
pub mod handle;
pub mod homology;
pub mod mapper;
pub mod metrics;
pub mod morse;
pub mod persistence;
pub mod ptx_kernels;
pub mod vector;
pub mod witness;

pub use error::{TdaError, TdaResult};

#[cfg(test)]
mod e2e_tests;
