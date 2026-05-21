//! Topological Data Analysis (TDA) algorithms.
//!
//! Provides:
//! - [`persistent_homology`] — Vietoris-Rips persistent homology (H0 and H1) and the Mapper algorithm.

pub mod persistent_homology;

pub use persistent_homology::{
    MapperConfig, MapperGraph, MapperNode, PersistenceDiagram, PersistencePair, VietorisRipsConfig,
    bottleneck_distance, mapper, persistence_betti, vietoris_rips_persistence,
};
