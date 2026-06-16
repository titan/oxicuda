//! Topological Data Analysis (TDA) algorithms.
//!
//! Provides:
//! - [`persistent_homology`] — Vietoris-Rips persistent homology (H0 and H1) and the Mapper algorithm.
//! - [`heat_method`]         — Heat method for geodesic distance computation on point clouds (Crane 2013).

pub mod heat_method;
pub mod persistent_homology;

pub use heat_method::{HeatMethodConfig, HeatMethodResult, heat_method_geodesic};
pub use persistent_homology::{
    MapperConfig, MapperGraph, MapperNode, PersistenceDiagram, PersistencePair, VietorisRipsConfig,
    bottleneck_distance, mapper, persistence_betti, vietoris_rips_persistence,
};
