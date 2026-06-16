//! Community detection algorithms for graphs.
//!
//! Currently provides Louvain modularity maximisation (Blondel et al. 2008).

pub mod louvain;

pub use louvain::{LouvainConfig, LouvainResult, louvain_communities};
