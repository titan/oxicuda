//! Community detection algorithms.

pub mod girvan_newman;
pub mod label_propagation;
pub mod louvain;

pub use girvan_newman::girvan_newman_communities;
pub use label_propagation::label_propagation;
pub use louvain::louvain_communities;
