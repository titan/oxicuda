//! Clique enumeration and maximum clique algorithms.

pub mod bron_kerbosch;

pub use bron_kerbosch::{bron_kerbosch, bron_kerbosch_degeneracy, maximum_clique};
