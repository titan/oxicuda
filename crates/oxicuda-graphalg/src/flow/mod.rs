//! Cut-structure algorithms built on max-flow / min-cut.
//!
//! - [`gomory_hu`] — the Gomory-Hu cut tree (Gusfield's `n − 1` max-flow construction),
//!   encoding every pairwise minimum cut of an undirected weighted graph.
//! - [`stoer_wagner`] — the Stoer-Wagner global minimum cut, found without any max-flow
//!   computation via maximum adjacency ordering.

pub mod gomory_hu;
pub mod stoer_wagner;

pub use gomory_hu::{GomoryHuTree, gomory_hu_tree};
pub use stoer_wagner::{GlobalMinCut, stoer_wagner_min_cut};
