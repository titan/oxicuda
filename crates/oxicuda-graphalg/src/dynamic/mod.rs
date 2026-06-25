//! Streaming dynamic-graph algorithms.
//!
//! Algorithms that maintain a result under a *stream* of edge insertions and
//! deletions, recomputing only what each update can change instead of solving
//! from scratch every time.
//!
//! - [`dynamic_graph`] — [`DynamicGraph`], a mutable directed graph with a
//!   reverse index and edge-multiplicity tracking (the streaming substrate).
//! - [`incremental_pagerank`] — [`IncrementalPageRank`], warm-started power
//!   iteration that resumes from the previous stationary vector after each
//!   update, converging to the same ranks a cold solve would.
//! - [`incremental_scc`] — [`IncrementalScc`], strongly-connected-component
//!   maintenance that merges components on cycle-closing insertions and re-splits
//!   only the affected component on deletions.

pub mod dynamic_graph;
pub mod incremental_pagerank;
pub mod incremental_scc;

pub use dynamic_graph::DynamicGraph;
pub use incremental_pagerank::IncrementalPageRank;
pub use incremental_scc::IncrementalScc;
