//! Node-centrality measures for directed graphs.
//!
//! - [`mod@pagerank`]    PageRank (Brin & Page 1998) — stationary distribution of
//!   a damped random walk, computed by power iteration.
//! - [`mod@betweenness`] Betweenness centrality (Brandes 2001) — fraction of
//!   shortest paths passing through each node.
//! - [`mod@closeness`]   Closeness centrality (Bavelas 1950, Wasserman-Faust
//!   normalisation) — inverse mean shortest-path distance to all reachable nodes.

pub mod betweenness;
pub mod closeness;
pub mod pagerank;

pub use betweenness::betweenness_centrality;
pub use closeness::closeness_centrality;
pub use pagerank::{PageRankConfig, pagerank};
