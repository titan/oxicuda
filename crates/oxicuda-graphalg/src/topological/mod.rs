//! Topological sort algorithms (Kahn, DFS-based).

pub mod dfs_topo;
pub mod kahn;

pub use dfs_topo::topo_sort_dfs;
pub use kahn::topo_sort_kahn;
