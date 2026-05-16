//! Minimum spanning tree algorithms.

pub mod boruvka;
pub mod kruskal;
pub mod prim;
pub mod union_find;

pub use boruvka::boruvka_mst;
pub use kruskal::kruskal_mst;
pub use prim::prim_mst;
pub use union_find::UnionFind;
