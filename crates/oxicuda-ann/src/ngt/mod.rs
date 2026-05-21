//! NGT — Neighborhood Graph and Tree (ANNG variant).
//!
//! Provides [`NgtIndex`], an incrementally-built Approximate Neighborhood Graph
//! with ε-relaxed greedy best-first graph search.

pub mod index;

pub use index::{NgtConfig, NgtIndex};
