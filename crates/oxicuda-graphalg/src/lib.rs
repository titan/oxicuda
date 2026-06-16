//! `oxicuda-graphalg` — Classical Graph Algorithms for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-graphalg
//! ├── repr/            — Graph representations (AdjacencyList, AdjacencyMatrix,
//! │                      EdgeList, CSR, WeightedGraph)
//! ├── traversal/       — BFS, DFS (iterative + post-order), IDDFS, bidirectional BFS
//! ├── topological/     — Kahn's algorithm and DFS-based topological sort
//! ├── shortest_path/   — Dijkstra, Bellman-Ford, SPFA, Floyd-Warshall, Johnson,
//! │                      A*, Yen K-shortest, bidirectional Dijkstra
//! ├── mst/             — Prim, Kruskal, Borůvka, Union-Find
//! ├── max_flow/        — Edmonds-Karp, Dinic, push-relabel, min-cut
//! ├── matching/        — Hopcroft-Karp bipartite, Hungarian (Munkres) assignment,
//! │                      simplified blossom for general unweighted matching
//! ├── flow/            — Gomory-Hu cut tree (Gusfield), Stoer-Wagner global min cut
//! ├── path/            — Suurballe vertex-disjoint shortest path pair
//! ├── connectivity/    — Tarjan / Kosaraju / Gabow SCC, bridges, articulation
//! │                      points, biconnected components
//! ├── centrality/      — Degree, betweenness (Brandes), closeness, eigenvector,
//! │                      PageRank, Katz
//! ├── community/       — Louvain, label propagation, Girvan-Newman
//! ├── arborescence/    — Chu-Liu-Edmonds minimum spanning arborescence
//! ├── isomorphism/     — VF2 subgraph isomorphism
//! ├── coloring/        — Greedy, DSATUR, Welsh-Powell
//! ├── tsp/             — Christofides approximation, nearest-neighbor, 2-opt
//! ├── eulerian/        — Hierholzer's Eulerian circuit
//! ├── hamiltonian/     — Held-Karp DP exact TSP
//! └── metrics/         — Diameter, radius, density, clustering coefficient, transitivity
//! ```
//!
//! All algorithms are implemented in pure Rust with no external graph libraries.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::single_match)]
#![allow(clippy::redundant_closure)]

pub mod arborescence;
pub mod centrality;
pub mod clique;
pub mod coloring;
pub mod community;
pub mod connectivity;
pub mod error;
pub mod eulerian;
pub mod flow;
pub mod hamiltonian;
pub mod handle;
pub mod isomorphism;
pub mod matching;
pub mod max_flow;
pub mod metrics;
pub mod min_cost_flow;
pub mod mst;
pub mod path;
pub mod ptx_kernels;
pub mod repr;
pub mod separation;
pub mod shortest_path;
pub mod topological;
pub mod traversal;
pub mod tsp;

pub use error::{GraphalgError, GraphalgResult};
pub use handle::{GraphalgHandle, LcgRng, SmVersion};

#[cfg(test)]
mod e2e_tests;
