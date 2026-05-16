//! Error types for `oxicuda-graphalg`.

use thiserror::Error;

/// Top-level error type for graph algorithm operations.
#[derive(Debug, Error)]
pub enum GraphalgError {
    #[error("invalid graph: {0}")]
    InvalidGraph(String),
    #[error("negative weight detected on edge {edge:?}: {weight}")]
    NegativeWeight { edge: (usize, usize), weight: f64 },
    #[error("negative cycle detected in graph")]
    NegativeCycle,
    #[error("source node {node} out of range for graph of {n} nodes")]
    SourceOutOfRange { node: usize, n: usize },
    #[error("graph is disconnected: {0}")]
    DisconnectedGraph(String),
    #[error("graph is not bipartite: {0}")]
    NotABipartiteGraph(String),
    #[error("graph is not a DAG: a cycle was detected")]
    NotADag,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("no solution exists: {0}")]
    NoSolution(String),
    #[error("invalid edge weight: {0}")]
    InvalidEdgeWeight(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Result alias for graph algorithm operations.
pub type GraphalgResult<T> = Result<T, GraphalgError>;
