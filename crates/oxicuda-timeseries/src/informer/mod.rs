//! Informer: ProbSparse self-attention for long-sequence time-series forecasting.
//!
//! Implements the Informer architecture from Zhou et al. (2021 AAAI Best Paper),
//! which achieves O(L log L) time and memory complexity via sparse query selection.
//!
//! Reference: "Informer: Beyond Efficient Transformer for Long Sequence
//! Time-Series Forecasting", Zhou et al., AAAI 2021.

pub mod prob_sparse;

pub use prob_sparse::{
    InformerBlock, InformerEncoder, InformerEncoderConfig, InformerEncoderWeights, InformerResult,
    ProbSparseConfig, ProbSparseWeights,
};
