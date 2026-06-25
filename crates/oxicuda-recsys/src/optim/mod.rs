//! Optimizers for recommender embedding tables.
//!
//! Currently provides [`sparse_adamw::SparseAdamW`], a row-wise sparse-gradient
//! AdamW optimizer suited to very large embedding tables where each mini-batch
//! touches only a handful of rows.

pub mod sparse_adamw;

pub use sparse_adamw::{RowGrad, SparseAdamW, SparseAdamWConfig};
