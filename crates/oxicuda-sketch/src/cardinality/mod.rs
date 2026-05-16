//! Cardinality / distinct-count estimation sketches.
//!
//! Includes HyperLogLog, HyperLogLog++, and Linear Counting.

pub mod hll;
pub mod hll_plus;
pub mod linear_counting;

pub use hll::HyperLogLog;
pub use hll_plus::HyperLogLogPlus;
pub use linear_counting::LinearCounter;
