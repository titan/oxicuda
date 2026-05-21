//! Cardinality / distinct-count estimation sketches.
//!
//! Includes HyperLogLog, HyperLogLog++, Linear Counting, and Theta Sketch.

pub mod hll;
pub mod hll_plus;
pub mod linear_counting;
pub mod sliding_window_hll;
pub mod theta_sketch;

pub use hll::HyperLogLog;
pub use hll_plus::HyperLogLogPlus;
pub use linear_counting::LinearCounter;
pub use sliding_window_hll::{SlidingWindowHll, SlidingWindowHllConfig};
pub use theta_sketch::ThetaSketch;
