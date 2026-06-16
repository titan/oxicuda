//! Cardinality / distinct-count estimation sketches.
//!
//! Includes HyperLogLog, HyperLogLog++, HLL-TailCut, Linear Counting, and Theta
//! Sketch.

pub mod hll;
pub mod hll_plus;
pub mod hll_tailcut;
pub mod hyperloglog;
pub mod linear_counting;
pub mod sliding_window_hll;
pub mod theta_sketch;

pub use hll::HyperLogLog;
pub use hll_plus::HyperLogLogPlus;
pub use hll_tailcut::HllTailCut;
pub use hyperloglog::{HllConfig, HyperLogLog as HyperLogLogBytes};
pub use linear_counting::LinearCounter;
pub use sliding_window_hll::{SlidingWindowHll, SlidingWindowHllConfig};
pub use theta_sketch::ThetaSketch;
