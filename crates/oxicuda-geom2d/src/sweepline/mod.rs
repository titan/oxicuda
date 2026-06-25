//! Sweepline algorithms.

pub mod bentley_ottmann;
pub mod streaming_bentley_ottmann;

pub use bentley_ottmann::bentley_ottmann;
pub use streaming_bentley_ottmann::{
    CollectingSink, CountingSink, IntersectionSink, StreamingSweep, count_intersections,
    report_intersections,
};
