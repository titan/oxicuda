//! Frequency estimation sketches: Count-Min, Count Sketch, Conservative Update,
//! Ada-Sketch, and a differentially-private Count-Min.

pub mod ada_sketch;
pub mod cm_dp;
pub mod conservative_update;
pub mod count_min;
pub mod count_sketch;
pub mod sliding_window_cm;

pub use ada_sketch::AdaSketch;
pub use cm_dp::DpCountMin;
pub use conservative_update::ConservativeUpdateCm;
pub use count_min::CountMinSketch;
pub use count_sketch::CountSketch;
pub use sliding_window_cm::SlidingWindowCm;
