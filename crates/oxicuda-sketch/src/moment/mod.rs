//! Moment / norm estimation sketches.

pub mod ams_l2;
pub mod johnson_lindenstrauss;
pub mod lp_norm;

pub use ams_l2::AmsL2Sketch;
pub use johnson_lindenstrauss::JlProjection;
pub use lp_norm::LpNormSketch;
