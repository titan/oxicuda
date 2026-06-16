//! Random-cut-forest anomaly detection.
//!
//! Currently exposes the Robust Random Cut Forest (RRCF) of Guha et al. (2016)
//! with streaming insert/forget and Collusive-Displacement (CoDisp) scoring.
pub mod rrcf;

pub use rrcf::{BoundingBox, RobustRandomCutForest, RrcNode, RrcTree, RrcfConfig};
