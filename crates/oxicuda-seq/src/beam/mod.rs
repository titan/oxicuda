//! Generic beam search infrastructure.

pub mod beam;
pub mod diverse;

pub use beam::{BeamConfig, BeamSearch};
pub use diverse::{DiverseBeam, DiverseBeamConfig};
