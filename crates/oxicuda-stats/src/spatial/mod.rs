//! Spatial statistics: Moran's I, Geary's C, Ripley's K.

pub mod spatial;

pub use spatial::{GearyCResult, MoransIResult, geary_c, moran_i, ripleys_k};
