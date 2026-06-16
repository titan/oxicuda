//! Lazy Witness Complex: landmark selection and complex construction.

pub mod strong_witness;
pub mod witness;

pub use strong_witness::{StrongWitnessConfig, StrongWitnessResult, strong_witness_complex};
pub use witness::{WitnessConfig, lazy_witness_complex, maxmin_landmarks};
