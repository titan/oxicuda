//! Quantum error-correction codes and decoders.
//!
//! * [`surface_code`] — distance-`d` rotated surface code (Fowler 2012) with a
//!   minimum-weight perfect-matching decoder, in the classical Pauli-frame
//!   picture used by surface-code threshold studies.
//!
//! The simpler 3-qubit repetition codes (bit-flip / phase-flip) live in
//! [`crate::algorithm::error_correction`].

pub mod surface_code;

pub use surface_code::{
    PauliError, StabKind, Stabilizer, SurfaceCode, SurfaceCodeConfig, Syndrome,
};
