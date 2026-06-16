//! Open-set, language-grounded detection.
//!
//! - [`gdino`] — Grounding-DINO cross-modality fusion, language-guided query
//!   selection, and box + alignment heads.

pub mod gdino;

pub use gdino::{GroundingDino, GroundingDinoConfig, GroundingDinoWeights};
