//! Object detection components.
//!
//! Provides:
//! - **`roi_align`**: CPU reference RoI Align matching the `roi_align_ptx` kernel.
//! - **`DetrDecoder`**: multi-layer DETR decoder (pre-norm self-attn + cross-attn + FFN).
//! - **`bipartite_match`**: greedy+2-opt bipartite matching for DETR set-prediction loss.

pub mod detr_decoder;
pub mod roi_align;
pub mod set_match;

pub use detr_decoder::{DetrConfig, DetrDecoder, DetrDecoderLayer};
pub use roi_align::roi_align;
pub use set_match::{MatchCost, bipartite_match};
