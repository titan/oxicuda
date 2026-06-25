//! Neural audio codec primitives for `oxicuda-audio`.
//!
//! Provides:
//! - **`rvq`**: Residual Vector Quantization (SoundStream / EnCodec / Bark
//!   acoustic-token core) — greedy stage-by-stage residual nearest-neighbour
//!   quantization with exact round-trip, monotone residual descent (reserved
//!   zero code), and stage-wise k-means codebook adaptation.
//! - **`bark`**: thin coarse/fine acoustic-token *layout* wrapper over the RVQ
//!   stages.  The trained Bark token-generation transformers are out of scope
//!   (documented in [`bark`]).

pub mod bark;
pub mod rvq;

pub use bark::{BarkAcousticTokens, BarkCodec};
pub use rvq::{ResidualVectorQuantizer, RvqFitReport};
