//! Encoder submodules for `oxicuda-audio`.
//!
//! Provides Wav2Vec2 CNN feature extraction and Conformer blocks
//! for end-to-end speech encoding pipelines.

pub mod conformer_block;
pub mod conv_module;
pub mod wav2vec_cnn;

pub use conformer_block::{ConformerBlock, ConformerConfig, ConformerEncoder};
pub use conv_module::ConvModule;
pub use wav2vec_cnn::{Wav2VecCnnConfig, Wav2VecCnnEncoder, Wav2VecCnnLayer};
