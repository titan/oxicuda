//! Encoder submodules for `oxicuda-audio`.
//!
//! Provides Wav2Vec2 CNN feature extraction, Conformer blocks, and a
//! Whisper-style transformer encoder for end-to-end speech encoding pipelines.

pub mod conformer_block;
pub mod conv_module;
pub mod hubert_ssl;
pub mod quantized;
pub mod streaming_conformer;
pub mod wav2vec_cnn;
pub mod whisper;

pub use conformer_block::{ConformerBlock, ConformerConfig, ConformerEncoder};
pub use conv_module::ConvModule;
pub use hubert_ssl::{
    HubertPretrainConfig, HubertPretrainer, KMeansQuantizer, MaskedPredictionHead, SpanMaskConfig,
    apply_span_mask, compute_mask_indices,
};
pub use quantized::{
    QuantizedFfn, QuantizedLinear, QuantizedTensor, compute_scale_symmetric, dequantize_symmetric,
    quantization_error_rms, quantize_symmetric,
};
pub use streaming_conformer::{
    LeftContextCache, StreamingConformerAttention, StreamingConformerConfig,
};
pub use wav2vec_cnn::{Wav2VecCnnConfig, Wav2VecCnnEncoder, Wav2VecCnnLayer};
pub use whisper::{WhisperEncoder, WhisperEncoderConfig};
