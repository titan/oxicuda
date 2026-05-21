//! Cross-modal alignment losses and heads.

pub mod contrastive;
pub mod llava_projector;
pub mod matching;
pub mod whisper_log_mel;

pub use llava_projector::{LlavaProjector, LlavaProjectorConfig};
pub use whisper_log_mel::{WhisperLogMel, WhisperLogMelConfig};
