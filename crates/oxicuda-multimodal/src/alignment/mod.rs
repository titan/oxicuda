//! Cross-modal alignment losses and heads.

pub mod contrastive;
pub mod llava_projector;
pub mod matching;
pub mod siglip;
pub mod whisper_log_mel;

pub use llava_projector::{LlavaProjector, LlavaProjectorConfig};
pub use siglip::{SigLipConfig, siglip_labels, siglip_loss, siglip_loss_from_sim};
pub use whisper_log_mel::{WhisperLogMel, WhisperLogMelConfig};
