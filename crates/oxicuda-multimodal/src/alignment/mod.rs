//! Cross-modal alignment losses and heads.

pub mod audio_clip;
pub mod contrastive;
pub mod hard_negative;
pub mod llava_projector;
pub mod matching;
pub mod siglip;
pub mod whisper_log_mel;

pub use audio_clip::{AudioClipConfig, AudioClipLoss, audio_clip_loss};
pub use hard_negative::{hard_negative_infonce, mine_hard_negatives, vse_plus_plus_loss};
pub use llava_projector::{LlavaProjector, LlavaProjectorConfig};
pub use siglip::{SigLipConfig, siglip_labels, siglip_loss, siglip_loss_from_sim};
pub use whisper_log_mel::{WhisperLogMel, WhisperLogMelConfig};
