//! CLIP (Contrastive Language–Image Pre-Training) vision module.
//!
//! Provides:
//! - **`ClipVisionEncoder`**: ViT-based image encoder that produces a single
//!   CLS-token embedding per image.
//! - **`ProjectionHead`**: linear projection + L2 normalisation mapping
//!   encoder embeddings to a shared CLIP embedding space.
//! - **`info_nce_loss`**: numerically-stable symmetric InfoNCE / NT-Xent loss.

pub mod contrastive;
pub mod projection;
pub mod vision_encoder;

pub use contrastive::info_nce_loss;
pub use projection::{ProjectionHead, ProjectionWeights};
pub use vision_encoder::{ClipVisionConfig, ClipVisionEncoder};
