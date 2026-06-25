//! Self-supervised learning (SSL) methods for vision backbones.
//!
//! Provides:
//! - **`dinov2`**: the DINOv2 self-distillation recipe (Oquab et al. 2023),
//!   combining the image-level DINO objective (Caron et al. 2021) with the
//!   patch-level iBOT masked-image-modelling objective (Zhou et al. 2022) — a
//!   ViT backbone returning `[CLS]` + patch tokens, a weight-normalised
//!   prototype projection head, a centred-and-sharpened teacher / softer
//!   student cross-entropy loss, an EMA teacher update, a centering buffer, and
//!   the iBOT masked-patch term.

pub mod dinov2;

pub use dinov2::{
    BackboneOutput, CenteringBuffer, DinoBackbone, DinoHead, cross_entropy, dino_loss, ibot_loss,
    koleo_loss, student_softmax, teacher_softmax,
};
