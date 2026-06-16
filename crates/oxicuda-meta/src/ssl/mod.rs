//! Self-supervised auxiliary tasks for few-shot representation learning.
//!
//! Currently exposes the rotation-prediction pretext task ([`rotation`]) used by
//! S2M2 / RotNet to regularise the backbone with a self-supervised objective.

pub mod rotation;

pub use rotation::{
    NUM_ROTATIONS, RotationConfig, RotationHead, rotate_chw, rotation_pretext_loss,
};
