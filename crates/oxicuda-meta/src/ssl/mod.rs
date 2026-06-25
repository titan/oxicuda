//! Self-supervised auxiliary tasks for few-shot representation learning.
//!
//! Exposes:
//!
//! * the rotation-prediction pretext task ([`rotation`]) used by S2M2 / RotNet to
//!   regularise the backbone with a self-supervised objective;
//! * ProtoTransfer / ProtoCLR ([`proto_transfer`]) — self-supervised
//!   prototypical contrastive pre-training that transfers a frozen embedding to a
//!   downstream ProtoNet few-shot episode (Medina, Devos & Grossglauser 2020).

pub mod proto_transfer;
pub mod rotation;

pub use proto_transfer::{ProtoTransferConfig, ProtoTransferHead, l2_normalize, proto_clr_loss};
pub use rotation::{
    NUM_ROTATIONS, RotationConfig, RotationHead, rotate_chw, rotation_pretext_loss,
};
