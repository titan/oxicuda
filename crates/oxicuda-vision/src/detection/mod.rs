//! Object detection components.
//!
//! Provides:
//! - **`roi_align`**: CPU reference RoI Align matching the `roi_align_ptx` kernel.
//! - **`DetrDecoder`**: multi-layer DETR decoder (pre-norm self-attn + cross-attn + FFN).
//! - **`bipartite_match`**: greedy+2-opt bipartite matching for DETR set-prediction loss.
//! - **`anchor_nms`**: multi-scale anchor generator, IoU, greedy NMS + Soft-NMS.
//! - **`mask_head`**: Mask R-CNN segmentation head (FCN + 2× deconv + per-class 1×1).

pub mod anchor_nms;
pub mod detr_decoder;
pub mod hungarian;
pub mod mask_head;
pub mod roi_align;
pub mod set_match;

pub use anchor_nms::{AnchorConfig, AnchorGenerator, iou, nms, soft_nms};
pub use detr_decoder::{DetrConfig, DetrDecoder, DetrDecoderLayer};
pub use hungarian::{exact_bipartite_match, hungarian};
pub use mask_head::{MaskHead, MaskHeadConfig};
pub use roi_align::roi_align;
pub use set_match::{MatchCost, bipartite_match};
