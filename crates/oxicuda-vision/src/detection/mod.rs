//! Object detection components.
//!
//! Provides:
//! - **`roi_align`**: CPU reference RoI Align matching the `roi_align_ptx` kernel.
//! - **`DetrDecoder`**: multi-layer DETR decoder (pre-norm self-attn + cross-attn + FFN).
//! - **`bipartite_match`**: greedy+2-opt bipartite matching for DETR set-prediction loss.
//! - **`anchor_nms`**: multi-scale anchor generator, IoU, greedy NMS + Soft-NMS.
//! - **`mask_head`**: Mask R-CNN segmentation head (FCN + 2× deconv + per-class 1×1).
//! - **`iou_losses`**: IoU / GIoU / DIoU / CIoU bounding-box regression losses.

pub mod anchor_nms;
pub mod detr_decoder;
pub mod hungarian;
pub mod iou_losses;
pub mod mask_head;
pub mod nms;
pub mod owl_vit;
pub mod roi_align;
pub mod rtmdet;
pub mod set_match;

pub use anchor_nms::{AnchorConfig, AnchorGenerator, iou, nms, soft_nms};
pub use detr_decoder::{DetrConfig, DetrDecoder, DetrDecoderLayer};
pub use hungarian::{exact_bipartite_match, hungarian};
// `iou_losses::{iou, giou}` operate on `IouBox`; the existing `anchor_nms::iou`
// and `set_match::giou` operate on `[f32; 4]`. To avoid a name clash, the bare
// `iou` / `giou` from `iou_losses` are reached via `iou_losses::…`; only the
// loss helpers and the `IouBox` type are re-exported here.
pub use iou_losses::{
    IouBox, IouLossKind, ciou, ciou_loss, diou, diou_loss, giou_loss, iou_loss, iou_loss_pairs,
};
pub use mask_head::{MaskHead, MaskHeadConfig};
pub use nms::BBox;
pub use owl_vit::{OwlVit, OwlVitConfig, OwlVitOutput};
pub use roi_align::roi_align;
pub use rtmdet::{
    Bottleneck, Conv2d, CspLayer, CspNeXtBackbone, DecoupledHead, DwConv2d, Pafpn, RtmDet,
    RtmDetConfig, RtmDetOutput, decode_level, simota_cost,
};
pub use set_match::{MatchCost, bipartite_match};
