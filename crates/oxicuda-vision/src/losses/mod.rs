//! Loss functions for classification, detection, and segmentation heads.
//!
//! Provides:
//! - **`focal`**: sigmoid (binary) and softmax (multiclass) focal loss
//!   (Lin et al. 2017) for class-imbalanced dense prediction.
//! - **`dice`**: soft Dice / V-Net Dice loss (Milletari et al. 2016) for
//!   overlap-based segmentation training.
//! - **`quality`**: reconstruction / image-quality metrics — MSE, PSNR, SSIM,
//!   MS-SSIM (also usable as differentiable training objectives).
//!
//! IoU-family bounding-box regression losses live in
//! [`crate::detection::iou_losses`].

pub mod dice;
pub mod focal;
pub mod quality;

pub use dice::{dice_loss, dice_loss_batch, dice_loss_default, dice_loss_squared};
pub use focal::{
    Reduction, binary_focal_loss, binary_focal_loss_one, multiclass_focal_loss,
    multiclass_focal_loss_one,
};
pub use quality::{MS_SSIM_WEIGHTS, SSIM_C1, SSIM_C2, ms_ssim, mse, psnr, ssim, ssim_default};
