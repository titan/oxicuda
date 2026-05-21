//! Standard SSL data augmentation helpers operating on `[C, H, W]` CHW tensors.

pub mod color;
pub mod multi_crop;
pub mod rand_augment;
pub mod solarize_blur;

pub use rand_augment::{
    AugOp, AutoAugPolicy, AutoAugmentConfig, RandAugmentConfig, SubPolicy, all_aug_ops,
    apply_aug_op, auto_augment, rand_augment,
};
pub use solarize_blur::{
    SimClrBlurSolarConfig, add_gaussian_noise, gaussian_blur_chw, random_gaussian_blur_chw,
    random_solarize, simclr_blur_solar, solarize,
};
