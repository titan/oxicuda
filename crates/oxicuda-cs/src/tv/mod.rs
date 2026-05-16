//! Total Variation denoising in 1D and 2D.

pub mod total_variation_denoise;
pub mod tv_1d_chambolle;
pub mod tv_2d_chambolle;

pub use total_variation_denoise::total_variation_denoise;
pub use tv_1d_chambolle::tv_1d_chambolle;
pub use tv_2d_chambolle::tv_2d_chambolle;
