//! 3D Gaussian splatting primitives.

pub mod bvh;
#[allow(clippy::module_inception)]
pub mod gaussian;
pub mod gaussian_2d;
pub mod mip_splat;
pub mod project;
pub mod project_grad;
pub mod raster_grad;
pub mod rasterize;
pub mod tile_raster;

pub use bvh::{GaussianBvh, GaussianHit};
pub use mip_splat::{MipSplatConfig, apply_2d_mip_filter, apply_3d_smoothing};
pub use project_grad::{ProjectGrad, project_backward};
pub use raster_grad::{
    DiffSplat2d, ForwardCache, SplatGrad, rasterize_backward_2d, rasterize_forward_2d,
};
pub use tile_raster::{
    TileBinning, TileRasterConfig, bin_gaussians_to_tiles, rasterize_gaussians_tiled,
};
