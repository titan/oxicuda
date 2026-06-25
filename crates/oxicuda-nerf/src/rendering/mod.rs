//! Rendering modules: ray generation, sampling, volume rendering, occupancy.

pub mod aabb;
pub mod block_nerf;
pub mod contraction;
pub mod deformable_3dgs;
pub mod distortion;
pub mod emernerf;
pub mod gaussian_splat_3d;
pub mod ndc;
pub mod occupancy;
pub mod proposal_network;
pub mod ray;
pub mod ref_nerf;
pub mod sampling;
pub mod sparse_voxel_octree;
pub mod volume_render;
pub mod zip_nerf;

pub use aabb::{Aabb as RayAabb, AabbHit};
pub use block_nerf::{Block, BlockNerfConfig, BlockNerfScene};
pub use contraction::{
    ContractionConfig, contract_batch, contract_point, contracted_norm, is_inner, uncontract_batch,
    uncontract_point,
};
pub use deformable_3dgs::{
    DeformableGaussians, DeformationConfig, DeformationField, GaussianDelta,
};
pub use distortion::{distortion_loss, distortion_loss_batch, distortion_loss_midpoints};
pub use emernerf::{Composite, DynamicOutput, EmerNerf, EmerNerfConfig, warp};
pub use gaussian_splat_3d::{
    Gaussian3d, Splat2d, SplatCamera, SplatImage, SplatPixelGrad, backward_pixel, project_gaussian,
    project_scene, quat_to_rotation, rasterize, rasterize_pixel,
};
pub use ndc::{ndc_depth_to_world, ndc_ray};
pub use proposal_network::{ProposalHistogram, ProposalMlpConfig, ProposalNetwork};
pub use ref_nerf::{
    RefNerf, RefNerfConfig, SpatialOutputs, attenuation_per_degree, ide_encode, reflect,
};
pub use sparse_voxel_octree::{
    Aabb, OctreeNode, RayHit, SparseVoxelOctree, SparseVoxelOctreeConfig,
};
pub use zip_nerf::{Multisample, ZipNerf, ZipNerfConfig};
