//! Rendering modules: ray generation, sampling, volume rendering, occupancy.

pub mod contraction;
pub mod distortion;
pub mod occupancy;
pub mod proposal_network;
pub mod ray;
pub mod sampling;
pub mod sparse_voxel_octree;
pub mod volume_render;

pub use contraction::{
    ContractionConfig, contract_batch, contract_point, contracted_norm, is_inner, uncontract_batch,
    uncontract_point,
};
pub use distortion::{distortion_loss, distortion_loss_batch, distortion_loss_midpoints};
pub use proposal_network::{ProposalHistogram, ProposalMlpConfig, ProposalNetwork};
pub use sparse_voxel_octree::{
    Aabb, OctreeNode, RayHit, SparseVoxelOctree, SparseVoxelOctreeConfig,
};
