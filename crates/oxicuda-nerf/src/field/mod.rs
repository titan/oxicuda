//! Neural field representations.
//!
//! - `tensorf`: TensoRF CP decomposition field
//! - `hash_field`: Instant-NGP style hash grid + tiny MLP decoder
//! - `kplanes`: K-Planes factorised coordinate-plane field
//! - `plenoxel`: Plenoxel voxel grid (density + SH coefficients, no MLP)

pub mod hash_field;
pub mod kplanes;
pub mod plenoxel;
pub mod tensorf;
pub mod vm_field;

pub use kplanes::{KPlanes, KPlanesConfig};
pub use plenoxel::{PlenoxelConfig, PlenoxelGrid};
pub use vm_field::{VmField, VmFieldConfig};
