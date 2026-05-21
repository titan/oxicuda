//! Structured SVM with cutting-plane optimisation for linear-chain prediction.

pub mod cutting_plane;
pub mod cutting_plane_full;
pub mod ssvm;

pub use cutting_plane::CuttingPlaneConfig;
pub use cutting_plane_full::{FullCuttingPlaneConfig, FullCuttingPlaneResult, FullCuttingPlaneSvm};
pub use ssvm::StructuredSvm;
