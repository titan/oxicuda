//! Structured SVM with cutting-plane optimisation for linear-chain prediction.

pub mod cutting_plane;
pub mod ssvm;

pub use cutting_plane::CuttingPlaneConfig;
pub use ssvm::StructuredSvm;
