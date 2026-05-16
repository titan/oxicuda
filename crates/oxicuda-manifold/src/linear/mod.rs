//! Linear dimensionality reduction methods.
//!
//! - [`pca`] Principal Component Analysis via covariance eigendecomposition.
//! - [`kernel_pca()`] Kernel PCA on centered Gram matrices.
//! - [`fast_ica()`] FastICA fixed-point algorithm.

pub mod fast_ica;
pub mod kernel_pca;
pub mod pca;

pub use fast_ica::{IcaNonlinearity, IcaResult, fast_ica};
pub use kernel_pca::{KernelKind, KernelPcaResult, kernel_pca};
pub use pca::{PcaResult, pca_fit};
