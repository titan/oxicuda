//! Linear dimensionality reduction methods.
//!
//! - [`pca`] Principal Component Analysis via covariance eigendecomposition.
//! - [`incremental_pca`] Incremental PCA (Ross-Lim-Lin-Yang 2008).
//! - [`kernel_pca()`] Kernel PCA on centered Gram matrices.
//! - [`fast_ica()`] FastICA fixed-point algorithm.
//! - [`sparse_pca()`] Sparse PCA via Penalized Matrix Decomposition (Witten-Tibshirani-Hastie 2009).
//! - [`cca_pls`] CCA and PLS cross-decomposition variants.

pub mod cca_pls;
pub mod fast_ica;
pub mod incremental_pca;
pub mod kernel_pca;
pub mod pca;
pub mod sparse_pca;

pub use cca_pls::{
    CcaConfig, CcaFit, PlsConfig, PlsFit, PlsSvdFit, cca_fit, cca_transform, pls_fit, pls_predict,
    pls_svd_fit, pls_transform,
};
pub use fast_ica::{IcaNonlinearity, IcaResult, fast_ica};
pub use incremental_pca::{IncrementalPca, IncrementalPcaConfig};
pub use kernel_pca::{KernelKind, KernelPcaResult, kernel_pca};
pub use pca::{PcaResult, pca_fit};
pub use sparse_pca::{SparsePcaConfig, SparsePcaResult, sparse_pca};
