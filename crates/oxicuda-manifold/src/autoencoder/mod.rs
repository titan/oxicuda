//! Auto-encoder-style manifold learning hooks.
//!
//! Provides the interface between manifold learning algorithms and neural-network training,
//! exporting embedding coordinates, reconstruction, loss, and gradients.
//!
//! # Key types
//!
//! - [`ManifoldHook`] — trait defining the hook interface
//! - [`PcaManifoldHook`] — PCA-based hook (linear encoder/decoder, orthonormal components)
//! - [`TsneRegHook`] — t-SNE regularized gradient hook combining reconstruction + manifold grads
//! - [`EmbeddingExport`] — complete export bundle for neural-network consumers
//!
//! # Export function
//!
//! - [`manifold_encode_and_export`] — one-shot forward+backward for a `PcaManifoldHook`

pub mod manifold_hooks;

pub use manifold_hooks::{
    EmbeddingExport, ManifoldHook, PcaManifoldHook, TsneRegHook, manifold_encode_and_export,
};
