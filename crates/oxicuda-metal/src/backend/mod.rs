//! [`MetalBackend`] — the main entry point for the oxicuda-metal crate.
//!
//! Implements the [`oxicuda_backend::ComputeBackend`] trait using Apple's
//! Metal API for GPU compute on macOS.
//!
//! Submodule layout:
//! - [`types`]: the [`MetalBackend`] struct, init helpers, and Metal/stub
//!   dispatch implementations split by `target_os`.
//! - [`trait_impls`]: the public `Default` and `ComputeBackend` impls.
//! - [`functions`]: small utility helpers and the `mod tests` integration
//!   suite.

pub mod functions;
pub mod trait_impls;
pub mod types;

pub use types::MetalBackend;
