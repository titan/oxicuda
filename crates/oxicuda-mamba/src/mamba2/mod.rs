//! Mamba-2 (SSD) architecture: State Space Duality framework.
//!
//! # Submodules
//!
//! - [`ssd`]               — Core SSD algorithm: naive O(L²) and recurrent O(L·N) forms.
//! - [`chunk_scan`]        — Chunk-wise scan for efficient Mamba-2 computation.
//! - [`mamba2_block`]      — Full Mamba-2 residual block with multi-head SSM.
//! - [`ssd_chunk_layer`]   — Simplified SSD Chunk-Scan layer (CPU reference).

pub mod chunk_scan;
pub mod mamba2_block;
pub mod ssd;
pub mod ssd_chunk_layer;
pub use ssd_chunk_layer::{SsdChunk, SsdChunkConfig};
