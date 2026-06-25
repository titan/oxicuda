//! Hardware-accelerated memory compression support (Ampere+).
//!
//! This module groups the host-side bookkeeping for the CUDA virtual-memory
//! compression feature.  See [`compressed_buffer`] for the descriptor and
//! allocation-planning types.

pub mod compressed_buffer;

pub use compressed_buffer::{
    CompressedDeviceBuffer, CompressionPlan, CompressionSupport, CompressionType,
    DEFAULT_COMPRESSION_GRANULARITY,
};
