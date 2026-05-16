//! `oxicuda-hdc` — Hyperdimensional Computing (HDC) / Vector Symbolic Architectures for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-hdc
//! ├── vector/       — Binary {±1}^D, integer Z^D, complex unit (FHRR) hypervectors
//! ├── ops/          — Binding (XOR/multiply/circular-conv), bundling, permutation
//! ├── memory/       — Item memory (symbol→HV), associative (Hopfield-style) memory
//! ├── classifier/   — Online HD classifier with error-corrective update
//! ├── encoding/     — Record-based, n-gram, spatial pattern encoding
//! ├── distance/     — Hamming, cosine, Jaccard similarity metrics
//! └── metrics/      — Capacity bounds, dimensionality analysis, accuracy
//! ```

pub mod classifier;
pub mod distance;
pub mod encoding;
pub mod error;
pub mod handle;
pub mod memory;
pub mod metrics;
pub mod ops;
pub mod ptx_kernels;
pub mod vector;

#[cfg(test)]
mod e2e_tests;
