//! `oxicuda-hdc` — Hyperdimensional Computing (HDC) / Vector Symbolic Architectures for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-hdc
//! ├── vector/       — Binary {±1}^D, integer Z^D, complex unit (FHRR), phasor-only FHRR
//! ├── ops/          — Binding (XOR/multiply/circular-conv), bundling, permutation
//! ├── memory/       — Item memory (symbol→HV), associative (Hopfield-style) memory
//! ├── classifier/   — Online HD classifier with error-corrective update
//! ├── learning/     — Adaptive (iteratively retrained) HD classifier, HD ridge regression
//! ├── encoding/     — Record, n-gram, spatial pattern, level/thermometer, graph encoding
//! ├── distance/     — Hamming, cosine, Jaccard similarity metrics
//! ├── metrics/      — Capacity bounds, dimensionality analysis, accuracy
//! └── analysis/     — Empirical scaling-law characterisation (capacity vs D, bundle SNR vs k)
//! ```

pub mod analysis;
pub mod classifier;
pub mod distance;
pub mod encoding;
pub mod error;
pub mod handle;
pub mod learning;
pub mod memory;
pub mod metrics;
pub mod ops;
pub mod ptx_kernels;
pub mod vector;

#[cfg(test)]
mod e2e_tests;

/// On-device GPU validation tests (feature-gated): JIT-compile each hand-written
/// PTX kernel, launch it on a real CUDA device, and assert numerical equivalence
/// to the crate's CPU references. Tests skip when no CUDA device is present.
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;
