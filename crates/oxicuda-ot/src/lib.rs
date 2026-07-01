//! `oxicuda-ot` — Optimal Transport primitives for OxiCUDA.
//!
//! Pure-Rust CPU-side simulation library covering the canonical Optimal Transport
//! algorithm spectrum: entropic OT (Sinkhorn-Knopp), exact OT (network simplex),
//! Wasserstein-1/2 distances, sliced and max-sliced approximations,
//! Gromov-Wasserstein and Fused-GW for unaligned domains, KL-relaxed Unbalanced OT,
//! Wasserstein barycenters (free and fixed support), JKO gradient flow,
//! Schrödinger Bridge / IPF, multi-marginal OT, Wasserstein k-means clustering,
//! and OT-based domain adaptation. Each domain module is paired with PTX kernels
//! emitted at runtime for SM 7.5 through SM 10.0.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-ot
//! ├── sinkhorn/     — Sinkhorn-Knopp (log-stab), divergence, low-level steps
//! ├── exact/        — Network simplex, EMD-1D
//! ├── wasserstein/  — W1, W2, Sliced, Max-Sliced
//! ├── gromov/       — Entropic Gromov-Wasserstein, Fused-GW
//! ├── unbalanced/   — KL-relaxed marginals
//! ├── barycenter/   — Free-support and fixed-support Wasserstein barycenters
//! ├── jko/          — Jordan-Kinderlehrer-Otto proximal gradient flow
//! ├── bridge/       — Schrödinger Bridge / IPF
//! ├── multi/        — Multi-marginal optimal transport
//! ├── clustering/   — Wasserstein k-means
//! ├── domain/       — OT-based domain adaptation (barycentric mapping)
//! ├── metrics/      — Diagnostics: marginal violation, KL, transport cost, entropy
//! ├── handle        — SmVersion, LcgRng, OtHandle
//! ├── error         — OtError / OtResult
//! └── ptx_kernels   — 7 GPU PTX kernel strings × 6 SM versions
//! ```

pub mod barycenter;
pub mod bridge;
pub mod clustering;
pub mod domain;
pub mod error;
pub mod exact;
pub mod gromov;
pub mod handle;
pub mod jko;
pub mod metrics;
pub mod multi;
pub mod ptx_kernels;
pub mod sinkhorn;
pub mod unbalanced;
pub mod wasserstein;

#[cfg(test)]
mod e2e_tests;

/// On-device GPU validation tests (feature-gated): JIT-compile each hand-written
/// PTX kernel, launch it on a real CUDA device, and assert numerical equivalence
/// to the matching CPU reference. Compiled only under `--features gpu-tests` and
/// only in test builds; every test skips gracefully if no GPU is available.
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;
