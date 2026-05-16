//! `oxicuda-manifold` — Manifold Learning, Dimensionality Reduction, and Riemannian Geometry.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-manifold
//! ├── linear/      — PCA, Kernel PCA, FastICA
//! ├── tsne/        — t-SNE (perplexity, Barnes-Hut, gradient descent)
//! ├── umap/        — UMAP (kNN graph, fuzzy simplicial set, SGD embedding)
//! ├── local/       — LLE, MLLE, Isomap, Laplacian Eigenmaps
//! ├── diffusion/   — Diffusion Maps (Coifman-Lafon)
//! ├── mds/         — Classical MDS and SMACOF stress majorisation
//! ├── neighbor/    — Brute-force kNN, KD-tree, ball tree
//! ├── linalg/      — Jacobi eigendecomp, power iteration, Lanczos, Householder QR
//! ├── riemannian/  — Stiefel, Grassmann, SPD, Poincaré ball
//! ├── optim/       — Riemannian SGD with retractions
//! └── metrics/     — Trustworthiness, continuity, KL, neighbourhood preservation
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod diffusion;
pub mod error;
pub mod handle;
pub mod linalg;
pub mod linear;
pub mod local;
pub mod mds;
pub mod metrics;
pub mod neighbor;
pub mod optim;
pub mod ptx_kernels;
pub mod riemannian;
pub mod tsne;
pub mod umap;

pub use error::{ManifoldError, ManifoldResult};
pub use handle::{LcgRng, ManifoldHandle, SmVersion};

#[cfg(test)]
mod e2e_tests;
