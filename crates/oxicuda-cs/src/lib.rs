//! `oxicuda-cs` — Compressed Sensing, Sparse Recovery, and Low-Rank Matrix Completion for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-cs
//! ├── greedy/             — OMP, StOMP, ROMP, CoSaMP, Subspace Pursuit
//! ├── thresholding/       — IHT, NIHT, HTP, Accelerated IHT
//! ├── amp/                — AMP, VAMP, Empirical-Bayes AMP
//! ├── basis_pursuit/      — Basis Pursuit (ADMM), BPDN, Dantzig Selector
//! ├── lasso/              — Coord descent (Friedman et al.), LARS, FISTA-LASSO,
//! │                         group/fused LASSO, Elastic Net
//! ├── tv/                 — 1D/2D Chambolle Total Variation denoising
//! ├── matrix_completion/  — SVT, Nuclear-norm minimisation, ADMM matrix completion
//! ├── robust_pca/         — Principal Component Pursuit (PCP), GoDec
//! ├── sparse_pca/         — Witten-Tibshirani-Hastie penalised matrix decomposition
//! ├── sbl/                — Sparse Bayesian Learning, Fast Marginal Likelihood
//! ├── dictionary/         — K-SVD, MOD, Online dictionary learning
//! ├── measurement/        — Gaussian, Bernoulli, Partial Fourier matrices, RIP estimator
//! ├── linalg/             — Private: Jacobi SVD, Householder QR, Cholesky, LSQR, normal equations
//! └── metrics/            — sparsity, recovery error, support recovery rate, MSE, PSNR, SNR
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_range_contains)]

pub mod amp;
pub mod basis_pursuit;
pub mod dictionary;
pub mod error;
pub mod greedy;
pub mod handle;
pub mod lasso;
pub mod linalg;
pub mod matrix_completion;
pub mod measurement;
pub mod metrics;
pub mod ptx_kernels;
pub mod robust_pca;
pub mod sbl;
pub mod sparse_pca;
pub mod thresholding;
pub mod tv;

pub use error::{CsError, CsResult};
pub use handle::{CsHandle, LcgRng, SmVersion};

#[cfg(test)]
mod e2e_tests;
