//! `oxicuda-privacy` — Differential Privacy primitives for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-privacy
//! ├── mechanism/          — Exponential, Report-Noisy-Max, Propose-Test-Release, Sampled-Gaussian
//! ├── selection/          — Sparse Vector Technique, AboveThreshold
//! ├── accounting/         — f-DP/GDP, zCDP/tCDP, CDP (mean-concentrated), PRV (numerical composition)
//! ├── composition/        — Strong composition, subsampling/shuffling/iteration amplification
//! ├── optimizer/          — DP-FTRL (tree agg), DP-Adam, DP-SGD (moments accountant)
//! ├── local/              — GRR, OUE, RAPPOR (local differential privacy)
//! ├── sensitivity/        — Local sensitivity, smooth sensitivity
//! ├── sanitisation/       — k-anonymity generalisation + suppression (Sweeney 2002)
//! └── metrics/            — Budget tracking, MSE, SNR, utility
//! ```
//!
//! This crate provides **complementary** DP primitives to `oxicuda-federated`.
//! Do not duplicate `GaussianMechanism`, `LaplacianMechanism`, `MomentsAccountant`,
//! `PateConfig`, or the RDP accountant — those live in `oxicuda-federated::privacy`.

pub mod accounting;
pub mod composition;
pub mod error;
pub mod handle;
pub mod local;
pub mod mechanism;
pub mod metrics;
pub mod noise;
pub mod optimizer;
pub mod ptx_kernels;
pub mod rng;
pub mod sanitisation;
pub mod selection;
pub mod sensitivity;

#[cfg(test)]
mod e2e_tests;
