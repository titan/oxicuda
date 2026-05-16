//! `oxicuda-survival` — Survival analysis & time-to-event modelling for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-survival
//! ├── data/            — Observation, Dataset, RiskSet primitives
//! ├── nonparametric/   — Kaplan-Meier, Nelson-Aalen, life table, S(t) curves
//! ├── test/            — Log-rank, stratified log-rank, Peto-Peto, Gehan-Breslow
//! ├── cox/             — Cox PH (Breslow/Efron ties, Newton-Raphson, Schoenfeld)
//! ├── aft/             — Parametric AFT (Exp, Weibull, log-normal, log-logistic, GG)
//! ├── time_varying/    — Counting-process Cox with time-varying covariates
//! ├── competing/       — Cumulative incidence, cause-specific Cox, Fine-Gray
//! ├── rmst/            — Restricted mean survival time
//! ├── concordance/     — Harrell's C, Uno's C
//! ├── calibration/     — Brier, IPCW Brier, integrated Brier, time-dependent AUC
//! ├── deep/            — DeepSurv head, Cox partial-likelihood gradient, loss callables
//! ├── special/         — gammaln, digamma
//! ├── linalg/          — Cholesky, Gauss-Jordan inverse, matmul (crate-private)
//! └── metrics/         — Median survival, RMST, S(τ) summaries
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod aft;
pub mod calibration;
pub mod competing;
pub mod concordance;
pub mod cox;
pub mod data;
pub mod deep;
pub mod error;
pub mod handle;
pub mod linalg;
pub mod metrics;
pub mod nonparametric;
pub mod ptx_kernels;
pub mod rmst;
pub mod special;
pub mod test;
pub mod time_varying;

pub use error::{SurvivalError, SurvivalResult};
pub use handle::{LcgRng, SmVersion, SurvivalHandle};

#[cfg(test)]
mod e2e_tests;
