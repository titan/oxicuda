//! Architecture performance predictors for hardware-aware neural architecture search.
//!
//! Predictors take an architecture description (operation list, channel widths,
//! spatial dims) and return cheap proxy estimates of:
//! - **Latency** (LUT- or MLP-based) — surrogate for measured wall-clock time
//! - **Accuracy** (kNN / RBF) — surrogate for held-out validation accuracy
//! - **FLOPs** (analytic) — exact arithmetic cost, used as an HW-agnostic proxy
//!
//! Predictors enable BRP-NAS / FBNetV3-style multi-objective search where every
//! candidate architecture is scored without an expensive train-then-evaluate
//! cycle.
//!
//! # Modules
//!
//! - [`flops`] — analytic FLOP / parameter accountant per `OpKind`
//! - [`latency`] — LUT and small-MLP latency surrogates
//! - [`accuracy`] — kNN / RBF accuracy regressor over architecture features
//! - [`bayesian_gp`] — Gaussian-Process accuracy regressor with closed-form
//!   posterior mean + variance (Cholesky solve) and an Expected-Improvement /
//!   UCB acquisition function for sample-efficient Bayesian-optimisation search
//! - [`gnn_predictor`] — BANANAS path-encoded MLP and NPENAS message-passing
//!   GNN accuracy predictors over the cell DAG
//! - [`predictor_io`] — feature extraction shared by all predictors

pub mod accuracy;
pub mod bayesian_gp;
pub mod flops;
pub mod gnn_predictor;
pub mod latency;
pub mod predictor_io;
