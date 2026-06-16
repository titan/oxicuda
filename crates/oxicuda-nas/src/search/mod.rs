//! Differentiable and hardware-aware NAS search primitives.
//!
//! - [`darts_ops`] — `DartsMixedOp`: softmax-weighted mixture of candidate
//!   operations with architecture-weight gradient updates (Liu 2018 DARTS).
//! - [`latency_predictor`] — `LatencyPredictor` + `train_latency_predictor`:
//!   linear regression over MBConv-spec features for hardware-aware cost
//!   estimation.
//! - [`local_search`] — `LocalSearchNas`: best-improvement hill-climbing over
//!   single-op architecture perturbations (White 2021).
//! - [`successive_halving`] — `SuccessiveHalving` / `Hyperband`: multi-fidelity
//!   resource-allocation search (Jamieson 2016 / Li 2017).

pub mod darts_ops;
pub mod latency_predictor;
pub mod local_search;
pub mod successive_halving;

pub use darts_ops::{DartsConfig, DartsMixedOp};
pub use latency_predictor::{LatencyPredictor, latency_features, train_latency_predictor};
pub use local_search::{
    ArchSpace, LocalSearchConfig, LocalSearchNas, SearchResult, single_op_neighbors,
};
pub use successive_halving::{
    BracketResult, Hyperband, HyperbandConfig, HyperbandResult, RoundInfo, ShaConfig, ShaResult,
    SuccessiveHalving,
};
