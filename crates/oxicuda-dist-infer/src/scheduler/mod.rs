//! Request schedulers for distributed serving.
//!
//! These are the host-side *admission & batching policies* that decide which
//! in-flight requests are advanced on each iteration of the serving loop. They
//! are pure data-structure / accounting logic with exact oracles — no GPU
//! kernels.
//!
//! | Module | Scheduler |
//! |--------|-----------|
//! | [`continuous_batch`] | [`continuous_batch::ContinuousBatcher`] — Orca-style iteration-level (continuous) batching with a paged-KV block budget and admission control |
//! | [`disagg_pd`] | [`disagg_pd::DisaggPdScheduler`] — DistServe-style disaggregated prefill/decode worker pools with KV-cache hand-off |
//! | [`rebalance`] | [`rebalance::RebalanceMonitor`] — autonomous cache/MoE load-imbalance trigger that synthesises conservation-checked migration plans |
//! | [`elastic`] | [`elastic::ElasticScaler`] — host-side add/remove-a-rank planner that recomputes the TP×SP×EP grid and redistributes cache + experts |
//!
//! # References
//! - Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based
//!   Generative Models." OSDI — iteration-level scheduling.
//! - Zhong et al. (2024) "DistServe: Disaggregating Prefill and Decoding for
//!   Goodput-optimized Large Language Model Serving." OSDI/SOSP.

pub mod continuous_batch;
pub mod disagg_pd;
pub mod elastic;
pub mod rebalance;

pub use continuous_batch::{BatchPlan, ContinuousBatcher, SeqState};
pub use disagg_pd::{DisaggPdScheduler, PdPhase, PdStats, PrefillHandoff};
pub use elastic::{ElasticAxis, ElasticPlan, ElasticScaler, ExpertMove};
pub use rebalance::{MigrationMove, MigrationPlan, RebalanceMonitor};
