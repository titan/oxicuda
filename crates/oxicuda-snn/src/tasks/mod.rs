//! Benchmark task suites that drive the crate's real dynamics end-to-end and
//! *measure* the textbook properties they should exhibit.
//!
//! * [`reservoir_tasks`] — NARMA-10 and linear memory-capacity benchmarks for
//!   the Liquid State Machine reservoir + ridge readout.
//! * [`stdp_protocols`] — controlled spike-pair / pairing protocols that
//!   verify the pair- and triplet-STDP learning rules (sign/shape of the STDP
//!   window, Poisson-statistics convergence, triplet rate dependence).

/// NARMA-10 and memory-capacity reservoir benchmarks.
pub mod reservoir_tasks;
/// Pair- and triplet-STDP verification protocols.
pub mod stdp_protocols;

pub use reservoir_tasks::{
    LsmTaskConfig, MemoryCapacityResult, Narma10Result, memory_capacity, narma10_lsm_nmse,
    narma10_sequence, nmse, squared_correlation,
};
pub use stdp_protocols::{pair_stdp_poisson_final_weight, pair_stdp_window, triplet_pairing_dw};
