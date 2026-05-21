//! `oxicuda-snn` — Spiking Neural Network primitives for OxiCUDA.
//!
//! Pure-Rust CPU-side simulation library covering the spiking ML stack: classical
//! neuron models (LIF, IF, Izhikevich, AdEx, Poisson), surrogate-gradient training
//! (sigmoid, atan, triangle, super-spike, fast-sigmoid), BPTT/STBP/SLAYER, pair-
//! and triplet- STDP plasticity with reward modulation, ANN→SNN conversion via
//! threshold balancing, rate/TTFS/phase encodings, spiking layers
//! (linear/conv/pool/recurrent), Liquid State Machines, and analytical metrics
//! (van Rossum, Victor-Purpura, sync index). Each domain module is paired with
//! PTX kernels emitted at runtime for SM 7.5 through SM 10.0.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-snn
//! ├── neuron/        — LIF, IF, Izhikevich, AdEx, Poisson neurons
//! ├── surrogate/     — sigmoid, atan, triangle, super-spike, fast-sigmoid grads
//! ├── training/      — BPTT, STBP, SLAYER spike-response training
//! ├── plasticity/    — STDP, R-STDP, triplet STDP
//! ├── conversion/    — ANN→SNN rate conversion, threshold balancing
//! ├── encoding/      — Rate, TTFS, phase, Poisson input encodings
//! ├── layer/         — Spiking linear, conv, pool, recurrent
//! ├── reservoir/     — Liquid State Machine
//! ├── metrics/       — Firing rate, ISI, CV, van Rossum, Victor-Purpura, sync
//! ├── handle         — SmVersion, LcgRng, SnnHandle
//! ├── error          — SnnError / SnnResult
//! └── ptx_kernels    — 7 GPU PTX kernel strings × 6 SM versions
//! ```

pub mod conversion;
pub mod encoding;
pub mod error;
pub mod handle;
pub mod layer;
pub mod metrics;
pub mod neuron;
pub mod plasticity;
pub mod ptx_kernels;
pub mod reservoir;
pub mod surrogate;
pub mod synapse;
pub mod training;

#[cfg(test)]
mod e2e_tests;
