//! `oxicuda-nas` — Neural Architecture Search primitives for OxiCUDA.
//!
//! Pure-Rust implementation of differentiable (DARTS), evolutionary (NSGA-II),
//! and one-shot (supernet) neural architecture search building blocks suitable
//! for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-nas
//! ├── darts/          — DartsCell/Network, BilevelOptimizer, derive discrete arch
//! ├── evolution/      — ArchEncoding, NSGA-II selection, Population
//! ├── ops/            — Primitives (8 DARTS ops), MixedOp, SearchSpace
//! ├── supernet/       — Weight-shared Supernet, PathSampler, SlimmableNet
//! ├── error           — NasError / NasResult
//! ├── handle          — NasHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod darts;
pub mod error;
pub mod evolution;
pub mod handle;
pub mod ops;
pub mod ptx_kernels;
pub mod supernet;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common neural architecture search types.
pub mod prelude {
    pub use crate::darts::bilevel::{BilevelConfig, BilevelOptimizer};
    pub use crate::darts::cell::DartsCell;
    pub use crate::darts::derive::{
        DiscretizedCell, DiscretizedNetwork, derive_discrete_cell, derive_network,
    };
    pub use crate::darts::network::DartsNetwork;
    pub use crate::error::{NasError, NasResult};
    pub use crate::evolution::encoding::ArchEncoding;
    pub use crate::evolution::nsga2::{
        Individual, crowding_distance, fast_non_dominated_sort, nsga2_select, tournament_select,
    };
    pub use crate::evolution::population::Population;
    pub use crate::handle::{LcgRng, NasHandle, SmVersion};
    pub use crate::ops::mixed_op::MixedOp;
    pub use crate::ops::primitives::{OpKind, OpWeights};
    pub use crate::ops::search_space::{CellSpace, NetworkSpace, SearchSpace};
    pub use crate::ptx_kernels::{
        arch_grad_ptx, arch_softmax_ptx, crossover_uniform_ptx, f32_hex, flops_accumulate_ptx,
        gumbel_softmax_ptx, mixed_op_blend_ptx, pareto_dominate_ptx,
    };
    pub use crate::supernet::path_sample::{PathSampler, SamplingStrategy};
    pub use crate::supernet::slimmable::{BnStats, SlimmableNet, WIDTH_MULTIPLIERS};
    pub use crate::supernet::weight_share::Supernet;
}
