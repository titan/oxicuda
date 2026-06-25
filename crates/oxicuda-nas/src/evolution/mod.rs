//! Evolutionary neural architecture search (NSGA-II and friends).

pub mod encoding;
pub mod nas_bench;
pub mod nsga2;
pub mod population;
pub mod regularized_evolution;

pub use nas_bench::{NasBenchCache, TrialResult, arch_key, arch_rng, derive_arch_seed};
pub use regularized_evolution::{RegEvoConfig, RegEvoResult, RegularizedEvolution};
