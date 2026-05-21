//! Evolutionary neural architecture search (NSGA-II and friends).

pub mod encoding;
pub mod nsga2;
pub mod population;
pub mod regularized_evolution;

pub use regularized_evolution::{RegEvoConfig, RegEvoResult, RegularizedEvolution};
