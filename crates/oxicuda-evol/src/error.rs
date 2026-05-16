//! Error types for the oxicuda-evol crate.

use thiserror::Error;

/// All errors that can occur in evolutionary / genetic algorithm operations.
#[derive(Debug, Error)]
pub enum EvolError {
    /// A configuration parameter is outside its valid range or logically inconsistent.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Operation was attempted on an empty population.
    #[error("empty population")]
    EmptyPopulation,

    /// Vectors or matrices have incompatible dimensionalities.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Population is too small to support the requested operation.
    #[error("population size {size} too small for {op}")]
    PopulationTooSmall { size: usize, op: &'static str },

    /// Iterative algorithm failed to converge within the allowed budget.
    #[error("convergence failed after {iter} iterations")]
    ConvergenceFailed { iter: usize },

    /// Objective vectors have inconsistent lengths across individuals.
    #[error("objective count mismatch")]
    ObjectiveCountMismatch,

    /// A genome has zero genes.
    #[error("empty genome")]
    EmptyGenome,

    /// An innovation number is invalid or cannot be resolved.
    #[error("invalid innovation: {0}")]
    InvalidInnovation(String),

    /// A species index was not found in the current species list.
    #[error("species not found: {0}")]
    SpeciesNotFound(usize),

    /// Swarm has zero particles.
    #[error("swarm empty")]
    SwarmEmpty,

    /// Pheromone matrix dimensions are inconsistent with city count.
    #[error("pheromone matrix dimension mismatch")]
    PheromoneDimensionMismatch,

    /// An ant tour is shorter than the required number of cities.
    #[error("tour incomplete: {0} cities visited, {1} required")]
    TourIncomplete(usize, usize),

    /// Jacobi eigendecomposition did not converge within the sweep budget.
    #[error("eigendecomposition failed after {0} sweeps")]
    EigenFailed(usize),
}

/// Convenience alias for `Result<T, EvolError>`.
pub type EvolResult<T> = Result<T, EvolError>;
