//! `oxicuda-evol` — Evolutionary & Genetic Algorithms for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-evol
//! ├── genetic/        — Canonical GA: individuals, population, selection, crossover, mutation
//! ├── evolution/
//! │   ├── cmaes/      — CMA-ES: full covariance matrix adaptation evolution strategy
//! │   └── de/         — Differential Evolution: DE/rand/1, DE/best/1, jDE adaptive
//! ├── multiobjective/ — NSGA-II (fast non-dominated sort + crowding), MOEA/D (Tchebycheff)
//! ├── neuroevolution/ — NEAT: topology evolution, innovation tracking, speciation
//! ├── swarm/          — PSO (inertia weight), ACO (Elitist, TSP)
//! └── metrics/        — Hypervolume (2D), IGD, GD, spacing, Pareto front extraction
//! ```

pub mod error;
pub mod evolution;
pub mod genetic;
pub mod handle;
pub mod metrics;
pub mod multiobjective;
pub mod neuroevolution;
pub mod ptx_kernels;
pub mod swarm;

pub use error::{EvolError, EvolResult};

#[cfg(test)]
mod e2e_tests;
