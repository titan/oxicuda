//! Canonical Genetic Algorithm primitives.
//!
//! # Components
//! - [`individual`] — `Individual` type (genome + fitness)
//! - [`population`] — `Population` management
//! - [`selection`] — tournament, roulette, rank selection
//! - [`crossover`] — one-point, two-point, uniform, SBX
//! - [`mutation`] — Gaussian, polynomial, swap

pub mod crossover;
pub mod individual;
pub mod mutation;
pub mod population;
pub mod selection;
