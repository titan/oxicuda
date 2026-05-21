//! Canonical Genetic Algorithm primitives.
//!
//! # Components
//! - [`individual`] — `Individual` type (genome + fitness)
//! - [`population`] — `Population` management
//! - [`selection`] — tournament, roulette, rank selection
//! - [`crossover`] — one-point, two-point, uniform, SBX
//! - [`mutation`] — Gaussian, polynomial, swap

pub mod constraints;
pub mod crossover;
pub mod encoding;
pub mod individual;
pub mod mutation;
pub mod parallel;
pub mod population;
pub mod selection;

// ─── Constraint handling re-exports ─────────────────────────────────────────
pub use constraints::{
    AdaptivePenaltyState, Constraint, ConstraintFn, ConstraintKind, ConstraintResult,
    ConstraintViolation, EpsilonState, adaptive_penalty_obj, constrained_tournament_select,
    deb_feasible_better, epsilon_feasible, evaluate_constraints, static_penalty, total_violation,
    update_epsilon, update_penalty,
};
