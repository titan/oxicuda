//! Data sanitisation primitives (syntactic anonymity models).
//!
//! Unlike differential privacy, syntactic models such as k-anonymity provide
//! guarantees about the *released table* rather than about a randomised
//! algorithm. They are included here as a complementary, non-randomised
//! anonymisation toolkit. See [`suppression`] for the k-anonymity
//! generalisation/suppression algorithm of Sweeney (2002).

pub mod suppression;

pub use suppression::{KAnonymiseSuppressor, SuppressionReport};
