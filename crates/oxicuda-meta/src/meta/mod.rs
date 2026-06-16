//! Meta-learning algorithm implementations (non-MAML variants).
//!
//! This module contains standalone meta-learner structs with explicit parameter
//! ownership, complementing the closure-based APIs in [`crate::maml`].

pub mod meta_sgd_learner;

pub use meta_sgd_learner::{MetaSgdLearner, MetaSgdLearnerConfig};
