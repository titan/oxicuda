//! Bayesian layer implementations.

pub mod bayes_conv;
pub mod bayes_gru;
pub mod bayes_linear;
pub mod flipout;

pub use bayes_gru::{BayesGru, BayesGruConfig, BayesGruState, BayesGruWeights};
