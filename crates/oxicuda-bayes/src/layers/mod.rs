//! Bayesian layer implementations.

pub mod bayes_conv;
pub mod bayes_gru;
pub mod bayes_linear;
pub mod bayes_lstm;
pub mod flipout;

pub use bayes_gru::{BayesGru, BayesGruConfig, BayesGruState, BayesGruWeights};
pub use bayes_lstm::{BayesLstm, BayesLstmConfig, BayesLstmSampledWeights, BayesLstmWeights};
