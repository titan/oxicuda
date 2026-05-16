//! Linear-chain Conditional Random Fields (CRF).

pub mod crf_train;
pub mod linear_chain_crf;
pub mod viterbi_decode;

pub use crf_train::{LbfgsConfig, crf_log_likelihood_and_gradient, train_crf_lbfgs};
pub use linear_chain_crf::LinearChainCrf;
pub use viterbi_decode::viterbi_decode;
