//! Hidden Markov Models: discrete + Gaussian emissions, forward-backward,
//! Viterbi decoding, and Baum-Welch (EM) parameter learning.

pub mod baum_welch;
pub mod forward_backward;
pub mod hmm;
pub mod viterbi;

pub use baum_welch::{BaumWelchResult, baum_welch_discrete};
pub use forward_backward::{ForwardBackward, forward_backward};
pub use hmm::{HmmDiscrete, HmmGaussian, log_safe};
pub use viterbi::{ViterbiResult, viterbi};
