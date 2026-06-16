//! Hidden Markov Models: discrete + Gaussian emissions, forward-backward,
//! Viterbi decoding, and Baum-Welch (EM) parameter learning.
//! Also includes Variational Bayes EM (Dirichlet priors) and Hidden
//! Semi-Markov Models with explicit duration distributions.

pub mod baum_welch;
pub mod forward_backward;
pub mod hmm;
pub mod posterior_decoding;
pub mod scaling;
pub mod semimarkov;
pub mod variational;
pub mod viterbi;

pub use baum_welch::{BaumWelchResult, baum_welch_discrete};
pub use forward_backward::{ForwardBackward, forward_backward};
pub use hmm::{HmmDiscrete, HmmGaussian, log_safe};
pub use posterior_decoding::{PosteriorDecode, posterior_decode, posterior_path_is_feasible};
pub use scaling::{
    ScaledBackwardResult, ScaledForwardBackwardResult, ScaledForwardResult, scaled_backward,
    scaled_baum_welch_step, scaled_forward, scaled_forward_backward, scaled_viterbi,
};
pub use semimarkov::{DurationDistrib, HsmConfig, HsmResult, Hsmm, hsm_fit};
pub use variational::{VbHmmConfig, VbHmmResult, digamma, log_gamma, variational_hmm};
pub use viterbi::{ViterbiResult, viterbi};
