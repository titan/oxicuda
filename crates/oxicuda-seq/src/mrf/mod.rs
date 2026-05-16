//! General Markov Random Fields (graph + Ising), Gibbs sampler, loopy belief propagation.

pub mod belief_prop;
pub mod gibbs;
pub mod mrf;

pub use belief_prop::{BpConfig, BpResult, loopy_bp_map, loopy_bp_marginals};
pub use gibbs::{GibbsConfig, ising_gibbs};
pub use mrf::{IsingModel, Mrf};
