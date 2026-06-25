//! General Markov Random Fields (graph + Ising), Gibbs sampler, loopy belief propagation.

pub mod belief_prop;
pub mod gibbs;
pub mod junction_tree;
pub mod mrf;
pub mod swendsen_wang;
pub mod wolff;

pub use belief_prop::{BpConfig, BpResult, loopy_bp_map, loopy_bp_marginals};
pub use gibbs::{GibbsConfig, ising_gibbs};
pub use junction_tree::{Clique, JunctionTree, JunctionTreeConfig};
pub use mrf::{IsingModel, Mrf};
pub use swendsen_wang::{SwendsenWang, SwendsenWangConfig};
pub use wolff::{Wolff, WolffConfig};
