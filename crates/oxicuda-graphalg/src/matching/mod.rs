//! Graph matching algorithms.

pub mod blossom_v_simple;
pub mod hopcroft_karp;
pub mod hungarian_munkres;
pub mod weighted_general;

pub use blossom_v_simple::blossom_match_unweighted;
pub use hopcroft_karp::hopcroft_karp_matching;
pub use hungarian_munkres::hungarian_assignment;
pub use weighted_general::{WeightedGeneralMatching, WeightedMatchingResult};
