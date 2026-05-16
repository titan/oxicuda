//! Similarity sketches: MinHash, SimHash, Weighted MinHash.

pub mod minhash;
pub mod simhash;
pub mod weighted_minhash;

pub use minhash::MinHash;
pub use simhash::SimHash;
pub use weighted_minhash::WeightedMinHash;
