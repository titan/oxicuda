//! Similarity sketches: MinHash, SimHash, Weighted MinHash, KMV.

pub mod kmv;
pub mod minhash;
pub mod simhash;
pub mod weighted_minhash;

pub use kmv::KmvSketch;
pub use minhash::MinHash;
pub use simhash::SimHash;
pub use weighted_minhash::WeightedMinHash;
