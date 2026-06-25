//! Similarity sketches: MinHash, SimHash, Weighted MinHash, KMV, and weighted
//! bottom-k MinHash.

pub mod kmv;
pub mod minhash;
pub mod simhash;
pub mod weighted_bottom_k;
pub mod weighted_minhash;

pub use kmv::KmvSketch;
pub use minhash::MinHash;
pub use simhash::SimHash;
pub use weighted_bottom_k::{WeightedBottomK, WeightedSlot};
pub use weighted_minhash::WeightedMinHash;
