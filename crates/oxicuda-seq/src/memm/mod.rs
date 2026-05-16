//! Maximum-Entropy Markov Models (MEMM): per-state softmax classifiers over
//! the previous label plus features.

pub mod memm;

pub use memm::Memm;
