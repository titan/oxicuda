//! `oxicuda-sketch` — Streaming data sketches and sublinear algorithms for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-sketch
//! ├── hash/         — Hash families: Murmur3, FNV-1a, xxHash3-min, 2-universal, tabulation
//! ├── cardinality/  — HyperLogLog, HyperLogLog++, Linear Counting
//! ├── frequency/    — Count-Min Sketch, Count Sketch, Conservative-Update CM
//! ├── membership/   — Bloom filter, Counting Bloom, Cuckoo filter
//! ├── quantile/     — KLL, t-Digest, Greenwald-Khanna, P-square
//! ├── topk/         — Misra-Gries, Space-Saving, Frequent
//! ├── similarity/   — MinHash (Jaccard), SimHash (cosine), Weighted MinHash
//! ├── lsh/          — Cosine LSH, Jaccard LSH (banded MinHash), generic LSH index
//! ├── sampling/     — Reservoir (Vitter), Weighted Reservoir (Efraimidis-Spirakis), Bernoulli, Priority
//! ├── moment/       — AMS L2 sketch, Johnson-Lindenstrauss, Lp-norm via stable projections
//! ├── stream/       — Welford online mean/var, exponential decay, sliding window count
//! └── metrics/      — Relative error, MAE, accuracy, recall-at-k
//! ```
//!
//! All algorithms are implemented in pure Rust with no external dependencies beyond `thiserror`.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod cardinality;
pub mod error;
pub mod frequency;
pub mod handle;
pub mod hash;
pub mod lsh;
pub mod matrix;
pub mod membership;
pub mod metrics;
pub mod moment;
pub mod ptx_kernels;
pub mod quantile;
pub mod sampling;
pub mod similarity;
pub mod stream;
pub mod topk;

pub use error::{SketchError, SketchResult};
pub use handle::{LcgRng, SketchHandle, SmVersion};

#[cfg(test)]
mod e2e_tests;
