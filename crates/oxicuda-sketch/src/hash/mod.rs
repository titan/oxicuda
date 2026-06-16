//! Hash families used by streaming sketches.
//!
//! Provides Murmur3-32, FNV-1a 64, simplified xxHash3, 2-universal hash family,
//! and tabulation hashing.

pub mod fnv64;
pub mod fourwise;
pub mod murmur3;
pub mod tabulation;
pub mod twouniv;
pub mod universal;
pub mod xxh3_min;

pub use fnv64::fnv1a_64;
pub use fourwise::FourWiseHash;
pub use murmur3::{murmur3_32, murmur3_32_bytes};
pub use tabulation::TabulationHash;
pub use twouniv::TwoUniversal;
pub use universal::hash_u64_to_range;
pub use xxh3_min::xxh3_64;
