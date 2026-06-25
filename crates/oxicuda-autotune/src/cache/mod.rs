//! Persistent, capacity-bounded caches for autotuning results.
//!
//! This module groups *on-disk caches* that key tuning results by the full
//! identity of the situation that produced them — the kernel source, the GPU
//! architecture, and the CUDA driver version — so a cached configuration is
//! only reused when *all three* match.  This is deliberately distinct from the
//! [`export`](crate::export) path, which is an export-/import-only bundle
//! format for *sharing* results across machines and performs no eviction,
//! recency tracking, or schema migration of its own.
//!
//! - [`persistent_cache`] — a least-recently-used (LRU) on-disk cache keyed by
//!   `(kernel-hash, GPU-arch, CUDA-driver-version)` with a versioned schema and
//!   forward migration from a legacy bare-map layout; [`PersistentTuneCache`].
//!
//! The cache is pure-Rust and depends only on `serde`/`serde_json`, matching
//! the rest of the crate.

pub mod persistent_cache;

pub use persistent_cache::{
    CacheKey, CacheSchemaVersion, CacheStats, PersistentTuneCache, kernel_source_hash,
};
