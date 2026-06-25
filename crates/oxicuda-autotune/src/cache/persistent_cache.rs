//! On-disk LRU tuning cache keyed by `(kernel-hash, GPU-arch, driver-version)`.
//!
//! [`PersistentTuneCache`] is a bounded, least-recently-used (LRU) cache that
//! maps a [`CacheKey`] — the triple of *which kernel* (a content hash of its
//! source), *which GPU architecture* (e.g. `"sm_90"`), and *which CUDA driver
//! version* (e.g. `"12.4"`) — to the best [`BenchmarkResult`] measured for that
//! exact combination.  Because all three identity components are part of the
//! key, a cached configuration is only ever reused when the kernel, the
//! hardware *and* the driver all match; a driver upgrade or an architecture
//! change naturally produces a different key and therefore a cache miss.
//!
//! # Why a separate cache (vs. [`ResultDb`](crate::result_db::ResultDb) / [`export`](crate::export))
//!
//! [`ResultDb`](crate::result_db::ResultDb) is keyed by `(GPU-name, kernel-name,
//! problem-size)` and grows without bound; [`export`](crate::export) is a
//! share-only bundle format.  This cache is the *runtime working set*: it is
//! **capacity-bounded** and **evicts the least-recently-used entry** when full,
//! it tracks **recency** so hot configurations survive, and it carries a
//! **versioned on-disk schema** with forward migration distinct from
//! [`db_migration`](crate::db_migration).  Keying on the driver version is the
//! distinguishing feature: it lets a system re-tune transparently after a CUDA
//! toolkit upgrade without serving stale, potentially-slower configs.
//!
//! # On-disk format
//!
//! The cache serialises to JSON as a [`CacheSchemaVersion`]-stamped envelope:
//!
//! ```json
//! {
//!   "schema": { "major": 1, "minor": 0 },
//!   "capacity": 256,
//!   "seq": 1234,
//!   "entries": [ { "key": {..}, "result": {..}, "last_access": 1233, "inserted": 12 }, ... ]
//! }
//! ```
//!
//! A *legacy* file that is a bare JSON array of `{key,result}` objects (no
//! envelope) is recognised on load and migrated in place; see
//! [`PersistentTuneCache::load_at`].
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::cache::persistent_cache::{CacheKey, PersistentTuneCache};
//! use oxicuda_autotune::{BenchmarkResult, Config};
//!
//! # fn make_result(us: f64) -> BenchmarkResult {
//! #     BenchmarkResult { config: Config::new(), median_us: us, min_us: us,
//! #         max_us: us, stddev_us: 0.0, gflops: None, efficiency: None }
//! # }
//! let mut cache = PersistentTuneCache::new(2);
//! let k = CacheKey::from_source("__global__ void g(){}", "sm_90", "12.4");
//! cache.put(k.clone(), make_result(42.0));
//! assert!(cache.get(&k).is_some());
//! assert_eq!(cache.stats().hits, 1);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// CacheSchemaVersion
// ---------------------------------------------------------------------------

/// Semantic version of the persistent-cache on-disk schema.
///
/// Compatibility is by **major** version: a file whose major number differs
/// from [`CacheSchemaVersion::CURRENT`] cannot be parsed by this build and is
/// treated as an unrecognised (legacy) layout to be migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheSchemaVersion {
    /// Major version — incompatible layout changes.
    pub major: u32,
    /// Minor version — backwards-compatible additions.
    pub minor: u32,
}

impl CacheSchemaVersion {
    /// The current schema version written by this build.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    /// Returns `true` when `self` is forward-compatible with `other`
    /// (same major number).
    #[must_use]
    pub fn is_compatible_with(&self, other: &CacheSchemaVersion) -> bool {
        self.major == other.major
    }
}

impl std::fmt::Display for CacheSchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

// ---------------------------------------------------------------------------
// kernel hashing
// ---------------------------------------------------------------------------

/// Computes a stable 64-bit content hash of kernel source text using the
/// FNV-1a algorithm.
///
/// The hash is deterministic and platform-independent (unlike
/// [`std::hash::DefaultHasher`], which is intentionally unspecified), so a
/// cache file written on one machine identifies the same kernel on another.
/// Identical source produces an identical hash; a single-byte change produces a
/// different one.
#[must_use]
pub fn kernel_source_hash(source: &str) -> u64 {
    // FNV-1a 64-bit: offset basis and prime per Fowler–Noll–Vo.
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// The full identity under which a tuning result is cached.
///
/// Two results are interchangeable only when their kernel content, GPU
/// architecture, and CUDA driver version all coincide.  The architecture and
/// driver strings are stored verbatim (e.g. `"sm_90"`, `"12.4"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Content hash of the kernel source (see [`kernel_source_hash`]).
    pub kernel_hash: u64,
    /// GPU architecture / compute-capability tag, e.g. `"sm_80"`, `"sm_90"`.
    pub gpu_arch: String,
    /// CUDA driver version string, e.g. `"12.4"`, `"11.8"`.
    pub driver_version: String,
}

impl CacheKey {
    /// Builds a key from a pre-computed kernel hash.
    #[must_use]
    pub fn new(
        kernel_hash: u64,
        gpu_arch: impl Into<String>,
        driver_version: impl Into<String>,
    ) -> Self {
        Self {
            kernel_hash,
            gpu_arch: gpu_arch.into(),
            driver_version: driver_version.into(),
        }
    }

    /// Builds a key by hashing the kernel `source` with [`kernel_source_hash`].
    #[must_use]
    pub fn from_source(
        source: &str,
        gpu_arch: impl Into<String>,
        driver_version: impl Into<String>,
    ) -> Self {
        Self::new(kernel_source_hash(source), gpu_arch, driver_version)
    }
}

// ---------------------------------------------------------------------------
// CacheEntry (on-disk + in-memory)
// ---------------------------------------------------------------------------

/// One stored result together with its recency bookkeeping.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// The identity this entry was cached under.
    key: CacheKey,
    /// The best measured result for `key`.
    result: BenchmarkResult,
    /// Logical clock value at the most recent access (get/put).  Higher means
    /// more recently used; the lowest is evicted first.
    last_access: u64,
    /// Logical clock value at first insertion (for diagnostics / tie-breaks).
    inserted: u64,
}

/// Legacy (pre-envelope) on-disk entry: just a key and a result.
#[derive(Debug, Deserialize)]
struct LegacyEntry {
    key: CacheKey,
    result: BenchmarkResult,
}

/// The versioned on-disk envelope.
#[derive(Debug, Serialize, Deserialize)]
struct CacheFile {
    /// Schema version of this file.
    schema: CacheSchemaVersion,
    /// Capacity the cache was created with.
    capacity: usize,
    /// Logical clock at the time of writing.
    seq: u64,
    /// All live entries.
    entries: Vec<CacheEntry>,
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Runtime hit/miss/eviction counters for a [`PersistentTuneCache`].
///
/// These counters are in-memory only (not persisted) and reset whenever a
/// cache is constructed or loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of [`get`](PersistentTuneCache::get) calls that found an entry.
    pub hits: u64,
    /// Number of [`get`](PersistentTuneCache::get) calls that found nothing.
    pub misses: u64,
    /// Number of entries evicted to respect the capacity bound.
    pub evictions: u64,
}

impl CacheStats {
    /// Hit rate in `[0, 1]`; returns `0.0` when there were no lookups.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// PersistentTuneCache
// ---------------------------------------------------------------------------

/// A capacity-bounded LRU cache of tuning results, persistable to JSON.
///
/// Insertion past the capacity evicts the least-recently-used entry.  A
/// successful [`get`](Self::get) counts as a use and refreshes the entry's
/// recency.  Capacity `0` is rejected at construction.
#[derive(Debug, Clone)]
pub struct PersistentTuneCache {
    /// Maximum number of live entries (always ≥ 1).
    capacity: usize,
    /// Live entries keyed by identity.
    entries: HashMap<CacheKey, CacheEntry>,
    /// Monotonic logical clock; incremented on every access for LRU ordering.
    seq: u64,
    /// In-memory hit/miss/eviction counters.
    stats: CacheStats,
}

impl PersistentTuneCache {
    /// Creates an empty cache with the given `capacity`.
    ///
    /// A `capacity` of `0` is silently raised to `1` so the cache can always
    /// hold at least one entry.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            seq: 0,
            stats: CacheStats::default(),
        }
    }

    /// Returns the configured capacity (number of entries).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of live entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the current hit/miss/eviction statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Advances and returns the logical clock.
    fn tick(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Looks up the result for `key`, refreshing its recency on a hit.
    ///
    /// Increments [`CacheStats::hits`] on success and [`CacheStats::misses`]
    /// otherwise.  Returns a reference to the stored [`BenchmarkResult`].
    pub fn get(&mut self, key: &CacheKey) -> Option<&BenchmarkResult> {
        let now = self.tick();
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.last_access = now;
                self.stats.hits += 1;
                Some(&entry.result)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Returns the stored result for `key` **without** affecting recency or
    /// statistics (a read-only peek, useful for diagnostics).
    #[must_use]
    pub fn peek(&self, key: &CacheKey) -> Option<&BenchmarkResult> {
        self.entries.get(key).map(|e| &e.result)
    }

    /// Returns `true` when `key` is present (without affecting recency).
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Inserts (or replaces) the result for `key`, evicting the
    /// least-recently-used entry first if the cache is at capacity.
    ///
    /// Replacing an existing key refreshes its recency and never triggers
    /// eviction (the live count does not grow).  Returns the result that was
    /// evicted to make room, if any.
    pub fn put(&mut self, key: CacheKey, result: BenchmarkResult) -> Option<BenchmarkResult> {
        let now = self.tick();

        if let Some(entry) = self.entries.get_mut(&key) {
            // In-place update: keep the original insertion time, bump recency.
            entry.result = result;
            entry.last_access = now;
            return None;
        }

        let mut evicted = None;
        if self.entries.len() >= self.capacity {
            if let Some(victim) = self.lru_key() {
                if let Some(old) = self.entries.remove(&victim) {
                    self.stats.evictions += 1;
                    evicted = Some(old.result);
                }
            }
        }

        self.entries.insert(
            key.clone(),
            CacheEntry {
                key,
                result,
                last_access: now,
                inserted: now,
            },
        );
        evicted
    }

    /// Removes and returns the result for `key`, if present.  Does not affect
    /// hit/miss statistics.
    pub fn remove(&mut self, key: &CacheKey) -> Option<BenchmarkResult> {
        self.entries.remove(key).map(|e| e.result)
    }

    /// Drops every entry (capacity and statistics are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the key of the least-recently-used live entry, or `None` when
    /// the cache is empty.  Ties on `last_access` are broken by the older
    /// insertion time so behaviour is deterministic.
    fn lru_key(&self) -> Option<CacheKey> {
        self.entries
            .values()
            .min_by(|a, b| {
                a.last_access
                    .cmp(&b.last_access)
                    .then(a.inserted.cmp(&b.inserted))
            })
            .map(|e| e.key.clone())
    }

    /// Returns the keys currently held, ordered most-recently-used first.
    #[must_use]
    pub fn keys_by_recency(&self) -> Vec<CacheKey> {
        let mut entries: Vec<&CacheEntry> = self.entries.values().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.last_access));
        entries.into_iter().map(|e| e.key.clone()).collect()
    }

    // -- persistence --------------------------------------------------------

    /// Serialises the cache to a JSON string in the current schema.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::SerdeError`] if serialisation fails.
    pub fn to_json(&self) -> Result<String, AutotuneError> {
        let file = CacheFile {
            schema: CacheSchemaVersion::CURRENT,
            capacity: self.capacity,
            seq: self.seq,
            entries: self.entries.values().cloned().collect(),
        };
        Ok(serde_json::to_string_pretty(&file)?)
    }

    /// Reconstructs a cache from a JSON string previously produced by
    /// [`to_json`](Self::to_json), or from a legacy bare-array layout.
    ///
    /// On a recognised current-schema envelope, the capacity, logical clock and
    /// recency ordering are restored exactly.  On a legacy bare array (no
    /// envelope), the entries are imported with synthesised recency in array
    /// order and `capacity` is set to the entry count (at least 1); pass an
    /// explicit `override_capacity` to bound it differently.
    ///
    /// Statistics always reset to zero.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::SerdeError`] if the text is neither a valid
    /// current envelope nor a valid legacy array.
    pub fn from_json(json: &str, override_capacity: Option<usize>) -> Result<Self, AutotuneError> {
        // Try the current envelope first.
        if let Ok(file) = serde_json::from_str::<CacheFile>(json) {
            if file.schema.is_compatible_with(&CacheSchemaVersion::CURRENT) {
                let capacity = override_capacity
                    .unwrap_or(file.capacity)
                    .max(file.entries.len())
                    .max(1);
                let mut entries = HashMap::with_capacity(file.entries.len());
                let mut max_seq = file.seq;
                for entry in file.entries {
                    max_seq = max_seq.max(entry.last_access).max(entry.inserted);
                    entries.insert(entry.key.clone(), entry);
                }
                return Ok(Self {
                    capacity,
                    entries,
                    seq: max_seq,
                    stats: CacheStats::default(),
                });
            }
        }

        // Fall back to the legacy bare-array layout, synthesising recency.
        let legacy: Vec<LegacyEntry> = serde_json::from_str(json)?;
        let mut cache = Self::new(override_capacity.unwrap_or(legacy.len()).max(1));
        for legacy_entry in legacy {
            let now = cache.tick();
            cache.entries.insert(
                legacy_entry.key.clone(),
                CacheEntry {
                    key: legacy_entry.key,
                    result: legacy_entry.result,
                    last_access: now,
                    inserted: now,
                },
            );
        }
        Ok(cache)
    }

    /// Writes the cache to `path` as JSON, creating parent directories.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] on a filesystem failure or
    /// [`AutotuneError::SerdeError`] on a serialisation failure.
    pub fn save_at(&self, path: impl AsRef<Path>) -> Result<(), AutotuneError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Loads a cache from `path`.
    ///
    /// A missing file yields an empty cache with the supplied
    /// `default_capacity` (so first-run callers need no special case).  A
    /// legacy bare-array file is migrated transparently (see
    /// [`from_json`](Self::from_json)).
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::IoError`] if the file exists but cannot be
    /// read, or [`AutotuneError::SerdeError`] if its contents parse as neither
    /// a current envelope nor a legacy array.
    pub fn load_at(path: impl AsRef<Path>, default_capacity: usize) -> Result<Self, AutotuneError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new(default_capacity));
        }
        let contents = std::fs::read_to_string(path)?;
        if contents.trim().is_empty() {
            return Ok(Self::new(default_capacity));
        }
        Self::from_json(&contents, Some(default_capacity.max(1)))
    }

    /// Default on-disk location for the persistent cache:
    /// `~/.cache/oxicuda/autotune/tune_cache.json`, falling back to
    /// `$TMPDIR/oxicuda_autotune/tune_cache.json`.
    #[must_use]
    pub fn default_path() -> PathBuf {
        let base = std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".cache"))
            .unwrap_or_else(|| std::env::temp_dir().join("oxicuda_autotune"));
        base.join("oxicuda")
            .join("autotune")
            .join("tune_cache.json")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// Builds a benchmark result whose median time encodes `us` (used to tell
    /// stored entries apart in assertions).
    fn result_with(us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new(),
            median_us: us,
            min_us: us,
            max_us: us,
            stddev_us: 0.0,
            gflops: None,
            efficiency: None,
        }
    }

    fn key(hash: u64) -> CacheKey {
        CacheKey::new(hash, "sm_90", "12.4")
    }

    fn temp_path(name: &str) -> PathBuf {
        let suffix = format!(
            "{}_{:?}_{}",
            std::process::id(),
            std::thread::current().id(),
            name
        );
        std::env::temp_dir()
            .join("oxicuda_persistent_cache")
            .join(suffix)
            .join("tune_cache.json")
    }

    #[test]
    fn kernel_hash_is_deterministic_and_sensitive() {
        let a = kernel_source_hash("__global__ void k(){ /* a */ }");
        let b = kernel_source_hash("__global__ void k(){ /* a */ }");
        let c = kernel_source_hash("__global__ void k(){ /* b */ }");
        assert_eq!(a, b, "identical source must hash identically");
        assert_ne!(a, c, "a one-character change must change the hash");
        // Known FNV-1a vector: the empty string maps to the offset basis.
        assert_eq!(kernel_source_hash(""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn key_distinguishes_arch_and_driver() {
        let base = CacheKey::from_source("src", "sm_80", "12.4");
        let diff_arch = CacheKey::from_source("src", "sm_90", "12.4");
        let diff_driver = CacheKey::from_source("src", "sm_80", "12.5");
        let same = CacheKey::from_source("src", "sm_80", "12.4");
        assert_ne!(base, diff_arch, "different arch => different key");
        assert_ne!(base, diff_driver, "different driver => different key");
        assert_eq!(base, same, "same triple => same key");
    }

    #[test]
    fn put_then_get_returns_stored_result() {
        let mut cache = PersistentTuneCache::new(4);
        cache.put(key(1), result_with(10.0));
        let got = cache.get(&key(1)).expect("entry present");
        assert!((got.median_us - 10.0).abs() < 1e-9);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn get_miss_counts_and_returns_none() {
        let mut cache = PersistentTuneCache::new(4);
        assert!(cache.get(&key(99)).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        assert!((cache.stats().hit_rate() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn capacity_zero_is_raised_to_one() {
        let cache = PersistentTuneCache::new(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn put_replaces_existing_without_eviction() {
        let mut cache = PersistentTuneCache::new(2);
        cache.put(key(1), result_with(10.0));
        let evicted = cache.put(key(1), result_with(20.0));
        assert!(evicted.is_none(), "replacing a key must not evict");
        assert_eq!(cache.len(), 1);
        let got = cache.peek(&key(1)).expect("present");
        assert!(
            (got.median_us - 20.0).abs() < 1e-9,
            "value updated in place"
        );
    }

    #[test]
    fn lru_eviction_removes_least_recently_used() {
        let mut cache = PersistentTuneCache::new(2);
        cache.put(key(1), result_with(1.0));
        cache.put(key(2), result_with(2.0));
        // Touch key(1) so key(2) becomes the LRU victim.
        let _ = cache.get(&key(1));
        let evicted = cache.put(key(3), result_with(3.0));
        let evicted = evicted.expect("an entry must be evicted at capacity");
        assert!(
            (evicted.median_us - 2.0).abs() < 1e-9,
            "key(2) was least-recently-used and should be evicted, got {}",
            evicted.median_us
        );
        assert!(cache.contains(&key(1)), "recently-used key survives");
        assert!(cache.contains(&key(3)), "freshly-inserted key present");
        assert!(!cache.contains(&key(2)), "LRU victim is gone");
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.len(), 2, "capacity bound respected");
    }

    #[test]
    fn eviction_chain_keeps_only_capacity_entries() {
        let mut cache = PersistentTuneCache::new(3);
        for i in 0..10u64 {
            cache.put(key(i), result_with(i as f64));
        }
        assert_eq!(cache.len(), 3, "never exceeds capacity");
        // With no intervening gets, the three most-recent insertions remain.
        assert!(cache.contains(&key(7)));
        assert!(cache.contains(&key(8)));
        assert!(cache.contains(&key(9)));
        assert!(!cache.contains(&key(0)));
        assert_eq!(cache.stats().evictions, 7);
    }

    #[test]
    fn peek_and_contains_do_not_affect_recency_or_stats() {
        let mut cache = PersistentTuneCache::new(2);
        cache.put(key(1), result_with(1.0));
        cache.put(key(2), result_with(2.0));
        // Peek key(1): must NOT save it from eviction (peek is recency-neutral).
        assert!(cache.peek(&key(1)).is_some());
        assert!(cache.contains(&key(1)));
        assert_eq!(cache.stats().hits, 0, "peek/contains do not count as hits");
        // key(1) is still the LRU (inserted before key(2), never `get`-touched).
        cache.put(key(3), result_with(3.0));
        assert!(!cache.contains(&key(1)), "peek did not refresh recency");
        assert!(cache.contains(&key(2)));
    }

    #[test]
    fn remove_and_clear() {
        let mut cache = PersistentTuneCache::new(4);
        cache.put(key(1), result_with(1.0));
        cache.put(key(2), result_with(2.0));
        let removed = cache.remove(&key(1)).expect("removed value");
        assert!((removed.median_us - 1.0).abs() < 1e-9);
        assert!(!cache.contains(&key(1)));
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn keys_by_recency_orders_mru_first() {
        let mut cache = PersistentTuneCache::new(4);
        cache.put(key(1), result_with(1.0));
        cache.put(key(2), result_with(2.0));
        cache.put(key(3), result_with(3.0));
        let _ = cache.get(&key(1)); // make key(1) most-recent
        let order = cache.keys_by_recency();
        assert_eq!(order.first().map(|k| k.kernel_hash), Some(1));
    }

    #[test]
    fn json_roundtrip_preserves_entries_and_capacity() {
        let mut cache = PersistentTuneCache::new(5);
        cache.put(key(1), result_with(11.0));
        cache.put(key(2), result_with(22.0));
        let _ = cache.get(&key(2));
        let json = cache.to_json().expect("serialise");
        let restored = PersistentTuneCache::from_json(&json, None).expect("deserialise");
        assert_eq!(restored.capacity(), 5);
        assert_eq!(restored.len(), 2);
        assert!((restored.peek(&key(1)).expect("k1").median_us - 11.0).abs() < 1e-9);
        assert!((restored.peek(&key(2)).expect("k2").median_us - 22.0).abs() < 1e-9);
        // Recency order survives: key(2) was touched last, so it is MRU.
        assert_eq!(
            restored.keys_by_recency().first().map(|k| k.kernel_hash),
            Some(2)
        );
    }

    #[test]
    fn save_and_load_roundtrip_on_disk() {
        let path = temp_path("save_load");
        let mut cache = PersistentTuneCache::new(8);
        cache.put(key(1), result_with(7.0));
        cache.put(key(2), result_with(8.0));
        cache.save_at(&path).expect("save");

        let loaded = PersistentTuneCache::load_at(&path, 8).expect("load");
        assert_eq!(loaded.len(), 2);
        assert!((loaded.peek(&key(1)).expect("k1").median_us - 7.0).abs() < 1e-9);

        let _ = std::fs::remove_dir_all(path.parent().and_then(|p| p.parent()).unwrap_or(&path));
    }

    #[test]
    fn load_missing_file_yields_empty_cache() {
        let path = temp_path("missing");
        let _ = std::fs::remove_file(&path);
        let cache = PersistentTuneCache::load_at(&path, 16).expect("load missing");
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 16);
    }

    #[test]
    fn legacy_bare_array_is_migrated() {
        // A legacy file: bare JSON array of {key,result}, no envelope.
        let legacy = serde_json::json!([
            {
                "key": { "kernel_hash": 1, "gpu_arch": "sm_80", "driver_version": "11.8" },
                "result": {
                    "config": Config::new(),
                    "median_us": 5.0, "min_us": 5.0, "max_us": 5.0,
                    "stddev_us": 0.0, "gflops": null, "efficiency": null
                }
            }
        ])
        .to_string();
        let cache = PersistentTuneCache::from_json(&legacy, Some(4)).expect("migrate legacy");
        assert_eq!(cache.len(), 1);
        let k = CacheKey::new(1, "sm_80", "11.8");
        assert!((cache.peek(&k).expect("migrated").median_us - 5.0).abs() < 1e-9);
        // Re-serialising must now produce the *current* envelope, not the array.
        let json = cache.to_json().expect("serialise");
        assert!(
            json.contains("\"schema\""),
            "migrated file gains an envelope"
        );
    }

    #[test]
    fn schema_version_compatibility() {
        let cur = CacheSchemaVersion::CURRENT;
        let same_major = CacheSchemaVersion { major: 1, minor: 9 };
        let diff_major = CacheSchemaVersion { major: 2, minor: 0 };
        assert!(cur.is_compatible_with(&same_major));
        assert!(!cur.is_compatible_with(&diff_major));
        assert_eq!(format!("{cur}"), "v1.0");
    }

    #[test]
    fn incompatible_envelope_falls_back_to_legacy_error() {
        // A future-major envelope is not a legacy array either => parse error.
        let future = serde_json::json!({
            "schema": { "major": 99, "minor": 0 },
            "capacity": 4, "seq": 0, "entries": []
        })
        .to_string();
        assert!(PersistentTuneCache::from_json(&future, None).is_err());
    }

    #[test]
    fn hit_rate_reflects_lookups() {
        let mut cache = PersistentTuneCache::new(4);
        cache.put(key(1), result_with(1.0));
        let _ = cache.get(&key(1)); // hit
        let _ = cache.get(&key(2)); // miss
        let _ = cache.get(&key(1)); // hit
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }
}
