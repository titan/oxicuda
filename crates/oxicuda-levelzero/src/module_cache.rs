//! Host-side module-binary cache and EU-occupancy heuristics.
//!
//! Compiling SPIR-V to an Intel native binary (the `zeModuleCreate` JIT) is
//! expensive. Level Zero lets an application retrieve the compiled native
//! binary via `zeModuleGetNativeBinary` and re-create the module from it on a
//! later run, skipping the JIT. The *cache bookkeeping* — keying compiled
//! binaries by a stable hash of the SPIR-V input and build flags, tracking hit
//! / miss statistics — is pure host-side logic modelled here by
//! [`ModuleBinaryCache`].
//!
//! This module also provides [`EuOccupancyAdvisor`], a tile-size heuristic that
//! consumes an EU-utilisation fraction (which on hardware comes from sysman
//! `zesDeviceProcessesGetState`) and recommends a workgroup size. The heuristic
//! itself is CPU-testable; only the utilisation *reading* is device-gated.

use std::collections::HashMap;

use crate::error::{LevelZeroError, LevelZeroResult};

// ─── Stable SPIR-V hashing ───────────────────────────────────

/// 64-bit FNV-1a hash over a SPIR-V word stream and its build flags.
///
/// FNV-1a is deterministic across runs and platforms (unlike `DefaultHasher`,
/// whose seed varies), so the same SPIR-V + flags always produce the same key.
/// That stability is what makes an on-disk native-binary cache valid across
/// process invocations.
#[must_use]
pub fn spirv_cache_key(spirv: &[u32], build_flags: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &word in spirv {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    // Mix the build flags so e.g. `-ze-opt-disable` keys a distinct binary.
    for byte in build_flags.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── A cached native binary ──────────────────────────────────

/// A compiled Intel native binary retrieved from `zeModuleGetNativeBinary`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBinary {
    /// The opaque native-binary blob (driver-specific ISA + metadata).
    pub bytes: Vec<u8>,
    /// Build flags the binary was compiled with (part of its identity).
    pub build_flags: String,
}

impl NativeBinary {
    /// Wrap a native-binary blob compiled with `build_flags`.
    #[must_use]
    pub fn new(bytes: Vec<u8>, build_flags: impl Into<String>) -> Self {
        Self {
            bytes,
            build_flags: build_flags.into(),
        }
    }

    /// Size of the native binary in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// `true` when the binary is empty (an invalid/placeholder entry).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

// ─── Module-binary cache ─────────────────────────────────────

/// An in-memory cache mapping `(SPIR-V, build flags)` to compiled native
/// binaries, with hit/miss accounting.
///
/// A backend looks up a key before calling `zeModuleCreate`; on a miss it JITs
/// the SPIR-V, retrieves the native binary, and [`insert`](Self::insert)s it so
/// later identical modules skip the JIT. The cache is the host-side half of
/// `zeModuleGetNativeBinary` persistence — serialising the map to disk (a
/// trivial extension) gives cross-run reuse.
#[derive(Debug, Default)]
pub struct ModuleBinaryCache {
    entries: HashMap<u64, NativeBinary>,
    hits: u64,
    misses: u64,
}

impl ModuleBinaryCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached binaries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when nothing is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cumulative cache hits.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cumulative cache misses.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate in `[0, 1]`; `0.0` when there have been no lookups.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Look up a compiled binary for `spirv` + `build_flags`, recording a
    /// hit or miss. Returns `None` on a miss (the caller must JIT and insert).
    pub fn lookup(&mut self, spirv: &[u32], build_flags: &str) -> Option<&NativeBinary> {
        let key = spirv_cache_key(spirv, build_flags);
        if self.entries.contains_key(&key) {
            self.hits += 1;
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a compiled native binary for `spirv`, returning its cache key.
    ///
    /// The binary's own `build_flags` are folded into the key so two builds of
    /// the same SPIR-V with different flags do not collide.
    pub fn insert(&mut self, spirv: &[u32], binary: NativeBinary) -> u64 {
        let key = spirv_cache_key(spirv, &binary.build_flags);
        self.entries.insert(key, binary);
        key
    }

    /// Fetch a binary by its precomputed key without affecting statistics.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<&NativeBinary> {
        self.entries.get(&key)
    }

    /// Remove all cached binaries (statistics are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Total bytes occupied by all cached native binaries.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.entries.values().map(NativeBinary::len).sum()
    }
}

// ─── EU occupancy advisor ────────────────────────────────────

/// Recommends a workgroup size from an EU-utilisation fraction.
///
/// When EUs are already busy (high utilisation) a smaller workgroup reduces
/// per-launch pressure and improves overlap; when EUs are idle a larger
/// workgroup maximises throughput. The advisor clamps recommendations to the
/// device's `[min, max]` workgroup-size envelope and to powers of two (Intel
/// sub-group sizes are 8/16/32, so workgroup sizes are typically power-of-two
/// multiples).
#[derive(Debug, Clone)]
pub struct EuOccupancyAdvisor {
    min_wg: u32,
    max_wg: u32,
    eu_count: u32,
}

impl EuOccupancyAdvisor {
    /// Create an advisor for a device with `eu_count` execution units and a
    /// workgroup-size envelope of `[min_wg, max_wg]` (both powers of two).
    pub fn new(eu_count: u32, min_wg: u32, max_wg: u32) -> LevelZeroResult<Self> {
        if eu_count == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "EU count must be > 0".into(),
            ));
        }
        if min_wg == 0 || max_wg == 0 || !min_wg.is_power_of_two() || !max_wg.is_power_of_two() {
            return Err(LevelZeroError::InvalidArgument(
                "workgroup-size bounds must be non-zero powers of two".into(),
            ));
        }
        if min_wg > max_wg {
            return Err(LevelZeroError::InvalidArgument(
                "min workgroup size exceeds max".into(),
            ));
        }
        Ok(Self {
            min_wg,
            max_wg,
            eu_count,
        })
    }

    /// The device EU count.
    #[must_use]
    pub fn eu_count(&self) -> u32 {
        self.eu_count
    }

    /// Recommend a workgroup size given the current EU-utilisation fraction.
    ///
    /// `utilisation` is clamped to `[0, 1]`. At full idle (`0.0`) the advisor
    /// returns `max_wg`; at full saturation (`1.0`) it returns `min_wg`;
    /// intermediate utilisations interpolate geometrically (halving per
    /// quartile of busy-ness) and snap down to the nearest power of two within
    /// the envelope.
    #[must_use]
    pub fn recommend_workgroup_size(&self, utilisation: f64) -> u32 {
        let u = utilisation.clamp(0.0, 1.0);
        // Geometric scale: idle → max, busy → progressively smaller.
        // shift = round(u * log2(max/min)); recommended = max >> shift.
        let span_log2 = (self.max_wg / self.min_wg).trailing_zeros();
        let shift = (u * f64::from(span_log2)).round() as u32;
        let candidate = (self.max_wg >> shift).max(self.min_wg);
        // Snap to power of two within the envelope.
        let snapped = if candidate.is_power_of_two() {
            candidate
        } else {
            candidate.next_power_of_two()
        };
        snapped.clamp(self.min_wg, self.max_wg)
    }

    /// Estimate the number of concurrent workgroups the device can host for a
    /// given workgroup size, assuming one sub-group of `subgroup_size` per EU
    /// thread. This feeds the dispatch grid sizing.
    #[must_use]
    pub fn estimated_resident_workgroups(&self, workgroup_size: u32, subgroup_size: u32) -> u32 {
        if workgroup_size == 0 || subgroup_size == 0 {
            return 0;
        }
        let subgroups_per_wg = workgroup_size.div_ceil(subgroup_size).max(1);
        // Each EU can host (roughly) one sub-group concurrently in this model.
        (self.eu_count / subgroups_per_wg).max(1)
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_deterministic_and_flag_sensitive() {
        let spirv = [0x0723_0203, 0x0001_0200, 1, 2, 3];
        let k1 = spirv_cache_key(&spirv, "");
        let k2 = spirv_cache_key(&spirv, "");
        assert_eq!(k1, k2, "same input → same key");
        let k3 = spirv_cache_key(&spirv, "-ze-opt-disable");
        assert_ne!(k1, k3, "build flags must change the key");
        let mut other = spirv;
        other[4] = 99;
        assert_ne!(
            k1,
            spirv_cache_key(&other, ""),
            "different SPIR-V → different key"
        );
    }

    #[test]
    fn native_binary_fields() {
        let b = NativeBinary::new(vec![1, 2, 3, 4], "-cl-fast-relaxed-math");
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
        assert_eq!(b.build_flags, "-cl-fast-relaxed-math");
        assert!(NativeBinary::new(vec![], "").is_empty());
    }

    #[test]
    fn cache_miss_then_hit() {
        let mut cache = ModuleBinaryCache::new();
        assert!(cache.is_empty());
        let spirv = [0x0723_0203, 1, 2, 3, 4];

        // First lookup misses.
        assert!(cache.lookup(&spirv, "").is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        // Insert and look up again → hit.
        let key = cache.insert(&spirv, NativeBinary::new(vec![9, 8, 7], ""));
        assert_eq!(cache.len(), 1);
        let got = cache.lookup(&spirv, "").expect("hit");
        assert_eq!(got.bytes, vec![9, 8, 7]);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.get(key).map(NativeBinary::len), Some(3));
        assert_eq!(cache.total_bytes(), 3);
    }

    #[test]
    fn cache_hit_rate_and_clear() {
        let mut cache = ModuleBinaryCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
        let spirv = [1u32, 2, 3, 4, 5];
        cache.insert(&spirv, NativeBinary::new(vec![1], ""));
        assert!(cache.lookup(&spirv, "").is_some()); // hit
        assert!(cache.lookup(&[9u32, 9, 9, 9, 9], "").is_none()); // miss
        assert!(cache.lookup(&[9u32, 9, 9, 9, 9], "").is_none()); // miss
        // 1 hit, 2 misses → 1/3.
        assert!((cache.hit_rate() - (1.0 / 3.0)).abs() < 1e-9);
        cache.clear();
        assert!(cache.is_empty());
        // Statistics survive clear.
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn cache_distinguishes_build_flags() {
        let mut cache = ModuleBinaryCache::new();
        let spirv = [1u32, 2, 3, 4, 5];
        cache.insert(&spirv, NativeBinary::new(vec![1], ""));
        cache.insert(&spirv, NativeBinary::new(vec![2, 2], "-O3"));
        // Two distinct entries despite identical SPIR-V.
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.lookup(&spirv, "").unwrap().bytes, vec![1]);
        assert_eq!(cache.lookup(&spirv, "-O3").unwrap().bytes, vec![2, 2]);
    }

    #[test]
    fn occupancy_advisor_idle_vs_busy() {
        let adv = EuOccupancyAdvisor::new(512, 8, 256).unwrap();
        assert_eq!(adv.eu_count(), 512);
        // Idle → maximum workgroup.
        assert_eq!(adv.recommend_workgroup_size(0.0), 256);
        // Fully busy → minimum workgroup.
        assert_eq!(adv.recommend_workgroup_size(1.0), 8);
        // Midway → somewhere strictly between, power of two.
        let mid = adv.recommend_workgroup_size(0.5);
        assert!((8..=256).contains(&mid));
        assert!(mid.is_power_of_two());
    }

    #[test]
    fn occupancy_advisor_monotonic() {
        let adv = EuOccupancyAdvisor::new(128, 16, 512).unwrap();
        let low = adv.recommend_workgroup_size(0.1);
        let high = adv.recommend_workgroup_size(0.9);
        assert!(
            low >= high,
            "busier device should not recommend a larger workgroup (low={low}, high={high})"
        );
        // Out-of-range utilisation is clamped, not panicking.
        assert_eq!(adv.recommend_workgroup_size(-5.0), 512);
        assert_eq!(adv.recommend_workgroup_size(99.0), 16);
    }

    #[test]
    fn occupancy_advisor_resident_estimate() {
        let adv = EuOccupancyAdvisor::new(512, 8, 256).unwrap();
        // wg 32, subgroup 16 → 2 subgroups/wg → 256 resident.
        assert_eq!(adv.estimated_resident_workgroups(32, 16), 256);
        // Degenerate inputs return 0.
        assert_eq!(adv.estimated_resident_workgroups(0, 16), 0);
        assert_eq!(adv.estimated_resident_workgroups(32, 0), 0);
    }

    #[test]
    fn occupancy_advisor_rejects_bad_args() {
        assert!(EuOccupancyAdvisor::new(0, 8, 256).is_err());
        assert!(EuOccupancyAdvisor::new(128, 0, 256).is_err());
        assert!(
            EuOccupancyAdvisor::new(128, 24, 256).is_err(),
            "non-pow2 min"
        );
        assert!(EuOccupancyAdvisor::new(128, 512, 256).is_err(), "min > max");
    }
}
