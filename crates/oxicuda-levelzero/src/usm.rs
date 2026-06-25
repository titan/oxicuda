//! Unified Shared Memory (USM) allocation logic for Intel Level Zero.
//!
//! Level Zero exposes three flavours of *Unified Shared Memory*:
//!
//! - **Device USM** (`zeMemAllocDevice`) — resident in device-local memory,
//!   not directly CPU-accessible; fastest for kernels.
//! - **Host USM** (`zeMemAllocHost`) — resident in host memory, accessible by
//!   both CPU and GPU (the GPU reads it across the bus).
//! - **Shared USM** (`zeMemAllocShared`) — migratable between host and device;
//!   the driver moves pages on demand.
//!
//! The *device-gated* part is the raw `zeMemAlloc*` call (which lives in
//! [`crate::memory`]). Everything in this module is **host-side bookkeeping**:
//! the suballocation arithmetic, alignment rounding, memory-ordinal selection
//! from a queried property table, and the memory-advise model. All of it is
//! CPU-testable without a physical Intel GPU.
//!
//! A backend allocates a few large `zeMemAlloc*` *blocks* and sub-allocates
//! individual tensors out of them via [`UsmSuballocator`], returning a byte
//! *offset* into the block. The caller then pointer-offsets the block base —
//! that pointer arithmetic is the only step needing a real allocation.

use crate::error::{LevelZeroError, LevelZeroResult};

// ─── USM memory kinds ────────────────────────────────────────

/// The kind of Unified Shared Memory backing an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UsmKind {
    /// Device-local memory (`zeMemAllocDevice`). Not CPU-accessible.
    Device,
    /// Host memory (`zeMemAllocHost`). CPU + GPU accessible.
    Host,
    /// Migratable shared memory (`zeMemAllocShared`).
    Shared,
}

impl UsmKind {
    /// Returns `true` when this kind is directly readable/writable by the CPU.
    #[must_use]
    pub fn is_host_accessible(self) -> bool {
        matches!(self, UsmKind::Host | UsmKind::Shared)
    }

    /// The `ze_structure_type_t` value of the corresponding alloc descriptor.
    ///
    /// Device → `ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC` (0x1),
    /// Host → `ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC` (0x2). Shared uses the
    /// device descriptor with a chained host descriptor in `pNext`.
    #[must_use]
    pub fn alloc_desc_stype(self) -> u32 {
        match self {
            UsmKind::Device | UsmKind::Shared => 0x1,
            UsmKind::Host => 0x2,
        }
    }
}

/// Minimum alignment Level Zero guarantees for any USM allocation (bytes).
///
/// The L0 spec mandates allocations are at least 8-byte aligned; intel-compute
/// -runtime returns 64-byte-aligned pointers in practice. We default to 64 so
/// SIMD16 float loads never straddle a cache line.
pub const USM_DEFAULT_ALIGNMENT: u64 = 64;

/// Round `value` up to the nearest multiple of `alignment` (a power of two).
#[must_use]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

// ─── Memory ordinal / property-table selection ───────────────

/// A single entry of `zeDeviceGetMemoryProperties` output.
///
/// Each Intel GPU reports one or more memory ordinals (e.g. HBM on Ponte
/// Vecchio, GDDR6 on Arc, system DRAM on Xe-LP). The backend picks an ordinal
/// for device allocations based on this table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryOrdinalInfo {
    /// The ordinal index passed to `zeMemAllocDevice`'s descriptor.
    pub ordinal: u32,
    /// Human-readable memory name reported by the driver (e.g. `"HBM"`).
    pub name: String,
    /// Total bytes of this memory pool.
    pub total_bytes: u64,
    /// Peak bandwidth in bytes/sec (0 when the driver does not report it).
    pub max_bandwidth: u64,
}

/// A queried memory-property table for one device.
#[derive(Debug, Clone, Default)]
pub struct MemoryPropertyTable {
    ordinals: Vec<MemoryOrdinalInfo>,
}

impl MemoryPropertyTable {
    /// Build a table from a list of ordinal records.
    #[must_use]
    pub fn new(ordinals: Vec<MemoryOrdinalInfo>) -> Self {
        Self { ordinals }
    }

    /// Number of distinct memory ordinals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ordinals.len()
    }

    /// Returns `true` when the table has no ordinals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ordinals.is_empty()
    }

    /// All ordinals, in driver-reported order.
    #[must_use]
    pub fn ordinals(&self) -> &[MemoryOrdinalInfo] {
        &self.ordinals
    }

    /// Select the device-memory ordinal best suited for a `bytes`-sized
    /// allocation.
    ///
    /// Strategy: among ordinals with enough capacity, prefer the one with the
    /// highest reported bandwidth (ties broken by larger capacity, then lower
    /// ordinal index). This routes large tensors to HBM on Ponte Vecchio while
    /// falling back to whatever fits on memory-constrained parts.
    ///
    /// Returns [`LevelZeroError::OutOfMemory`] when no ordinal can hold `bytes`,
    /// or [`LevelZeroError::NoSuitableDevice`] when the table is empty.
    pub fn select_device_ordinal(&self, bytes: u64) -> LevelZeroResult<u32> {
        if self.ordinals.is_empty() {
            return Err(LevelZeroError::NoSuitableDevice);
        }
        let best = self
            .ordinals
            .iter()
            .filter(|o| o.total_bytes >= bytes)
            .max_by(|a, b| {
                a.max_bandwidth
                    .cmp(&b.max_bandwidth)
                    .then(a.total_bytes.cmp(&b.total_bytes))
                    .then(b.ordinal.cmp(&a.ordinal))
            });
        match best {
            Some(info) => Ok(info.ordinal),
            None => Err(LevelZeroError::OutOfMemory),
        }
    }

    /// Total capacity across all reported memory ordinals.
    #[must_use]
    pub fn total_capacity(&self) -> u64 {
        self.ordinals.iter().map(|o| o.total_bytes).sum()
    }
}

// ─── Memory advise model ─────────────────────────────────────

/// Memory-advise hints applied to a shared/device USM range.
///
/// These mirror `ze_memory_advice_t`. The hints are pure metadata that the
/// driver uses to steer page migration for shared USM; we model the *intent*
/// here so a planner can validate and accumulate hints before issuing the
/// device-gated `zeCommandListAppendMemAdvise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// `ZE_MEMORY_ADVICE_SET_READ_MOSTLY` — pages are read far more than written.
    SetReadMostly,
    /// `ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY`.
    ClearReadMostly,
    /// `ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION` — pin to the device.
    SetPreferredLocation,
    /// `ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION`.
    ClearPreferredLocation,
    /// `ZE_MEMORY_ADVICE_SET_ACCESSED_BY_DEVICE`.
    SetAccessedByDevice,
    /// `ZE_MEMORY_ADVICE_CLEAR_ACCESSED_BY_DEVICE`.
    ClearAccessedByDevice,
    /// `ZE_MEMORY_ADVICE_BIAS_CACHED`.
    BiasCached,
    /// `ZE_MEMORY_ADVICE_BIAS_UNCACHED`.
    BiasUncached,
}

impl MemoryAdvice {
    /// The `ze_memory_advice_t` enum value.
    #[must_use]
    pub fn ze_value(self) -> u32 {
        match self {
            MemoryAdvice::SetReadMostly => 0,
            MemoryAdvice::ClearReadMostly => 1,
            MemoryAdvice::SetPreferredLocation => 2,
            MemoryAdvice::ClearPreferredLocation => 3,
            MemoryAdvice::SetAccessedByDevice => 4,
            MemoryAdvice::ClearAccessedByDevice => 5,
            MemoryAdvice::BiasCached => 6,
            MemoryAdvice::BiasUncached => 7,
        }
    }

    /// The advice that *clears* this one, if it is a set/clear pair.
    #[must_use]
    pub fn clearing_advice(self) -> Option<MemoryAdvice> {
        match self {
            MemoryAdvice::SetReadMostly => Some(MemoryAdvice::ClearReadMostly),
            MemoryAdvice::SetPreferredLocation => Some(MemoryAdvice::ClearPreferredLocation),
            MemoryAdvice::SetAccessedByDevice => Some(MemoryAdvice::ClearAccessedByDevice),
            _ => None,
        }
    }
}

/// Accumulates memory-advise hints for a single USM range.
///
/// Applying a *set* hint records it; applying the matching *clear* hint removes
/// it. `BiasCached` / `BiasUncached` are mutually exclusive — the most recent
/// wins. This lets a planner fold a sequence of advise calls into the minimal
/// effective set before touching the driver.
#[derive(Debug, Clone, Default)]
pub struct MemoryAdviseState {
    read_mostly: bool,
    preferred_location: bool,
    accessed_by_device: bool,
    /// `Some(true)` = cached bias, `Some(false)` = uncached, `None` = default.
    bias_cached: Option<bool>,
}

impl MemoryAdviseState {
    /// A fresh state with no advice applied.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one advice hint, folding it into the effective state.
    pub fn apply(&mut self, advice: MemoryAdvice) {
        match advice {
            MemoryAdvice::SetReadMostly => self.read_mostly = true,
            MemoryAdvice::ClearReadMostly => self.read_mostly = false,
            MemoryAdvice::SetPreferredLocation => self.preferred_location = true,
            MemoryAdvice::ClearPreferredLocation => self.preferred_location = false,
            MemoryAdvice::SetAccessedByDevice => self.accessed_by_device = true,
            MemoryAdvice::ClearAccessedByDevice => self.accessed_by_device = false,
            MemoryAdvice::BiasCached => self.bias_cached = Some(true),
            MemoryAdvice::BiasUncached => self.bias_cached = Some(false),
        }
    }

    /// `true` when `SetReadMostly` is currently in effect.
    #[must_use]
    pub fn is_read_mostly(&self) -> bool {
        self.read_mostly
    }

    /// `true` when the range is pinned to its preferred device location.
    #[must_use]
    pub fn is_preferred_location(&self) -> bool {
        self.preferred_location
    }

    /// `true` when the device is marked as an accessor of this range.
    #[must_use]
    pub fn is_accessed_by_device(&self) -> bool {
        self.accessed_by_device
    }

    /// The cache bias: `Some(true)` cached, `Some(false)` uncached, `None`
    /// driver-default.
    #[must_use]
    pub fn cache_bias(&self) -> Option<bool> {
        self.bias_cached
    }

    /// The minimal sequence of advise calls that reproduces this state from a
    /// freshly-allocated range. Used to replay folded hints to the driver.
    #[must_use]
    pub fn effective_advice(&self) -> Vec<MemoryAdvice> {
        let mut out = Vec::new();
        if self.read_mostly {
            out.push(MemoryAdvice::SetReadMostly);
        }
        if self.preferred_location {
            out.push(MemoryAdvice::SetPreferredLocation);
        }
        if self.accessed_by_device {
            out.push(MemoryAdvice::SetAccessedByDevice);
        }
        match self.bias_cached {
            Some(true) => out.push(MemoryAdvice::BiasCached),
            Some(false) => out.push(MemoryAdvice::BiasUncached),
            None => {}
        }
        out
    }
}

// ─── USM sub-allocator ───────────────────────────────────────

/// A sub-allocation carved out of a USM block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsmSubAllocation {
    /// Byte offset of the allocation within the backing block.
    pub offset: u64,
    /// Requested usable size (not the alignment-padded size).
    pub size: u64,
}

/// One contiguous free span within the backing USM block.
#[derive(Debug, Clone, Copy)]
struct FreeSpan {
    offset: u64,
    size: u64,
}

/// First-fit free-list sub-allocator over a single USM block.
///
/// Mirrors the device-memory-allocator pattern: tracks free spans, splits on
/// allocation, coalesces adjacent spans on free, and honours arbitrary
/// power-of-two alignment (accounting for the introduced leading padding).
///
/// The [`UsmKind`] is carried so a backend can keep separate suballocators per
/// memory flavour and never mix host and device offsets.
#[derive(Debug)]
pub struct UsmSuballocator {
    kind: UsmKind,
    block_size: u64,
    free: Vec<FreeSpan>,
    /// Returned-offset → carved span `(offset, size)` (including padding) so
    /// `free` restores the original region exactly.
    live: Vec<(u64, FreeSpan)>,
}

impl UsmSuballocator {
    /// Create a suballocator managing `block_size` bytes of `kind` memory.
    pub fn new(kind: UsmKind, block_size: u64) -> LevelZeroResult<Self> {
        if block_size == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "USM block_size must be > 0".into(),
            ));
        }
        Ok(Self {
            kind,
            block_size,
            free: vec![FreeSpan {
                offset: 0,
                size: block_size,
            }],
            live: Vec::new(),
        })
    }

    /// The USM flavour this suballocator manages.
    #[must_use]
    pub fn kind(&self) -> UsmKind {
        self.kind
    }

    /// Total managed size in bytes.
    #[must_use]
    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    /// Sum of all currently-free bytes (possibly fragmented).
    #[must_use]
    pub fn free_bytes(&self) -> u64 {
        self.free.iter().map(|s| s.size).sum()
    }

    /// The largest single contiguous free span.
    #[must_use]
    pub fn largest_free_span(&self) -> u64 {
        self.free.iter().map(|s| s.size).max().unwrap_or(0)
    }

    /// Number of live allocations.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Allocate `size` bytes aligned to `alignment` (a power of two).
    pub fn alloc(&mut self, size: u64, alignment: u64) -> LevelZeroResult<UsmSubAllocation> {
        if size == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "USM suballocation size must be > 0".into(),
            ));
        }
        if !alignment.is_power_of_two() {
            return Err(LevelZeroError::InvalidArgument(format!(
                "alignment {alignment} must be a power of two"
            )));
        }

        let mut chosen: Option<usize> = None;
        let mut aligned_off = 0u64;
        for (i, span) in self.free.iter().enumerate() {
            let a = align_up(span.offset, alignment);
            let pad = a - span.offset;
            if pad + size <= span.size {
                chosen = Some(i);
                aligned_off = a;
                break;
            }
        }
        let Some(idx) = chosen else {
            return Err(LevelZeroError::OutOfMemory);
        };

        let span = self.free[idx];
        let pad = aligned_off - span.offset;
        let carved = FreeSpan {
            offset: span.offset,
            size: pad + size,
        };

        let remaining_off = carved.offset + carved.size;
        let remaining_size = span.size - carved.size;
        if remaining_size == 0 {
            self.free.remove(idx);
        } else {
            self.free[idx] = FreeSpan {
                offset: remaining_off,
                size: remaining_size,
            };
        }

        self.live.push((aligned_off, carved));
        Ok(UsmSubAllocation {
            offset: aligned_off,
            size,
        })
    }

    /// Allocate `size` bytes using the default USM alignment.
    pub fn alloc_default(&mut self, size: u64) -> LevelZeroResult<UsmSubAllocation> {
        self.alloc(size, USM_DEFAULT_ALIGNMENT)
    }

    /// Free a previously-returned allocation by its `offset`.
    pub fn free(&mut self, offset: u64) -> LevelZeroResult<()> {
        let pos = self
            .live
            .iter()
            .position(|(o, _)| *o == offset)
            .ok_or_else(|| {
                LevelZeroError::InvalidArgument(format!("USM free of unknown offset {offset}"))
            })?;
        let (_, carved) = self.live.remove(pos);
        self.insert_and_coalesce(carved);
        Ok(())
    }

    /// Insert a freed span and merge with adjacent free spans.
    fn insert_and_coalesce(&mut self, span: FreeSpan) {
        self.free.push(span);
        self.free.sort_by_key(|s| s.offset);
        let mut merged: Vec<FreeSpan> = Vec::with_capacity(self.free.len());
        for s in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.offset + last.size == s.offset {
                    last.size += s.size;
                    continue;
                }
            }
            merged.push(s);
        }
        self.free = merged;
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usm_kind_host_accessibility() {
        assert!(!UsmKind::Device.is_host_accessible());
        assert!(UsmKind::Host.is_host_accessible());
        assert!(UsmKind::Shared.is_host_accessible());
    }

    #[test]
    fn usm_kind_desc_stypes() {
        assert_eq!(UsmKind::Device.alloc_desc_stype(), 0x1);
        assert_eq!(UsmKind::Host.alloc_desc_stype(), 0x2);
        assert_eq!(UsmKind::Shared.alloc_desc_stype(), 0x1);
    }

    #[test]
    fn align_up_powers_of_two() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(200, 256), 256);
    }

    #[test]
    fn memory_table_select_prefers_bandwidth() {
        let table = MemoryPropertyTable::new(vec![
            MemoryOrdinalInfo {
                ordinal: 0,
                name: "DDR".into(),
                total_bytes: 64 << 30,
                max_bandwidth: 50 << 30,
            },
            MemoryOrdinalInfo {
                ordinal: 1,
                name: "HBM".into(),
                total_bytes: 48 << 30,
                max_bandwidth: 1200 << 30,
            },
        ]);
        // HBM (ordinal 1) has much higher bandwidth and fits 1 GiB.
        assert_eq!(table.select_device_ordinal(1 << 30).unwrap(), 1);
        assert_eq!(table.len(), 2);
        assert_eq!(table.total_capacity(), (64 << 30) + (48 << 30));
    }

    #[test]
    fn memory_table_select_skips_too_small() {
        let table = MemoryPropertyTable::new(vec![
            MemoryOrdinalInfo {
                ordinal: 0,
                name: "small-fast".into(),
                total_bytes: 1 << 20,
                max_bandwidth: 1000 << 30,
            },
            MemoryOrdinalInfo {
                ordinal: 1,
                name: "big-slow".into(),
                total_bytes: 16 << 30,
                max_bandwidth: 50 << 30,
            },
        ]);
        // 8 GiB only fits in the big-slow pool despite its lower bandwidth.
        assert_eq!(table.select_device_ordinal(8 << 30).unwrap(), 1);
    }

    #[test]
    fn memory_table_errors() {
        let empty = MemoryPropertyTable::default();
        assert!(empty.is_empty());
        assert!(matches!(
            empty.select_device_ordinal(1),
            Err(LevelZeroError::NoSuitableDevice)
        ));

        let tiny = MemoryPropertyTable::new(vec![MemoryOrdinalInfo {
            ordinal: 0,
            name: "tiny".into(),
            total_bytes: 1024,
            max_bandwidth: 1,
        }]);
        assert!(matches!(
            tiny.select_device_ordinal(1 << 30),
            Err(LevelZeroError::OutOfMemory)
        ));
    }

    #[test]
    fn memory_advise_set_clear_pairing() {
        assert_eq!(
            MemoryAdvice::SetReadMostly.clearing_advice(),
            Some(MemoryAdvice::ClearReadMostly)
        );
        assert_eq!(MemoryAdvice::BiasCached.clearing_advice(), None);
        assert_eq!(MemoryAdvice::SetReadMostly.ze_value(), 0);
        assert_eq!(MemoryAdvice::BiasUncached.ze_value(), 7);
    }

    #[test]
    fn memory_advise_state_folds() {
        let mut st = MemoryAdviseState::new();
        st.apply(MemoryAdvice::SetReadMostly);
        st.apply(MemoryAdvice::SetPreferredLocation);
        st.apply(MemoryAdvice::BiasCached);
        assert!(st.is_read_mostly());
        assert!(st.is_preferred_location());
        assert_eq!(st.cache_bias(), Some(true));

        // Clearing read-mostly removes it; later bias overrides earlier.
        st.apply(MemoryAdvice::ClearReadMostly);
        st.apply(MemoryAdvice::BiasUncached);
        assert!(!st.is_read_mostly());
        assert_eq!(st.cache_bias(), Some(false));

        let eff = st.effective_advice();
        assert!(eff.contains(&MemoryAdvice::SetPreferredLocation));
        assert!(eff.contains(&MemoryAdvice::BiasUncached));
        assert!(!eff.contains(&MemoryAdvice::SetReadMostly));
    }

    #[test]
    fn usm_suballoc_basic() {
        let mut a = UsmSuballocator::new(UsmKind::Device, 1024).unwrap();
        assert_eq!(a.kind(), UsmKind::Device);
        assert_eq!(a.block_size(), 1024);
        assert_eq!(a.free_bytes(), 1024);

        let x = a.alloc(256, 64).unwrap();
        assert_eq!(x.offset, 0);
        assert_eq!(x.size, 256);
        assert_eq!(a.free_bytes(), 768);
        assert_eq!(a.live_count(), 1);

        let y = a.alloc_default(128).unwrap();
        assert_eq!(y.offset, 256);

        a.free(x.offset).unwrap();
        a.free(y.offset).unwrap();
        assert_eq!(a.free_bytes(), 1024);
        assert_eq!(a.largest_free_span(), 1024);
        assert_eq!(a.live_count(), 0);
    }

    #[test]
    fn usm_suballoc_alignment_padding() {
        let mut a = UsmSuballocator::new(UsmKind::Shared, 4096).unwrap();
        let head = a.alloc(10, 1).unwrap();
        assert_eq!(head.offset, 0);
        // Next 256-aligned alloc must skip to offset 256.
        let aligned = a.alloc(16, 256).unwrap();
        assert_eq!(aligned.offset, 256);
        a.free(head.offset).unwrap();
        a.free(aligned.offset).unwrap();
        assert_eq!(a.free_bytes(), 4096);
    }

    #[test]
    fn usm_suballoc_oom_and_bad_args() {
        let mut a = UsmSuballocator::new(UsmKind::Host, 128).unwrap();
        assert!(matches!(a.alloc(256, 1), Err(LevelZeroError::OutOfMemory)));
        assert!(a.alloc(0, 1).is_err());
        assert!(a.alloc(8, 3).is_err(), "non-pow2 alignment rejected");
        assert!(a.free(999).is_err(), "unknown offset rejected");
        assert!(UsmSuballocator::new(UsmKind::Host, 0).is_err());
    }

    #[test]
    fn usm_suballoc_coalesce_middle_hole() {
        let mut a = UsmSuballocator::new(UsmKind::Device, 300).unwrap();
        let x = a.alloc(100, 1).unwrap();
        let y = a.alloc(100, 1).unwrap();
        let z = a.alloc(100, 1).unwrap();
        assert_eq!(a.free_bytes(), 0);
        a.free(x.offset).unwrap();
        a.free(z.offset).unwrap();
        // Two disjoint holes — must not merge across the live middle.
        assert_eq!(a.largest_free_span(), 100);
        a.free(y.offset).unwrap();
        assert_eq!(a.largest_free_span(), 300);
    }
}
