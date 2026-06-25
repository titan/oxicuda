//! CPU model of the CUDA Runtime stream-ordered memory pool
//! (`cudaMallocAsync` / `cudaFreeAsync` / the `cudaMemPool*` family).
//!
//! This is a self-contained, deterministic, GPU-free model of the *caching
//! allocator* the CUDA Runtime exposes at the `cudaMemPool_t` surface.  It
//! tracks a pool's reserved / used bytes, honours stream-ordered free
//! semantics (a block freed on a stream is only reusable once that stream has
//! reached the free point), reuses freed blocks for same-or-larger requests,
//! and trims the free list against a release threshold attribute.
//!
//! It is deliberately a *runtime-layer* model: it works in terms of the
//! runtime's [`CudaStream`] handle and a `cudaMemPool_t`-style attribute table,
//! and it does not call the driver.  (The sibling `oxicuda-driver` crate has its
//! own lower-level `StreamMemoryPool`; this models the higher cudart surface and
//! its attribute set, not the driver internals.)
//!
//! # Stream-ordered semantics
//!
//! `cudaMallocAsync(size, stream)` hands back a pointer immediately, but the
//! allocation is only valid on the GPU once `stream` reaches that point.
//! `cudaFreeAsync(ptr, stream)` records a free that only releases the block
//! once `stream` reaches the free point.  Until then the block is still in use
//! by earlier work on the stream, so reuse across streams is only safe after
//! the freeing stream is synchronised (modelled here by [`MemPool::synchronize`]).

use std::collections::HashMap;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── Pool identity & attributes ──────────────────────────────────────────────

/// A `cudaMemPool_t`-style opaque pool handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemPoolHandle(pub u64);

impl MemPoolHandle {
    /// The reserved value used for "no pool".
    pub const NULL: Self = Self(0);
}

/// Attribute selectors for [`MemPool::set_attribute`] / [`MemPool::get_attribute`].
///
/// Mirrors `cudaMemPoolAttr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemPoolAttr {
    /// `cudaMemPoolAttrReleaseThreshold`: bytes to keep on the free list when
    /// the pool is trimmed at a stream/event sync point.
    ReleaseThreshold,
    /// `cudaMemPoolReuseFollowEventDependencies`: allow reuse of memory freed on
    /// a stream that an allocating stream waits on via an event.
    ReuseFollowEventDependencies,
    /// `cudaMemPoolReuseAllowOpportunistic`: allow reuse of memory whose free
    /// has already completed (the stream has passed the free point).
    ReuseAllowOpportunistic,
    /// `cudaMemPoolReuseAllowInternalDependencies`: allow the pool to insert
    /// internal stream dependencies to enable reuse.
    ReuseAllowInternalDependencies,
}

/// The full attribute state of a pool (the values `cudaMemPoolGetAttribute`
/// would return).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemPoolAttributes {
    /// Bytes retained on the free list when trimmed (`u64::MAX` ≈ never release).
    pub release_threshold: u64,
    /// Reuse memory freed on a stream the allocator waits on via an event.
    pub reuse_follow_event_dependencies: bool,
    /// Reuse memory whose free has already completed on its stream.
    pub reuse_allow_opportunistic: bool,
    /// Permit the pool to add internal stream dependencies to enable reuse.
    pub reuse_allow_internal_dependencies: bool,
}

impl Default for MemPoolAttributes {
    fn default() -> Self {
        // Matches the CUDA driver defaults: release-threshold 0 (free everything
        // at the next sync), all three reuse policies enabled.
        Self {
            release_threshold: 0,
            reuse_follow_event_dependencies: true,
            reuse_allow_opportunistic: true,
            reuse_allow_internal_dependencies: true,
        }
    }
}

/// Live usage statistics for a pool (mirrors the readable `cudaMemPoolAttr`
/// counters `cudaMemPoolAttrReservedMemCurrent` etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemPoolStats {
    /// Bytes currently carved from the (virtual) device by the pool.
    pub reserved_current: usize,
    /// High-water mark of [`Self::reserved_current`].
    pub reserved_high: usize,
    /// Bytes currently handed out to live allocations.
    pub used_current: usize,
    /// High-water mark of [`Self::used_current`].
    pub used_high: usize,
}

// ─── Internal block bookkeeping ──────────────────────────────────────────────

/// A virtual block tracked by the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    /// Virtual address (non-zero).
    ptr: u64,
    /// Capacity in bytes (may exceed the live request that owns it).
    capacity: usize,
}

/// A free recorded on a stream, pending the stream reaching the free point.
#[derive(Debug, Clone, Copy)]
struct PendingFree {
    block: Block,
    /// Raw stream token the free was issued on.
    stream: u64,
    /// Sequence number at which the free was submitted on that stream.
    seq: u64,
}

/// Per-stream logical clock: `submit` is the next sequence number, `reached`
/// is how far the stream has executed.
#[derive(Debug, Clone, Copy, Default)]
struct StreamClock {
    submit: u64,
    reached: u64,
}

// ─── MemPool ─────────────────────────────────────────────────────────────────

/// A CPU model of a `cudaMemPool_t` caching allocator with stream-ordered reuse.
#[derive(Debug)]
pub struct MemPool {
    handle: MemPoolHandle,
    device: i32,
    attrs: MemPoolAttributes,
    /// Next virtual address to carve when the free list cannot satisfy a request.
    next_addr: u64,
    /// Free, immediately-reusable blocks (their free has completed).
    free_list: Vec<Block>,
    /// Live allocations: address → block.
    live: HashMap<u64, Block>,
    /// Stream-ordered frees awaiting their stream to reach the free point.
    pending: Vec<PendingFree>,
    /// Per-stream clocks (keyed by raw stream token).
    clocks: HashMap<u64, StreamClock>,
    stats: MemPoolStats,
}

impl MemPool {
    /// Address space base; deliberately non-zero so a real pointer is never NULL.
    const ADDR_BASE: u64 = 0x1_0000_0000;
    /// Minimum allocation granularity (CUDA pools round up to 512 bytes).
    const GRANULARITY: usize = 512;

    /// Create a pool for `device` with default attributes.
    #[must_use]
    pub fn new(device: i32) -> Self {
        Self::with_attributes(device, MemPoolAttributes::default())
    }

    /// Create a pool with explicit attributes.
    #[must_use]
    pub fn with_attributes(device: i32, attrs: MemPoolAttributes) -> Self {
        // A stable, per-(device) handle token; high bit set so it never collides
        // with a NULL handle.
        let handle = MemPoolHandle(0x8000_0000_0000_0000 | ((device as u64) & 0xFFFF));
        Self {
            handle,
            device,
            attrs,
            next_addr: Self::ADDR_BASE,
            free_list: Vec::new(),
            live: HashMap::new(),
            pending: Vec::new(),
            clocks: HashMap::new(),
            stats: MemPoolStats::default(),
        }
    }

    /// This pool's opaque handle.
    #[must_use]
    pub fn handle(&self) -> MemPoolHandle {
        self.handle
    }

    /// The device this pool belongs to.
    #[must_use]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Current attribute snapshot.
    #[must_use]
    pub fn attributes(&self) -> MemPoolAttributes {
        self.attrs
    }

    /// Current usage statistics (`cudaMemPoolAttrReserved*` / `Used*`).
    #[must_use]
    pub fn stats(&self) -> MemPoolStats {
        self.stats
    }

    fn round_up(size: usize) -> usize {
        size.div_ceil(Self::GRANULARITY) * Self::GRANULARITY
    }

    fn clock(&mut self, stream: u64) -> &mut StreamClock {
        self.clocks.entry(stream).or_default()
    }

    /// Set one attribute (`cudaMemPoolSetAttribute`).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidValue`] if `value` is out of range for the boolean
    /// attributes (only 0 / 1 permitted).
    pub fn set_attribute(&mut self, attr: MemPoolAttr, value: u64) -> CudaRtResult<()> {
        let as_bool = |v: u64| -> CudaRtResult<bool> {
            match v {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(CudaRtError::InvalidValue),
            }
        };
        match attr {
            MemPoolAttr::ReleaseThreshold => self.attrs.release_threshold = value,
            MemPoolAttr::ReuseFollowEventDependencies => {
                self.attrs.reuse_follow_event_dependencies = as_bool(value)?;
            }
            MemPoolAttr::ReuseAllowOpportunistic => {
                self.attrs.reuse_allow_opportunistic = as_bool(value)?;
            }
            MemPoolAttr::ReuseAllowInternalDependencies => {
                self.attrs.reuse_allow_internal_dependencies = as_bool(value)?;
            }
        }
        Ok(())
    }

    /// Read one attribute (`cudaMemPoolGetAttribute`).
    #[must_use]
    pub fn get_attribute(&self, attr: MemPoolAttr) -> u64 {
        match attr {
            MemPoolAttr::ReleaseThreshold => self.attrs.release_threshold,
            MemPoolAttr::ReuseFollowEventDependencies => {
                u64::from(self.attrs.reuse_follow_event_dependencies)
            }
            MemPoolAttr::ReuseAllowOpportunistic => u64::from(self.attrs.reuse_allow_opportunistic),
            MemPoolAttr::ReuseAllowInternalDependencies => {
                u64::from(self.attrs.reuse_allow_internal_dependencies)
            }
        }
    }

    /// Stream-ordered allocation (`cudaMallocAsync`).
    ///
    /// Returns the device address immediately.  A freed block of sufficient
    /// capacity is reused (best-fit, preferring an exact match); otherwise fresh
    /// virtual address space is carved.
    ///
    /// # Errors
    ///
    /// Returns `Ok(NULL)` for a zero-size request (matching the runtime); never
    /// fails in the model otherwise (the virtual address space is unbounded).
    pub fn malloc_async(&mut self, size: usize, stream: CudaStream) -> CudaRtResult<u64> {
        if size == 0 {
            return Ok(0);
        }
        let want = Self::round_up(size);

        // Try to reuse from the free list: pick the smallest block that fits
        // (best-fit) so we do not waste large blocks on small requests.
        let mut best: Option<usize> = None;
        for (i, blk) in self.free_list.iter().enumerate() {
            if blk.capacity >= want {
                match best {
                    Some(b) if self.free_list[b].capacity <= blk.capacity => {}
                    _ => best = Some(i),
                }
            }
        }

        let block = if let Some(idx) = best {
            self.free_list.swap_remove(idx)
        } else {
            let ptr = self.next_addr;
            self.next_addr = self.next_addr.saturating_add(want as u64);
            let block = Block {
                ptr,
                capacity: want,
            };
            // Newly carved memory grows the reserved footprint.
            self.stats.reserved_current += want;
            self.stats.reserved_high = self.stats.reserved_high.max(self.stats.reserved_current);
            block
        };

        // Advance the allocating stream's submit clock (the alloc is an op).
        self.clock(stream.raw().0 as u64).submit += 1;

        self.live.insert(block.ptr, block);
        self.stats.used_current += block.capacity;
        self.stats.used_high = self.stats.used_high.max(self.stats.used_current);
        Ok(block.ptr)
    }

    /// Stream-ordered free (`cudaFreeAsync`).
    ///
    /// The block is moved onto the pending-free queue tagged with `stream` and
    /// the current submit sequence; it only becomes reusable once `stream`
    /// reaches that point (see [`Self::synchronize`]).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidDevicePointer`] if `ptr` is not a live allocation
    /// of this pool.  Freeing the NULL pointer is a no-op.
    pub fn free_async(&mut self, ptr: u64, stream: CudaStream) -> CudaRtResult<()> {
        if ptr == 0 {
            return Ok(());
        }
        let block = self
            .live
            .remove(&ptr)
            .ok_or(CudaRtError::InvalidDevicePointer)?;
        self.stats.used_current -= block.capacity;

        let raw = stream.raw().0 as u64;
        let clk = self.clock(raw);
        let seq = clk.submit;
        clk.submit += 1;
        self.pending.push(PendingFree {
            block,
            stream: raw,
            seq,
        });
        Ok(())
    }

    /// Synchronise `stream` (`cudaStreamSynchronize`): every operation submitted
    /// on it is now complete, so any pending frees on `stream` are retired to the
    /// free list and the pool is trimmed against the release threshold.
    pub fn synchronize(&mut self, stream: CudaStream) {
        let raw = stream.raw().0 as u64;
        if let Some(clk) = self.clocks.get_mut(&raw) {
            clk.reached = clk.submit;
        }
        self.retire_completed();
        self.trim_to_threshold();
    }

    /// Retire every pending free whose stream has reached the free point.
    fn retire_completed(&mut self) {
        let clocks = &self.clocks;
        let mut still_pending = Vec::with_capacity(self.pending.len());
        for pf in self.pending.drain(..) {
            let reached = clocks
                .get(&pf.stream)
                .map(|c| c.reached > pf.seq)
                .unwrap_or(false);
            if reached {
                self.free_list.push(pf.block);
            } else {
                still_pending.push(pf);
            }
        }
        self.pending = still_pending;
    }

    /// Release free-list bytes above the release threshold back to the device
    /// (`reserved` shrinks).  Models the trim the runtime performs at sync points.
    fn trim_to_threshold(&mut self) {
        let threshold = self.attrs.release_threshold as usize;
        let mut free_bytes: usize = self.free_list.iter().map(|b| b.capacity).sum();
        // Release largest-first until at or below the threshold.
        self.free_list
            .sort_by_key(|b| std::cmp::Reverse(b.capacity));
        while free_bytes > threshold {
            let Some(blk) = self.free_list.pop() else {
                break;
            };
            free_bytes -= blk.capacity;
            self.stats.reserved_current = self.stats.reserved_current.saturating_sub(blk.capacity);
        }
    }

    /// Explicitly trim the pool to retain at most `min_bytes_to_keep` free bytes
    /// (`cudaMemPoolTrimTo`).  First retires any pending frees that have
    /// completed, then releases the excess.
    pub fn trim_to(&mut self, min_bytes_to_keep: usize) {
        self.retire_completed();
        self.free_list
            .sort_by_key(|b| std::cmp::Reverse(b.capacity));
        let mut free_bytes: usize = self.free_list.iter().map(|b| b.capacity).sum();
        while free_bytes > min_bytes_to_keep {
            let Some(blk) = self.free_list.pop() else {
                break;
            };
            free_bytes -= blk.capacity;
            self.stats.reserved_current = self.stats.reserved_current.saturating_sub(blk.capacity);
        }
    }

    /// Reset the reserved / used high-water marks to the current values
    /// (`cudaMemPoolAttrReserved/UsedMemHigh` are resettable counters).
    pub fn reset_peak_stats(&mut self) {
        self.stats.reserved_high = self.stats.reserved_current;
        self.stats.used_high = self.stats.used_current;
    }

    /// Number of immediately-reusable free blocks (test/inspection helper).
    #[must_use]
    pub fn free_block_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of stream-ordered frees not yet retired (test/inspection helper).
    #[must_use]
    pub fn pending_free_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of live allocations (test/inspection helper).
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_driver::ffi::CUstream;

    /// Build a distinct non-default stream token for the model.
    fn stream(token: usize) -> CudaStream {
        // SAFETY: the model never dereferences the handle; it is used purely as
        // a stable u64 token to key per-stream clocks.
        unsafe { CudaStream::from_raw(CUstream(token as *mut std::ffi::c_void)) }
    }

    #[test]
    fn malloc_zero_returns_null() {
        let mut pool = MemPool::new(0);
        assert_eq!(pool.malloc_async(0, CudaStream::DEFAULT).unwrap_or(1), 0);
    }

    #[test]
    fn malloc_rounds_up_and_tracks_used() {
        let mut pool = MemPool::new(0);
        let p = pool.malloc_async(100, stream(1)).expect("alloc");
        assert_ne!(p, 0);
        // 100 rounds up to one 512-byte granule.
        assert_eq!(pool.stats().used_current, 512);
        assert_eq!(pool.stats().reserved_current, 512);
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn free_requires_live_pointer() {
        let mut pool = MemPool::new(0);
        assert_eq!(
            pool.free_async(0xDEAD_BEEF, stream(1)),
            Err(CudaRtError::InvalidDevicePointer)
        );
    }

    #[test]
    fn freed_block_is_not_reusable_until_stream_synced() {
        let mut pool = MemPool::new(0);
        // Keep everything on the free list so we can observe reuse.
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let s = stream(1);
        let p = pool.malloc_async(512, s).expect("alloc");
        pool.free_async(p, s).expect("free");
        // Free is pending — not yet on the free list.
        assert_eq!(pool.pending_free_count(), 1);
        assert_eq!(pool.free_block_count(), 0);
        // After the stream is synchronised the free retires.
        pool.synchronize(s);
        assert_eq!(pool.pending_free_count(), 0);
        assert_eq!(pool.free_block_count(), 1);
    }

    #[test]
    fn freed_block_of_right_size_is_reused() {
        let mut pool = MemPool::new(0);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let s = stream(1);
        let p1 = pool.malloc_async(2048, s).expect("alloc1");
        let reserved_after_first = pool.stats().reserved_current;
        pool.free_async(p1, s).expect("free");
        pool.synchronize(s);
        assert_eq!(pool.free_block_count(), 1);
        // A same-size request must reuse the freed block — reserved must NOT grow
        // and the returned pointer must equal the recycled address.
        let p2 = pool.malloc_async(2048, s).expect("alloc2");
        assert_eq!(p2, p1);
        assert_eq!(pool.stats().reserved_current, reserved_after_first);
        assert_eq!(pool.free_block_count(), 0);
    }

    #[test]
    fn best_fit_picks_smallest_sufficient_block() {
        let mut pool = MemPool::new(0);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let s = stream(1);
        let small = pool.malloc_async(512, s).expect("small");
        let large = pool.malloc_async(8192, s).expect("large");
        pool.free_async(small, s).expect("free small");
        pool.free_async(large, s).expect("free large");
        pool.synchronize(s);
        assert_eq!(pool.free_block_count(), 2);
        // Request 512 → must reuse the small block, leaving the large one free.
        let reuse = pool.malloc_async(512, s).expect("reuse");
        assert_eq!(reuse, small);
        assert_eq!(pool.free_block_count(), 1);
    }

    #[test]
    fn release_threshold_zero_returns_memory_on_sync() {
        // Default threshold 0 → the free list is emptied (reserved drops) at sync.
        let mut pool = MemPool::new(0);
        let s = stream(1);
        let p = pool.malloc_async(4096, s).expect("alloc");
        assert_eq!(pool.stats().reserved_current, 4096);
        pool.free_async(p, s).expect("free");
        pool.synchronize(s);
        // With a 0 threshold the trimmed pool reserves nothing.
        assert_eq!(pool.free_block_count(), 0);
        assert_eq!(pool.stats().reserved_current, 0);
    }

    #[test]
    fn trim_to_retains_requested_bytes() {
        let mut pool = MemPool::new(0);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let s = stream(1);
        let a = pool.malloc_async(4096, s).expect("a");
        let b = pool.malloc_async(4096, s).expect("b");
        pool.free_async(a, s).expect("free a");
        pool.free_async(b, s).expect("free b");
        pool.synchronize(s);
        assert_eq!(pool.stats().reserved_current, 8192);
        // Keep only ~4 KiB; one block must be released back.
        pool.trim_to(4096);
        assert_eq!(pool.free_block_count(), 1);
        assert_eq!(pool.stats().reserved_current, 4096);
    }

    #[test]
    fn cross_stream_free_not_reused_until_its_stream_syncs() {
        // A block freed on stream A is not reusable from stream B's perspective
        // until A is synchronised — only then does it land on the free list.
        let mut pool = MemPool::new(0);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let a = stream(1);
        let b = stream(2);
        let p = pool.malloc_async(1024, a).expect("alloc on A");
        pool.free_async(p, a).expect("free on A");
        // Synchronising B does NOT retire a free recorded on A.
        pool.synchronize(b);
        assert_eq!(pool.pending_free_count(), 1);
        assert_eq!(pool.free_block_count(), 0);
        // Synchronising A retires it.
        pool.synchronize(a);
        assert_eq!(pool.free_block_count(), 1);
    }

    #[test]
    fn attribute_round_trip_and_validation() {
        let mut pool = MemPool::new(3);
        assert_eq!(pool.device(), 3);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, 1 << 20)
            .expect("set threshold");
        assert_eq!(pool.get_attribute(MemPoolAttr::ReleaseThreshold), 1 << 20);
        pool.set_attribute(MemPoolAttr::ReuseAllowOpportunistic, 0)
            .expect("clear");
        assert_eq!(pool.get_attribute(MemPoolAttr::ReuseAllowOpportunistic), 0);
        // Out-of-range boolean value is rejected.
        assert_eq!(
            pool.set_attribute(MemPoolAttr::ReuseAllowOpportunistic, 2),
            Err(CudaRtError::InvalidValue)
        );
    }

    #[test]
    fn high_water_marks_track_peak() {
        let mut pool = MemPool::new(0);
        pool.set_attribute(MemPoolAttr::ReleaseThreshold, u64::MAX)
            .expect("set");
        let s = stream(1);
        let a = pool.malloc_async(4096, s).expect("a");
        let b = pool.malloc_async(4096, s).expect("b");
        assert_eq!(pool.stats().used_high, 8192);
        pool.free_async(a, s).expect("free a");
        pool.free_async(b, s).expect("free b");
        pool.synchronize(s);
        // Used drops but the high-water mark is retained until reset.
        assert_eq!(pool.stats().used_current, 0);
        assert_eq!(pool.stats().used_high, 8192);
        pool.reset_peak_stats();
        assert_eq!(pool.stats().used_high, 0);
    }
}
