//! Faithful CPU model of CUDA stream-ordered memory allocation.
//!
//! This module implements the *semantics* of the CUDA stream-ordered
//! allocator (`cuMemAllocAsync` / `cuMemFreeAsync` / the `cuMemPool*` family)
//! without any GPU.  It is a self-contained, deterministic simulation that the
//! [`StreamMemoryPool`](crate::stream_ordered_alloc::StreamMemoryPool) drives
//! on every platform so that the allocator's behaviour can be exercised and
//! reasoned about on a plain CPU.
//!
//! # What "stream-ordered" means
//!
//! In CUDA, `cuMemAllocAsync(ptr, size, stream)` returns a pointer
//! *immediately* to the host, but the memory is only **valid on the GPU once
//! the stream reaches the allocation point**.  Symmetrically,
//! `cuMemFreeAsync(ptr, stream)` records a free that only takes effect once
//! the stream reaches the free point — until then the memory is still in use
//! by earlier work on that stream.  A pointer freed on stream `A` may be
//! reused by a later allocation, and that reuse is only safe with respect to
//! work on another stream `B` if `B` has been ordered after the free (e.g. via
//! an event).
//!
//! # The model
//!
//! Each stream is modelled by a `StreamClock`: a monotonically increasing
//! *submit* counter (the position at which the next operation is enqueued) and
//! a *reached* counter (how far the stream has actually executed).  Submitting
//! an operation returns its sequence number; the operation is **complete** once
//! `reached >= seq`.  Advancing a stream (the model's analogue of
//! `cuStreamSynchronize`) sets `reached = submit`, retiring every pending
//! operation in FIFO order — exactly the in-order guarantee CUDA gives.
//!
//! Memory is modelled by a flat virtual address space.  Live allocations own a
//! `Block`; a stream-ordered free moves the block onto a *pending-free*
//! queue tagged with the freeing stream and the free's sequence number.  When
//! that stream reaches the free point the block is returned to the pool's free
//! list and becomes eligible for reuse by a later same-or-larger request
//! (first-fit over the free list, preferring exact size).  `reserved` bytes
//! count everything the pool has carved from the (virtual) device, whereas
//! `used` bytes count only currently-live allocations; trimming releases
//! free-list bytes above the release threshold back to the device.

use std::collections::HashMap;

use crate::error::{CudaError, CudaResult};

/// Identifier of a stream within the model.
///
/// Derived from a real [`Stream`](crate::stream::Stream)'s raw handle (or any
/// stable `u64` token), this lets the CPU model order operations per stream
/// without owning the stream itself.  The reserved value [`StreamOrderId::NULL`]
/// models the default (NULL) stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamOrderId(pub u64);

impl StreamOrderId {
    /// The default (NULL) stream identifier.
    pub const NULL: StreamOrderId = StreamOrderId(0);

    /// Returns the raw token backing this identifier.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl From<u64> for StreamOrderId {
    #[inline]
    fn from(value: u64) -> Self {
        StreamOrderId(value)
    }
}

/// Per-stream logical clock modelling in-order execution.
///
/// `submit` is the sequence number that will be handed to the *next* operation
/// enqueued on the stream; `reached` is how far the stream has executed.  An
/// operation with sequence `s` is complete once `reached >= s`.
#[derive(Debug, Clone, Copy, Default)]
struct StreamClock {
    /// Sequence number of the next operation to be submitted.
    submit: u64,
    /// Sequence number the stream has executed up to (inclusive of all
    /// operations with a strictly smaller sequence number).
    reached: u64,
}

impl StreamClock {
    /// Enqueue an operation and return its sequence number.
    fn enqueue(&mut self) -> u64 {
        let seq = self.submit;
        self.submit = self.submit.saturating_add(1);
        seq
    }

    /// Advance the stream to the head (model of `cuStreamSynchronize`):
    /// every submitted operation is now complete.
    fn advance_to_head(&mut self) {
        self.reached = self.submit;
    }

    /// Returns `true` once the operation with sequence `seq` has executed.
    fn has_reached(&self, seq: u64) -> bool {
        self.reached > seq
    }
}

/// A contiguous virtual memory block tracked by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    /// Virtual device address (non-zero).
    ptr: u64,
    /// Capacity of the block in bytes (may exceed the request that owns it
    /// when the block was reused from the free list).
    capacity: usize,
}

/// A stream-ordered free awaiting completion of its stream.
#[derive(Debug, Clone, Copy)]
struct PendingFree {
    /// The block being freed.
    block: Block,
    /// The stream the free was ordered on.
    stream: StreamOrderId,
    /// The free's sequence number on `stream`.
    seq: u64,
}

/// Record of a live allocation produced by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAllocation {
    /// Virtual device pointer (non-zero).
    pub ptr: u64,
    /// Bytes requested by the caller (`<= capacity`).
    pub size: usize,
    /// Capacity of the backing block.
    pub capacity: usize,
    /// Stream the allocation was ordered on.
    pub stream: StreamOrderId,
    /// Sequence number at which the allocation becomes valid on `stream`.
    pub ready_seq: u64,
}

/// Configuration knobs the model needs from the pool.
#[derive(Debug, Clone, Copy)]
pub struct ModelLimits {
    /// Hard cap on `reserved` bytes (`0` == unlimited).
    pub max_pool_size: usize,
    /// Bytes kept reserved on an implicit trim / release.
    pub release_threshold: usize,
}

/// The stream-ordered allocation engine.
///
/// All accounting is in *virtual* bytes; no host or device memory is actually
/// reserved.  The engine is deterministic: identical operation sequences
/// produce identical pointers and statistics.
#[derive(Debug)]
pub struct StreamOrderModel {
    /// Per-stream logical clocks.
    clocks: HashMap<StreamOrderId, StreamClock>,
    /// Free blocks available for reuse (returned by completed frees).
    free_list: Vec<Block>,
    /// Frees whose stream has not yet reached the free point.
    pending_frees: Vec<PendingFree>,
    /// Set of pointers that are currently live (alloc'd, not yet freed).
    live: HashMap<u64, Block>,
    /// Next virtual address to hand out for a brand-new block.
    next_addr: u64,
    /// Bytes carved from the (virtual) device — live + free-list + pending.
    reserved: usize,
    /// Bytes belonging to currently-live allocations.
    used: usize,
    /// Peak `reserved` since creation or last reset.
    reserved_high: usize,
    /// Peak `used` since creation or last reset.
    used_high: usize,
    /// Number of currently-live allocations.
    active: usize,
    /// Peak number of concurrent live allocations.
    peak_active: usize,
    /// Pool limits mirrored from the configuration.
    limits: ModelLimits,
}

/// Base of the model's virtual address space (avoids null and small ints).
const VIRTUAL_BASE: u64 = 0x0000_7F00_0000_0000;
/// Allocation granularity — every block is rounded up to this, matching the
/// 512-byte minimum granularity CUDA's stream-ordered allocator uses.
const GRANULARITY: usize = 512;

impl StreamOrderModel {
    /// Create a fresh model with the given pool limits.
    pub fn new(limits: ModelLimits) -> Self {
        Self {
            clocks: HashMap::new(),
            free_list: Vec::new(),
            pending_frees: Vec::new(),
            live: HashMap::new(),
            next_addr: VIRTUAL_BASE,
            reserved: 0,
            used: 0,
            reserved_high: 0,
            used_high: 0,
            active: 0,
            peak_active: 0,
            limits,
        }
    }

    /// Round a request up to the allocation granularity.
    fn align(size: usize) -> usize {
        // size is always >= 1 here; round up to the next multiple of GRANULARITY.
        size.saturating_add(GRANULARITY - 1) / GRANULARITY * GRANULARITY
    }

    /// Mirror an updated release threshold from the pool configuration.
    pub fn set_release_threshold(&mut self, threshold: usize) {
        self.limits.release_threshold = threshold;
    }

    /// Allocate `size` bytes ordered on `stream`.
    ///
    /// Returns a [`ModelAllocation`] describing the (possibly reused) block and
    /// the sequence number at which it becomes valid on the stream.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size == 0`.
    /// * [`CudaError::OutOfMemory`] if satisfying the request would push
    ///   `reserved` past `max_pool_size`.
    pub fn alloc(&mut self, size: usize, stream: StreamOrderId) -> CudaResult<ModelAllocation> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Opportunistically retire any frees whose streams have already
        // advanced, so their blocks are reuse-eligible for this request.
        self.collect_ready_frees();

        let want = Self::align(size);

        // First-fit over the free list, preferring an exact-capacity match so
        // that small requests don't permanently consume large reclaimed blocks.
        let block = if let Some(idx) = self.pick_free_block(want) {
            self.free_list.swap_remove(idx)
        } else {
            // No reusable block — carve a fresh one from the virtual device.
            // This grows `reserved`, so enforce the pool ceiling here.
            if self.limits.max_pool_size > 0
                && self.reserved.saturating_add(want) > self.limits.max_pool_size
            {
                return Err(CudaError::OutOfMemory);
            }
            let ptr = self.next_addr;
            self.next_addr = self.next_addr.saturating_add(want as u64);
            self.reserved = self.reserved.saturating_add(want);
            if self.reserved > self.reserved_high {
                self.reserved_high = self.reserved;
            }
            Block {
                ptr,
                capacity: want,
            }
        };

        // Order the allocation on the stream and mark the block live.
        let ready_seq = self.clock_mut(stream).enqueue();
        self.live.insert(block.ptr, block);
        self.used = self.used.saturating_add(block.capacity);
        if self.used > self.used_high {
            self.used_high = self.used;
        }
        self.active = self.active.saturating_add(1);
        if self.active > self.peak_active {
            self.peak_active = self.active;
        }

        Ok(ModelAllocation {
            ptr: block.ptr,
            size,
            capacity: block.capacity,
            stream,
            ready_seq,
        })
    }

    /// Pick a reusable free block of at least `want` bytes.
    ///
    /// Prefers an exact-capacity match; otherwise returns the first block large
    /// enough (first-fit).  Returns the index into [`Self::free_list`].
    fn pick_free_block(&self, want: usize) -> Option<usize> {
        let mut first_fit: Option<usize> = None;
        for (idx, block) in self.free_list.iter().enumerate() {
            if block.capacity == want {
                return Some(idx);
            }
            if block.capacity > want && first_fit.is_none() {
                first_fit = Some(idx);
            }
        }
        first_fit
    }

    /// Record a stream-ordered free of `ptr` on `stream`.
    ///
    /// The block is not returned to the free list until `stream` reaches the
    /// free point; until then it remains counted in `reserved` but no longer in
    /// `used`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `ptr` is not a live allocation
    ///   (covers double-free and free of a foreign pointer).
    pub fn free(&mut self, ptr: u64, stream: StreamOrderId) -> CudaResult<()> {
        let block = self.live.remove(&ptr).ok_or(CudaError::InvalidValue)?;

        self.used = self.used.saturating_sub(block.capacity);
        self.active = self.active.saturating_sub(1);

        let seq = self.clock_mut(stream).enqueue();
        self.pending_frees.push(PendingFree { block, stream, seq });

        // A free that is already ordered-complete (its stream has been advanced
        // past this point) can be reclaimed straight away.
        self.collect_ready_frees();
        Ok(())
    }

    /// Returns `true` if `ptr` is a currently-live allocation in this model.
    pub fn is_live(&self, ptr: u64) -> bool {
        self.live.contains_key(&ptr)
    }

    /// Advance a stream to its head: every operation submitted so far is now
    /// complete (model of `cuStreamSynchronize`).  Completed frees are
    /// reclaimed into the free list.
    pub fn synchronize(&mut self, stream: StreamOrderId) {
        self.clock_mut(stream).advance_to_head();
        self.collect_ready_frees();
    }

    /// Returns `true` if the allocation `alloc` is valid for use on its own
    /// ordering stream (i.e. the stream has executed past the allocation
    /// point).  This is the same-stream visibility rule.
    pub fn is_ready_same_stream(&self, alloc: &ModelAllocation) -> bool {
        self.clocks
            .get(&alloc.stream)
            .is_some_and(|c| c.has_reached(alloc.ready_seq))
    }

    /// Returns `true` if `alloc` (made on stream `A`) is safe to use on
    /// `consumer` (stream `B`) given that `B` has been ordered after sequence
    /// `wait_seq` on `A` (the sequence captured by an event `B` waited on).
    ///
    /// Cross-stream use is only safe when the event was recorded **after** the
    /// allocation became ready (`wait_seq > ready_seq`) and `consumer != A`
    /// degenerates to the same-stream rule when they are equal.
    pub fn is_ready_cross_stream(
        &self,
        alloc: &ModelAllocation,
        consumer: StreamOrderId,
        wait_seq: u64,
    ) -> bool {
        if consumer == alloc.stream {
            return self.is_ready_same_stream(alloc);
        }
        // The consumer stream observes the allocation only if the event it
        // waited on was recorded at or after the allocation point.
        wait_seq > alloc.ready_seq
    }

    /// Record an event on `stream`, returning the sequence number it captures.
    ///
    /// A later `cuStreamWaitEvent` on another stream is ordered after every
    /// operation submitted on `stream` before this point — modelled by handing
    /// back the stream's current submit position.
    pub fn record_event(&mut self, stream: StreamOrderId) -> u64 {
        self.clock_mut(stream).submit
    }

    /// Reclaim every pending free whose stream has reached the free point.
    fn collect_ready_frees(&mut self) {
        let mut still_pending = Vec::with_capacity(self.pending_frees.len());
        // Take ownership to avoid borrowing `self` across the closure.
        let drained = std::mem::take(&mut self.pending_frees);
        for pf in drained {
            let reached = self
                .clocks
                .get(&pf.stream)
                .is_some_and(|c| c.has_reached(pf.seq));
            if reached {
                self.free_list.push(pf.block);
            } else {
                still_pending.push(pf);
            }
        }
        self.pending_frees = still_pending;
        // After reclaiming, optionally shrink the free list down to the
        // release threshold (implicit release, mirroring driver behaviour).
        self.release_excess();
    }

    /// Release free-list bytes that exceed the release threshold back to the
    /// (virtual) device, lowering `reserved`.
    fn release_excess(&mut self) {
        // Bytes that must stay reserved regardless: live + pending-free bytes.
        let pending_bytes: usize = self.pending_frees.iter().map(|p| p.block.capacity).sum();
        let pinned = self.used.saturating_add(pending_bytes);
        let keep_floor = self.limits.release_threshold.max(pinned);

        while self.reserved > keep_floor {
            let Some(block) = self.free_list.pop() else {
                break;
            };
            // Dropping the block from the free list returns its bytes to the
            // device.  We do not recycle the virtual address — fresh blocks
            // always take a new address, which keeps pointers unambiguous.
            self.reserved = self.reserved.saturating_sub(block.capacity);
        }
    }

    /// Explicit trim (model of `cuMemPoolTrimTo`): keep at least
    /// `min_bytes_to_keep` reserved, releasing free-list blocks above that.
    pub fn trim_to(&mut self, min_bytes_to_keep: usize) {
        // First make sure completed frees are on the free list.
        self.collect_ready_frees();

        let pending_bytes: usize = self.pending_frees.iter().map(|p| p.block.capacity).sum();
        let pinned = self.used.saturating_add(pending_bytes);
        let keep_floor = min_bytes_to_keep.max(pinned);

        while self.reserved > keep_floor {
            let Some(block) = self.free_list.pop() else {
                break;
            };
            self.reserved = self.reserved.saturating_sub(block.capacity);
        }
    }

    /// Reset the peak (`reserved_high` / `used_high`) statistics to the current
    /// values.
    pub fn reset_peaks(&mut self) {
        self.reserved_high = self.reserved;
        self.used_high = self.used;
        self.peak_active = self.active;
    }

    /// Current `reserved` byte count.
    #[inline]
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Current `used` byte count.
    #[inline]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Peak `reserved` byte count.
    #[inline]
    pub fn reserved_high(&self) -> usize {
        self.reserved_high
    }

    /// Peak `used` byte count.
    #[inline]
    pub fn used_high(&self) -> usize {
        self.used_high
    }

    /// Number of currently-live allocations.
    #[inline]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Peak number of concurrent live allocations.
    #[inline]
    pub fn peak_active(&self) -> usize {
        self.peak_active
    }

    /// Number of reuse-eligible free blocks currently held.
    #[inline]
    pub fn free_block_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of pending (not-yet-complete) stream-ordered frees.
    #[inline]
    pub fn pending_free_count(&self) -> usize {
        self.pending_frees.len()
    }

    /// Get a mutable handle to a stream's clock, creating it on first use.
    fn clock_mut(&mut self, stream: StreamOrderId) -> &mut StreamClock {
        self.clocks.entry(stream).or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max: usize, release: usize) -> ModelLimits {
        ModelLimits {
            max_pool_size: max,
            release_threshold: release,
        }
    }

    #[test]
    fn alloc_rounds_up_to_granularity() {
        let mut m = StreamOrderModel::new(limits(0, 0));
        let a = m.alloc(1, StreamOrderId::NULL).expect("alloc");
        assert_eq!(a.size, 1);
        assert_eq!(a.capacity, GRANULARITY);
        assert_eq!(m.reserved(), GRANULARITY);
        assert_eq!(m.used(), GRANULARITY);
        assert_ne!(a.ptr, 0);
    }

    #[test]
    fn zero_size_rejected() {
        let mut m = StreamOrderModel::new(limits(0, 0));
        assert_eq!(
            m.alloc(0, StreamOrderId::NULL),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn free_of_foreign_pointer_rejected() {
        let mut m = StreamOrderModel::new(limits(0, 0));
        assert_eq!(
            m.free(0xDEAD_BEEF, StreamOrderId::NULL),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn double_free_rejected() {
        let mut m = StreamOrderModel::new(limits(0, 0));
        let s = StreamOrderId(7);
        let a = m.alloc(256, s).expect("alloc");
        assert!(m.free(a.ptr, s).is_ok());
        assert_eq!(m.free(a.ptr, s), Err(CudaError::InvalidValue));
    }

    #[test]
    fn pending_free_holds_block_until_stream_advances() {
        // Keep everything reserved so the released bytes don't vanish.
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(1);
        let a = m.alloc(512, s).expect("alloc");
        // Submit one more op so the free does not sit at the stream head with
        // reached already past it.
        let _b = m.alloc(512, s).expect("alloc2");
        m.free(a.ptr, s).expect("free");
        // The freeing op is pending: stream has not been advanced.
        assert_eq!(m.pending_free_count(), 1);
        assert_eq!(m.free_block_count(), 0);
        // Advancing the stream retires the free → block becomes reusable.
        m.synchronize(s);
        assert_eq!(m.pending_free_count(), 0);
        assert_eq!(m.free_block_count(), 1);
    }

    #[test]
    fn freed_block_is_reused_by_same_size_alloc() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(2);
        let a = m.alloc(1024, s).expect("alloc");
        let ptr_a = a.ptr;
        let reserved_after_first = m.reserved();
        m.free(a.ptr, s).expect("free");
        m.synchronize(s); // retire the free
        // A same-size allocation must reuse the freed block (same pointer) and
        // must not grow reserved.
        let b = m.alloc(1024, s).expect("alloc reuse");
        assert_eq!(b.ptr, ptr_a, "freed block should be reused");
        assert_eq!(m.reserved(), reserved_after_first, "no new reservation");
        assert_eq!(m.free_block_count(), 0);
    }

    #[test]
    fn same_stream_visibility_rule() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(3);
        let a = m.alloc(64, s).expect("alloc");
        // Not yet executed: stream has not advanced past the alloc point.
        assert!(!m.is_ready_same_stream(&a));
        m.synchronize(s);
        assert!(m.is_ready_same_stream(&a));
    }

    #[test]
    fn cross_stream_requires_event_after_alloc() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let producer = StreamOrderId(10);
        let consumer = StreamOrderId(20);
        let a = m.alloc(128, producer).expect("alloc");
        // Event recorded BEFORE the alloc would have captured seq 0.
        let early_wait = 0u64;
        assert!(!m.is_ready_cross_stream(&a, consumer, early_wait));
        // Event recorded AFTER the alloc captures the post-alloc submit pos.
        let late_wait = m.record_event(producer);
        assert!(late_wait > a.ready_seq);
        assert!(m.is_ready_cross_stream(&a, consumer, late_wait));
    }

    #[test]
    fn reserved_vs_used_accounting_consistent() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(4);
        let a = m.alloc(1000, s).expect("a");
        let b = m.alloc(2000, s).expect("b");
        let total = a.capacity + b.capacity;
        assert_eq!(m.used(), total);
        assert_eq!(m.reserved(), total);
        m.free(a.ptr, s).expect("free a");
        // used drops immediately; reserved stays until trimmed.
        assert_eq!(m.used(), b.capacity);
        assert_eq!(m.reserved(), total);
        m.synchronize(s);
        // Block is on the free list but threshold keeps it reserved.
        assert_eq!(m.reserved(), total);
        assert_eq!(m.free_block_count(), 1);
    }

    #[test]
    fn trim_releases_free_blocks_above_threshold() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(5);
        let a = m.alloc(4096, s).expect("a");
        let reserved_full = m.reserved();
        m.free(a.ptr, s).expect("free");
        m.synchronize(s);
        assert_eq!(m.free_block_count(), 1);
        assert_eq!(m.reserved(), reserved_full);
        // Trim to zero releases the free block.
        m.trim_to(0);
        assert_eq!(m.free_block_count(), 0);
        assert_eq!(m.reserved(), 0);
    }

    #[test]
    fn release_threshold_keeps_some_reserved() {
        let mut m = StreamOrderModel::new(limits(0, 4096));
        let s = StreamOrderId(6);
        let a = m.alloc(4096, s).expect("a");
        m.free(a.ptr, s).expect("free");
        m.synchronize(s);
        // Threshold == 4096 keeps the block reserved on the implicit release.
        assert_eq!(m.reserved(), 4096);
        assert_eq!(m.free_block_count(), 1);
    }

    #[test]
    fn max_pool_size_enforced_on_fresh_blocks() {
        let mut m = StreamOrderModel::new(limits(1024, 0));
        let s = StreamOrderId::NULL;
        assert!(m.alloc(1024, s).is_ok());
        // A second fresh block would exceed the 1024-byte ceiling.
        assert_eq!(m.alloc(1, s), Err(CudaError::OutOfMemory));
    }

    #[test]
    fn larger_request_reuses_oversized_free_block() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(8);
        // Free a 2048-byte block, then request 1024 — first-fit reuse.
        let big = m.alloc(2048, s).expect("big");
        let big_ptr = big.ptr;
        m.free(big.ptr, s).expect("free big");
        m.synchronize(s);
        let small = m.alloc(1024, s).expect("small reuse");
        assert_eq!(small.ptr, big_ptr, "oversized block reused");
        assert_eq!(small.capacity, 2048, "capacity retained from block");
    }

    #[test]
    fn peak_stats_track_and_reset() {
        let mut m = StreamOrderModel::new(limits(0, usize::MAX));
        let s = StreamOrderId(9);
        let a = m.alloc(1024, s).expect("a");
        let _b = m.alloc(2048, s).expect("b");
        assert_eq!(m.peak_active(), 2);
        assert_eq!(m.used_high(), a.capacity + 2048);
        m.free(a.ptr, s).expect("free a");
        m.synchronize(s);
        m.reset_peaks();
        assert_eq!(m.peak_active(), 1);
        assert_eq!(m.used_high(), m.used());
    }
}
