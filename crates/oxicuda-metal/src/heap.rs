//! `MTLHeap`-style suballocator and unified-memory budgeting.
//!
//! Large model weights can exceed comfortable unified-memory pressure if every
//! tensor is an independent `MTLBuffer`.  Metal's answer is `MTLHeap`: one big
//! backing allocation from which individual resources are sub-allocated with
//! the right alignment.  This module implements the *placement logic* — a
//! first-fit free-list suballocator with coalescing — as a pure data structure
//! that mirrors `MTLHeap`'s `newBufferWithLength:options:offset:` placement
//! model.  It is fully unit-testable without a device; on macOS the backend can
//! drive a real `MTLHeap` using the offsets this planner produces.

use crate::error::{MetalError, MetalResult};
use crate::storage::{METAL_BUFFER_ALIGNMENT, MetalStorageMode, align_up};

// ─── HeapBlock ─────────────────────────────────────────────────────────────────

/// One sub-allocation placed within a heap, identified by an opaque handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapBlock {
    /// Opaque allocation handle (unique within the owning heap).
    pub handle: u64,
    /// Byte offset of this block from the start of the heap.
    pub offset: usize,
    /// Aligned byte length of this block.
    pub size: usize,
}

/// A free region inside the heap (offset + length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeSpan {
    offset: usize,
    size: usize,
}

// ─── MetalHeapAllocator ────────────────────────────────────────────────────────

/// A first-fit suballocator over a fixed-size heap.
///
/// All allocations are rounded up to [`METAL_BUFFER_ALIGNMENT`] and placed at
/// aligned offsets.  Freed spans are coalesced with adjacent free spans so the
/// heap does not fragment under alloc/free churn.
#[derive(Debug, Clone)]
pub struct MetalHeapAllocator {
    total_size: usize,
    storage_mode: MetalStorageMode,
    free_spans: Vec<FreeSpan>,
    /// Live (offset, size) per handle, for `free`/coalesce.
    live: Vec<(u64, usize, usize)>,
    next_handle: u64,
    used: usize,
}

impl MetalHeapAllocator {
    /// Create a heap of `total_size` bytes (rounded up to alignment) backed by
    /// the given storage mode.
    pub fn new(total_size: usize, storage_mode: MetalStorageMode) -> Self {
        let total = align_up(total_size, METAL_BUFFER_ALIGNMENT);
        Self {
            total_size: total,
            storage_mode,
            free_spans: if total == 0 {
                Vec::new()
            } else {
                vec![FreeSpan {
                    offset: 0,
                    size: total,
                }]
            },
            live: Vec::new(),
            next_handle: 1,
            used: 0,
        }
    }

    /// Total heap capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.total_size
    }

    /// Currently allocated bytes (sum of aligned block sizes).
    pub fn used(&self) -> usize {
        self.used
    }

    /// Bytes available across all free spans (may be fragmented).
    pub fn available(&self) -> usize {
        self.total_size - self.used
    }

    /// Storage mode this heap was created with.
    pub fn storage_mode(&self) -> MetalStorageMode {
        self.storage_mode
    }

    /// Size of the largest single contiguous free span.
    pub fn largest_free_span(&self) -> usize {
        self.free_spans.iter().map(|s| s.size).max().unwrap_or(0)
    }

    /// Place a sub-allocation of `size` bytes using first-fit.
    ///
    /// Returns [`MetalError::OutOfMemory`] when no single free span is large
    /// enough (even if total free bytes would suffice — Metal heaps require a
    /// contiguous region).
    pub fn allocate(&mut self, size: usize) -> MetalResult<HeapBlock> {
        if size == 0 {
            return Err(MetalError::InvalidArgument(
                "heap allocation size must be > 0".into(),
            ));
        }
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);

        // First-fit search.
        let idx = self
            .free_spans
            .iter()
            .position(|s| s.size >= aligned)
            .ok_or(MetalError::OutOfMemory)?;

        let span = self.free_spans[idx];
        let offset = span.offset;

        if span.size == aligned {
            self.free_spans.remove(idx);
        } else {
            self.free_spans[idx] = FreeSpan {
                offset: span.offset + aligned,
                size: span.size - aligned,
            };
        }

        let handle = self.next_handle;
        self.next_handle += 1;
        self.live.push((handle, offset, aligned));
        self.used += aligned;

        Ok(HeapBlock {
            handle,
            offset,
            size: aligned,
        })
    }

    /// Free a previously allocated block by handle, coalescing free spans.
    ///
    /// Unknown handles return [`MetalError::InvalidArgument`].
    pub fn free(&mut self, handle: u64) -> MetalResult<()> {
        let pos = self
            .live
            .iter()
            .position(|(h, _, _)| *h == handle)
            .ok_or_else(|| MetalError::InvalidArgument(format!("unknown heap handle {handle}")))?;
        let (_, offset, size) = self.live.remove(pos);
        self.used -= size;
        self.insert_and_coalesce(FreeSpan { offset, size });
        Ok(())
    }

    /// Reset the heap to fully free, dropping all live allocations.
    pub fn reset(&mut self) {
        self.live.clear();
        self.used = 0;
        self.free_spans = if self.total_size == 0 {
            Vec::new()
        } else {
            vec![FreeSpan {
                offset: 0,
                size: self.total_size,
            }]
        };
    }

    /// Insert a freed span and merge it with any adjacent free spans.
    fn insert_and_coalesce(&mut self, span: FreeSpan) {
        self.free_spans.push(span);
        self.free_spans.sort_by_key(|s| s.offset);

        let mut merged: Vec<FreeSpan> = Vec::with_capacity(self.free_spans.len());
        for s in self.free_spans.drain(..) {
            match merged.last_mut() {
                Some(prev) if prev.offset + prev.size == s.offset => {
                    prev.size += s.size;
                }
                _ => merged.push(s),
            }
        }
        self.free_spans = merged;
    }
}

// ─── MemoryBudget ──────────────────────────────────────────────────────────────

/// Tracks unified-memory pressure against a soft budget.
///
/// Apple Silicon shares one memory pool between CPU and GPU; allocating more
/// than the recommended working-set size triggers paging and stalls.  This
/// tracker lets the backend reserve/release against a budget and report when a
/// request would exceed it, so callers can fall back to streaming or spill.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    limit: usize,
    reserved: usize,
}

impl MemoryBudget {
    /// Create a budget tracker with the given soft `limit` in bytes.
    pub fn new(limit: usize) -> Self {
        Self { limit, reserved: 0 }
    }

    /// The configured soft limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved.
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Bytes still available under the budget (saturating at zero).
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.reserved)
    }

    /// `true` when reserving `bytes` more would stay within budget.
    pub fn can_reserve(&self, bytes: usize) -> bool {
        self.reserved.saturating_add(bytes) <= self.limit
    }

    /// Reserve `bytes`, returning [`MetalError::OutOfMemory`] if it would
    /// exceed the soft budget.
    pub fn reserve(&mut self, bytes: usize) -> MetalResult<()> {
        if !self.can_reserve(bytes) {
            return Err(MetalError::OutOfMemory);
        }
        self.reserved += bytes;
        Ok(())
    }

    /// Release `bytes` previously reserved (saturating at zero).
    pub fn release(&mut self, bytes: usize) {
        self.reserved = self.reserved.saturating_sub(bytes);
    }

    /// Fraction of the budget currently in use, in `[0.0, 1.0]`.
    pub fn pressure(&self) -> f32 {
        if self.limit == 0 {
            return 1.0;
        }
        (self.reserved as f32 / self.limit as f32).min(1.0)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_rounds_capacity_to_alignment() {
        let h = MetalHeapAllocator::new(1000, MetalStorageMode::Private);
        assert_eq!(h.capacity(), 1024); // 1000 → 1024
        assert_eq!(h.used(), 0);
        assert_eq!(h.available(), 1024);
        assert_eq!(h.storage_mode(), MetalStorageMode::Private);
    }

    #[test]
    fn allocate_aligns_offsets_and_sizes() {
        let mut h = MetalHeapAllocator::new(4096, MetalStorageMode::Shared);
        let a = h.allocate(100).expect("alloc a");
        assert_eq!(a.offset, 0);
        assert_eq!(a.size, 256); // 100 → 256
        let b = h.allocate(300).expect("alloc b");
        assert_eq!(b.offset, 256); // placed after a's aligned block
        assert_eq!(b.size, 512);
        assert_eq!(h.used(), 256 + 512);
    }

    #[test]
    fn allocate_zero_is_error() {
        let mut h = MetalHeapAllocator::new(1024, MetalStorageMode::Shared);
        assert!(matches!(h.allocate(0), Err(MetalError::InvalidArgument(_))));
    }

    #[test]
    fn out_of_memory_when_no_span_fits() {
        let mut h = MetalHeapAllocator::new(512, MetalStorageMode::Private);
        let _a = h.allocate(256).expect("alloc 256");
        // Only 256 left; 300 → 512 aligned won't fit.
        assert!(matches!(h.allocate(300), Err(MetalError::OutOfMemory)));
    }

    #[test]
    fn free_then_realloc_reuses_space() {
        let mut h = MetalHeapAllocator::new(1024, MetalStorageMode::Private);
        let a = h.allocate(256).expect("a");
        let _b = h.allocate(256).expect("b");
        h.free(a.handle).expect("free a");
        assert_eq!(h.used(), 256);
        // Re-allocate; first-fit should reuse a's slot at offset 0.
        let c = h.allocate(256).expect("c");
        assert_eq!(c.offset, 0);
    }

    #[test]
    fn coalesce_adjacent_free_spans() {
        let mut h = MetalHeapAllocator::new(768, MetalStorageMode::Private);
        let a = h.allocate(256).expect("a");
        let b = h.allocate(256).expect("b");
        let c = h.allocate(256).expect("c");
        assert_eq!(h.largest_free_span(), 0);
        // Free the middle then the neighbours — should all coalesce into one span.
        h.free(b.handle).expect("free b");
        h.free(a.handle).expect("free a");
        h.free(c.handle).expect("free c");
        assert_eq!(h.largest_free_span(), 768);
        assert_eq!(h.available(), 768);
    }

    #[test]
    fn unknown_handle_free_errors() {
        let mut h = MetalHeapAllocator::new(1024, MetalStorageMode::Shared);
        assert!(matches!(h.free(999), Err(MetalError::InvalidArgument(_))));
    }

    #[test]
    fn reset_clears_all() {
        let mut h = MetalHeapAllocator::new(1024, MetalStorageMode::Shared);
        let _a = h.allocate(256).expect("a");
        let _b = h.allocate(256).expect("b");
        h.reset();
        assert_eq!(h.used(), 0);
        assert_eq!(h.largest_free_span(), 1024);
    }

    #[test]
    fn fragmentation_blocks_large_alloc() {
        // Total free can exceed request but no contiguous span fits.
        let mut h = MetalHeapAllocator::new(768, MetalStorageMode::Private);
        let a = h.allocate(256).expect("a");
        let _b = h.allocate(256).expect("b");
        let c = h.allocate(256).expect("c");
        h.free(a.handle).expect("free a");
        h.free(c.handle).expect("free c");
        // 512 bytes free but split into two 256 spans; a 512 request fails.
        assert_eq!(h.available(), 512);
        assert!(matches!(h.allocate(512), Err(MetalError::OutOfMemory)));
    }

    // ── MemoryBudget ──

    #[test]
    fn budget_reserve_and_release() {
        let mut b = MemoryBudget::new(1000);
        assert_eq!(b.remaining(), 1000);
        b.reserve(600).expect("reserve 600");
        assert_eq!(b.reserved(), 600);
        assert_eq!(b.remaining(), 400);
        assert!(b.can_reserve(400));
        assert!(!b.can_reserve(401));
        b.release(200);
        assert_eq!(b.reserved(), 400);
    }

    #[test]
    fn budget_over_limit_errors() {
        let mut b = MemoryBudget::new(100);
        assert!(matches!(b.reserve(200), Err(MetalError::OutOfMemory)));
        assert_eq!(b.reserved(), 0);
    }

    #[test]
    fn budget_pressure() {
        let mut b = MemoryBudget::new(1000);
        b.reserve(250).expect("reserve");
        assert!((b.pressure() - 0.25).abs() < 1e-6);
        // Zero-limit budget is always fully pressured.
        let z = MemoryBudget::new(0);
        assert_eq!(z.pressure(), 1.0);
        assert!(!z.can_reserve(1));
    }

    #[test]
    fn budget_release_saturates() {
        let mut b = MemoryBudget::new(1000);
        b.reserve(100).expect("reserve");
        b.release(500); // more than reserved → clamp to 0
        assert_eq!(b.reserved(), 0);
    }
}
