//! Host-side stream-ordered HIP memory pool model (`hipMemPool_t`).
//!
//! Models the *allocation bookkeeping* of `hipMallocAsync` / `hipFreeAsync`
//! over a `hipMemPool_t`: a suballocator that carves device-address ranges out
//! of a backing arena, rounds requests up to the HIP allocation granularity,
//! coalesces freed ranges, and tracks a high-water mark — all without a HIP
//! runtime, so it is fully CPU-testable.
//!
//! This is *modeling* only: the "device addresses" are synthetic offsets into a
//! virtual arena base, never dereferenced.

use crate::error::{RocmError, RocmResult};

/// The natural device-memory allocation alignment HIP uses for `hipMalloc`
/// (256 bytes — one cache line on CDNA).
pub const HIP_MALLOC_ALIGN: u64 = 256;

/// Round `bytes` up to the next multiple of `align` (a power of two).
///
/// Returns `bytes` unchanged if `align` is 0 or 1.
pub fn align_up(bytes: u64, align: u64) -> u64 {
    if align <= 1 {
        return bytes;
    }
    // `align` is expected to be a power of two; use the masked form when so,
    // else fall back to division to remain correct for any alignment.
    if align.is_power_of_two() {
        (bytes + (align - 1)) & !(align - 1)
    } else {
        bytes.div_ceil(align) * align
    }
}

// ─── Allocation record ──────────────────────────────────────────────────────

/// A live suballocation handed out by the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    /// Synthetic device offset (relative to the arena base).
    offset: u64,
    /// Padded byte size (multiple of [`HIP_MALLOC_ALIGN`]).
    size: u64,
    /// `true` while the block is owned by the caller.
    in_use: bool,
}

// ─── MemPoolStats ───────────────────────────────────────────────────────────

/// A snapshot of pool utilisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemPoolStats {
    /// Bytes currently handed out to callers (padded).
    pub bytes_in_use: u64,
    /// Total bytes ever reserved from the arena (high-water mark).
    pub high_water_mark: u64,
    /// Number of live allocations.
    pub live_allocations: usize,
    /// Bytes available in free blocks that can be reused without growing.
    pub reusable_bytes: u64,
}

// ─── MemoryPool ─────────────────────────────────────────────────────────────

/// A stream-ordered suballocating memory pool over a fixed-size virtual arena.
///
/// Allocations return opaque `u64` handles; the underlying device offset is an
/// implementation detail. Freed blocks are returned to a free list and reused
/// (best-fit) before the arena is grown.
#[derive(Debug)]
pub struct MemoryPool {
    /// Total arena capacity in bytes.
    capacity: u64,
    /// Next never-before-used offset (the "brk" of the arena).
    brk: u64,
    /// All blocks, both live and freed, keyed by handle.
    blocks: Vec<(u64, Block)>,
    /// Next handle to assign (starts at 1; 0 is reserved as "null").
    next_handle: u64,
    /// Peak `brk` reached over the pool's lifetime.
    high_water: u64,
}

impl MemoryPool {
    /// Create a pool backed by a virtual arena of `capacity` bytes.
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            brk: 0,
            blocks: Vec::new(),
            next_handle: 1,
            high_water: 0,
        }
    }

    /// The arena capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Allocate `bytes` of device memory, padded to the HIP alignment.
    ///
    /// Reuses a freed block whenever a best-fit one is available; otherwise
    /// grows the arena. Returns an opaque non-zero handle.
    ///
    /// # Errors
    ///
    /// - [`RocmError::InvalidArgument`] if `bytes` is 0.
    /// - [`RocmError::OutOfMemory`] if neither reuse nor growth can satisfy the
    ///   request within `capacity`.
    pub fn alloc(&mut self, bytes: u64) -> RocmResult<u64> {
        if bytes == 0 {
            return Err(RocmError::InvalidArgument(
                "cannot allocate zero bytes".into(),
            ));
        }
        let need = align_up(bytes, HIP_MALLOC_ALIGN);

        // Best-fit reuse over free blocks.
        let mut best: Option<usize> = None;
        for (i, (_, blk)) in self.blocks.iter().enumerate() {
            if !blk.in_use && blk.size >= need {
                match best {
                    Some(b) if self.blocks[b].1.size <= blk.size => {}
                    _ => best = Some(i),
                }
            }
        }
        if let Some(i) = best {
            let handle = self.next_handle;
            self.next_handle += 1;
            // Re-key the reused block under a fresh handle and mark live.
            let blk = self.blocks[i].1;
            self.blocks[i] = (
                handle,
                Block {
                    offset: blk.offset,
                    size: blk.size,
                    in_use: true,
                },
            );
            return Ok(handle);
        }

        // Grow the arena.
        if self.brk + need > self.capacity {
            return Err(RocmError::OutOfMemory);
        }
        let offset = self.brk;
        self.brk += need;
        self.high_water = self.high_water.max(self.brk);

        let handle = self.next_handle;
        self.next_handle += 1;
        self.blocks.push((
            handle,
            Block {
                offset,
                size: need,
                in_use: true,
            },
        ));
        Ok(handle)
    }

    /// Free the allocation for `handle`, returning its block to the free list.
    ///
    /// Unknown or already-freed handles are a no-op (idempotent free), matching
    /// HIP's tolerance of double-free on pooled memory.
    pub fn free(&mut self, handle: u64) {
        if let Some((_, blk)) = self.blocks.iter_mut().find(|(h, _)| *h == handle) {
            blk.in_use = false;
        }
        self.coalesce();
    }

    /// Merge adjacent free blocks into a single larger free block.
    fn coalesce(&mut self) {
        // Sort by offset to find neighbours.
        self.blocks.sort_by_key(|(_, b)| b.offset);
        let mut i = 0;
        while i + 1 < self.blocks.len() {
            let (_, a) = self.blocks[i];
            let (_, b) = self.blocks[i + 1];
            if !a.in_use && !b.in_use && a.offset + a.size == b.offset {
                // Merge b into a.
                let merged = Block {
                    offset: a.offset,
                    size: a.size + b.size,
                    in_use: false,
                };
                let handle = self.blocks[i].0;
                self.blocks[i] = (handle, merged);
                self.blocks.remove(i + 1);
                // Stay at i to attempt further merges.
            } else {
                i += 1;
            }
        }
    }

    /// The device offset backing `handle` (for descriptor wiring / tests).
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] for unknown or freed handles.
    pub fn offset_of(&self, handle: u64) -> RocmResult<u64> {
        self.blocks
            .iter()
            .find(|(h, b)| *h == handle && b.in_use)
            .map(|(_, b)| b.offset)
            .ok_or_else(|| RocmError::InvalidArgument(format!("invalid handle {handle}")))
    }

    /// Release all *free* blocks past the live high-water region back to the
    /// arena (`hipMemPoolTrimTo`), lowering the brk where possible.
    pub fn trim(&mut self) {
        self.coalesce();
        // Drop trailing free blocks and lower brk accordingly.
        self.blocks.sort_by_key(|(_, b)| b.offset);
        while let Some((_, last)) = self.blocks.last().copied() {
            if !last.in_use && last.offset + last.size == self.brk {
                self.brk = last.offset;
                self.blocks.pop();
            } else {
                break;
            }
        }
    }

    /// Current utilisation snapshot.
    pub fn stats(&self) -> MemPoolStats {
        let mut bytes_in_use = 0u64;
        let mut reusable = 0u64;
        let mut live = 0usize;
        for (_, b) in &self.blocks {
            if b.in_use {
                bytes_in_use += b.size;
                live += 1;
            } else {
                reusable += b.size;
            }
        }
        MemPoolStats {
            bytes_in_use,
            high_water_mark: self.high_water,
            live_allocations: live,
            reusable_bytes: reusable,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_pow2() {
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(100, 1), 100);
        assert_eq!(align_up(100, 0), 100);
    }

    #[test]
    fn align_up_non_pow2() {
        assert_eq!(align_up(10, 3), 12);
        assert_eq!(align_up(9, 3), 9);
    }

    #[test]
    fn alloc_pads_to_alignment() {
        let mut pool = MemoryPool::new(1 << 20);
        let h = pool.alloc(100).expect("alloc");
        let s = pool.stats();
        // 100 padded up to 256.
        assert_eq!(s.bytes_in_use, 256);
        assert_eq!(s.live_allocations, 1);
        assert_eq!(pool.offset_of(h).unwrap(), 0);
    }

    #[test]
    fn zero_alloc_rejected() {
        let mut pool = MemoryPool::new(1024);
        assert!(matches!(pool.alloc(0), Err(RocmError::InvalidArgument(_))));
    }

    #[test]
    fn out_of_memory_when_arena_full() {
        let mut pool = MemoryPool::new(512);
        let _ = pool.alloc(256).expect("first 256");
        let _ = pool.alloc(256).expect("second 256");
        assert!(matches!(pool.alloc(1), Err(RocmError::OutOfMemory)));
    }

    #[test]
    fn free_then_reuse_does_not_grow_arena() {
        let mut pool = MemoryPool::new(1 << 20);
        let a = pool.alloc(1024).expect("a");
        let hw_before = pool.stats().high_water_mark;
        pool.free(a);
        // Same-size request must reuse the freed block, not grow.
        let b = pool.alloc(1024).expect("b");
        assert_eq!(pool.stats().high_water_mark, hw_before);
        assert_eq!(pool.stats().live_allocations, 1);
        assert_ne!(a, b); // fresh handle for the reused block
    }

    #[test]
    fn coalesce_merges_adjacent_frees() {
        let mut pool = MemoryPool::new(1 << 20);
        let a = pool.alloc(1024).expect("a");
        let b = pool.alloc(1024).expect("b");
        pool.free(a);
        pool.free(b);
        // Both 1024-byte blocks should coalesce into a single 2048 free region.
        let big = pool.alloc(2048).expect("coalesced reuse");
        assert_eq!(pool.stats().high_water_mark, 2048);
        assert_eq!(pool.offset_of(big).unwrap(), 0);
    }

    #[test]
    fn high_water_mark_tracks_peak() {
        let mut pool = MemoryPool::new(1 << 20);
        let a = pool.alloc(4096).expect("a");
        let _b = pool.alloc(4096).expect("b");
        assert_eq!(pool.stats().high_water_mark, 8192);
        pool.free(a);
        // Freeing does not lower the high-water mark.
        assert_eq!(pool.stats().high_water_mark, 8192);
        assert_eq!(pool.stats().reusable_bytes, 4096);
    }

    #[test]
    fn trim_lowers_brk_for_trailing_free() {
        let mut pool = MemoryPool::new(1 << 20);
        let _a = pool.alloc(4096).expect("a");
        let b = pool.alloc(4096).expect("b");
        pool.free(b);
        pool.trim();
        // After trimming the trailing free block, a fresh 4096 alloc reuses the
        // same offset b had.
        let c = pool.alloc(4096).expect("c");
        assert_eq!(pool.offset_of(c).unwrap(), 4096);
    }

    #[test]
    fn offset_of_invalid_handle_errors() {
        let mut pool = MemoryPool::new(1024);
        let a = pool.alloc(256).expect("a");
        pool.free(a);
        assert!(pool.offset_of(a).is_err());
        assert!(pool.offset_of(9999).is_err());
    }

    #[test]
    fn double_free_is_noop() {
        let mut pool = MemoryPool::new(1024);
        let a = pool.alloc(256).expect("a");
        pool.free(a);
        pool.free(a); // must not panic
        assert_eq!(pool.stats().live_allocations, 0);
    }

    #[test]
    fn capacity_reported() {
        let pool = MemoryPool::new(2048);
        assert_eq!(pool.capacity(), 2048);
    }
}
