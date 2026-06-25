//! Pure-Rust device-memory sub-allocator (VMA-style).
//!
//! Calling `vkAllocateMemory` per buffer is slow and bounded by
//! `maxMemoryAllocationCount` (often ~4096). Production Vulkan code allocates a
//! few large `VkDeviceMemory` *blocks* and **sub-allocates** buffers out of
//! them, mirroring the [Vulkan Memory Allocator] design.
//!
//! This module implements the sub-allocation *bookkeeping* — the part that is
//! entirely host-side arithmetic and therefore CPU-testable without a device.
//! Two strategies are provided:
//!
//! - [`FreeListSuballocator`] — a first-fit free-list with boundary-merge on
//!   free; honours arbitrary power-of-two alignment. General purpose.
//! - [`BuddySuballocator`] — a binary-buddy allocator over a power-of-two
//!   block; O(log n) alloc/free with bounded external fragmentation. Ideal for
//!   transient tensors.
//!
//! Both return a byte *offset* into the backing block. The caller binds a
//! `VkBuffer` to `(block_memory, offset)` via `vkBindBufferMemory` — that bind
//! is the only device-gated step and lives in [`crate::memory`].
//!
//! [Vulkan Memory Allocator]: https://gpuopen.com/vulkan-memory-allocator/

use crate::error::{VulkanError, VulkanResult};

// ─── Free-list sub-allocator ─────────────────────────────────

/// A sub-allocation handed out by [`FreeListSuballocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubAllocation {
    /// Byte offset of the allocation within the backing block.
    pub offset: u64,
    /// Usable size of the allocation in bytes (the requested size, not padded).
    pub size: u64,
}

/// One contiguous free span within the backing block.
#[derive(Debug, Clone, Copy)]
struct FreeSpan {
    offset: u64,
    size: u64,
}

/// First-fit free-list sub-allocator over a single fixed-size block.
///
/// Tracks free spans, splits on allocation, and coalesces adjacent spans on
/// free. Alignment is honoured by rounding the chosen offset up and accounting
/// for the introduced padding.
#[derive(Debug)]
pub struct FreeListSuballocator {
    block_size: u64,
    free: Vec<FreeSpan>,
    /// Map from a returned offset to the *actual* span `(offset, size)` that was
    /// carved (including alignment padding), so `free` restores it exactly.
    live: Vec<(u64, FreeSpan)>,
}

impl FreeListSuballocator {
    /// Create an allocator managing `block_size` bytes (must be non-zero).
    pub fn new(block_size: u64) -> VulkanResult<Self> {
        if block_size == 0 {
            return Err(VulkanError::InvalidArgument(
                "suballocator block_size must be > 0".into(),
            ));
        }
        Ok(Self {
            block_size,
            free: vec![FreeSpan {
                offset: 0,
                size: block_size,
            }],
            live: Vec::new(),
        })
    }

    /// Total managed size in bytes.
    #[must_use]
    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    /// Sum of all currently-free bytes (may be fragmented across spans).
    #[must_use]
    pub fn free_bytes(&self) -> u64 {
        self.free.iter().map(|s| s.size).sum()
    }

    /// The largest single contiguous free span.
    #[must_use]
    pub fn largest_free_span(&self) -> u64 {
        self.free.iter().map(|s| s.size).max().unwrap_or(0)
    }

    /// Allocate `size` bytes aligned to `alignment` (must be a power of two).
    ///
    /// Returns the offset/size, or [`VulkanError::OutOfMemory`] if no span fits.
    pub fn alloc(&mut self, size: u64, alignment: u64) -> VulkanResult<SubAllocation> {
        if size == 0 {
            return Err(VulkanError::InvalidArgument(
                "suballocation size must be > 0".into(),
            ));
        }
        if !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidArgument(format!(
                "alignment {alignment} must be a power of two"
            )));
        }

        // First-fit: find a span whose aligned start + size fits.
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
            return Err(VulkanError::OutOfMemory);
        };

        let span = self.free[idx];
        let pad = aligned_off - span.offset;
        // The carved region (including leading padding) we must restore on free.
        let carved = FreeSpan {
            offset: span.offset,
            size: pad + size,
        };

        // Replace the span with whatever remains after the carved region.
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
        Ok(SubAllocation {
            offset: aligned_off,
            size,
        })
    }

    /// Free a previously-returned allocation by its `offset`.
    ///
    /// Returns [`VulkanError::InvalidArgument`] if the offset was not live
    /// (double free or bogus offset).
    pub fn free(&mut self, offset: u64) -> VulkanResult<()> {
        let pos = self
            .live
            .iter()
            .position(|(o, _)| *o == offset)
            .ok_or_else(|| {
                VulkanError::InvalidArgument(format!("free of unknown offset {offset}"))
            })?;
        let (_, carved) = self.live.remove(pos);
        self.insert_and_coalesce(carved);
        Ok(())
    }

    /// Insert a freed span and merge it with any adjacent free spans.
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

    /// Number of live allocations.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

/// Round `value` up to the nearest multiple of `alignment` (power of two).
#[must_use]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

// ─── Buddy sub-allocator ─────────────────────────────────────

/// Binary-buddy sub-allocator over a power-of-two block.
///
/// Allocation sizes are rounded up to a power of two between `min_block` and
/// the total block size. Free recombines buddies upward. This bounds external
/// fragmentation and gives O(log n) operations.
#[derive(Debug)]
pub struct BuddySuballocator {
    /// `total_size == min_block << max_order`.
    min_block: u64,
    max_order: u32,
    /// `free_lists[k]` holds offsets of free blocks of size `min_block << k`.
    free_lists: Vec<Vec<u64>>,
    /// Live allocations: `offset -> order`.
    live: Vec<(u64, u32)>,
}

impl BuddySuballocator {
    /// Create a buddy allocator over `total_size` bytes with `min_block` leaf
    /// size. Both must be powers of two and `min_block <= total_size`.
    pub fn new(total_size: u64, min_block: u64) -> VulkanResult<Self> {
        if !total_size.is_power_of_two() || !min_block.is_power_of_two() {
            return Err(VulkanError::InvalidArgument(
                "buddy allocator sizes must be powers of two".into(),
            ));
        }
        if min_block == 0 || min_block > total_size {
            return Err(VulkanError::InvalidArgument(
                "require 0 < min_block <= total_size".into(),
            ));
        }
        let max_order = (total_size / min_block).trailing_zeros();
        let mut free_lists = vec![Vec::new(); (max_order + 1) as usize];
        // The whole block starts free at the top order.
        free_lists[max_order as usize].push(0);
        Ok(Self {
            min_block,
            max_order,
            free_lists,
            live: Vec::new(),
        })
    }

    /// Total managed size in bytes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.min_block << self.max_order
    }

    /// Smallest order (block size = `min_block << order`) that holds `size`.
    fn order_for(&self, size: u64) -> VulkanResult<u32> {
        if size == 0 {
            return Err(VulkanError::InvalidArgument("size must be > 0".into()));
        }
        let mut order = 0u32;
        while (self.min_block << order) < size {
            order += 1;
            if order > self.max_order {
                return Err(VulkanError::OutOfMemory);
            }
        }
        Ok(order)
    }

    /// Allocate `size` bytes, returning the offset of a power-of-two block.
    pub fn alloc(&mut self, size: u64) -> VulkanResult<SubAllocation> {
        let need = self.order_for(size)?;

        // Find the smallest available order >= need.
        let mut order = need;
        while order <= self.max_order && self.free_lists[order as usize].is_empty() {
            order += 1;
        }
        if order > self.max_order {
            return Err(VulkanError::OutOfMemory);
        }

        // Pop a block and split it down to `need`, pushing the buddies back.
        // The scan above guarantees this list is non-empty.
        let offset = match self.free_lists[order as usize].pop() {
            Some(o) => o,
            None => return Err(VulkanError::OutOfMemory),
        };
        while order > need {
            order -= 1;
            let buddy = offset + (self.min_block << order);
            self.free_lists[order as usize].push(buddy);
        }

        self.live.push((offset, need));
        Ok(SubAllocation {
            offset,
            size: self.min_block << need,
        })
    }

    /// Free a previously-allocated block by its `offset`.
    pub fn free(&mut self, offset: u64) -> VulkanResult<()> {
        let pos = self
            .live
            .iter()
            .position(|(o, _)| *o == offset)
            .ok_or_else(|| {
                VulkanError::InvalidArgument(format!("buddy free of unknown offset {offset}"))
            })?;
        let (mut off, mut order) = self.live.remove(pos);

        // Recombine with the buddy while it is free at the same order.
        while order < self.max_order {
            let block = self.min_block << order;
            let buddy = off ^ block; // XOR flips the buddy bit.
            if let Some(idx) = self.free_lists[order as usize]
                .iter()
                .position(|&o| o == buddy)
            {
                self.free_lists[order as usize].swap_remove(idx);
                off = off.min(buddy);
                order += 1;
            } else {
                break;
            }
        }
        self.free_lists[order as usize].push(off);
        Ok(())
    }

    /// Total free bytes across all orders.
    #[must_use]
    pub fn free_bytes(&self) -> u64 {
        let mut total = 0u64;
        for (k, list) in self.free_lists.iter().enumerate() {
            total += (self.min_block << k as u32) * list.len() as u64;
        }
        total
    }

    /// Number of live allocations.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(100, 64), 128);
    }

    #[test]
    fn freelist_basic_alloc_free() {
        let mut a = FreeListSuballocator::new(1024).unwrap();
        assert_eq!(a.free_bytes(), 1024);
        let x = a.alloc(256, 256).unwrap();
        assert_eq!(x.offset, 0);
        assert_eq!(a.free_bytes(), 768);
        let y = a.alloc(128, 64).unwrap();
        assert_eq!(y.offset, 256);
        a.free(x.offset).unwrap();
        a.free(y.offset).unwrap();
        // Everything coalesced back to one span.
        assert_eq!(a.free_bytes(), 1024);
        assert_eq!(a.largest_free_span(), 1024);
        assert_eq!(a.live_count(), 0);
    }

    #[test]
    fn freelist_alignment_padding_accounted() {
        let mut a = FreeListSuballocator::new(1024).unwrap();
        // Carve 10 bytes first so the next aligned alloc must pad.
        let head = a.alloc(10, 1).unwrap();
        assert_eq!(head.offset, 0);
        let aligned = a.alloc(16, 256).unwrap();
        assert_eq!(aligned.offset, 256, "must round up to 256");
        // Free both → fully coalesced.
        a.free(head.offset).unwrap();
        a.free(aligned.offset).unwrap();
        assert_eq!(a.free_bytes(), 1024);
    }

    #[test]
    fn freelist_out_of_memory() {
        let mut a = FreeListSuballocator::new(128).unwrap();
        assert!(a.alloc(256, 1).is_err());
        let _ = a.alloc(128, 1).unwrap();
        assert!(matches!(a.alloc(1, 1), Err(VulkanError::OutOfMemory)));
    }

    #[test]
    fn freelist_rejects_bad_args() {
        assert!(FreeListSuballocator::new(0).is_err());
        let mut a = FreeListSuballocator::new(64).unwrap();
        assert!(a.alloc(0, 1).is_err());
        assert!(a.alloc(8, 3).is_err(), "non-pow2 alignment rejected");
        assert!(a.free(999).is_err(), "unknown offset rejected");
    }

    #[test]
    fn freelist_coalesce_middle_hole() {
        let mut a = FreeListSuballocator::new(300).unwrap();
        let x = a.alloc(100, 1).unwrap();
        let y = a.alloc(100, 1).unwrap();
        let z = a.alloc(100, 1).unwrap();
        assert_eq!(a.free_bytes(), 0);
        // Free outer two — should NOT merge across the live middle.
        a.free(x.offset).unwrap();
        a.free(z.offset).unwrap();
        assert_eq!(a.free_bytes(), 200);
        assert_eq!(a.largest_free_span(), 100, "two disjoint 100-byte holes");
        // Free the middle — now all three coalesce.
        a.free(y.offset).unwrap();
        assert_eq!(a.largest_free_span(), 300);
    }

    #[test]
    fn buddy_basic() {
        let mut b = BuddySuballocator::new(1024, 64).unwrap();
        assert_eq!(b.total_size(), 1024);
        assert_eq!(b.free_bytes(), 1024);
        // 100 bytes rounds up to a 128-byte block.
        let x = b.alloc(100).unwrap();
        assert_eq!(x.size, 128);
        assert_eq!(b.free_bytes(), 1024 - 128);
        b.free(x.offset).unwrap();
        assert_eq!(b.free_bytes(), 1024);
        assert_eq!(b.live_count(), 0);
    }

    #[test]
    fn buddy_splits_and_recombines() {
        let mut b = BuddySuballocator::new(1024, 64).unwrap();
        let a0 = b.alloc(64).unwrap();
        let a1 = b.alloc(64).unwrap();
        // Two adjacent leaves carved from the same 128 block.
        assert_ne!(a0.offset, a1.offset);
        assert_eq!(a0.size, 64);
        b.free(a0.offset).unwrap();
        b.free(a1.offset).unwrap();
        // Full recombination back to a single 1024 block.
        assert_eq!(b.free_bytes(), 1024);
        // A full-size allocation must now succeed.
        let full = b.alloc(1024).unwrap();
        assert_eq!(full.offset, 0);
        assert_eq!(full.size, 1024);
    }

    #[test]
    fn buddy_exhaustion() {
        let mut b = BuddySuballocator::new(256, 64).unwrap();
        let _a = b.alloc(64).unwrap();
        let _c = b.alloc(64).unwrap();
        let _d = b.alloc(64).unwrap();
        let _e = b.alloc(64).unwrap();
        assert_eq!(b.free_bytes(), 0);
        assert!(matches!(b.alloc(64), Err(VulkanError::OutOfMemory)));
    }

    #[test]
    fn buddy_rejects_bad_args() {
        assert!(BuddySuballocator::new(1000, 64).is_err(), "non-pow2 total");
        assert!(BuddySuballocator::new(1024, 100).is_err(), "non-pow2 leaf");
        assert!(BuddySuballocator::new(64, 128).is_err(), "leaf > total");
        let mut b = BuddySuballocator::new(1024, 64).unwrap();
        assert!(b.alloc(0).is_err());
        assert!(b.alloc(2048).is_err(), "larger than block");
        assert!(b.free(123).is_err(), "unknown offset");
    }

    #[test]
    fn buddy_offsets_are_block_aligned() {
        let mut b = BuddySuballocator::new(4096, 256).unwrap();
        let x = b.alloc(256).unwrap();
        let y = b.alloc(512).unwrap();
        assert_eq!(x.offset % 256, 0);
        assert_eq!(y.offset % 512, 0, "512-block must be 512-aligned");
    }
}
