//! `VK_KHR_push_descriptor` write recording + push-constant layout helpers.
//!
//! `VK_KHR_push_descriptor` lets a dispatch bind its buffers *directly into the
//! command buffer* via `vkCmdPushDescriptorSetKHR`, with no
//! `VkDescriptorPool`/`VkDescriptorSet` allocation — the lowest-overhead path
//! for short-lived compute dispatches. The host assembles a list of descriptor
//! writes (`VkWriteDescriptorSet`); recording them into the command buffer is
//! the only device-gated step.
//!
//! This module models that write list as a CPU-testable data structure
//! ([`PushDescriptorSet`]) plus a [`PushConstantLayout`] helper that validates
//! and lays out push-constant ranges against the common 128-byte guaranteed
//! limit.

use crate::error::{VulkanError, VulkanResult};

/// One push-descriptor write: bind a buffer region to a `binding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushDescriptorWrite {
    /// Shader `binding` number (descriptor set 0).
    pub binding: u32,
    /// Opaque buffer handle (the backend's `u64` memory handle).
    pub buffer: u64,
    /// Byte offset into the buffer.
    pub offset: u64,
    /// Byte range bound (`WHOLE_SIZE` is modelled as `u64::MAX`).
    pub range: u64,
}

/// Sentinel for `VK_WHOLE_SIZE`.
pub const WHOLE_SIZE: u64 = u64::MAX;

/// Host-side accumulator of push-descriptor writes for one dispatch.
///
/// Bindings must be unique; the assembled slice is fed to
/// `vkCmdPushDescriptorSetKHR` at record time.
#[derive(Debug, Default, Clone)]
pub struct PushDescriptorSet {
    writes: Vec<PushDescriptorWrite>,
}

impl PushDescriptorSet {
    /// Create an empty push-descriptor set.
    #[must_use]
    pub fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Bind `buffer[offset .. offset+range]` to `binding`.
    ///
    /// Returns an error if `binding` was already written (a command buffer
    /// cannot push two descriptors to the same binding in one call).
    pub fn bind_buffer(
        &mut self,
        binding: u32,
        buffer: u64,
        offset: u64,
        range: u64,
    ) -> VulkanResult<&mut Self> {
        if self.writes.iter().any(|w| w.binding == binding) {
            return Err(VulkanError::InvalidArgument(format!(
                "binding {binding} already pushed"
            )));
        }
        if range == 0 {
            return Err(VulkanError::InvalidArgument(
                "descriptor range must be > 0 (use WHOLE_SIZE for the rest)".into(),
            ));
        }
        self.writes.push(PushDescriptorWrite {
            binding,
            buffer,
            offset,
            range,
        });
        Ok(self)
    }

    /// Bind the whole of `buffer` to `binding`.
    pub fn bind_whole_buffer(&mut self, binding: u32, buffer: u64) -> VulkanResult<&mut Self> {
        self.bind_buffer(binding, buffer, 0, WHOLE_SIZE)
    }

    /// The accumulated writes, sorted by binding (the order a command buffer
    /// records them is irrelevant, but a stable order eases testing/hashing).
    #[must_use]
    pub fn writes(&self) -> Vec<PushDescriptorWrite> {
        let mut w = self.writes.clone();
        w.sort_by_key(|x| x.binding);
        w
    }

    /// Number of writes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.writes.len()
    }

    /// Whether no writes have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

// ─── Push-constant layout ────────────────────────────────────

/// One push-constant range within the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushConstantRange {
    /// Byte offset of the range.
    pub offset: u32,
    /// Byte size of the range.
    pub size: u32,
}

/// Push-constant layout validator/builder.
///
/// Vulkan guarantees at least 128 bytes of push-constant space
/// (`maxPushConstantsSize`). This helper packs typed fields into ranges,
/// enforcing the 4-byte alignment Vulkan requires for offsets/sizes and the
/// chosen size limit.
#[derive(Debug, Clone)]
pub struct PushConstantLayout {
    limit: u32,
    cursor: u32,
    ranges: Vec<PushConstantRange>,
}

impl PushConstantLayout {
    /// The Vulkan-guaranteed minimum push-constant size.
    pub const GUARANTEED_LIMIT: u32 = 128;

    /// Create a layout bounded by `limit` bytes (use [`Self::GUARANTEED_LIMIT`]
    /// for the portable minimum; pass the device's `maxPushConstantsSize`
    /// otherwise).
    pub fn new(limit: u32) -> VulkanResult<Self> {
        if limit == 0 || limit % 4 != 0 {
            return Err(VulkanError::InvalidArgument(
                "push-constant limit must be a positive multiple of 4".into(),
            ));
        }
        Ok(Self {
            limit,
            cursor: 0,
            ranges: Vec::new(),
        })
    }

    /// Append a field of `size` bytes, returning its byte offset.
    ///
    /// `size` is rounded up to a multiple of 4 (Vulkan requires 4-byte aligned
    /// offsets and sizes). Returns [`VulkanError::OutOfMemory`] if the field
    /// would exceed the limit.
    pub fn push_field(&mut self, size: u32) -> VulkanResult<u32> {
        if size == 0 {
            return Err(VulkanError::InvalidArgument(
                "push-constant field size must be > 0".into(),
            ));
        }
        let padded = (size + 3) & !3;
        if self.cursor + padded > self.limit {
            return Err(VulkanError::OutOfMemory);
        }
        let offset = self.cursor;
        self.ranges.push(PushConstantRange {
            offset,
            size: padded,
        });
        self.cursor += padded;
        Ok(offset)
    }

    /// Append `count` `u32`/`f32` scalars (4 bytes each).
    pub fn push_u32_array(&mut self, count: u32) -> VulkanResult<u32> {
        self.push_field(count.checked_mul(4).ok_or_else(|| {
            VulkanError::InvalidArgument("push-constant array size overflow".into())
        })?)
    }

    /// The total bytes consumed so far (always a multiple of 4).
    #[must_use]
    pub fn total_size(&self) -> u32 {
        self.cursor
    }

    /// The byte limit this layout enforces.
    #[must_use]
    pub fn limit(&self) -> u32 {
        self.limit
    }

    /// The recorded ranges.
    #[must_use]
    pub fn ranges(&self) -> &[PushConstantRange] {
        &self.ranges
    }

    /// The single coalesced range `(offset 0 .. total_size)` to feed a
    /// `VkPushConstantRange` for the compute stage.
    #[must_use]
    pub fn coalesced_range(&self) -> Option<PushConstantRange> {
        if self.cursor == 0 {
            None
        } else {
            Some(PushConstantRange {
                offset: 0,
                size: self.cursor,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_descriptor_basic() {
        let mut set = PushDescriptorSet::new();
        assert!(set.is_empty());
        set.bind_whole_buffer(0, 0xAAAA).unwrap();
        set.bind_buffer(1, 0xBBBB, 256, 1024).unwrap();
        assert_eq!(set.len(), 2);
        let writes = set.writes();
        assert_eq!(writes[0].binding, 0);
        assert_eq!(writes[0].range, WHOLE_SIZE);
        assert_eq!(writes[1].binding, 1);
        assert_eq!(writes[1].offset, 256);
        assert_eq!(writes[1].range, 1024);
    }

    #[test]
    fn push_descriptor_duplicate_binding_rejected() {
        let mut set = PushDescriptorSet::new();
        set.bind_whole_buffer(0, 1).unwrap();
        assert!(set.bind_whole_buffer(0, 2).is_err());
    }

    #[test]
    fn push_descriptor_zero_range_rejected() {
        let mut set = PushDescriptorSet::new();
        assert!(set.bind_buffer(0, 1, 0, 0).is_err());
    }

    #[test]
    fn push_descriptor_writes_sorted() {
        let mut set = PushDescriptorSet::new();
        set.bind_whole_buffer(2, 1).unwrap();
        set.bind_whole_buffer(0, 1).unwrap();
        set.bind_whole_buffer(1, 1).unwrap();
        let w = set.writes();
        assert_eq!(
            w.iter().map(|x| x.binding).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn push_constant_layout_packs_and_aligns() {
        let mut l = PushConstantLayout::new(PushConstantLayout::GUARANTEED_LIMIT).unwrap();
        // A 3-byte field rounds up to 4.
        let a = l.push_field(3).unwrap();
        assert_eq!(a, 0);
        let b = l.push_u32_array(4).unwrap();
        assert_eq!(b, 4, "previous field padded to 4 bytes");
        assert_eq!(l.total_size(), 4 + 16);
        assert_eq!(l.ranges().len(), 2);
        let coalesced = l.coalesced_range().unwrap();
        assert_eq!(coalesced.offset, 0);
        assert_eq!(coalesced.size, 20);
    }

    #[test]
    fn push_constant_layout_enforces_limit() {
        let mut l = PushConstantLayout::new(16).unwrap();
        l.push_u32_array(4).unwrap(); // exactly 16
        assert!(matches!(l.push_field(4), Err(VulkanError::OutOfMemory)));
    }

    #[test]
    fn push_constant_layout_rejects_bad_limit() {
        assert!(PushConstantLayout::new(0).is_err());
        assert!(PushConstantLayout::new(13).is_err(), "not a multiple of 4");
    }

    #[test]
    fn push_constant_empty_has_no_range() {
        let l = PushConstantLayout::new(128).unwrap();
        assert!(l.coalesced_range().is_none());
        assert_eq!(l.total_size(), 0);
        assert_eq!(l.limit(), 128);
    }

    #[test]
    fn push_constant_array_overflow_rejected() {
        let mut l = PushConstantLayout::new(128).unwrap();
        assert!(l.push_u32_array(u32::MAX).is_err());
    }
}
