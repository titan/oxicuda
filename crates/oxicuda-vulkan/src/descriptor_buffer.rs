//! `VK_EXT_descriptor_buffer` bindless descriptor layout math.
//!
//! `VK_EXT_descriptor_buffer` (Vulkan 2023) replaces opaque
//! `VkDescriptorPool`/`VkDescriptorSet` objects with plain device memory: a
//! *descriptor buffer* into which the application writes raw descriptor bytes at
//! computed offsets. This removes descriptor-set allocation from the hot path
//! and is ideal for binding the many weight tensors of a large model.
//!
//! Using it on a device requires the per-descriptor *sizes* and *alignment* from
//! `VkPhysicalDeviceDescriptorBufferPropertiesEXT`, then
//! `vkGetDescriptorSetLayoutSizeEXT` / `vkGetDescriptorSetLayoutBindingOffsetEXT`
//! to place descriptors — but the underlying placement is deterministic
//! arithmetic. This module reproduces that arithmetic as a host-side, fully
//! CPU-testable [`DescriptorBuffer`] layout planner.

use crate::error::{VulkanError, VulkanResult};

/// The descriptor types relevant to a compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    /// A storage buffer (SSBO) — the only type the SPIR-V generators bind.
    StorageBuffer,
    /// A uniform buffer (UBO).
    UniformBuffer,
    /// A storage image.
    StorageImage,
    /// A combined image sampler.
    CombinedImageSampler,
}

/// Per-device descriptor sizes/alignment from
/// `VkPhysicalDeviceDescriptorBufferPropertiesEXT`.
///
/// The defaults are representative mid-range values; on a real device they are
/// queried and supplied to [`DescriptorBuffer::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBufferProps {
    /// Size in bytes of a storage-buffer descriptor.
    pub storage_buffer_size: u64,
    /// Size in bytes of a uniform-buffer descriptor.
    pub uniform_buffer_size: u64,
    /// Size in bytes of a storage-image descriptor.
    pub storage_image_size: u64,
    /// Size in bytes of a combined-image-sampler descriptor.
    pub combined_image_sampler_size: u64,
    /// Required alignment of a descriptor-set within the buffer.
    pub descriptor_buffer_offset_alignment: u64,
}

impl Default for DescriptorBufferProps {
    fn default() -> Self {
        Self {
            storage_buffer_size: 16,
            uniform_buffer_size: 16,
            storage_image_size: 32,
            combined_image_sampler_size: 96,
            descriptor_buffer_offset_alignment: 64,
        }
    }
}

impl DescriptorBufferProps {
    /// Descriptor byte-size for `kind`.
    #[must_use]
    pub fn size_of(self, kind: DescriptorKind) -> u64 {
        match kind {
            DescriptorKind::StorageBuffer => self.storage_buffer_size,
            DescriptorKind::UniformBuffer => self.uniform_buffer_size,
            DescriptorKind::StorageImage => self.storage_image_size,
            DescriptorKind::CombinedImageSampler => self.combined_image_sampler_size,
        }
    }
}

/// One binding within a descriptor-set layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutBinding {
    /// `binding` number referenced by the shader.
    pub binding: u32,
    /// Descriptor type.
    pub kind: DescriptorKind,
    /// Array size (`1` for a plain binding; larger for descriptor arrays /
    /// bindless tables).
    pub count: u32,
}

impl LayoutBinding {
    /// Convenience constructor for a single storage buffer.
    #[must_use]
    pub fn storage_buffer(binding: u32) -> Self {
        Self {
            binding,
            kind: DescriptorKind::StorageBuffer,
            count: 1,
        }
    }
}

/// The computed byte offset of a binding within the descriptor buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingOffset {
    /// The binding number.
    pub binding: u32,
    /// Byte offset from the start of the descriptor set.
    pub offset: u64,
    /// Total byte size of this binding's descriptors (`size_of(kind) * count`).
    pub size: u64,
}

/// Host-side descriptor-buffer layout planner.
///
/// Computes the total descriptor-buffer size for a set layout and the offset of
/// every binding, exactly as `vkGetDescriptorSetLayoutSizeEXT` /
/// `vkGetDescriptorSetLayoutBindingOffsetEXT` would, without a device.
#[derive(Debug, Clone)]
pub struct DescriptorBuffer {
    props: DescriptorBufferProps,
    offsets: Vec<BindingOffset>,
    set_size: u64,
}

impl DescriptorBuffer {
    /// Plan the layout for `bindings` under device `props`.
    ///
    /// Bindings are laid out in ascending `binding` order; each descriptor's
    /// offset is the running total (descriptors of the same type pack tightly,
    /// as the Vulkan spec lays them out contiguously). The set size is rounded
    /// up to `descriptor_buffer_offset_alignment` so consecutive sets in one
    /// buffer stay aligned.
    ///
    /// Returns an error on duplicate binding numbers or a zero `count`.
    pub fn new(
        props: DescriptorBufferProps,
        mut bindings: Vec<LayoutBinding>,
    ) -> VulkanResult<Self> {
        if props.descriptor_buffer_offset_alignment == 0
            || !props.descriptor_buffer_offset_alignment.is_power_of_two()
        {
            return Err(VulkanError::InvalidArgument(
                "descriptor_buffer_offset_alignment must be a power of two".into(),
            ));
        }
        bindings.sort_by_key(|b| b.binding);
        // Reject duplicate binding numbers.
        for pair in bindings.windows(2) {
            if pair[0].binding == pair[1].binding {
                return Err(VulkanError::InvalidArgument(format!(
                    "duplicate binding number {}",
                    pair[0].binding
                )));
            }
        }

        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(bindings.len());
        for b in &bindings {
            if b.count == 0 {
                return Err(VulkanError::InvalidArgument(format!(
                    "binding {} has count 0",
                    b.binding
                )));
            }
            let size = props.size_of(b.kind) * u64::from(b.count);
            offsets.push(BindingOffset {
                binding: b.binding,
                offset,
                size,
            });
            offset += size;
        }

        let set_size = align_up(offset, props.descriptor_buffer_offset_alignment);
        Ok(Self {
            props,
            offsets,
            set_size,
        })
    }

    /// The padded size in bytes of one descriptor set within the buffer.
    #[must_use]
    pub fn set_size(&self) -> u64 {
        self.set_size
    }

    /// The unpadded size in bytes (sum of all binding sizes).
    #[must_use]
    pub fn used_size(&self) -> u64 {
        self.offsets.iter().map(|o| o.size).sum()
    }

    /// Per-binding offsets, in ascending binding order.
    #[must_use]
    pub fn binding_offsets(&self) -> &[BindingOffset] {
        &self.offsets
    }

    /// The byte offset of `binding`, or `None` if not present.
    #[must_use]
    pub fn offset_of(&self, binding: u32) -> Option<u64> {
        self.offsets
            .iter()
            .find(|o| o.binding == binding)
            .map(|o| o.offset)
    }

    /// Device properties this layout was planned against.
    #[must_use]
    pub fn props(&self) -> DescriptorBufferProps {
        self.props
    }

    /// Total descriptor-buffer size needed to hold `n` copies of this set
    /// (e.g. `n` frames-in-flight or `n` model layers), keeping every copy
    /// aligned.
    #[must_use]
    pub fn buffer_size_for(&self, n: u32) -> u64 {
        self.set_size * u64::from(n)
    }
}

/// Round `value` up to the nearest multiple of `alignment` (power of two).
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_storage_buffers_pack_tightly() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(
            props,
            vec![
                LayoutBinding::storage_buffer(0),
                LayoutBinding::storage_buffer(1),
                LayoutBinding::storage_buffer(2),
            ],
        )
        .unwrap();
        // 3 * 16 = 48 used; offsets at 0, 16, 32.
        assert_eq!(db.used_size(), 48);
        assert_eq!(db.offset_of(0), Some(0));
        assert_eq!(db.offset_of(1), Some(16));
        assert_eq!(db.offset_of(2), Some(32));
        // Padded up to 64-byte alignment.
        assert_eq!(db.set_size(), 64);
    }

    #[test]
    fn mixed_types_use_per_type_sizes() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(
            props,
            vec![
                LayoutBinding {
                    binding: 0,
                    kind: DescriptorKind::UniformBuffer,
                    count: 1,
                },
                LayoutBinding {
                    binding: 1,
                    kind: DescriptorKind::CombinedImageSampler,
                    count: 1,
                },
            ],
        )
        .unwrap();
        assert_eq!(db.offset_of(0), Some(0));
        // CIS follows the 16-byte UBO at offset 16.
        assert_eq!(db.offset_of(1), Some(16));
        assert_eq!(db.used_size(), 16 + 96);
    }

    #[test]
    fn descriptor_array_multiplies_size() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(
            props,
            vec![LayoutBinding {
                binding: 0,
                kind: DescriptorKind::StorageBuffer,
                count: 8,
            }],
        )
        .unwrap();
        // Bindless array of 8 SSBOs: 8 * 16 = 128.
        assert_eq!(db.used_size(), 128);
        assert_eq!(db.binding_offsets()[0].size, 128);
    }

    #[test]
    fn bindings_sorted_regardless_of_input_order() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(
            props,
            vec![
                LayoutBinding::storage_buffer(2),
                LayoutBinding::storage_buffer(0),
                LayoutBinding::storage_buffer(1),
            ],
        )
        .unwrap();
        let offs = db.binding_offsets();
        assert_eq!(offs[0].binding, 0);
        assert_eq!(offs[1].binding, 1);
        assert_eq!(offs[2].binding, 2);
    }

    #[test]
    fn buffer_size_for_multiple_sets() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(props, vec![LayoutBinding::storage_buffer(0)]).unwrap();
        // One SSBO → 16 used, padded to 64.
        assert_eq!(db.set_size(), 64);
        assert_eq!(db.buffer_size_for(4), 256);
    }

    #[test]
    fn duplicate_binding_rejected() {
        let props = DescriptorBufferProps::default();
        assert!(
            DescriptorBuffer::new(
                props,
                vec![
                    LayoutBinding::storage_buffer(0),
                    LayoutBinding::storage_buffer(0),
                ],
            )
            .is_err()
        );
    }

    #[test]
    fn zero_count_rejected() {
        let props = DescriptorBufferProps::default();
        assert!(
            DescriptorBuffer::new(
                props,
                vec![LayoutBinding {
                    binding: 0,
                    kind: DescriptorKind::StorageBuffer,
                    count: 0,
                }],
            )
            .is_err()
        );
    }

    #[test]
    fn non_pow2_alignment_rejected() {
        let props = DescriptorBufferProps {
            descriptor_buffer_offset_alignment: 48,
            ..DescriptorBufferProps::default()
        };
        assert!(DescriptorBuffer::new(props, vec![LayoutBinding::storage_buffer(0)]).is_err());
    }

    #[test]
    fn unknown_binding_offset_is_none() {
        let props = DescriptorBufferProps::default();
        let db = DescriptorBuffer::new(props, vec![LayoutBinding::storage_buffer(0)]).unwrap();
        assert_eq!(db.offset_of(7), None);
    }
}
