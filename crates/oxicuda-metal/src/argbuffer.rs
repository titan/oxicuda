//! Argument-buffer (`MTLArgumentEncoder`) layout planning.
//!
//! Argument buffers let a shader access many resources through a single bound
//! buffer of GPU addresses, instead of binding each resource individually on
//! every dispatch — the bindless model.  The Rust-side work is to lay out the
//! argument *table*: assign each argument an `[[id(n)]]` slot, compute its byte
//! offset within the encoded buffer, and report the total encoded size so a
//! backing buffer can be allocated.
//!
//! This module models that layout as a pure data structure that mirrors what
//! `MTLArgumentEncoder` produces (`encodedLength`, `setBuffer:offset:atIndex:`),
//! and is fully unit-testable without a device.

use crate::error::{MetalError, MetalResult};
use crate::storage::align_up;
use std::fmt;

/// Encoded size of a single GPU resource handle (64-bit address) in an
/// argument buffer.
pub const ARGUMENT_HANDLE_SIZE: usize = 8;

/// Alignment of each argument slot within an argument buffer.
pub const ARGUMENT_SLOT_ALIGNMENT: usize = 8;

// ─── ArgumentKind ──────────────────────────────────────────────────────────────

/// The kind of resource bound at an argument slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgumentKind {
    /// A `device`/`constant` buffer pointer.
    Buffer,
    /// A texture handle.
    Texture,
    /// A sampler state.
    Sampler,
    /// An inline constant of the given byte size (POD struct embedded in the
    /// argument buffer rather than referenced).
    InlineConstant(usize),
}

impl ArgumentKind {
    /// Encoded byte size of this argument within the argument buffer.
    pub fn encoded_size(self) -> usize {
        match self {
            // Buffers, textures and samplers are encoded as 64-bit handles.
            Self::Buffer | Self::Texture | Self::Sampler => ARGUMENT_HANDLE_SIZE,
            Self::InlineConstant(bytes) => bytes,
        }
    }

    /// Whether this argument references a separate `MTLResource` that must be
    /// made resident via `useResource:` (buffers/textures), vs. inline data.
    pub fn references_resource(self) -> bool {
        matches!(self, Self::Buffer | Self::Texture | Self::Sampler)
    }
}

impl fmt::Display for ArgumentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffer => f.write_str("Buffer"),
            Self::Texture => f.write_str("Texture"),
            Self::Sampler => f.write_str("Sampler"),
            Self::InlineConstant(n) => write!(f, "InlineConstant({n})"),
        }
    }
}

// ─── ArgumentSlot ──────────────────────────────────────────────────────────────

/// One laid-out argument: its `[[id(n)]]`, kind, and byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgumentSlot {
    /// The MSL `[[id(n)]]` index.
    pub id: u32,
    /// What is bound here.
    pub kind: ArgumentKind,
    /// Byte offset of this slot within the encoded argument buffer.
    pub offset: usize,
}

// ─── ArgumentBufferLayout ──────────────────────────────────────────────────────

/// A complete argument-buffer layout built from an ordered list of arguments.
///
/// Build incrementally with [`ArgumentBufferLayout::builder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgumentBufferLayout {
    slots: Vec<ArgumentSlot>,
    encoded_length: usize,
}

impl ArgumentBufferLayout {
    /// Start building a layout.
    pub fn builder() -> ArgumentBufferLayoutBuilder {
        ArgumentBufferLayoutBuilder::default()
    }

    /// All laid-out slots in `[[id]]` order.
    pub fn slots(&self) -> &[ArgumentSlot] {
        &self.slots
    }

    /// Total encoded length in bytes (`MTLArgumentEncoder.encodedLength`).
    ///
    /// Rounded up to [`crate::storage::METAL_BUFFER_ALIGNMENT`] so the backing
    /// buffer satisfies Metal's offset-alignment requirement.
    pub fn encoded_length(&self) -> usize {
        self.encoded_length
    }

    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// `true` when there are no arguments.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Look up a slot by its `[[id(n)]]` index.
    pub fn slot(&self, id: u32) -> Option<&ArgumentSlot> {
        self.slots.iter().find(|s| s.id == id)
    }

    /// Count of slots that reference a separate `MTLResource` (need
    /// `useResource:` residency calls before dispatch).
    pub fn resource_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.kind.references_resource())
            .count()
    }
}

// ─── ArgumentBufferLayoutBuilder ───────────────────────────────────────────────

/// Incremental builder for an [`ArgumentBufferLayout`].
///
/// Arguments are assigned sequential `[[id(n)]]` indices starting from 0 and
/// packed at 8-byte-aligned offsets, exactly as `MTLArgumentEncoder` does.
#[derive(Debug, Default, Clone)]
pub struct ArgumentBufferLayoutBuilder {
    kinds: Vec<ArgumentKind>,
}

impl ArgumentBufferLayoutBuilder {
    /// Append a `device`/`constant` buffer argument.
    pub fn buffer(mut self) -> Self {
        self.kinds.push(ArgumentKind::Buffer);
        self
    }

    /// Append a texture argument.
    pub fn texture(mut self) -> Self {
        self.kinds.push(ArgumentKind::Texture);
        self
    }

    /// Append a sampler argument.
    pub fn sampler(mut self) -> Self {
        self.kinds.push(ArgumentKind::Sampler);
        self
    }

    /// Append an inline constant of `bytes` bytes.
    pub fn inline_constant(mut self, bytes: usize) -> Self {
        self.kinds.push(ArgumentKind::InlineConstant(bytes));
        self
    }

    /// Append an explicitly-kinded argument.
    pub fn arg(mut self, kind: ArgumentKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Finalise the layout, computing offsets and total encoded length.
    ///
    /// Returns [`MetalError::InvalidArgument`] if no arguments were added or if
    /// an inline constant declares a zero size.
    pub fn build(self) -> MetalResult<ArgumentBufferLayout> {
        if self.kinds.is_empty() {
            return Err(MetalError::InvalidArgument(
                "argument buffer must have at least one argument".into(),
            ));
        }
        let mut slots = Vec::with_capacity(self.kinds.len());
        let mut cursor = 0usize;
        for (i, kind) in self.kinds.into_iter().enumerate() {
            if let ArgumentKind::InlineConstant(0) = kind {
                return Err(MetalError::InvalidArgument(
                    "inline constant size must be > 0".into(),
                ));
            }
            let offset = align_up(cursor, ARGUMENT_SLOT_ALIGNMENT);
            slots.push(ArgumentSlot {
                id: i as u32,
                kind,
                offset,
            });
            cursor = offset + kind.encoded_size();
        }
        let encoded_length = align_up(cursor, crate::storage::METAL_BUFFER_ALIGNMENT);
        Ok(ArgumentBufferLayout {
            slots,
            encoded_length,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argument_kind_sizes() {
        assert_eq!(ArgumentKind::Buffer.encoded_size(), 8);
        assert_eq!(ArgumentKind::Texture.encoded_size(), 8);
        assert_eq!(ArgumentKind::Sampler.encoded_size(), 8);
        assert_eq!(ArgumentKind::InlineConstant(20).encoded_size(), 20);
    }

    #[test]
    fn references_resource_classification() {
        assert!(ArgumentKind::Buffer.references_resource());
        assert!(ArgumentKind::Texture.references_resource());
        assert!(!ArgumentKind::InlineConstant(4).references_resource());
    }

    #[test]
    fn layout_assigns_sequential_ids_and_offsets() {
        let layout = ArgumentBufferLayout::builder()
            .buffer() // id 0, offset 0
            .buffer() // id 1, offset 8
            .texture() // id 2, offset 16
            .build()
            .expect("build layout");
        assert_eq!(layout.len(), 3);
        assert_eq!(layout.slots()[0].id, 0);
        assert_eq!(layout.slots()[0].offset, 0);
        assert_eq!(layout.slots()[1].offset, 8);
        assert_eq!(layout.slots()[2].offset, 16);
        // 3 * 8 = 24 → rounds up to 256.
        assert_eq!(layout.encoded_length(), 256);
        assert_eq!(layout.resource_count(), 3);
    }

    #[test]
    fn inline_constant_advances_by_its_size_and_aligns() {
        let layout = ArgumentBufferLayout::builder()
            .buffer() // id 0, offset 0, size 8
            .inline_constant(20) // id 1, offset 8, size 20 → cursor 28
            .buffer() // id 2, offset align_up(28,8)=32
            .build()
            .expect("build");
        assert_eq!(layout.slots()[1].offset, 8);
        assert_eq!(layout.slots()[2].offset, 32);
        // Only the two buffers reference resources.
        assert_eq!(layout.resource_count(), 2);
    }

    #[test]
    fn slot_lookup_by_id() {
        let layout = ArgumentBufferLayout::builder()
            .buffer()
            .sampler()
            .build()
            .expect("build");
        assert_eq!(layout.slot(1).unwrap().kind, ArgumentKind::Sampler);
        assert!(layout.slot(99).is_none());
    }

    #[test]
    fn empty_layout_is_error() {
        let err = ArgumentBufferLayout::builder().build().unwrap_err();
        assert!(matches!(err, MetalError::InvalidArgument(_)));
    }

    #[test]
    fn zero_size_inline_constant_is_error() {
        let err = ArgumentBufferLayout::builder()
            .inline_constant(0)
            .build()
            .unwrap_err();
        assert!(matches!(err, MetalError::InvalidArgument(_)));
    }

    #[test]
    fn arg_explicit_kind() {
        let layout = ArgumentBufferLayout::builder()
            .arg(ArgumentKind::Buffer)
            .arg(ArgumentKind::InlineConstant(4))
            .build()
            .expect("build");
        assert_eq!(layout.len(), 2);
        assert!(!layout.is_empty());
    }

    #[test]
    fn display_argument_kinds() {
        assert_eq!(ArgumentKind::Buffer.to_string(), "Buffer");
        assert_eq!(
            ArgumentKind::InlineConstant(16).to_string(),
            "InlineConstant(16)"
        );
    }
}
