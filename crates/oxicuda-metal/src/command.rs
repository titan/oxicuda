//! In-memory command recording: indirect command buffers and blit operations.
//!
//! Two related host-side recording surfaces live here:
//!
//! * [`IndirectCommandBuffer`] models `MTLIndirectCommandBuffer` — a
//!   pre-encoded list of compute dispatches that the GPU can execute without
//!   per-command CPU encoding overhead (GPU-driven rendering / compute).  We
//!   record each `ICBComputeCommand` into a fixed-capacity list, exactly like
//!   `indirectComputeCommandAtIndex:`.
//!
//! * [`BlitCommandList`] models the operation list a `MTLBlitCommandEncoder`
//!   would record: buffer↔buffer copies, fills, and synchronise-resource
//!   operations, with overlap/bounds validation.
//!
//! Both are pure data structures, fully unit-testable without a device; the
//! backend replays them onto real Metal encoders on macOS.

use crate::error::{MetalError, MetalResult};

// ─── Indirect command buffer ───────────────────────────────────────────────────

/// A single compute dispatch recorded in an indirect command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ICBComputeCommand {
    /// Index of the pipeline state to use (into a backend-managed table).
    pub pipeline_index: u32,
    /// Threadgroups per grid (x, y, z).
    pub threadgroups: [u32; 3],
    /// Threads per threadgroup (x, y, z).
    pub threads_per_group: [u32; 3],
}

impl ICBComputeCommand {
    /// Total threadgroup count (`x * y * z`).
    pub fn total_threadgroups(&self) -> u64 {
        u64::from(self.threadgroups[0])
            * u64::from(self.threadgroups[1])
            * u64::from(self.threadgroups[2])
    }

    /// `true` when no dimension is zero (a non-empty dispatch).
    pub fn is_nonempty(&self) -> bool {
        self.threadgroups.iter().all(|&d| d > 0) && self.threads_per_group.iter().all(|&d| d > 0)
    }
}

/// A fixed-capacity, pre-encoded list of compute dispatches.
///
/// Mirrors `MTLIndirectCommandBuffer`: created with a maximum command count,
/// then each slot is filled via [`Self::set_compute_command`].  Slots may be left
/// empty (a no-op when executed) and the whole buffer can be `reset`.
#[derive(Debug, Clone)]
pub struct IndirectCommandBuffer {
    commands: Vec<Option<ICBComputeCommand>>,
}

impl IndirectCommandBuffer {
    /// Create an ICB with room for `max_command_count` commands.
    ///
    /// Returns [`MetalError::InvalidArgument`] if the capacity is zero.
    pub fn new(max_command_count: usize) -> MetalResult<Self> {
        if max_command_count == 0 {
            return Err(MetalError::InvalidArgument(
                "indirect command buffer capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            commands: vec![None; max_command_count],
        })
    }

    /// Maximum number of commands this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.commands.len()
    }

    /// Number of populated command slots.
    pub fn populated(&self) -> usize {
        self.commands.iter().filter(|c| c.is_some()).count()
    }

    /// Record `command` at `index` (`indirectComputeCommandAtIndex:`).
    ///
    /// Returns [`MetalError::InvalidArgument`] for an out-of-range index or an
    /// empty (zero-dimension) dispatch.
    pub fn set_compute_command(
        &mut self,
        index: usize,
        command: ICBComputeCommand,
    ) -> MetalResult<()> {
        if index >= self.commands.len() {
            return Err(MetalError::InvalidArgument(format!(
                "ICB index {index} out of range (capacity {})",
                self.commands.len()
            )));
        }
        if !command.is_nonempty() {
            return Err(MetalError::InvalidArgument(
                "ICB compute command has a zero dispatch dimension".into(),
            ));
        }
        self.commands[index] = Some(command);
        Ok(())
    }

    /// Get the command recorded at `index`, if any.
    pub fn command(&self, index: usize) -> Option<ICBComputeCommand> {
        self.commands.get(index).copied().flatten()
    }

    /// Clear a range of slots (`resetWithRange:`), inclusive of `start`,
    /// exclusive of `end`.
    pub fn reset_range(&mut self, start: usize, end: usize) -> MetalResult<()> {
        if start > end || end > self.commands.len() {
            return Err(MetalError::InvalidArgument(format!(
                "ICB reset range [{start}, {end}) invalid for capacity {}",
                self.commands.len()
            )));
        }
        for slot in &mut self.commands[start..end] {
            *slot = None;
        }
        Ok(())
    }

    /// Clear all slots.
    pub fn reset_all(&mut self) {
        for slot in &mut self.commands {
            *slot = None;
        }
    }

    /// Iterate the populated commands in slot order.
    pub fn iter_commands(&self) -> impl Iterator<Item = (usize, ICBComputeCommand)> + '_ {
        self.commands
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|cmd| (i, cmd)))
    }
}

// ─── Blit command list ─────────────────────────────────────────────────────────

/// A single blit operation recorded by a (virtual) blit-command encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlitOp {
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    CopyBuffer {
        /// Source buffer handle.
        src: u64,
        /// Source byte offset.
        src_offset: usize,
        /// Destination buffer handle.
        dst: u64,
        /// Destination byte offset.
        dst_offset: usize,
        /// Number of bytes to copy.
        size: usize,
    },
    /// `fillBuffer:range:value:`.
    Fill {
        /// Buffer handle.
        buffer: u64,
        /// Byte offset of the fill range.
        offset: usize,
        /// Length of the fill range in bytes.
        size: usize,
        /// Byte value written across the range.
        value: u8,
    },
    /// `synchronizeResource:` (flush a `Managed` buffer's GPU copy to CPU).
    Synchronize {
        /// Buffer handle to synchronise.
        buffer: u64,
    },
}

/// An ordered list of blit operations with bounds/overlap validation.
#[derive(Debug, Default, Clone)]
pub struct BlitCommandList {
    ops: Vec<BlitOp>,
}

impl BlitCommandList {
    /// Create an empty blit command list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// `true` when no operations are recorded.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// All recorded operations in order.
    pub fn ops(&self) -> &[BlitOp] {
        &self.ops
    }

    /// Record a buffer→buffer copy.
    ///
    /// Returns [`MetalError::InvalidArgument`] for a zero-size copy, or for a
    /// same-buffer copy whose source and destination ranges overlap (Metal
    /// requires non-overlapping ranges within a single buffer).
    pub fn copy_buffer(
        &mut self,
        src: u64,
        src_offset: usize,
        dst: u64,
        dst_offset: usize,
        size: usize,
    ) -> MetalResult<()> {
        if size == 0 {
            return Err(MetalError::InvalidArgument(
                "blit copy size must be > 0".into(),
            ));
        }
        if src == dst {
            let s0 = src_offset;
            let s1 = src_offset + size;
            let d0 = dst_offset;
            let d1 = dst_offset + size;
            if s0 < d1 && d0 < s1 {
                return Err(MetalError::InvalidArgument(
                    "blit copy source and destination ranges overlap within one buffer".into(),
                ));
            }
        }
        self.ops.push(BlitOp::CopyBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
        Ok(())
    }

    /// Record a buffer fill.
    pub fn fill_buffer(
        &mut self,
        buffer: u64,
        offset: usize,
        size: usize,
        value: u8,
    ) -> MetalResult<()> {
        if size == 0 {
            return Err(MetalError::InvalidArgument(
                "blit fill size must be > 0".into(),
            ));
        }
        self.ops.push(BlitOp::Fill {
            buffer,
            offset,
            size,
            value,
        });
        Ok(())
    }

    /// Record a resource synchronisation (flush a Managed buffer to the CPU).
    pub fn synchronize(&mut self, buffer: u64) {
        self.ops.push(BlitOp::Synchronize { buffer });
    }

    /// Total number of bytes moved by copy + fill operations (ignores
    /// synchronise ops).  Useful for blit-bandwidth accounting.
    pub fn total_bytes_moved(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                BlitOp::CopyBuffer { size, .. } | BlitOp::Fill { size, .. } => *size,
                BlitOp::Synchronize { .. } => 0,
            })
            .sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(pipeline: u32) -> ICBComputeCommand {
        ICBComputeCommand {
            pipeline_index: pipeline,
            threadgroups: [8, 4, 1],
            threads_per_group: [32, 1, 1],
        }
    }

    #[test]
    fn icb_capacity_validation() {
        assert!(IndirectCommandBuffer::new(0).is_err());
        let icb = IndirectCommandBuffer::new(4).expect("create icb");
        assert_eq!(icb.capacity(), 4);
        assert_eq!(icb.populated(), 0);
    }

    #[test]
    fn icb_set_and_get_command() {
        let mut icb = IndirectCommandBuffer::new(4).expect("icb");
        icb.set_compute_command(2, dispatch(7)).expect("set");
        assert_eq!(icb.populated(), 1);
        let c = icb.command(2).expect("command present");
        assert_eq!(c.pipeline_index, 7);
        assert_eq!(c.total_threadgroups(), 32); // 8*4*1
        assert!(icb.command(0).is_none());
    }

    #[test]
    fn icb_rejects_out_of_range_and_empty() {
        let mut icb = IndirectCommandBuffer::new(2).expect("icb");
        assert!(icb.set_compute_command(5, dispatch(0)).is_err());
        let empty = ICBComputeCommand {
            pipeline_index: 0,
            threadgroups: [0, 1, 1],
            threads_per_group: [32, 1, 1],
        };
        assert!(!empty.is_nonempty());
        assert!(icb.set_compute_command(0, empty).is_err());
    }

    #[test]
    fn icb_reset_range_and_all() {
        let mut icb = IndirectCommandBuffer::new(4).expect("icb");
        for i in 0..4 {
            icb.set_compute_command(i, dispatch(i as u32)).expect("set");
        }
        assert_eq!(icb.populated(), 4);
        icb.reset_range(1, 3).expect("reset range");
        assert_eq!(icb.populated(), 2);
        assert!(icb.command(0).is_some());
        assert!(icb.command(1).is_none());
        assert!(icb.command(3).is_some());
        assert!(icb.reset_range(3, 1).is_err()); // start > end
        icb.reset_all();
        assert_eq!(icb.populated(), 0);
    }

    #[test]
    fn icb_iter_commands_in_order() {
        let mut icb = IndirectCommandBuffer::new(4).expect("icb");
        icb.set_compute_command(3, dispatch(30)).expect("set");
        icb.set_compute_command(1, dispatch(10)).expect("set");
        let collected: Vec<_> = icb.iter_commands().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, 1);
        assert_eq!(collected[0].1.pipeline_index, 10);
        assert_eq!(collected[1].0, 3);
    }

    // ── Blit list ──

    #[test]
    fn blit_copy_records_op() {
        let mut list = BlitCommandList::new();
        list.copy_buffer(1, 0, 2, 0, 256).expect("copy");
        assert_eq!(list.len(), 1);
        assert_eq!(list.total_bytes_moved(), 256);
    }

    #[test]
    fn blit_rejects_zero_size() {
        let mut list = BlitCommandList::new();
        assert!(list.copy_buffer(1, 0, 2, 0, 0).is_err());
        assert!(list.fill_buffer(1, 0, 0, 0).is_err());
    }

    #[test]
    fn blit_rejects_overlapping_same_buffer_copy() {
        let mut list = BlitCommandList::new();
        // src [0,100) and dst [50,150) on buffer 1 overlap.
        assert!(list.copy_buffer(1, 0, 1, 50, 100).is_err());
        // Non-overlapping ranges on the same buffer are fine.
        assert!(list.copy_buffer(1, 0, 1, 100, 100).is_ok());
        // Different buffers never "overlap".
        assert!(list.copy_buffer(1, 0, 2, 0, 100).is_ok());
    }

    #[test]
    fn blit_fill_and_synchronize() {
        let mut list = BlitCommandList::new();
        list.fill_buffer(1, 0, 128, 0xFF).expect("fill");
        list.synchronize(1);
        assert_eq!(list.len(), 2);
        // Synchronise moves no bytes; only the fill counts.
        assert_eq!(list.total_bytes_moved(), 128);
        assert!(matches!(list.ops()[1], BlitOp::Synchronize { buffer: 1 }));
    }

    #[test]
    fn blit_empty_list() {
        let list = BlitCommandList::new();
        assert!(list.is_empty());
        assert_eq!(list.total_bytes_moved(), 0);
    }
}
