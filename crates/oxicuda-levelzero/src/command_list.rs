//! Host-side Level Zero command-list recording and dispatch planning.
//!
//! A Level Zero `ze_command_list_handle_t` is *recorded* on the host: the
//! application appends memory copies, kernel launches, and barriers, then
//! *closes* the list and submits it to a queue. The recording itself —
//! validating ordinals, computing `ze_group_count_t` dispatch dimensions,
//! sequencing barriers and events — is pure host-side logic that this module
//! models as an in-memory [`CommandListRecorder`]. The only device-gated step
//! is replaying the recorded commands through `zeCommandListAppend*`.
//!
//! This module also provides:
//! - [`GroupCountPlanner`] — converts a global problem size into the
//!   `(group_count_x, group_count_y, group_count_z)` launch grid for a given
//!   workgroup size, mirroring `zeKernelSuggestGroupSize` arithmetic.
//! - [`CommandQueueGroupTable`] — selects a compute vs. copy queue-group
//!   ordinal from a queried `zeDeviceGetCommandQueueGroupProperties` table.
//! - [`EventPoolDesc`] / [`Barrier`] — event-pool sizing and barrier
//!   dependency construction.
//! - [`ReusableCommandList`] — models `zeCommandListReset` + re-record for
//!   repeated launches without re-allocation.

use crate::error::{LevelZeroError, LevelZeroResult};

// ─── Group-count dispatch planning ───────────────────────────

/// A `ze_group_count_t` — the number of workgroups launched in each dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupCount {
    /// Workgroups in X.
    pub x: u32,
    /// Workgroups in Y.
    pub y: u32,
    /// Workgroups in Z.
    pub z: u32,
}

impl GroupCount {
    /// Total number of workgroups across all three dimensions.
    #[must_use]
    pub fn total(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

/// Plans `ze_group_count_t` launch grids from global problem sizes.
///
/// Given a per-dimension workgroup (local) size, `plan_*` computes the smallest
/// group count whose `group_count * local_size >= global_size`, i.e. the
/// ceiling division used by `zeCommandListAppendLaunchKernel`. The kernel's
/// bounds check (`if (gid < count)`) masks the overhang.
#[derive(Debug, Clone, Copy)]
pub struct GroupCountPlanner {
    local_x: u32,
    local_y: u32,
    local_z: u32,
}

impl GroupCountPlanner {
    /// Create a 1-D planner with the given workgroup size.
    pub fn new_1d(local_x: u32) -> LevelZeroResult<Self> {
        Self::new_3d(local_x, 1, 1)
    }

    /// Create a 3-D planner. Every local dimension must be non-zero.
    pub fn new_3d(local_x: u32, local_y: u32, local_z: u32) -> LevelZeroResult<Self> {
        if local_x == 0 || local_y == 0 || local_z == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "workgroup size must be non-zero in every dimension".into(),
            ));
        }
        Ok(Self {
            local_x,
            local_y,
            local_z,
        })
    }

    /// The configured local (workgroup) size.
    #[must_use]
    pub fn local_size(&self) -> (u32, u32, u32) {
        (self.local_x, self.local_y, self.local_z)
    }

    /// Plan a 1-D launch grid covering `global_x` work-items.
    pub fn plan_1d(&self, global_x: u64) -> LevelZeroResult<GroupCount> {
        self.plan_3d(global_x, 1, 1)
    }

    /// Plan a 3-D launch grid covering `(global_x, global_y, global_z)`
    /// work-items.
    ///
    /// Returns [`LevelZeroError::InvalidArgument`] if any dimension's group
    /// count would overflow `u32` (the Level Zero limit).
    pub fn plan_3d(
        &self,
        global_x: u64,
        global_y: u64,
        global_z: u64,
    ) -> LevelZeroResult<GroupCount> {
        let x = ceil_div_u32(global_x, self.local_x)?;
        let y = ceil_div_u32(global_y, self.local_y)?;
        let z = ceil_div_u32(global_z, self.local_z)?;
        Ok(GroupCount { x, y, z })
    }
}

/// Ceiling division `ceil(num / den)` clamped to the `u32` group-count limit.
fn ceil_div_u32(num: u64, den: u32) -> LevelZeroResult<u32> {
    if num == 0 {
        return Ok(0);
    }
    let groups = num.div_ceil(u64::from(den));
    u32::try_from(groups).map_err(|_| {
        LevelZeroError::InvalidArgument(format!("group count {groups} exceeds u32 dispatch limit"))
    })
}

// ─── Command-queue group selection ───────────────────────────

/// One entry of `zeDeviceGetCommandQueueGroupProperties`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandQueueGroupInfo {
    /// The ordinal passed to command-list / command-queue descriptors.
    pub ordinal: u32,
    /// `true` when this group can execute compute kernels.
    pub compute: bool,
    /// `true` when this group can execute memory-copy commands.
    pub copy: bool,
    /// Number of physical queues in this group.
    pub num_queues: u32,
}

/// A queried command-queue-group property table for one device.
///
/// Intel GPUs expose a primary compute+copy group (ordinal 0) and, on
/// discrete parts, dedicated copy-only "link" engines. Selecting the right
/// ordinal lets copies overlap compute on a separate engine.
#[derive(Debug, Clone, Default)]
pub struct CommandQueueGroupTable {
    groups: Vec<CommandQueueGroupInfo>,
}

impl CommandQueueGroupTable {
    /// Build a table from queue-group records.
    #[must_use]
    pub fn new(groups: Vec<CommandQueueGroupInfo>) -> Self {
        Self { groups }
    }

    /// All queue groups, in driver-reported order.
    #[must_use]
    pub fn groups(&self) -> &[CommandQueueGroupInfo] {
        &self.groups
    }

    /// Select an ordinal that can run compute kernels (lowest ordinal first).
    pub fn select_compute_ordinal(&self) -> LevelZeroResult<u32> {
        self.groups
            .iter()
            .filter(|g| g.compute)
            .min_by_key(|g| g.ordinal)
            .map(|g| g.ordinal)
            .ok_or_else(|| {
                LevelZeroError::Unsupported("device exposes no compute queue group".into())
            })
    }

    /// Select an ordinal for asynchronous copies.
    ///
    /// Prefers a **copy-only** group (so it does not contend with compute);
    /// falls back to any copy-capable group. This is the basis for async copy
    /// queues discovered separately from compute queues.
    pub fn select_copy_ordinal(&self) -> LevelZeroResult<u32> {
        // First choice: a dedicated copy-only engine.
        if let Some(g) = self
            .groups
            .iter()
            .filter(|g| g.copy && !g.compute)
            .min_by_key(|g| g.ordinal)
        {
            return Ok(g.ordinal);
        }
        // Fallback: any copy-capable group.
        self.groups
            .iter()
            .filter(|g| g.copy)
            .min_by_key(|g| g.ordinal)
            .map(|g| g.ordinal)
            .ok_or_else(|| LevelZeroError::Unsupported("device exposes no copy queue group".into()))
    }

    /// `true` when a dedicated copy-only engine exists (enabling overlap).
    #[must_use]
    pub fn has_dedicated_copy_engine(&self) -> bool {
        self.groups.iter().any(|g| g.copy && !g.compute)
    }
}

// ─── Event pools and barriers ────────────────────────────────

/// Memory/host visibility scope for events in a pool (`ze_event_pool_flags_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventScope {
    /// Events visible only to the device (`0`).
    Device,
    /// `ZE_EVENT_POOL_FLAG_HOST_VISIBLE` — the host can wait on the event.
    HostVisible,
    /// `ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP` — events capture kernel timestamps.
    KernelTimestamp,
}

impl EventScope {
    /// The `ze_event_pool_flags_t` bitmask for this scope.
    #[must_use]
    pub fn ze_flags(self) -> u32 {
        match self {
            EventScope::Device => 0,
            EventScope::HostVisible => 0x1,
            EventScope::KernelTimestamp => 0x2,
        }
    }
}

/// Descriptor for a `zeEventPoolCreate` call.
///
/// An event pool is a fixed-capacity arena of events. The recorder hands out
/// monotonically-increasing event indices via [`EventPoolDesc::alloc_event`]
/// until the pool is exhausted, modelling the index bookkeeping that the
/// device-gated `zeEventCreate` performs.
#[derive(Debug, Clone)]
pub struct EventPoolDesc {
    /// Maximum number of events the pool can hold.
    pub capacity: u32,
    /// Visibility scope applied to all events.
    pub scope: EventScope,
    /// Next free event index.
    next_index: u32,
}

impl EventPoolDesc {
    /// Create a pool descriptor with `capacity` slots and `scope` visibility.
    pub fn new(capacity: u32, scope: EventScope) -> LevelZeroResult<Self> {
        if capacity == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "event pool capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity,
            scope,
            next_index: 0,
        })
    }

    /// Reserve the next event index in the pool.
    ///
    /// Returns [`LevelZeroError::OutOfMemory`] when the pool is exhausted.
    pub fn alloc_event(&mut self) -> LevelZeroResult<u32> {
        if self.next_index >= self.capacity {
            return Err(LevelZeroError::OutOfMemory);
        }
        let idx = self.next_index;
        self.next_index += 1;
        Ok(idx)
    }

    /// Number of events still available in the pool.
    #[must_use]
    pub fn remaining(&self) -> u32 {
        self.capacity - self.next_index
    }

    /// `true` when host-side `zeEventHostSynchronize` is permitted.
    #[must_use]
    pub fn is_host_visible(&self) -> bool {
        matches!(self.scope, EventScope::HostVisible)
    }
}

/// A barrier appended into a command list.
///
/// Models `zeCommandListAppendBarrier`: it signals `signal_event` (if any)
/// once all `wait_events` have completed. With no wait events it is a global
/// execution barrier across the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Barrier {
    /// Event signalled when the barrier completes (`None` = no signal).
    pub signal_event: Option<u32>,
    /// Events that must complete before the barrier passes.
    pub wait_events: Vec<u32>,
}

impl Barrier {
    /// A global barrier with no event dependencies.
    #[must_use]
    pub fn global() -> Self {
        Self {
            signal_event: None,
            wait_events: Vec::new(),
        }
    }

    /// A barrier that signals `event` once recorded work completes.
    #[must_use]
    pub fn signalling(event: u32) -> Self {
        Self {
            signal_event: Some(event),
            wait_events: Vec::new(),
        }
    }

    /// A barrier that waits on `events` before passing.
    #[must_use]
    pub fn waiting(events: Vec<u32>) -> Self {
        Self {
            signal_event: None,
            wait_events: events,
        }
    }
}

// ─── Recorded commands ───────────────────────────────────────

/// A single command appended into a [`CommandListRecorder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedCommand {
    /// `zeCommandListAppendMemoryCopy` of `bytes` bytes from src→dst handles.
    MemoryCopy {
        /// Destination buffer handle.
        dst: u64,
        /// Source buffer handle.
        src: u64,
        /// Number of bytes copied.
        bytes: u64,
    },
    /// `zeCommandListAppendMemoryFill` writing `pattern_len`-byte pattern.
    MemoryFill {
        /// Destination buffer handle.
        dst: u64,
        /// Bytes filled.
        bytes: u64,
        /// Pattern width in bytes.
        pattern_len: u32,
    },
    /// `zeCommandListAppendLaunchKernel` with a name and launch grid.
    LaunchKernel {
        /// Entry-point name of the launched kernel.
        kernel: String,
        /// The launch grid.
        groups: GroupCount,
        /// Event signalled on completion (`None` = no signal).
        signal_event: Option<u32>,
    },
    /// `zeCommandListAppendBarrier`.
    Barrier(Barrier),
}

/// State of a command list in the host-side model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandListState {
    /// Open for `append_*` calls.
    Recording,
    /// Closed; ready for submission, no further appends until reset.
    Closed,
}

/// An in-memory recording of a Level Zero command list.
///
/// Appends are buffered in order; [`close`](Self::close) seals the list (after
/// which appends are rejected) and [`reset`](Self::reset) clears it back to the
/// recording state for reuse — exactly the `zeCommandListReset` lifecycle.
#[derive(Debug, Clone)]
pub struct CommandListRecorder {
    /// Compute/copy queue-group ordinal this list targets.
    ordinal: u32,
    commands: Vec<RecordedCommand>,
    state: CommandListState,
}

impl CommandListRecorder {
    /// Begin recording a command list targeting queue-group `ordinal`.
    #[must_use]
    pub fn new(ordinal: u32) -> Self {
        Self {
            ordinal,
            commands: Vec::new(),
            state: CommandListState::Recording,
        }
    }

    /// The queue-group ordinal this list was created against.
    #[must_use]
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Current state of the list.
    #[must_use]
    pub fn state(&self) -> CommandListState {
        self.state
    }

    /// Number of recorded commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// `true` when no commands have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// The recorded command sequence.
    #[must_use]
    pub fn commands(&self) -> &[RecordedCommand] {
        &self.commands
    }

    fn ensure_recording(&self) -> LevelZeroResult<()> {
        if self.state != CommandListState::Recording {
            return Err(LevelZeroError::CommandListError(
                "command list is closed; reset before appending".into(),
            ));
        }
        Ok(())
    }

    /// Append a host→device or device→device memory copy.
    pub fn append_memory_copy(&mut self, dst: u64, src: u64, bytes: u64) -> LevelZeroResult<()> {
        self.ensure_recording()?;
        if bytes == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "memory copy size must be > 0".into(),
            ));
        }
        self.commands
            .push(RecordedCommand::MemoryCopy { dst, src, bytes });
        Ok(())
    }

    /// Append a memory fill writing a repeating pattern.
    pub fn append_memory_fill(
        &mut self,
        dst: u64,
        bytes: u64,
        pattern_len: u32,
    ) -> LevelZeroResult<()> {
        self.ensure_recording()?;
        if pattern_len == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "fill pattern length must be > 0".into(),
            ));
        }
        if bytes % u64::from(pattern_len) != 0 {
            return Err(LevelZeroError::InvalidArgument(
                "fill size must be a multiple of the pattern length".into(),
            ));
        }
        self.commands.push(RecordedCommand::MemoryFill {
            dst,
            bytes,
            pattern_len,
        });
        Ok(())
    }

    /// Append a kernel launch with the given grid and optional signal event.
    pub fn append_launch_kernel(
        &mut self,
        kernel: impl Into<String>,
        groups: GroupCount,
        signal_event: Option<u32>,
    ) -> LevelZeroResult<()> {
        self.ensure_recording()?;
        if groups.total() == 0 {
            return Err(LevelZeroError::InvalidArgument(
                "kernel launch grid must contain at least one workgroup".into(),
            ));
        }
        self.commands.push(RecordedCommand::LaunchKernel {
            kernel: kernel.into(),
            groups,
            signal_event,
        });
        Ok(())
    }

    /// Append an execution/event barrier.
    pub fn append_barrier(&mut self, barrier: Barrier) -> LevelZeroResult<()> {
        self.ensure_recording()?;
        self.commands.push(RecordedCommand::Barrier(barrier));
        Ok(())
    }

    /// Close the list, sealing it for submission.
    pub fn close(&mut self) -> LevelZeroResult<()> {
        self.ensure_recording()?;
        self.state = CommandListState::Closed;
        Ok(())
    }

    /// Reset the list back to the recording state, discarding all commands.
    pub fn reset(&mut self) {
        self.commands.clear();
        self.state = CommandListState::Recording;
    }

    /// Number of kernel-launch commands recorded.
    #[must_use]
    pub fn launch_count(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| matches!(c, RecordedCommand::LaunchKernel { .. }))
            .count()
    }
}

// ─── Reusable command list ───────────────────────────────────

/// A command list designed for repeated reset-and-re-record launch loops.
///
/// Mirrors the L0 §3.5.6 pattern: record once, submit, then `zeCommandListReset`
/// and re-record for the next iteration *without re-allocating* the list. The
/// [`reuse_count`](Self::reuse_count) tracks how many times the underlying list
/// has been recycled, which a backend uses to amortise allocation cost.
#[derive(Debug, Clone)]
pub struct ReusableCommandList {
    inner: CommandListRecorder,
    reuse_count: u64,
}

impl ReusableCommandList {
    /// Create a reusable list targeting queue-group `ordinal`.
    #[must_use]
    pub fn new(ordinal: u32) -> Self {
        Self {
            inner: CommandListRecorder::new(ordinal),
            reuse_count: 0,
        }
    }

    /// Mutable access to the underlying recorder for the current iteration.
    pub fn recorder_mut(&mut self) -> &mut CommandListRecorder {
        &mut self.inner
    }

    /// Shared access to the underlying recorder.
    #[must_use]
    pub fn recorder(&self) -> &CommandListRecorder {
        &self.inner
    }

    /// How many times the list has been reset for reuse.
    #[must_use]
    pub fn reuse_count(&self) -> u64 {
        self.reuse_count
    }

    /// Reset for the next iteration, incrementing the reuse counter.
    ///
    /// Returns [`LevelZeroError::CommandListError`] if the list was never
    /// closed (resetting an open recording would lose un-submitted work).
    pub fn recycle(&mut self) -> LevelZeroResult<()> {
        if self.inner.state() != CommandListState::Closed {
            return Err(LevelZeroError::CommandListError(
                "close the command list before recycling it".into(),
            ));
        }
        self.inner.reset();
        self.reuse_count += 1;
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_count_total() {
        let g = GroupCount { x: 4, y: 2, z: 3 };
        assert_eq!(g.total(), 24);
    }

    #[test]
    fn planner_1d_ceil_div() {
        let p = GroupCountPlanner::new_1d(256).unwrap();
        assert_eq!(p.local_size(), (256, 1, 1));
        // 1000 / 256 = ceil 4.
        assert_eq!(p.plan_1d(1000).unwrap(), GroupCount { x: 4, y: 1, z: 1 });
        // Exact multiple → no overhang.
        assert_eq!(p.plan_1d(512).unwrap(), GroupCount { x: 2, y: 1, z: 1 });
        // Zero work-items → zero groups.
        assert_eq!(p.plan_1d(0).unwrap(), GroupCount { x: 0, y: 1, z: 1 });
    }

    #[test]
    fn planner_3d_per_dimension() {
        let p = GroupCountPlanner::new_3d(16, 16, 1).unwrap();
        let g = p.plan_3d(100, 50, 8).unwrap();
        assert_eq!(g.x, 7); // ceil(100/16)
        assert_eq!(g.y, 4); // ceil(50/16)
        assert_eq!(g.z, 8); // ceil(8/1)
    }

    #[test]
    fn planner_rejects_zero_local_and_overflow() {
        assert!(GroupCountPlanner::new_3d(0, 1, 1).is_err());
        let p = GroupCountPlanner::new_1d(1).unwrap();
        // 2^33 work-items / local 1 overflows u32 group count.
        assert!(matches!(
            p.plan_1d(1u64 << 33),
            Err(LevelZeroError::InvalidArgument(_))
        ));
    }

    #[test]
    fn queue_group_selection() {
        let table = CommandQueueGroupTable::new(vec![
            CommandQueueGroupInfo {
                ordinal: 0,
                compute: true,
                copy: true,
                num_queues: 4,
            },
            CommandQueueGroupInfo {
                ordinal: 1,
                compute: false,
                copy: true,
                num_queues: 1,
            },
        ]);
        assert_eq!(table.select_compute_ordinal().unwrap(), 0);
        // Dedicated copy engine (ordinal 1) preferred for copies.
        assert_eq!(table.select_copy_ordinal().unwrap(), 1);
        assert!(table.has_dedicated_copy_engine());
        assert_eq!(table.groups().len(), 2);
    }

    #[test]
    fn queue_group_fallback_to_shared_copy() {
        let table = CommandQueueGroupTable::new(vec![CommandQueueGroupInfo {
            ordinal: 0,
            compute: true,
            copy: true,
            num_queues: 1,
        }]);
        // No dedicated engine → falls back to the compute+copy group.
        assert_eq!(table.select_copy_ordinal().unwrap(), 0);
        assert!(!table.has_dedicated_copy_engine());
    }

    #[test]
    fn queue_group_errors_when_missing_capability() {
        let copy_only = CommandQueueGroupTable::new(vec![CommandQueueGroupInfo {
            ordinal: 0,
            compute: false,
            copy: true,
            num_queues: 1,
        }]);
        assert!(matches!(
            copy_only.select_compute_ordinal(),
            Err(LevelZeroError::Unsupported(_))
        ));

        let compute_only = CommandQueueGroupTable::new(vec![CommandQueueGroupInfo {
            ordinal: 0,
            compute: true,
            copy: false,
            num_queues: 1,
        }]);
        assert!(matches!(
            compute_only.select_copy_ordinal(),
            Err(LevelZeroError::Unsupported(_))
        ));
    }

    #[test]
    fn event_pool_allocation() {
        let mut pool = EventPoolDesc::new(2, EventScope::HostVisible).unwrap();
        assert!(pool.is_host_visible());
        assert_eq!(pool.scope.ze_flags(), 0x1);
        assert_eq!(pool.remaining(), 2);
        assert_eq!(pool.alloc_event().unwrap(), 0);
        assert_eq!(pool.alloc_event().unwrap(), 1);
        assert_eq!(pool.remaining(), 0);
        assert!(matches!(
            pool.alloc_event(),
            Err(LevelZeroError::OutOfMemory)
        ));
    }

    #[test]
    fn event_pool_rejects_zero_capacity() {
        assert!(EventPoolDesc::new(0, EventScope::Device).is_err());
        assert_eq!(EventScope::KernelTimestamp.ze_flags(), 0x2);
        assert_eq!(EventScope::Device.ze_flags(), 0);
    }

    #[test]
    fn barrier_constructors() {
        let g = Barrier::global();
        assert!(g.signal_event.is_none());
        assert!(g.wait_events.is_empty());

        let s = Barrier::signalling(3);
        assert_eq!(s.signal_event, Some(3));

        let w = Barrier::waiting(vec![1, 2]);
        assert_eq!(w.wait_events, vec![1, 2]);
        assert!(w.signal_event.is_none());
    }

    #[test]
    fn recorder_records_in_order() {
        let mut rec = CommandListRecorder::new(0);
        assert_eq!(rec.ordinal(), 0);
        assert!(rec.is_empty());
        assert_eq!(rec.state(), CommandListState::Recording);

        rec.append_memory_copy(2, 1, 256).unwrap();
        rec.append_launch_kernel("gemm_f32", GroupCount { x: 4, y: 1, z: 1 }, Some(0))
            .unwrap();
        rec.append_barrier(Barrier::global()).unwrap();
        assert_eq!(rec.len(), 3);
        assert_eq!(rec.launch_count(), 1);

        match &rec.commands()[0] {
            RecordedCommand::MemoryCopy { dst, src, bytes } => {
                assert_eq!((*dst, *src, *bytes), (2, 1, 256));
            }
            other => panic!("expected MemoryCopy, got {other:?}"),
        }
    }

    #[test]
    fn recorder_validates_args() {
        let mut rec = CommandListRecorder::new(0);
        assert!(rec.append_memory_copy(1, 0, 0).is_err());
        assert!(rec.append_memory_fill(1, 10, 0).is_err());
        assert!(
            rec.append_memory_fill(1, 10, 4).is_err(),
            "10 not a multiple of 4"
        );
        rec.append_memory_fill(1, 12, 4).unwrap();
        assert!(
            rec.append_launch_kernel("k", GroupCount { x: 0, y: 1, z: 1 }, None)
                .is_err()
        );
    }

    #[test]
    fn recorder_close_blocks_append_until_reset() {
        let mut rec = CommandListRecorder::new(0);
        rec.append_memory_copy(2, 1, 128).unwrap();
        rec.close().unwrap();
        assert_eq!(rec.state(), CommandListState::Closed);
        assert!(matches!(
            rec.append_memory_copy(3, 1, 64),
            Err(LevelZeroError::CommandListError(_))
        ));
        // Double close is rejected.
        assert!(rec.close().is_err());

        rec.reset();
        assert_eq!(rec.state(), CommandListState::Recording);
        assert!(rec.is_empty());
        rec.append_memory_copy(3, 1, 64).unwrap();
    }

    #[test]
    fn reusable_command_list_recycle_loop() {
        let mut list = ReusableCommandList::new(0);
        assert_eq!(list.reuse_count(), 0);

        for expected in 0..3u64 {
            assert_eq!(list.reuse_count(), expected);
            let rec = list.recorder_mut();
            rec.append_launch_kernel("k", GroupCount { x: 1, y: 1, z: 1 }, None)
                .unwrap();
            rec.close().unwrap();
            list.recycle().unwrap();
        }
        assert_eq!(list.reuse_count(), 3);
        // After recycle the recorder is empty and recording again.
        assert!(list.recorder().is_empty());
        assert_eq!(list.recorder().state(), CommandListState::Recording);
    }

    #[test]
    fn reusable_rejects_recycle_while_open() {
        let mut list = ReusableCommandList::new(0);
        list.recorder_mut()
            .append_launch_kernel("k", GroupCount { x: 1, y: 1, z: 1 }, None)
            .unwrap();
        // Not closed yet → recycle must fail to avoid losing work.
        assert!(matches!(
            list.recycle(),
            Err(LevelZeroError::CommandListError(_))
        ));
        assert_eq!(list.reuse_count(), 0);
    }
}
