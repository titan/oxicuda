//! Host-side HIP stream / event ordering model and command recording.
//!
//! Models the *ordering semantics* of `hipStream_t` and `hipEvent_t` without a
//! live HIP runtime:
//!
//! - A [`StreamPlan`] records a sequence of commands (kernel launches, memory
//!   copies, event records, and cross-stream waits) on one or more streams.
//! - Each stream is a FIFO: commands on the same stream execute in submission
//!   order.
//! - `hipEventRecord` marks a point in a stream; `hipStreamWaitEvent` makes a
//!   stream block until that point is reached on its source stream.
//! - [`StreamPlan::validate`] simulates the partial order and detects
//!   unsatisfiable / cyclic waits (deadlocks) — exactly the analysis a HIP
//!   scheduler must satisfy, done entirely on CPU.

use crate::error::{RocmError, RocmResult};
use std::collections::HashMap;

// ─── Command ────────────────────────────────────────────────────────────────

/// A single command recorded onto a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamCommand {
    /// A kernel launch identified by its function name.
    KernelLaunch {
        /// Kernel entry-point name.
        name: String,
    },
    /// A host↔device or device↔device copy of `bytes` bytes.
    Memcpy {
        /// Direction of the transfer.
        kind: MemcpyKind,
        /// Number of bytes transferred.
        bytes: u64,
    },
    /// Records `event` at this point in the stream (`hipEventRecord`).
    RecordEvent {
        /// Event handle being recorded.
        event: u64,
    },
    /// Blocks this stream until `event` has been recorded on its source stream
    /// (`hipStreamWaitEvent`).
    WaitEvent {
        /// Event handle awaited.
        event: u64,
    },
}

/// HIP memory-copy direction (`hipMemcpyKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    /// `hipMemcpyHostToDevice` (kind = 1).
    HostToDevice,
    /// `hipMemcpyDeviceToHost` (kind = 2).
    DeviceToHost,
    /// `hipMemcpyDeviceToDevice` (kind = 3).
    DeviceToDevice,
    /// `hipMemcpyHostToHost` (kind = 0).
    HostToHost,
}

impl MemcpyKind {
    /// The numeric `hipMemcpyKind` enum value.
    pub fn hip_value(self) -> i32 {
        match self {
            MemcpyKind::HostToHost => 0,
            MemcpyKind::HostToDevice => 1,
            MemcpyKind::DeviceToHost => 2,
            MemcpyKind::DeviceToDevice => 3,
        }
    }

    /// `true` if the transfer touches device memory on either side.
    pub fn involves_device(self) -> bool {
        !matches!(self, MemcpyKind::HostToHost)
    }
}

// ─── StreamPlan ─────────────────────────────────────────────────────────────

/// A multi-stream command recording with event-based ordering.
///
/// Commands are appended in submission order; the `(stream, command)` pairs
/// preserve global submission order while each stream individually behaves as a
/// FIFO during simulation.
#[derive(Debug, Clone, Default)]
pub struct StreamPlan {
    ops: Vec<(u64, StreamCommand)>,
}

impl StreamPlan {
    /// Create an empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a kernel launch on `stream`.
    pub fn launch(&mut self, stream: u64, name: impl Into<String>) -> &mut Self {
        self.ops
            .push((stream, StreamCommand::KernelLaunch { name: name.into() }));
        self
    }

    /// Append a memory copy on `stream`.
    pub fn memcpy(&mut self, stream: u64, kind: MemcpyKind, bytes: u64) -> &mut Self {
        self.ops
            .push((stream, StreamCommand::Memcpy { kind, bytes }));
        self
    }

    /// Record `event` at the current point of `stream`.
    pub fn record_event(&mut self, stream: u64, event: u64) -> &mut Self {
        self.ops
            .push((stream, StreamCommand::RecordEvent { event }));
        self
    }

    /// Make `stream` wait for `event`.
    pub fn wait_event(&mut self, stream: u64, event: u64) -> &mut Self {
        self.ops.push((stream, StreamCommand::WaitEvent { event }));
        self
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// `true` if no commands are recorded.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// All distinct stream handles referenced by the plan.
    pub fn streams(&self) -> Vec<u64> {
        let mut seen: Vec<u64> = Vec::new();
        for (s, _) in &self.ops {
            if !seen.contains(s) {
                seen.push(*s);
            }
        }
        seen
    }

    /// Total bytes copied across all streams.
    pub fn total_copy_bytes(&self) -> u64 {
        self.ops
            .iter()
            .filter_map(|(_, c)| match c {
                StreamCommand::Memcpy { bytes, .. } => Some(*bytes),
                _ => None,
            })
            .fold(0u64, |acc, b| acc.saturating_add(b))
    }

    /// Validate that the ordering is realisable: every `WaitEvent` must wait on
    /// an event that is recorded *somewhere*, and the partial order must not
    /// deadlock.
    ///
    /// The simulation advances each stream's program counter whenever its head
    /// command is runnable (a wait becomes runnable once its event has been
    /// recorded). If a full pass makes no progress while commands remain, the
    /// blocked waits are unsatisfiable — a deadlock.
    ///
    /// # Errors
    ///
    /// - [`RocmError::InvalidArgument`] if a wait references an event that is
    ///   never recorded.
    /// - [`RocmError::DeviceError`] if the schedule deadlocks.
    pub fn validate(&self) -> RocmResult<()> {
        // Every event that is recorded anywhere.
        let mut recorded_anywhere: Vec<u64> = Vec::new();
        for (_, c) in &self.ops {
            if let StreamCommand::RecordEvent { event } = c {
                if !recorded_anywhere.contains(event) {
                    recorded_anywhere.push(*event);
                }
            }
        }
        // Reject waits on events that are never recorded.
        for (_, c) in &self.ops {
            if let StreamCommand::WaitEvent { event } = c {
                if !recorded_anywhere.contains(event) {
                    return Err(RocmError::InvalidArgument(format!(
                        "stream waits on event {event} that is never recorded"
                    )));
                }
            }
        }

        // Per-stream FIFO command queues.
        let mut queues: HashMap<u64, Vec<StreamCommand>> = HashMap::new();
        for (s, c) in &self.ops {
            queues.entry(*s).or_default().push(c.clone());
        }
        let mut pc: HashMap<u64, usize> = queues.keys().map(|&s| (s, 0usize)).collect();
        let mut events_done: Vec<u64> = Vec::new();

        let total = self.ops.len();
        let mut completed = 0usize;

        loop {
            let mut progressed = false;
            for (&stream, cmds) in &queues {
                loop {
                    // Single lookup reused for both the read (`idx`) and the
                    // write (`idx + 1`) below, instead of an `Index` read
                    // followed by a redundant `get_mut` re-lookup. `pc` is
                    // seeded from `queues.keys()` and `stream` is a key of
                    // `queues`, so this can only miss if that invariant is
                    // ever violated -- in which case we fail loudly with a
                    // Result instead of panicking.
                    let Some(pc_entry) = pc.get_mut(&stream) else {
                        return Err(RocmError::DeviceError(format!(
                            "internal error: no program counter tracked for stream {stream}"
                        )));
                    };
                    let idx = *pc_entry;
                    if idx >= cmds.len() {
                        break;
                    }
                    let runnable = match &cmds[idx] {
                        StreamCommand::WaitEvent { event } => events_done.contains(event),
                        _ => true,
                    };
                    if !runnable {
                        break;
                    }
                    // Execute the head command.
                    if let StreamCommand::RecordEvent { event } = &cmds[idx] {
                        if !events_done.contains(event) {
                            events_done.push(*event);
                        }
                    }
                    *pc_entry = idx + 1;
                    completed += 1;
                    progressed = true;
                }
            }
            if completed == total {
                return Ok(());
            }
            if !progressed {
                return Err(RocmError::DeviceError(
                    "stream plan deadlock: cyclic or unsatisfiable event waits".into(),
                ));
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memcpy_kind_values() {
        assert_eq!(MemcpyKind::HostToHost.hip_value(), 0);
        assert_eq!(MemcpyKind::HostToDevice.hip_value(), 1);
        assert_eq!(MemcpyKind::DeviceToHost.hip_value(), 2);
        assert_eq!(MemcpyKind::DeviceToDevice.hip_value(), 3);
        assert!(MemcpyKind::HostToDevice.involves_device());
        assert!(!MemcpyKind::HostToHost.involves_device());
    }

    #[test]
    fn empty_plan_validates() {
        let plan = StreamPlan::new();
        assert!(plan.is_empty());
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn single_stream_fifo_is_valid() {
        let mut plan = StreamPlan::new();
        plan.memcpy(1, MemcpyKind::HostToDevice, 1024)
            .launch(1, "gemm_f32")
            .memcpy(1, MemcpyKind::DeviceToHost, 1024);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.streams(), vec![1]);
        assert_eq!(plan.total_copy_bytes(), 2048);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn cross_stream_event_dependency_is_valid() {
        // Stream 1 produces, records event 100; stream 2 waits then consumes.
        let mut plan = StreamPlan::new();
        plan.launch(1, "producer")
            .record_event(1, 100)
            .wait_event(2, 100)
            .launch(2, "consumer");
        assert!(plan.validate().is_ok());
        assert_eq!(plan.streams().len(), 2);
    }

    #[test]
    fn wait_on_unrecorded_event_errors() {
        let mut plan = StreamPlan::new();
        plan.wait_event(2, 999).launch(2, "consumer");
        let err = plan.validate().unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn mutual_wait_deadlocks() {
        // Stream 1 waits for event 200 (recorded on stream 2 *after* its wait),
        // stream 2 waits for event 100 (recorded on stream 1 after *its* wait).
        let mut plan = StreamPlan::new();
        plan.wait_event(1, 200)
            .record_event(1, 100)
            .wait_event(2, 100)
            .record_event(2, 200);
        // Stream 1 head = wait(200) needs stream2 to record 200, but stream2
        // head = wait(100) needs stream1 to record 100 → cycle.
        let err = plan.validate().unwrap_err();
        assert!(matches!(err, RocmError::DeviceError(_)));
    }

    #[test]
    fn event_recorded_before_wait_same_pass() {
        // record on stream 1 happens, then both later waits succeed.
        let mut plan = StreamPlan::new();
        plan.record_event(1, 7)
            .wait_event(2, 7)
            .wait_event(3, 7)
            .launch(2, "k2")
            .launch(3, "k3");
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn fan_out_fan_in_is_valid() {
        // Stream 0 forks to 1 and 2 (event 1), they each record (events 11, 12),
        // stream 0 joins by waiting on both.
        let mut plan = StreamPlan::new();
        plan.launch(0, "split")
            .record_event(0, 1)
            .wait_event(1, 1)
            .wait_event(2, 1)
            .launch(1, "branch_a")
            .record_event(1, 11)
            .launch(2, "branch_b")
            .record_event(2, 12)
            .wait_event(0, 11)
            .wait_event(0, 12)
            .launch(0, "join");
        assert!(plan.validate().is_ok());
    }
}
