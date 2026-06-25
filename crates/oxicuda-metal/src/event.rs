//! `MTLEvent` / `MTLSharedEvent` and `MTLFence` ordering logic.
//!
//! Metal synchronises work across command queues with monotonic *events*: a
//! producer encodes `encodeSignalEvent:value:` to raise an event to some value,
//! and a consumer encodes `encodeWaitForEvent:value:` to block until the event
//! reaches (at least) that value.  `MTLFence` gives intra-queue resource
//! ordering between encoders.
//!
//! The *ordering semantics* — "is this wait already satisfied?", "does this
//! signal/wait graph contain a deadlock cycle?" — are pure logic and fully
//! unit-testable without a device.  This module provides an in-memory
//! event timeline and a dependency-graph checker the backend can use to
//! validate a synchronisation plan before recording it.

use crate::error::{MetalError, MetalResult};
use std::collections::HashMap;

// ─── MetalEvent ────────────────────────────────────────────────────────────────

/// A monotonic event value tracker, mirroring `MTLSharedEvent.signaledValue`.
///
/// The signalled value may only increase; attempting to lower it is rejected.
#[derive(Debug, Clone)]
pub struct MetalEvent {
    id: u64,
    signaled_value: u64,
    shared: bool,
}

impl MetalEvent {
    /// Create a device-local event (`MTLEvent`).
    pub fn new(id: u64) -> Self {
        Self {
            id,
            signaled_value: 0,
            shared: false,
        }
    }

    /// Create a cross-process shared event (`MTLSharedEvent`).
    pub fn new_shared(id: u64) -> Self {
        Self {
            id,
            signaled_value: 0,
            shared: true,
        }
    }

    /// Opaque event id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// `true` for a `MTLSharedEvent` (cross-queue / cross-process).
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Current signalled value.
    pub fn signaled_value(&self) -> u64 {
        self.signaled_value
    }

    /// Raise the event to `value`.  Must be strictly monotonic increasing.
    ///
    /// Returns [`MetalError::InvalidArgument`] if `value` is not greater than
    /// the current value (Metal events are monotonic).
    pub fn signal(&mut self, value: u64) -> MetalResult<()> {
        if value <= self.signaled_value {
            return Err(MetalError::InvalidArgument(format!(
                "event {} signal value {value} must exceed current {}",
                self.id, self.signaled_value
            )));
        }
        self.signaled_value = value;
        Ok(())
    }

    /// `true` when a wait for `value` is already satisfied.
    pub fn is_satisfied(&self, value: u64) -> bool {
        self.signaled_value >= value
    }
}

// ─── SyncOp ────────────────────────────────────────────────────────────────────

/// A single synchronisation operation on a queue's timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOp {
    /// Signal `event` to `value` (`encodeSignalEvent:value:`).
    Signal { event: u64, value: u64 },
    /// Wait until `event` reaches `value` (`encodeWaitForEvent:value:`).
    Wait { event: u64, value: u64 },
}

// ─── EventTimeline ─────────────────────────────────────────────────────────────

/// An ordered list of signal/wait operations across multiple queues, used to
/// validate a synchronisation plan for satisfiability and freedom from deadlock.
#[derive(Debug, Default, Clone)]
pub struct EventTimeline {
    /// Operations as `(queue_id, op)` in submission order.
    ops: Vec<(u64, SyncOp)>,
}

impl EventTimeline {
    /// Create an empty timeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a signal of `event` to `value` on `queue`.
    pub fn signal(&mut self, queue: u64, event: u64, value: u64) -> &mut Self {
        self.ops.push((queue, SyncOp::Signal { event, value }));
        self
    }

    /// Record a wait for `event` to reach `value` on `queue`.
    pub fn wait(&mut self, queue: u64, event: u64, value: u64) -> &mut Self {
        self.ops.push((queue, SyncOp::Wait { event, value }));
        self
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// `true` when no operations are recorded.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Validate that every `Wait` can eventually be satisfied by some earlier-or
    /// -later `Signal`, and that no waiting cycle deadlocks the plan.
    ///
    /// The check simulates progress: it repeatedly advances any op whose
    /// preconditions are met (signals always run; a wait runs once its event
    /// has reached the required value).  If a full pass makes no progress while
    /// operations remain, the remaining waits are unsatisfiable — a deadlock.
    ///
    /// Returns [`MetalError::CommandBufferError`] describing the stuck wait.
    pub fn validate(&self) -> MetalResult<()> {
        // Highest value each event will EVER reach (any signal, any order).
        let mut max_signal: HashMap<u64, u64> = HashMap::new();
        for (_, op) in &self.ops {
            if let SyncOp::Signal { event, value } = op {
                let e = max_signal.entry(*event).or_insert(0);
                *e = (*e).max(*value);
            }
        }

        // Per-queue program counters; a queue is sequential.
        let mut queues: HashMap<u64, Vec<SyncOp>> = HashMap::new();
        for (q, op) in &self.ops {
            queues.entry(*q).or_default().push(*op);
        }
        let mut pc: HashMap<u64, usize> = queues.keys().map(|&q| (q, 0)).collect();
        let mut current: HashMap<u64, u64> = HashMap::new();

        loop {
            let mut progressed = false;
            let mut all_done = true;
            for (&q, ops) in &queues {
                let i = pc[&q];
                if i >= ops.len() {
                    continue;
                }
                all_done = false;
                match ops[i] {
                    SyncOp::Signal { event, value } => {
                        let e = current.entry(event).or_insert(0);
                        *e = (*e).max(value);
                        *pc.get_mut(&q).expect("queue pc") = i + 1;
                        progressed = true;
                    }
                    SyncOp::Wait { event, value } => {
                        let have = current.get(&event).copied().unwrap_or(0);
                        if have >= value {
                            *pc.get_mut(&q).expect("queue pc") = i + 1;
                            progressed = true;
                        }
                    }
                }
            }
            if all_done {
                return Ok(());
            }
            if !progressed {
                // Find a stuck wait to report.
                for (&q, ops) in &queues {
                    let i = pc[&q];
                    if i < ops.len() {
                        if let SyncOp::Wait { event, value } = ops[i] {
                            return Err(MetalError::CommandBufferError(format!(
                                "deadlock: queue {q} waiting for event {event} value {value}, \
                                 max ever signalled = {}",
                                max_signal.get(&event).copied().unwrap_or(0)
                            )));
                        }
                    }
                }
                return Err(MetalError::CommandBufferError(
                    "synchronisation plan made no progress".into(),
                ));
            }
        }
    }
}

// ─── MetalFence ────────────────────────────────────────────────────────────────

/// An intra-queue `MTLFence` for ordering resource access between encoders.
///
/// A fence must be *updated* by a producing encoder before it is *waited on*
/// by a consuming encoder.  This tracker enforces that ordering invariant.
#[derive(Debug, Clone)]
pub struct MetalFence {
    id: u64,
    updated: bool,
}

impl MetalFence {
    /// Create a fence with the given id.
    pub fn new(id: u64) -> Self {
        Self { id, updated: false }
    }

    /// Fence id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// `true` once the fence has been updated by a producer.
    pub fn is_updated(&self) -> bool {
        self.updated
    }

    /// Mark the fence as updated (`updateFence:` after the producing encoder).
    pub fn update(&mut self) {
        self.updated = true;
    }

    /// Wait on the fence (`waitForFence:` before the consuming encoder).
    ///
    /// Returns [`MetalError::CommandBufferError`] if the fence was never
    /// updated — a wait that would block forever.
    pub fn wait(&self) -> MetalResult<()> {
        if !self.updated {
            return Err(MetalError::CommandBufferError(format!(
                "fence {} waited on before being updated",
                self.id
            )));
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_monotonic_signal() {
        let mut e = MetalEvent::new(1);
        assert_eq!(e.signaled_value(), 0);
        e.signal(5).expect("signal 5");
        assert_eq!(e.signaled_value(), 5);
        // Lowering or repeating is rejected.
        assert!(e.signal(5).is_err());
        assert!(e.signal(3).is_err());
        e.signal(6).expect("signal 6");
        assert_eq!(e.signaled_value(), 6);
    }

    #[test]
    fn event_satisfaction_and_shared_flag() {
        let mut e = MetalEvent::new_shared(7);
        assert!(e.is_shared());
        assert!(!e.is_satisfied(1));
        e.signal(3).expect("signal");
        assert!(e.is_satisfied(1));
        assert!(e.is_satisfied(3));
        assert!(!e.is_satisfied(4));
        assert_eq!(e.id(), 7);
    }

    #[test]
    fn timeline_valid_producer_consumer() {
        let mut t = EventTimeline::new();
        // Queue 1 signals event 10 to 1; queue 2 waits for it.
        t.signal(1, 10, 1).wait(2, 10, 1);
        assert_eq!(t.len(), 2);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn timeline_wait_before_signal_across_queues_ok() {
        // Even if the wait is recorded first, the cross-queue signal satisfies it.
        let mut t = EventTimeline::new();
        t.wait(2, 5, 1);
        t.signal(1, 5, 1);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn timeline_unsatisfiable_wait_is_deadlock() {
        let mut t = EventTimeline::new();
        // Nothing ever signals event 99 to 1.
        t.wait(1, 99, 1);
        let err = t.validate().unwrap_err();
        assert!(matches!(err, MetalError::CommandBufferError(_)));
    }

    #[test]
    fn timeline_cyclic_deadlock() {
        // Queue 1 waits on event B then signals A; queue 2 waits on A then
        // signals B. Neither can start → deadlock.
        let mut t = EventTimeline::new();
        t.wait(1, 2, 1).signal(1, 1, 1);
        t.wait(2, 1, 1).signal(2, 2, 1);
        assert!(t.validate().is_err());
    }

    #[test]
    fn timeline_empty_is_valid() {
        let t = EventTimeline::new();
        assert!(t.is_empty());
        assert!(t.validate().is_ok());
    }

    #[test]
    fn timeline_chained_dependencies() {
        // q1 -> event1 -> q2 -> event2 -> q3
        let mut t = EventTimeline::new();
        t.signal(1, 1, 1);
        t.wait(2, 1, 1).signal(2, 2, 1);
        t.wait(3, 2, 1);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn fence_update_then_wait() {
        let mut f = MetalFence::new(1);
        assert!(!f.is_updated());
        // Waiting before update would block forever.
        assert!(f.wait().is_err());
        f.update();
        assert!(f.is_updated());
        assert!(f.wait().is_ok());
        assert_eq!(f.id(), 1);
    }
}
