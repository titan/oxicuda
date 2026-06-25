//! Collective-communication *step schedules* for distributed inference.
//!
//! Real multi-GPU collectives (NCCL `ncclAllReduce`, `ncclAllGather`,
//! `ncclReduceScatter`) decompose a logical operation into a sequence of
//! point-to-point send/recv steps over a virtual topology (ring, tree). This
//! module generates those **step schedules** and *executes them over in-memory
//! buffers* so the schedule's correctness can be verified bit-exactly on a
//! single host — independently of any real interconnect.
//!
//! The schedules here are the algorithmic skeletons used by `oxicuda-driver` /
//! a future `oxicuda-collective` backend to drive real device transfers; the
//! in-memory executors are the *oracle* against which the device path is
//! checked.
//!
//! | Module | Algorithm |
//! |--------|-----------|
//! | [`ring`] | Ring all-reduce / reduce-scatter / all-gather (Baidu / NCCL ring) |
//! | [`tree`] | Recursive-halving/doubling & binary-tree reduce schedules |
//!
//! # What is *not* here
//!
//! Real device kernels and NIC/NVLink transfers are out of scope — those need
//! actual multi-GPU hardware. The PTX collective kernel *sources* live in
//! [`crate::ptx_kernels`]; this module is the host-side scheduling logic.
//!
//! # References
//! - Patarasuk & Yuan (2009) "Bandwidth optimal all-reduce algorithms for
//!   clusters of workstations." JPDC.
//! - Thakur, Rabenseifner, Gropp (2005) "Optimization of Collective
//!   Communication Operations in MPICH." IJHPCA — recursive halving/doubling.

pub mod ring;
pub mod tree;

pub use ring::{
    RingCollective, RingStep, execute_ring_all_gather, execute_ring_all_reduce,
    execute_ring_reduce_scatter, ring_all_gather_schedule, ring_all_reduce_schedule,
    ring_reduce_scatter_schedule,
};
pub use tree::{
    TreeStep, execute_recursive_doubling_all_gather, execute_recursive_halving_all_reduce,
    recursive_doubling_all_gather_schedule, recursive_halving_all_reduce_schedule,
};
