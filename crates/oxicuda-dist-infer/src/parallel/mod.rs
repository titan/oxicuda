//! Host-side collective simulations for tensor parallelism.
//!
//! These routines reproduce the arithmetic of Megatron-style tensor-parallel
//! collectives (shard / all-reduce / all-gather) entirely in-process, so the
//! correctness of a parallel decomposition can be validated without a real
//! multi-GPU NCCL backend.
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`tensor_parallel`] | `shard_matmul`, `all_reduce_sum`, `all_gather`, `all_gather_columns` |

pub mod tensor_parallel;

pub use tensor_parallel::{all_gather, all_gather_columns, all_reduce_sum, shard_matmul};
