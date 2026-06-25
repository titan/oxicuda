//! NUMA-aware host allocation support.
//!
//! This module groups the host-side NUMA topology and allocation-binding
//! bookkeeping.  See [`numa_buffer`] for the topology, node-selection, and
//! per-node accounting types.

pub mod numa_buffer;

pub use numa_buffer::{
    LOCAL_NUMA_DISTANCE, NumaAllocTracker, NumaBuffer, NumaTopology, closest_node_to_gpu,
};
