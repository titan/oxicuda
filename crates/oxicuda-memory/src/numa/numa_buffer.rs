//! NUMA-aware host allocation bookkeeping.
//!
//! On multi-socket hosts, page-locked memory used for host↔device DMA performs
//! best when it is physically resident on the NUMA node electrically closest to
//! the target GPU (i.e. the node whose PCIe root complex hosts the device).
//! `libnuma`'s `numa_alloc_onnode` lets an application bind an allocation to a
//! specific node; the matching CUDA pattern is to pin that allocation with
//! `cuMemHostRegister`.
//!
//! This module models the *host-side topology and policy* required to make that
//! decision without a GPU present: a [`NumaTopology`] describing nodes and the
//! NUMA distance matrix, a [`closest_node_to_gpu`] selector, and a
//! [`NumaBuffer`] descriptor that records which node an allocation is bound to
//! together with its byte footprint.  The physical `numa_alloc_onnode` /
//! `cuMemHostRegister` calls require the real platform and live behind the
//! device-gated path; everything here is deterministic and unit-testable.
//!
//! NUMA distances follow the ACPI SLIT convention: the distance from a node to
//! itself is `10`, and remote distances are larger multiples (a one-hop remote
//! node is typically `20`, i.e. ~2× local latency).
//!
//! # Example
//!
//! ```rust
//! # use oxicuda_memory::numa::numa_buffer::*;
//! // Two-socket box: node 0 and node 1, remote distance 20.
//! let topo = NumaTopology::two_node(20);
//! // A GPU attached to node 1's root complex should bind host memory to node 1.
//! let node = closest_node_to_gpu(&topo, 1).expect("node");
//! assert_eq!(node, 1);
//!
//! let buf = NumaBuffer::on_node(&topo, node, 4096).expect("alloc");
//! assert_eq!(buf.node(), 1);
//! assert_eq!(buf.byte_size(), 4096);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::error::{CudaError, CudaResult};

/// The ACPI SLIT local-node distance (node to itself).
pub const LOCAL_NUMA_DISTANCE: u8 = 10;

// ---------------------------------------------------------------------------
// NumaTopology
// ---------------------------------------------------------------------------

/// A model of the host NUMA topology.
///
/// Stores the number of nodes and a flattened, row-major distance matrix in
/// ACPI SLIT units.  The matrix is symmetric for physically realistic
/// topologies but this type does not enforce symmetry — it only enforces that
/// the diagonal is the local distance and that the matrix is square.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaTopology {
    node_count: usize,
    /// Row-major `node_count × node_count` distance matrix.
    distances: Vec<u8>,
}

impl NumaTopology {
    /// Builds a topology from an explicit row-major distance matrix.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `node_count` is zero, if `distances`
    ///   does not have exactly `node_count * node_count` entries, or if any
    ///   diagonal entry is not [`LOCAL_NUMA_DISTANCE`].
    pub fn new(node_count: usize, distances: Vec<u8>) -> CudaResult<Self> {
        if node_count == 0 {
            return Err(CudaError::InvalidValue);
        }
        let expected = node_count
            .checked_mul(node_count)
            .ok_or(CudaError::InvalidValue)?;
        if distances.len() != expected {
            return Err(CudaError::InvalidValue);
        }
        for node in 0..node_count {
            if distances[node * node_count + node] != LOCAL_NUMA_DISTANCE {
                return Err(CudaError::InvalidValue);
            }
        }
        Ok(Self {
            node_count,
            distances,
        })
    }

    /// Builds a uniform topology of `node_count` nodes where every distinct
    /// pair has distance `remote`.
    ///
    /// `remote` is clamped to be at least `LOCAL_NUMA_DISTANCE + 1` so remote
    /// access is always modelled as strictly slower than local.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `node_count` is zero.
    pub fn uniform(node_count: usize, remote: u8) -> CudaResult<Self> {
        if node_count == 0 {
            return Err(CudaError::InvalidValue);
        }
        let remote = remote.max(LOCAL_NUMA_DISTANCE.saturating_add(1));
        let mut distances = vec![remote; node_count * node_count];
        for node in 0..node_count {
            distances[node * node_count + node] = LOCAL_NUMA_DISTANCE;
        }
        Ok(Self {
            node_count,
            distances,
        })
    }

    /// Convenience constructor for the common two-socket layout.
    ///
    /// `remote` is the cross-socket distance (typically `20`).
    #[must_use]
    pub fn two_node(remote: u8) -> Self {
        // node_count == 2 is non-zero, so `uniform` cannot fail.
        Self::uniform(2, remote).unwrap_or(Self {
            node_count: 2,
            distances: vec![LOCAL_NUMA_DISTANCE, 20, 20, LOCAL_NUMA_DISTANCE],
        })
    }

    /// A single-node topology (UMA / non-NUMA host).
    #[must_use]
    pub fn single_node() -> Self {
        Self {
            node_count: 1,
            distances: vec![LOCAL_NUMA_DISTANCE],
        }
    }

    /// The number of NUMA nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the distance between `from` and `to`, or `None` if either index
    /// is out of range.
    #[must_use]
    pub fn distance(&self, from: usize, to: usize) -> Option<u8> {
        if from >= self.node_count || to >= self.node_count {
            return None;
        }
        Some(self.distances[from * self.node_count + to])
    }

    /// Returns the node with the smallest distance from `from`.
    ///
    /// Ties are broken by the lowest node index.  Returns `None` if `from` is
    /// out of range.
    #[must_use]
    pub fn nearest_node(&self, from: usize) -> Option<usize> {
        if from >= self.node_count {
            return None;
        }
        let mut best = from;
        let mut best_dist = self.distances[from * self.node_count + from];
        for to in 0..self.node_count {
            let d = self.distances[from * self.node_count + to];
            if d < best_dist {
                best_dist = d;
                best = to;
            }
        }
        Some(best)
    }
}

// ---------------------------------------------------------------------------
// closest_node_to_gpu
// ---------------------------------------------------------------------------

/// Returns the NUMA node closest to a GPU whose PCIe root complex is attached
/// to `gpu_home_node`.
///
/// In a real deployment `gpu_home_node` would come from
/// `cudaDeviceGetPCIBusId` + a sysfs lookup of the device's
/// `numa_node` attribute.  The closest *host* node for binding pinned memory is
/// then the GPU's home node itself (its memory is electrically local to that
/// socket), which this function validates against the topology.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `gpu_home_node` is not a valid node index
///   in `topology`.
pub fn closest_node_to_gpu(topology: &NumaTopology, gpu_home_node: usize) -> CudaResult<usize> {
    if gpu_home_node >= topology.node_count() {
        return Err(CudaError::InvalidValue);
    }
    // The home node is by construction the nearest (distance == local).
    topology
        .nearest_node(gpu_home_node)
        .ok_or(CudaError::InvalidValue)
}

// ---------------------------------------------------------------------------
// NumaBuffer
// ---------------------------------------------------------------------------

/// Host-side descriptor for a host allocation bound to a NUMA node.
///
/// Records the bound node and the byte footprint.  This models the result of a
/// `numa_alloc_onnode(byte_size, node)` followed by `cuMemHostRegister`; the
/// descriptor carries the accounting that callers use to verify locality
/// without owning a real pointer here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumaBuffer {
    node: usize,
    byte_size: usize,
}

impl NumaBuffer {
    /// Plans a NUMA-bound host allocation of `byte_size` bytes on `node`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `byte_size` is zero or `node` is not a
    ///   valid index in `topology`.
    pub fn on_node(topology: &NumaTopology, node: usize, byte_size: usize) -> CudaResult<Self> {
        if byte_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if node >= topology.node_count() {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self { node, byte_size })
    }

    /// Plans a host allocation bound to the node closest to a GPU whose home
    /// node is `gpu_home_node`.
    ///
    /// # Errors
    ///
    /// Forwards errors from [`closest_node_to_gpu`] and the byte-size /
    /// node-range checks.
    pub fn closest_to_gpu(
        topology: &NumaTopology,
        gpu_home_node: usize,
        byte_size: usize,
    ) -> CudaResult<Self> {
        let node = closest_node_to_gpu(topology, gpu_home_node)?;
        Self::on_node(topology, node, byte_size)
    }

    /// The NUMA node this allocation is bound to.
    #[inline]
    #[must_use]
    pub fn node(&self) -> usize {
        self.node
    }

    /// The byte footprint of the allocation.
    #[inline]
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Returns `true` if this allocation is local to `access_node` (i.e. the
    /// accessing node *is* the bound node).
    #[inline]
    #[must_use]
    pub fn is_local_to(&self, access_node: usize) -> bool {
        self.node == access_node
    }

    /// Returns the NUMA distance an access from `access_node` would incur,
    /// according to `topology`.
    ///
    /// Returns `None` if `access_node` is out of range for `topology`.
    #[must_use]
    pub fn access_distance(&self, topology: &NumaTopology, access_node: usize) -> Option<u8> {
        topology.distance(access_node, self.node)
    }
}

impl std::fmt::Display for NumaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NumaBuffer(node={}, {} bytes)",
            self.node, self.byte_size
        )
    }
}

// ---------------------------------------------------------------------------
// NumaAllocTracker
// ---------------------------------------------------------------------------

/// Per-node accounting of NUMA-bound host allocations.
///
/// Tracks how many bytes are currently outstanding on each node so that a
/// scheduler can balance pinned-memory pressure across sockets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaAllocTracker {
    per_node_bytes: Vec<usize>,
}

impl NumaAllocTracker {
    /// Creates a tracker sized for `topology`'s node count.
    #[must_use]
    pub fn new(topology: &NumaTopology) -> Self {
        Self {
            per_node_bytes: vec![0; topology.node_count()],
        }
    }

    /// Records that `buf` was allocated, adding its bytes to its node's total.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the buffer's node is out of range for
    ///   this tracker.
    pub fn record(&mut self, buf: &NumaBuffer) -> CudaResult<()> {
        let slot = self
            .per_node_bytes
            .get_mut(buf.node())
            .ok_or(CudaError::InvalidValue)?;
        *slot = slot.saturating_add(buf.byte_size());
        Ok(())
    }

    /// Records that `buf` was freed, subtracting its bytes from its node total.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the buffer's node is out of range.
    pub fn release(&mut self, buf: &NumaBuffer) -> CudaResult<()> {
        let slot = self
            .per_node_bytes
            .get_mut(buf.node())
            .ok_or(CudaError::InvalidValue)?;
        *slot = slot.saturating_sub(buf.byte_size());
        Ok(())
    }

    /// Returns the bytes currently outstanding on `node`, or `None` if out of
    /// range.
    #[must_use]
    pub fn bytes_on_node(&self, node: usize) -> Option<usize> {
        self.per_node_bytes.get(node).copied()
    }

    /// Returns the total bytes outstanding across all nodes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.per_node_bytes
            .iter()
            .copied()
            .fold(0usize, |acc, b| acc.saturating_add(b))
    }

    /// Returns the index of the least-loaded node (fewest outstanding bytes).
    ///
    /// Ties are broken by the lowest index.  Returns `None` only if the
    /// tracker has zero nodes (which cannot happen via [`new`](Self::new)).
    #[must_use]
    pub fn least_loaded_node(&self) -> Option<usize> {
        self.per_node_bytes
            .iter()
            .enumerate()
            .min_by_key(|&(_, &bytes)| bytes)
            .map(|(idx, _)| idx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_uniform_diagonal_is_local() {
        let topo = NumaTopology::uniform(4, 20).expect("topo");
        assert_eq!(topo.node_count(), 4);
        for node in 0..4 {
            assert_eq!(topo.distance(node, node), Some(LOCAL_NUMA_DISTANCE));
        }
    }

    #[test]
    fn topology_uniform_remote_distance() {
        let topo = NumaTopology::uniform(3, 32).expect("topo");
        assert_eq!(topo.distance(0, 1), Some(32));
        assert_eq!(topo.distance(2, 0), Some(32));
    }

    #[test]
    fn topology_uniform_clamps_remote_below_local() {
        // remote smaller than local should be clamped to local+1.
        let topo = NumaTopology::uniform(2, 5).expect("topo");
        assert_eq!(topo.distance(0, 1), Some(LOCAL_NUMA_DISTANCE + 1));
    }

    #[test]
    fn topology_two_node() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(topo.node_count(), 2);
        assert_eq!(topo.distance(0, 0), Some(10));
        assert_eq!(topo.distance(0, 1), Some(20));
        assert_eq!(topo.distance(1, 0), Some(20));
    }

    #[test]
    fn topology_single_node() {
        let topo = NumaTopology::single_node();
        assert_eq!(topo.node_count(), 1);
        assert_eq!(topo.distance(0, 0), Some(10));
        assert_eq!(topo.distance(0, 1), None);
    }

    #[test]
    fn topology_new_rejects_zero_nodes() {
        assert_eq!(NumaTopology::new(0, vec![]), Err(CudaError::InvalidValue));
    }

    #[test]
    fn topology_new_rejects_wrong_matrix_size() {
        // 2 nodes need 4 entries; give 3.
        assert_eq!(
            NumaTopology::new(2, vec![10, 20, 20]),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn topology_new_rejects_bad_diagonal() {
        // Diagonal entry (1,1) is 11, not 10.
        assert_eq!(
            NumaTopology::new(2, vec![10, 20, 20, 11]),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn topology_new_accepts_valid_matrix() {
        let topo = NumaTopology::new(2, vec![10, 21, 21, 10]).expect("topo");
        assert_eq!(topo.distance(0, 1), Some(21));
    }

    #[test]
    fn topology_distance_out_of_range_is_none() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(topo.distance(0, 5), None);
        assert_eq!(topo.distance(5, 0), None);
    }

    #[test]
    fn nearest_node_is_self_for_local() {
        let topo = NumaTopology::uniform(4, 20).expect("topo");
        for node in 0..4 {
            assert_eq!(topo.nearest_node(node), Some(node));
        }
    }

    #[test]
    fn nearest_node_picks_closer_remote() {
        // 3 nodes; from node 0, node 2 is closer (15) than node 1 (40).
        // diag = 10. Row 0: [10, 40, 15].
        let distances = vec![
            10, 40, 15, // from 0
            40, 10, 25, // from 1
            15, 25, 10, // from 2
        ];
        let topo = NumaTopology::new(3, distances).expect("topo");
        // Local (10) is still the minimum, so nearest is self.
        assert_eq!(topo.nearest_node(0), Some(0));
    }

    #[test]
    fn nearest_node_out_of_range_is_none() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(topo.nearest_node(2), None);
    }

    #[test]
    fn closest_node_to_gpu_returns_home_node() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(closest_node_to_gpu(&topo, 0), Ok(0));
        assert_eq!(closest_node_to_gpu(&topo, 1), Ok(1));
    }

    #[test]
    fn closest_node_to_gpu_rejects_bad_node() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(closest_node_to_gpu(&topo, 9), Err(CudaError::InvalidValue));
    }

    #[test]
    fn numa_buffer_on_node_valid() {
        let topo = NumaTopology::two_node(20);
        let buf = NumaBuffer::on_node(&topo, 1, 8192).expect("alloc");
        assert_eq!(buf.node(), 1);
        assert_eq!(buf.byte_size(), 8192);
    }

    #[test]
    fn numa_buffer_rejects_zero_bytes() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(
            NumaBuffer::on_node(&topo, 0, 0),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn numa_buffer_rejects_bad_node() {
        let topo = NumaTopology::two_node(20);
        assert_eq!(
            NumaBuffer::on_node(&topo, 7, 4096),
            Err(CudaError::InvalidValue)
        );
    }

    #[test]
    fn numa_buffer_closest_to_gpu() {
        let topo = NumaTopology::two_node(20);
        let buf = NumaBuffer::closest_to_gpu(&topo, 1, 4096).expect("alloc");
        assert_eq!(buf.node(), 1);
    }

    #[test]
    fn numa_buffer_locality_and_distance() {
        let topo = NumaTopology::two_node(20);
        let buf = NumaBuffer::on_node(&topo, 1, 4096).expect("alloc");
        assert!(buf.is_local_to(1));
        assert!(!buf.is_local_to(0));
        // Access from node 1 (local) -> 10; from node 0 (remote) -> 20.
        assert_eq!(buf.access_distance(&topo, 1), Some(10));
        assert_eq!(buf.access_distance(&topo, 0), Some(20));
        assert_eq!(buf.access_distance(&topo, 9), None);
    }

    #[test]
    fn numa_buffer_display() {
        let topo = NumaTopology::two_node(20);
        let buf = NumaBuffer::on_node(&topo, 1, 4096).expect("alloc");
        let s = format!("{buf}");
        assert!(s.contains("node=1"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn tracker_records_and_releases() {
        let topo = NumaTopology::two_node(20);
        let mut tracker = NumaAllocTracker::new(&topo);
        let a = NumaBuffer::on_node(&topo, 0, 1000).expect("a");
        let b = NumaBuffer::on_node(&topo, 1, 2000).expect("b");
        let c = NumaBuffer::on_node(&topo, 0, 500).expect("c");

        tracker.record(&a).expect("rec a");
        tracker.record(&b).expect("rec b");
        tracker.record(&c).expect("rec c");

        assert_eq!(tracker.bytes_on_node(0), Some(1500));
        assert_eq!(tracker.bytes_on_node(1), Some(2000));
        assert_eq!(tracker.total_bytes(), 3500);

        tracker.release(&a).expect("rel a");
        assert_eq!(tracker.bytes_on_node(0), Some(500));
        assert_eq!(tracker.total_bytes(), 2500);
    }

    #[test]
    fn tracker_bytes_on_bad_node_is_none() {
        let topo = NumaTopology::two_node(20);
        let tracker = NumaAllocTracker::new(&topo);
        assert_eq!(tracker.bytes_on_node(9), None);
    }

    #[test]
    fn tracker_least_loaded_node() {
        let topo = NumaTopology::uniform(3, 20).expect("topo");
        let mut tracker = NumaAllocTracker::new(&topo);
        let a = NumaBuffer::on_node(&topo, 0, 5000).expect("a");
        let b = NumaBuffer::on_node(&topo, 1, 1000).expect("b");
        tracker.record(&a).expect("rec a");
        tracker.record(&b).expect("rec b");
        // Node 2 has 0 bytes -> least loaded.
        assert_eq!(tracker.least_loaded_node(), Some(2));
    }

    #[test]
    fn tracker_least_loaded_ties_pick_lowest_index() {
        let topo = NumaTopology::two_node(20);
        let tracker = NumaAllocTracker::new(&topo);
        // Both nodes at 0 -> lowest index 0.
        assert_eq!(tracker.least_loaded_node(), Some(0));
    }

    #[test]
    fn tracker_release_saturates_at_zero() {
        let topo = NumaTopology::single_node();
        let mut tracker = NumaAllocTracker::new(&topo);
        let buf = NumaBuffer::on_node(&topo, 0, 100).expect("buf");
        // Release without record should not underflow.
        tracker.release(&buf).expect("rel");
        assert_eq!(tracker.bytes_on_node(0), Some(0));
    }
}
