//! Extended launch configuration types for `cuLaunchKernelEx` (CUDA 12.0+).
//!
//! Thread block cluster launch attributes, launch configuration, and related
//! types for the modern CUDA 12.x kernel launch API.

use super::CUstream;

// =========================================================================
// CuLaunchAttributeId — attribute discriminant
// =========================================================================

/// Attribute identifier for `CuLaunchAttribute`.
///
/// Controls which extended kernel launch feature is configured.
/// Used with `cuLaunchKernelEx` (CUDA 12.0+).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuLaunchAttributeId {
    /// No attribute (`CU_LAUNCH_ATTRIBUTE_IGNORE`).
    Ignore = 0,
    /// Access-policy window for persisting L2 cache
    /// (`CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW`).
    AccessPolicyWindow = 1,
    /// Cooperative kernel launch (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).
    Cooperative = 2,
    /// Stream synchronization policy
    /// (`CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY`).
    SynchronizationPolicy = 3,
    /// Specifies thread block cluster dimensions (sm_90+).
    ClusterDimension = 4,
    /// Controls cluster scheduling policy preference.
    ClusterSchedulingPolicyPreference = 5,
    /// Enables programmatic stream serialization.
    ProgrammaticStreamSerialization = 6,
    /// Specifies a programmatic completion event.
    ProgrammaticEvent = 7,
    /// Specifies kernel launch priority.
    Priority = 8,
    /// Maps memory synchronization domains.
    MemSyncDomainMap = 9,
    /// Sets memory synchronization domain.
    MemSyncDomain = 10,
    /// Specifies a launch completion event.
    LaunchCompletionEvent = 12,
    /// Configures device-updatable kernel node.
    DeviceUpdatableKernelNode = 13,
}

// =========================================================================
// CuLaunchAttributeClusterDim — cluster geometry
// =========================================================================

/// Cluster dimension for thread block clusters (sm_90+).
///
/// Specifies how many thread blocks form one cluster in each dimension.
/// Used inside [`CuLaunchAttributeValue`] when the attribute id is
/// [`CuLaunchAttributeId::ClusterDimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CuLaunchAttributeClusterDim {
    /// Cluster extent in X dimension.
    pub x: u32,
    /// Cluster extent in Y dimension.
    pub y: u32,
    /// Cluster extent in Z dimension.
    pub z: u32,
}

// =========================================================================
// CuLaunchAttributeValue — attribute value union
// =========================================================================

/// Value union for `CuLaunchAttribute`.
///
/// # Safety
///
/// This is a C union — callers must only read the field that matches
/// the accompanying [`CuLaunchAttributeId`] discriminant.
/// Padding ensures the union is always 64 bytes, matching the CUDA ABI.
#[repr(C)]
pub union CuLaunchAttributeValue {
    /// Cluster dimension configuration (when id == `ClusterDimension`).
    pub cluster_dim: CuLaunchAttributeClusterDim,
    /// Scalar u32 value (for single-word attributes).
    pub value_u32: u32,
    /// Raw padding to maintain 64-byte ABI alignment.
    pub pad: [u8; 64],
}

// Manual Clone/Copy for the union (derive cannot handle unions with non-Copy
// fields, but all union fields here are effectively POD).
// `Copy` is declared first so that the `Clone` impl can delegate to it.
impl Copy for CuLaunchAttributeValue {}

impl Clone for CuLaunchAttributeValue {
    fn clone(&self) -> Self {
        // Delegate to Copy — canonical approach for Copy types.
        *self
    }
}

// =========================================================================
// CuLaunchAttribute — single attribute entry
// =========================================================================

/// A single extended kernel launch attribute (id + value pair).
///
/// Used in the `attrs` array of [`CuLaunchConfig`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CuLaunchAttribute {
    /// Which feature this attribute configures.
    pub id: CuLaunchAttributeId,
    /// Alignment padding (must be zero).
    pub pad: [u8; 4],
    /// The attribute value — interpret according to `id`.
    pub value: CuLaunchAttributeValue,
}

// =========================================================================
// CuLaunchConfig — full launch configuration
// =========================================================================

/// Extended kernel launch configuration for `cuLaunchKernelEx` (CUDA 12.0+).
///
/// Supersedes the individual parameters of `cuLaunchKernel` and adds
/// support for thread block clusters, launch priorities, and other
/// CUDA 12.x features.
///
/// # Example
///
/// ```rust
/// use oxicuda_driver::ffi::{
///     CuLaunchConfig, CuLaunchAttribute, CuLaunchAttributeId,
///     CuLaunchAttributeValue, CuLaunchAttributeClusterDim, CUstream,
/// };
///
/// // Build a cluster-launch config for a 2×1×1 cluster.
/// let cluster_attr = CuLaunchAttribute {
///     id: CuLaunchAttributeId::ClusterDimension,
///     pad: [0u8; 4],
///     value: CuLaunchAttributeValue {
///         cluster_dim: CuLaunchAttributeClusterDim { x: 2, y: 1, z: 1 },
///     },
/// };
/// let _config = CuLaunchConfig {
///     grid_dim_x: 8,
///     grid_dim_y: 1,
///     grid_dim_z: 1,
///     block_dim_x: 256,
///     block_dim_y: 1,
///     block_dim_z: 1,
///     shared_mem_bytes: 0,
///     stream: CUstream::default(),
///     attrs: std::ptr::null(),
///     num_attrs: 0,
/// };
/// ```
#[repr(C)]
pub struct CuLaunchConfig {
    /// Grid dimension in X.
    pub grid_dim_x: u32,
    /// Grid dimension in Y.
    pub grid_dim_y: u32,
    /// Grid dimension in Z.
    pub grid_dim_z: u32,
    /// Block dimension in X (threads per block in X).
    pub block_dim_x: u32,
    /// Block dimension in Y.
    pub block_dim_y: u32,
    /// Block dimension in Z.
    pub block_dim_z: u32,
    /// Dynamic shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Stream to submit the kernel on.
    pub stream: CUstream,
    /// Pointer to an array of `num_attrs` attributes (may be null if zero).
    pub attrs: *const CuLaunchAttribute,
    /// Number of entries in `attrs`.
    pub num_attrs: u32,
}

// SAFETY: CuLaunchConfig is a plain data structure mirroring the CUDA ABI.
// The raw pointer `attrs` must be valid for the lifetime of the config, but
// the struct itself is Send + Sync because no interior mutation occurs.
unsafe impl Send for CuLaunchConfig {}
unsafe impl Sync for CuLaunchConfig {}

#[cfg(test)]
mod tests {
    use super::CuLaunchAttributeId;

    /// Discriminants must match `CUlaunchAttributeID_enum` from `cuda.h`
    /// exactly, otherwise `cuLaunchKernelEx` silently misconfigures kernels.
    #[test]
    fn launch_attribute_id_matches_header() {
        assert_eq!(CuLaunchAttributeId::Ignore as u32, 0);
        assert_eq!(CuLaunchAttributeId::AccessPolicyWindow as u32, 1);
        assert_eq!(CuLaunchAttributeId::Cooperative as u32, 2);
        assert_eq!(CuLaunchAttributeId::SynchronizationPolicy as u32, 3);
        assert_eq!(CuLaunchAttributeId::ClusterDimension as u32, 4);
        assert_eq!(
            CuLaunchAttributeId::ClusterSchedulingPolicyPreference as u32,
            5
        );
        assert_eq!(
            CuLaunchAttributeId::ProgrammaticStreamSerialization as u32,
            6
        );
        assert_eq!(CuLaunchAttributeId::ProgrammaticEvent as u32, 7);
        assert_eq!(CuLaunchAttributeId::Priority as u32, 8);
        assert_eq!(CuLaunchAttributeId::MemSyncDomainMap as u32, 9);
        assert_eq!(CuLaunchAttributeId::MemSyncDomain as u32, 10);
        assert_eq!(CuLaunchAttributeId::LaunchCompletionEvent as u32, 12);
        assert_eq!(CuLaunchAttributeId::DeviceUpdatableKernelNode as u32, 13);
    }
}
