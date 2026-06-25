//! NVLink fabric memory sharing — shareable handle export / import model.
//!
//! On a multi-GPU host wired with NVSwitch (or a multi-host NVLink fabric),
//! peer processes can share a single physical VMM allocation by exchanging an
//! opaque *fabric handle*.  The owning process calls
//! `cuMemExportToShareableHandle` with handle type
//! [`crate::ffi::CUmemAllocationHandleType::Fabric`] to materialise a 64-byte
//! `CUmemFabricHandle`; that blob is shipped to a peer (over a socket, MPI,
//! the file system, …) which calls `cuMemImportFromShareableHandle` to obtain
//! a [`crate::ffi::CUmemGenericAllocationHandle`] that maps the very same
//! physical memory.
//!
//! This module provides a **host-side behavioural model** of that protocol so
//! that the export/import lifecycle, ref-counting, and the (many) failure
//! modes can be exercised without a live multi-GPU fabric.  The real driver
//! calls are unavailable on CI, but the *state machine* — who may import,
//! when a handle becomes invalid, how a serialized blob round-trips — is fully
//! deterministic and is what this models.
//!
//! # Lifecycle
//!
//! ```text
//!   FabricMemory::allocate(props)         // exporter reserves physical memory
//!         │
//!         ▼
//!   .export_shareable()  ──►  FabricHandle (64-byte opaque, serializable)
//!         │                          │  ship to peer process
//!         │                          ▼
//!         │                   FabricHandle::from_bytes(blob)
//!         │                          │
//!         │                          ▼
//!         │                   importer.import(&handle)  ──►  FabricImport
//!         ▼
//!   .release()   (export count must reach zero before underlying free)
//! ```
//!
//! # What is and is not modelled
//!
//! * Modelled (host-side, deterministic): handle serialization to/from the
//!   64-byte wire form, fabric-uuid + offset addressing, import access-control
//!   against the exporter's permitted [`FabricAccess`] set, export/import
//!   reference counting, and every documented error path
//!   (`ERROR_NOT_PERMITTED`, `ERROR_INVALID_HANDLE`, `ERROR_NOT_SUPPORTED`).
//! * **Not** modelled (requires hardware): the actual DMA mapping, NVSwitch
//!   routing, and physical page table installation.

use crate::error::{CudaError, CudaResult};
use crate::ffi::{CUmemAllocationHandleType, CUmemGenericAllocationHandle};

/// Size in bytes of an opaque `CUmemFabricHandle` on the wire.
///
/// The CUDA C struct is `struct CUmemFabricHandle_st { unsigned char data[64]; }`.
pub const FABRIC_HANDLE_BYTES: usize = 64;

// ---------------------------------------------------------------------------
// FabricAccess
// ---------------------------------------------------------------------------

/// Access rights an importer is granted to a shared fabric allocation.
///
/// Mirrors the read / read-write distinction carried by `CUmemAccess_flags`
/// for VMM allocations, applied here to cross-process fabric imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FabricAccess {
    /// Importers may not map the memory at all (export is private).
    #[default]
    None,
    /// Importers may map the memory read-only.
    ReadOnly,
    /// Importers may map the memory for read and write.
    ReadWrite,
}

impl FabricAccess {
    /// Whether this access level permits any mapping at all.
    #[must_use]
    pub fn is_mappable(self) -> bool {
        !matches!(self, FabricAccess::None)
    }

    /// Whether this access level permits writes.
    #[must_use]
    pub fn is_writable(self) -> bool {
        matches!(self, FabricAccess::ReadWrite)
    }
}

// ---------------------------------------------------------------------------
// FabricAllocationProps
// ---------------------------------------------------------------------------

/// Properties describing a fabric-exportable VMM allocation.
///
/// These correspond to the fields of `CUmemAllocationProp` that matter for a
/// fabric export: the owning device, the granularity-rounded size, and the
/// requested handle types (which must include `Fabric`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FabricAllocationProps {
    /// Ordinal of the device that physically owns the memory.
    pub device_ordinal: i32,
    /// Allocation size in bytes (already rounded up to the fabric granularity).
    pub size: usize,
    /// Access level granted to importing peers.
    pub peer_access: FabricAccess,
    /// Handle types the allocation is exportable as (must contain `Fabric`).
    pub handle_type: CUmemAllocationHandleType,
}

impl FabricAllocationProps {
    /// Construct fabric props for the common case: a single device exporting a
    /// read-write region to its NVLink peers.
    #[must_use]
    pub fn read_write(device_ordinal: i32, size: usize) -> Self {
        Self {
            device_ordinal,
            size,
            peer_access: FabricAccess::ReadWrite,
            handle_type: CUmemAllocationHandleType::Fabric,
        }
    }

    /// Validate that these props are usable for a fabric export.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotSupported`] if `handle_type` is not `Fabric`.
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    fn validate(&self) -> CudaResult<()> {
        if self.handle_type != CUmemAllocationHandleType::Fabric {
            return Err(CudaError::NotSupported);
        }
        if self.size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FabricHandle — the 64-byte opaque wire form
// ---------------------------------------------------------------------------

/// An opaque, serializable fabric handle — the value produced by
/// `cuMemExportToShareableHandle` and consumed by
/// `cuMemImportFromShareableHandle`.
///
/// Internally this models the 64-byte `CUmemFabricHandle` as: an 8-byte magic
/// tag, a 16-byte fabric UUID identifying the physical allocation, an 8-byte
/// size, a 1-byte access level, and padding to 64 bytes.  Callers treat it as
/// opaque; only [`FabricHandle::to_bytes`] / [`FabricHandle::from_bytes`] are
/// meaningful, exactly as with the real driver type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FabricHandle {
    bytes: [u8; FABRIC_HANDLE_BYTES],
}

impl FabricHandle {
    /// 8-byte magic prefix identifying an OxiCUDA-modelled fabric handle.
    const MAGIC: [u8; 8] = *b"OXIFABRC";

    /// Build a fabric handle from its logical fields and serialize to the wire
    /// representation.
    fn encode(fabric_uuid: u128, size: usize, access: FabricAccess) -> Self {
        let mut bytes = [0u8; FABRIC_HANDLE_BYTES];
        bytes[0..8].copy_from_slice(&Self::MAGIC);
        bytes[8..24].copy_from_slice(&fabric_uuid.to_le_bytes());
        bytes[24..32].copy_from_slice(&(size as u64).to_le_bytes());
        bytes[32] = match access {
            FabricAccess::None => 0,
            FabricAccess::ReadOnly => 1,
            FabricAccess::ReadWrite => 2,
        };
        // Bytes 33..64 remain zero padding, matching the opaque driver struct.
        Self { bytes }
    }

    /// The raw 64-byte wire form, suitable for transmission to a peer process.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; FABRIC_HANDLE_BYTES] {
        self.bytes
    }

    /// Reconstruct a fabric handle from bytes received from a peer.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the slice is not exactly
    ///   [`FABRIC_HANDLE_BYTES`] long.
    /// * [`CudaError::InvalidHandle`] if the magic prefix does not match (the
    ///   blob was corrupted, truncated, or is not a fabric handle at all).
    pub fn from_bytes(raw: &[u8]) -> CudaResult<Self> {
        if raw.len() != FABRIC_HANDLE_BYTES {
            return Err(CudaError::InvalidValue);
        }
        if raw[0..8] != Self::MAGIC {
            return Err(CudaError::InvalidHandle);
        }
        let mut bytes = [0u8; FABRIC_HANDLE_BYTES];
        bytes.copy_from_slice(raw);
        Ok(Self { bytes })
    }

    /// The fabric UUID identifying the physical allocation this handle names.
    #[must_use]
    pub fn fabric_uuid(&self) -> u128 {
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&self.bytes[8..24]);
        u128::from_le_bytes(buf)
    }

    /// The size of the shared allocation, in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.bytes[24..32]);
        u64::from_le_bytes(buf) as usize
    }

    /// The access level the exporter granted to importers.
    #[must_use]
    pub fn access(&self) -> FabricAccess {
        match self.bytes[32] {
            1 => FabricAccess::ReadOnly,
            2 => FabricAccess::ReadWrite,
            _ => FabricAccess::None,
        }
    }
}

// ---------------------------------------------------------------------------
// FabricMemory — exporter side
// ---------------------------------------------------------------------------

/// A fabric-exportable physical allocation owned by the exporting process.
///
/// Models the allocation produced by `cuMemCreate` with a fabric-capable
/// `CUmemAllocationProp`.  Exports are reference-counted: the underlying memory
/// must not be released while any shareable handle is still outstanding, which
/// this type enforces via [`FabricMemory::release`].
#[derive(Debug)]
pub struct FabricMemory {
    handle: CUmemGenericAllocationHandle,
    fabric_uuid: u128,
    props: FabricAllocationProps,
    export_count: u32,
    released: bool,
}

impl FabricMemory {
    /// Reserve a fabric-exportable allocation.
    ///
    /// `handle` and `fabric_uuid` stand in for the driver-assigned allocation
    /// handle and the fabric-global identifier the driver would mint.  In the
    /// model the caller supplies them; on hardware they come from `cuMemCreate`
    /// and the fabric manager respectively.
    ///
    /// # Errors
    ///
    /// Propagates `FabricAllocationProps::validate` errors.
    pub fn allocate(
        handle: CUmemGenericAllocationHandle,
        fabric_uuid: u128,
        props: FabricAllocationProps,
    ) -> CudaResult<Self> {
        props.validate()?;
        Ok(Self {
            handle,
            fabric_uuid,
            props,
            export_count: 0,
            released: false,
        })
    }

    /// The driver allocation handle backing this memory.
    #[must_use]
    pub fn allocation_handle(&self) -> CUmemGenericAllocationHandle {
        self.handle
    }

    /// The allocation properties.
    #[must_use]
    pub fn props(&self) -> FabricAllocationProps {
        self.props
    }

    /// Number of shareable handles currently outstanding for this allocation.
    #[must_use]
    pub fn export_count(&self) -> u32 {
        self.export_count
    }

    /// Export a shareable [`FabricHandle`] for transmission to a peer.
    ///
    /// Equivalent to `cuMemExportToShareableHandle(&handle, alloc,
    /// CU_MEM_HANDLE_TYPE_FABRIC, 0)`.  Each successful export bumps the
    /// reference count.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidHandle`] if the allocation has been released.
    /// * [`CudaError::NotPermitted`] if `peer_access` is [`FabricAccess::None`]
    ///   (an allocation that grants no peer access cannot be shared).
    pub fn export_shareable(&mut self) -> CudaResult<FabricHandle> {
        if self.released {
            return Err(CudaError::InvalidHandle);
        }
        if !self.props.peer_access.is_mappable() {
            return Err(CudaError::NotPermitted);
        }
        self.export_count = self.export_count.saturating_add(1);
        Ok(FabricHandle::encode(
            self.fabric_uuid,
            self.props.size,
            self.props.peer_access,
        ))
    }

    /// Note that a previously exported handle has been retired by all its
    /// consumers, decrementing the export reference count.
    ///
    /// # Errors
    ///
    /// [`CudaError::IllegalState`] if the count is already zero (double-free of
    /// an export token).
    pub fn retire_export(&mut self) -> CudaResult<()> {
        if self.export_count == 0 {
            return Err(CudaError::IllegalState);
        }
        self.export_count -= 1;
        Ok(())
    }

    /// Release the underlying physical allocation (`cuMemRelease`).
    ///
    /// # Errors
    ///
    /// [`CudaError::IllegalState`] if any shareable handles are still
    /// outstanding — the fabric contract forbids freeing memory that a peer may
    /// still import, mirroring the driver's refusal to release a live handle.
    pub fn release(&mut self) -> CudaResult<()> {
        if self.export_count != 0 {
            return Err(CudaError::IllegalState);
        }
        self.released = true;
        Ok(())
    }

    /// Whether the underlying allocation has been released.
    #[must_use]
    pub fn is_released(&self) -> bool {
        self.released
    }
}

// ---------------------------------------------------------------------------
// FabricImporter / FabricImport — importer side
// ---------------------------------------------------------------------------

/// The importing process's view of the fabric.
///
/// Holds the importing device's ordinal and the synthetic allocation-handle
/// counter the driver would assign on each successful import.  A real importer
/// must reside on a device that is fabric-reachable from the exporter; the
/// model captures that via [`FabricImporter::set_reachable`].
#[derive(Debug)]
pub struct FabricImporter {
    device_ordinal: i32,
    reachable: bool,
    next_handle: u64,
}

impl FabricImporter {
    /// Create an importer bound to a device ordinal, assumed fabric-reachable.
    #[must_use]
    pub fn new(device_ordinal: i32) -> Self {
        Self {
            device_ordinal,
            reachable: true,
            next_handle: 1,
        }
    }

    /// The device ordinal this importer maps memory into.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Mark whether this importer's device is reachable across the fabric.
    ///
    /// An unreachable device models `cuMemImportFromShareableHandle` failing
    /// with `CUDA_ERROR_NOT_SUPPORTED` (no NVLink/NVSwitch path).
    pub fn set_reachable(&mut self, reachable: bool) {
        self.reachable = reachable;
    }

    /// Import a peer's shareable handle, producing a [`FabricImport`] that maps
    /// the shared physical memory into this importer's address space.
    ///
    /// Equivalent to `cuMemImportFromShareableHandle(&imported, handle,
    /// CU_MEM_HANDLE_TYPE_FABRIC)`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotSupported`] if this importer's device is not
    ///   fabric-reachable.
    /// * [`CudaError::NotPermitted`] if the exporter granted no mappable
    ///   access ([`FabricAccess::None`]).
    pub fn import(&mut self, handle: &FabricHandle) -> CudaResult<FabricImport> {
        if !self.reachable {
            return Err(CudaError::NotSupported);
        }
        let access = handle.access();
        if !access.is_mappable() {
            return Err(CudaError::NotPermitted);
        }
        let imported = self.next_handle;
        self.next_handle += 1;
        Ok(FabricImport {
            handle: imported,
            fabric_uuid: handle.fabric_uuid(),
            size: handle.size(),
            access,
            device_ordinal: self.device_ordinal,
            mapped: false,
        })
    }
}

/// A successfully imported fabric allocation in the importing process.
///
/// Carries the locally-assigned allocation handle plus the access level the
/// exporter permitted, which bounds whether [`FabricImport::map_writable`]
/// may succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FabricImport {
    handle: u64,
    fabric_uuid: u128,
    size: usize,
    access: FabricAccess,
    device_ordinal: i32,
    mapped: bool,
}

impl FabricImport {
    /// The importer-local allocation handle.
    #[must_use]
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// The fabric UUID — identical to the exporter's, proving both sides name
    /// the same physical allocation.
    #[must_use]
    pub fn fabric_uuid(&self) -> u128 {
        self.fabric_uuid
    }

    /// Size of the shared region in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The access level granted by the exporter.
    #[must_use]
    pub fn access(&self) -> FabricAccess {
        self.access
    }

    /// The device ordinal this import is mapped into.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Map the import read-only into the importer's virtual address space.
    ///
    /// # Errors
    ///
    /// [`CudaError::NotPermitted`] if the exporter granted no access.
    pub fn map_readable(&mut self) -> CudaResult<()> {
        if !self.access.is_mappable() {
            return Err(CudaError::NotPermitted);
        }
        self.mapped = true;
        Ok(())
    }

    /// Map the import read-write.
    ///
    /// # Errors
    ///
    /// [`CudaError::NotPermitted`] if the exporter granted only read-only (or
    /// no) access.
    pub fn map_writable(&mut self) -> CudaResult<()> {
        if !self.access.is_writable() {
            return Err(CudaError::NotPermitted);
        }
        self.mapped = true;
        Ok(())
    }

    /// Whether this import is currently mapped.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;

    fn make_exporter(access: FabricAccess) -> FabricMemory {
        let props = FabricAllocationProps {
            device_ordinal: 0,
            size: 2 * 1024 * 1024,
            peer_access: access,
            handle_type: CUmemAllocationHandleType::Fabric,
        };
        FabricMemory::allocate(42, UUID, props).expect("valid props")
    }

    #[test]
    fn props_require_fabric_handle_type() {
        let props = FabricAllocationProps {
            device_ordinal: 0,
            size: 4096,
            peer_access: FabricAccess::ReadWrite,
            handle_type: CUmemAllocationHandleType::PosixFileDescriptor,
        };
        assert!(matches!(
            FabricMemory::allocate(1, UUID, props),
            Err(CudaError::NotSupported)
        ));
    }

    #[test]
    fn props_reject_zero_size() {
        let props = FabricAllocationProps::read_write(0, 0);
        assert!(matches!(
            FabricMemory::allocate(1, UUID, props),
            Err(CudaError::InvalidValue)
        ));
    }

    #[test]
    fn handle_round_trips_through_bytes() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let handle = mem.export_shareable().expect("export ok");
        let wire = handle.to_bytes();
        assert_eq!(wire.len(), FABRIC_HANDLE_BYTES);

        let restored = FabricHandle::from_bytes(&wire).expect("decode ok");
        assert_eq!(restored, handle);
        assert_eq!(restored.fabric_uuid(), UUID);
        assert_eq!(restored.size(), 2 * 1024 * 1024);
        assert_eq!(restored.access(), FabricAccess::ReadWrite);
    }

    #[test]
    fn from_bytes_rejects_wrong_length() {
        assert!(matches!(
            FabricHandle::from_bytes(&[0u8; 32]),
            Err(CudaError::InvalidValue)
        ));
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut blob = [0u8; FABRIC_HANDLE_BYTES];
        blob[0] = b'X';
        assert!(matches!(
            FabricHandle::from_bytes(&blob),
            Err(CudaError::InvalidHandle)
        ));
    }

    #[test]
    fn export_bumps_reference_count() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        assert_eq!(mem.export_count(), 0);
        let _h1 = mem.export_shareable().unwrap();
        let _h2 = mem.export_shareable().unwrap();
        assert_eq!(mem.export_count(), 2);
    }

    #[test]
    fn none_access_cannot_be_exported() {
        let mut mem = make_exporter(FabricAccess::None);
        assert!(matches!(
            mem.export_shareable(),
            Err(CudaError::NotPermitted)
        ));
    }

    #[test]
    fn release_blocked_while_exports_outstanding() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let _h = mem.export_shareable().unwrap();
        assert!(matches!(mem.release(), Err(CudaError::IllegalState)));
        // After retiring the export, release succeeds.
        mem.retire_export().unwrap();
        mem.release().unwrap();
        assert!(mem.is_released());
    }

    #[test]
    fn retire_export_underflow_is_illegal_state() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        assert!(matches!(mem.retire_export(), Err(CudaError::IllegalState)));
    }

    #[test]
    fn export_after_release_fails() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        mem.release().unwrap();
        assert!(matches!(
            mem.export_shareable(),
            Err(CudaError::InvalidHandle)
        ));
    }

    #[test]
    fn import_maps_same_physical_memory() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let handle = mem.export_shareable().unwrap();
        let wire = handle.to_bytes();

        // Peer process side.
        let received = FabricHandle::from_bytes(&wire).unwrap();
        let mut importer = FabricImporter::new(1);
        let import = importer.import(&received).expect("import ok");

        // Same fabric UUID proves it is the same physical allocation.
        assert_eq!(import.fabric_uuid(), UUID);
        assert_eq!(import.size(), 2 * 1024 * 1024);
        assert_eq!(import.device_ordinal(), 1);
        assert_eq!(import.access(), FabricAccess::ReadWrite);
    }

    #[test]
    fn import_assigns_distinct_handles() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let handle = mem.export_shareable().unwrap();
        let mut importer = FabricImporter::new(2);
        let a = importer.import(&handle).unwrap();
        let b = importer.import(&handle).unwrap();
        assert_ne!(a.handle(), b.handle());
    }

    #[test]
    fn unreachable_importer_gets_not_supported() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let handle = mem.export_shareable().unwrap();
        let mut importer = FabricImporter::new(3);
        importer.set_reachable(false);
        assert!(matches!(
            importer.import(&handle),
            Err(CudaError::NotSupported)
        ));
    }

    #[test]
    fn read_only_export_blocks_writable_map() {
        let mut mem = make_exporter(FabricAccess::ReadOnly);
        let handle = mem.export_shareable().unwrap();
        let mut importer = FabricImporter::new(1);
        let mut import = importer.import(&handle).unwrap();
        // Read-only mapping succeeds…
        import.map_readable().unwrap();
        assert!(import.is_mapped());
        // …but a writable mapping is refused.
        assert!(matches!(
            import.map_writable(),
            Err(CudaError::NotPermitted)
        ));
    }

    #[test]
    fn read_write_export_allows_writable_map() {
        let mut mem = make_exporter(FabricAccess::ReadWrite);
        let handle = mem.export_shareable().unwrap();
        let mut importer = FabricImporter::new(1);
        let mut import = importer.import(&handle).unwrap();
        import.map_writable().unwrap();
        assert!(import.is_mapped());
    }

    #[test]
    fn fabric_access_predicates() {
        assert!(!FabricAccess::None.is_mappable());
        assert!(FabricAccess::ReadOnly.is_mappable());
        assert!(!FabricAccess::ReadOnly.is_writable());
        assert!(FabricAccess::ReadWrite.is_writable());
    }
}
