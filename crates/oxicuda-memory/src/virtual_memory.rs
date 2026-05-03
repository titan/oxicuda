//! Virtual memory management for fine-grained GPU address space control.
//!
//! This module provides abstractions for CUDA's virtual memory management
//! API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, etc.), which
//! allows separating the concepts of virtual address reservation and
//! physical memory allocation.
//!
//! # Concepts
//!
//! * **Virtual Address Range** — A reservation of contiguous virtual
//!   addresses in the GPU address space. No physical memory is committed
//!   until explicitly mapped.
//!
//! * **Physical Allocation** — A chunk of physical GPU memory that can
//!   be mapped to one or more virtual address ranges.
//!
//! * **Mapping** — The association of a physical allocation with a region
//!   of a virtual address range.
//!
//! # Use Cases
//!
//! * **Sparse arrays** — Reserve a large virtual range but only commit
//!   physical memory for the tiles/pages that are actually used.
//!
//! * **Resizable buffers** — Reserve a large virtual range up-front and
//!   map additional physical memory as the buffer grows, without changing
//!   the base address.
//!
//! * **Multi-GPU memory** — Map physical allocations from different devices
//!   into the same virtual address space.
//!
//! # Status
//!
//! The CUDA virtual-memory management entry points (`cuMemAddressReserve`,
//! `cuMemCreate`, `cuMemMap`, `cuMemUnmap`, `cuMemSetAccess`,
//! `cuMemRelease`, `cuMemAddressFree`) are now wired through
//! `oxicuda-driver`.  Operations forward to the driver when it is
//! available; on platforms without a CUDA driver (such as macOS),
//! [`oxicuda_driver::loader::try_driver`] returns
//! [`CudaError::NotInitialized`].  When the driver loads but a particular
//! VMM symbol is missing (older drivers), the corresponding method
//! returns [`CudaError::NotSupported`].
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_memory::virtual_memory::VirtualMemoryManager;
//!
//! // Reserve 1 GiB of virtual address space with 2 MiB alignment.
//! let va = VirtualMemoryManager::reserve(1 << 30, 1 << 21)?;
//! assert_eq!(va.size(), 1 << 30);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::fmt;

use oxicuda_driver::error::{CudaError, CudaResult, check};
use oxicuda_driver::ffi::{
    CUdeviceptr, CUmemAccessDesc, CUmemAllocationHandleType, CUmemAllocationProp,
    CUmemAllocationType, CUmemGenericAllocationHandle, CUmemLocation, CUmemLocationType,
};

// ---------------------------------------------------------------------------
// AccessFlags
// ---------------------------------------------------------------------------

/// Memory access permission flags for virtual memory mappings.
///
/// These flags control how a mapped virtual address range can be accessed
/// by a given device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AccessFlags {
    /// No access permitted. The mapping exists but cannot be read or written.
    #[default]
    None,
    /// Read-only access. The device can read but not write.
    Read,
    /// Full read-write access.
    ReadWrite,
}

impl fmt::Display for AccessFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Read => write!(f, "Read"),
            Self::ReadWrite => write!(f, "ReadWrite"),
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualAddressRange
// ---------------------------------------------------------------------------

/// A reserved range of virtual addresses in the GPU address space.
///
/// This represents a contiguous block of virtual addresses that has been
/// reserved but not necessarily backed by physical memory. Physical memory
/// is associated with the range via [`VirtualMemoryManager::map`].
///
/// # Note
///
/// On systems without CUDA virtual memory support, the `base` address
/// is set to 0 and operations on the range will return
/// [`CudaError::NotSupported`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualAddressRange {
    base: u64,
    size: usize,
    alignment: usize,
}

impl VirtualAddressRange {
    /// Returns the base virtual address of the range.
    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Returns the size of the range in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the alignment of the range in bytes.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns whether the range contains the given virtual address.
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base.saturating_add(self.size as u64)
    }

    /// Returns the end address (exclusive) of the range.
    #[inline]
    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.size as u64)
    }
}

impl fmt::Display for VirtualAddressRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VA[0x{:016x}..0x{:016x}, {} bytes, align={}]",
            self.base,
            self.end(),
            self.size,
            self.alignment,
        )
    }
}

// ---------------------------------------------------------------------------
// PhysicalAllocation
// ---------------------------------------------------------------------------

/// A physical memory allocation on a specific GPU device.
///
/// Physical allocations represent actual GPU VRAM that can be mapped
/// into virtual address ranges. Multiple virtual ranges can map to
/// the same physical allocation (aliasing).
///
/// # Note
///
/// On systems without CUDA virtual memory support, the `handle` is
/// set to 0 and the allocation is not backed by real memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalAllocation {
    handle: u64,
    size: usize,
    device_ordinal: i32,
}

impl PhysicalAllocation {
    /// Returns the opaque handle for this physical allocation.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Returns the size of this allocation in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device ordinal this allocation belongs to.
    #[inline]
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }
}

impl fmt::Display for PhysicalAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhysAlloc[handle=0x{:016x}, {} bytes, dev={}]",
            self.handle, self.size, self.device_ordinal,
        )
    }
}

// ---------------------------------------------------------------------------
// MappingRecord — tracks virtual-to-physical mappings
// ---------------------------------------------------------------------------

/// A record of a virtual-to-physical memory mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingRecord {
    /// Offset within the virtual address range where the mapping starts.
    pub va_offset: usize,
    /// Size of the mapped region in bytes.
    pub size: usize,
    /// Handle of the physical allocation backing this mapping.
    pub phys_handle: u64,
    /// Access permissions for this mapping.
    pub access: AccessFlags,
}

// ---------------------------------------------------------------------------
// VirtualMemoryManager
// ---------------------------------------------------------------------------

/// Manager for GPU virtual memory operations.
///
/// Provides methods for reserving virtual address ranges, allocating
/// physical memory, mapping/unmapping, and setting access permissions.
///
/// # Status
///
/// The underlying CUDA virtual memory driver functions
/// (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemUnmap`,
/// `cuMemSetAccess`, `cuMemRelease`, `cuMemAddressFree`) are wired
/// through `oxicuda-driver`.  On systems without a CUDA driver
/// the calls fail with [`CudaError::NotInitialized`]; on systems
/// with a driver that lacks a specific VMM symbol the calls fail
/// with [`CudaError::NotSupported`].
pub struct VirtualMemoryManager;

impl VirtualMemoryManager {
    /// Reserves a range of virtual addresses in the GPU address space.
    ///
    /// The reserved range is not backed by physical memory until
    /// [`map`](Self::map) is called.
    ///
    /// # Parameters
    ///
    /// * `size` - Size of the virtual range to reserve in bytes.
    ///   Must be a multiple of `alignment`.
    /// * `alignment` - Alignment requirement in bytes. Must be a power
    ///   of two and non-zero.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero, `alignment` is
    ///   zero, `alignment` is not a power of two, or `size` is not a
    ///   multiple of `alignment`.
    pub fn reserve(size: usize, alignment: usize) -> CudaResult<VirtualAddressRange> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CudaError::InvalidValue);
        }
        if size % alignment != 0 {
            return Err(CudaError::InvalidValue);
        }

        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_address_reserve.ok_or(CudaError::NotSupported)?;
        let mut base: CUdeviceptr = 0;
        // addr=0 lets the driver choose; flags=0 (reserved for future use).
        check(unsafe { f(&mut base, size, alignment, 0, 0) })?;

        Ok(VirtualAddressRange {
            base,
            size,
            alignment,
        })
    }

    /// Releases a previously reserved virtual address range.
    ///
    /// After this call, the virtual addresses are no longer reserved
    /// and may be reused by future reservations.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemAddressFree`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn release(va: VirtualAddressRange) -> CudaResult<()> {
        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_address_free.ok_or(CudaError::NotSupported)?;
        check(unsafe { f(va.base, va.size) })
    }

    /// Allocates physical memory on the specified device.
    ///
    /// The allocated memory is not accessible until mapped into a
    /// virtual address range via [`map`](Self::map).
    ///
    /// # Parameters
    ///
    /// * `size` - Size of the allocation in bytes. Must be non-zero.
    /// * `device_ordinal` - Ordinal of the device to allocate on.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero or `device_ordinal`
    ///   is negative.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemCreate`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn alloc_physical(size: usize, device_ordinal: i32) -> CudaResult<PhysicalAllocation> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if device_ordinal < 0 {
            return Err(CudaError::InvalidValue);
        }

        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_create.ok_or(CudaError::NotSupported)?;

        let prop = CUmemAllocationProp {
            alloc_type: CUmemAllocationType::Pinned as u32,
            requested_handle_types: CUmemAllocationHandleType::None as u32,
            location: CUmemLocation {
                loc_type: CUmemLocationType::Device as u32,
                id: device_ordinal,
            },
            ..CUmemAllocationProp::default()
        };

        let mut handle: CUmemGenericAllocationHandle = 0;
        check(unsafe { f(&mut handle, size, &prop, 0) })?;

        Ok(PhysicalAllocation {
            handle,
            size,
            device_ordinal,
        })
    }

    /// Frees a physical memory allocation.
    ///
    /// The allocation must not be currently mapped to any virtual range.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemRelease`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn free_physical(phys: PhysicalAllocation) -> CudaResult<()> {
        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_release.ok_or(CudaError::NotSupported)?;
        check(unsafe { f(phys.handle) })
    }

    /// Maps a physical allocation to a region of a virtual address range.
    ///
    /// After mapping, GPU kernels can access the virtual addresses and
    /// reads/writes will be routed to the physical memory.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to map into.
    /// * `phys` - The physical allocation to map.
    /// * `offset` - Byte offset within the virtual range at which to
    ///   start the mapping. Must be aligned to the VA's alignment.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `offset` is not aligned, or if
    ///   the physical allocation would extend past the end of the virtual
    ///   range.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemMap`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn map(
        va: &VirtualAddressRange,
        phys: &PhysicalAllocation,
        offset: usize,
    ) -> CudaResult<()> {
        // Validate alignment
        if va.alignment > 0 && offset % va.alignment != 0 {
            return Err(CudaError::InvalidValue);
        }
        // Validate bounds
        let end = offset
            .checked_add(phys.size)
            .ok_or(CudaError::InvalidValue)?;
        if end > va.size {
            return Err(CudaError::InvalidValue);
        }

        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_map.ok_or(CudaError::NotSupported)?;

        // ptr = base + offset (VA), offset_into_phys = 0, size = phys.size, flags = 0.
        let target_va: CUdeviceptr = va.base.saturating_add(offset as u64);
        check(unsafe { f(target_va, phys.size, 0, phys.handle, 0) })
    }

    /// Unmaps a region of a virtual address range.
    ///
    /// After unmapping, accesses to the affected virtual addresses will
    /// fault. The physical memory is not freed — it can be remapped
    /// elsewhere.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to unmap from.
    /// * `offset` - Byte offset within the range where unmapping starts.
    /// * `size` - Number of bytes to unmap.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the offset+size exceeds the
    ///   virtual range bounds.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemUnmap`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn unmap(va: &VirtualAddressRange, offset: usize, size: usize) -> CudaResult<()> {
        let end = offset.checked_add(size).ok_or(CudaError::InvalidValue)?;
        if end > va.size {
            return Err(CudaError::InvalidValue);
        }

        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_unmap.ok_or(CudaError::NotSupported)?;

        let target_va: CUdeviceptr = va.base.saturating_add(offset as u64);
        check(unsafe { f(target_va, size) })
    }

    /// Sets access permissions for a virtual address range on a device.
    ///
    /// This controls whether the specified device can read and/or write
    /// to the mapped virtual addresses.
    ///
    /// # Parameters
    ///
    /// * `va` - The virtual address range to set permissions on.
    /// * `device_ordinal` - The device to grant/deny access for.
    /// * `flags` - The access permission flags.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available
    ///   (e.g. on macOS).
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemSetAccess`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn set_access(
        va: &VirtualAddressRange,
        device_ordinal: i32,
        flags: AccessFlags,
    ) -> CudaResult<()> {
        let api = oxicuda_driver::loader::try_driver()?;
        let f = api.cu_mem_set_access.ok_or(CudaError::NotSupported)?;

        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                loc_type: CUmemLocationType::Device as u32,
                id: device_ordinal,
            },
            flags: match flags {
                AccessFlags::None => 0,
                AccessFlags::Read => 1,
                AccessFlags::ReadWrite => 3,
            },
        };

        check(unsafe { f(va.base, va.size, &desc, 1) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns `true` when the driver-failure error kind is acceptable for a
    /// no-GPU host.  The driver may be loaded but have no real GPU hardware or
    /// VMM granularity support, in which case it returns `InvalidValue`,
    /// `InvalidDevice`, `NoDevice`, or `InvalidContext` in addition to the
    /// canonical `NotInitialized` (no driver; macOS) and `NotSupported`
    /// (driver loaded but symbol missing) variants.
    fn is_driver_unavailable(err: &CudaError) -> bool {
        matches!(
            err,
            CudaError::NotInitialized
                | CudaError::NotSupported
                | CudaError::InvalidValue
                | CudaError::InvalidDevice
                | CudaError::NoDevice
                | CudaError::InvalidContext
        )
    }

    // -- Reservation: argument-validation paths --------------------------------

    #[test]
    fn reserve_zero_size_fails() {
        let result = VirtualMemoryManager::reserve(0, 4096);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_zero_alignment_fails() {
        let result = VirtualMemoryManager::reserve(4096, 0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_non_power_of_two_alignment_fails() {
        let result = VirtualMemoryManager::reserve(4096, 3);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn reserve_misaligned_size_fails() {
        // 4096+1 is not a multiple of 4096
        let result = VirtualMemoryManager::reserve(4097, 4096);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    // -- Reservation: driver-call path on hosts without a CUDA driver ----------

    /// On a host without a CUDA driver, `reserve` must fail cleanly with one of
    /// the driver-unavailability error kinds rather than panicking.
    #[test]
    fn reserve_no_driver_returns_driver_unavailable() {
        let result = VirtualMemoryManager::reserve(4096, 4096);
        match result {
            Ok(va) => {
                // Real CUDA driver present: the driver gave us a base.
                assert_eq!(va.size(), 4096);
                assert_eq!(va.alignment(), 4096);
            }
            Err(e) => assert!(
                is_driver_unavailable(&e),
                "unexpected error from reserve: {e:?}"
            ),
        }
    }

    // -- VirtualAddressRange accessor methods ---------------------------------

    #[test]
    fn virtual_address_range_contains_synthetic() {
        // Build the value-object directly so the test runs everywhere.
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 8192,
            alignment: 4096,
        };
        assert!(va.contains(va.base()));
        assert!(va.contains(va.base() + 1));
        assert!(va.contains(va.base() + 8191));
        assert!(!va.contains(va.end()));
        assert!(!va.contains(va.base().wrapping_sub(1)));
    }

    #[test]
    fn virtual_address_range_end_synthetic() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        assert_eq!(va.end(), va.base() + 4096);
    }

    #[test]
    fn virtual_address_range_display_synthetic() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        let disp = format!("{va}");
        assert!(disp.contains("VA["));
        assert!(disp.contains("4096 bytes"));
    }

    // -- Physical allocation: argument-validation and driver-unavailable path --

    #[test]
    fn alloc_physical_zero_size_fails() {
        let result = VirtualMemoryManager::alloc_physical(0, 0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn alloc_physical_negative_device_fails() {
        let result = VirtualMemoryManager::alloc_physical(4096, -1);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn alloc_physical_no_driver_returns_driver_unavailable() {
        let result = VirtualMemoryManager::alloc_physical(4096, 0);
        if let Err(e) = result {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
        // On a real CUDA box the call may succeed; we only require not-panic.
    }

    #[test]
    fn release_no_driver_returns_driver_unavailable() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        if let Err(e) = VirtualMemoryManager::release(va) {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
    }

    #[test]
    fn free_physical_no_driver_returns_driver_unavailable() {
        // Calling cuMemRelease with a fake handle when the driver is loaded is
        // undefined behaviour that can SIGSEGV the process.  This test only
        // covers the "driver not available" path.
        if oxicuda_driver::loader::try_driver().is_ok() {
            return;
        }
        let phys = PhysicalAllocation {
            handle: 1,
            size: 4096,
            device_ordinal: 0,
        };
        if let Err(e) = VirtualMemoryManager::free_physical(phys) {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
    }

    // -- map / unmap / set_access argument-validation paths --------------------

    #[test]
    fn map_validates_alignment() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 8192,
            alignment: 4096,
        };
        let phys = PhysicalAllocation {
            handle: 1,
            size: 4096,
            device_ordinal: 0,
        };
        // Offset 1 is not aligned to 4096
        let result = VirtualMemoryManager::map(&va, &phys, 1);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn map_validates_bounds() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        let phys = PhysicalAllocation {
            handle: 1,
            size: 8192, // larger than VA range
            device_ordinal: 0,
        };
        let result = VirtualMemoryManager::map(&va, &phys, 0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn map_no_driver_returns_driver_unavailable() {
        // Calling cuMemMap with a fake virtual address and fake handle when the
        // driver is loaded is undefined behaviour that can SIGSEGV the process.
        // This test only covers the "driver not available" path.
        if oxicuda_driver::loader::try_driver().is_ok() {
            return;
        }
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 8192,
            alignment: 4096,
        };
        let phys = PhysicalAllocation {
            handle: 1,
            size: 4096,
            device_ordinal: 0,
        };
        if let Err(e) = VirtualMemoryManager::map(&va, &phys, 0) {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
    }

    #[test]
    fn unmap_validates_bounds() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        let result = VirtualMemoryManager::unmap(&va, 0, 8192);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn unmap_no_driver_returns_driver_unavailable() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        if let Err(e) = VirtualMemoryManager::unmap(&va, 0, 4096) {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
    }

    #[test]
    fn set_access_no_driver_returns_driver_unavailable() {
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        if let Err(e) = VirtualMemoryManager::set_access(&va, 0, AccessFlags::ReadWrite) {
            assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            );
        }
    }

    // -- Plain value-object tests (platform-independent) -----------------------

    #[test]
    fn access_flags_default() {
        assert_eq!(AccessFlags::default(), AccessFlags::None);
    }

    #[test]
    fn access_flags_display() {
        assert_eq!(format!("{}", AccessFlags::None), "None");
        assert_eq!(format!("{}", AccessFlags::Read), "Read");
        assert_eq!(format!("{}", AccessFlags::ReadWrite), "ReadWrite");
    }

    #[test]
    fn physical_allocation_display() {
        let phys = PhysicalAllocation {
            handle: 0x1234,
            size: 4096,
            device_ordinal: 0,
        };
        let disp = format!("{phys}");
        assert!(disp.contains("4096 bytes"));
        assert!(disp.contains("dev=0"));
    }

    #[test]
    fn mapping_record_fields() {
        let record = MappingRecord {
            va_offset: 0,
            size: 4096,
            phys_handle: 42,
            access: AccessFlags::ReadWrite,
        };
        assert_eq!(record.va_offset, 0);
        assert_eq!(record.size, 4096);
        assert_eq!(record.phys_handle, 42);
        assert_eq!(record.access, AccessFlags::ReadWrite);
    }

    /// On macOS specifically, every driver-calling method must return
    /// [`CudaError::NotInitialized`] (no library to load).
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_paths_return_not_initialized() {
        assert_eq!(
            VirtualMemoryManager::reserve(4096, 4096),
            Err(CudaError::NotInitialized)
        );
        assert_eq!(
            VirtualMemoryManager::alloc_physical(4096, 0),
            Err(CudaError::NotInitialized)
        );
        let phys = PhysicalAllocation {
            handle: 1,
            size: 4096,
            device_ordinal: 0,
        };
        assert_eq!(
            VirtualMemoryManager::free_physical(phys.clone()),
            Err(CudaError::NotInitialized)
        );
        let va = VirtualAddressRange {
            base: 0x1_0000_0000,
            size: 4096,
            alignment: 4096,
        };
        assert_eq!(
            VirtualMemoryManager::release(va.clone()),
            Err(CudaError::NotInitialized)
        );
        assert_eq!(
            VirtualMemoryManager::map(&va, &phys, 0),
            Err(CudaError::NotInitialized)
        );
        assert_eq!(
            VirtualMemoryManager::unmap(&va, 0, 4096),
            Err(CudaError::NotInitialized)
        );
        assert_eq!(
            VirtualMemoryManager::set_access(&va, 0, AccessFlags::ReadWrite),
            Err(CudaError::NotInitialized)
        );
    }
}
