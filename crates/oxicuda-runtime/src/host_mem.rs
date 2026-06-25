//! CPU model of CUDA Runtime host-memory registration, mapped-memory
//! bookkeeping, IPC handle tables, and the peer-access matrix.
//!
//! These runtime facilities are mostly *bookkeeping*: the runtime maintains
//! tables mapping registered host ranges to device-visible pointers, mapping
//! IPC handles to the allocations they export, and tracking which device pairs
//! have peer access enabled.  The *table logic* (range lookup, double-register
//! detection, handle round-tripping, symmetric/asymmetric peer enable rules) is
//! fully CPU-modelable and is what this module implements.  Actual page-locking
//! and cross-process sharing require a GPU and are left to [`crate::memory`].
//!
//! Everything here is deterministic and GPU-free.

use std::collections::HashMap;

use crate::error::{CudaRtError, CudaRtResult};
use crate::memory::DevicePtr;

// ─── Host-register flags ─────────────────────────────────────────────────────

/// Flags for [`HostMemoryRegistry::register`] (mirrors `cudaHostRegister*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HostRegisterFlags(pub u32);

impl HostRegisterFlags {
    /// Default: page-locked, not mapped, not portable.
    pub const DEFAULT: Self = Self(0x00);
    /// `cudaHostRegisterPortable`: memory is page-locked for all contexts.
    pub const PORTABLE: Self = Self(0x01);
    /// `cudaHostRegisterMapped`: maps the allocation into the device address space.
    pub const MAPPED: Self = Self(0x02);
    /// `cudaHostRegisterIoMemory`: the range is I/O memory.
    pub const IO_MEMORY: Self = Self(0x04);
    /// `cudaHostRegisterReadOnly`: the range is read-only to the device.
    pub const READ_ONLY: Self = Self(0x08);

    /// `true` if the [`Self::MAPPED`] bit is set.
    #[must_use]
    pub fn is_mapped(self) -> bool {
        self.0 & Self::MAPPED.0 != 0
    }
}

/// A registered host range and the device pointer it maps to (if mapped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HostRange {
    base: u64,
    size: usize,
    flags: HostRegisterFlags,
    device_ptr: DevicePtr,
}

impl HostRange {
    fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base + self.size as u64
    }
}

/// CPU model of the runtime's registered-host-memory table.
///
/// Models `cudaHostRegister` / `cudaHostUnregister` / `cudaHostGetDevicePointer`
/// and the validity rules around them (no overlapping double-register, mapped
/// ranges expose a stable device pointer, lookups resolve interior addresses).
#[derive(Debug, Default)]
pub struct HostMemoryRegistry {
    /// Registered ranges keyed by base host address.
    ranges: HashMap<u64, HostRange>,
    /// Monotonic source of synthetic device addresses for mapped ranges.
    next_device_addr: u64,
}

impl HostMemoryRegistry {
    /// Synthetic mapped-device-address base (non-zero so it is never NULL).
    const MAP_BASE: u64 = 0x7F00_0000_0000;

    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ranges: HashMap::new(),
            next_device_addr: Self::MAP_BASE,
        }
    }

    /// Register a host range (`cudaHostRegister`).
    ///
    /// A mapped registration assigns a stable device pointer returned later by
    /// [`Self::device_pointer`].
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::InvalidValue`] for a zero-size range or zero base.
    /// - [`CudaRtError::HostMemoryAlreadyRegistered`] if the range overlaps an
    ///   existing registration.
    pub fn register(
        &mut self,
        host_base: u64,
        size: usize,
        flags: HostRegisterFlags,
    ) -> CudaRtResult<()> {
        if host_base == 0 || size == 0 {
            return Err(CudaRtError::InvalidValue);
        }
        let new_end = host_base + size as u64;
        for r in self.ranges.values() {
            let existing_end = r.base + r.size as u64;
            // Overlap test for half-open intervals [base, end).
            if host_base < existing_end && r.base < new_end {
                return Err(CudaRtError::HostMemoryAlreadyRegistered);
            }
        }
        let device_ptr = if flags.is_mapped() {
            let addr = self.next_device_addr;
            self.next_device_addr = self.next_device_addr.saturating_add(size as u64);
            DevicePtr(addr)
        } else {
            DevicePtr::NULL
        };
        self.ranges.insert(
            host_base,
            HostRange {
                base: host_base,
                size,
                flags,
                device_ptr,
            },
        );
        Ok(())
    }

    /// Unregister a previously registered host range (`cudaHostUnregister`).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::HostMemoryNotRegistered`] if `host_base` is not a
    /// registered base address.
    pub fn unregister(&mut self, host_base: u64) -> CudaRtResult<()> {
        self.ranges
            .remove(&host_base)
            .map(|_| ())
            .ok_or(CudaRtError::HostMemoryNotRegistered)
    }

    /// Resolve the device pointer for a mapped host address
    /// (`cudaHostGetDevicePointer`).
    ///
    /// Interior addresses resolve to the corresponding interior device address
    /// (preserving the offset within the range).
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::HostMemoryNotRegistered`] if no registered range covers
    ///   `host_addr`.
    /// - [`CudaRtError::InvalidValue`] if the covering range was not registered
    ///   with [`HostRegisterFlags::MAPPED`].
    pub fn device_pointer(&self, host_addr: u64) -> CudaRtResult<DevicePtr> {
        let range = self
            .ranges
            .values()
            .find(|r| r.contains(host_addr))
            .ok_or(CudaRtError::HostMemoryNotRegistered)?;
        if !range.flags.is_mapped() {
            return Err(CudaRtError::InvalidValue);
        }
        let offset = host_addr - range.base;
        Ok(DevicePtr(range.device_ptr.0 + offset))
    }

    /// Flags a registered range was registered with.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::HostMemoryNotRegistered`] if `host_base` is not registered.
    pub fn flags(&self, host_base: u64) -> CudaRtResult<HostRegisterFlags> {
        self.ranges
            .get(&host_base)
            .map(|r| r.flags)
            .ok_or(CudaRtError::HostMemoryNotRegistered)
    }

    /// Number of currently-registered ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// `true` if no ranges are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

// ─── IPC handles ─────────────────────────────────────────────────────────────

/// An opaque IPC memory handle (mirrors `cudaIpcMemHandle_t`, modelled as a
/// 64-byte token).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcMemHandle(pub [u8; 64]);

/// An opaque IPC event handle (mirrors `cudaIpcEventHandle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcEventHandle(pub [u8; 64]);

/// Flags for [`IpcRegistry::open_mem_handle`] (mirrors `cudaIpcMemLazyEnablePeerAccess`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IpcMemFlags(pub u32);

impl IpcMemFlags {
    /// Lazily enable peer access when the handle is opened.
    pub const LAZY_ENABLE_PEER_ACCESS: Self = Self(0x01);
}

/// CPU model of the runtime's IPC handle table.
///
/// Models `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`
/// and `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` as a bookkeeping table
/// that round-trips a device allocation through an opaque handle and back to a
/// (process-local model of a) device pointer, with reference counting on opens.
#[derive(Debug, Default)]
pub struct IpcRegistry {
    /// Handle token → exported device pointer.
    mem_exports: HashMap<[u8; 64], DevicePtr>,
    /// Open mem handles → (mapped pointer, open refcount).
    mem_opens: HashMap<[u8; 64], (DevicePtr, u32)>,
    /// Event handle token → exported event raw token.
    event_exports: HashMap<[u8; 64], u64>,
    /// Monotonic counters for synthesising handle tokens / mapped pointers.
    next_handle_id: u64,
    next_open_addr: u64,
}

impl IpcRegistry {
    /// Base address for opened-handle device pointers in the model.
    const OPEN_BASE: u64 = 0x6000_0000_0000;

    /// Create an empty IPC registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mem_exports: HashMap::new(),
            mem_opens: HashMap::new(),
            event_exports: HashMap::new(),
            next_handle_id: 1,
            next_open_addr: Self::OPEN_BASE,
        }
    }

    /// Encode a `u64` id into a 64-byte handle token.
    fn token(id: u64) -> [u8; 64] {
        let mut t = [0u8; 64];
        t[..8].copy_from_slice(&id.to_le_bytes());
        t
    }

    /// Export a device allocation as an IPC handle (`cudaIpcGetMemHandle`).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidDevicePointer`] for a NULL pointer.
    pub fn get_mem_handle(&mut self, ptr: DevicePtr) -> CudaRtResult<IpcMemHandle> {
        if ptr.is_null() {
            return Err(CudaRtError::InvalidDevicePointer);
        }
        let id = self.next_handle_id;
        self.next_handle_id += 1;
        let token = Self::token(id);
        self.mem_exports.insert(token, ptr);
        Ok(IpcMemHandle(token))
    }

    /// Open an IPC memory handle in this (model) process (`cudaIpcOpenMemHandle`).
    ///
    /// Returns a device pointer valid in the importing context.  Re-opening the
    /// same handle bumps a refcount and returns the same pointer.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidResourceHandle`] if the handle was never exported.
    pub fn open_mem_handle(
        &mut self,
        handle: IpcMemHandle,
        _flags: IpcMemFlags,
    ) -> CudaRtResult<DevicePtr> {
        if !self.mem_exports.contains_key(&handle.0) {
            return Err(CudaRtError::InvalidResourceHandle);
        }
        if let Some((ptr, refs)) = self.mem_opens.get_mut(&handle.0) {
            *refs += 1;
            return Ok(*ptr);
        }
        let addr = self.next_open_addr;
        self.next_open_addr = self.next_open_addr.saturating_add(0x1000);
        let ptr = DevicePtr(addr);
        self.mem_opens.insert(handle.0, (ptr, 1));
        Ok(ptr)
    }

    /// Close an opened IPC memory handle (`cudaIpcCloseMemHandle`).
    ///
    /// Decrements the open refcount; the mapping is removed when it hits zero.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidDevicePointer`] if `ptr` is not an open mapping.
    pub fn close_mem_handle(&mut self, ptr: DevicePtr) -> CudaRtResult<()> {
        let key = self
            .mem_opens
            .iter()
            .find(|(_, (p, _))| *p == ptr)
            .map(|(k, _)| *k)
            .ok_or(CudaRtError::InvalidDevicePointer)?;
        if let Some((_, refs)) = self.mem_opens.get_mut(&key) {
            *refs -= 1;
            if *refs == 0 {
                self.mem_opens.remove(&key);
            }
        }
        Ok(())
    }

    /// Export an event as an IPC handle (`cudaIpcGetEventHandle`).
    ///
    /// `event_token` is the raw stable token of the event being exported.
    #[must_use]
    pub fn get_event_handle(&mut self, event_token: u64) -> IpcEventHandle {
        let id = self.next_handle_id;
        self.next_handle_id += 1;
        let token = Self::token(id);
        self.event_exports.insert(token, event_token);
        IpcEventHandle(token)
    }

    /// Open an IPC event handle (`cudaIpcOpenEventHandle`), returning the
    /// exported event's raw token.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidResourceHandle`] if the handle was never exported.
    pub fn open_event_handle(&self, handle: IpcEventHandle) -> CudaRtResult<u64> {
        self.event_exports
            .get(&handle.0)
            .copied()
            .ok_or(CudaRtError::InvalidResourceHandle)
    }

    /// Number of distinct currently-open memory mappings.
    #[must_use]
    pub fn open_mapping_count(&self) -> usize {
        self.mem_opens.len()
    }
}

// ─── Peer-access matrix ──────────────────────────────────────────────────────

/// CPU model of the runtime's peer-access state
/// (`cudaDeviceCanAccessPeer` / `cudaDeviceEnablePeerAccess` /
/// `cudaDeviceDisablePeerAccess`).
///
/// CUDA peer access is *directional*: enabling access from device `a` to `b`
/// does not implicitly enable `b → a`.  Enabling an already-enabled direction
/// and disabling a not-enabled direction are both errors, exactly as the
/// runtime reports them.  A `capable` predicate models the hardware
/// `cudaDeviceCanAccessPeer` topology (e.g. an NVLink/PCIe adjacency matrix).
#[derive(Debug)]
pub struct PeerAccessMatrix {
    device_count: u32,
    /// Enabled directed edges `(from, to)`.
    enabled: std::collections::HashSet<(u32, u32)>,
    /// Capability predicate: which directed pairs *can* be enabled.
    capable: Vec<(u32, u32)>,
}

impl PeerAccessMatrix {
    /// Create a matrix for `device_count` devices where every distinct pair is
    /// peer-capable (a fully-connected topology).
    #[must_use]
    pub fn fully_connected(device_count: u32) -> Self {
        let mut capable = Vec::new();
        for a in 0..device_count {
            for b in 0..device_count {
                if a != b {
                    capable.push((a, b));
                }
            }
        }
        Self {
            device_count,
            enabled: std::collections::HashSet::new(),
            capable,
        }
    }

    /// Create a matrix with an explicit set of peer-capable directed pairs.
    #[must_use]
    pub fn with_capable_pairs(device_count: u32, capable: &[(u32, u32)]) -> Self {
        Self {
            device_count,
            enabled: std::collections::HashSet::new(),
            capable: capable.to_vec(),
        }
    }

    fn valid_device(&self, d: u32) -> bool {
        d < self.device_count
    }

    /// Whether `from` *can* access `to` (`cudaDeviceCanAccessPeer`).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidDevice`] if either ordinal is out of range.
    pub fn can_access_peer(&self, from: u32, to: u32) -> CudaRtResult<bool> {
        if !self.valid_device(from) || !self.valid_device(to) {
            return Err(CudaRtError::InvalidDevice);
        }
        if from == to {
            // A device cannot be its own peer.
            return Ok(false);
        }
        Ok(self.capable.contains(&(from, to)))
    }

    /// Enable peer access `from → to` (`cudaDeviceEnablePeerAccess`).
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::InvalidDevice`] for an out-of-range ordinal.
    /// - [`CudaRtError::PeerAccessUnsupported`] if the pair is not peer-capable.
    /// - [`CudaRtError::PeerAccessAlreadyEnabled`] if already enabled.
    pub fn enable_peer_access(&mut self, from: u32, to: u32) -> CudaRtResult<()> {
        if !self.can_access_peer(from, to)? {
            return Err(CudaRtError::PeerAccessUnsupported);
        }
        if !self.enabled.insert((from, to)) {
            return Err(CudaRtError::PeerAccessAlreadyEnabled);
        }
        Ok(())
    }

    /// Disable peer access `from → to` (`cudaDeviceDisablePeerAccess`).
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::InvalidDevice`] for an out-of-range ordinal.
    /// - [`CudaRtError::PeerAccessNotEnabled`] if it was not enabled.
    pub fn disable_peer_access(&mut self, from: u32, to: u32) -> CudaRtResult<()> {
        if !self.valid_device(from) || !self.valid_device(to) {
            return Err(CudaRtError::InvalidDevice);
        }
        if !self.enabled.remove(&(from, to)) {
            return Err(CudaRtError::PeerAccessNotEnabled);
        }
        Ok(())
    }

    /// Whether peer access `from → to` is currently enabled.
    #[must_use]
    pub fn is_enabled(&self, from: u32, to: u32) -> bool {
        self.enabled.contains(&(from, to))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Host registration ---------------------------------------------------

    #[test]
    fn register_and_unregister_round_trip() {
        let mut reg = HostMemoryRegistry::new();
        reg.register(0x1000, 4096, HostRegisterFlags::DEFAULT)
            .expect("register");
        assert_eq!(reg.len(), 1);
        reg.unregister(0x1000).expect("unregister");
        assert!(reg.is_empty());
    }

    #[test]
    fn register_rejects_zero_and_overlap() {
        let mut reg = HostMemoryRegistry::new();
        assert_eq!(
            reg.register(0, 16, HostRegisterFlags::DEFAULT),
            Err(CudaRtError::InvalidValue)
        );
        assert_eq!(
            reg.register(0x1000, 0, HostRegisterFlags::DEFAULT),
            Err(CudaRtError::InvalidValue)
        );
        reg.register(0x1000, 4096, HostRegisterFlags::DEFAULT)
            .expect("first");
        // Overlapping the existing [0x1000, 0x2000) range.
        assert_eq!(
            reg.register(0x1800, 4096, HostRegisterFlags::DEFAULT),
            Err(CudaRtError::HostMemoryAlreadyRegistered)
        );
        // Adjacent, non-overlapping range is fine.
        reg.register(0x2000, 4096, HostRegisterFlags::DEFAULT)
            .expect("adjacent");
    }

    #[test]
    fn unregister_unknown_is_error() {
        let mut reg = HostMemoryRegistry::new();
        assert_eq!(
            reg.unregister(0xDEAD),
            Err(CudaRtError::HostMemoryNotRegistered)
        );
    }

    #[test]
    fn mapped_range_resolves_interior_device_pointer() {
        let mut reg = HostMemoryRegistry::new();
        reg.register(0x1_0000, 8192, HostRegisterFlags::MAPPED)
            .expect("register mapped");
        let base = reg.device_pointer(0x1_0000).expect("base devptr");
        assert!(!base.is_null());
        // An interior host address maps to base + same offset.
        let mid = reg.device_pointer(0x1_0000 + 256).expect("mid devptr");
        assert_eq!(mid.0, base.0 + 256);
    }

    #[test]
    fn unmapped_range_has_no_device_pointer() {
        let mut reg = HostMemoryRegistry::new();
        reg.register(0x2_0000, 4096, HostRegisterFlags::DEFAULT)
            .expect("register");
        assert_eq!(reg.device_pointer(0x2_0000), Err(CudaRtError::InvalidValue));
        // An address outside any range is "not registered".
        assert_eq!(
            reg.device_pointer(0x9_9999),
            Err(CudaRtError::HostMemoryNotRegistered)
        );
    }

    // --- IPC -----------------------------------------------------------------

    #[test]
    fn ipc_mem_handle_round_trips() {
        let mut ipc = IpcRegistry::new();
        let exported = DevicePtr(0x1234_5000);
        let handle = ipc.get_mem_handle(exported).expect("get handle");
        let opened = ipc
            .open_mem_handle(handle, IpcMemFlags::default())
            .expect("open");
        assert!(!opened.is_null());
        assert_eq!(ipc.open_mapping_count(), 1);
        ipc.close_mem_handle(opened).expect("close");
        assert_eq!(ipc.open_mapping_count(), 0);
    }

    #[test]
    fn ipc_get_handle_rejects_null() {
        let mut ipc = IpcRegistry::new();
        assert_eq!(
            ipc.get_mem_handle(DevicePtr::NULL),
            Err(CudaRtError::InvalidDevicePointer)
        );
    }

    #[test]
    fn ipc_open_unknown_handle_is_error() {
        let mut ipc = IpcRegistry::new();
        let bogus = IpcMemHandle([0xAB; 64]);
        assert_eq!(
            ipc.open_mem_handle(bogus, IpcMemFlags::default()),
            Err(CudaRtError::InvalidResourceHandle)
        );
    }

    #[test]
    fn ipc_open_is_refcounted() {
        let mut ipc = IpcRegistry::new();
        let handle = ipc.get_mem_handle(DevicePtr(0xABC000)).expect("get");
        let p1 = ipc
            .open_mem_handle(handle, IpcMemFlags::default())
            .expect("open1");
        let p2 = ipc
            .open_mem_handle(handle, IpcMemFlags::default())
            .expect("open2");
        assert_eq!(p1, p2);
        assert_eq!(ipc.open_mapping_count(), 1);
        // Two closes are needed to remove the mapping (refcount 2 → 0).
        ipc.close_mem_handle(p1).expect("close1");
        assert_eq!(ipc.open_mapping_count(), 1);
        ipc.close_mem_handle(p2).expect("close2");
        assert_eq!(ipc.open_mapping_count(), 0);
    }

    #[test]
    fn ipc_event_handle_round_trips() {
        let mut ipc = IpcRegistry::new();
        let handle = ipc.get_event_handle(0xEEFF);
        assert_eq!(ipc.open_event_handle(handle).expect("open"), 0xEEFF);
        let bogus = IpcEventHandle([0x00; 64]);
        assert_eq!(
            ipc.open_event_handle(bogus),
            Err(CudaRtError::InvalidResourceHandle)
        );
    }

    // --- Peer access ---------------------------------------------------------

    #[test]
    fn peer_self_is_never_accessible() {
        let m = PeerAccessMatrix::fully_connected(4);
        assert!(!m.can_access_peer(2, 2).expect("self"));
    }

    #[test]
    fn peer_out_of_range_is_invalid_device() {
        let m = PeerAccessMatrix::fully_connected(2);
        assert_eq!(m.can_access_peer(0, 5), Err(CudaRtError::InvalidDevice));
    }

    #[test]
    fn peer_enable_disable_is_directional() {
        let mut m = PeerAccessMatrix::fully_connected(3);
        m.enable_peer_access(0, 1).expect("enable 0->1");
        assert!(m.is_enabled(0, 1));
        // The reverse direction is independent and still disabled.
        assert!(!m.is_enabled(1, 0));
        // Double-enable is an error.
        assert_eq!(
            m.enable_peer_access(0, 1),
            Err(CudaRtError::PeerAccessAlreadyEnabled)
        );
        m.disable_peer_access(0, 1).expect("disable 0->1");
        assert!(!m.is_enabled(0, 1));
        // Disabling a not-enabled edge is an error.
        assert_eq!(
            m.disable_peer_access(0, 1),
            Err(CudaRtError::PeerAccessNotEnabled)
        );
    }

    #[test]
    fn peer_enable_requires_capability() {
        // Only 0->1 is peer-capable; 1->0 is not.
        let mut m = PeerAccessMatrix::with_capable_pairs(2, &[(0, 1)]);
        m.enable_peer_access(0, 1).expect("0->1 capable");
        assert_eq!(
            m.enable_peer_access(1, 0),
            Err(CudaRtError::PeerAccessUnsupported)
        );
    }
}
