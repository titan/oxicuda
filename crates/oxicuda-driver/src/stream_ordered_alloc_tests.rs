//! Unit tests for [`stream_ordered_alloc`](super).
//!
//! Split into its own file (referenced via `#[path]`) to keep
//! `stream_ordered_alloc.rs` under the 2000-line module limit while these
//! tests retain access to the parent module's private items via `super::*`.
use super::*;

/// Returns `true` when a real CUDA driver is loadable on this host.
///
/// `new` / `default_pool` on non-macOS platforms perform genuine driver
/// calls; without a driver they must fail with a clean typed error rather
/// than succeeding or panicking.  Tests that need a live driver pool gate
/// on this helper; CPU-model tests use [`StreamMemoryPool::cpu_pool`].
#[cfg(not(target_os = "macos"))]
fn driver_present() -> bool {
    crate::loader::try_driver().is_ok()
}

/// A CPU-model pool with default (unlimited) configuration.
fn cpu_pool() -> StreamMemoryPool {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    StreamMemoryPool::cpu_pool(config).expect("cpu_pool should always succeed")
}

// -- Config validation -------------------------------------------------

#[test]
fn config_validate_valid_sizes() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 1024,
        max_pool_size: 4096,
        release_threshold: 512,
        device: 0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_unlimited_max() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 1024 * 1024,
        max_pool_size: 0, // unlimited
        release_threshold: 512,
        device: 0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_initial_exceeds_max() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 8192,
        max_pool_size: 4096,
        release_threshold: 0,
        device: 0,
    };
    assert_eq!(config.validate(), Err(CudaError::InvalidValue));
}

#[test]
fn config_validate_negative_device() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: 0,
        device: -1,
    };
    assert_eq!(config.validate(), Err(CudaError::InvalidValue));
}

#[test]
fn config_validate_threshold_exceeds_max() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 1024,
        release_threshold: 2048,
        device: 0,
    };
    assert_eq!(config.validate(), Err(CudaError::InvalidValue));
}

// -- Default config ----------------------------------------------------

#[test]
fn default_config_for_device() {
    let config = StreamOrderedAllocConfig::default_for_device(2);
    assert_eq!(config.device, 2);
    assert_eq!(config.initial_pool_size, 0);
    assert_eq!(config.max_pool_size, 0);
    assert_eq!(config.release_threshold, 0);
    assert!(config.validate().is_ok());
}

// -- CPU-model pool creation -------------------------------------------

#[test]
fn cpu_pool_creation_succeeds_everywhere() {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    let pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool should succeed");
    assert_eq!(pool.device(), 0);
    assert_eq!(pool.stats().active_allocations, 0);
    assert_eq!(pool.stats().used_current, 0);
    assert_eq!(pool.handle(), 0);
}

#[test]
fn cpu_pool_creation_invalid_config() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: 0,
        device: -1,
    };
    assert!(matches!(
        StreamMemoryPool::cpu_pool(config),
        Err(CudaError::InvalidValue)
    ));
}

// -- Pool creation (driver-backed `new`) -------------------------------

/// On macOS, `new` creates a local-only model-backed pool (no driver).
#[cfg(target_os = "macos")]
#[test]
fn pool_creation() {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    let pool = StreamMemoryPool::new(config);
    assert!(pool.is_ok());
    if let Ok(p) = pool {
        assert_eq!(p.device(), 0);
        assert_eq!(p.stats().active_allocations, 0);
        assert_eq!(p.stats().used_current, 0);
    }
}

/// On non-macOS, `new` performs a real `cuMemPoolCreate`: it succeeds when
/// a driver is present and otherwise fails with a clean typed error.
#[cfg(not(target_os = "macos"))]
#[test]
fn pool_creation() {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    let pool = StreamMemoryPool::new(config);
    if driver_present() {
        // A live driver may still reject the pool (e.g. no device);
        // either way the result must be a typed Result, not a panic.
        if let Ok(p) = pool {
            assert_eq!(p.device(), 0);
            assert_eq!(p.stats().active_allocations, 0);
            assert_eq!(p.stats().used_current, 0);
        } else {
            assert!(matches!(
                pool,
                Err(CudaError::NotSupported)
                    | Err(CudaError::NoDevice)
                    | Err(CudaError::InvalidDevice)
                    | Err(CudaError::InvalidContext)
                    | Err(CudaError::NotInitialized)
            ));
        }
    } else {
        // No driver: must surface a clean NotInitialized error.
        assert_eq!(pool.err(), Some(CudaError::NotInitialized));
    }
}

#[test]
fn pool_creation_invalid_config() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: 0,
        device: -1,
    };
    let result = StreamMemoryPool::new(config);
    assert!(matches!(result, Err(CudaError::InvalidValue)));
}

// -- alloc_on / free_on round-trip (CPU model) -------------------------

#[test]
fn alloc_on_creates_allocation() {
    let mut pool = cpu_pool();
    let alloc = pool.alloc_on(1024, StreamOrderId::from(7)).expect("alloc");
    assert_eq!(alloc.size(), 1024);
    assert!(!alloc.is_freed());
    assert_ne!(alloc.as_ptr(), 0);
    assert_eq!(alloc.stream(), 7);
    assert_eq!(alloc.stream_id(), StreamOrderId::from(7));
}

#[test]
fn alloc_async_then_free_async_round_trips() {
    let mut pool = cpu_pool();
    let mut alloc = pool.alloc_async(2048, 3).expect("alloc");
    assert!(!alloc.is_freed());
    assert_eq!(pool.stats().active_allocations, 1);
    assert!(pool.free_async(&mut alloc).is_ok());
    assert!(alloc.is_freed());
    assert_eq!(pool.stats().active_allocations, 0);
}

#[test]
fn free_on_marks_freed() {
    let mut pool = cpu_pool();
    let s = StreamOrderId::from(11);
    let mut alloc = pool.alloc_on(2048, s).expect("alloc");
    assert!(!alloc.is_freed());
    assert!(pool.free_on(&mut alloc, s).is_ok());
    assert!(alloc.is_freed());
    assert_eq!(pool.stats().active_allocations, 0);
}

#[test]
fn double_free_returns_error() {
    let mut pool = cpu_pool();
    let s = StreamOrderId::from(1);
    let mut alloc = pool.alloc_on(512, s).expect("alloc");
    assert!(pool.free_on(&mut alloc, s).is_ok());
    assert_eq!(pool.free_on(&mut alloc, s), Err(CudaError::InvalidValue));
}

#[test]
fn free_of_foreign_pointer_returns_error() {
    // Two independent pools; an allocation from one is foreign to the other.
    let mut pool_a = cpu_pool();
    let mut pool_b = cpu_pool();
    let s = StreamOrderId::from(2);
    let mut alloc = pool_a.alloc_on(256, s).expect("alloc in A");
    // Freeing A's allocation against pool B must be rejected.
    assert_eq!(pool_b.free_on(&mut alloc, s), Err(CudaError::InvalidValue));
    // The allocation is still live in A and frees cleanly there.
    assert!(pool_a.free_on(&mut alloc, s).is_ok());
}

#[test]
fn alloc_zero_size_returns_error() {
    let mut pool = cpu_pool();
    assert!(matches!(
        pool.alloc_on(0, StreamOrderId::NULL),
        Err(CudaError::InvalidValue)
    ));
    assert!(matches!(
        pool.alloc_async(0, 0),
        Err(CudaError::InvalidValue)
    ));
}

// -- Pool reuse --------------------------------------------------------

/// A freed block, once its stream reaches the free point, is reused by a
/// subsequent same-size allocation: same pointer, no new reservation.
#[test]
fn freed_block_is_reused_by_same_size_alloc() {
    // Hold released bytes so reuse is observable via reserved bytes.
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(5);

    let a = pool.alloc_on(1024, s).expect("alloc a");
    let ptr_a = a.as_ptr();
    let reserved_after_first = pool.stats().reserved_current;

    let mut a = a;
    pool.free_on(&mut a, s).expect("free a");
    pool.synchronize_stream(s); // retire the stream-ordered free

    let b = pool.alloc_on(1024, s).expect("alloc b reuses block");
    assert_eq!(b.as_ptr(), ptr_a, "freed block must be reused (same ptr)");
    assert_eq!(
        pool.stats().reserved_current,
        reserved_after_first,
        "reuse must not grow reserved bytes"
    );
}

/// Reuse is only available after the freeing stream has reached the free
/// point; before that, the block is still pending and a new alloc must
/// carve a fresh block.
#[test]
fn block_not_reused_before_free_completes() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(6);

    let a = pool.alloc_on(1024, s).expect("alloc a");
    let ptr_a = a.as_ptr();
    // Submit a later op so the free does not sit at an already-reached head.
    let _b = pool.alloc_on(1024, s).expect("alloc b");

    let mut a = a;
    pool.free_on(&mut a, s).expect("free a");
    // The free is pending (stream not advanced): a new alloc cannot reuse it.
    let c = pool.alloc_on(1024, s).expect("alloc c");
    assert_ne!(
        c.as_ptr(),
        ptr_a,
        "pending (incomplete) free must not be reused"
    );
}

// -- Reserved vs used accounting ---------------------------------------

#[test]
fn reserved_and_used_accounting_consistent() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(4);

    let a = pool.alloc_on(1024, s).expect("a");
    let _b = pool.alloc_on(2048, s).expect("b");
    let total = 1024 + 2048;
    let stats = pool.stats();
    assert_eq!(stats.used_current, total);
    assert_eq!(stats.reserved_current, total);
    assert_eq!(stats.active_allocations, 2);

    let mut a = a;
    pool.free_on(&mut a, s).expect("free a");
    // used drops immediately; reserved holds until the free completes/trims.
    let stats = pool.stats();
    assert_eq!(stats.used_current, 2048);
    assert_eq!(stats.reserved_current, total);
    assert_eq!(stats.active_allocations, 1);

    pool.synchronize_stream(s);
    // Threshold keeps the freed block reserved and reuse-eligible.
    assert_eq!(pool.stats().reserved_current, total);
}

#[test]
fn stats_track_peaks() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(8);

    let a = pool.alloc_on(1024, s).expect("a");
    let _b = pool.alloc_on(2048, s).expect("b");
    let stats = pool.stats();
    assert_eq!(stats.active_allocations, 2);
    assert_eq!(stats.used_current, 3072);
    assert_eq!(stats.used_high, 3072);
    assert_eq!(stats.peak_allocations, 2);

    let mut a = a;
    pool.free_on(&mut a, s).expect("free a");
    let stats = pool.stats();
    assert_eq!(stats.active_allocations, 1);
    assert_eq!(stats.used_current, 2048);
    // Peak retained.
    assert_eq!(stats.used_high, 3072);
}

#[test]
fn reset_peak_stats_lowers_peaks() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(9);

    let a = pool.alloc_on(1024, s).expect("a");
    let _b = pool.alloc_on(2048, s).expect("b");
    assert_eq!(pool.stats().peak_allocations, 2);
    assert_eq!(pool.stats().used_high, 3072);

    let mut a = a;
    pool.free_on(&mut a, s).expect("free a");
    pool.synchronize_stream(s);
    pool.reset_peak_stats();

    let stats = pool.stats();
    assert_eq!(stats.used_high, stats.used_current);
    assert_eq!(stats.peak_allocations, 1);
}

// -- Cross-stream ordering ---------------------------------------------

/// An allocation made on stream A is only safe to use on its own stream
/// once that stream advances past the allocation point.
#[test]
fn same_stream_visibility() {
    let mut pool = cpu_pool();
    let s = StreamOrderId::from(10);
    let alloc = pool.alloc_on(64, s).expect("alloc");
    assert!(!pool.is_ready(&alloc), "not ready before stream advances");
    pool.synchronize_stream(s);
    assert!(pool.is_ready(&alloc), "ready after stream advances");
}

/// Cross-stream use is only safe once the consuming stream has waited on an
/// event recorded *after* the allocation point on the producing stream.
#[test]
fn cross_stream_ordering_requires_event_after_alloc() {
    let mut pool = cpu_pool();
    let producer = StreamOrderId::from(100);
    let consumer = StreamOrderId::from(200);
    let alloc = pool.alloc_on(128, producer).expect("alloc");

    // An event recorded before the alloc (seq 0) does not order the
    // consumer after the allocation: unsafe.
    assert!(!pool.is_ready_on(&alloc, consumer, 0));

    // An event recorded after the alloc captures the post-alloc position
    // and makes the consumer's use safe.
    let wait_seq = pool.record_event(producer);
    assert!(wait_seq > alloc.ready_seq());
    assert!(pool.is_ready_on(&alloc, consumer, wait_seq));
}

// -- Trim (CPU model) --------------------------------------------------

#[test]
fn model_trim_releases_free_blocks() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 0,
        release_threshold: usize::MAX,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    let s = StreamOrderId::from(12);

    let a = pool.alloc_on(4096, s).expect("a");
    let reserved_full = pool.stats().reserved_current;
    let mut a = a;
    pool.free_on(&mut a, s).expect("free");
    pool.synchronize_stream(s);
    assert_eq!(pool.stats().reserved_current, reserved_full);

    // Trim to zero releases the reclaimed free block.
    pool.model_trim(0);
    assert_eq!(pool.stats().reserved_current, 0);
}

// -- Max pool size -----------------------------------------------------

#[test]
fn alloc_exceeds_max_pool_size() {
    let config = StreamOrderedAllocConfig {
        initial_pool_size: 0,
        max_pool_size: 1024,
        release_threshold: 0,
        device: 0,
    };
    let mut pool = StreamMemoryPool::cpu_pool(config).expect("cpu_pool");
    assert!(pool.alloc_on(1024, StreamOrderId::NULL).is_ok());
    // A second fresh block would exceed the 1024-byte ceiling.
    assert!(matches!(
        pool.alloc_on(1, StreamOrderId::NULL),
        Err(CudaError::OutOfMemory)
    ));
}

#[test]
fn large_allocation_size() {
    let mut pool = cpu_pool();
    // 16 GiB allocation (large but valid in the virtual model).
    let size = 16 * 1024 * 1024 * 1024_usize;
    let alloc = pool.alloc_on(size, StreamOrderId::NULL).expect("alloc");
    assert_eq!(alloc.size(), size);
}

// -- Pool attribute setting --------------------------------------------

#[test]
fn set_attribute_release_threshold_cpu_pool() {
    let mut pool = cpu_pool();
    let result = pool.set_attribute(PoolAttribute::ReleaseThreshold(4096));
    // On a CPU-model pool (handle 0) the driver passthrough is a no-op on
    // macOS and `cuMemPoolSetAttribute` with a null handle elsewhere; the
    // config + model threshold are updated regardless of the passthrough.
    let _ = result;
    assert_eq!(pool.config().release_threshold, 4096);
}

#[test]
fn set_attribute_readonly_returns_error() {
    let mut pool = cpu_pool();
    assert_eq!(
        pool.set_attribute(PoolAttribute::ReservedMemCurrent),
        Err(CudaError::InvalidValue)
    );
    assert_eq!(
        pool.set_attribute(PoolAttribute::UsedMemCurrent),
        Err(CudaError::InvalidValue)
    );
    assert_eq!(
        pool.set_attribute(PoolAttribute::ReservedMemHigh),
        Err(CudaError::InvalidValue)
    );
    assert_eq!(
        pool.set_attribute(PoolAttribute::UsedMemHigh),
        Err(CudaError::InvalidValue)
    );
}

// -- StreamAllocation accessors ----------------------------------------

#[test]
fn allocation_accessors() {
    let mut pool = cpu_pool();
    let alloc = pool.alloc_on(4096, StreamOrderId::from(42)).expect("alloc");
    assert_eq!(alloc.size(), 4096);
    assert_eq!(alloc.stream(), 42);
    assert!(!alloc.is_freed());
    assert_ne!(alloc.as_ptr(), 0);
    assert_eq!(alloc.pool(), 0);
    // Debug formatting should not panic.
    let _debug = format!("{alloc:?}");
}

// -- Convenience stream_free -------------------------------------------

#[test]
fn convenience_stream_free_marks_and_rejects_double() {
    let mut pool = cpu_pool();
    let mut alloc = pool.alloc_on(128, StreamOrderId::from(1)).expect("alloc");
    assert!(stream_free(&mut alloc).is_ok());
    assert!(alloc.is_freed());
    // Double free via convenience function.
    assert_eq!(stream_free(&mut alloc), Err(CudaError::InvalidValue));
}

// -- Peer access (same-device pre-flight, CPU pool) --------------------

#[test]
fn peer_access_same_device_error() {
    let pool = cpu_pool();
    assert_eq!(pool.enable_peer_access(0), Err(CudaError::InvalidDevice));
    assert_eq!(pool.disable_peer_access(0), Err(CudaError::InvalidDevice));
}

// -- Default pool ------------------------------------------------------

/// On macOS, the default pool is a local-only model-backed pool.
#[cfg(target_os = "macos")]
#[test]
fn default_pool_valid_device() {
    let pool = StreamMemoryPool::default_pool(0);
    assert!(pool.is_ok());
}

/// On non-macOS, `default_pool` performs a real
/// `cuDeviceGetDefaultMemPool`: success with a driver, a clean typed
/// error without one — never a panic.
#[cfg(not(target_os = "macos"))]
#[test]
fn default_pool_valid_device() {
    let pool = StreamMemoryPool::default_pool(0);
    if driver_present() {
        if let Ok(p) = pool {
            assert_eq!(p.device(), 0);
        } else {
            assert!(matches!(
                pool,
                Err(CudaError::NotSupported)
                    | Err(CudaError::NoDevice)
                    | Err(CudaError::InvalidDevice)
                    | Err(CudaError::InvalidContext)
                    | Err(CudaError::NotInitialized)
            ));
        }
    } else {
        assert_eq!(pool.err(), Some(CudaError::NotInitialized));
    }
}

#[test]
fn default_pool_negative_device() {
    assert!(matches!(
        StreamMemoryPool::default_pool(-1),
        Err(CudaError::InvalidValue)
    ));
}

// -- PoolAttribute::to_raw ---------------------------------------------

#[test]
fn pool_attribute_to_raw() {
    assert_eq!(
        PoolAttribute::ReuseFollowEventDependencies.to_raw(),
        CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
    );
    assert_eq!(
        PoolAttribute::ReuseAllowOpportunistic.to_raw(),
        CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
    );
    assert_eq!(
        PoolAttribute::ReuseAllowInternalDependencies.to_raw(),
        CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
    );
    assert_eq!(
        PoolAttribute::ReleaseThreshold(0).to_raw(),
        CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
    );
    assert_eq!(
        PoolAttribute::ReservedMemCurrent.to_raw(),
        CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
    );
    assert_eq!(
        PoolAttribute::ReservedMemHigh.to_raw(),
        CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
    );
    assert_eq!(
        PoolAttribute::UsedMemCurrent.to_raw(),
        CU_MEMPOOL_ATTR_USED_MEM_CURRENT
    );
    assert_eq!(
        PoolAttribute::UsedMemHigh.to_raw(),
        CU_MEMPOOL_ATTR_USED_MEM_HIGH
    );
}

// -- ShareableHandleType default ---------------------------------------

#[test]
fn shareable_handle_type_default() {
    assert_eq!(ShareableHandleType::default(), ShareableHandleType::None);
}

// -- PoolExportDescriptor construction ---------------------------------

#[test]
fn pool_export_descriptor() {
    let desc = PoolExportDescriptor {
        shareable_handle_type: ShareableHandleType::PosixFileDescriptor,
        pool_device: 0,
    };
    assert_eq!(
        desc.shareable_handle_type,
        ShareableHandleType::PosixFileDescriptor
    );
    assert_eq!(desc.pool_device, 0);
}

// -- GPU driver bindings: real-FFI / absent-driver path ----------------
//
// These tests exercise the `gpu_*` bindings against whatever driver the
// host provides.
//
// * Without a driver, every binding must surface a clean typed error
//   (`NotInitialized`) — never a panic, never a fake `Ok`.
// * With a driver, the bindings reach the real CUDA FFI.  The driver
//   *dereferences* pool / device-pointer handles without validating
//   them, so a fabricated handle would segfault.  Handle-consuming
//   bindings are therefore only ever called with handles obtained from
//   a genuine `cuMemPoolCreate` / `cuMemAllocAsync`.

/// Create a real driver-backed pool, or `None` when the host cannot
/// (no driver, no device, or a graphless/poolless driver).
#[cfg(not(target_os = "macos"))]
fn make_real_pool() -> Option<StreamMemoryPool> {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    StreamMemoryPool::new(config).ok()
}

/// `gpu_create_pool` is deref-free: it builds `CUmemPoolProps` and calls
/// `cuMemPoolCreate`.  Without a driver it fails cleanly; with one it
/// returns a real handle or a typed driver error.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_create_pool_real_or_clean_error() {
    let config = StreamOrderedAllocConfig::default_for_device(0);
    let result = StreamMemoryPool::gpu_create_pool(&config);
    if !driver_present() {
        assert_eq!(result.err(), Some(CudaError::NotInitialized));
    } else {
        match result {
            Ok(handle) => assert_ne!(handle, 0, "a created pool has a non-null handle"),
            Err(e) => assert!(matches!(
                e,
                CudaError::NotSupported
                    | CudaError::NoDevice
                    | CudaError::InvalidDevice
                    | CudaError::InvalidContext
            )),
        }
    }
}

/// `gpu_default_pool` is deref-free: it only needs a device ordinal.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_default_pool_real_or_clean_error() {
    let result = StreamMemoryPool::gpu_default_pool(0);
    if !driver_present() {
        assert_eq!(result.err(), Some(CudaError::NotInitialized));
    } else {
        match result {
            Ok(handle) => assert_ne!(handle, 0, "the default pool has a non-null handle"),
            Err(e) => assert!(matches!(
                e,
                CudaError::NotSupported | CudaError::NoDevice | CudaError::InvalidDevice
            )),
        }
    }
}

/// `gpu_alloc_async` default-pool path (`handle == 0`) calls
/// `cuMemAllocAsync`, which dereferences no caller handle.  Without a
/// current context the driver returns `InvalidContext`; without a
/// driver, `NotInitialized`.  Either way: a clean typed error, never a
/// panic.  Any device pointer actually returned is freed immediately.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_alloc_async_default_pool_is_clean() {
    let result = StreamMemoryPool::gpu_alloc_async(0, 1024, 0);
    if !driver_present() {
        assert_eq!(result.err(), Some(CudaError::NotInitialized));
    } else {
        match result {
            Ok(ptr) => {
                // A live allocation: return it on the same null stream.
                assert_ne!(ptr, 0);
                let _ = StreamMemoryPool::gpu_free_async(ptr, 0);
            }
            Err(e) => assert!(matches!(
                e,
                CudaError::InvalidContext
                    | CudaError::NotSupported
                    | CudaError::NoDevice
                    | CudaError::InvalidDevice
                    | CudaError::OutOfMemory
            )),
        }
    }
}

/// `gpu_trim` on a *real* pool handle: trimming an empty pool succeeds.
/// Without a driver the binding fails cleanly before any dereference.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_trim_on_real_pool_or_clean_error() {
    if !driver_present() {
        // The function must fail cleanly; with no driver `try_driver`
        // returns the error before a handle is ever dereferenced.
        // (A fabricated handle is never passed to a live driver.)
        return;
    }
    let Some(pool) = make_real_pool() else {
        return; // driver present but no usable pool — nothing to trim.
    };
    // `cuMemPoolTrimTo` on a real, empty pool is a valid no-op.
    let result = StreamMemoryPool::gpu_trim(pool.handle(), 0);
    assert!(
        result.is_ok() || result.is_err(),
        "gpu_trim must return a typed Result, not panic"
    );
    if let Err(e) = result {
        assert!(matches!(e, CudaError::NotSupported));
    }
}

/// `gpu_set_attribute` on a *real* pool handle, for both the reuse-policy
/// (`int` value) and release-threshold (`cuuint64_t` value) branches.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_set_attribute_on_real_pool_or_clean_error() {
    if !driver_present() {
        return;
    }
    let Some(pool) = make_real_pool() else {
        return;
    };
    let reuse =
        StreamMemoryPool::gpu_set_attribute(pool.handle(), PoolAttribute::ReuseAllowOpportunistic);
    let threshold =
        StreamMemoryPool::gpu_set_attribute(pool.handle(), PoolAttribute::ReleaseThreshold(8192));
    // On a CUDA 11.2+ driver both branches succeed; an older driver
    // yields a clean `NotSupported`.
    for r in [reuse, threshold] {
        if let Err(e) = r {
            assert!(matches!(e, CudaError::NotSupported));
        }
    }
}

/// `gpu_enable_peer_access` / `gpu_disable_peer_access` on a *real* pool.
/// Granting access to a non-existent peer device is a typed driver error
/// (`InvalidDevice`), not a panic.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_peer_access_on_real_pool_or_clean_error() {
    if !driver_present() {
        return;
    }
    let Some(pool) = make_real_pool() else {
        return;
    };
    // Device 1 may or may not exist; either outcome must be a typed
    // Result.  `cuMemPoolSetAccess` dereferences only the real pool
    // handle, never a fabricated one.
    let enable = StreamMemoryPool::gpu_enable_peer_access(pool.handle(), 1);
    let disable = StreamMemoryPool::gpu_disable_peer_access(pool.handle(), 1);
    for r in [enable, disable] {
        if let Err(e) = r {
            assert!(matches!(
                e,
                CudaError::InvalidDevice | CudaError::InvalidValue | CudaError::NotSupported
            ));
        }
    }
}

/// The `gpu_*` bindings surface `NotInitialized` (never a panic) when no
/// driver is loadable.  This is a no-op assertion on a host *with* a
/// driver; the per-binding tests above cover the live-FFI behaviour.
#[cfg(not(target_os = "macos"))]
#[test]
fn gpu_bindings_clean_error_without_driver() {
    if driver_present() {
        return;
    }
    // No driver: every binding fails before touching a handle.
    let config = StreamOrderedAllocConfig::default_for_device(0);
    assert_eq!(
        StreamMemoryPool::gpu_create_pool(&config).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_default_pool(0).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_alloc_async(0, 1024, 0).err(),
        Some(CudaError::NotInitialized)
    );
    // Handle-consuming bindings also fail before any dereference.
    assert_eq!(
        StreamMemoryPool::gpu_alloc_async(0x1, 1024, 0).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_free_async(0x1, 0).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_trim(0x1, 0).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_set_attribute(0x1, PoolAttribute::ReleaseThreshold(1)).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_enable_peer_access(0x1, 1).err(),
        Some(CudaError::NotInitialized)
    );
    assert_eq!(
        StreamMemoryPool::gpu_disable_peer_access(0x1, 1).err(),
        Some(CudaError::NotInitialized)
    );
}

/// `map_pool_attribute` maps every variant to the matching driver enum.
#[cfg(not(target_os = "macos"))]
#[test]
fn map_pool_attribute_covers_all_variants() {
    use crate::ffi::CUmemPoolAttribute;
    let cases = [
        (
            PoolAttribute::ReuseFollowEventDependencies,
            CUmemPoolAttribute::ReuseFollowEventDependencies,
        ),
        (
            PoolAttribute::ReuseAllowOpportunistic,
            CUmemPoolAttribute::ReuseAllowOpportunistic,
        ),
        (
            PoolAttribute::ReuseAllowInternalDependencies,
            CUmemPoolAttribute::ReuseAllowInternalDependencies,
        ),
        (
            PoolAttribute::ReleaseThreshold(0),
            CUmemPoolAttribute::ReleaseThreshold,
        ),
        (
            PoolAttribute::ReservedMemCurrent,
            CUmemPoolAttribute::ReservedMemCurrent,
        ),
        (
            PoolAttribute::ReservedMemHigh,
            CUmemPoolAttribute::ReservedMemHigh,
        ),
        (
            PoolAttribute::UsedMemCurrent,
            CUmemPoolAttribute::UsedMemCurrent,
        ),
        (PoolAttribute::UsedMemHigh, CUmemPoolAttribute::UsedMemHigh),
    ];
    for (attr, expected) in cases {
        let mapped = StreamMemoryPool::map_pool_attribute(attr);
        assert_eq!(mapped, Ok(expected));
    }
}

/// `stream_alloc` convenience: a clean typed error without a driver, and
/// a typed `Result` (allocation, or a context/driver error) with one.
#[cfg(not(target_os = "macos"))]
#[test]
fn convenience_stream_alloc_real_or_clean_error() {
    let result = stream_alloc(256, 0);
    if !driver_present() {
        assert_eq!(result.err(), Some(CudaError::NotInitialized));
    } else {
        match result {
            Ok(mut alloc) => {
                assert_eq!(alloc.size(), 256);
                let _ = stream_free(&mut alloc);
            }
            Err(e) => assert!(matches!(
                e,
                CudaError::InvalidContext
                    | CudaError::NotSupported
                    | CudaError::NoDevice
                    | CudaError::InvalidDevice
                    | CudaError::OutOfMemory
            )),
        }
    }
}
