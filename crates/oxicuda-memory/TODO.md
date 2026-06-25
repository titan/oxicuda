# oxicuda-memory TODO

Type-safe GPU memory management with Rust ownership semantics. RAII-based wrappers around CUDA memory allocation and transfer operations. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 5,297** across **18 files** (estimated 70K-112K for all Vol.1 combined)

The memory crate provides the core buffer types and copy operations that all higher-level OxiCUDA crates depend on. Current implementation covers the essential buffer types with stubs for advanced features.

### Completed [x]

- [x] `device_buffer.rs` -- DeviceBuffer<T> (VRAM allocation, RAII Drop), DeviceSlice<T> (borrowed sub-range)
- [x] `host_buffer.rs` -- PinnedBuffer<T> (page-locked host memory for fast DMA transfers)
- [x] `unified.rs` -- UnifiedBuffer<T> (managed memory accessible from both host and device)
- [x] `zero_copy.rs` -- MappedBuffer<T> (zero-copy host-mapped memory)
- [x] `copy.rs` -- H2D/D2H/D2D copy helpers (copy_htod, copy_dtoh, copy_dtod) with type safety
- [x] `pool.rs` -- MemoryPool and PooledBuffer (stream-ordered allocation, feature-gated under `pool`)
- [x] `lib.rs` -- Prelude module, feature flags (pool, gpu-tests)

### Future Enhancements [ ]

- [x] Async memory pool enhancements (pool.rs) -- PoolStats (allocated/peak/count), trim(), set_threshold() (P0)
- [x] Memory pool statistics -- AllocationHistogram, FragmentationMetrics, PoolReport, PoolStatsTracker (pool_stats.rs) (P1)
- [x] CUDA 12+ managed memory hints -- ManagedMemoryHints, MigrationPolicy, PrefetchPlan (managed_hints.rs) (P1)
- [x] Virtual memory management (virtual_memory.rs) -- VirtualAddressRange, PhysicalAllocation, VirtualMemoryManager (P1)
- [x] Memory advice / prefetch hints -- cuMemAdvise, cuMemPrefetchAsync for unified memory (P1)
- [x] Multi-GPU peer copy (peer_copy.rs) -- can_access_peer, enable/disable_peer_access, copy_peer, copy_peer_async (P0)
- [x] Memory bandwidth profiling hooks -- transfer timing, throughput measurement (P2)
- [x] Async copy operations (copy.rs) -- copy_htod_async_raw, copy_dtoh_async_raw, copy_dtod_async with stream ordering (P0)
- [x] 2D/3D memory copy (copy_2d3d.rs) -- Memcpy2DParams, Memcpy3DParams, copy_2d/3d functions (P1)
- [x] Memory alignment guarantees -- 256-byte / 512-byte aligned allocation options (P2)
- [x] Buffer views and reinterpret cast -- type-safe buffer reinterpretation (P1)
- [x] Host-registered memory -- cuMemHostRegister for existing host allocations (P2)
- [x] Memory usage query -- cuMemGetInfo (free/total VRAM) (P1)
- [x] Memory compression hints (`compression/compressed_buffer.rs`) -- cuMemCreate with CU_MEM_PROPERTY_COMPRESSION flag for hardware-accelerated lossless memory compression on Ampere+; `CompressedDeviceBuffer` (P1) -- CPU-modelable bookkeeping implemented: `CompressionType`, `CompressionSupport` (CC>=8.0 gate, 2 MiB granularity), `CompressionPlan` (granularity-aligned reservation), `CompressedDeviceBuffer` (logical/physical footprint, compression-ratio + effective-bandwidth model). The `cuMemCreate` device call remains GPU-gated.
- [x] NUMA-aware host allocation (`numa/numa_buffer.rs`) -- numa_alloc_onnode / libnuma integration for host-pinned memory physically allocated on the NUMA node closest to the target GPU; `NumaBuffer` (P1) -- CPU-modelable topology + policy implemented: `NumaTopology` (ACPI SLIT distance matrix), `closest_node_to_gpu`, `NumaBuffer` (node-bound footprint + access-distance), `NumaAllocTracker` (per-node byte accounting / least-loaded balancing). The `numa_alloc_onnode`/`cuMemHostRegister` calls remain platform-gated.
- [x] Memory pressure monitoring (`pool_pressure.rs`, exposes `MemoryPressureMonitor`) -- poll cuMemGetInfo periodically + configurable OOM threshold callback + eviction hook for proactive pool trim under memory pressure; `MemoryPressureMonitor` (P2) -- CPU-modelable control loop implemented: `PressureLevel` (Nominal/Warning/Critical), warning/critical used-fraction thresholds, `observe()` state machine with transition + eviction hooks (fires once on entering Critical, accumulates reclaimed bytes), `PressureSample` escalation tracking. `poll()` queries live `memory_info()` and is GPU-gated; `observe()` is fully unit-tested on synthetic samples. (Placed at top level rather than `pool/` since the `pool` module is a single feature-gated file; this module is always available.)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API bindings | Yes |
| tracing | Warning logs in Drop impls (instead of panic) | Yes |

## Quality Status

- Warnings: 0
- Tests: 256 unit + 20 doc passing (all-features); 250 unit + 20 doc (default features)
- unwrap() calls: 0
- Drop implementations log errors via tracing::warn, never panic

## Performance Targets

Memory operations are bandwidth-bound. Key targets:
- H2D/D2H copy: approach PCIe bandwidth limit (15-30 GB/s depending on gen)
- D2D copy: approach device memory bandwidth (e.g. 2 TB/s on H100)
- Pinned buffer allocation: avoid kernel round-trip overhead via page-locking
- Pool allocation: sub-microsecond from pre-allocated pool

## Notes

- MappedBuffer is implemented via `cuMemAllocHost_v2` + `cuMemHostGetDevicePointer_v2`
- MemoryPool is feature-gated under `pool` and currently uses an in-process reuse pool backed by `cuMemAlloc_v2`/`cuMemFree_v2`
- All buffer types implement Drop for automatic deallocation
- Size mismatches and zero-length allocations return CudaError::InvalidValue

---

## Blueprint Quality Gates (Vol.1 Sec 7)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F6 | `DeviceBuffer` alloc / free / copy — memory leak test under stress | P0 | [x] |
| F7 | `PinnedBuffer` async copy with stream verified correct | P0 | [x] |
| F8 | `MemoryPool` `alloc_async` / `free_async` continuous benchmark | P0 | [x] |

### Non-Functional Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| NF2 | H2D / D2H copy bandwidth | ≥ 95% of PCIe theoretical bandwidth (same as `cuMemcpy`) | [~] Verify |
| NF4 | Memory leak detection | Zero leaks via `compute-sanitizer --tool memcheck` in CI | [ ] Verify |

---

## Numerical Accuracy / Correctness Requirements

| Operation | Requirement |
|-----------|-------------|
| `copy_from_host` → `copy_to_host` round-trip | Bitwise identical to source data for all `T: Copy` types |
| Async copy with stream sync | Same bitwise correctness, verified after `stream.synchronize()` |
| Unified memory access | CPU and GPU reads return same values after `memset`/init |

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Test Coverage Gaps
- [x] Unit test suite expansion (currently very few unit tests; rely on doc-tests)
- [x] H2D / D2H bandwidth benchmark added to `benches/` to verify ≥ 95% PCIe (NF2)
- [x] `MemoryPool` stress test: 10K concurrent alloc_async / free_async cycles without fragmentation
- [x] Memory leak detection CI job using `compute-sanitizer --tool memcheck` (NF4)
- [ ] Peer copy correctness test on 2+ GPU systems (copy D0→D1, verify D1 matches)

### Implementation Deepening
- [x] `DeviceBuffer::alloc_async` / `free_async` with `cuMemAllocAsync` / `cuMemFreeAsync` fully exercised (requires CUDA 11.2+ driver) — CPU-side API verified; requires driver for actual execution
- [x] Pool trim / `cuMemPoolTrimTo` to release unused pool memory to system
- [x] `MemoryPool` per-stream allocation tracking for debugging

---

## Performance Verification Harness Status (2026-04-26)

- **NF2** (H2D / D2H bandwidth): harness implemented at `benches/bandwidth_copy.rs` with five criterion groups (`h2d_pageable`, `h2d_pinned`, `d2h_pageable`, `d2h_pinned`, `d2d`) sweeping 4 KiB → 256 MiB, each annotated with `Throughput::Bytes(...)`. The bench skips on macOS / no-GPU (logs `skip:` to stderr) and on Linux + NVIDIA emits a `report_nf2` line comparing the measured peak vs. PCIe Gen3 / Gen4 / Gen5 × 16 theoretical bandwidth. Default reference is **PCIe Gen4 ×16** (overridable via `OXI_PCIE_GEN={3|4|5}`). Awaiting Linux + NVIDIA verification run to confirm the ≥ 95 % gate.
