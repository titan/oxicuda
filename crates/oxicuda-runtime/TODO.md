# oxicuda-runtime TODO

Pure-Rust implementation of the **CUDA Runtime API** (`libcudart`) surface, built on top of `oxicuda-driver`'s dynamic driver loader. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Zero CUDA SDK build-time dependency; only the NVIDIA driver (`libcuda.so` / `nvcuda.dll`) is required at run time.

## Implementation Status

**Actual: ~4,856 SLoC across 14 files** (production; ~7,540 incl. tests)

Covers the day-to-day surface of `cudart`: device enumeration, memory management, streams, events, kernel launch, peer-to-peer access, profiler control, and the full texture / surface object family. The crate is a thin ergonomic façade over `oxicuda-driver` -- strong Rust types for streams, events, device pointers, kernel dimensions, with `Result`-typed errors and no unwrap in production code.

In addition, four **GPU-free CPU-model** modules implement the runtime semantics that are fully testable on the host (launch/occupancy arithmetic, the stream-ordered memory pool, graph capture/construction, and host-register/IPC/peer bookkeeping). These run and self-verify on every platform with no NVIDIA driver present.

### Completed [x]

#### Top-level entry points
- [x] `lib.rs` (182 SLoC) -- module wiring, flat-namespace convenience functions (`get_device_count`, `set_device`, `get_device`, `device_synchronize`, `cuda_malloc`, `cuda_free`, `cuda_memset`, `memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d`), plus 6 doc / unit tests that exercise the flat API without requiring a GPU

#### Error types
- [x] `error.rs` (744 SLoC) -- `CudaRtError` enum mirroring `cudaError_t` (NotInitialized, InvalidValue, OutOfMemory, InvalidDevicePointer, InvalidMemcpyDirection, NoDevice, InvalidDevice, ECCUncorrectable, IllegalAddress, LaunchTimeout, LaunchOutOfResources, ContextIsDestroyed, DriverNotAvailable, ...); `from_code(c_int) -> Option<Self>`; `CudaRtResult<T>` alias; 7 unit tests for code round-trips and Display

#### Device API (`device.rs`, 397 SLoC)
- [x] `cudaGetDeviceCount` -- `get_device_count() -> CudaRtResult<u32>`
- [x] `cudaSetDevice` / `cudaGetDevice` -- thread-local current-device tracking
- [x] `cudaGetDeviceProperties` -- `CudaDeviceProp` struct (name, total_global_mem, sm_count, compute_capability_major/minor, warp_size, max_threads_per_block, max_block_dim, max_grid_dim, shared_mem_per_block, regs_per_block, ...)
- [x] `cudaDeviceSynchronize`, `cudaDeviceReset`
- [x] 3 unit tests (count query, ordinal validation, properties shape)

#### Memory API (`memory.rs`)
- [x] `DevicePtr(u64)` newtype with `NULL`, `is_null`, `offset(isize)` arithmetic, `cast::<T>()` reinterpret, `as_raw_ptr::<T>()`, `as_typed_slice_meta::<T>(len)` (host-side typed-length descriptor with overflow checks)
- [x] `MemLocation { Host, Device }` + `MemcpyKind::resolve` / `src_is_device` / `dst_is_device` -- unified-addressing direction classification model
- [x] `cudaMalloc` / `cudaFree` -- generic device-side allocations
- [x] `cudaMallocHost` / `cudaFreeHost` -- pinned host memory
- [x] `cudaMallocManaged` -- unified memory
- [x] `cudaMallocPitch` -- 2-D pitched allocations with row-pitch return
- [x] `cudaMemcpy` / `cudaMemcpyAsync` -- with `MemcpyKind { HostToHost, HostToDevice, DeviceToHost, DeviceToDevice, Default }`
- [x] Typed helpers `memcpy_h2d<T: Copy>(dst, &[T])`, `memcpy_d2h<T: Copy>(&mut [T], src)`, `memcpy_d2d(dst, src, bytes)` -- no raw pointers required
- [x] `cudaMemset` / `cudaMemsetAsync`
- [x] `cudaMemGetInfo` -- free + total device bytes
- [x] 13 unit tests covering null-pointer handling, pointer arithmetic, async/zero-size edge cases, cast round-trip, typed-length + overflow rejection, offset round-trip, and the 5×5 `MemcpyKind` direction matrix

#### Stream API (`stream.rs`, 285 SLoC)
- [x] `CudaStream` newtype + `StreamFlags { DEFAULT, NON_BLOCKING }` bit-flags
- [x] `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`
- [x] `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`
- [x] `cudaStreamWaitEvent` -- cross-stream synchronization
- [x] `cudaStreamGetPriority`, `cudaStreamGetFlags`
- [x] `StreamIdAllocator` -- GPU-free monotonic stream-id bookkeeping (create/destroy/live_count/peek_next_id, double-free rejection)
- [x] 7 unit tests for flag constants, default stream, creation/destruction smoke, id-allocator start/double-free, and the 10,000-stream create/destroy stress (monotonic ids, no collisions, clean teardown)

#### Event API (`event.rs`, 245 SLoC)
- [x] `CudaEvent` newtype + `EventFlags { DEFAULT, DISABLE_TIMING, BLOCKING_SYNC, INTERPROCESS }`
- [x] `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`
- [x] `cudaEventRecord` -- attach an event to a stream
- [x] `cudaEventSynchronize` -- block until the event completes
- [x] `cudaEventQuery` -- non-blocking ready check
- [x] `cudaEventElapsedTime` -- milliseconds between two recorded events
- [x] `EventIdAllocator` -- GPU-free monotonic event-id bookkeeping (create/destroy/live_count/peek_next_id, double-free rejection)
- [x] 7 unit tests for flag constants, creation/destruction smoke, id-allocator start/double-free, the 10,000-event create/destroy stress, and a 10,000-event retain-then-teardown variant

#### Kernel launch API (`launch.rs`, 354 SLoC)
- [x] `Dim3 { x, y, z }` -- `one_d`, `two_d`, `three_d`, `volume()`
- [x] `CudaFunction = CUfunction` and `CudaModule = CUmodule` type aliases
- [x] `cudaLaunchKernel` -- explicit-handle launch with grid, block, shared-mem, stream, packed parameter buffer
- [x] `cudaFuncGetAttributes` -- `FuncAttributes { max_threads_per_block, shared_size_bytes, const_size_bytes, local_size_bytes, num_regs, ptx_version, binary_version, cache_mode_ca, max_dynamic_shared_size_bytes, preferred_shared_carveout }`
- [x] `cudaFuncSetAttribute` -- `FuncAttribute` enum for runtime attribute mutation
- [x] `module_load_ptx` / `module_get_function` / `module_unload` -- PTX module lifecycle bridge to `oxicuda-driver`
- [x] 5 unit tests (Dim3 helpers, packed-param round-trip, attribute defaults)

#### Peer access API (`peer.rs`, 181 SLoC)
- [x] `cudaDeviceCanAccessPeer` -- `device_can_access_peer(device, peer) -> CudaRtResult<bool>`
- [x] `cudaDeviceEnablePeerAccess` (auto-retains the peer's primary context)
- [x] `cudaDeviceDisablePeerAccess`
- [x] `cudaMemcpyPeer` / `cudaMemcpyPeerAsync` -- explicit cross-device byte copies
- [x] 1 unit test (peer-self detection)

#### Profiler API (`profiler.rs`, 108 SLoC)
- [x] `cudaProfilerStart` / `cudaProfilerStop` -- start/stop the external profiler collection window
- [x] `ProfilerGuard` -- RAII helper that stops the profiler on drop, so callers can scope a section with `let _g = ProfilerGuard::start()?;`
- [x] 2 unit tests (manual start/stop sequence, guard scoping)

#### Texture & surface objects (`texture.rs`, 1,004 SLoC)
- [x] `ArrayFormat` -- UnsignedInt8/16/32, SignedInt8/16/32, Half, Float; `as_cu_format`, `bytes_per_channel`
- [x] `AddressMode` -- Wrap, Clamp, Mirror, Border
- [x] `FilterMode` -- Point, Linear
- [x] `CudaArray` (1-D / 2-D) -- `cudaMallocArray`, `cudaFreeArray`, `cudaArrayGetInfo`
- [x] `CudaArray3D` -- `cudaMalloc3DArray` with `Array3DFlags { LAYERED, SURFACE_LDST, CUBEMAP, TEXTURE_GATHER }`
- [x] Host ↔ array copies -- `cudaMemcpyToArray`, `cudaMemcpyFromArray`, `cudaMemcpyToArrayAsync`, `cudaMemcpyFromArrayAsync`
- [x] `ResourceDesc` -- Linear / Pitched / Array / MipmappedArray resource discriminators
- [x] `TextureDesc` -- `address_modes[3]`, `filter_mode`, `read_as_normalized_int`, `normalized_coords`, `srgb`, `max_anisotropy`, mipmap params (now derives `PartialEq`)
- [x] `TextureDescBuilder` -- fluent builder seeded from `default_2d` (`address_mode`/`address_modes`/`filter_mode`/`normalized_coords`/`read_as_integer`/`srgb`/`max_anisotropy`/`mipmap_filter`/`mipmap_levels`/`border_color`/`build`)
- [x] `ResourceViewDesc` -- format override and dimension specification
- [x] `CudaTextureObject` (bindless) -- `cudaCreateTextureObject` / `cudaDestroyTextureObject` / `cudaGetTextureObjectResourceDesc`
- [x] `CudaSurfaceObject` (bindless) -- `cudaCreateSurfaceObject` / `cudaDestroySurfaceObject`
- [x] 13 unit tests covering format byte width, address mode round-trip, descriptor builders, null-safety on destroy, and `TextureDescBuilder` (defaults match `default_2d`, builder vs manual full-custom equality, `address_mode` sets all axes)

#### Launch-config & occupancy CPU model (`launch_config.rs`, ~600 SLoC)
- [x] `LaunchConfig` + `DeviceLaunchLimits` (`from_prop` / `for_compute_capability` Turing→Blackwell) -- grid/block/shared/launch-bound validation
- [x] `OccupancyCalculator` -- `active_blocks_per_sm` (warp/register/shared/block-cap minimum with allocation granularity), `max_potential_block_size`, cooperative-grid sizing; `Occupancy` + `OccupancyLimiter` attribution
- [x] 15 unit tests with hand-computed expected occupancies (warp/register/shared/block-cap-limited, cooperative validation)

#### Stream-ordered memory pool CPU model (`mem_pool.rs`, ~560 SLoC)
- [x] `MemPool` (`cudaMallocAsync`/`cudaFreeAsync`) -- immediate-return alloc, stream-clock pending-free retirement, best-fit reuse, granularity rounding
- [x] `MemPoolAttr` / `MemPoolAttributes` (`cudaMemPool*` attribute table), release-threshold trim, `trim_to`, `MemPoolStats` high-water marks
- [x] 12 unit tests (right-size reuse, best-fit, cross-stream ordering, threshold trim, attribute round-trip)

#### Graph capture CPU model (`graph_capture.rs`, ~560 SLoC)
- [x] `CudaGraph` (`cudaGraphAdd*Node`, dependencies, child graphs) + Kahn topological sort with cycle rejection
- [x] `CudaGraphExec` (`cudaGraphInstantiate`/`cudaGraphExecUpdate`) -- precomputed exec order, topology-match update enforcement, clone
- [x] `StreamCapture` (`cudaStreamBeginCapture`/`EndCapture`) -- idle/active/invalidated state machine, in-order chaining, per-stream `EventFlags` tracking (`begin_with_flags`/`event_flags`), `capture_info() -> (CaptureStatus, EventFlags)` (`cudaStreamGetCaptureInfo`), `end_in_place` non-consuming end
- [x] 18 unit tests (linear/diamond topo order, cycle rejection, capture chain, exec-update accept/reject, capture-info status+flags / end→None / invalidated-keeps-flags / default-flags)

#### Host-mem / IPC / peer CPU model (`host_mem.rs`, ~640 SLoC)
- [x] `HostMemoryRegistry` (`cudaHostRegister`/`Unregister`/`HostGetDevicePointer`) -- overlap detection, mapped interior-offset resolution
- [x] `IpcRegistry` (`cudaIpc*MemHandle`/`cudaIpc*EventHandle`) -- handle round-trip with open refcounting
- [x] `PeerAccessMatrix` (`cudaDeviceCanAccessPeer`/`Enable`/`DisablePeerAccess`) -- directional enable/disable, capability predicate
- [x] 13 unit tests (register/overlap, mapped lookup, IPC refcount, directional peer enable)

### Future Enhancements

> **CPU-model status (2026-06-21).** The genuinely-missing *CPU-modelable runtime
> semantics* of this crate have now been implemented as deterministic, GPU-free
> models with full unit-test coverage. The driver FFI passthrough (`memory.rs`,
> `stream.rs`, `event.rs`, `launch.rs`, ...) still requires a GPU at run time and
> stays device-gated. New CPU-model modules:
>
> - `launch_config.rs` -- launch-config validation + occupancy calculator
> - `mem_pool.rs` -- `cudaMallocAsync`/`cudaFreeAsync`/`cudaMemPool*` CPU model
> - `graph_capture.rs` -- stream-capture state machine + `cudaGraph_t` builder
> - `host_mem.rs` -- host-register / IPC-handle / peer-access bookkeeping tables

#### P0 -- Surface completeness
- [x] `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` -- CPU bookkeeping model in `host_mem.rs` (`IpcRegistry`: export/open/close with refcounting + round-trip). Real cross-process page sharing **(requires GPU hardware)**.
- [x] `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` -- CPU bookkeeping model in `host_mem.rs` (`IpcRegistry::get_event_handle`/`open_event_handle`). Real cross-process event **(requires GPU hardware)**.
- [x] `cudaHostRegister` / `cudaHostUnregister` -- CPU bookkeeping model in `host_mem.rs` (`HostMemoryRegistry`: overlap/double-register detection, range table). Real page-locking **(requires GPU hardware)**.
- [x] `cudaHostGetDevicePointer` -- CPU model in `host_mem.rs` (`HostMemoryRegistry::device_pointer`, interior-offset resolution, MAPPED-flag enforcement). Real mapped address **(requires GPU hardware)**.
- [ ] `cudaStreamAttachMemAsync` -- attach managed memory to a stream **(requires GPU hardware; managed-memory migration is a device/driver operation)**

#### P0 -- Graph capture wiring
- [x] `cudaStreamBeginCapture` / `cudaStreamEndCapture` -- CPU state machine in `graph_capture.rs` (`StreamCapture`: idle/active/invalidated transitions, in-order single-stream node chaining, default-stream-capture rejection)
- [x] `cudaGraphInstantiate` / `cudaGraphExecUpdate` / clone -- CPU model in `graph_capture.rs` (`CudaGraph::instantiate` → `CudaGraphExec` with precomputed topo order; `clone_graph`; `update` with topology-match enforcement). `cudaGraphLaunch` device execution **(requires GPU hardware)**.
- [x] `cudaGraphAddKernelNode` / `cudaGraphAddMemcpyNode` / `cudaGraphAddMemsetNode` -- CPU model in `graph_capture.rs` (`CudaGraph::add_*_node`, `add_empty_node`, `add_child_graph_node`, `add_dependency`, Kahn topological sort + cycle rejection)
- [x] These models are runtime-surface (`cudaGraph_t`) types; the sibling `oxicuda-driver` `graph.rs` provides the lower driver-side `Graph`/`GraphExec`/`StreamCapture` for the eventual device binding

#### P1 -- Event polish
- [x] `EventFlags::BLOCKING_SYNC` (0x1) and `EventFlags::INTERPROCESS` (0x4) constants -- `event.rs` (BLOCKING_SYNC added 2026-06-21; INTERPROCESS already present)
- [x] `cudaEventRecordWithFlags` -- already present as `event::event_record_with_flags` (`event.rs:145`)

#### P1 -- Async memory pool
- [x] `cudaMallocAsync` / `cudaFreeAsync` -- stream-ordered alloc CPU model in `mem_pool.rs` (`MemPool::malloc_async`/`free_async`: immediate-return alloc, pending-free queue, stream-clock retirement, best-fit reuse). Device-backed allocation **(requires GPU hardware)**.
- [x] `cudaMemPoolCreate` / `cudaMemPoolDestroy` (+ attribute control) -- CPU model in `mem_pool.rs` (`MemPool::new`/`with_attributes`, `set_attribute`/`get_attribute`, `MemPoolAttr` family, release-threshold trim, usage stats)
- [ ] `cudaDeviceGetDefaultMemPool` / `cudaDeviceSetMemPool` -- per-device default-pool binding **(requires GPU hardware; device-context state)**. The pool *object* it would return is modelled by `mem_pool.rs`.

#### P1 -- Occupancy & launch validation (CPU-modelable, NEW)
- [x] `cudaOccupancyMaxActiveBlocksPerMultiprocessor` -- `launch_config.rs` (`OccupancyCalculator::active_blocks_per_sm`: warp/register/shared/block-cap min with allocation granularity + limiter attribution)
- [x] `cudaOccupancyMaxPotentialBlockSize` -- `launch_config.rs` (`max_potential_block_size`, warp-step sweep + shared-mem callback)
- [x] Launch-config validation -- `launch_config.rs` (`LaunchConfig::validate`: grid/block/shared/launch-bound vs `DeviceLaunchLimits`, InvalidConfiguration vs LaunchOutOfResources)
- [x] Cooperative-launch grid sizing -- `launch_config.rs` (`max_cooperative_grid_size`, `validate_cooperative_grid`)
- [x] Peer-access matrix model -- `host_mem.rs` (`PeerAccessMatrix`: directional enable/disable, capability predicate, can-access query)

#### P2 -- Quality of life
- [ ] Drop the unused `gpu-tests` feature flag from `Cargo.toml` -- **still referenced** in compiled code (`device.rs:356` gates a `#[cfg(not(feature = "gpu-tests"))]` test), so it is **not** safe to remove
- [x] Builder pattern for `TextureDesc` -- `texture.rs` (`TextureDescBuilder`: `new`/`address_mode`/`address_modes`/`filter_mode`/`normalized_coords`/`read_as_integer`/`srgb`/`max_anisotropy`/`mipmap_filter`/`mipmap_levels`/`border_color`/`build`, seeded from `default_2d`; `TextureDesc` now derives `PartialEq` for verification; 3 unit tests: defaults-match-default_2d, builder-vs-manual full custom, address_mode-sets-all-axes; +1 doctest)
- [x] Convenience `DevicePtr::cast::<T>()` and typed-length helper -- `memory.rs` (`DevicePtr::cast::<T>()` address-preserving reinterpret, `as_raw_ptr::<T>()` FFI hand-off, `as_typed_slice_meta::<T>(len) -> (addr, byte_len)` with `len*size_of::<T>()` and address-space overflow checks; host-side pointer arithmetic only, never dereferences device memory; 4 unit tests incl. count-overflow and address-overflow rejection)
- [ ] Async variants `memcpy_h2d_async<T>`, `memcpy_d2h_async<T>` mirroring the existing typed helpers **(requires GPU hardware for the underlying async transfer)**

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading FFI, function pointer table) | Yes (runtime FFI only) |
| thiserror | `#[derive(Error)]` for `CudaRtError` | Yes |
| criterion (dev) | Benchmark harness for `benches/runtime_ops.rs` | Yes |

## Quality Status

- Warnings: 0
- Tests: 121 unit tests + 3 doc-tests across 16 modules (all CPU-side, no GPU required for the unit suite)
- unwrap() calls: 0 (production code)
- clippy: clean (pedantic + nursery; `--all-features --all-targets -D warnings`)
- Benchmark harness: `benches/runtime_ops.rs` via criterion

## Performance Targets

The runtime layer is a thin ergonomic shim over `oxicuda-driver` -- there is no measurable compute work in this crate. The relevant latency targets are:

| Operation | Target | Notes |
|-----------|--------|-------|
| `set_device(0)` (warm) | < 1 μs above driver call | thread-local cache |
| `cuda_malloc(N)` (uncached) | bounded by driver `cuMemAlloc` | no extra allocation in the wrapper |
| `memcpy_h2d::<T>(slice)` | < 100 ns above raw `cuMemcpyHtoD` | slice-length validation only |
| Stream/event create+destroy | < 5 μs per pair | direct FFI |

## Notes

- macOS builds compile but every device-touching call returns `CudaRtError::DriverNotAvailable` at runtime (NVIDIA dropped macOS support; the driver loader returns `Err(DriverLoadError)`)
- GPU integration tests for this crate live in higher-level crates (`oxicuda-blas`, `oxicuda-dnn`); the `cudart` surface is exercised transitively
- All FFI calls go through `oxicuda_driver::loader::try_driver()` -- no link-time `libcudart.so` symbol resolution
- `DevicePtr` is a `u64` newtype, not a raw pointer, so it crosses thread boundaries (`Send + Sync`) without unsafe

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89)
- [ ] Surface `cudaStreamSetAttribute` for `cudaStreamAttributeAccessPolicyWindow` (L2 cache residency hints introduced in CUDA 11)
- [ ] `cudaStreamCreateWithPriority` priority range query (`cudaDeviceGetStreamPriorityRange`) -- the constructor is wired but the range query is not

### Hopper (sm_90) / Blackwell (sm_100)
- [ ] `cudaLaunchKernelEx` -- the extended launch entry point that accepts cluster dimensions and launch attributes; the underlying driver call is in `oxicuda-driver`
- [ ] `cudaDeviceGetTexture1DLinearMaxWidth` / mipmap query helpers for sm_90+ texture limits

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the gap between the present runtime surface and what NVIDIA's `libcudart` exposes.

### Test Coverage Gaps
- [x] Round-trip property test: any `DevicePtr` value `p`, `p.offset(d).offset(-d) == p` for `d` not overflowing `i64` -- `memory.rs` (`device_ptr_offset_round_trip`: representative ptrs × deltas, skips forward-overflow combos via `checked_add` guard)
- [x] `MemcpyKind` exhaustive matrix (5 × 5) -- `memory.rs` (`memcpy_kind_direction_matrix_5x5` walks all 25 cells; new `MemLocation` enum + `MemcpyKind::resolve`/`src_is_device`/`dst_is_device` model unified-addressing resolution of `Default`; plus a direct 4-way resolve truth table)
- [ ] GPU-gated suite that allocates, copies, launches a no-op PTX kernel, records an event, and frees -- end-to-end smoke for every public function **(requires GPU hardware)**
- [x] Stress test that creates / destroys 10,000 streams + events -- `stream.rs` (`StreamIdAllocator` + `stream_stress_create_destroy_10k_no_collision`) and `event.rs` (`EventIdAllocator` + `event_stress_create_destroy_10k_no_collision`, `event_stress_retain_then_teardown_10k`): pure host-side id bookkeeping, strictly-monotonic ids, zero collisions, double-free rejection, clean teardown -- no device. (Driver-side handle leak detection still **requires GPU hardware**.)

### Implementation Deepening
- [ ] Bench `cudaLaunchKernel` overhead vs raw `cuLaunchKernel` and document the headroom (target: < 100 ns wrapper cost) **(requires GPU hardware)**
- [x] Wire the `cudaStream` family to track per-stream `EventFlags` and surface `cudaStreamGetCaptureInfo` -- `graph_capture.rs` (`StreamCapture` now carries an `event_flags: EventFlags` field; `begin_with_flags` records it, `event_flags()` getter, `capture_info() -> (CaptureStatus, EventFlags)` mirroring `cudaStreamGetCaptureInfo`; `end_in_place` keeps the handle observable so post-end None status is testable; 3 unit tests: active+flags / end→None+DEFAULT / invalidated-keeps-flags, plus default-flags case)
- [ ] Add doc-tests showing each public function used in isolation -- today the doc-tests are concentrated in `lib.rs`'s flat-API "Quick start" (a `TextureDescBuilder` doctest was added)
