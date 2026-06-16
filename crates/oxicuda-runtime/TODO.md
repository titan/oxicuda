# oxicuda-runtime TODO

Pure-Rust implementation of the **CUDA Runtime API** (`libcudart`) surface, built on top of `oxicuda-driver`'s dynamic driver loader. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Zero CUDA SDK build-time dependency; only the NVIDIA driver (`libcuda.so` / `nvcuda.dll`) is required at run time.

## Implementation Status

**Actual: 2,521 SLoC across 12 files**

Covers the day-to-day surface of `cudart`: device enumeration, memory management, streams, events, kernel launch, peer-to-peer access, profiler control, and the full texture / surface object family. The crate is a thin ergonomic façade over `oxicuda-driver` -- strong Rust types for streams, events, device pointers, kernel dimensions, with `Result`-typed errors and no unwrap in production code.

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

#### Memory API (`memory.rs`, 505 SLoC)
- [x] `DevicePtr(u64)` newtype with `NULL`, `is_null`, `offset(isize)` arithmetic
- [x] `cudaMalloc` / `cudaFree` -- generic device-side allocations
- [x] `cudaMallocHost` / `cudaFreeHost` -- pinned host memory
- [x] `cudaMallocManaged` -- unified memory
- [x] `cudaMallocPitch` -- 2-D pitched allocations with row-pitch return
- [x] `cudaMemcpy` / `cudaMemcpyAsync` -- with `MemcpyKind { HostToHost, HostToDevice, DeviceToHost, DeviceToDevice, Default }`
- [x] Typed helpers `memcpy_h2d<T: Copy>(dst, &[T])`, `memcpy_d2h<T: Copy>(&mut [T], src)`, `memcpy_d2d(dst, src, bytes)` -- no raw pointers required
- [x] `cudaMemset` / `cudaMemsetAsync`
- [x] `cudaMemGetInfo` -- free + total device bytes
- [x] 6 unit tests covering null-pointer handling, pointer arithmetic, async/zero-size edge cases

#### Stream API (`stream.rs`, 285 SLoC)
- [x] `CudaStream` newtype + `StreamFlags { DEFAULT, NON_BLOCKING }` bit-flags
- [x] `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`
- [x] `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`
- [x] `cudaStreamWaitEvent` -- cross-stream synchronization
- [x] `cudaStreamGetPriority`, `cudaStreamGetFlags`
- [x] 4 unit tests for flag constants, default stream, creation/destruction smoke

#### Event API (`event.rs`, 245 SLoC)
- [x] `CudaEvent` newtype + `EventFlags { DEFAULT, DISABLE_TIMING, BLOCKING_SYNC, INTERPROCESS }`
- [x] `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`
- [x] `cudaEventRecord` -- attach an event to a stream
- [x] `cudaEventSynchronize` -- block until the event completes
- [x] `cudaEventQuery` -- non-blocking ready check
- [x] `cudaEventElapsedTime` -- milliseconds between two recorded events
- [x] 3 unit tests for flag constants and creation/destruction smoke

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
- [x] `TextureDesc` -- `address_modes[3]`, `filter_mode`, `read_as_normalized_int`, `normalized_coords`, `srgb`, `max_anisotropy`, mipmap params
- [x] `ResourceViewDesc` -- format override and dimension specification
- [x] `CudaTextureObject` (bindless) -- `cudaCreateTextureObject` / `cudaDestroyTextureObject` / `cudaGetTextureObjectResourceDesc`
- [x] `CudaSurfaceObject` (bindless) -- `cudaCreateSurfaceObject` / `cudaDestroySurfaceObject`
- [x] 10 unit tests covering format byte width, address mode round-trip, descriptor builders, and null-safety on destroy

### Future Enhancements [ ]

#### P0 -- Surface completeness
- [ ] `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` -- inter-process device-memory sharing handles
- [ ] `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` -- inter-process event sharing
- [ ] `cudaHostRegister` / `cudaHostUnregister` -- pin existing host allocations (currently only `cudaMallocHost` is wired)
- [ ] `cudaHostGetDevicePointer` -- mapped-memory device-side address for a registered host buffer
- [ ] `cudaStreamAttachMemAsync` -- attach managed memory to a stream

#### P0 -- Graph capture wiring
- [ ] `cudaStreamBeginCapture` / `cudaStreamEndCapture` -- record stream operations as a `cudaGraph_t`
- [ ] `cudaGraphInstantiate` / `cudaGraphLaunch` / `cudaGraphExecDestroy` -- materialize and launch captured graphs
- [ ] `cudaGraphAddKernelNode` / `cudaGraphAddMemcpyNode` / `cudaGraphAddMemsetNode` -- programmatic graph construction
- [ ] These bind on top of `oxicuda-driver`'s `graph.rs` which already exposes the driver-side `Graph`, `GraphNode`, `GraphExec`, `StreamCapture` types

#### P1 -- Event polish
- [ ] Expose `EventFlags::BLOCKING_SYNC` and `EventFlags::INTERPROCESS` constants symmetric with `cudaEventBlockingSync` / `cudaEventInterprocess`
- [ ] `cudaEventRecordWithFlags` -- the CUDA 11+ variant that lets callers pass `cudaEventRecordDefault` / `cudaEventRecordExternal`

#### P1 -- Async memory pool
- [ ] `cudaMallocAsync` / `cudaFreeAsync` -- stream-ordered allocation (CUDA 11.2+) backed by `oxicuda-driver`'s `StreamMemoryPool`
- [ ] `cudaMemPoolCreate` / `cudaMemPoolDestroy` -- custom memory pools with attribute control
- [ ] `cudaDeviceGetDefaultMemPool` / `cudaDeviceSetMemPool`

#### P2 -- Quality of life
- [ ] Drop the unused `gpu-tests` feature flag from `Cargo.toml` (see driver TODO "gpu-tests-feature-gate-platforms" item) if confirmed unused at audit time
- [ ] Builder pattern for `TextureDesc` -- there are 8+ optional fields and the current struct-literal style is verbose
- [ ] Convenience `DevicePtr::cast::<T>()` and `as_typed_slice::<T>(len)` helpers for callers that want a typed view of a device allocation
- [ ] Async variants `memcpy_h2d_async<T>`, `memcpy_d2h_async<T>` mirroring the existing typed helpers

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading FFI, function pointer table) | Yes (runtime FFI only) |
| thiserror | `#[derive(Error)]` for `CudaRtError` | Yes |
| criterion (dev) | Benchmark harness for `benches/runtime_ops.rs` | Yes |

## Quality Status

- Warnings: 0
- Tests: 46 unit tests across 12 modules (all CPU-side, no GPU required for the unit suite)
- unwrap() calls: 0 (production code)
- clippy: clean (pedantic + nursery)
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
- [ ] Round-trip property test: any `DevicePtr` value `p`, `p.offset(d).offset(-d) == p` for `d` not overflowing `i64`
- [ ] `MemcpyKind` exhaustive matrix (5 × 5) -- assert that `Default` is correctly resolved at runtime via unified addressing
- [ ] GPU-gated suite that allocates, copies, launches a no-op PTX kernel, records an event, and frees -- end-to-end smoke for every public function
- [ ] Stress test that creates / destroys 10,000 streams + events to surface any leak in the driver-side handle tracking

### Implementation Deepening
- [ ] Bench `cudaLaunchKernel` overhead vs raw `cuLaunchKernel` and document the headroom (target: < 100 ns wrapper cost)
- [ ] Wire the `cudaStream` family to actually track per-stream `EventFlags` and surface `cudaStreamGetCaptureInfo` (needed for graph capture support)
- [ ] Add doc-tests showing each public function used in isolation -- today the doc-tests are concentrated in `lib.rs`'s flat-API "Quick start"
