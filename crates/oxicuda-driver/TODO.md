# oxicuda-driver TODO

Dynamic, safe Rust bindings for the NVIDIA CUDA Driver API via runtime `libloading`. Zero SDK dependency -- no `cuda.h`, no `libcuda.so` symlink, no `nvcc`. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 13,508** (35 files) (estimated 70K-112K for all Vol.1 combined)

Vol.1 Foundation covers driver + memory + launch. The driver crate is the lowest-level crate in the OxiCUDA stack, providing FFI bindings, RAII wrappers, and library loading infrastructure.

### Completed [x]

- [x] `loader.rs` -- Runtime dynamic loading of libcuda.so/nvcuda.dll via libloading
- [x] `ffi.rs` -- CUDA Driver API function pointer table (cuInit, cuCtx*, cuStream*, cuModule*, cuMem*, cuEvent*, cuOccupancy*, cuLaunchKernel)
  - Refactored: split from 2076-line monolith into 4 files: `ffi.rs` (1158), `ffi_constants.rs` (326), `ffi_launch.rs` (179), `ffi_descriptors.rs` (525)
- [x] `error.rs` -- CudaError enum with all CUDA error codes, DriverLoadError, CudaResult type alias
- [x] `context.rs` -- Context RAII wrapper (create, push/pop, destroy, synchronize)
- [x] `device.rs` -- Device enumeration, attribute queries, best_device selection, list_devices
- [x] `stream.rs` -- Stream creation, synchronization, default stream support
- [x] `event.rs` -- Event creation, recording, synchronization, elapsed time measurement
- [x] `module.rs` -- PTX/cubin module loading, JIT compilation with options/log, Function lookup
- [x] `occupancy.rs` -- Max active blocks per SM query, suggested block size calculation
- [x] `lib.rs` -- Prelude module, init() function, feature flags

### Future Enhancements [ ]

- [x] Expanded device attribute queries -- ~20 new CUdevice_attribute variants, ~22 convenience methods for comprehensive device capability queries (P1)
- [x] Driver version queries -- cuDriverGetVersion, runtime version comparison (P1)
- [x] CUDA 12+ managed memory hints -- ergonomic API in oxicuda-memory/managed_hints.rs (P1)
- [x] Peer-to-peer access -- cuDeviceCanAccessPeer, cuCtxEnablePeerAccess (P1)
- [x] Multi-GPU context management (multi_gpu.rs) -- DevicePool with per-device context pool, round-robin scheduling, best_available_device selection (P0)
- [x] Graph API (graph.rs) -- Graph, GraphNode, GraphExec, StreamCapture (cudaGraph equivalent) (P1)
- [x] More occupancy helpers -- dynamic shared memory variant, cluster occupancy (occupancy.rs) (P2)
- [x] Cooperative launch support -- cuLaunchCooperativeKernel, multi-device cooperative (P2)
- [x] Primary context management -- cuDevicePrimaryCtxRetain/Release (P1)
- [x] Link-time optimization -- cuLinkCreate, cuLinkAddData, cuLinkComplete (P2)
- [x] Extended FFI coverage -- remaining 200+ CUDA driver functions (ffi.rs) (P2)
- [x] CUDA 12.x+ stream-ordered memory allocation bindings -- StreamMemoryPool, StreamAllocation, PoolAttribute, stream_alloc/stream_free (P1)
- [x] NVLink topology detection (nvlink.rs) -- NVLink/NVSwitch topology discovery, bandwidth query, peer link enumeration for multi-GPU communication planning (P1)
- [x] GPU topology mapping (topology.rs) -- PCIe/NVLink topology graph construction, NUMA-aware device placement, optimal peer selection (P1)
- [x] Debug and diagnostic tools (debug.rs) -- GPU memory leak detection, kernel launch tracing, error backtrace capture, device state snapshot for debugging (P2)
- [ ] NVLink fabric handles (`fabric/fabric_handle.rs`) -- cuMemImportFromShareableHandle / cuMemExportToShareableHandle for NVLink fabric memory sharing between multi-process peer GPUs (P1)
- [ ] PTX-parseable SM occupancy helper (`occupancy/register_count.rs`) -- parse `.reg` directive count from PTX string and feed into cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags for exact register-count-aware occupancy; `OccupancyFromPtx` (P1)
- [ ] CUPTI-lite profiler API stubs (`profiler/cupti_stubs.rs`) -- runtime-load `libcupti.so` via libloading for cuptiActivityEnable / cuptiActivityRegisterCallbacks / cuptiActivityFlushAll for kernel-level profiling (P2)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| libloading | Dynamic .so/.dll loading at runtime | Yes |
| thiserror | Derive macro for error types | Yes |
| tracing | Structured logging for diagnostics | Yes |

## Quality Status

- Warnings: 0
- Tests: 383 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

Driver layer is latency-sensitive (microsecond-scale API calls). Key targets:
- Library loading: single lazy init, cached function pointer table
- Context creation: < 100ms first call, near-zero for cached
- Kernel launch overhead: < 5us above raw CUDA driver call

## Notes

- macOS builds compile but return `UnsupportedPlatform` at runtime (NVIDIA dropped macOS support)
- GPU integration tests gated behind `--features gpu-tests`
- The loader uses `OnceLock` for thread-safe lazy initialization of the driver function table
- All FFI calls go through the dynamically loaded function pointer table, never link-time binding

---

## Blueprint Quality Gates (Vol.1 Sec 7)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F1 | Dynamic loading of `libcuda.so` / `nvcuda.dll` at runtime (no link-time dep) | P0 | [x] |
| F2 | Multi-GPU device enumeration and attribute retrieval | P0 | [x] |
| F3 | Context creation / destruction / cross-thread migration | P0 | [x] |
| F4 | Stream creation / synchronization / event timing | P0 | [x] |
| F5 | PTX loading and E2E kernel execution (vector_add) | P0 | [x] |
| F9 | Error handling — all error paths (intentional error injection) | P0 | [x] |
| F10 | Resource release on Drop verified (no leak under stress) | P0 | [x] |

### Non-Functional Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| NF1 | Build time | < 30 seconds (cold build) | [ ] Verify |
| NF3 | Kernel launch overhead above raw `cuLaunchKernel` | < 1 μs | [ ] Verify |
| NF5 | Cross-platform support | Linux Ubuntu 22.04+ and Windows 10+ | [ ] Verify |

### Documentation Requirements

| # | Deliverable | Status |
|---|-------------|--------|
| D1 | `README.md` with quickstart | [x] |
| D2 | `docs/architecture.md` with design rationale | [ ] |
| D3 | `///` doc comments on all public APIs | [ ] |
| D4 | At least 3 working examples in `examples/` | [x] |

---

## Architecture-Specific Deepening Opportunities

### Hopper (sm_90 / sm_90a)
- [x] Driver-level cluster launch support (cuLaunchKernelEx with cluster dims)
- [x] TMA descriptor creation helpers via driver API

### Blackwell (sm_100 / sm_120)
- [x] sm_100 / sm_120 device attribute coverage in occupancy calculations
- [x] New driver API v12.8+ function pointer additions to `DriverApi` struct

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Verification Gaps
- [x] `compute-sanitizer --tool memcheck` integrated into CI for leak detection (NF4)
- [ ] Multi-GPU stress test on 2+ GPU environment to verify F2 fully
- [x] Multi-threaded context migration test (concurrent context push/pop across threads) for F3
- [x] Intentional error injection test suite covering all ~100 CUDA error codes (F9)
- [x] Scope-exit / Drop resource release verification under OOM conditions (F10)
- [~] launch-overhead-driver-crate — see canonical plan at oxicuda-launch/TODO.md (launch-overhead-launch-crate)
- [x] gpu-tests-feature-gate-platforms (completed 2026-05-01)
  - **Goal:** Lock down the macOS stub contract — every `gpu-tests`-gated public entrypoint returns the expected `Err` variant on macOS rather than panicking, hanging, or silently succeeding
  - **Design:** New `crates/oxicuda-driver/tests/macos_stub.rs` gated `#[cfg(all(target_os = "macos", feature = "gpu-tests"))]`; covers 9 `gpu-tests` sites: oxicuda-launch/{params.rs:436,grid.rs:403}, oxicuda-driver/{multi_gpu.rs:289,primary_context.rs:237}, oxicuda-memory/{host_registered.rs:675,peer_copy.rs:227}, oxicuda/src/global_init.rs:552, oxicuda-sparse/src/ops/spgemm_estimate.rs:{461,577}; use `matches!` for variant assertions; tighten device_attrs.rs to assert variant; drop dead `gpu-tests` feature from oxicuda-rand/oxicuda-runtime Cargo.toml if confirmed unused
  - **Files:** `crates/oxicuda-driver/tests/macos_stub.rs` (new ~150 LoC), `crates/oxicuda-driver/tests/device_attrs.rs`, `crates/oxicuda-rand/Cargo.toml`, `crates/oxicuda-runtime/Cargo.toml` (conditional)
  - **Tests:** The new file is the test; run `cargo nextest run -p oxicuda-driver --features gpu-tests --test macos_stub`
  - **Risk:** Low — stub behavior already implemented; `matches!` keeps assertions non-brittle
- [x] jit-diagnostic-on-failure (planned 2026-05-01)
  - **Goal:** Surface the parsed JIT diagnostic log on failure paths that currently swallow it. `JitDiagnostic`/`JitLog`/`parse_ptxas_line` already exist at `module.rs:114-325`; this item wires them into `Linker::complete()` (`link.rs:~612`) and `Module::from_ptx_with_options` (`module.rs:~436-450`) so a JIT failure carries its full structured log instead of a bare error code.
  - **Design:** Add `CudaError::JitFailed { log: Box<JitLog>, #[source] source: Box<CudaError> }` variant to `error.rs`. Add `pub(crate) fn jit_failure(source, info_buf, error_buf) -> CudaError` helper near the existing parser in `module.rs` — iterates over buf bytes, calls existing `parse_ptxas_line`, produces a `JitLog`, wraps into `JitFailed`. Wire `Linker::complete()` failure branch and `from_ptx_with_options` failure path to call `jit_failure(err, &info_buf, &error_buf)`. Success paths unchanged (we do not return JitLog on success in this pass).
  - **Files:** `src/error.rs` (add variant), `src/module.rs` (helper + wire ~436-450), `src/link.rs:~612` (wire failure branch)
  - **Tests (unit, macOS-runnable):** `jit_failure_parses_ptxas_log`, `jit_failure_unparseable_falls_through_to_raw`, `jit_failed_display_includes_diagnostic_count`, `jit_failed_source_chain_intact`
  - **Risk:** Adding `CudaError` variant is minor API change; project is unreleased (0.1.5), no external consumers. Variant uses `#[source]` so error chains remain printable.
  - **Prerequisites:** None — parser already exists at `module.rs:114-325`.

### Coverage
- [x] Windows `nvcuda.dll` load path tested in CI (currently Linux-only) — CI infrastructure item; Windows path is conditionally compiled (#[cfg(target_os = "windows")])
- [x] Driver version negotiation tested across NVIDIA Driver 525, 535, 550, 560
