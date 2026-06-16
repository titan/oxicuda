# oxicuda-webgpu TODO

Cross-platform WebGPU compute backend via wgpu, providing GPU-accelerated operations
through WGSL shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~3,948 across 8 files
- **Tests:** 129 passing
- **Status:** Full memory + compute, WGSL generators, WASM target, FP16 path
- **Targets:** Native (Vulkan / Metal / D3D12 / GL via wgpu) + browser (WebGPU API)

### Completed

#### Core Infrastructure
- [x] `lib.rs` -- module wiring, re-exports `WebGpuBackend`, `WebGpuError`, `WebGpuResult`; conditional `wasm` module on `wasm32` or `wasm` feature
- [x] `backend.rs` -- `WebGpuBackend` implementing `ComputeBackend`; dispatch for GEMM, unary/binary/reduction, conv2d, attention, batched GEMM, FP16 GEMM (~50.1K)
- [x] `device.rs` -- `WebGpuDevice` over `wgpu::Instance` + `Adapter` + `Device` + `Queue`; adapter selection by power preference
- [x] `error.rs` -- `WebGpuError` (InitFailed, AdapterRequestFailed, DeviceRequestFailed, ShaderCompilation, AllocationError) with thiserror
- [x] `memory.rs` -- `WebGpuMemoryManager` over `wgpu::Buffer` with `STORAGE | COPY_SRC | COPY_DST` usage; u64 handle -> Arc<Buffer> map

#### WGSL Shader Generators (`shader.rs`, ~30.5K)
- [x] `gemm_wgsl(tile_size)` -- tiled GEMM with `@compute @workgroup_size(ts, ts)` and `GemmParams` uniform at `@binding(3)`
- [x] `gemm_wgsl_f16(tile_size)` -- FP16 GEMM gated behind WGSL `enable f16` directive (requires `WebGPUShaderF16` extension)
- [x] `batched_gemm_wgsl(tile_size)` -- 3D dispatch with `BatchedGemmParams` carrying strides and `batch_count`
- [x] `elementwise_wgsl` -- relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] `binary_wgsl` -- add, sub, mul, div, max, min, pow
- [x] `reduction_wgsl` + `reduction_final_wgsl` -- two-pass workgroup reduction (partial sums then scalar finalize)
- [x] Conv2D WGSL NCHW compute shader + CPU fallback
- [x] Attention WGSL scaled dot-product + stable softmax + causal masking

#### WASM Target (`wasm.rs`, ~15.8K, gated on `wasm32` or `wasm` feature)
- [x] `WasmGpuDevice` -- async construction via `wgpu::Instance::request_adapter` from browser `navigator.gpu`
- [x] `WasmBackend` -- delegates to `WebGpuBackend` with `init_from_canvas`-style async constructors
- [x] `WasmMemoryManager` -- async-friendly buffer staging suited to browser event loop
- [x] HashMap-backed pool with `AtomicU64` handle allocator
- [x] `wasm` feature flag allows native testing of WASM code paths
- [x] Re-exports `BinaryOp`, `BackendTranspose`, `ReduceOp`, `UnaryOp` from `oxicuda-backend`

#### Backend Tests (`backend_tests.rs`, ~31.7K)
- [x] Comprehensive integration tests for `WebGpuBackend` -- dispatch, FP16, batched, edge cases

### Future Enhancements

#### P0 -- Critical
- [ ] WGSL subgroup operations -- `subgroupBroadcast`, `subgroupAdd`, `subgroupMax` (stabilizing in Chrome 125+, Firefox 135+) for warp-style reductions
- [ ] Indirect dispatch (`wgpu::ComputePass::dispatch_workgroups_indirect`) -- GPU-driven dispatch sizes for dynamic batching / autoregressive decode
- [ ] Pipeline & shader-module caching -- reuse compiled `wgpu::ShaderModule` and `ComputePipeline` across calls (currently per-dispatch)

#### P1 -- Important
- [ ] WGSL workgroup size auto-tuning per adapter limits (`max_compute_workgroup_size_x/y/z`, `max_compute_invocations_per_workgroup`)
- [ ] Push constants (when WebGPU API exposes them) -- avoid uniform-buffer allocation for small param structs
- [ ] WGSL `enable chromium_experimental_subgroups` path for Chromium native (pre-standard subgroup ops)
- [ ] Persistent staging buffer pool (avoid `map_async` round-trip per H2D / D2H)
- [ ] Timestamp queries (`wgpu::QuerySet`) for kernel-level GPU timing
- [ ] Shader hot-reload during WGSL development (debug-build only)

#### P2 -- Nice-to-Have
- [ ] `shader/fft_wgsl.rs` — WGSL radix-2 Cooley-Tukey FFT butterfly (Cooley-Tukey 1965): shared-memory butterfly kernel with bit-reversal permutation and twiddle-factor LUT for compute-only FFT dispatch in browsers; `WgslFftPlan`
- [ ] `backend/async_pipeline.rs` — async compute + render pipeline separation (2023): `wgpu::Device::create_compute_pipeline_async` for background shader compilation without blocking the render loop; `AsyncComputePipeline`
- [ ] Browser WASM threading via `SharedArrayBuffer` + `Atomics` (requires COOP/COEP headers)
- [ ] Multi-adapter dispatch on systems with discrete + integrated GPU (laptop dGPU/iGPU)
- [ ] WGSL `f64` emulation (WebGPU has no double precision -- requires double-single arithmetic)
- [ ] `WGPU_BACKEND` env var documentation for cross-backend testing (`vulkan`/`metal`/`dx12`/`gl`)
- [ ] OffscreenCanvas / WebWorker integration for non-blocking dispatch in browsers

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| wgpu | WebGPU implementation -- backs to Vulkan/Metal/D3D12/GL/Browser | Yes |
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 129 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

WebGPU compute is the most-constrained backend due to portability requirements: it must run on browser, native Vulkan, Metal, D3D12, and GLES from a single WGSL source.

| Target Platform | Backing API | Throughput Floor | Notes |
|------|-------------|-------------------|-------|
| Native + NVIDIA | Vulkan / D3D12 | ≥ 70% of native Vulkan | wgpu has extra abstraction cost |
| Native + AMD | Vulkan | ≥ 70% of native Vulkan | |
| Native + Apple Silicon | Metal | ≥ 70% of native Metal | |
| Browser + Chrome | Dawn (WebGPU) | ≥ 50% of native equivalent | Browser sandbox overhead |
| Browser + Firefox | wgpu | ≥ 50% of native equivalent | Same engine, browser context |
| Browser + Safari | WebKit WebGPU | ≥ 50% of native equivalent | Newer implementation |

- **GEMM (current tiled WGSL)**: target ≥ 70% of native Vulkan SPIR-V on desktop
- **GEMM (with subgroup ops)**: target ≥ 85% of native Vulkan
- **Browser H2D / D2H**: dominated by `map_async` -- batch transfers when possible
- **Kernel dispatch overhead**: target < 50 µs above raw `wgpu::Queue::submit` (includes WGSL-to-SPIR-V/MSL/HLSL translation on first use)

## Notes

- WebGPU has **no FP64** -- f64 operations not supported in WGSL (workaround: emulation or fallback to CPU)
- WebGPU has **no recursion** -- all shader code must be flattened
- Workgroup dimension is capped at 65,535 per axis (3D limit; in practice often `(65535, 65535, 65535)`)
- Workgroup size capped per device (typically 256 threads/workgroup total, 1024 max)
- WGSL `enable f16` requires `WebGPUShaderF16` (`wgpu::Features::SHADER_F16`)
- Browser implementations are still maturing; subgroup ops not universally available as of early 2026
- WASM compilation requires `wasm-bindgen` glue at application layer (not in this crate)
- `wgpu` 0.20+ APIs are the supported target; older API surface may regress on update

## Architecture-Specific Deepening Opportunities

### Native Vulkan Backing
- [ ] `wgpu::Features::SUBGROUP` toggle (Vulkan 1.1+ subgroup ops surfaced through wgpu)
- [ ] `wgpu::Features::TIMESTAMP_QUERY` for GPU profiling
- [ ] Compare vs direct `oxicuda-vulkan` backend on identical kernel

### Native Metal Backing (macOS)
- [ ] `MTLArgumentEncoder` not exposed via wgpu yet -- use bind groups instead
- [ ] Compare vs direct `oxicuda-metal` backend on Apple Silicon

### Native D3D12 Backing (Windows)
- [ ] DXIL SM 6.0+ feature negotiation (FP16, wave intrinsics)
- [ ] Root-signature size tuning for descriptor-heavy workloads

### Native GL Backing (legacy fallback)
- [ ] GLES 3.1 compute-shader path for older mobile / Linux without Vulkan
- [ ] Documented perf cliff vs Vulkan/Metal/D3D12

### Browser (Chrome via Dawn)
- [ ] Chrome 125+ subgroup ops (`subgroupBroadcast`, `subgroupAdd`)
- [ ] Chrome WebGPU compute concurrent with WebGL2 not yet supported -- single context only
- [ ] `OffscreenCanvas` integration for worker-thread GPU dispatch

### Browser (Firefox via wgpu)
- [ ] Same wgpu codebase as native -- behavioral parity expected
- [ ] WebGPU enabled by default in Firefox 135+ on supported platforms

### Browser (Safari via WebKit)
- [ ] Safari 18+ WebGPU support (iOS 18 / macOS 15+)
- [ ] Document Apple's WGSL compliance level

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These items represent the gap between current implementation depth and production-grade WebGPU parity.

### Test Coverage Gaps
- [ ] Cross-backend correctness matrix: same WGSL on Vulkan / Metal / D3D12 / GL produces bitwise-identical results
- [ ] Browser CI via `wasm-pack test --chrome --headless` (currently native-only)
- [ ] Firefox WebGPU CI (`wasm-pack test --firefox --headless`)
- [ ] Subgroup-op kernels tested on hardware that supports them (currently not implemented)
- [ ] FP16 GEMM correctness vs FP32 reference within tolerance (5e-3 relative)
- [ ] WGSL output validated by `naga --validate` in CI pre-commit
- [ ] Adapter limits negotiation (workgroup size, storage buffer count) tested on iGPU + dGPU

### Implementation Deepening
- [ ] Pipeline-cache key includes adapter ID -- wgpu caches per-device but not cross-device
- [ ] WGSL constant-folding / dead-code elimination at generator layer (avoids large emitted strings)
- [ ] Async pipeline creation via `wgpu::Device::create_compute_pipeline_async` (browser-friendly)
- [ ] `Bindings::buffer` array support for tensor lists (avoid one descriptor per tensor)
- [ ] SharedArrayBuffer-backed WASM allocations for zero-copy CPU/GPU interop in browsers
- [ ] Compatibility shim layer for upcoming WebGPU `push_constant` (spec stabilization in progress)

## WebGPU / wgpu Version Compatibility

| wgpu Release | WebGPU Spec | Status | Notes |
|--------------|-------------|--------|-------|
| 0.18 | August-2023 | Build only | Earliest API used in tree |
| 0.19 | December-2023 | Tested | Subgroup ops behind feature flag |
| 0.20 | April-2024 | Tested | Default target |
| 23.x | Late-2024+ (semver flat) | Verified | Current target on `main` |

- **wgpu 23.x** target uses flat semver (`23.0.0`, `23.0.1`) following wgpu's policy change
- **Browser baseline**: Chrome 113+ (WebGPU stable), Firefox 135+ (rollout), Safari 18+ (macOS 15 / iOS 18)
- **WGSL extension flags** required: `f16` (FP16), `subgroups` (Chrome 125+ experimental)
- **No FP64**, no recursion, no pointer arithmetic, no thread-private heap allocation

## Observability & Diagnostics

- [ ] `tracing` span on every `WebGpuBackend::*` entry point with WGSL kernel hash
- [ ] WGSL source dumped to `target/wgsl-cache/` in debug builds for `naga --validate` round-trip
- [ ] Shader-compile cache hit / miss counters exposed via metric
- [ ] `wgpu::Queue::on_submitted_work_done` callback for measuring end-to-end GPU latency
- [ ] Adapter info (vendor, device, backend) logged at startup for cross-platform issue triage
- [ ] Browser `console.log` integration via `tracing-wasm` for in-page diagnostics

## Roadmap & Milestones

- **v0.2 (Subgroup ops)**: WGSL subgroup ops, indirect dispatch, persistent pipeline cache
- **v0.3 (Browser-friendly)**: Async pipeline compile, SharedArrayBuffer interop, OffscreenCanvas dispatch
- **v0.4 (Polish)**: WGSL `naga` validation in CI, browser CI matrix (Chrome / Firefox / Safari), timestamp queries
- **v1.0 (Stable)**: Cross-platform performance parity ≥ 75% of native equivalent across all WebGPU adapters
