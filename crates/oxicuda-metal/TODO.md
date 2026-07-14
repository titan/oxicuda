# oxicuda-metal TODO

Apple Metal compute backend via metal-rs, providing GPU-accelerated operations
on macOS through MSL shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 7,287 across 22 files
- **Tests:** 262 unit + 3 doc = 265 passing (host/codegen surface; on-device tests are `#[cfg(target_os = "macos")]`-gated and skip without a GPU)
- **Status:** Full memory + compute backend with MSL, MPS interop, ANE hints, GPU FFT, plus host-side codegen/builders for `simdgroup_matrix`/df64-FP64/INT8 GEMM, MTLHeap suballocation, storage-mode planning, argument buffers, events/fences, indirect command + blit lists, GPU-family capability gating, and dispatch planning
- **Targets:** Apple Silicon (M1/M2/M3/M4 series) and Intel Mac (discrete + integrated)

### Completed

#### Core Infrastructure
- [x] `lib.rs` -- module wiring, re-exports `MetalBackend`, `MetalFftBuffer/Direction/Plan`
- [x] `backend/mod.rs` + `backend/{types,functions,trait_impls}.rs` -- `MetalBackend` implementing `ComputeBackend`; split per Refactoring Policy (single file < 2000 lines)
- [x] `device.rs` -- `MetalDevice` via `metal::Device::system_default()`; macOS-only initialization
- [x] `error.rs` -- `MetalError` (UnsupportedPlatform, ShaderCompilation, PipelineCreation, DeviceError, MemoryError) with thiserror
- [x] `memory.rs` -- `MetalMemoryManager` over `MTLBuffer` in `Shared` mode (unified memory), u64 handle pool
- [x] `pipeline.rs` -- `MetalComputePipeline` owns `ComputePipelineState` + `CommandQueue` together for dispatch

#### MSL Shader Generators (`msl.rs`, ~27.5K)
- [x] `gemm_msl` -- tiled GEMM with threadgroup shared memory and `GemmParams` constant buffer at `[[buffer(3)]]`
- [x] `gemm_msl_f16` -- half-precision GEMM using Metal `half` type for M-series unified-memory bandwidth optimization
- [x] `batched_gemm_msl` -- 3D dispatch with `threadgroup_position_in_grid.z` batch index, stride-based per-batch offsets
- [x] `elementwise_msl` -- unary kernels for relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] `binary_msl` -- add, sub, mul, div, max, min, pow generator
- [x] `reduction_msl` -- dedicated MSL shaders for sum/max/min/mean using `simdgroup_*` intrinsics
- [x] Conv2D MSL kernel with NCHW layout + CPU fallback for unsupported tile sizes
- [x] Attention MSL kernel -- scaled dot-product + stable softmax + causal masking
- [x] `softmax_msl` (`msl_nn.rs`) -- row-wise numerically-stable softmax (threadgroup max/sum reductions)
- [x] `layernorm_msl` (`msl_nn.rs`) -- row-wise layer-norm with affine `gamma`/`beta` and `rsqrt(var+eps)`
- [x] `scan_msl` (`msl_nn.rs`) -- Hillis-Steele inclusive/exclusive prefix sum with ping-pong threadgroup buffers

#### Host-side builders & planners (`*.rs`, CPU-testable)
- [x] `dispatch.rs` -- `DispatchPlanner`/`DispatchPlan`: 1D/2D/batched threadgroup + grid sizing, SIMD-aligned widths, threadgroup-scratch budgeting
- [x] `storage.rs` -- `MetalStorageMode` (Shared/Managed/Private/Memoryless), `StoragePlanner` mode selection, `align_up`/`align_down` buffer math
- [x] `command.rs` -- `BlitCommandList` (copy/fill/synchronize op list with overlap validation) alongside the `IndirectCommandBuffer` recorder

#### FFT Pipeline (`fft.rs`, ~35.3K)
- [x] `MetalFftPlan::new()` -- compiles MSL `fft_butterfly` and `bit_reverse` shaders **once** at plan creation
- [x] `ComputePipelineState` objects cached in struct fields (eliminates 100 ms+ recompile per call)
- [x] `CommandQueue` cached per plan, reused across all `execute()` calls
- [x] Radix-2 DIT Cooley-Tukey FFT for power-of-2 sizes (forward + inverse)
- [x] `MetalFftBuffer` host-side helper, `MetalFftDirection` enum (Forward/Inverse)
- [x] All struct fields `#[cfg(target_os = "macos")]` gated; manual `Debug` impl (PSO has no Debug)

#### MPS Interop (`mps.rs`, ~14.2K)
- [x] `MpsDataType` (Float32/Float16/UInt8) enum mirroring `MPSDataType`
- [x] `MpsMatrixDescriptor` -- rows/columns/row_bytes shape
- [x] `MpsMatrixMultiply` -- SGEMM via `MPSMatrixMultiplication` for tuned Apple GPU paths
- [x] `MpsImageConvolveConfig` -- 2-D convolution via `MPSImageConvolution`
- [x] Non-macOS paths return `MetalError::UnsupportedPlatform`

#### ANE Dispatch Hints (`ane.rs`, ~12.1K)
- [x] `AneGeneration` enum (None / Gen1..Gen5) per Apple chip family with `tops()` reporting Apple's published TOPS figures
- [x] Heuristic detection from device/chip names (M1=Gen3 @ 15.8, M2=Gen4 @ 15.8, M3=Gen5 @ 38.0)
- [x] Operation classification: which ops are ANE-friendly vs GPU-preferred
- [x] `AneDispatchHint` -- decision-layer only (actual ANE execution requires CoreML at app layer)

### Future Enhancements

#### P0 -- Critical
- [x] Native MSL `simdgroup_matrix` GEMM kernels -- `msl_nn::simdgroup_gemm_msl` emits the `simdgroup_float8x8` MMA-tile GEMM source (8x8 `simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store`); `device_family::MetalDeviceCapabilities::prefer_simdgroup_gemm` gates dispatch on family + 8-alignment. *Codegen + gating CPU-tested; on-GPU 4-8x speedup measurement requires Apple GPU/Metal hardware.*
- [x] `MTLHeap` / `MTLResourceStorageModeManaged` budgeting -- `heap::MetalHeapAllocator` (first-fit suballocator with coalescing) + `heap::MemoryBudget` (unified-memory pressure tracking) + `storage::StoragePlanner` (Shared/Managed/Private selection, alignment math). *Placement/budgeting logic CPU-tested; driving a real `MTLHeap` requires Apple GPU/Metal hardware.*
- [ ] `MPSGraph` integration -- use `MPSGraphExecutable` for fused op chains (transformer blocks) instead of per-op MSL dispatch *(requires Apple GPU/Metal hardware -- needs the MPSGraph runtime)*

#### P1 -- Important
- [x] f64 emulation path -- `msl_nn::gemm_msl_f64_ds` emits a double-single (`df64`) GEMM (Dekker `two_prod`/`fma` + Knuth `two_sum`, `float2` limb storage); `numeric::DoubleSingle` + `pack_df64`/`unpack_df64` provide the matching host-side split arithmetic. *Codegen + host math CPU-tested.*
- [x] INT8 dynamic quantization GEMM -- `msl_nn::int8_quant_gemm_msl` emits the `char`x`char`->`int`->dequant-`float` GEMM; `numeric::Int8Quantizer` (symmetric + asymmetric) derives the scale/zero-point constants the kernel consumes. *Codegen + quant math CPU-tested.*
- [x] Argument buffers (`MTLArgumentEncoder`) -- `argbuffer::ArgumentBufferLayout` + builder lay out `[[id(n)]]` slots, byte offsets, and `encodedLength` for bindless buffer/texture/sampler/inline-constant tables. *Layout logic CPU-tested; encoding into a real argument buffer requires Apple GPU/Metal hardware.*
- [x] `MTLEvent` / `MTLSharedEvent` cross-queue synchronization -- `event::MetalEvent`/`MetalFence`/`EventTimeline` model monotonic signal/wait values and validate a multi-queue sync plan for satisfiability + deadlock cycles. *Ordering logic CPU-tested; real cross-process events require Apple GPU/Metal hardware.* (0.5.0: `EventTimeline::validate` no longer `.expect()`s on the missing-program-counter invariant -- it now returns a proper `MetalResult` error instead of panicking, with new regression-test coverage.)
- [x] Indirect command buffers (`MTLIndirectCommandBuffer`) -- `command::IndirectCommandBuffer` records a fixed-capacity, pre-encoded compute-dispatch list (`set_compute_command`/`reset_range`). *Recording logic CPU-tested; GPU-driven replay requires Apple GPU/Metal hardware.*

#### P2 -- Nice-to-Have
- [ ] `fft/mps_fft.rs` — MPS FFT integration via `MPSMatrixFourierTransform` (Apple 2021): 1D/2D power-of-2 FFT through Metal Performance Shaders for tuned Apple GPU paths; `MpsFftPlan` *(requires Apple GPU/Metal hardware -- MPS runtime)*
- [x] `device_family.rs` — Metal GPU family and feature gating: `device_family::MetalGpuFamily` (Apple4..9, Mac2) + `MetalDeviceCapabilities` gate `simdgroup_matrix`, dynamic caching, mesh shaders, argument-buffer tier-2, threadgroup-memory budget, and unified-memory per chip generation; `from_device_name` heuristic + `dispatch::DispatchPlanner` consume it. *Capability table CPU-tested; live `supportsFamily:` query requires Apple GPU/Metal hardware.*
- [ ] Mac Pro multi-GPU peer copy via `MTLDevice::peerGroupID` (Intel Mac Pro era; legacy support) *(requires Apple GPU/Metal hardware -- multiple physical GPUs)*
- [ ] Tile shaders for compute-in-rasterization hybrid workloads (M-series tile memory) *(requires Apple GPU/Metal hardware)*
- [ ] Hardware mesh shader support for graphics-compute fusion (M3+) *(requires Apple GPU/Metal hardware)*
- [ ] CoreML stub integration -- emit `.mlmodel` blobs from kernel graphs for ANE offload *(requires Apple GPU/Metal hardware + CoreML)*
- [ ] Metal3 dynamic libraries (`MTLDynamicLibrary`) for shader hot-reload in development builds *(requires Apple GPU/Metal hardware)*

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| metal | Metal API bindings (Obj-C FFI) | Yes (FFI wrapper) |
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| num-complex | Complex number type for FFT | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 265 passing (262 unit + 3 doc)
- unwrap() calls: 0 (production code; tests use `.expect(...)`)
- Clippy: clean (`-D warnings`, all-features all-targets)

## Performance Targets

Apple Silicon GPUs share memory with CPU (unified memory architecture) -- bandwidth limits dominate over compute on most ML workloads.

| Chip | GPU Cores | Peak FP32 TFLOPS | Unified Memory BW | ANE TOPS |
|------|-----------|------------------|---------------------|----------|
| M1 | 7-8 | ~2.6 | 68 GB/s | 15.8 |
| M1 Pro | 14-16 | ~5.2 | 200 GB/s | 15.8 |
| M1 Max | 24-32 | ~10.4 | 400 GB/s | 15.8 |
| M1 Ultra | 48-64 | ~20.8 | 800 GB/s | 31.6 |
| M2 | 8-10 | ~3.6 | 100 GB/s | 15.8 |
| M2 Max | 30-38 | ~13.6 | 400 GB/s | 15.8 |
| M3 Max | 30-40 | ~14.0 | 400 GB/s | 18.0 |
| M3 Ultra | 60-80 | ~28.0 | 800 GB/s | 36.0 |

- **GEMM (current tiled MSL)**: target ≥ 60% of `MPSMatrixMultiplication` on M2
- **GEMM (future `simdgroup_matrix`)**: target ≥ 90% of MPS on M3
- **FFT (Radix-2 DIT)**: target ≥ 70% of vDSP for power-of-2 sizes up to 2^20
- **Unified memory H2D/D2H**: zero-copy (shared mode) -- no PCIe bottleneck on Apple Silicon

## Notes

- Metal has **no native FP64** -- f64 operations require emulation or fallback to CPU
- `MTLStorageMode::Shared` enables zero-copy CPU/GPU access on Apple Silicon (UMA)
- Intel Macs use discrete or integrated GPUs with `Managed` storage requiring explicit sync
- All MSL shaders compiled via `device.new_library_with_source()` at runtime
- Non-macOS targets compile successfully but every operation returns `MetalError::UnsupportedPlatform`
- `metal::ComputePipelineState` has no `Debug` impl -- our wrappers use `finish_non_exhaustive()`

## Architecture-Specific Deepening Opportunities

### Apple A-series (iOS/iPadOS leakage)
- [ ] iOS Metal-compute path for A14+ (currently macOS-only)
- [ ] `simdgroup` width 32 assumption verified on A-series (vs M-series)

### M1 family (Firestorm + Icestorm, 5 nm)
- [ ] Tile sizes tuned for 32 KiB threadgroup memory budget per workgroup
- [ ] M1 ANE Gen3 path documented (15.8 TOPS INT8)
- [ ] Performance vs MPS on M1 base (8 GPU cores) baseline established

### M2 family (Avalanche + Blizzard, 5 nm enhanced)
- [ ] `simdgroup_matrix` 8x8x8 FP16 kernel benchmarked on M2 Max
- [ ] Enhanced media engine offload for FFT-on-image workloads
- [ ] M2 Ultra dual-die UltraFusion interconnect awareness

### M3 family (Everest + Sawtooth, 3 nm TSMC N3B)
- [ ] Dynamic Caching exploited -- GPU memory allocator hints
- [ ] Hardware mesh shaders for compute (mesh shading compute pipeline)
- [ ] ANE Gen5 (38 TOPS) dispatch decisions updated
- [ ] Ray-tracing units NOT targeted (compute path only)

### Intel Mac (Mac Pro 2019, Mac mini 2018)
- [ ] AMD Vega/Navi discrete GPU path (Mac Pro 7,1) using `MTLStorageMode::Managed`
- [ ] Multi-GPU peer access via `MTLDevice::peerGroupID`

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These items represent the gap between current implementation depth and production-grade Metal parity.

### Test Coverage Gaps
- [ ] M1/M2/M3 dispatch matrix verified across all chip variants (currently single-device CI) *(requires Apple GPU/Metal hardware)*
- [ ] MPS interop accuracy vs `MPSMatrixMultiplication` reference for SGEMM *(requires Apple GPU/Metal hardware)*
- [ ] FFT correctness vs vDSP for sizes 64, 256, 1024, 4096, 16384, 65536 *(requires Apple GPU/Metal hardware)*
- [ ] ANE dispatch decisions verified against actual CoreML execution *(requires Apple GPU/Metal hardware + CoreML)*
- [ ] Conv2D NCHW vs MPS image-format reference correctness *(requires Apple GPU/Metal hardware)*

### Implementation Deepening
- [x] `simdgroup_matrix` kernel codegen for M2+ -- `msl_nn::simdgroup_gemm_msl` (on-hardware perf vs tiled scalar GEMM still requires Apple GPU/Metal hardware)
- [x] Heap-based allocation for large working sets -- `heap::MetalHeapAllocator` + `MemoryBudget` (placement logic; driving a real `MTLHeap` requires Apple GPU/Metal hardware)
- [ ] MPSGraph executable caching across `execute()` calls *(requires Apple GPU/Metal hardware -- MPSGraph runtime)*
- [ ] `MTLCounterSampleBuffer` integration for kernel-level GPU timing *(requires Apple GPU/Metal hardware)*
- [ ] Pipeline state cache shared across `MetalBackend` instances (currently per-pipeline) *(requires Apple GPU/Metal hardware to validate dispatch)*
- [ ] `MTLDynamicLibrary` for shader hot-reload during MSL development *(requires Apple GPU/Metal hardware)*

## macOS Version Compatibility

| macOS | Metal Feature Set | Supported | Notes |
|-------|--------------------|-----------|-------|
| 11 Big Sur | Metal 2.3 | Build only | Drops some PSO options |
| 12 Monterey | Metal 2.4 | Tested | Minimum for FP16 GEMM |
| 13 Ventura | Metal 3 | Tested | `simdgroup_matrix` available |
| 14 Sonoma | Metal 3.1 | Tested | Argument-buffer tier 2 default |
| 15 Sequoia | Metal 3.2 | Verified | Default target |

- **metal-rs**: `0.27+` required for `simdgroup_matrix` bindings
- **MPS**: framework auto-linked on macOS; no version pin needed
- **Apple Silicon native**: arm64-darwin -- compile with `cargo build --target aarch64-apple-darwin`

## Observability & Diagnostics

- [ ] `tracing` span instrumentation on every `MetalBackend::*` entry point
- [ ] `MetalError` Display surfaces NSError userInfo (file/line for shader-compile errors)
- [ ] Optional `--features metal-counters` for `MTLCounterSampleBuffer` GPU-side timing
- [ ] Pipeline-compile time logged at INFO level (large MSL strings can take 100+ ms)
- [ ] Memory-pressure callbacks integrated -- handle `MTLDevice::low-power` mode

## Roadmap & Milestones

- **v0.2 (Apple Silicon ML readiness)**: `simdgroup_matrix` GEMM, MPSGraph integration, INT8 path
- **v0.3 (Memory & Sync)**: `MTLHeap`, argument buffers, `MTLEvent`/`MTLSharedEvent`, indirect commands
- **v0.4 (Polish)**: M3 Dynamic Caching exploitation, FP64 emulation, hot-reload shaders
- **v1.0 (Stable)**: MPS / MPSGraph parity for inference workloads, full FP16/INT8 quantization
