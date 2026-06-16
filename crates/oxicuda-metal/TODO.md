# oxicuda-metal TODO

Apple Metal compute backend via metal-rs, providing GPU-accelerated operations
on macOS through MSL shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 4,395 across 13 files
- **Tests:** 152 passing
- **Status:** Full memory + compute backend with MSL, MPS interop, ANE hints, GPU FFT
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
- [ ] Native MSL `simdgroup_matrix` GEMM kernels -- use Apple's SIMD-group matrix multiply for M2/M3 (analogous to Tensor Cores) for 4-8x speedup over current tiled GEMM
- [ ] `MTLHeap` / `MTLResourceStorageModeManaged` budgeting for large model weights exceeding unified-memory pressure
- [ ] `MPSGraph` integration -- use `MPSGraphExecutable` for fused op chains (transformer blocks) instead of per-op MSL dispatch

#### P1 -- Important
- [ ] f64 emulation path -- Metal has no native `double` type; implement double-single arithmetic for scientific workloads
- [ ] INT8 dynamic quantization GEMM for ANE/CPU offload of inference workloads (Apple Silicon int8 path)
- [ ] Argument buffers (`MTLArgumentEncoder`) for bindless texture/buffer access -- reduces per-dispatch overhead
- [ ] `MTLEvent` / `MTLSharedEvent` cross-queue and cross-process synchronization
- [ ] Indirect command buffers (`MTLIndirectCommandBuffer`) for low-overhead GPU-driven dispatch

#### P2 -- Nice-to-Have
- [ ] `fft/mps_fft.rs` — MPS FFT integration via `MPSMatrixFourierTransform` (Apple 2021): 1D/2D power-of-2 FFT through Metal Performance Shaders for tuned Apple GPU paths; `MpsFftPlan`
- [ ] `device/device_family.rs` — Metal GPU family and feature gating (Apple 2022): query `MTLGPUFamily` (Apple5/6/7/8, Mac2) to gate `simdgroup_matrix`, dynamic caching, and mesh shaders per chip generation at runtime; `MetalDeviceFamily`
- [ ] Mac Pro multi-GPU peer copy via `MTLDevice::peerGroupID` (Intel Mac Pro era; legacy support)
- [ ] Tile shaders for compute-in-rasterization hybrid workloads (M-series tile memory)
- [ ] Hardware mesh shader support for graphics-compute fusion (M3+)
- [ ] CoreML stub integration -- emit `.mlmodel` blobs from kernel graphs for ANE offload
- [ ] Metal3 dynamic libraries (`MTLDynamicLibrary`) for shader hot-reload in development builds

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| metal | Metal API bindings (Obj-C FFI) | Yes (FFI wrapper) |
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| num-complex | Complex number type for FFT | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 152 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

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
- [ ] M1/M2/M3 dispatch matrix verified across all chip variants (currently single-device CI)
- [ ] MPS interop accuracy vs `MPSMatrixMultiplication` reference for SGEMM
- [ ] FFT correctness vs vDSP for sizes 64, 256, 1024, 4096, 16384, 65536
- [ ] ANE dispatch decisions verified against actual CoreML execution
- [ ] Conv2D NCHW vs MPS image-format reference correctness

### Implementation Deepening
- [ ] `simdgroup_matrix` kernels for M2+ (currently tiled scalar GEMM only)
- [ ] Heap-based allocation for >4 GiB working sets on M-series
- [ ] MPSGraph executable caching across `execute()` calls
- [ ] `MTLCounterSampleBuffer` integration for kernel-level GPU timing
- [ ] Pipeline state cache shared across `MetalBackend` instances (currently per-pipeline)
- [ ] `MTLDynamicLibrary` for shader hot-reload during MSL development

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
