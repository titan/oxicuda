# oxicuda-rocm TODO

AMD ROCm compute backend with HIP kernel string generators and host-side dispatch,
providing GPU-accelerated operations on AMD GPUs. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~3,739 across 9 files
- **Tests:** 104 passing
- **Status:** HIP kernel generators, hipBLAS/hipRTC runtime loaders, multi-GPU dispatcher
- **Targets:** AMD CDNA1/CDNA2/CDNA3 (MI100/MI200/MI300) and RDNA2/RDNA3 (Radeon RX 7000 series)

### Completed

#### Core Infrastructure
- [x] `lib.rs` -- module wiring, public exports of `RocmBackend`, `RocmError`, `RocmResult`
- [x] `backend.rs` -- `RocmBackend` implementing `ComputeBackend` trait; dispatch for GEMM, unary/binary/reduction, conv2d, attention, batched GEMM (~57.9K)
- [x] `device.rs` -- HIP device enumeration via `hipGetDeviceCount`, attribute queries, `RocmDevice` selection (~13.2K)
- [x] `error.rs` -- `RocmError` enum (LibraryNotFound, UnsupportedPlatform, KernelCompileError, RuntimeError, InvalidParameter, MemoryError) with thiserror
- [x] `memory.rs` -- `RocmMemoryManager` over `hipMalloc`/`hipFree`/`hipMemcpy*`; buffer pool with u64 handles (~11.9K)

#### HIP Kernel Generators (`hip_kernels.rs`, ~32.7K)
- [x] `gemm_hip(tile_size)` -- tiled `__global__ void gemm_f32`, configurable tile size, alpha/beta scaling
- [x] `gemm_hip_f16` / `gemm_hip_bf16` -- half-precision GEMM strings targeting MFMA instructions on CDNA2+
- [x] `batched_gemm_hip(tile_size)` -- 3D grid `(N/ts, M/ts, batch_count)` with `hipBlockIdx_z` batch index, stride-based per-batch offsets
- [x] Unary elementwise kernels -- relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu (with FP32/FP16 variants)
- [x] Binary elementwise kernels -- add, sub, mul, div, max, min, pow
- [x] Reduction kernels -- sum, max, min, mean using HIP `__shared__` LDS tiling
- [x] Attention kernel -- HIP fused scaled dot-product attention with softmax
- [x] Conv2D kernel -- HIP convolution with NCHW layout

#### Runtime Compilation (`hiprtc.rs`, ~13.8K)
- [x] `HipRtc::load()` -- runtime loader of `libhiprtc.so.{6,5,4}` via `libloading`; gracefully returns `LibraryNotFound` if absent
- [x] `HipRtcOptions` -- ergonomic builder for compile flags (`-O3`, `--gpu-architecture=gfx90a`, defines)
- [x] `compile_from_source` -- compiles HIP C++ to GCN ISA / AMDGPU binary at runtime

#### hipBLAS Interop (`hipblas.rs`, ~14.0K)
- [x] `HipBlas::load()` -- runtime loader of `libhipblas.so` for tuned SGEMM/DGEMM/HGEMM
- [x] `HipBlasOperation` and `HipBlasFillMode` C ABI enums
- [x] Falls back to in-tree HIP kernel path when library is missing

#### Multi-GPU (`multi_device.rs`, ~13.6K)
- [x] `MultiDeviceDispatcher` -- discovers all HIP-capable GPUs via `hipGetDeviceProperties`
- [x] `DeviceInfo` -- id, name, total_memory, compute_units, xGMI peer flag
- [x] Row-slab partitioning across multiple MI200/MI300 cards; single-device fallthrough

### Future Enhancements

#### P0 -- Critical
- [ ] CDNA3 (gfx940/gfx941/gfx942) FP8 (`OCP_FP8_E4M3` / `E5M2`) MFMA instruction emission for MI300X inference workloads
- [ ] Stream-K GEMM scheduling on CDNA3 -- atomic work-stealing across `XCD` chiplets to balance compute on MI300 split-die architecture
- [ ] RCCL collectives integration -- `ncclAllReduce`/`ncclAllGather`/`ncclReduceScatter` via runtime-loaded `librccl.so` for multi-GPU training
- [ ] xGMI peer-to-peer copy -- enable `hipDeviceEnablePeerAccess` and `hipMemcpyPeer` for cross-socket MI300A/MI300X transfers without staging through host memory

#### P1 -- Important
- [ ] hipGraph capture/instantiate/launch -- `hipStreamBeginCapture`/`hipGraphInstantiate` for repeated dispatch patterns (transformer decoder layers)
- [ ] Cooperative groups equivalent -- `hipCooperativeLaunch` for kernels requiring grid-wide synchronization (FlashAttention reductions)
- [ ] MFMA instruction variants -- emit `v_mfma_f32_16x16x16_f16`, `v_mfma_f32_32x32x8_bf16`, `v_mfma_f64_16x16x4_f64` directly via inline GCN asm
- [ ] wave32 vs wave64 dispatch heuristic -- RDNA3 prefers wave32 for divergent control flow, CDNA prefers wave64 for memory-bound kernels
- [ ] LDS (Local Data Share) bank conflict avoidance -- skew GEMM A/B tiles by 4 bytes to eliminate 32-bank conflicts

#### P2 -- Nice-to-Have
- [ ] `hip_kernels/rocm_sdma.rs` — ROCm 6.0 SDMA engine dispatch (AMD 2024): direct submission to System DMA engines via `hipMemcpyWithStream` SDMA path for CPU-GPU zero-copy on unified-memory APU (MI300A); `RocmSdmaTransfer`
- [ ] `hipblas/hipblaslt.rs` — hipBLASLt runtime loader (`libhipblaslt.so`) for matrix-layout-flexible GEMM with epilogue fusions (bias/activation) on CDNA3/RDNA3; `HipBlasLt`
- [ ] `device/flat_workgroup.rs` — AMDGPU flat work-group size attribute tuning (AMD 2022): emit `__attribute__((amdgpu_flat_work_group_size(min, max)))` hints to hipRTC for optimal occupancy on divergent kernels; `FlatWorkgroupHint`
- [ ] AMD MIGraphX backend interop for ONNX model execution
- [ ] AOMP/AOCC LLVM IR emission as alternative to hipRTC source compilation
- [ ] HSA (Heterogeneous System Architecture) queue management for fine-grained dispatch
- [ ] AMD Smart Access Memory (SAM) tuning -- BAR-mapped device memory for direct CPU writes on RDNA3 boards
- [ ] ROCm SMI integration -- power/thermal telemetry, clock query, ECC error reporting

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| libloading | Runtime `.so` loading for hipRTC and hipBLAS | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 104 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

ROCm performance is bound by CDNA matrix-core throughput and Infinity Fabric / xGMI bandwidth.

| GPU | Architecture | Peak FP16 (TFLOPS) | Peak BF16 | Peak FP8 | HBM BW |
|-----|--------------|--------------------|-----------|----------|--------|
| MI100 | CDNA1 (gfx908) | ~185 | -- | -- | 1.2 TB/s |
| MI210 | CDNA2 (gfx90a) | ~181 | ~181 | -- | 1.6 TB/s |
| MI250X | CDNA2 (gfx90a, 2 dies) | ~383 | ~383 | -- | 3.2 TB/s |
| MI300X | CDNA3 (gfx942) | ~1300 | ~1300 | ~2600 | 5.3 TB/s |
| RX 7900 XTX | RDNA3 (gfx1100) | ~123 | -- | -- | 960 GB/s |

- **GEMM (HIP kernel)**: target ≥ 70% of hipBLAS throughput for typical sizes (M=N=K=4096) on gfx90a
- **MFMA-accelerated GEMM** (future): target ≥ 90% of rocBLAS on gfx940
- **xGMI peer copy** (future): target ≥ 95% of theoretical 800 GB/s on MI300X dual-GPU setup
- **Kernel dispatch overhead**: target < 10 µs over raw `hipLaunchKernel`

## Notes

- All HIP runtime calls go through dynamically-loaded `libamdhip64.so` -- no link-time dependency
- macOS builds compile but return `RocmError::UnsupportedPlatform` (no ROCm support)
- Windows builds compile but return `UnsupportedPlatform` -- HIP-on-Windows is not yet integrated
- Generated HIP source uses `extern "C" __global__` ABI to remain compatible with both `hipcc` and `hiprtcCompileProgram`
- Buffer handles are 64-bit opaque IDs mapped to raw device pointers internally (poisoning-resistant)

## Architecture-Specific Deepening Opportunities

### CDNA1 (gfx906/gfx908, MI50/MI60/MI100)
- [ ] `v_mfma_f32_32x32x4f16` for native 32x32x4 FP16 tiles
- [ ] LDS tile sizes tuned for 64 KiB shared memory budget
- [ ] Test against rocBLAS-3.x reference for MI100

### CDNA2 (gfx90a, MI210/MI250/MI250X)
- [ ] BF16 MFMA path validated for MI210
- [ ] Multi-die scheduling for MI250X (two GCDs per package) via xGMI
- [ ] AOCC/aomp LLVM-IR pipeline as alternative to hiprtc

### CDNA3 (gfx940/gfx941/gfx942, MI300A/MI300X)
- [ ] FP8 E4M3 / E5M2 MFMA emission and accuracy validation
- [ ] Split-die XCD chiplet-aware tile dispatch
- [ ] APU shared-memory awareness for MI300A (unified CPU+GPU memory)
- [ ] Sparse MFMA (`v_mfma_f32_32x32x16f16_sparse`) for 2:4 structured sparsity

### RDNA2/RDNA3 (gfx1030/gfx1100, Radeon RX 6000/7000)
- [ ] wave32 SIMD path with reduced register pressure
- [ ] WMMA (Wave Matrix Multiply Accumulate) instruction emission for RDNA3
- [ ] Hardware ray-tracing units left untargeted -- compute path only

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These items represent the gap between current implementation depth and production-grade ROCm parity.

### Test Coverage Gaps
- [ ] Multi-GPU MI300X correctness suite -- run on 2+ GPU systems with xGMI links
- [ ] HIP graph capture/replay correctness against per-launch baseline
- [ ] MFMA numerical accuracy vs rocBLAS reference (FP16/BF16/FP8)
- [ ] hipBLAS interop tested across hipBLAS 1.x and 2.x ABIs
- [ ] gfx906/gfx908/gfx90a/gfx940 dispatch coverage matrix (currently only gfx90a path verified)
- [ ] Wave32 vs wave64 dispatch decision verified on RDNA3 vs CDNA hardware

### Implementation Deepening
- [ ] hipRTC compilation log surfaced through `RocmError::KernelCompileError` with structured ptxas-equivalent diagnostics
- [ ] hipMemPool / `hipMallocAsync` stream-ordered allocation (CUDA 11.2+ equivalent: HIP 5.0+)
- [ ] hipGraphInstantiate with kernel-node parameter updates for repeated launches
- [ ] AMDGPU instruction selection hints via `__attribute__((amdgpu_flat_work_group_size))`
- [ ] LDS tile padding tuned per CDNA generation (CDNA2: 64 KiB, CDNA3: 64 KiB + split mode)

## ROCm Version Compatibility

| ROCm Release | HIP Version | Status | Notes |
|--------------|-------------|--------|-------|
| ROCm 5.4 | HIP 5.4 | Tested | Minimum supported |
| ROCm 5.6 | HIP 5.6 | Tested | LTS branch |
| ROCm 6.0 | HIP 6.0 | Tested | MI300X support added |
| ROCm 6.1 | HIP 6.1 | Tested | gfx941/gfx942 stable |
| ROCm 6.2 | HIP 6.2 | Verified | Default target |

Library candidates searched (in order): `libhiprtc.so.6`, `libhiprtc.so.5`, `libhiprtc.so.4`, `libhiprtc.so`. Same fallback list applies to `libhipblas.so` and `libamdhip64.so`.

## Observability & Diagnostics

- [ ] `tracing` span instrumentation on every `RocmBackend::*` entry point with structured kernel name / dispatch dims
- [ ] `RocmError` Display includes failed function name (e.g., `hipLaunchKernelGGL`) + return-code constant name (e.g., `hipErrorOutOfMemory`)
- [ ] Optional `--features rocm-smi` for power/thermal telemetry queries through `librocm_smi64.so`
- [ ] Kernel launch event log (ring buffer) for post-mortem debugging on driver crash

## Roadmap & Milestones

- **v0.2 (CDNA3 readiness)**: FP8 OCP MFMA, Stream-K on MI300, xGMI peer copy
- **v0.3 (Collectives)**: RCCL integration, hipGraph capture, multi-process IPC
- **v0.4 (Polish)**: ROCm 7.0 support, AOMP LLVM IR backend, ROCm SMI telemetry
- **v1.0 (Stable)**: rocBLAS / MIOpen API parity for inference workloads, full multi-GPU training paths

## MFMA Tile Shape Reference (CDNA)

| Architecture | Instruction | Tile (M x N x K) | Dtype | Accumulator | Throughput (per CU) |
|--------------|-------------|------------------|-------|-------------|---------------------|
| CDNA1 (gfx908) | `v_mfma_f32_32x32x4f16` | 32 x 32 x 4 | FP16 | FP32 | 256 FLOPS/cycle |
| CDNA1 (gfx908) | `v_mfma_f32_16x16x4f16` | 16 x 16 x 4 | FP16 | FP32 | 256 FLOPS/cycle |
| CDNA2 (gfx90a) | `v_mfma_f32_16x16x16bf16_1k` | 16 x 16 x 16 | BF16 | FP32 | 512 FLOPS/cycle |
| CDNA2 (gfx90a) | `v_mfma_f64_16x16x4_f64` | 16 x 16 x 4 | FP64 | FP64 | 64 FLOPS/cycle |
| CDNA3 (gfx942) | `v_mfma_f32_16x16x32fp8_fp8` | 16 x 16 x 32 | FP8 (E4M3) | FP32 | 2048 FLOPS/cycle |
| CDNA3 (gfx942) | `v_mfma_f32_16x16x32bf8_bf8` | 16 x 16 x 32 | FP8 (E5M2) | FP32 | 2048 FLOPS/cycle |
| RDNA3 (gfx1100) | `v_wmma_f32_16x16x16_f16` | 16 x 16 x 16 | FP16 | FP32 | 256 FLOPS/cycle |

Tile-shape selection is currently hardcoded; future work moves to a `gfx*`-aware tile dispatcher mirroring the `oxicuda-blas` tile-config approach for CUDA.
